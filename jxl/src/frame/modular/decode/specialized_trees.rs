// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;
use std::ops::Range;

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Histograms, SymbolReader, unpack_signed};
use crate::error::Result;
use crate::frame::modular::decode::channel::{ModularChannelDecoder, sync_scratch};
use crate::frame::modular::decode::common::{make_pixel, precompute_references};
use crate::frame::modular::flat_tree::{FlatTreeNode, predict_flat};
use crate::frame::modular::predict::{PredictionData, WeightedPredictorState, clamped_gradient};
use crate::frame::modular::tree::{
    NUM_NONREF_PROPERTIES, PROPERTIES_PER_PREVCHAN, PredictionResult, TreeNode,
};
use crate::frame::modular::{ModularChannel, ModularStorage, Predictor, Tree};
use crate::headers::modular::GroupHeader;
use crate::image::{Image, ImageRectMut};

trait MaybeWeightedPredictor: Sized {
    fn wp_state(&mut self) -> Option<&mut WeightedPredictorState>;
    #[inline(always)]
    fn predict_and_prop(&mut self, pos: (usize, usize), data: &PredictionData) -> (i64, i32) {
        match self.wp_state() {
            Some(wp) => wp.predict_and_property(pos, data),
            None => (0, 0),
        }
    }
    #[inline(always)]
    fn predict(
        &mut self,
        nodes: &[FlatTreeNode],
        prediction_data: PredictionData,
        pos: (usize, usize),
        references: &Image<i32>,
        prop_buffer: &mut [i32; 256],
    ) -> PredictionResult {
        predict_flat(
            nodes,
            prediction_data,
            self.wp_state(),
            pos,
            references,
            prop_buffer,
        )
    }
    fn update_errors(&mut self, _val: i32, _pos: (usize, usize)) {}
}

impl MaybeWeightedPredictor for () {
    #[inline(always)]
    fn wp_state(&mut self) -> Option<&mut WeightedPredictorState> {
        None
    }
}

impl MaybeWeightedPredictor for WeightedPredictorState {
    #[inline(always)]
    fn wp_state(&mut self) -> Option<&mut WeightedPredictorState> {
        Some(self)
    }
    #[inline(always)]
    fn update_errors(&mut self, val: i32, pos: (usize, usize)) {
        self.update_errors(val, pos);
    }
}

trait Reader: Sized {
    fn read(
        &self,
        reader: &mut SymbolReader,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> i32;
}

impl Reader for i32 {
    #[inline(always)]
    fn read(&self, _: &mut SymbolReader, _: &Histograms, _: &mut BitReader, _: usize) -> i32 {
        *self
    }
}

struct Reader420NoLz;
impl Reader for Reader420NoLz {
    #[inline(always)]
    fn read(
        &self,
        reader: &mut SymbolReader,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> i32 {
        reader.read_signed_clustered_config_420(histograms, br, cluster)
    }
}

struct ReaderGeneric;
impl Reader for ReaderGeneric {
    #[inline(always)]
    fn read(
        &self,
        reader: &mut SymbolReader,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> i32 {
        reader.read_signed_clustered_inline(histograms, br, cluster)
    }
}

struct FlatTreeInner {
    nodes: Vec<FlatTreeNode>,
    references: Image<i32>,
    property_buffer: Box<[i32; 256]>,
    storage: ModularStorage,
}

impl FlatTreeInner {
    fn new(
        nodes: Vec<TreeNode>,
        max_property_count: usize,
        channel: usize,
        stream: usize,
        xsize: usize,
        storage: ModularStorage,
    ) -> Result<Self> {
        let num_ref_props = max_property_count
            .saturating_sub(NUM_NONREF_PROPERTIES)
            .next_multiple_of(PROPERTIES_PER_PREVCHAN);
        let references = Image::<i32>::new((num_ref_props, xsize))?;
        let mut property_buffer = Box::new([0; 256]);

        property_buffer[0] = channel as i32;
        property_buffer[1] = stream as i32;

        Ok(Self {
            nodes: Tree::build_flat_tree(&nodes)?,
            references,
            property_buffer,
            storage,
        })
    }
}

struct FlatTree<WP, R> {
    inner: FlatTreeInner,
    reader: R,
    wp_state: WP,
}

impl<WP: MaybeWeightedPredictor, R: Reader> FlatTree<WP, R> {
    fn new(inner: FlatTreeInner, reader: R, wp_state: WP) -> Self {
        Self {
            inner,
            reader,
            wp_state,
        }
    }
}

impl<WP: MaybeWeightedPredictor, R: Reader> ModularChannelDecoder for FlatTree<WP, R> {
    fn init_row(&mut self, buffers: &mut [&mut ModularChannel], chan: usize, y: usize) {
        precompute_references(
            buffers,
            chan,
            y,
            &mut self.inner.references,
            self.inner.storage,
        );
        self.inner.property_buffer[GRADIENT_PROPERTY as usize] = 0;
    }

    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let prediction_result = self.wp_state.predict(
            &self.inner.nodes,
            prediction_data,
            pos,
            &self.inner.references,
            &mut self.inner.property_buffer,
        );
        let dec = self
            .reader
            .read(reader, histograms, br, prediction_result.context as usize);
        let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
        self.wp_state.update_errors(val, pos);
        val
    }
}

const LUT_MAX_SPLITVAL: i32 = 1023;
const LUT_MIN_SPLITVAL: i32 = -1024;
pub const LUT_TABLE_SIZE: usize = (LUT_MAX_SPLITVAL - LUT_MIN_SPLITVAL + 1) as usize;
const _: () = assert!(LUT_TABLE_SIZE.is_power_of_two());

fn make_lut<'a>(
    tree: &[TreeNode],
    ans: &'a mut [u8; LUT_TABLE_SIZE],
) -> Option<&'a [u8; LUT_TABLE_SIZE]> {
    struct RangeAndNode {
        range: Range<i32>,
        node: u32,
    }
    let mut stack = vec![RangeAndNode {
        range: LUT_MIN_SPLITVAL..LUT_MAX_SPLITVAL + 1,
        node: 0,
    }];

    while let Some(RangeAndNode { range, node }) = stack.pop() {
        let v = tree[node as usize];
        match v {
            TreeNode::Split {
                val, left, right, ..
            } => {
                let first_left = val + 1;
                if first_left >= range.end || first_left <= range.start {
                    return None;
                }
                stack.push(RangeAndNode {
                    range: first_left..range.end,
                    node: left,
                });
                stack.push(RangeAndNode {
                    range: range.start..first_left,
                    node: right,
                });
            }
            TreeNode::Leaf {
                offset,
                multiplier,
                id,
                ..
            } => {
                if offset != 0 || multiplier != 1 {
                    return None;
                }
                let start = range.start - LUT_MIN_SPLITVAL;
                let end = range.end - LUT_MIN_SPLITVAL;
                ans[start as usize..end as usize].fill(id as u8);
            }
        }
    }

    Some(ans)
}

struct WpOnly<'a, R> {
    lut: &'a [u8; LUT_TABLE_SIZE],
    wp_state: WeightedPredictorState,
    reader: R,
}

impl<'a, R: Reader> WpOnly<'a, R> {
    fn new(
        tree: &[TreeNode],
        header: &GroupHeader,
        xsize: usize,
        reader: R,
        lut: &'a mut [u8; LUT_TABLE_SIZE],
    ) -> Option<Self> {
        let wp_state = WeightedPredictorState::new(&header.wp_header, xsize);
        let lut = make_lut(tree, lut)?;
        Some(Self {
            lut,
            wp_state,
            reader,
        })
    }
}

impl<'a, R: Reader> ModularChannelDecoder for WpOnly<'a, R> {
    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let (wp_pred, property) = self.wp_state.predict_and_property(pos, &prediction_data);
        let ctx = self.lut[(property as i64 - LUT_MIN_SPLITVAL as i64)
            .clamp(0, LUT_TABLE_SIZE as i64 - 1) as usize];
        let dec = self.reader.read(reader, histograms, br, ctx as usize);
        let val = dec.wrapping_add(wp_pred as i32);
        self.wp_state.update_errors(val, pos);
        val
    }
}

/// Property 9 is the "gradient property": left + top - topleft
const GRADIENT_PROPERTY: u8 = 9;
const WEIGHTED_PROPERTY: u8 = 15;

struct GradientOnly<'a, R> {
    lut: &'a [u8; LUT_TABLE_SIZE],
    reader: R,
}

impl<'a, R: Reader> GradientOnly<'a, R> {
    fn new(tree: &[TreeNode], reader: R, lut: &'a mut [u8; LUT_TABLE_SIZE]) -> Option<Self> {
        let lut = make_lut(tree, lut)?;
        Some(Self { lut, reader })
    }
}

impl<'a, R: Reader> ModularChannelDecoder for GradientOnly<'a, R> {
    #[inline(always)]
    fn needs_toptop(&self) -> bool {
        false
    }

    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        _: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let prop9 = prediction_data
            .left
            .wrapping_add(prediction_data.top)
            .wrapping_sub(prediction_data.topleft);

        let index =
            (prop9 as i64 - LUT_MIN_SPLITVAL as i64).clamp(0, LUT_TABLE_SIZE as i64 - 1) as usize;
        let cluster = self.lut[index];

        let pred = clamped_gradient(
            prediction_data.left as i64,
            prediction_data.top as i64,
            prediction_data.topleft as i64,
        );

        let dec = self.reader.read(reader, histograms, br, cluster as usize);
        dec.wrapping_add(pred as i32)
    }
}

struct SingleGradientOnly<R> {
    clustered_ctx: usize,
    reader: R,
}

impl<R: Reader> ModularChannelDecoder for SingleGradientOnly<R> {
    #[inline(always)]
    fn needs_toptop(&self) -> bool {
        false
    }

    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        _: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let pred = clamped_gradient(
            prediction_data.left as i64,
            prediction_data.top as i64,
            prediction_data.topleft as i64,
        );
        let dec = self.reader.read(reader, histograms, br, self.clustered_ctx);
        dec.wrapping_add(pred as i32)
    }
}

struct NoTreeZero {
    clustered_ctx: usize,
    single_value: Option<i32>,
    multiplier: u32,
    offset: i64,
}

impl ModularChannelDecoder for NoTreeZero {
    #[inline(never)]
    fn decode_one(
        &mut self,
        _prediction_data: PredictionData,
        _pos: (usize, usize),
        _reader: &mut SymbolReader,
        _br: &mut BitReader,
        _histograms: &Histograms,
    ) -> i32 {
        unreachable!()
    }
    #[inline(never)]
    fn decode_row(
        &mut self,
        buffers: &mut [&mut ModularChannel],
        chan: usize,
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
        mut scratch: Option<&mut [Vec<i32>; 3]>,
    ) {
        let storage = if scratch.is_some() {
            ModularStorage::I16
        } else {
            ModularStorage::I32
        };
        if let Some(sym) = self.single_value {
            match storage {
                ModularStorage::I16 => {
                    let mut rect = ImageRectMut::<i16>::from_raw(buffers[chan].data.as_rect_mut());
                    rect.row(y)
                        .fill(make_pixel(sym, self.multiplier, self.offset) as i16);
                }
                ModularStorage::I32 => {
                    let mut rect = ImageRectMut::<i32>::from_raw(buffers[chan].data.as_rect_mut());
                    rect.row(y)
                        .fill(make_pixel(sym, self.multiplier, self.offset));
                }
            }
            return;
        }
        let mut rect;
        let row: &mut [i32] = if let Some(scratch) = scratch.as_deref_mut() {
            &mut scratch[0][..xsize]
        } else {
            rect = ImageRectMut::<i32>::from_raw(buffers[chan].data.as_rect_mut());
            rect.row(y)
        };
        debug_assert_eq!(row.len(), xsize);
        if self.multiplier == 1 && self.offset == 0 {
            for r in row.iter_mut() {
                *r = reader.read_signed_clustered_inline(histograms, br, self.clustered_ctx);
            }
        } else {
            for r in row.iter_mut() {
                let residual =
                    reader.read_signed_clustered_inline(histograms, br, self.clustered_ctx);
                *r = make_pixel(residual, self.multiplier, self.offset);
            }
        }
        sync_scratch(buffers[chan], y, scratch);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct MultiPropCell {
    pub cluster: u8,
    pub predictor: Predictor,
}

impl Default for MultiPropCell {
    fn default() -> Self {
        Self {
            cluster: 0,
            predictor: Predictor::Zero,
        }
    }
}

fn make_lut_multi_prop<const N: usize>(
    tree: &[TreeNode],
    properties: [u8; N],
    cutoffs: &[Box<[i32]>; N],
    num_buckets: [usize; N],
    total_cells: usize,
    cells: &mut [MultiPropCell; LUT_TABLE_SIZE],
) -> Option<()> {
    struct StackItem<const N: usize> {
        node: u32,
        ranges: [(usize, usize); N],
    }

    let mut initial_ranges = [(0, 0); N];
    for i in 0..N {
        initial_ranges[i] = (0, num_buckets[i] - 1);
    }

    let mut stack = vec![StackItem {
        node: 0,
        ranges: initial_ranges,
    }];

    let mut filled_cells = 0usize;

    while let Some(StackItem { node, ranges }) = stack.pop() {
        if node as usize >= tree.len() {
            return None;
        }
        match tree[node as usize] {
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } => {
                let dim = properties.iter().position(|&p| p == property)?;
                let cutoff_idx = cutoffs[dim].binary_search(&val).ok()?;

                // Left child: property > val <=> bucket_index >= cutoff_idx + 1
                let left_lower = ranges[dim].0.max(cutoff_idx + 1);
                let left_upper = ranges[dim].1;
                if left_lower <= left_upper {
                    let mut left_ranges = ranges;
                    left_ranges[dim] = (left_lower, left_upper);
                    stack.push(StackItem {
                        node: left,
                        ranges: left_ranges,
                    });
                }

                // Right child: property <= val <=> bucket_index <= cutoff_idx
                let right_lower = ranges[dim].0;
                let right_upper = ranges[dim].1.min(cutoff_idx);
                if right_lower <= right_upper {
                    let mut right_ranges = ranges;
                    right_ranges[dim] = (right_lower, right_upper);
                    stack.push(StackItem {
                        node: right,
                        ranges: right_ranges,
                    });
                }
            }
            TreeNode::Leaf {
                offset,
                multiplier,
                id,
                predictor,
            } => {
                if offset != 0 || multiplier != 1 || id > 255 {
                    return None;
                }
                let cell_val = MultiPropCell {
                    cluster: id as u8,
                    predictor,
                };
                match N {
                    2 => {
                        let stride0 = num_buckets[1];
                        let len1 = ranges[1].1 - ranges[1].0 + 1;
                        for i in ranges[0].0..=ranges[0].1 {
                            let row_start = i * stride0;
                            let start = row_start + ranges[1].0;
                            let end = start + len1;
                            cells[start..end].fill(cell_val);
                        }
                        filled_cells += (ranges[0].1 - ranges[0].0 + 1) * len1;
                    }
                    3 => {
                        let stride1 = num_buckets[2];
                        let stride0 = num_buckets[1] * stride1;
                        let len2 = ranges[2].1 - ranges[2].0 + 1;
                        for i in ranges[0].0..=ranges[0].1 {
                            let plane_start = i * stride0;
                            for j in ranges[1].0..=ranges[1].1 {
                                let row_start = plane_start + j * stride1;
                                let start = row_start + ranges[2].0;
                                let end = start + len2;
                                cells[start..end].fill(cell_val);
                            }
                        }
                        filled_cells += (ranges[0].1 - ranges[0].0 + 1)
                            * (ranges[1].1 - ranges[1].0 + 1)
                            * len2;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    if filled_cells != total_cells {
        return None;
    }

    Some(())
}

#[inline(always)]
fn compute_property(
    property: u8,
    prediction_data: &PredictionData,
    pos: (usize, usize),
    prev_grad: i32,
    wp_prop: i32,
) -> i32 {
    match property {
        2 => pos.1 as i32,
        3 => pos.0 as i32,
        4 => prediction_data.top.wrapping_abs(),
        5 => prediction_data.left.wrapping_abs(),
        6 => prediction_data.top,
        7 => prediction_data.left,
        8 => prediction_data.left.wrapping_sub(prev_grad),
        9 => prediction_data
            .left
            .wrapping_add(prediction_data.top)
            .wrapping_sub(prediction_data.topleft),
        10 => prediction_data.left.wrapping_sub(prediction_data.topleft),
        11 => prediction_data.topleft.wrapping_sub(prediction_data.top),
        12 => prediction_data.top.wrapping_sub(prediction_data.topright),
        13 => prediction_data.top.wrapping_sub(prediction_data.toptop),
        14 => prediction_data.left.wrapping_sub(prediction_data.leftleft),
        15 => wp_prop,
        _ => unreachable!(),
    }
}

struct MultiPropTreeInner<const N: usize> {
    properties: [u8; N],
    prop_luts: Box<[[u16; LUT_TABLE_SIZE]; N]>,
    cells: Box<[MultiPropCell; LUT_TABLE_SIZE]>,
}

impl<const N: usize> MultiPropTreeInner<N> {
    fn new(tree: &[TreeNode]) -> Option<Self> {
        let mut properties = Vec::new();

        for node in tree {
            match node {
                TreeNode::Leaf {
                    offset,
                    multiplier,
                    id,
                    ..
                } => {
                    if *offset != 0 || *multiplier != 1 || *id > 255 {
                        return None;
                    }
                }
                TreeNode::Split { property, .. } => {
                    if *property >= NUM_NONREF_PROPERTIES as u8 {
                        return None;
                    }
                    if !properties.contains(property) {
                        properties.push(*property);
                    }
                }
            }
        }

        if properties.len() != N {
            return None;
        }
        properties.sort_unstable();

        let mut cutoffs_vec = Vec::with_capacity(N);
        let mut num_buckets = [0usize; N];
        let mut total_cells = 1usize;

        for (i, &prop) in properties.iter().enumerate() {
            let mut prop_cutoffs = Vec::new();
            for node in tree {
                if let TreeNode::Split { property, val, .. } = node
                    && *property == prop
                    && !prop_cutoffs.contains(val)
                {
                    prop_cutoffs.push(*val);
                }
            }
            prop_cutoffs.sort_unstable();
            num_buckets[i] = prop_cutoffs.len() + 1;
            total_cells = total_cells.checked_mul(num_buckets[i])?;
            cutoffs_vec.push(prop_cutoffs.into_boxed_slice());
        }

        if total_cells > LUT_TABLE_SIZE {
            return None;
        }

        for prop_cutoffs in &cutoffs_vec {
            for &c in prop_cutoffs.iter() {
                if !(LUT_MIN_SPLITVAL..LUT_MAX_SPLITVAL).contains(&c) {
                    return None;
                }
            }
        }

        let props: [u8; N] = properties.try_into().ok()?;
        let cutoffs: [Box<[i32]>; N] = cutoffs_vec.clone().try_into().ok()?;

        let mut cells = crate::util::box_array(MultiPropCell::default());
        make_lut_multi_prop(tree, props, &cutoffs, num_buckets, total_cells, &mut cells)?;

        let mut strides = [0usize; N];
        match N {
            2 => {
                strides[0] = num_buckets[1];
                strides[1] = 1;
            }
            3 => {
                strides[0] = num_buckets[1] * num_buckets[2];
                strides[1] = num_buckets[2];
                strides[2] = 1;
            }
            _ => unreachable!(),
        }

        let mut prop_luts = crate::util::box_array([0u16; LUT_TABLE_SIZE]);
        for d in 0..N {
            let cutoffs_d = &cutoffs_vec[d];
            let stride = strides[d];
            for (idx, slot) in prop_luts[d].iter_mut().enumerate() {
                let val = idx as i32 + LUT_MIN_SPLITVAL;
                let bucket = cutoffs_d.partition_point(|&c| c < val);
                *slot = (bucket * stride) as u16;
            }
        }

        Some(Self {
            properties: props,
            prop_luts,
            cells,
        })
    }
}

struct MultiPropTree<WP, R, const N: usize> {
    inner: MultiPropTreeInner<N>,
    prev_grad: i32,
    wp_state: WP,
    reader: R,
}

impl<WP, R, const N: usize> MultiPropTree<WP, R, N> {
    fn new(inner: MultiPropTreeInner<N>, wp_state: WP, reader: R) -> Self {
        Self {
            inner,
            prev_grad: 0,
            wp_state,
            reader,
        }
    }
}

impl<WP: MaybeWeightedPredictor, R: Reader, const N: usize> ModularChannelDecoder
    for MultiPropTree<WP, R, N>
{
    #[inline(always)]
    fn init_row(&mut self, _buffers: &mut [&mut ModularChannel], _chan: usize, _y: usize) {
        self.prev_grad = 0;
    }

    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let (wp_pred, wp_prop) = self.wp_state.predict_and_prop(pos, &prediction_data);

        let mut cell_index = 0usize;
        for d in 0..N {
            let prop_val = compute_property(
                self.inner.properties[d],
                &prediction_data,
                pos,
                self.prev_grad,
                wp_prop,
            );
            let lut_idx = (prop_val as i64 - LUT_MIN_SPLITVAL as i64)
                .clamp(0, LUT_TABLE_SIZE as i64 - 1) as usize;
            // We could use u8 + a multiply here, but that seems to be more expensive.
            cell_index += self.inner.prop_luts[d][lut_idx] as usize;
        }

        self.prev_grad = prediction_data
            .left
            .wrapping_add(prediction_data.top)
            .wrapping_sub(prediction_data.topleft);

        let cell_index = cell_index & (LUT_TABLE_SIZE - 1);
        let cell = self.inner.cells[cell_index];
        let pred = cell.predictor.predict_one(prediction_data, wp_pred);

        let dec = self
            .reader
            .read(reader, histograms, br, cell.cluster as usize);
        let val = dec.wrapping_add(pred as i32);
        self.wp_state.update_errors(val, pos);
        val
    }
}

fn run_multiprop<const N: usize, F: FnOnce(&mut dyn ModularChannelDecoder) -> Result<()>>(
    inner: MultiPropTreeInner<N>,
    header: &GroupHeader,
    xsize: usize,
    uses_non420: bool,
    uses_wp: bool,
    run: F,
) -> Result<()> {
    if !uses_wp {
        if !uses_non420 {
            let mut decoder = MultiPropTree::new(inner, (), Reader420NoLz);
            return run(&mut decoder);
        }
        let mut decoder = MultiPropTree::new(inner, (), ReaderGeneric);
        return run(&mut decoder);
    }

    let wp_state = WeightedPredictorState::new(&header.wp_header, xsize);
    if !uses_non420 {
        let mut decoder = MultiPropTree::new(inner, wp_state, Reader420NoLz);
        return run(&mut decoder);
    }
    let mut decoder = MultiPropTree::new(inner, wp_state, ReaderGeneric);
    run(&mut decoder)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_on_specialized_tree<F: FnOnce(&mut dyn ModularChannelDecoder) -> Result<()>>(
    tree: &Tree,
    channel: usize,
    stream: usize,
    xsize: usize,
    header: &GroupHeader,
    storage: ModularStorage,
    lut_scratch: &mut [u8; LUT_TABLE_SIZE],
    run: F,
) -> Result<()> {
    // TODO(veluca): consider skipping the pruning if header.uses_global_tree is true.
    let mut pruned_tree = Vec::new();
    let mut queue = VecDeque::new();
    pruned_tree.try_reserve(tree.nodes.len())?;
    queue.try_reserve(tree.nodes.len())?;
    queue.push_front(0);

    let mut uses_wp = false;
    let mut uses_non_wp = false;
    let mut max_property_count = 0;
    let mut uses_non_gradient = false;

    // If, after pruning the tree, `is_single_symbol` is true, then `single_symbol` is the
    // only symbol that could possibly be decoded by this tree.
    // TODO(veluca): The single-symbol special case corrupts the lz77 window, so it is
    // disabled for now if lz77 is enabled. Figure out how to make them work together.
    let mut is_single_symbol = !tree.histograms.lz77_params().enabled;
    let mut single_symbol = None;

    // Obtain a pruned tree without nodes that are not relevant in the current channel and stream.
    // Proceed in BFS order, so that we know that the children of a node will be adjacent.
    // Also re-maps context IDs to cluster IDs.
    while let Some(v) = queue.pop_front() {
        let mut node = tree.nodes[v as usize];
        match node {
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } if property < 2 => {
                // If the node splits on static properties, re-enqueue its correct child immediately.
                let vv = if property == 0 { channel } else { stream };
                queue.push_front(if vv as i32 > val { left } else { right });
                continue;
            }
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } => {
                uses_wp |= property == WEIGHTED_PROPERTY;
                uses_non_wp |= property != WEIGHTED_PROPERTY;
                uses_non_gradient |= property != GRADIENT_PROPERTY;
                max_property_count = max_property_count.max(property as usize + 1);
                let base = (queue.len() + pruned_tree.len() + 1) as u32;
                pruned_tree.push(TreeNode::Split {
                    property,
                    val,
                    left: base,
                    right: base + 1,
                });
                queue.push_back(left);
                queue.push_back(right);
            }
            TreeNode::Leaf { predictor, .. } => {
                uses_wp |= predictor == Predictor::Weighted;
                uses_non_wp |= predictor != Predictor::Weighted;
                uses_non_gradient |= predictor != Predictor::Gradient;
                let TreeNode::Leaf { id, .. } = &mut node else {
                    unreachable!()
                };
                *id = tree.histograms.map_context_to_cluster(*id as usize) as u32;
                if is_single_symbol {
                    if let Some(sym) = tree.histograms.single_symbol(*id as usize) {
                        if sym >= tree.histograms.uint(*id as usize).split_token() {
                            // This symbol would need extra bits. This is rare enough, so disable
                            // the optimization.
                            is_single_symbol = false;
                        }
                        if single_symbol.is_none() {
                            single_symbol = Some(sym);
                        }
                        if single_symbol != Some(sym) {
                            is_single_symbol = false;
                        }
                    } else {
                        is_single_symbol = false;
                    }
                }
                pruned_tree.push(node);
            }
        }
    }

    if !is_single_symbol {
        single_symbol = None;
    }

    if let [
        TreeNode::Leaf {
            predictor: Predictor::Zero,
            multiplier,
            offset,
            id,
        },
    ] = &*pruned_tree
    {
        return run(&mut NoTreeZero {
            clustered_ctx: *id as usize,
            single_value: single_symbol.map(unpack_signed),
            multiplier: *multiplier,
            offset: *offset as i64,
        });
    }

    if let [
        TreeNode::Leaf {
            predictor: Predictor::Gradient,
            multiplier: 1,
            offset: 0,
            id,
        },
    ] = &*pruned_tree
    {
        return run(&mut SingleGradientOnly {
            clustered_ctx: *id as usize,
            reader: ReaderGeneric,
        });
    }

    let uses_non420 = !tree.histograms.can_use_config_420_fast_path();

    if !uses_non_wp
        && !uses_non420
        && let Some(mut wp) = WpOnly::new(&pruned_tree, header, xsize, Reader420NoLz, lut_scratch)
    {
        return run(&mut wp);
    }

    if !uses_non_gradient
        && !uses_non420
        && let Some(mut grad) = GradientOnly::new(&pruned_tree, Reader420NoLz, lut_scratch)
    {
        return run(&mut grad);
    }

    if let Some(inner) = MultiPropTreeInner::<2>::new(&pruned_tree) {
        return run_multiprop(inner, header, xsize, uses_non420, uses_wp, run);
    }

    if let Some(inner) = MultiPropTreeInner::<3>::new(&pruned_tree) {
        return run_multiprop(inner, header, xsize, uses_non420, uses_wp, run);
    }

    let single_symbol = single_symbol.map(unpack_signed);

    let inner = FlatTreeInner::new(
        pruned_tree,
        max_property_count,
        channel,
        stream,
        xsize,
        storage,
    )?;

    // Non-WP trees (includes effort 2 encoding and some groups in effort > 3)
    if !uses_wp {
        if let Some(ss) = single_symbol {
            return run(&mut FlatTree::new(inner, ss, ()));
        }
        if !uses_non420 {
            return run(&mut FlatTree::new(inner, Reader420NoLz, ()));
        }
        return run(&mut FlatTree::new(inner, ReaderGeneric, ()));
    }

    let wp_state = WeightedPredictorState::new(&header.wp_header, xsize);

    if let Some(ss) = single_symbol {
        return run(&mut FlatTree::new(inner, ss, wp_state));
    }
    if !uses_non420 {
        return run(&mut FlatTree::new(inner, Reader420NoLz, wp_state));
    }
    run(&mut FlatTree::new(inner, ReaderGeneric, wp_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy_coding::decode::Histograms;
    use crate::frame::modular::Predictor;
    use crate::frame::modular::tree::TreeNode;
    use crate::image::ImageRect;

    fn walk_test_tree(tree: &[TreeNode], props: &[i32; 16]) -> u32 {
        let mut curr = 0;
        while let TreeNode::Split {
            property,
            val,
            left,
            right,
        } = tree[curr]
        {
            if props[property as usize] > val {
                curr = left as usize;
            } else {
                curr = right as usize;
            }
        }
        match tree[curr] {
            TreeNode::Leaf { id, .. } => id,
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_multiprop_2d_lut() {
        // Properties 7 (left) and 9 (gradient)
        // Splits:
        // Root: prop 7 > 10 -> left=1, right=2
        // Node 1: prop 9 > 0 -> left=3 (leaf 10), right=4 (leaf 20)
        // Node 2: prop 9 > 20 -> left=5 (leaf 30), right=6 (leaf 40)
        let tree = vec![
            TreeNode::Split {
                property: 7,
                val: 10,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 9,
                val: 0,
                left: 3,
                right: 4,
            },
            TreeNode::Split {
                property: 9,
                val: 20,
                left: 5,
                right: 6,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 10,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 20,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 30,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 40,
            },
        ];

        let inner = MultiPropTreeInner::<2>::new(&tree).unwrap();

        assert_eq!(inner.properties, [7, 9]);

        for p7 in [-2000, -20, 0, 10, 15, 50, 2000] {
            for p9 in [-2000, -50, -1, 0, 5, 20, 30, 100, 2000] {
                let mut props = [0i32; 16];
                props[7] = p7;
                props[9] = p9;

                let expected_cluster = walk_test_tree(&tree, &props);

                let idx0 = (p7 as i64 - LUT_MIN_SPLITVAL as i64).clamp(0, LUT_TABLE_SIZE as i64 - 1)
                    as usize;
                let idx1 = (p9 as i64 - LUT_MIN_SPLITVAL as i64).clamp(0, LUT_TABLE_SIZE as i64 - 1)
                    as usize;
                let idx = (inner.prop_luts[0][idx0] + inner.prop_luts[1][idx1]) as usize;
                let actual_cluster = inner.cells[idx].cluster as u32;

                assert_eq!(
                    actual_cluster, expected_cluster,
                    "Mismatch at p7={p7}, p9={p9}"
                );
            }
        }
    }

    #[test]
    fn test_multiprop_3d_lut() {
        // Properties 4 (|top|), 6 (top), 7 (left)
        // Root: prop 4 > 5 -> left=1, right=2
        // Node 1: prop 6 > 0 -> left=3, right=4
        // Node 2: leaf 1
        // Node 3: prop 7 > -10 -> left=5 (leaf 2), right=6 (leaf 3)
        // Node 4: leaf 4
        let tree = vec![
            TreeNode::Split {
                property: 4,
                val: 5,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 6,
                val: 0,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Split {
                property: 7,
                val: -10,
                left: 5,
                right: 6,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 3,
            },
        ];

        let inner = MultiPropTreeInner::<3>::new(&tree).unwrap();

        assert_eq!(inner.properties, [4, 6, 7]);

        for p4 in [-2000, 0, 5, 10, 2000] {
            for p6 in [-2000, -5, 0, 8, 2000] {
                for p7 in [-2000, -20, -10, 5, 2000] {
                    let mut props = [0i32; 16];
                    props[4] = p4;
                    props[6] = p6;
                    props[7] = p7;

                    let expected_cluster = walk_test_tree(&tree, &props);

                    let idx0 = (p4 as i64 - LUT_MIN_SPLITVAL as i64)
                        .clamp(0, LUT_TABLE_SIZE as i64 - 1)
                        as usize;
                    let idx1 = (p6 as i64 - LUT_MIN_SPLITVAL as i64)
                        .clamp(0, LUT_TABLE_SIZE as i64 - 1)
                        as usize;
                    let idx2 = (p7 as i64 - LUT_MIN_SPLITVAL as i64)
                        .clamp(0, LUT_TABLE_SIZE as i64 - 1)
                        as usize;
                    let idx = (inner.prop_luts[0][idx0]
                        + inner.prop_luts[1][idx1]
                        + inner.prop_luts[2][idx2]) as usize;
                    let actual_cluster = inner.cells[idx].cluster as u32;

                    assert_eq!(
                        actual_cluster, expected_cluster,
                        "Mismatch at p4={p4}, p6={p6}, p7={p7}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_multiprop_reject_invalid_trees() {
        // Offset != 0
        let tree_offset = vec![TreeNode::Leaf {
            predictor: Predictor::Zero,
            offset: 1,
            multiplier: 1,
            id: 0,
        }];
        assert!(MultiPropTreeInner::<2>::new(&tree_offset).is_none());

        // Multiplier != 1
        let tree_mul = vec![TreeNode::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 2,
            id: 0,
        }];
        assert!(MultiPropTreeInner::<2>::new(&tree_mul).is_none());

        // id > 255
        let tree_large_id = vec![TreeNode::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
            id: 256,
        }];
        assert!(MultiPropTreeInner::<2>::new(&tree_large_id).is_none());

        // Reference property (property >= 16)
        let tree_ref = vec![
            TreeNode::Split {
                property: 16,
                val: 0,
                left: 1,
                right: 2,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
        ];
        assert!(MultiPropTreeInner::<2>::new(&tree_ref).is_none());

        // Split value >= 1023
        let tree_high_split = vec![
            TreeNode::Split {
                property: 6,
                val: 1023,
                left: 1,
                right: 2,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
        ];
        assert!(MultiPropTreeInner::<2>::new(&tree_high_split).is_none());

        // Split value < -1024
        let tree_low_split = vec![
            TreeNode::Split {
                property: 6,
                val: -1025,
                left: 1,
                right: 2,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
        ];
        assert!(MultiPropTreeInner::<2>::new(&tree_low_split).is_none());
    }

    #[test]
    fn test_multiprop_reject_too_many_cells() {
        // Build a tree with 50 cutoffs for prop 6 and 50 cutoffs for prop 7:
        // (50 + 1) * (50 + 1) = 2601 > 2048 cells.
        let mut tree = Vec::new();
        for i in 0..50 {
            tree.push(TreeNode::Split {
                property: 6,
                val: i,
                left: (tree.len() + 2) as u32,
                right: (tree.len() + 1) as u32,
            });
            tree.push(TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            });
        }
        for i in 0..50 {
            tree.push(TreeNode::Split {
                property: 7,
                val: i,
                left: (tree.len() + 2) as u32,
                right: (tree.len() + 1) as u32,
            });
            tree.push(TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            });
        }
        tree.push(TreeNode::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
            id: 0,
        });

        assert!(MultiPropTreeInner::<2>::new(&tree).is_none());
    }

    #[test]
    fn test_multiprop_multiple_predictors() {
        let tree = vec![
            TreeNode::Split {
                property: 6,
                val: 0,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 7,
                val: 10,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Leaf {
                predictor: Predictor::North,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
        ];

        let inner = MultiPropTreeInner::<2>::new(&tree).unwrap();
        assert_eq!(inner.properties, [6, 7]);
        let p6_idx = (20i64 - LUT_MIN_SPLITVAL as i64).clamp(0, LUT_TABLE_SIZE as i64 - 1) as usize;
        let p7_idx = (20i64 - LUT_MIN_SPLITVAL as i64).clamp(0, LUT_TABLE_SIZE as i64 - 1) as usize;
        let cell = (inner.prop_luts[0][p6_idx] + inner.prop_luts[1][p7_idx]) as usize;
        assert_eq!(inner.cells[cell].cluster, 1);
        assert_eq!(inner.cells[cell].predictor, Predictor::Gradient);
    }

    #[test]
    fn test_multiprop_decode_one_execution() {
        // Test actual decode_one execution with Reader = i32 (constant read = 5)
        let tree = vec![
            TreeNode::Split {
                property: 6,
                val: 10,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 7,
                val: 20,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
        ];

        let inner = MultiPropTreeInner::<2>::new(&tree).unwrap();
        let mut decoder = MultiPropTree::new(inner, (), 5i32);

        let pred_data = PredictionData {
            left: 25,
            top: 15,
            toptop: 0,
            topleft: 10,
            topright: 0,
            leftleft: 0,
            toprightright: 0,
        };
        // prop 6 (top) = 15 > 10 -> left (node 1)
        // prop 7 (left) = 25 > 20 -> left (node 3 -> leaf 1)
        // clamped_gradient(25, 15, 10) = clamped to max (25) because topleft (10) < min (15)
        // val = 25 + 5 = 30
        let dummy_br = &mut BitReader::new(&[0u8; 16]);
        let dummy_histo = Histograms::dummy_for_test();
        let mut dummy_reader = SymbolReader::new(&dummy_histo, dummy_br, None).unwrap();

        let val = decoder.decode_one(pred_data, (5, 5), &mut dummy_reader, dummy_br, &dummy_histo);
        assert_eq!(val, 30);
    }

    #[test]
    fn test_multiprop_identical_to_flattree() {
        let tree = vec![
            TreeNode::Split {
                property: 6, // top
                val: 10,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 7, // left
                val: 20,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Leaf {
                predictor: Predictor::North,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
        ];

        let xsize = 16;
        let ysize = 8;
        let bit_depth =
            crate::headers::bit_depth::BitDepth::default(&crate::headers::encodings::Empty {});
        let mut buf_flat =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();
        let mut buf_multi =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();

        let inner_flat =
            FlatTreeInner::new(tree.clone(), 16, 0, 0, xsize, ModularStorage::I32).unwrap();
        let mut flat_decoder = FlatTree::new(inner_flat, 7i32, ());

        let inner_multi = MultiPropTreeInner::<2>::new(&tree).unwrap();
        let mut multi_decoder = MultiPropTree::new(inner_multi, (), 7i32);

        let dummy_br = &mut BitReader::new(&[0u8; 16]);
        let dummy_histo = Histograms::dummy_for_test();
        let mut dummy_reader = SymbolReader::new(&dummy_histo, dummy_br, None).unwrap();

        for y in 0..ysize {
            flat_decoder.decode_row(
                &mut [&mut buf_flat],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
            multi_decoder.decode_row(
                &mut [&mut buf_multi],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
        }

        let rect_flat = ImageRect::<i32>::from_raw(buf_flat.data.as_rect());
        let rect_multi = ImageRect::<i32>::from_raw(buf_multi.data.as_rect());
        for y in 0..ysize {
            assert_eq!(
                rect_flat.row(y),
                rect_multi.row(y),
                "Row {y} mismatch between FlatTree and MultiPropTree"
            );
        }
    }

    #[test]
    fn test_multiprop_3d_identical_to_flattree() {
        let tree = vec![
            TreeNode::Split {
                property: 4, // |top|
                val: 5,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 6, // top
                val: 0,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::North,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Split {
                property: 7, // left
                val: -10,
                left: 5,
                right: 6,
            },
            TreeNode::Leaf {
                predictor: Predictor::Select,
                offset: 0,
                multiplier: 1,
                id: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 3,
            },
        ];

        let xsize = 16;
        let ysize = 8;
        let bit_depth =
            crate::headers::bit_depth::BitDepth::default(&crate::headers::encodings::Empty {});
        let mut buf_flat =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();
        let mut buf_multi =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();

        let inner_flat =
            FlatTreeInner::new(tree.clone(), 16, 0, 0, xsize, ModularStorage::I32).unwrap();
        let mut flat_decoder = FlatTree::new(inner_flat, 13i32, ());

        let inner_multi = MultiPropTreeInner::<3>::new(&tree).unwrap();
        let mut multi_decoder = MultiPropTree::new(inner_multi, (), 13i32);

        let dummy_br = &mut BitReader::new(&[0u8; 16]);
        let dummy_histo = Histograms::dummy_for_test();
        let mut dummy_reader = SymbolReader::new(&dummy_histo, dummy_br, None).unwrap();

        for y in 0..ysize {
            flat_decoder.decode_row(
                &mut [&mut buf_flat],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
            multi_decoder.decode_row(
                &mut [&mut buf_multi],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
        }

        let rect_flat = ImageRect::<i32>::from_raw(buf_flat.data.as_rect());
        let rect_multi = ImageRect::<i32>::from_raw(buf_multi.data.as_rect());
        for y in 0..ysize {
            assert_eq!(
                rect_flat.row(y),
                rect_multi.row(y),
                "Row {y} mismatch between FlatTree and MultiPropTree 3D"
            );
        }
    }

    #[test]
    fn test_multiprop_wp_identical_to_flattree() {
        let tree = vec![
            TreeNode::Split {
                property: 6, // top
                val: 5,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 15, // WP property
                val: 0,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Weighted,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
        ];

        let xsize = 16;
        let ysize = 8;
        let bit_depth =
            crate::headers::bit_depth::BitDepth::default(&crate::headers::encodings::Empty {});
        let mut buf_flat =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();
        let mut buf_multi =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();

        let header = GroupHeader {
            use_global_tree: false,
            wp_header: crate::headers::modular::WeightedHeader::default(
                &crate::headers::encodings::Empty {},
            ),
            transforms: Vec::new(),
        };

        let wp_flat = WeightedPredictorState::new(&header.wp_header, xsize);
        let wp_multi = WeightedPredictorState::new(&header.wp_header, xsize);

        let inner_flat =
            FlatTreeInner::new(tree.clone(), 16, 0, 0, xsize, ModularStorage::I32).unwrap();
        let mut flat_decoder = FlatTree::new(inner_flat, 3i32, wp_flat);

        let inner_multi = MultiPropTreeInner::<2>::new(&tree).unwrap();
        let mut multi_decoder = MultiPropTree::new(inner_multi, wp_multi, 3i32);

        let dummy_br = &mut BitReader::new(&[0u8; 16]);
        let dummy_histo = Histograms::dummy_for_test();
        let mut dummy_reader = SymbolReader::new(&dummy_histo, dummy_br, None).unwrap();

        for y in 0..ysize {
            flat_decoder.decode_row(
                &mut [&mut buf_flat],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
            multi_decoder.decode_row(
                &mut [&mut buf_multi],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
        }

        let rect_flat = ImageRect::<i32>::from_raw(buf_flat.data.as_rect());
        let rect_multi = ImageRect::<i32>::from_raw(buf_multi.data.as_rect());
        for y in 0..ysize {
            assert_eq!(
                rect_flat.row(y),
                rect_multi.row(y),
                "Row {y} mismatch between FlatTree and MultiPropTree with WP"
            );
        }
    }

    #[test]
    fn test_multiprop_gradient_properties_identical_to_flattree() {
        let tree = vec![
            TreeNode::Split {
                property: 8, // local gradient (left - prev_grad)
                val: 0,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 9, // gradient (left + top - topleft)
                val: 10,
                left: 3,
                right: 4,
            },
            TreeNode::Leaf {
                predictor: Predictor::Zero,
                offset: 0,
                multiplier: 1,
                id: 0,
            },
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                id: 1,
            },
            TreeNode::Leaf {
                predictor: Predictor::West,
                offset: 0,
                multiplier: 1,
                id: 2,
            },
        ];

        let xsize = 16;
        let ysize = 8;
        let bit_depth =
            crate::headers::bit_depth::BitDepth::default(&crate::headers::encodings::Empty {});
        let mut buf_flat =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();
        let mut buf_multi =
            ModularChannel::new((xsize, ysize), ModularStorage::I32, bit_depth).unwrap();

        let inner_flat =
            FlatTreeInner::new(tree.clone(), 16, 0, 0, xsize, ModularStorage::I32).unwrap();
        let mut flat_decoder = FlatTree::new(inner_flat, 5i32, ());

        let inner_multi = MultiPropTreeInner::<2>::new(&tree).unwrap();
        let mut multi_decoder = MultiPropTree::new(inner_multi, (), 5i32);

        let dummy_br = &mut BitReader::new(&[0u8; 16]);
        let dummy_histo = Histograms::dummy_for_test();
        let mut dummy_reader = SymbolReader::new(&dummy_histo, dummy_br, None).unwrap();

        for y in 0..ysize {
            flat_decoder.decode_row(
                &mut [&mut buf_flat],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
            multi_decoder.decode_row(
                &mut [&mut buf_multi],
                0,
                &dummy_histo,
                &mut dummy_reader,
                dummy_br,
                y,
                xsize,
                None,
            );
        }

        let rect_flat = ImageRect::<i32>::from_raw(buf_flat.data.as_rect());
        let rect_multi = ImageRect::<i32>::from_raw(buf_multi.data.as_rect());
        for y in 0..ysize {
            assert_eq!(
                rect_flat.row(y),
                rect_multi.row(y),
                "Row {y} mismatch between FlatTree and MultiPropTree with gradient properties"
            );
        }
    }
}
