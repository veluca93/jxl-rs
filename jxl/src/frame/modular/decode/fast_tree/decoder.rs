// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::marker::PhantomData;

use super::backend::TreeBackend;
use super::mdd::{eval_mdd_fast, eval_mdd_pixel, MddPlan};
use super::nodes::{FastTreeLeaf, FastTreeNode, NodeEvalResult};
use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Histograms, SymbolReader};
use crate::frame::modular::decode::channel::{sync_scratch, ModularChannelDecoder};
use crate::frame::modular::decode::common::{precompute_references, References};
use crate::frame::modular::predict::{
    PredictionData, WeightedPredictorState, clamped_gradient,
};
use crate::frame::modular::{ModularChannel, ModularStorage, Predictor};
use crate::image::ImageRectMut;

pub struct FastTree<'a, B: TreeBackend> {
    pub(super) nodes: &'a [FastTreeNode],
    pub(super) mdd_plan: Option<MddPlan>,
    pub(super) mdd_tables: Option<&'a [u16]>,
    pub(super) mdd_leaves: &'a [FastTreeLeaf],
    pub(super) uses_wp: bool,
    pub(super) wp_state: Option<WeightedPredictorState<'a>>,
    pub(super) references: References<'a>,
    pub(super) needs_ext_neighbors: bool,
    pub(super) needs_toptop: bool,
    pub(super) needs_only_4_9: bool,
    pub(super) prev_prop9: i32,
    pub(super) storage: ModularStorage,
    pub(super) _backend: PhantomData<B>,
}

impl<'a, B: TreeBackend> FastTree<'a, B> {
    #[inline(always)]
    fn eval_leaf_static(
        nodes: &[FastTreeNode],
        mdd_plan: Option<&MddPlan>,
        mdd_tables: Option<&[u16]>,
        mdd_leaves: &[FastTreeLeaf],
        props: &[i32],
    ) -> FastTreeLeaf {
        if let Some(plan) = mdd_plan {
            let tables = mdd_tables.unwrap();
            let leaf_idx = eval_mdd_fast::<B>(plan, tables, props);
            if plan.all_simple_leaves {
                let pred_id = (leaf_idx >> 8) as u8;
                let context = (leaf_idx & 0xff) as u32;
                return FastTreeLeaf {
                    context,
                    predictor: Predictor::try_from(pred_id as u32).unwrap_or(Predictor::Zero),
                    offset: 0,
                    multiplier: 1,
                };
            }
            return mdd_leaves[leaf_idx as usize];
        }

        let mut node_idx = 0;
        loop {
            match nodes[node_idx].eval::<B>(props) {
                NodeEvalResult::Next(next_idx) => node_idx = next_idx,
                NodeEvalResult::Mdd(_stage_idx) => {
                    let tables = mdd_tables.unwrap();
                    let leaf_idx = eval_mdd_fast::<B>(mdd_plan.unwrap(), tables, props);
                    return mdd_leaves[leaf_idx as usize];
                }
                NodeEvalResult::Leaf(leaf) => return leaf,
            }
        }
    }

    #[inline(always)]
    fn eval_leaf(&self, props: &[i32]) -> FastTreeLeaf {
        Self::eval_leaf_static(
            self.nodes,
            self.mdd_plan.as_ref(),
            self.mdd_tables,
            self.mdd_leaves,
            props,
        )
    }

    #[inline(always)]
    fn decode_pixel_symbol(
        leaf: FastTreeLeaf,
        reader: &mut SymbolReader,
        histograms: &Histograms,
        br: &mut BitReader,
    ) -> i32 {
        let dec = reader.read_signed_clustered_inline(histograms, br, leaf.context as usize);
        dec.wrapping_mul(leaf.multiplier as i32).wrapping_add(leaf.offset)
    }

    fn decode_row_simple_4_9(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        y: usize,
        br: &mut BitReader,
        reader: &mut SymbolReader,
        histograms: &Histograms,
    ) {
        let mut prev_prop9 = self.prev_prop9;
        let mut props = [0i32; 12];
        let can_420 = !histograms.lz77_params().enabled && histograms.can_use_config_420_fast_path();

        if y == 0 {
            let mut left = 0i32;
            for r in row.iter_mut() {
                let abs_left = left.abs();
                props[4] = abs_left;
                props[5] = abs_left;
                props[6] = left;
                props[7] = left;
                props[8] = left.wrapping_sub(prev_prop9);
                props[9] = left;
                props[10] = 0;
                props[11] = 0;
                prev_prop9 = left;

                let leaf = self.eval_leaf(&props);
                let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                    reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                } else {
                    Self::decode_pixel_symbol(leaf, reader, histograms, br)
                };
                let pred = match leaf.predictor {
                    Predictor::Zero => 0,
                    Predictor::West => left,
                    Predictor::North => left,
                    Predictor::Select => left,
                    Predictor::Gradient => left,
                    other => other.predict_one(
                        PredictionData {
                            left,
                            top: left,
                            topleft: left,
                            ..Default::default()
                        },
                        0,
                    ) as i32,
                };
                let val = pred.wrapping_add(residual);
                *r = val;
                left = val;
            }
        } else if !row.is_empty() {
            let mut left = row_top[0];
            let mut topleft = left;
            for (r, &top) in row.iter_mut().zip(row_top) {
                let top_minus_tl = top.wrapping_sub(topleft);
                let prop9 = top_minus_tl.wrapping_add(left);
                props[4] = top.abs();
                props[5] = left.abs();
                props[6] = top;
                props[7] = left;
                props[8] = left.wrapping_sub(prev_prop9);
                props[9] = prop9;
                props[10] = left.wrapping_sub(topleft);
                props[11] = topleft.wrapping_sub(top);
                prev_prop9 = prop9;

                let leaf = self.eval_leaf(&props);
                let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                    reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                } else {
                    Self::decode_pixel_symbol(leaf, reader, histograms, br)
                };
                let pred = match leaf.predictor {
                    Predictor::Zero => 0,
                    Predictor::West => left,
                    Predictor::North => top,
                    Predictor::Select => {
                        let p = left.wrapping_add(top).wrapping_sub(topleft);
                        if (p - left).abs() < (p - top).abs() {
                            left
                        } else {
                            top
                        }
                    }
                    Predictor::Gradient => {
                        clamped_gradient(left as i64, top as i64, topleft as i64) as i32
                    }
                    other => other.predict_one(
                        PredictionData {
                            left,
                            top,
                            topleft,
                            ..Default::default()
                        },
                        0,
                    ) as i32,
                };
                let val = pred.wrapping_add(residual);
                *r = val;
                left = val;
                topleft = top;
            }
        }
        self.prev_prop9 = prev_prop9;
    }

    fn decode_row_simple_24props(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        y: usize,
        br: &mut BitReader,
        reader: &mut SymbolReader,
        histograms: &Histograms,
    ) {
        let mut prev_prop9 = self.prev_prop9;
        let references = &self.references;
        let mut props = [0i32; 24];

        let can_420 = !histograms.lz77_params().enabled && histograms.can_use_config_420_fast_path();

        if y == 0 {
            let mut left = 0i32;
            for (x, r) in row.iter_mut().enumerate() {
                let abs_left = left.abs();
                props[2] = 0;
                props[3] = x as i32;
                props[4] = abs_left;
                props[5] = abs_left;
                props[6] = left;
                props[7] = left;
                props[8] = left.wrapping_sub(prev_prop9);
                props[9] = left;
                props[10] = 0;
                props[11] = 0;
                prev_prop9 = left;

                let num_refs = references.num_ref_props;
                if num_refs != 0 {
                    let src = &references.data[x * num_refs..];
                    let copy_len = num_refs.min(8);
                    props[16..16 + copy_len].copy_from_slice(&src[..copy_len]);
                }

                let leaf = self.eval_leaf(&props);
                let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                    reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                } else {
                    Self::decode_pixel_symbol(leaf, reader, histograms, br)
                };
                let pred = match leaf.predictor {
                    Predictor::Zero => 0,
                    Predictor::West => left,
                    Predictor::North => left,
                    Predictor::Select => left,
                    Predictor::Gradient => left,
                    other => other.predict_one(
                        PredictionData {
                            left,
                            top: left,
                            topleft: left,
                            ..Default::default()
                        },
                        0,
                    ) as i32,
                };
                let val = pred.wrapping_add(residual);
                *r = val;
                left = val;
            }
        } else if !row.is_empty() {
            let mut left = row_top[0];
            let mut topleft = left;
            for (x, (r, &top)) in row.iter_mut().zip(row_top).enumerate() {
                let top_minus_tl = top.wrapping_sub(topleft);
                let prop9 = top_minus_tl.wrapping_add(left);
                props[2] = y as i32;
                props[3] = x as i32;
                props[4] = top.abs();
                props[5] = left.abs();
                props[6] = top;
                props[7] = left;
                props[8] = left.wrapping_sub(prev_prop9);
                props[9] = prop9;
                props[10] = left.wrapping_sub(topleft);
                props[11] = topleft.wrapping_sub(top);
                prev_prop9 = prop9;

                let num_refs = references.num_ref_props;
                if num_refs != 0 {
                    let src = &references.data[x * num_refs..];
                    let copy_len = num_refs.min(8);
                    props[16..16 + copy_len].copy_from_slice(&src[..copy_len]);
                }

                let leaf = self.eval_leaf(&props);
                let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                    reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                } else {
                    Self::decode_pixel_symbol(leaf, reader, histograms, br)
                };
                let pred = match leaf.predictor {
                    Predictor::Zero => 0,
                    Predictor::West => left,
                    Predictor::North => top,
                    Predictor::Select => {
                        let p = left.wrapping_add(top).wrapping_sub(topleft);
                        if (p - left).abs() < (p - top).abs() {
                            left
                        } else {
                            top
                        }
                    }
                    Predictor::Gradient => {
                        clamped_gradient(left as i64, top as i64, topleft as i64) as i32
                    }
                    other => other.predict_one(
                        PredictionData {
                            left,
                            top,
                            topleft,
                            ..Default::default()
                        },
                        0,
                    ) as i32,
                };
                let val = pred.wrapping_add(residual);
                *r = val;
                left = val;
                topleft = top;
            }
        }
        self.prev_prop9 = prev_prop9;
    }

    fn decode_row_wp(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        row_toptop: &[i32],
        y: usize,
        xsize: usize,
        br: &mut BitReader,
        reader: &mut SymbolReader,
        histograms: &Histograms,
    ) {
        let mut prev_prop9 = self.prev_prop9;
        let nodes = self.nodes;
        let mdd_plan = self.mdd_plan.as_ref();
        let mdd_tables = self.mdd_tables;
        let mdd_leaves = self.mdd_leaves;
        let references = &self.references;
        let wp_state = self.wp_state.as_mut().unwrap();
        let mut props = [0i32; 24];

        let can_420 = !histograms.lz77_params().enabled && histograms.can_use_config_420_fast_path();

        let (x0, x1) = if y == 0 || xsize <= 4 {
            (0, 0)
        } else {
            (2, xsize - 2)
        };

        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..x0 {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let (wp_pred, wp_prop) = wp_state.predict_and_property((x, y), &prediction_data);

            Self::populate_props_24(
                &mut props,
                &prediction_data,
                (x, y),
                wp_prop,
                &mut prev_prop9,
                references,
            );

            let leaf = Self::eval_leaf_static(nodes, mdd_plan, mdd_tables, mdd_leaves, &props);
            let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
            } else {
                Self::decode_pixel_symbol(leaf, reader, histograms, br)
            };
            let pred = match leaf.predictor {
                Predictor::Weighted => wp_pred as i32,
                Predictor::Gradient => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                Predictor::Zero => 0,
                Predictor::West => prediction_data.left,
                Predictor::North => prediction_data.top,
                other => other.predict_one(prediction_data, wp_pred) as i32,
            };
            last = pred.wrapping_add(residual);
            wp_state.update_errors(last, (x, y));
            row[x] = last;
        }

        if x0 < x1 {
            let row_mid = &mut row[x0..x1];
            let row_top_trr = &row_top[x0 + 2..x1 + 2];
            let row_toptop_mid = &row_toptop[x0..x1];
            for (i, ((r, &trr), &tt)) in row_mid
                .iter_mut()
                .zip(row_top_trr)
                .zip(row_toptop_mid)
                .enumerate()
            {
                let x = x0 + i;
                prediction_data = prediction_data.update_for_interior_row(last, trr, tt);
                let (wp_pred, wp_prop) = wp_state.predict_and_property((x, y), &prediction_data);

                Self::populate_props_24(
                    &mut props,
                    &prediction_data,
                    (x, y),
                    wp_prop,
                    &mut prev_prop9,
                    references,
                );

                let leaf = Self::eval_leaf_static(nodes, mdd_plan, mdd_tables, mdd_leaves, &props);
                let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                    reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                } else {
                    Self::decode_pixel_symbol(leaf, reader, histograms, br)
                };
                let pred = match leaf.predictor {
                    Predictor::Weighted => wp_pred as i32,
                    Predictor::Gradient => clamped_gradient(
                        prediction_data.left as i64,
                        prediction_data.top as i64,
                        prediction_data.topleft as i64,
                    ) as i32,
                    Predictor::Zero => 0,
                    Predictor::West => prediction_data.left,
                    Predictor::North => prediction_data.top,
                    other => other.predict_one(prediction_data, wp_pred) as i32,
                };
                let val = pred.wrapping_add(residual);
                wp_state.update_errors(val, (x, y));
                *r = val;
                last = val;
            }
        }

        for x in x1..xsize {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let (wp_pred, wp_prop) = wp_state.predict_and_property((x, y), &prediction_data);

            Self::populate_props_24(
                &mut props,
                &prediction_data,
                (x, y),
                wp_prop,
                &mut prev_prop9,
                references,
            );

            let leaf = Self::eval_leaf_static(nodes, mdd_plan, mdd_tables, mdd_leaves, &props);
            let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
            } else {
                Self::decode_pixel_symbol(leaf, reader, histograms, br)
            };
            let pred = match leaf.predictor {
                Predictor::Weighted => wp_pred as i32,
                Predictor::Gradient => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                Predictor::Zero => 0,
                Predictor::West => prediction_data.left,
                Predictor::North => prediction_data.top,
                other => other.predict_one(prediction_data, wp_pred) as i32,
            };
            let val = pred.wrapping_add(residual);
            wp_state.update_errors(val, (x, y));
            row[x] = val;
        }

        self.prev_prop9 = prev_prop9;
    }

    fn decode_row_ext(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        row_toptop: &[i32],
        y: usize,
        xsize: usize,
        br: &mut BitReader,
        reader: &mut SymbolReader,
        histograms: &Histograms,
    ) {
        let mut prev_prop9 = self.prev_prop9;
        let references = &self.references;
        let mut props = [0i32; 24];

        let can_420 = !histograms.lz77_params().enabled && histograms.can_use_config_420_fast_path();

        let (x0, x1) = if y == 0 || xsize <= 4 {
            (0, 0)
        } else {
            (2, xsize - 2)
        };

        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..x0 {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            Self::populate_props_24(
                &mut props,
                &prediction_data,
                (x, y),
                0,
                &mut prev_prop9,
                references,
            );
            let leaf = self.eval_leaf(&props);
            let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
            } else {
                Self::decode_pixel_symbol(leaf, reader, histograms, br)
            };
            let pred = match leaf.predictor {
                Predictor::Zero => 0,
                Predictor::West => prediction_data.left,
                Predictor::North => prediction_data.top,
                Predictor::Gradient => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                other => other.predict_one(prediction_data, 0) as i32,
            };
            last = pred.wrapping_add(residual);
            row[x] = last;
        }

        if x0 < x1 {
            let row_mid = &mut row[x0..x1];
            let row_top_trr = &row_top[x0 + 2..x1 + 2];
            if self.needs_toptop {
                let row_toptop_mid = &row_toptop[x0..x1];
                for (i, ((r, &trr), &tt)) in row_mid
                    .iter_mut()
                    .zip(row_top_trr)
                    .zip(row_toptop_mid)
                    .enumerate()
                {
                    let x = x0 + i;
                    prediction_data = prediction_data.update_for_interior_row(last, trr, tt);
                    Self::populate_props_24(
                        &mut props,
                        &prediction_data,
                        (x, y),
                        0,
                        &mut prev_prop9,
                        references,
                    );
                    let leaf = self.eval_leaf(&props);
                    let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                        reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                    } else {
                        Self::decode_pixel_symbol(leaf, reader, histograms, br)
                    };
                    let pred = match leaf.predictor {
                        Predictor::Zero => 0,
                        Predictor::West => prediction_data.left,
                        Predictor::North => prediction_data.top,
                        Predictor::Gradient => clamped_gradient(
                            prediction_data.left as i64,
                            prediction_data.top as i64,
                            prediction_data.topleft as i64,
                        ) as i32,
                        other => other.predict_one(prediction_data, 0) as i32,
                    };
                    let val = pred.wrapping_add(residual);
                    *r = val;
                    last = val;
                }
            } else {
                for (i, (r, &trr)) in row_mid.iter_mut().zip(row_top_trr).enumerate() {
                    let x = x0 + i;
                    prediction_data = prediction_data.update_for_interior_row(last, trr, 0);
                    Self::populate_props_24(
                        &mut props,
                        &prediction_data,
                        (x, y),
                        0,
                        &mut prev_prop9,
                        references,
                    );
                    let leaf = self.eval_leaf(&props);
                    let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                        reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
                    } else {
                        Self::decode_pixel_symbol(leaf, reader, histograms, br)
                    };
                    let pred = match leaf.predictor {
                        Predictor::Zero => 0,
                        Predictor::West => prediction_data.left,
                        Predictor::North => prediction_data.top,
                        Predictor::Gradient => clamped_gradient(
                            prediction_data.left as i64,
                            prediction_data.top as i64,
                            prediction_data.topleft as i64,
                        ) as i32,
                        other => other.predict_one(prediction_data, 0) as i32,
                    };
                    let val = pred.wrapping_add(residual);
                    *r = val;
                    last = val;
                }
            }
        }

        for x in x1..xsize {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            Self::populate_props_24(
                &mut props,
                &prediction_data,
                (x, y),
                0,
                &mut prev_prop9,
                references,
            );
            let leaf = self.eval_leaf(&props);
            let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
            } else {
                Self::decode_pixel_symbol(leaf, reader, histograms, br)
            };
            let pred = match leaf.predictor {
                Predictor::Zero => 0,
                Predictor::West => prediction_data.left,
                Predictor::North => prediction_data.top,
                Predictor::Gradient => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                other => other.predict_one(prediction_data, 0) as i32,
            };
            let val = pred.wrapping_add(residual);
            row[x] = val;
        }

        self.prev_prop9 = prev_prop9;
    }

    #[inline(always)]
    fn get_single_mdd_prop(
        prop: u8,
        data: &PredictionData,
        pos: (usize, usize),
        wp_prop: i32,
        prev_prop9: i32,
        references: &References,
    ) -> i32 {
        match prop {
            9 => data.left.wrapping_add(data.top).wrapping_sub(data.topleft),
            10 => data.left.wrapping_sub(data.topleft),
            11 => data.topleft.wrapping_sub(data.top),
            12 => data.top.wrapping_sub(data.topright),
            13 => data.top.wrapping_sub(data.toptop),
            14 => data.left.wrapping_sub(data.leftleft),
            15 => wp_prop,
            4 => data.top.wrapping_abs(),
            5 => data.left.wrapping_abs(),
            6 => data.top,
            7 => data.left,
            8 => data.left.wrapping_sub(prev_prop9),
            2 => pos.1 as i32,
            3 => pos.0 as i32,
            p if p >= 16 => {
                let num_refs = references.num_ref_props;
                if num_refs != 0 && ((p - 16) as usize) < num_refs {
                    references.data[pos.0 * num_refs + (p - 16) as usize]
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    #[inline(always)]
    fn extract_mdd_props(
        plan: &MddPlan,
        data: &PredictionData,
        pos: (usize, usize),
        wp_prop: i32,
        prev_prop9: i32,
        references: &References,
    ) -> [i32; 6] {
        let mut v = [0i32; 6];
        match plan.active_dims {
            1 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
            }
            2 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
                v[1] = Self::get_single_mdd_prop(plan.props[1], data, pos, wp_prop, prev_prop9, references);
            }
            3 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
                v[1] = Self::get_single_mdd_prop(plan.props[1], data, pos, wp_prop, prev_prop9, references);
                v[2] = Self::get_single_mdd_prop(plan.props[2], data, pos, wp_prop, prev_prop9, references);
            }
            4 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
                v[1] = Self::get_single_mdd_prop(plan.props[1], data, pos, wp_prop, prev_prop9, references);
                v[2] = Self::get_single_mdd_prop(plan.props[2], data, pos, wp_prop, prev_prop9, references);
                v[3] = Self::get_single_mdd_prop(plan.props[3], data, pos, wp_prop, prev_prop9, references);
            }
            5 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
                v[1] = Self::get_single_mdd_prop(plan.props[1], data, pos, wp_prop, prev_prop9, references);
                v[2] = Self::get_single_mdd_prop(plan.props[2], data, pos, wp_prop, prev_prop9, references);
                v[3] = Self::get_single_mdd_prop(plan.props[3], data, pos, wp_prop, prev_prop9, references);
                v[4] = Self::get_single_mdd_prop(plan.props[4], data, pos, wp_prop, prev_prop9, references);
            }
            6 => {
                v[0] = Self::get_single_mdd_prop(plan.props[0], data, pos, wp_prop, prev_prop9, references);
                v[1] = Self::get_single_mdd_prop(plan.props[1], data, pos, wp_prop, prev_prop9, references);
                v[2] = Self::get_single_mdd_prop(plan.props[2], data, pos, wp_prop, prev_prop9, references);
                v[3] = Self::get_single_mdd_prop(plan.props[3], data, pos, wp_prop, prev_prop9, references);
                v[4] = Self::get_single_mdd_prop(plan.props[4], data, pos, wp_prop, prev_prop9, references);
                v[5] = Self::get_single_mdd_prop(plan.props[5], data, pos, wp_prop, prev_prop9, references);
            }
            _ => {}
        }
        v
    }

    #[inline(always)]
    fn decode_mdd_pixel(
        plan: &MddPlan,
        tables: &[u16],
        mdd_leaves: &[FastTreeLeaf],
        prediction_data: &PredictionData,
        pos: (usize, usize),
        wp_pred: i64,
        wp_prop: i32,
        prev_prop9: i32,
        can_420: bool,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
        references: &References,
    ) -> i32 {
        let v = Self::extract_mdd_props(
            plan,
            prediction_data,
            pos,
            wp_prop,
            prev_prop9,
            references,
        );
        let leaf_idx = eval_mdd_pixel(plan, tables, &v);
        let (pred, residual) = if plan.all_simple_leaves {
            let cluster = (leaf_idx & 0xff) as usize;
            let pred_id = (leaf_idx >> 8) as u8;
            let residual = if can_420 {
                reader.read_signed_clustered_config_420(histograms, br, cluster)
            } else {
                reader.read_signed_clustered_inline(histograms, br, cluster)
            };
            let pred = match pred_id {
                p if p == Predictor::Weighted as u8 => wp_pred as i32,
                p if p == Predictor::Gradient as u8 => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                p if p == Predictor::Zero as u8 => 0,
                p if p == Predictor::West as u8 => prediction_data.left,
                p if p == Predictor::North as u8 => prediction_data.top,
                other => Predictor::try_from(other as u32).unwrap_or(Predictor::Zero).predict_one(*prediction_data, wp_pred) as i32,
            };
            (pred, residual)
        } else {
            let leaf = mdd_leaves[leaf_idx as usize];
            let residual = if can_420 && leaf.multiplier == 1 && leaf.offset == 0 {
                reader.read_signed_clustered_config_420(histograms, br, leaf.context as usize)
            } else {
                Self::decode_pixel_symbol(leaf, reader, histograms, br)
            };
            let pred = match leaf.predictor {
                Predictor::Weighted => wp_pred as i32,
                Predictor::Gradient => clamped_gradient(
                    prediction_data.left as i64,
                    prediction_data.top as i64,
                    prediction_data.topleft as i64,
                ) as i32,
                Predictor::Zero => 0,
                Predictor::West => prediction_data.left,
                Predictor::North => prediction_data.top,
                other => other.predict_one(*prediction_data, wp_pred) as i32,
            };
            (pred, residual)
        };
        pred.wrapping_add(residual)
    }

    fn decode_row_mdd(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        row_toptop: &[i32],
        y: usize,
        xsize: usize,
        br: &mut BitReader,
        reader: &mut SymbolReader,
        histograms: &Histograms,
    ) {
        let plan = self.mdd_plan.as_ref().unwrap();
        let tables = self.mdd_tables.unwrap();
        let mdd_leaves = self.mdd_leaves;
        let references = &self.references;
        let uses_wp = self.uses_wp;
        let can_420 = !histograms.lz77_params().enabled && histograms.can_use_config_420_fast_path();
        let mut prev_prop9 = self.prev_prop9;

        let (x0, x1) = if y == 0 || xsize <= 4 {
            (0, 0)
        } else {
            (2, xsize - 2)
        };

        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..x0 {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let (wp_pred, wp_prop) = if uses_wp {
                self.wp_state.as_mut().unwrap().predict_and_property((x, y), &prediction_data)
            } else {
                (0, 0)
            };
            last = Self::decode_mdd_pixel(
                plan,
                tables,
                mdd_leaves,
                &prediction_data,
                (x, y),
                wp_pred,
                wp_prop,
                prev_prop9,
                can_420,
                reader,
                br,
                histograms,
                references,
            );
            if uses_wp {
                self.wp_state.as_mut().unwrap().update_errors(last, (x, y));
            }
            prev_prop9 = prediction_data.left.wrapping_add(prediction_data.top).wrapping_sub(prediction_data.topleft);
            row[x] = last;
        }

        if uses_wp && x0 < x1 {
            let wp_state = self.wp_state.as_mut().unwrap();
            let row_mid = &mut row[x0..x1];
            let row_top_trr = &row_top[x0 + 2..x1 + 2];
            let row_toptop_mid = &row_toptop[x0..x1];
            for (i, ((r, &trr), &tt)) in row_mid
                .iter_mut()
                .zip(row_top_trr)
                .zip(row_toptop_mid)
                .enumerate()
            {
                let x = x0 + i;
                prediction_data = prediction_data.update_for_interior_row(last, trr, tt);
                let (wp_pred, wp_prop) = wp_state.predict_and_property((x, y), &prediction_data);
                let val = Self::decode_mdd_pixel(
                    plan,
                    tables,
                    mdd_leaves,
                    &prediction_data,
                    (x, y),
                    wp_pred,
                    wp_prop,
                    prev_prop9,
                    can_420,
                    reader,
                    br,
                    histograms,
                    references,
                );
                wp_state.update_errors(val, (x, y));
                prev_prop9 = prediction_data.left.wrapping_add(prediction_data.top).wrapping_sub(prediction_data.topleft);
                *r = val;
                last = val;
            }
        } else if !uses_wp && x0 < x1 {
            let row_mid = &mut row[x0..x1];
            let row_top_trr = &row_top[x0 + 2..x1 + 2];
            for (i, (r, &trr)) in row_mid.iter_mut().zip(row_top_trr).enumerate() {
                let x = x0 + i;
                prediction_data = prediction_data.update_for_interior_row(last, trr, 0);
                let val = Self::decode_mdd_pixel(
                    plan,
                    tables,
                    mdd_leaves,
                    &prediction_data,
                    (x, y),
                    0,
                    0,
                    prev_prop9,
                    can_420,
                    reader,
                    br,
                    histograms,
                    references,
                );
                prev_prop9 = prediction_data.left.wrapping_add(prediction_data.top).wrapping_sub(prediction_data.topleft);
                *r = val;
                last = val;
            }
        }

        for x in x1..xsize {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let (wp_pred, wp_prop) = if uses_wp {
                self.wp_state.as_mut().unwrap().predict_and_property((x, y), &prediction_data)
            } else {
                (0, 0)
            };
            let val = Self::decode_mdd_pixel(
                plan,
                tables,
                mdd_leaves,
                &prediction_data,
                (x, y),
                wp_pred,
                wp_prop,
                prev_prop9,
                can_420,
                reader,
                br,
                histograms,
                references,
            );
            if uses_wp {
                self.wp_state.as_mut().unwrap().update_errors(val, (x, y));
            }
            prev_prop9 = prediction_data.left.wrapping_add(prediction_data.top).wrapping_sub(prediction_data.topleft);
            row[x] = val;
        }

        self.prev_prop9 = prev_prop9;
    }

    fn populate_props_24(
        props: &mut [i32; 24],
        data: &PredictionData,
        pos: (usize, usize),
        wp_prop: i32,
        prev_prop9: &mut i32,
        references: &References,
    ) {
        let (left, top, topleft) = (data.left, data.top, data.topleft);
        let top_minus_tl = top.wrapping_sub(topleft);
        let prop9 = top_minus_tl.wrapping_add(left);

        props[0] = 0;
        props[1] = 0;
        props[2] = pos.1 as i32; // y
        props[3] = pos.0 as i32; // x
        props[4] = top.wrapping_abs();
        props[5] = left.wrapping_abs();
        props[6] = top;
        props[7] = left;
        props[8] = left.wrapping_sub(*prev_prop9);
        props[9] = prop9;
        props[10] = left.wrapping_sub(topleft);
        props[11] = topleft.wrapping_sub(top);
        props[12] = top.wrapping_sub(data.topright);
        props[13] = top.wrapping_sub(data.toptop);
        props[14] = left.wrapping_sub(data.leftleft);
        props[15] = wp_prop;

        *prev_prop9 = prop9;

        let num_refs = references.num_ref_props;
        if num_refs != 0 {
            let src = &references.data[pos.0 * num_refs..];
            let copy_len = num_refs.min(8);
            props[16..16 + copy_len].copy_from_slice(&src[..copy_len]);
        }
    }
}

macro_rules! rows {
    ($row: ident, $row_top: ident, $row_toptop: ident, $buffers: expr, $y: expr, $scratch: expr, $xsize: expr) => {
        let mut rect;
        let ($row, $row_top, $row_toptop): (&mut [i32], &[i32], &[i32]) = if let Some(scratch) =
            $scratch.as_deref_mut()
        {
            let [row, row_top, row_toptop] = scratch;
            (
                &mut row[..$xsize],
                &row_top[..$xsize],
                if $y == 1 {
                    &row_top[..$xsize]
                } else {
                    &row_toptop[..$xsize]
                },
            )
        } else {
            rect = ImageRectMut::<i32>::from_raw($buffers.data.as_rect_mut());
            match $y {
                0 => (&mut rect.row(0)[..$xsize], &[], &[]),
                1 => {
                    let [row, row_top] = rect.distinct_rows_mut([1, 0]);
                    (&mut row[..$xsize], &row_top[..$xsize], &row_top[..$xsize])
                }
                _ => {
                    let [row, row_top, row_toptop] = rect.distinct_rows_mut([$y, $y - 1, $y - 2]);
                    (
                        &mut row[..$xsize],
                        &row_top[..$xsize],
                        &row_toptop[..$xsize],
                    )
                }
            }
        };
    };
}

impl<'a, B: TreeBackend> ModularChannelDecoder for FastTree<'a, B> {
    #[inline(always)]
    fn needs_toptop(&self) -> bool {
        self.needs_toptop
    }

    fn init_row(&mut self, buffers: &mut [&mut ModularChannel], chan: usize, y: usize) {
        if self.references.num_ref_props > 0 {
            precompute_references(buffers, chan, y, &mut self.references, self.storage);
        }
        self.prev_prop9 = 0;
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
        let (wp_pred, wp_prop) = self
            .wp_state
            .as_mut()
            .map(|wp| wp.predict_and_property(pos, &prediction_data))
            .unwrap_or((0, 0));

        let mut props = [0i32; 24];
        Self::populate_props_24(
            &mut props,
            &prediction_data,
            pos,
            wp_prop,
            &mut self.prev_prop9,
            &self.references,
        );

        let leaf = Self::eval_leaf_static(
            self.nodes,
            self.mdd_plan.as_ref(),
            self.mdd_tables,
            self.mdd_leaves,
            &props,
        );
        let residual = Self::decode_pixel_symbol(leaf, reader, histograms, br);
        let pred = leaf.predictor.predict_one(prediction_data, wp_pred) as i32;
        let val = pred.wrapping_add(residual);
        if let Some(wp) = self.wp_state.as_mut() {
            wp.update_errors(val, pos);
        }
        val
    }

    #[allow(clippy::too_many_arguments)]
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
        self.init_row(buffers, chan, y);

        rows!(row, row_top, row_toptop, buffers[chan], y, scratch, xsize);

        if self.mdd_plan.is_some() {
            self.decode_row_mdd(row, row_top, row_toptop, y, xsize, br, reader, histograms);
        } else if self.needs_only_4_9 && !self.uses_wp && chan == 0 {
            self.decode_row_simple_4_9(row, row_top, y, br, reader, histograms);
        } else if !self.needs_ext_neighbors && !self.uses_wp {
            self.decode_row_simple_24props(row, row_top, y, br, reader, histograms);
        } else if self.uses_wp {
            self.decode_row_wp(row, row_top, row_toptop, y, xsize, br, reader, histograms);
        } else {
            self.decode_row_ext(row, row_top, row_toptop, y, xsize, br, reader, histograms);
        }

        sync_scratch(buffers[chan], y, scratch);
    }
}

