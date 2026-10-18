// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![allow(dead_code)]

use super::backend::TreeBackend;
use super::nodes::FastTreeLeaf;
use crate::error::Result;
use crate::frame::modular::tree::TreeNode;

pub const MAX_Q_SPAN: usize = 2048;
pub const MIN_Q_VAL: i32 = -(MAX_Q_SPAN as i32 / 2);
pub const MAX_Q_VAL: i32 = (MAX_Q_SPAN as i32 / 2) - 1;
pub const Q_STRIDE: usize = MAX_Q_SPAN + 32;
pub const MAX_STAGE0_FUSED_ENTRIES: usize = 4096;
pub const MAX_FULL_LUT_ENTRIES: usize = 4096;
pub const MAX_FUSED_2D_ENTRIES: usize = 8192;
pub const MAX_MDD_TOTAL_ENTRIES: usize = 16384;
pub const MAX_RESIDUAL_SUBTREES: usize = 64;
pub const TABLES_OFFSET: usize = 6 * Q_STRIDE;
pub const MDD_BUF_LEN: usize = TABLES_OFFSET + MAX_MDD_TOTAL_ENTRIES;

const DAG_HASH_BITS: usize = 14;
const DAG_HASH_SIZE: usize = 1 << DAG_HASH_BITS;
const DAG_HASH_MASK: usize = DAG_HASH_SIZE - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum MddStageKind {
    Lut1D = 0,
    S1_2D = 1,
    S1_3D = 2,
    S1_4D = 3,
    S1_5D = 4,
    S1_6D = 5,
    S2_4D = 6,
    S2_5D = 7,
    S3_4D = 8,
    S3_5D = 9,
    S3_6D = 10,
    S4_5D = 11,
    S5_6D = 12,
}

#[derive(Clone, Copy, Debug)]
pub struct MddPlan {
    pub kind: MddStageKind,
    pub props: [u8; 6],
    pub active_dims: usize,
    pub min: [i32; 6],
    pub max: [i32; 6],
    pub bias: [i32; 6],
    pub all_simple_leaves: bool,
}

impl Default for MddPlan {
    fn default() -> Self {
        Self {
            kind: MddStageKind::Lut1D,
            props: [0; 6],
            active_dims: 0,
            min: [MIN_Q_VAL; 6],
            max: [MAX_Q_VAL; 6],
            bias: [0; 6],
            all_simple_leaves: false,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct HashSlot {
    key: u64,
    val: u32,
    generation: u32,
}

#[derive(Clone, Copy, Default)]
struct MemoSlot {
    key: u32,
    val: u32,
    generation: u32,
}

#[derive(Clone, Copy, Default)]
struct DagNode {
    rank: u8,
    threshold_idx: u16,
    left: u32,
    right: u32,
}

pub struct MddCompiler {
    generation: u32,
    leaf_generation: u32,
    dag_nodes: Vec<DagNode>,
    dag_table: Vec<HashSlot>,
    state_table: Vec<HashSlot>,
    leaf_table: Vec<HashSlot>,
    leaf_tuples: Vec<(u8, i32, u32, u32)>,
    memo: Vec<MemoSlot>,
    level_tables: [Vec<u16>; 5],
    level_states: [Vec<u32>; 5],
    residual_cache: Vec<(u32, u16)>,
    dirty_start: [u16; 6],
}

struct MddScratchInner {
    compiler: MddCompiler,
    buf: Box<[u16; MDD_BUF_LEN]>,
}

#[derive(Default)]
pub struct MddScratch {
    thresholds: [Vec<i32>; 6],
    lut1d: Option<Box<[u16; MAX_Q_SPAN]>>,
    inner: Option<Box<MddScratchInner>>,
}

impl MddScratch {
    #[inline(always)]
    pub fn get_or_init_lut1d(&mut self) -> &mut [u16; MAX_Q_SPAN] {
        self.lut1d
            .get_or_insert_with(|| crate::util::box_array(0u16))
    }

    pub fn get_or_init(
        &mut self,
    ) -> (
        &mut [Vec<i32>; 6],
        &mut MddCompiler,
        &mut [u16; MDD_BUF_LEN],
    ) {
        let inner = self.inner.get_or_insert_with(|| {
            Box::new(MddScratchInner {
                compiler: MddCompiler {
                    generation: 1,
                    leaf_generation: 1,
                    dag_nodes: Vec::new(),
                    dag_table: Vec::new(),
                    state_table: Vec::new(),
                    leaf_table: Vec::new(),
                    leaf_tuples: Vec::new(),
                    memo: Vec::new(),
                    level_tables: [const { Vec::new() }; 5],
                    level_states: [const { Vec::new() }; 5],
                    residual_cache: Vec::new(),
                    dirty_start: [MAX_Q_SPAN as u16; 6],
                },
                buf: crate::util::box_array(0u16),
            })
        });
        (&mut self.thresholds, &mut inner.compiler, &mut inner.buf)
    }
}

impl MddCompiler {
    #[inline(always)]
    fn ensure_leaf_table(&mut self) {
        if self.leaf_table.is_empty() {
            self.leaf_table.resize(DAG_HASH_SIZE, HashSlot::default());
        }
    }

    #[inline(always)]
    fn ensure_dag_tables(&mut self) {
        if self.dag_table.is_empty() {
            self.dag_table.resize(DAG_HASH_SIZE, HashSlot::default());
            self.state_table.resize(DAG_HASH_SIZE, HashSlot::default());
        }
    }

    #[inline(always)]
    fn bump_gen(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.dag_table.fill(HashSlot::default());
            self.state_table.fill(HashSlot::default());
            self.memo.fill(MemoSlot::default());
            self.generation = 1;
        }
    }

    #[inline(always)]
    fn bump_leaf_gen(&mut self) {
        self.leaf_generation = self.leaf_generation.wrapping_add(1);
        if self.leaf_generation == 0 {
            self.leaf_table.fill(HashSlot::default());
            self.leaf_tuples.clear();
            self.leaf_generation = 1;
        } else {
            self.leaf_tuples.clear();
        }
    }

    #[inline(always)]
    fn hash_u64(k: u64) -> usize {
        (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - DAG_HASH_BITS)) as usize
    }

    #[inline(always)]
    fn intern_leaf(
        &mut self,
        node: TreeNode,
        leaves: &mut Vec<FastTreeLeaf>,
        all_simple_leaves: bool,
    ) -> u16 {
        let TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id,
        } = node
        else {
            unreachable!()
        };
        if all_simple_leaves {
            return ((predictor as u16) << 8) | (id as u16);
        }
        let tuple = (predictor as u8, offset, multiplier, id);
        let key = ((predictor as u64) << 56)
            ^ ((offset as u32 as u64) << 24)
            ^ ((multiplier as u64) << 12)
            ^ (id as u64);
        let mut idx = Self::hash_u64(key);
        let leaf_gen = self.leaf_generation;
        let mut probes = 0;
        loop {
            let slot = &mut self.leaf_table[idx & DAG_HASH_MASK];
            if slot.generation != leaf_gen {
                let leaf_idx = leaves.len() as u16;
                leaves.push(FastTreeLeaf {
                    context: id,
                    predictor,
                    offset,
                    multiplier,
                });
                let tuple_idx = self.leaf_tuples.len() as u32;
                self.leaf_tuples.push(tuple);
                slot.generation = leaf_gen;
                slot.key = ((leaf_idx as u64) << 32) | (tuple_idx as u64);
                slot.val = tuple_idx;
                return leaf_idx;
            }
            let tuple_idx = slot.val as usize;
            if self.leaf_tuples[tuple_idx] == tuple {
                return (slot.key >> 32) as u16;
            }
            probes += 1;
            if probes >= DAG_HASH_SIZE {
                panic!("leaf_table overflow");
            }
            idx = (idx + 1) & DAG_HASH_MASK;
        }
    }

    #[inline(always)]
    fn intern_dag(&mut self, key: u64, node: DagNode) -> u32 {
        let mut idx = Self::hash_u64(key);
        let generation = self.generation;
        let mut probes = 0;
        loop {
            let slot = &mut self.dag_table[idx & DAG_HASH_MASK];
            if slot.generation != generation {
                let id = self.dag_nodes.len() as u32;
                self.dag_nodes.push(node);
                slot.generation = generation;
                slot.key = key;
                slot.val = id;
                return id;
            }
            if slot.key == key {
                return slot.val;
            }
            probes += 1;
            if probes >= DAG_HASH_SIZE {
                panic!("dag_table overflow");
            }
            idx = (idx + 1) & DAG_HASH_MASK;
        }
    }

    fn build_dag(
        &mut self,
        tree_nodes: &[TreeNode],
        thresholds: &[Vec<i32>; 6],
        rank_of_prop: &[u8; 256],
        leaves: &mut Vec<FastTreeLeaf>,
        all_simple_leaves: bool,
    ) -> Option<u32> {
        let n = tree_nodes.len();
        if n == 0 {
            return None;
        }
        let mut node_to_dag = vec![0u32; n];
        for i in (0..n).rev() {
            match tree_nodes[i] {
                TreeNode::Leaf { .. } => {
                    let leaf_idx = self.intern_leaf(tree_nodes[i], leaves, all_simple_leaves);
                    node_to_dag[i] = 0x8000_0000 | (leaf_idx as u32);
                }
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    let rank = rank_of_prop[property as usize];
                    if rank == 0xff {
                        return None;
                    }
                    let Ok(t_idx) = thresholds[rank as usize].binary_search(&val) else {
                        return None;
                    };
                    let l_idx = left as usize;
                    let r_idx = right as usize;
                    if l_idx >= n || r_idx >= n || l_idx <= i || r_idx <= i {
                        return None;
                    }
                    let l_id = node_to_dag[l_idx];
                    let r_id = node_to_dag[r_idx];
                    if l_id == r_id {
                        node_to_dag[i] = l_id;
                        continue;
                    }
                    let key = ((rank as u64) << 56)
                        ^ ((t_idx as u64) << 40)
                        ^ ((l_id as u64) << 20)
                        ^ (r_id as u64);
                    let id = self.intern_dag(
                        key,
                        DagNode {
                            rank,
                            threshold_idx: t_idx as u16,
                            left: l_id,
                            right: r_id,
                        },
                    );
                    node_to_dag[i] = id;
                }
            }
        }
        node_to_dag.first().copied()
    }

    fn restrict_rec(&mut self, node_id: u32, rank: usize, bucket: u16) -> u32 {
        if (node_id & 0x8000_0000) != 0 {
            return node_id;
        }
        let memo_idx = node_id as usize;
        let memo = self.memo[memo_idx];
        if memo.generation == self.generation && memo.key == bucket as u32 {
            return memo.val;
        }
        let node = self.dag_nodes[node_id as usize];
        let r = node.rank as usize;
        let res = if r == rank {
            let child = if bucket > node.threshold_idx {
                node.left
            } else {
                node.right
            };
            self.restrict_rec(child, rank, bucket)
        } else {
            let new_l = self.restrict_rec(node.left, rank, bucket);
            let new_r = self.restrict_rec(node.right, rank, bucket);
            if new_l == new_r {
                new_l
            } else if new_l == node.left && new_r == node.right {
                node_id
            } else {
                let key = ((node.rank as u64) << 56)
                    ^ ((node.threshold_idx as u64) << 40)
                    ^ ((new_l as u64) << 20)
                    ^ (new_r as u64);
                self.intern_dag(
                    key,
                    DagNode {
                        rank: node.rank,
                        threshold_idx: node.threshold_idx,
                        left: new_l,
                        right: new_r,
                    },
                )
            }
        };

        self.memo[memo_idx] = MemoSlot {
            key: bucket as u32,
            val: res,
            generation: self.generation,
        };
        res
    }

    fn restrict_level(
        &mut self,
        root: u32,
        level: usize,
        prev_level: usize,
        m_prev: usize,
    ) -> Option<()> {
        let num_states = if prev_level == 0 {
            1
        } else {
            self.level_states[prev_level - 1].len()
        };
        let table_len = num_states * m_prev;
        if table_len > MAX_MDD_TOTAL_ENTRIES {
            return None;
        }

        self.level_tables[prev_level].clear();
        self.level_tables[prev_level].reserve(table_len);
        self.level_states[prev_level].clear();

        if m_prev == 1 {
            if prev_level == 0 {
                self.level_states[0].push(root);
                self.level_tables[0].push(0);
            } else {
                for s_idx in 0..num_states {
                    let dag = self.level_states[prev_level - 1][s_idx];
                    self.level_states[prev_level].push(dag);
                    self.level_tables[prev_level].push(s_idx as u16);
                }
            }
            return Some(());
        }

        self.bump_gen();
        let state_gen = self.generation;

        for s_idx in 0..num_states {
            let s = if prev_level == 0 {
                root
            } else {
                self.level_states[prev_level - 1][s_idx]
            };
            for b in 0..m_prev {
                self.bump_gen();
                if self.memo.len() < self.dag_nodes.len() {
                    self.memo
                        .resize(self.dag_nodes.len() + 64, MemoSlot::default());
                }
                let mut state = self.restrict_rec(s, prev_level, b as u16);
                for intermediate_rank in (prev_level + 1)..level {
                    state = self.restrict_rec(state, intermediate_rank, 0);
                }

                let key = (state as u64) + 1;
                let mut h_idx = Self::hash_u64(key);
                let mut probes = 0;
                let state_idx = loop {
                    let slot = &mut self.state_table[h_idx & DAG_HASH_MASK];
                    if slot.generation != state_gen {
                        let idx = self.level_states[prev_level].len() as u32;
                        if idx >= 8192 {
                            return None;
                        }
                        self.level_states[prev_level].push(state);
                        slot.generation = state_gen;
                        slot.key = key;
                        slot.val = idx;
                        break idx as u16;
                    }
                    if slot.key == key {
                        break slot.val as u16;
                    }
                    probes += 1;
                    if probes >= DAG_HASH_SIZE {
                        return None;
                    }
                    h_idx = (h_idx + 1) & DAG_HASH_MASK;
                };
                self.level_tables[prev_level].push(state_idx);
            }
        }

        Some(())
    }

    fn fill_1d_dag(
        &mut self,
        dag_id: u32,
        lo: usize,
        hi: usize,
        leaves: &mut Vec<FastTreeLeaf>,
        out: &mut [u16],
    ) {
        if (dag_id & 0x8000_0000) != 0 {
            let leaf_idx = (dag_id & 0x7fff_ffff) as u16;
            out[lo..hi].fill(leaf_idx);
            return;
        }
        let node = self.dag_nodes[dag_id as usize];
        let sb = (node.threshold_idx + 1) as usize;
        if sb <= lo {
            self.fill_1d_dag(node.left, lo, hi, leaves, out);
        } else if sb >= hi {
            self.fill_1d_dag(node.right, lo, hi, leaves, out);
        } else {
            self.fill_1d_dag(node.right, lo, sb, leaves, out);
            self.fill_1d_dag(node.left, sb, hi, leaves, out);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_2d_dag(
        &mut self,
        dag_id: u32,
        b0: (usize, usize),
        b1: (usize, usize),
        rank0: usize,
        stride0: usize,
        leaves: &mut Vec<FastTreeLeaf>,
        out: &mut [u16],
    ) {
        if (dag_id & 0x8000_0000) != 0 {
            let leaf_idx = (dag_id & 0x7fff_ffff) as u16;
            let (lo0, hi0) = b0;
            let (lo1, hi1) = b1;
            if lo1 == 0 && hi1 == stride0 {
                out[lo0 * stride0..hi0 * stride0].fill(leaf_idx);
            } else {
                for r0 in lo0..hi0 {
                    let row = r0 * stride0;
                    out[row + lo1..row + hi1].fill(leaf_idx);
                }
            }
            return;
        }
        let node = self.dag_nodes[dag_id as usize];
        let prop_rank = node.rank as usize;
        let sb = (node.threshold_idx + 1) as usize;
        if prop_rank == rank0 {
            let (lo, hi) = b0;
            if sb <= lo {
                self.fill_2d_dag(node.left, b0, b1, rank0, stride0, leaves, out);
            } else if sb >= hi {
                self.fill_2d_dag(node.right, b0, b1, rank0, stride0, leaves, out);
            } else {
                self.fill_2d_dag(node.right, (lo, sb), b1, rank0, stride0, leaves, out);
                self.fill_2d_dag(node.left, (sb, hi), b1, rank0, stride0, leaves, out);
            }
        } else {
            let (lo, hi) = b1;
            if sb <= lo {
                self.fill_2d_dag(node.left, b0, b1, rank0, stride0, leaves, out);
            } else if sb >= hi {
                self.fill_2d_dag(node.right, b0, b1, rank0, stride0, leaves, out);
            } else {
                self.fill_2d_dag(node.right, b0, (lo, sb), rank0, stride0, leaves, out);
                self.fill_2d_dag(node.left, b0, (sb, hi), rank0, stride0, leaves, out);
            }
        }
    }
}

fn fill_quantizer(q: &mut [u16], thresholds: &[i32], stride: u16) {
    let mut t_idx = 0;
    let n = thresholds.len();
    for (val, slot) in (MIN_Q_VAL..=MAX_Q_VAL).zip(q.iter_mut()) {
        while t_idx < n && val > thresholds[t_idx] {
            t_idx += 1;
        }
        *slot = (t_idx as u16) * stride;
    }
}

pub fn try_build_mdd<'a>(
    tree_nodes: &[TreeNode],
    mdd_props: &[u8; 6],
    active_dims: usize,
    has_residual: bool,
    leaves: &mut Vec<FastTreeLeaf>,
    mdd_scratch: &'a mut MddScratch,
) -> Result<Option<(MddPlan, &'a [u16])>> {
    if active_dims == 0 || active_dims > 6 || tree_nodes.len() > 2048 {
        return Ok(None);
    }

    leaves.clear();
    leaves.try_reserve(1 + tree_nodes.len().div_ceil(2))?;

    let mut all_simple_leaves = true;
    for node in tree_nodes {
        if let TreeNode::Leaf { offset, multiplier, id, .. } = *node {
            if offset != 0 || multiplier != 1 || id > 255 {
                all_simple_leaves = false;
                break;
            }
        }
    }

    // Fast path for 1-property LUT:
    if active_dims == 1 && !has_residual {
        let p0 = mdd_props[0];
        let mut plan = MddPlan::default();
        plan.kind = MddStageKind::Lut1D;
        plan.props[0] = p0;
        plan.active_dims = 1;
        plan.min[0] = MIN_Q_VAL;
        plan.max[0] = MAX_Q_VAL;
        plan.bias[0] = -MIN_Q_VAL;
        plan.all_simple_leaves = all_simple_leaves;

        let (_, _, buf) = mdd_scratch.get_or_init();
        let mut leaf_cache = [((0u8, 0i32, 0u32, 0u32), 0u16); 64];
        let mut leaf_cache_len = 0;
        fill_1d_tree_light(
            tree_nodes,
            0,
            0,
            MAX_Q_SPAN,
            leaves,
            &mut leaf_cache,
            &mut leaf_cache_len,
            &mut buf[..MAX_Q_SPAN],
            all_simple_leaves,
        );
        return Ok(Some((plan, &buf[..MAX_Q_SPAN])));
    }

    let (thresholds, scratch, buf) = mdd_scratch.get_or_init();
    for t in thresholds.iter_mut() {
        t.clear();
    }
    scratch.bump_gen();
    scratch.bump_leaf_gen();
    scratch.ensure_leaf_table();
    scratch.ensure_dag_tables();
    scratch.dag_nodes.clear();

    let mut rank_of_prop = [0xffu8; 256];
    for (r, &p) in mdd_props[..active_dims].iter().enumerate() {
        rank_of_prop[p as usize] = r as u8;
    }

    for node in tree_nodes {
        if let TreeNode::Split { property, val, .. } = *node {
            let r = rank_of_prop[property as usize];
            if r != 0xff {
                let r = r as usize;
                if (MIN_Q_VAL..MAX_Q_VAL).contains(&val) {
                    thresholds[r].push(val);
                }
            }
        }
    }

    for t in thresholds[..active_dims].iter_mut() {
        t.sort_unstable();
        t.dedup();
    }

    let root = match scratch.build_dag(tree_nodes, thresholds, &rank_of_prop, leaves, all_simple_leaves) {
        Some(r) => r,
        None => return Ok(None),
    };

    let mut m = [1usize; 6];
    for i in 0..active_dims {
        m[i] = thresholds[i].len() + 1;
    }

    let mut stage_res = None;

    // 1. Single-stage fused table:
    let s0_entries: usize = m[..active_dims].iter().product();
    if s0_entries <= MAX_MDD_TOTAL_ENTRIES {
        let (kind, dims) = match active_dims {
            2 => (MddStageKind::S1_2D, 2),
            3 => (MddStageKind::S1_3D, 3),
            4 => (MddStageKind::S1_4D, 4),
            5 => (MddStageKind::S1_5D, 5),
            6 => (MddStageKind::S1_6D, 6),
            _ => (MddStageKind::S1_2D, 2),
        };
        let mut strides = [0usize; 6];
        strides[active_dims - 1] = 1;
        for i in (0..active_dims - 1).rev() {
            strides[i] = strides[i + 1] * m[i + 1];
        }
        let full_bounds: [(usize, usize); 6] = std::array::from_fn(|r| (0, m[r]));
        fill_box_tree(
            tree_nodes,
            0,
            full_bounds,
            &strides,
            active_dims,
            &rank_of_prop,
            &thresholds,
            scratch,
            leaves,
            &mut buf[TABLES_OFFSET..TABLES_OFFSET + s0_entries],
            all_simple_leaves,
        );
        let quant_strides: [u16; 6] = std::array::from_fn(|r| {
            if r < active_dims {
                strides[r] as u16
            } else {
                0
            }
        });
        stage_res = Some((quant_strides, kind, dims));
    }

    // 2. Multi-stage options:
    if stage_res.is_none() && active_dims >= 4 {
        let r0 = scratch.restrict_level(root, 0, 0, m[0]);
        let r1 = scratch.restrict_level(root, 1, 1, m[1]);
        let r2 = scratch.restrict_level(root, 2, 2, m[2]);
        if r0.is_some() && r1.is_some() && r2.is_some() {
            let b012_len = m[0] * m[1] * m[2];
            let states_3_len = scratch.level_states[2].len();

            let fill_stage0_3d = |scratch: &MddCompiler, tables: &mut [u16], next_stride: usize| {
                let t0 = &scratch.level_tables[0];
                let t1 = &scratch.level_tables[1];
                let t2 = &scratch.level_tables[2];
                for (b0, &s1_u16) in t0.iter().enumerate().take(m[0]) {
                    let s1 = s1_u16 as usize;
                    for b1 in 0..m[1] {
                        let s2 = t1[s1 * m[1] + b1] as usize;
                        let row_s2 = s2 * m[2];
                        let out_base = (b0 * m[1] + b1) * m[2];
                        for b2 in 0..m[2] {
                            let s3 = t2[row_s2 + b2] as usize;
                            tables[out_base + b2] = (b012_len + s3 * next_stride) as u16;
                        }
                    }
                }
            };

            // 2a. Try 2-Stage `3 + 2` Fused MDD (`S2_5D`) when `m[5] == 1`.
            if m[5] == 1 && b012_len <= MAX_STAGE0_FUSED_ENTRIES {
                let b34_len = m[3] * m[4];
                let stage1_2d_len = states_3_len * b34_len;
                if b012_len + stage1_2d_len <= MAX_MDD_TOTAL_ENTRIES {
                    let tables = &mut buf[TABLES_OFFSET..];
                    fill_stage0_3d(scratch, tables, b34_len);
                    for s3_idx in 0..states_3_len {
                        let dag_id = scratch.level_states[2][s3_idx];
                        let start = b012_len + s3_idx * b34_len;
                        scratch.fill_2d_dag(
                            dag_id,
                            (0, m[3]),
                            (0, m[4]),
                            3,
                            m[4],
                            leaves,
                            &mut tables[start..start + b34_len],
                        );
                    }
                    let strides: [u16; 6] =
                        [(m[1] * m[2]) as u16, m[2] as u16, 1, m[4] as u16, 1, 0];
                    stage_res = Some((strides, MddStageKind::S2_5D, 5));
                }
            }

            // 2b. Try 2-Stage `3 + 1` MDD (`S2_4D`) when `m[4] == 1 && m[5] == 1`.
            if stage_res.is_none()
                && m[4] == 1
                && m[5] == 1
                && b012_len <= MAX_STAGE0_FUSED_ENTRIES
            {
                let stage1_1d_len = states_3_len * m[3];
                if b012_len + stage1_1d_len <= MAX_MDD_TOTAL_ENTRIES {
                    let tables = &mut buf[TABLES_OFFSET..];
                    fill_stage0_3d(scratch, tables, m[3]);
                    for s3_idx in 0..states_3_len {
                        let dag_id = scratch.level_states[2][s3_idx];
                        let start = b012_len + s3_idx * m[3];
                        scratch.fill_1d_dag(
                            dag_id,
                            0,
                            m[3],
                            leaves,
                            &mut tables[start..start + m[3]],
                        );
                    }
                    let strides: [u16; 6] =
                        [(m[1] * m[2]) as u16, m[2] as u16, 1, 1, 0, 0];
                    stage_res = Some((strides, MddStageKind::S2_4D, 4));
                }
            }

            // 2c. Try 3-Stage `3 + 1 + 1` (or `3 + 1 + 2`) (`S3`, 2 dependent state loads).
            if stage_res.is_none() && b012_len <= MAX_STAGE0_FUSED_ENTRIES {
                if scratch.restrict_level(root, 3, 3, m[3]).is_some() {
                    let states_4_len = scratch.level_states[3].len();
                    let b45_len = m[4] * m[5];
                    let s1_len = states_3_len * m[3];
                    let s2_len = states_4_len * b45_len;
                    let total_len = b012_len + s1_len + s2_len;
                    let limit = if m[5] == 1 {
                        MAX_MDD_TOTAL_ENTRIES
                    } else {
                        MAX_FUSED_2D_ENTRIES
                    };
                    if total_len <= limit {
                        let tables = &mut buf[TABLES_OFFSET..];
                        fill_stage0_3d(scratch, tables, m[3]);
                        let base2 = b012_len + s1_len;
                        for (i, &s4) in scratch.level_tables[3].iter().enumerate().take(s1_len) {
                            tables[b012_len + i] = (base2 + (s4 as usize) * b45_len) as u16;
                        }
                        for s4_idx in 0..states_4_len {
                            let dag_id = scratch.level_states[3][s4_idx];
                            let start = base2 + s4_idx * b45_len;
                            if m[5] == 1 {
                                scratch.fill_1d_dag(
                                    dag_id,
                                    0,
                                    m[4],
                                    leaves,
                                    &mut tables[start..start + m[4]],
                                );
                            } else {
                                scratch.fill_2d_dag(
                                    dag_id,
                                    (0, m[4]),
                                    (0, m[5]),
                                    4,
                                    m[5],
                                    leaves,
                                    &mut tables[start..start + b45_len],
                                );
                            }
                        }
                        let strides: [u16; 6] =
                            [(m[1] * m[2]) as u16, m[2] as u16, 1, 1, m[5] as u16, 1];
                        let stage = if m[5] == 1 {
                            MddStageKind::S3_5D
                        } else {
                            MddStageKind::S3_6D
                        };
                        stage_res = Some((strides, stage, if m[5] == 1 { 5 } else { 6 }));
                    }
                }
            }
        }
    }

    let Some((quant_strides, stage_kind, dims)) = stage_res else {
        return Ok(None);
    };

    // Fill quantizers:
    for i in 0..dims {
        let q_slice = &mut buf[i * Q_STRIDE..(i + 1) * Q_STRIDE];
        fill_quantizer(q_slice, &thresholds[i], quant_strides[i]);
    }

    let mut plan = MddPlan::default();
    plan.kind = stage_kind;
    plan.props = *mdd_props;
    plan.active_dims = dims;
    for i in 0..dims {
        plan.min[i] = MIN_Q_VAL;
        plan.max[i] = MAX_Q_VAL;
        plan.bias[i] = (i * Q_STRIDE + 1024) as i32;
    }
    plan.all_simple_leaves = all_simple_leaves;

    Ok(Some((plan, &buf[..])))
}

fn write_box(
    tables: &mut [u16],
    bounds: &[(usize, usize); 6],
    strides: &[usize; 6],
    r: usize,
    active_dims: usize,
    base: usize,
    token: u16,
) {
    let (lo, hi) = bounds[r];
    if r + 1 >= active_dims {
        let st = strides[r];
        if st == 1 {
            tables[base + lo..base + hi].fill(token);
        } else {
            for b in lo..hi {
                tables[base + b * st] = token;
            }
        }
    } else {
        let st = strides[r];
        for b in lo..hi {
            write_box(
                tables,
                bounds,
                strides,
                r + 1,
                active_dims,
                base + b * st,
                token,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_box_tree(
    tree_nodes: &[TreeNode],
    node_idx: usize,
    bounds: [(usize, usize); 6],
    strides: &[usize; 6],
    active_dims: usize,
    rank_of_prop: &[u8; 256],
    thresholds: &[Vec<i32>; 6],
    scratch: &mut MddCompiler,
    leaves: &mut Vec<FastTreeLeaf>,
    tables: &mut [u16],
    all_simple_leaves: bool,
) {
    match tree_nodes[node_idx] {
        TreeNode::Leaf { .. } => {
            let token = scratch.intern_leaf(tree_nodes[node_idx], leaves, all_simple_leaves);
            write_box(tables, &bounds, strides, 0, active_dims, 0, token);
        }
        TreeNode::Split {
            property,
            val,
            left,
            right,
        } => {
            let r = rank_of_prop[property as usize] as usize;
            if r >= active_dims {
                return;
            }
            let Ok(t_idx) = thresholds[r].binary_search(&val) else {
                return;
            };
            let split_b = t_idx + 1;
            let (lo, hi) = bounds[r];
            if split_b <= lo {
                fill_box_tree(
                    tree_nodes,
                    left as usize,
                    bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    scratch,
                    leaves,
                    tables,
                    all_simple_leaves,
                );
            } else if split_b >= hi {
                fill_box_tree(
                    tree_nodes,
                    right as usize,
                    bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    scratch,
                    leaves,
                    tables,
                    all_simple_leaves,
                );
            } else {
                let mut left_bounds = bounds;
                left_bounds[r] = (split_b, hi);
                fill_box_tree(
                    tree_nodes,
                    left as usize,
                    left_bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    scratch,
                    leaves,
                    tables,
                    all_simple_leaves,
                );
                let mut right_bounds = bounds;
                right_bounds[r] = (lo, split_b);
                fill_box_tree(
                    tree_nodes,
                    right as usize,
                    right_bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    scratch,
                    leaves,
                    tables,
                    all_simple_leaves,
                );
            }
        }
    }
}

fn fill_1d_tree_light(
    tree_nodes: &[TreeNode],
    node_idx: usize,
    lo: usize,
    hi: usize,
    leaves: &mut Vec<FastTreeLeaf>,
    cache: &mut [((u8, i32, u32, u32), u16); 64],
    cache_len: &mut usize,
    lut: &mut [u16],
    all_simple_leaves: bool,
) {
    if lo >= hi {
        return;
    }
    match tree_nodes[node_idx] {
        TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id,
        } => {
            if all_simple_leaves {
                let token = ((predictor as u16) << 8) | (id as u16);
                lut[lo..hi].fill(token);
                return;
            }
            let tuple = (predictor as u8, offset, multiplier, id);
            let mut found = None;
            for &(t, leaf_idx) in &cache[..*cache_len] {
                if t == tuple {
                    found = Some(leaf_idx);
                    break;
                }
            }
            let leaf_idx = match found {
                Some(idx) => idx,
                None => {
                    let idx = leaves.len() as u16;
                    leaves.push(FastTreeLeaf {
                        context: id,
                        predictor,
                        offset,
                        multiplier,
                    });
                    if *cache_len < 64 {
                        cache[*cache_len] = (tuple, idx);
                        *cache_len += 1;
                    }
                    idx
                }
            };
            lut[lo..hi].fill(leaf_idx);
        }
        TreeNode::Split {
            val, left, right, ..
        } => {
            let split_bucket = (val - MIN_Q_VAL + 1).clamp(0, MAX_Q_SPAN as i32) as usize;
            if split_bucket <= lo {
                fill_1d_tree_light(
                    tree_nodes, left as usize, lo, hi, leaves, cache, cache_len, lut, all_simple_leaves,
                );
            } else if split_bucket >= hi {
                fill_1d_tree_light(
                    tree_nodes, right as usize, lo, hi, leaves, cache, cache_len, lut, all_simple_leaves,
                );
            } else {
                fill_1d_tree_light(
                    tree_nodes, right as usize, lo, split_bucket, leaves, cache, cache_len, lut, all_simple_leaves,
                );
                fill_1d_tree_light(
                    tree_nodes, left as usize, split_bucket, hi, leaves, cache, cache_len, lut, all_simple_leaves,
                );
            }
        }
    }
}

pub const TABLE_MASK: usize = MAX_MDD_TOTAL_ENTRIES - 1;

#[inline(always)]
pub fn q(val: i32, stride: usize) -> usize {
    stride + (((val.clamp(MIN_Q_VAL, MAX_Q_VAL) - MIN_Q_VAL) as usize) & (MAX_Q_SPAN - 1))
}

#[inline(always)]
pub fn eval_mdd_pixel(
    plan: &MddPlan,
    tables: &[u16],
    v: &[i32; 6],
) -> u16 {
    match plan.kind {
        MddStageKind::Lut1D => {
            let q0 = ((v[0].clamp(MIN_Q_VAL, MAX_Q_VAL) - MIN_Q_VAL) as usize) & (MAX_Q_SPAN - 1);
            tables[q0]
        }
        MddStageKind::S1_2D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            tables[TABLES_OFFSET + ((q0 + q1) & TABLE_MASK)]
        }
        MddStageKind::S1_3D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            tables[TABLES_OFFSET + ((q0 + q1 + q2) & TABLE_MASK)]
        }
        MddStageKind::S1_4D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            tables[TABLES_OFFSET + ((q0 + q1 + q2 + q3) & TABLE_MASK)]
        }
        MddStageKind::S1_5D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            tables[TABLES_OFFSET + ((q0 + q1 + q2 + q3 + q4) & TABLE_MASK)]
        }
        MddStageKind::S1_6D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let q5 = if plan.active_dims > 5 {
                tables[q(v[5], 5 * Q_STRIDE)] as usize
            } else {
                0
            };
            tables[TABLES_OFFSET + ((q0 + q1 + q2 + q3 + q4 + q5) & TABLE_MASK)]
        }
        MddStageKind::S2_4D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + ((q0 + q1 + q2) & TABLE_MASK)] as usize;
            tables[TABLES_OFFSET + ((s1 + q3) & TABLE_MASK)]
        }
        MddStageKind::S2_5D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + ((q0 + q1 + q2) & TABLE_MASK)] as usize;
            tables[TABLES_OFFSET + ((s1 + q3 + q4) & TABLE_MASK)]
        }
        MddStageKind::S3_4D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + ((q0 + q1) & TABLE_MASK)] as usize;
            let s2 = tables[TABLES_OFFSET + ((s1 + q2) & TABLE_MASK)] as usize;
            tables[TABLES_OFFSET + ((s2 + q3) & TABLE_MASK)]
        }
        MddStageKind::S3_5D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + ((q0 + q1 + q2) & TABLE_MASK)] as usize;
            let s2 = tables[TABLES_OFFSET + ((s1 + q3) & TABLE_MASK)] as usize;
            tables[TABLES_OFFSET + ((s2 + q4) & TABLE_MASK)]
        }
        MddStageKind::S3_6D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let q5 = tables[q(v[5], 5 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + ((q0 + q1 + q2) & TABLE_MASK)] as usize;
            let s2 = tables[TABLES_OFFSET + ((s1 + q3) & TABLE_MASK)] as usize;
            tables[TABLES_OFFSET + ((s2 + q4 + q5) & TABLE_MASK)]
        }
        MddStageKind::S4_5D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + q0 + q1] as usize;
            let s2 = tables[TABLES_OFFSET + s1 + q2] as usize;
            let s3 = tables[TABLES_OFFSET + s2 + q3] as usize;
            tables[TABLES_OFFSET + s3 + q4]
        }
        MddStageKind::S5_6D => {
            let q0 = tables[q(v[0], 0)] as usize;
            let q1 = tables[q(v[1], Q_STRIDE)] as usize;
            let q2 = tables[q(v[2], 2 * Q_STRIDE)] as usize;
            let q3 = tables[q(v[3], 3 * Q_STRIDE)] as usize;
            let q4 = tables[q(v[4], 4 * Q_STRIDE)] as usize;
            let q5 = tables[q(v[5], 5 * Q_STRIDE)] as usize;
            let s1 = tables[TABLES_OFFSET + q0 + q1] as usize;
            let s2 = tables[TABLES_OFFSET + s1 + q2] as usize;
            let s3 = tables[TABLES_OFFSET + s2 + q3] as usize;
            let s4 = tables[TABLES_OFFSET + s3 + q4] as usize;
            tables[TABLES_OFFSET + s4 + q5]
        }
    }
}

pub fn eval_mdd_fast<B: TreeBackend>(
    plan: &MddPlan,
    tables: &[u16],
    props: &[i32],
) -> u16 {
    let mut v = [0i32; 6];
    for i in 0..plan.active_dims {
        v[i] = props[plan.props[i] as usize];
    }
    eval_mdd_pixel(plan, tables, &v)
}
