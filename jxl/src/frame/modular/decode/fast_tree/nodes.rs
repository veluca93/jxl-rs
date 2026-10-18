// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::backend::TreeBackend;
use crate::frame::modular::Predictor;

#[allow(unused)]
pub const PATH_LUT: [u8; 128] = {
    let mut lut = [0u8; 128];
    let mut mask = 0usize;
    while mask < 128 {
        let b0 = mask & 1;
        let child_idx = if b0 == 0 {
            let b1 = (mask >> 1) & 1;
            if b1 == 0 {
                (mask >> 3) & 1
            } else {
                2 + ((mask >> 4) & 1)
            }
        } else {
            let b2 = (mask >> 2) & 1;
            if b2 == 0 {
                4 + ((mask >> 5) & 1)
            } else {
                6 + ((mask >> 6) & 1)
            }
        };
        lut[mask] = child_idx as u8;
        mask += 1;
    }
    lut
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastTreeLeaf {
    pub context: u32,
    pub predictor: Predictor,
    pub offset: i32,
    pub multiplier: u32,
}

impl Default for FastTreeLeaf {
    fn default() -> Self {
        Self {
            context: 0,
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum FastTreeNode {
    /// Collapsed run of $K \in 1..=8$ splits on the same property.
    /// Sibling children are stored contiguously at `child_base + branch`.
    Range {
        prop: u8,
        count: u8,
        thresholds: [i32; 8],
        child_base: u32,
    },
    /// Depth-2 supernode (3 splits $\to$ 4 children).
    D2 {
        props: [u8; 3],
        splits: [i32; 3],
        child_base: u32,
    },
    /// Depth-3 supernode (7 splits $\to$ 8 children).
    D3 {
        props: [u8; 7],
        splits: [i32; 7],
        child_base: u32,
    },
    /// Multi-Valued Decision Diagram (MDD) stage.
    Mdd {
        stage_idx: u32,
    },
    /// Resolved terminal leaf.
    Leaf(FastTreeLeaf),
}

impl Default for FastTreeNode {
    fn default() -> Self {
        FastTreeNode::Leaf(FastTreeLeaf::default())
    }
}

pub enum NodeEvalResult {
    Next(usize),
    Mdd(u32),
    Leaf(FastTreeLeaf),
}

impl FastTreeNode {
    #[inline(always)]
    pub fn eval<B: TreeBackend>(&self, props: &[i32]) -> NodeEvalResult {
        match *self {
            FastTreeNode::Range {
                prop,
                count,
                thresholds,
                child_base,
            } => {
                let val = props[prop as usize];
                let branch = B::eval_range(val, &thresholds, count as usize);
                NodeEvalResult::Next((child_base + branch as u32) as usize)
            }
            FastTreeNode::D2 {
                props: p,
                splits,
                child_base,
            } => {
                let vals = [
                    props[p[0] as usize],
                    props[p[1] as usize],
                    props[p[2] as usize],
                ];
                let branch = B::eval_d2(vals, splits);
                NodeEvalResult::Next((child_base + branch as u32) as usize)
            }
            FastTreeNode::D3 {
                props: p,
                splits,
                child_base,
            } => {
                let vals = [
                    props[p[0] as usize],
                    props[p[1] as usize],
                    props[p[2] as usize],
                    props[p[3] as usize],
                    props[p[4] as usize],
                    props[p[5] as usize],
                    props[p[6] as usize],
                ];
                let branch = B::eval_d3(vals, splits);
                NodeEvalResult::Next((child_base + branch as u32) as usize)
            }
            FastTreeNode::Mdd { stage_idx } => NodeEvalResult::Mdd(stage_idx),
            FastTreeNode::Leaf(leaf) => NodeEvalResult::Leaf(leaf),
        }
    }
}
