// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::nodes::{FastTreeLeaf, FastTreeNode};
use crate::error::Result;
use crate::frame::modular::tree::TreeNode;

pub trait TreeSource {
    fn as_split(&self, idx: usize) -> Option<(u8, i32, usize, usize)>;
    fn make_leaf(&self, idx: usize) -> FastTreeLeaf;
}

pub struct PrunedTreeSource<'a> {
    pub nodes: &'a [TreeNode],
}

impl TreeSource for PrunedTreeSource<'_> {
    #[inline(always)]
    fn as_split(&self, idx: usize) -> Option<(u8, i32, usize, usize)> {
        match self.nodes[idx] {
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } => Some((property, val, left as usize, right as usize)),
            TreeNode::Leaf { .. } => None,
        }
    }

    #[inline(always)]
    fn make_leaf(&self, idx: usize) -> FastTreeLeaf {
        let TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id,
        } = self.nodes[idx]
        else {
            unreachable!()
        };
        FastTreeLeaf {
            context: id,
            predictor,
            offset,
            multiplier,
        }
    }
}

pub fn extract_range_chain<S: TreeSource>(
    root_idx: usize,
    source: &S,
) -> (u8, usize, [i32; 8], [usize; 9]) {
    let Some((root_prop, root_val, root_left, root_right)) = source.as_split(root_idx) else {
        unreachable!()
    };
    let mut thresholds = [i32::MAX; 8];
    let mut targets = [0usize; 9];
    thresholds[0] = root_val;
    targets[0] = root_right;
    targets[1] = root_left;
    let mut k = 1usize;

    loop {
        let mut progressed = false;
        let mut c = 0usize;
        while c <= k {
            if let Some((prop, val, left, right)) = source.as_split(targets[c])
                && prop == root_prop
            {
                let min_val = if c == 0 {
                    i32::MIN
                } else {
                    thresholds[c - 1].saturating_add(1)
                };
                let max_val = if c == k { i32::MAX } else { thresholds[c] };
                if val < min_val {
                    targets[c] = left;
                    progressed = true;
                    continue;
                } else if val >= max_val {
                    targets[c] = right;
                    progressed = true;
                    continue;
                } else if k < 8 {
                    for j in (c..k).rev() {
                        thresholds[j + 1] = thresholds[j];
                    }
                    for j in (c + 1..=k).rev() {
                        targets[j + 1] = targets[j];
                    }
                    thresholds[c] = val;
                    targets[c] = right;
                    targets[c + 1] = left;
                    k += 1;
                    progressed = true;
                    continue;
                }
            }
            c += 1;
        }
        if !progressed {
            break;
        }
    }

    (root_prop, k, thresholds, targets)
}

#[inline(always)]
pub fn is_leaf_or_split_of_leaves<S: TreeSource>(idx: usize, source: &S) -> bool {
    match source.as_split(idx) {
        None => true,
        Some((_, _, left, right)) => {
            source.as_split(left).is_none() && source.as_split(right).is_none()
        }
    }
}

#[inline(always)]
pub fn extract_d2<S: TreeSource>(
    root_idx: usize,
    source: &S,
) -> ([u8; 3], [i32; 3], [usize; 4]) {
    let Some((root_prop, root_val, root_left, root_right)) = source.as_split(root_idx) else {
        unreachable!()
    };
    let (r_prop, r_val, rr, rl) = match source.as_split(root_right) {
        Some((p, v, l, r)) => (p, v, r, l),
        None => (root_prop, i32::MAX, root_right, root_right),
    };
    let (l_prop, l_val, lr, ll) = match source.as_split(root_left) {
        Some((p, v, l, r)) => (p, v, r, l),
        None => (root_prop, i32::MAX, root_left, root_left),
    };

    let props = [root_prop, r_prop, l_prop];
    let splits = [root_val, r_val, l_val];
    let children = [rr, rl, lr, ll];
    (props, splits, children)
}

#[inline(always)]
pub fn extract_d3<S: TreeSource>(
    root_idx: usize,
    source: &S,
) -> ([u8; 7], [i32; 7], [usize; 8]) {
    let mut slot_nodes = [0usize; 7];
    slot_nodes[0] = root_idx;
    let mut props = [0u8; 7];
    let mut splits = [i32::MAX; 7];
    let mut children = [0usize; 8];

    for slot in 0..7 {
        let idx = slot_nodes[slot];
        let (r_child, l_child) = match source.as_split(idx) {
            Some((property, val, left, right)) => {
                props[slot] = property;
                splits[slot] = val;
                (right, left)
            }
            None => {
                props[slot] = 0;
                splits[slot] = i32::MAX;
                (idx, idx)
            }
        };
        if slot < 3 {
            slot_nodes[2 * slot + 1] = r_child;
            slot_nodes[2 * slot + 2] = l_child;
        } else {
            let c_base = (slot - 3) * 2;
            children[c_base] = r_child;
            children[c_base + 1] = l_child;
        }
    }

    (props, splits, children)
}

pub fn compile_opcode_subtree<S: TreeSource>(
    root_orig_idx: usize,
    source: &S,
    nodes: &mut Vec<FastTreeNode>,
) -> Result<usize> {
    let start_idx = nodes.len();
    if source.as_split(root_orig_idx).is_none() {
        let leaf = source.make_leaf(root_orig_idx);
        nodes.try_reserve(1)?;
        nodes.push(FastTreeNode::Leaf(leaf));
        return Ok(start_idx);
    }

    // Dummy placeholder for root node
    nodes.try_reserve(1)?;
    nodes.push(FastTreeNode::Range {
        prop: 0,
        count: 0,
        thresholds: [i32::MAX; 8],
        child_base: root_orig_idx as u32,
    });

    let mut level_start = start_idx;
    while level_start < nodes.len() {
        let level_end = nodes.len();
        for i in level_start..level_end {
            // Check if node is an uncompiled placeholder
            if let FastTreeNode::Range { count: 0, child_base, .. } = nodes[i] {
                let orig_idx = child_base as usize;
                let Some((_, _, root_left, root_right)) = source.as_split(orig_idx) else {
                    unreachable!()
                };
                let (root_prop, k, range_splits, range_targets) =
                    extract_range_chain(orig_idx, source);

                let all_range_leaves = range_targets[..=k]
                    .iter()
                    .all(|&t| source.as_split(t).is_none());
                let both_children_same_prop = match (
                    source.as_split(root_left),
                    source.as_split(root_right),
                ) {
                    (Some((pl, _, _, _)), Some((pr, _, _, _))) => {
                        pl == root_prop && pr == root_prop
                    }
                    _ => false,
                };
                let use_range = all_range_leaves || k >= 7 || (k >= 4 && both_children_same_prop);

                let mut children_buf = [0usize; 9];
                let (compiled_node, num_children) = if use_range {
                    children_buf[..=k].copy_from_slice(&range_targets[..=k]);
                    (
                        FastTreeNode::Range {
                            prop: root_prop,
                            count: k as u8,
                            thresholds: range_splits,
                            child_base: 0, // set below
                        },
                        k + 1,
                    )
                } else if is_leaf_or_split_of_leaves(root_left, source)
                    && is_leaf_or_split_of_leaves(root_right, source)
                {
                    let (props, splits, ch4) = extract_d2(orig_idx, source);
                    children_buf[..4].copy_from_slice(&ch4);
                    (
                        FastTreeNode::D2 {
                            props,
                            splits,
                            child_base: 0,
                        },
                        4,
                    )
                } else {
                    let (props, splits, ch8) = extract_d3(orig_idx, source);
                    children_buf[..8].copy_from_slice(&ch8);
                    (
                        FastTreeNode::D3 {
                            props,
                            splits,
                            child_base: 0,
                        },
                        8,
                    )
                };

                let child_base_idx = nodes.len() as u32;
                match compiled_node {
                    FastTreeNode::Range { prop, count, thresholds, .. } => {
                        nodes[i] = FastTreeNode::Range {
                            prop,
                            count,
                            thresholds,
                            child_base: child_base_idx,
                        };
                    }
                    FastTreeNode::D2 { props, splits, .. } => {
                        nodes[i] = FastTreeNode::D2 {
                            props,
                            splits,
                            child_base: child_base_idx,
                        };
                    }
                    FastTreeNode::D3 { props, splits, .. } => {
                        nodes[i] = FastTreeNode::D3 {
                            props,
                            splits,
                            child_base: child_base_idx,
                        };
                    }
                    _ => unreachable!(),
                }

                nodes.try_reserve(num_children)?;
                for &c_orig_idx in &children_buf[..num_children] {
                    if source.as_split(c_orig_idx).is_some() {
                        nodes.push(FastTreeNode::Range {
                            prop: 0,
                            count: 0,
                            thresholds: [i32::MAX; 8],
                            child_base: c_orig_idx as u32,
                        });
                    } else {
                        let leaf = source.make_leaf(c_orig_idx);
                        nodes.push(FastTreeNode::Leaf(leaf));
                    }
                }
            }
        }
        level_start = level_end;
    }

    Ok(start_idx)
}
