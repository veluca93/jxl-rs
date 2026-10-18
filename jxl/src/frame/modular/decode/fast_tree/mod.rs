// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod backend;
pub mod compile;
pub mod decoder;
pub mod mdd;
pub mod nodes;

pub use backend::ScalarBackend;
#[cfg(target_arch = "x86_64")]
pub use backend::Avx2Backend;
#[cfg(target_arch = "aarch64")]
pub use backend::NeonBackend;
pub use decoder::FastTree;
pub use mdd::MddScratch;
pub use nodes::{FastTreeLeaf, FastTreeNode};

use std::marker::PhantomData;

use crate::entropy_coding::decode::Histograms;
use crate::error::Result;
use crate::frame::modular::decode::channel::ModularChannelDecoder;
use crate::frame::modular::decode::common::References;
use crate::frame::modular::predict::{WeightedPredictorScratch, WeightedPredictorState};
use crate::frame::modular::tree::{NUM_NONREF_PROPERTIES, TreeNode};
use crate::frame::modular::{ModularStorage, Predictor};
use crate::headers::modular::GroupHeader;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_fast_tree<F: FnOnce(&mut dyn ModularChannelDecoder) -> Result<()>>(
    pruned_tree: &[TreeNode],
    histograms: &Histograms,
    header: &GroupHeader,
    _channel: usize,
    xsize: usize,
    ysize: usize,
    storage: ModularStorage,
    uses_wp: bool,
    max_property_count: usize,
    nodes: &mut Vec<FastTreeNode>,
    mdd_scratch: &mut MddScratch,
    mdd_leaves: &mut Vec<FastTreeLeaf>,
    wp_scratch: &mut WeightedPredictorScratch,
    references_scratch: &mut Vec<i32>,
    run: &mut Option<F>,
) -> Result<bool> {
    if pruned_tree.is_empty() || histograms.lz77_params().enabled {
        return Ok(false);
    }

    let mut needs_toptop = uses_wp;
    let mut needs_16props = false;
    let mut needs_24props = false;
    let mut needs_ext_leaf_preds = false;
    let mut num_splits = 0usize;
    let mut min_splitval = i32::MAX;
    let mut max_splitval = i32::MIN;
    let mut prop_counts = [0u16; 24];
    let mut num_props = 0usize;

    for &node in pruned_tree {
        match node {
            TreeNode::Split { property, val, .. } => {
                if !(2..24).contains(&property) {
                    return Ok(false);
                }
                num_splits += 1;
                needs_toptop |= property == 13;
                if property >= 16 {
                    needs_24props = true;
                } else if !(4..=11).contains(&property) {
                    needs_16props = true;
                }
                min_splitval = min_splitval.min(val);
                max_splitval = max_splitval.max(val);
                let p = property as usize;
                if prop_counts[p] == 0 {
                    num_props += 1;
                }
                prop_counts[p] = prop_counts[p].saturating_add(1);
            }
            TreeNode::Leaf { predictor, .. } => {
                needs_toptop |= predictor == Predictor::AverageAll;
                needs_ext_leaf_preds |= matches!(
                    predictor,
                    Predictor::NorthEast
                        | Predictor::WestWest
                        | Predictor::AverageNorthAndNorthEast
                        | Predictor::AverageAll
                );
            }
        }
    }

    if num_splits == 0 || (needs_24props && num_splits <= 3) {
        return Ok(false);
    }

    let needs_ext_neighbors = uses_wp || needs_16props || needs_ext_leaf_preds;
    let num_pixels = xsize.saturating_mul(ysize);
    let can_try_mdd = min_splitval >= mdd::MIN_Q_VAL
        && max_splitval < mdd::MAX_Q_VAL
        && ((num_props == 1 && num_splits > 8) || (num_props <= 6 && pruned_tree.len() >= 8))
        && pruned_tree.len() <= 2048
        && num_pixels >= 512;

    let mdd_result = if can_try_mdd {
        let mut active_props = [0u8; 6];
        let mut ap_len = 0usize;
        for (p, &cnt) in prop_counts.iter().enumerate() {
            if cnt > 0 {
                active_props[ap_len] = p as u8;
                ap_len += 1;
            }
        }
        mdd::try_build_mdd(
            pruned_tree,
            &active_props,
            ap_len,
            false,
            mdd_leaves,
            mdd_scratch,
        )?
    } else {
        None
    };

    let (mdd_plan, mdd_tables) = match mdd_result {
        Some((plan, buf)) => (Some(plan), Some(buf)),
        None => (None, None),
    };

    if mdd_plan.is_none() {
        nodes.clear();
        let source = compile::PrunedTreeSource { nodes: pruned_tree };
        compile::compile_opcode_subtree(0, &source, nodes)?;
    }
    let references = if needs_24props && max_property_count > NUM_NONREF_PROPERTIES {
        let num_ref_props = (max_property_count - NUM_NONREF_PROPERTIES)
            .next_multiple_of(4)
            .min(8);
        References::new(references_scratch, num_ref_props, xsize)
    } else {
        References {
            data: &mut [],
            num_ref_props: 0,
        }
    };

    let wp_state = if uses_wp {
        Some(WeightedPredictorState::new(
            &header.wp_header,
            xsize,
            wp_scratch,
        ))
    } else {
        None
    };

    let needs_only_4_9 = !needs_ext_neighbors
        && !needs_24props
        && !needs_toptop
        && (max_property_count <= 11);

    // Pick backend based on architecture and runtime CPU features
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let mut decoder = FastTree::<Avx2Backend> {
            nodes,
            mdd_plan,
            mdd_tables,
            mdd_leaves,
            uses_wp,
            wp_state,
            references,
            needs_ext_neighbors,
            needs_toptop,
            needs_only_4_9,
            prev_prop9: 0,
            storage,
            _backend: PhantomData,
        };
        (run.take().unwrap())(&mut decoder)?;
        return Ok(true);
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut decoder = FastTree::<NeonBackend> {
            nodes,
            mdd_plan,
            mdd_tables,
            mdd_leaves,
            uses_wp,
            wp_state,
            references,
            needs_ext_neighbors,
            needs_toptop,
            needs_only_4_9,
            prev_prop9: 0,
            storage,
            _backend: PhantomData,
        };
        (run.take().unwrap())(&mut decoder)?;
        return Ok(true);
    }

    #[allow(unreachable_code)]
    {
        let mut decoder = FastTree::<ScalarBackend> {
            nodes,
            mdd_plan,
            mdd_tables,
            mdd_leaves,
            uses_wp,
            wp_state,
            references,
            needs_ext_neighbors,
            needs_toptop,
            needs_only_4_9,
            prev_prop9: 0,
            storage,
            _backend: PhantomData,
        };
        (run.take().unwrap())(&mut decoder)?;
        Ok(true)
    }
}
