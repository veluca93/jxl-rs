// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod codegen;
pub mod compiler;
pub mod decoder;

use std::collections::VecDeque;

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{SymbolReader, unpack_signed};
use crate::error::Result;
use crate::frame::modular::tree::TreeNode;
use crate::frame::modular::{ModularChannel, Tree};
use crate::headers::modular::GroupHeader;

pub fn decode_modular_channel_jit(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<bool> {
    if tree.histograms.lz77_params().enabled {
        return Ok(false);
    }

    let size = buffers[chan].data.size();
    let xsize = size.0;
    let ysize = size.1;
    if xsize == 0 || ysize == 0 {
        return Ok(true);
    }

    // 2. Prune tree on static properties (0 = channel, 1 = stream_id)
    let mut pruned_tree = Vec::new();
    let mut queue = VecDeque::new();
    pruned_tree.try_reserve(tree.nodes.len())?;
    queue.try_reserve(tree.nodes.len())?;
    queue.push_front(0);

    let mut is_single_symbol = true;
    let mut single_symbol = None;
    let mut max_property_count = 0;

    while let Some(v) = queue.pop_front() {
        let mut node = tree.nodes[v as usize];
        match node {
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } if property < 2 => {
                let vv = if property == 0 { chan } else { stream_id };
                queue.push_front(if vv as i32 > val { left } else { right });
                continue;
            }
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } => {
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
            TreeNode::Leaf { .. } => {
                let TreeNode::Leaf { id, .. } = &mut node else {
                    unreachable!()
                };
                *id = tree.histograms.map_context_to_cluster(*id as usize) as u32;
                if is_single_symbol {
                    if let Some(sym) = tree.histograms.single_symbol(*id as usize) {
                        if sym >= tree.histograms.uint(*id as usize).split_token() {
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

    let single_symbol = if is_single_symbol {
        single_symbol.map(unpack_signed)
    } else {
        None
    };

    // Calculate number of reference channels needed
    let num_refs_needed = if max_property_count > 16 {
        (max_property_count - 16).div_ceil(4)
    } else {
        0
    };

    let mut num_ref_channels = 0;
    for i in 0..chan {
        if num_ref_channels >= num_refs_needed {
            break;
        }
        let j = chan - 1 - i;
        if buffers[j].data.size() == buffers[chan].data.size()
            && buffers[j].shift == buffers[chan].shift
        {
            num_ref_channels += 1;
        }
    }

    let codegen_input = codegen::CodegenInput {
        tree: &pruned_tree,
        xsize,
        header,
        histograms: &tree.histograms,
        single_symbol,
        num_ref_channels,
    };

    let c_code = codegen::generate_c_code(&codegen_input);

    let module = match compiler::get_or_compile_module(&c_code) {
        Ok(m) => m,
        Err(_e) => {
            crate::util::tracing_wrappers::warn!("JIT compilation failed, falling back: {_e}");
            return Ok(false);
        }
    };

    decoder::decode_channel_with_jit(
        &module,
        buffers,
        chan,
        &tree.histograms,
        reader,
        br,
        num_ref_channels,
    )?;

    Ok(true)
}
