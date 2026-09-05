// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::ffi::c_void;

use super::compiler::JitModule;
use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Codes, Histograms, SymbolReader};
use crate::entropy_coding::huffman::TableEntry;
use crate::error::Result;
use crate::frame::modular::ModularChannel;

#[repr(C)]
pub struct CBitReader {
    pub data: *const u8,
    pub data_len: usize,
    pub bit_buf: u64,
    pub bits_in_buf: usize,
    pub total_bits_read: usize,
}

#[repr(C)]
pub struct CReferenceChannel {
    pub data: *const i32,
    pub stride: usize,
}

#[repr(C)]
pub struct CAnsBucket {
    pub alias_symbol: u8,
    pub alias_cutoff: u8,
    pub dist: u16,
    pub alias_offset: u16,
    pub alias_dist_xor: u16,
}

#[repr(C)]
pub struct CAnsHistogram {
    pub buckets: *const CAnsBucket,
    pub log_bucket_size: u32,
    pub bucket_mask: u32,
}

pub fn decode_channel_with_jit(
    module: &JitModule,
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    histograms: &Histograms,
    reader: &mut SymbolReader,
    br: &mut BitReader,
    num_ref_channels: usize,
) -> Result<()> {
    let size = buffers[chan].data.size();
    let xsize = size.0;
    let ysize = size.1;
    if xsize == 0 || ysize == 0 {
        return Ok(());
    }

    let channel_ptr = buffers[chan].data.row_mut(0).as_mut_ptr();
    let stride = if ysize > 1 {
        let p0 = buffers[chan].data.row(0).as_ptr();
        let p1 = buffers[chan].data.row(1).as_ptr();
        (p1 as usize - p0 as usize) / std::mem::size_of::<i32>()
    } else {
        xsize
    };

    // Prepare reference channels
    let mut ref_channels = Vec::new();
    for i in 0..chan {
        if ref_channels.len() >= num_ref_channels {
            break;
        }
        let j = chan - 1 - i;
        if buffers[j].data.size() == buffers[chan].data.size()
            && buffers[j].shift == buffers[chan].shift
        {
            let ref_stride = if buffers[j].data.size().1 > 1 {
                let p0 = buffers[j].data.row(0).as_ptr();
                let p1 = buffers[j].data.row(1).as_ptr();
                (p1 as usize - p0 as usize) / std::mem::size_of::<i32>()
            } else {
                buffers[j].data.size().0
            };
            ref_channels.push(CReferenceChannel {
                data: buffers[j].data.row(0).as_ptr(),
                stride: ref_stride,
            });
        }
    }

    let mut c_br = CBitReader {
        data: br.data.as_ptr(),
        data_len: br.data.len(),
        bit_buf: br.bit_buf,
        bits_in_buf: br.bits_in_buf,
        total_bits_read: br.total_bits_read,
    };

    let mut ans_state = 0u32;
    let (_table_ptrs, _ans_histograms, entropy_tables_ptr, ans_state_ptr) = match histograms.codes()
    {
        Codes::Huffman(hc) => {
            let table_ptrs: Vec<*const TableEntry> =
                hc.tables().iter().map(|t| t.entries().as_ptr()).collect();
            let ptr = table_ptrs.as_ptr() as *const c_void;
            (Some(table_ptrs), None, ptr, std::ptr::null_mut())
        }
        Codes::Ans(ans) => {
            let ans_histograms: Vec<CAnsHistogram> = ans
                .histograms()
                .iter()
                .map(|h| CAnsHistogram {
                    buckets: h.buckets().as_ptr() as *const CAnsBucket,
                    log_bucket_size: h.log_bucket_size() as u32,
                    bucket_mask: h.bucket_mask(),
                })
                .collect();
            ans_state = reader.ans_reader().state();
            let ptr = ans_histograms.as_ptr() as *const c_void;
            (None, Some(ans_histograms), ptr, &mut ans_state as *mut u32)
        }
    };

    let ref_ptr = if ref_channels.is_empty() {
        std::ptr::null()
    } else {
        ref_channels.as_ptr() as *const c_void
    };

    // SAFETY: channel_ptr and ref_ptr point to allocated modular channel buffers,
    // c_br points to a valid CBitReader, and entropy_tables_ptr points to valid table data.
    unsafe {
        (module.decode_channel)(
            channel_ptr,
            stride,
            ysize,
            &mut c_br as *mut CBitReader as *mut c_void,
            entropy_tables_ptr,
            ans_state_ptr,
            ref_ptr,
        );
    }

    // Synchronize BitReader
    let consumed_bytes = br.data.len() - c_br.data_len;
    br.data = &br.data[consumed_bytes..];
    br.bit_buf = c_br.bit_buf;
    br.bits_in_buf = c_br.bits_in_buf;
    br.total_bits_read = c_br.total_bits_read;

    // Synchronize ANS state
    if matches!(histograms.codes(), Codes::Ans(_)) {
        reader.ans_reader_mut().set_state(ans_state);
    }

    br.check_for_error()?;

    Ok(())
}
