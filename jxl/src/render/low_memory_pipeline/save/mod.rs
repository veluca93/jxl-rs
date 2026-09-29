// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::row_buffers::RowBuffer;
use crate::api::{Endianness, JxlDataFormat};
use crate::error::Result;
use crate::headers::Orientation;
use crate::image::ImageDataType;
use crate::render::buffer_splitter::OutputChannelRef;
use crate::render::save::{ChannelConversion, SaveStage};
use crate::util::{SmallVec, StackOnly};

mod identity;

impl SaveStage {
    // Takes as input only those channels that are *actually* saved.
    #[allow(clippy::too_many_arguments)]
    pub fn save_lowmem(
        &self,
        data: &[&RowBuffer],
        buffers: &mut [Option<OutputChannelRef>],
        group_size: (usize, usize),
        frame_y: usize,
        group_origin: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
        save_scratch: &mut Vec<u8>,
    ) -> Result<()> {
        let Some(buf) = buffers[self.output_buffer_index].as_mut() else {
            return Ok(());
        };

        let group_y = frame_y - group_origin.1;

        let relative_full_image_start = (
            -frame_origin.0 - (group_origin.0 as isize),
            -frame_origin.1 - (group_origin.1 as isize),
        );

        let relative_full_image_end = (
            relative_full_image_start.0 + full_image_size.0 as isize,
            relative_full_image_start.1 + full_image_size.1 as isize,
        );

        let save_start = (
            relative_full_image_start.0.max(0) as usize,
            relative_full_image_start.1.max(0) as usize,
        );

        let save_end = (
            relative_full_image_end.0.clamp(0, group_size.0 as isize) as usize,
            relative_full_image_end.1.clamp(0, group_size.1 as isize) as usize,
        );

        // If the visible area were empty, we'd have gotten None for the buffer.
        assert!(save_start.0 < save_end.0);
        assert!(save_start.1 < save_end.1);

        if !(save_start.1..save_end.1).contains(&group_y) {
            // The current row is outside the visible area - skip rendering it.
            return Ok(());
        }

        let relative_y = group_y - save_start.1;
        let save_size = (save_end.0 - save_start.0, save_end.1 - save_start.1);
        let xlen = save_size.0;
        let out_channels = self.output_channels();
        let bps = self.data_format.bytes_per_sample();
        let pixel_bytes = out_channels * bps;
        let row_bytes = xlen * pixel_bytes;

        let is_native_endian = match self.data_format {
            JxlDataFormat::U8 { .. } => true,
            JxlDataFormat::U16 { endianness, .. }
            | JxlDataFormat::F16 { endianness }
            | JxlDataFormat::F32 { endianness } => endianness == Endianness::native(),
        };

        let direct_out_y = if is_native_endian {
            match self.orientation {
                Orientation::Identity => Some(relative_y),
                Orientation::FlipVertical => Some(save_size.1 - 1 - relative_y),
                _ => None,
            }
        } else {
            None
        };

        let position = (group_origin.0 + save_start.0, frame_y);

        if let Some(out_y) = direct_out_y
            && buf.row_mut(out_y).as_ptr().align_offset(bps) == 0
        {
            self.convert_and_interleave_row(
                data,
                save_start.0,
                xlen,
                frame_y,
                position,
                &mut buf.row_mut(out_y)[..row_bytes],
            );
            return Ok(());
        }

        let needed = row_bytes + 4;
        if save_scratch.len() < needed {
            save_scratch.resize(needed, 0);
        }
        let align = save_scratch.as_ptr().align_offset(4);
        let target_bytes = &mut save_scratch[align..align + row_bytes];

        self.convert_and_interleave_row(data, save_start.0, xlen, frame_y, position, target_bytes);

        if !is_native_endian {
            match bps {
                2 => {
                    for chunk in target_bytes.chunks_exact_mut(2) {
                        chunk.swap(0, 1);
                    }
                }
                4 => {
                    for chunk in target_bytes.chunks_exact_mut(4) {
                        chunk.reverse();
                    }
                }
                _ => {}
            }
        }
        let (x0, y0) = self.orientation.display_pixel((0, relative_y), save_size);
        let (dx, dy) = self.orientation.display_row_step();
        for (ix, px_bytes) in target_bytes.chunks_exact(pixel_bytes).enumerate() {
            let y = (y0 as isize + dy * ix as isize) as usize;
            let x = (x0 as isize + dx * ix as isize) as usize;
            buf.row_mut(y)[x * pixel_bytes..][..pixel_bytes].copy_from_slice(px_bytes);
        }

        Ok(())
    }

    fn convert_and_interleave_row(
        &self,
        data: &[&RowBuffer],
        x_start: usize,
        xlen: usize,
        frame_y: usize,
        position: (usize, usize),
        target_bytes: &mut [u8],
    ) {
        match self.data_format {
            JxlDataFormat::U8 { .. } => {
                let mut sources: SmallVec<identity::ChannelSourceU8, 4, StackOnly> =
                    SmallVec::new();
                for (c, d) in data.iter().enumerate() {
                    let src = match self.conversions[c] {
                        ChannelConversion::F32ToU8 {
                            bit_depth,
                            dither_channel,
                        } => {
                            let off = RowBuffer::x0_offset::<f32>() + x_start;
                            identity::ChannelSourceU8::F32 {
                                slice: &d.get_row::<f32>(frame_y)[off..off + xlen],
                                max: ((1u32 << bit_depth) - 1) as f32,
                                dither_channel,
                            }
                        }
                        ChannelConversion::I16ToU8 { multiplier, max } => {
                            let off = RowBuffer::x0_offset::<i16>() + x_start;
                            identity::ChannelSourceU8::I16 {
                                slice: &d.get_row::<i16>(frame_y)[off..off + xlen],
                                mult: multiplier as i16,
                                max: max as i16,
                            }
                        }
                        ChannelConversion::I32ToU8 { multiplier, max } => {
                            let off = RowBuffer::x0_offset::<i32>() + x_start;
                            identity::ChannelSourceU8::I32 {
                                slice: &d.get_row::<i32>(frame_y)[off..off + xlen],
                                mult: multiplier,
                                max,
                            }
                        }
                        _ => unreachable!("unsupported conversion to U8"),
                    };
                    sources.push(src);
                }
                identity::store_fused_u8(&sources, self.fill_opaque_alpha, position, target_bytes);
            }
            JxlDataFormat::U16 { bit_depth, .. } => {
                let off = RowBuffer::x0_offset::<f32>() + x_start;
                let slices: SmallVec<&[f32], 4, StackOnly> = data
                    .iter()
                    .map(|d| &d.get_row::<f32>(frame_y)[off..off + xlen])
                    .collect();
                let max = ((1u32 << bit_depth) - 1) as f32;
                let target_u16 = u16::cast_slice_mut(target_bytes);
                identity::store_fused_u16(&slices, max, self.fill_opaque_alpha, target_u16);
            }
            JxlDataFormat::F16 { .. } => {
                let off = RowBuffer::x0_offset::<f32>() + x_start;
                let slices: SmallVec<&[f32], 4, StackOnly> = data
                    .iter()
                    .map(|d| &d.get_row::<f32>(frame_y)[off..off + xlen])
                    .collect();
                let target_u16 = u16::cast_slice_mut(target_bytes);
                identity::store_fused_f16(&slices, self.fill_opaque_alpha, target_u16);
            }
            JxlDataFormat::F32 { .. } => {
                let off = RowBuffer::x0_offset::<f32>() + x_start;
                let slices: SmallVec<&[f32], 4, StackOnly> = data
                    .iter()
                    .map(|d| &d.get_row::<f32>(frame_y)[off..off + xlen])
                    .collect();
                let target_f32 = f32::cast_slice_mut(target_bytes);
                identity::store_fused_f32(&slices, self.fill_opaque_alpha, target_f32);
            }
        }
    }
}
