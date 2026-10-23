// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use super::LowMemoryRenderPipeline;
use super::row_buffers::RowBuffer;
use crate::error::Result;
use crate::image::{DataTypeTag, OwnedRawImage, Rect};
use crate::render::buffer_splitter::{OutputChannelRef, extend_canvas_rect};
use crate::render::internal::{ChannelInfo, Stage};
use crate::render::low_memory_pipeline::LowMemoryRenderPipelinePerThread;
use crate::render::low_memory_pipeline::helpers::get_distinct_indices;
use crate::render::low_memory_pipeline::run_stage::ExtraInfo;
use crate::util::sync::{RwLock, RwLockReadGuard};
use crate::util::tracing_wrappers::*;
use crate::util::{ChannelVec, ShiftRightCeil, SmallVec, mirror};

fn apply_x_padding(
    input_type: DataTypeTag,
    row: &mut [u8],
    to_pad: Range<isize>,
    valid_pixels: Range<isize>,
) {
    let x0_offset = RowBuffer::x0_byte_offset() as isize;
    let num_valid = valid_pixels.clone().count();
    let sz = input_type.size();
    match sz {
        1 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx) as usize;
                let to = (x0_offset + x) as usize;
                row[to] = row[from];
            }
        }
        2 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 2) as usize;
                let to = (x0_offset + x * 2) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
            }
        }
        4 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 4) as usize;
                let to = (x0_offset + x * 4) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
                row[to + 2] = row[from + 2];
                row[to + 3] = row[from + 3];
            }
        }
        _ => {
            unimplemented!("only 1, 2 or 4 byte data types supported");
        }
    }
}

struct BufferFiller<'a> {
    c: usize,
    ty: DataTypeTag,
    group_y0: usize,
    group_ysize: usize,
    top_y_offset: usize,
    bot_y_offset: usize,
    images: [Option<RwLockReadGuard<'a, Option<OwnedRawImage>>>; 9],
    copy_byte_offset_initial: usize,
    src_byte_offset_left_topbottom: usize,
    src_byte_offset_left_center: usize,
    to_copy_left: usize,
    copy_start: usize,
    copy_end: usize,
    to_copy_main: usize,
    to_copy_right: usize,
    padding_range: Option<(isize, isize)>,
}

impl<'a> BufferFiller<'a> {
    fn new(
        rp: &'a LowMemoryRenderPipeline,
        c: usize,
        (x0, xsize): (usize, usize),
        (gx, gy): (usize, usize),
        vyrange: Range<usize>,
    ) -> Option<Self> {
        if !rp.shared.channel_is_used[c] || vyrange.is_empty() {
            return None;
        }
        let ChannelInfo {
            ty,
            downsample: (dx, dy),
        } = rp.shared.channel_info[0][c];

        let scaled_y_border = rp.input_border_pixels[c].1 << dy;
        let stage_vy_start = vyrange.start as isize - scaled_y_border as isize;
        let stage_vy_end = (vyrange.end as isize - 1) + scaled_y_border as isize;
        let min_y = stage_vy_start >> dy;
        let max_y = stage_vy_end >> dy;
        let max_valid_y = rp.shared.input_size.1.shrc(dy) as isize - 1;

        let yrange = if max_y < 0 || min_y > max_valid_y {
            0..0
        } else {
            let start = min_y.max(0) as usize;
            let end = (max_y.min(max_valid_y) as usize) + 1;
            start..end
        };

        if yrange.is_empty() {
            return None;
        }

        let x0 = x0 >> dx;
        let xsize = xsize.shrc(dx);
        let group_ysize = 1 << (rp.shared.log_group_size - dy as usize);
        let group_xsize = 1 << (rp.shared.log_group_size - dx as usize);

        let (bx, by) = rp.border_size;

        let group_y0 = gy * group_ysize;
        let group_x0 = gx << (rp.shared.log_group_size - dx as usize);
        let group_x1 = group_x0 + group_xsize;

        let top_y_offset = (by >> dy) * 4;
        let bot_y_offset = group_y0 + group_ysize;

        let has_top = yrange.start < group_y0;
        let has_center = yrange.start < group_y0 + group_ysize && yrange.end > group_y0;
        let has_bot = yrange.end > group_y0 + group_ysize;

        let copy_x0 = x0.saturating_sub(rp.input_border_pixels[c].0);
        let copy_x1 =
            (x0 + xsize + rp.input_border_pixels[c].0).min(rp.shared.input_size.0.shrc(dx));

        debug_assert!(copy_x1 >= group_x0);

        let copy_byte_offset_initial = RowBuffer::x0_byte_offset() - (x0 - copy_x0) * ty.size();

        let gw = rp.shared.group_count.0;

        let mut images: [Option<RwLockReadGuard<'_, Option<OwnedRawImage>>>; 9] =
            std::array::from_fn(|_| None);

        let has_left = copy_x0 < group_x0;
        let has_right = copy_x1 > group_x1;

        let to_copy_left = if has_left {
            (group_x0 - copy_x0) * ty.size()
        } else {
            0
        };

        let src_byte_offset_left_topbottom = group_xsize * ty.size() - to_copy_left;
        let src_byte_offset_left_center = 4 * (bx >> dx) * ty.size() - to_copy_left;

        fn make_ref(
            g: &RwLock<Option<OwnedRawImage>>,
        ) -> Option<RwLockReadGuard<'_, Option<OwnedRawImage>>> {
            Some(g.try_read().unwrap())
        }

        if has_top {
            let base_gid = (gy - 1) * gw + gx;
            if has_left {
                images[0] = make_ref(&rp.input_buffers.get(base_gid - 1).topbottom[c]);
            }
            images[1] = make_ref(&rp.input_buffers.get(base_gid).topbottom[c]);
            if has_right {
                images[2] = make_ref(&rp.input_buffers.get(base_gid + 1).topbottom[c]);
            }
        }

        if has_center {
            let base_gid = gy * gw + gx;
            if has_left {
                images[3] = make_ref(&rp.input_buffers.get(base_gid - 1).leftright[c]);
            }
            images[4] = make_ref(&rp.input_buffers.get(base_gid).data[c]);
            if has_right {
                images[5] = make_ref(&rp.input_buffers.get(base_gid + 1).leftright[c]);
            }
        }

        if has_bot {
            let base_gid = (gy + 1) * gw + gx;
            if has_left {
                images[6] = make_ref(&rp.input_buffers.get(base_gid - 1).topbottom[c]);
            }
            images[7] = make_ref(&rp.input_buffers.get(base_gid).topbottom[c]);
            if has_right {
                images[8] = make_ref(&rp.input_buffers.get(base_gid + 1).topbottom[c]);
            }
        }

        let copy_start = copy_x0.saturating_sub(group_x0) * ty.size();
        let copy_end = (copy_x1.min(group_x1) - group_x0) * ty.size();
        let to_copy_main = copy_end - copy_start;

        let (to_copy_right, padding_range) = if has_right {
            let gid = gy * gw + gx;
            let next_group_xsize = rp.shared.group_size(gid + 1).0.shrc(dx);
            let border_x = (copy_x1 - group_x1).min(next_group_xsize);
            let to_copy_right = border_x * ty.size();
            let pad = if border_x + group_x1 < copy_x1 {
                let pad_from = (xsize + border_x) as isize;
                let pad_to = (xsize + copy_x1 - group_x1) as isize;
                Some((pad_from, pad_to))
            } else {
                None
            };
            (to_copy_right, pad)
        } else {
            (0, None)
        };

        Some(Self {
            c,
            ty,
            group_y0,
            group_ysize,
            top_y_offset,
            bot_y_offset,
            images,
            copy_byte_offset_initial,
            src_byte_offset_left_topbottom,
            src_byte_offset_left_center,
            to_copy_left,
            copy_start,
            copy_end,
            to_copy_main,
            to_copy_right,
            padding_range,
        })
    }

    fn fill(&self, data: &mut LowMemoryRenderPipelinePerThread, y: usize) {
        let (row_idx, input_y) = if y < self.group_y0 {
            (0, y + self.top_y_offset - self.group_y0)
        } else if y >= self.group_y0 + self.group_ysize {
            (2, y - self.bot_y_offset)
        } else {
            (1, y - self.group_y0)
        };

        let base = row_idx * 3;
        let output_row = data.row_buffers[0][self.c].get_row_mut::<u8>(y);
        let mut copy_byte_offset = self.copy_byte_offset_initial;

        if let Some(left_buf_guard) = &self.images[base] {
            let left_buf = left_buf_guard.as_ref().unwrap();
            let input_row = left_buf.row(input_y);
            let src_byte_offset = if row_idx != 1 {
                self.src_byte_offset_left_topbottom
            } else {
                self.src_byte_offset_left_center
            };
            output_row[copy_byte_offset..copy_byte_offset + self.to_copy_left]
                .copy_from_slice(&input_row[src_byte_offset..src_byte_offset + self.to_copy_left]);
            copy_byte_offset += self.to_copy_left;
        }

        let center_buf = self.images[base + 1].as_ref().unwrap().as_ref().unwrap();
        let input_row = center_buf.row(input_y);
        output_row[copy_byte_offset..copy_byte_offset + self.to_copy_main]
            .copy_from_slice(&input_row[self.copy_start..self.copy_end]);
        copy_byte_offset += self.to_copy_main;

        if let Some(right_buf_guard) = &self.images[base + 2] {
            let right_buf = right_buf_guard.as_ref().unwrap();
            let input_row = right_buf.row(input_y);
            output_row[copy_byte_offset..copy_byte_offset + self.to_copy_right]
                .copy_from_slice(&input_row[..self.to_copy_right]);
            if let Some((pad_from, pad_to)) = self.padding_range {
                apply_x_padding(self.ty, output_row, pad_from..pad_to, 0..pad_from);
            }
        }
    }
}

impl LowMemoryRenderPipeline {
    #[allow(clippy::too_many_arguments)]
    fn run_stage_on_row(
        &self,
        i: usize,
        stage: &Stage<RowBuffer>,
        data: &mut LowMemoryRenderPipelinePerThread,
        buffers: &mut [Option<OutputChannelRef>],
        previous_call_was_previous_row: &mut [bool],
        cur_rect: Rect,
        save_clip_rect: Rect,
        space_size: (usize, usize),
        post_extend_zero_copy: Option<((usize, usize), (isize, isize))>,
        y: usize,
        start_of_row: bool,
        end_of_row: bool,
    ) -> Result<()> {
        let (in_dx, in_dy) = (
            self.stage_input_downsample[i].0 as usize,
            self.stage_input_downsample[i].1 as usize,
        );
        let (out_dx, out_dy) = (
            self.stage_output_downsample[i].0 as usize,
            self.stage_output_downsample[i].1 as usize,
        );
        let out_extra_x = self.stage_output_border_pixels[i].0;
        let shifted_xsize = cur_rect.size.0.shrc(in_dx);
        let shifted_ysize = space_size.1.shrc(in_dy);

        match stage {
            Stage::InPlace(s) => {
                let mut row_bufs =
                    get_distinct_indices(&mut data.row_buffers, &self.sorted_buffer_indices[i]);
                let prev = std::mem::replace(&mut previous_call_was_previous_row[i], true);
                s.run_stage_on(
                    ExtraInfo {
                        xsize: shifted_xsize,
                        current_row: y,
                        group_x0: cur_rect.origin.0 >> in_dx,
                        out_extra_x,
                        start_of_row,
                        end_of_row,
                        image_height: shifted_ysize,
                        previous_call_was_previous_row: prev,
                    },
                    &mut row_bufs,
                    data.local_states[i].as_deref_mut(),
                );
            }
            Stage::Downsample(d) => {
                let (inb, outb) = data.row_buffers.split_at_mut(i + 1);
                let (si, ci) = self.stage_input_buffer_index[i][0];
                let in_buf = &inb[si][ci];
                let x0_off = RowBuffer::x0_offset::<f32>();
                let in_y0 = y << d.shift();
                let in_y1 = (in_y0 + (1usize << d.shift())).min(shifted_ysize);
                let mut input_rows: [&[f32]; 8] = [&[]; 8];
                for (r, slot) in (in_y0..in_y1).zip(input_rows.iter_mut()) {
                    *slot = &in_buf.get_row::<f32>(r)[x0_off..x0_off + shifted_xsize];
                }
                let out_xsize = shifted_xsize.shrc(d.shift());
                let out_row = &mut outb[0][0].get_row_mut::<f32>(y)[x0_off..x0_off + out_xsize];
                d.process_row_chunk(shifted_xsize, &input_rows[..in_y1 - in_y0], out_row);
            }
            Stage::Save(s) => {
                let input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                    .iter()
                    .map(|(si, ci)| &data.row_buffers[*si][*ci])
                    .collect();
                let (full_size, frame_origin) =
                    if let Some((ext_size, ext_origin)) = post_extend_zero_copy {
                        (
                            (ext_size.0.shrc(out_dx), ext_size.1.shrc(out_dy)),
                            ext_origin,
                        )
                    } else {
                        let saved_rect = save_clip_rect
                            .downsample((out_dx as u8, out_dy as u8))
                            .clip((space_size.0.shrc(out_dx), space_size.1.shrc(out_dy)));
                        if saved_rect.size.0 == 0 || saved_rect.size.1 == 0 {
                            return Ok(());
                        }
                        (
                            saved_rect.size,
                            (
                                -(saved_rect.origin.0 as isize),
                                -(saved_rect.origin.1 as isize),
                            ),
                        )
                    };
                s.save_lowmem(
                    &input_data,
                    data.save_scratch.as_mut().unwrap(),
                    &mut *buffers,
                    (cur_rect.size.0.shrc(out_dx), cur_rect.size.1.shrc(out_dy)),
                    y,
                    (cur_rect.origin.0 >> out_dx, cur_rect.origin.1 >> out_dy),
                    full_size,
                    frame_origin,
                )?;
            }
            Stage::Extend(_) => unreachable!(),
            Stage::InOut(s) => {
                let borderx = s.border().0 as usize;
                let bordery = s.border().1 as isize;
                let prev = std::mem::replace(&mut previous_call_was_previous_row[i], true);
                let iy_range = if prev {
                    bordery..=bordery
                } else {
                    -bordery..=bordery
                };
                if start_of_row && borderx != 0 {
                    for (si, ci) in self.stage_input_buffer_index[i].iter() {
                        for iy in iy_range.clone() {
                            let py = mirror(y as isize + iy, shifted_ysize);
                            apply_x_padding(
                                s.input_type(),
                                data.row_buffers[*si][*ci].get_row_mut::<u8>(py),
                                -(borderx as isize)..0,
                                0..shifted_xsize as isize,
                            );
                        }
                    }
                }
                if end_of_row && borderx != 0 {
                    for (si, ci) in self.stage_input_buffer_index[i].iter() {
                        for iy in iy_range.clone() {
                            let py = mirror(y as isize + iy, shifted_ysize);
                            apply_x_padding(
                                s.input_type(),
                                data.row_buffers[*si][*ci].get_row_mut::<u8>(py),
                                shifted_xsize as isize..(shifted_xsize + borderx) as isize,
                                -(borderx as isize)..shifted_xsize as isize,
                            );
                        }
                    }
                }
                let (inb, outb) = data.row_buffers.split_at_mut(i + 1);
                let input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                    .iter()
                    .map(|(si, ci)| &inb[*si][*ci])
                    .collect();
                s.run_stage_on(
                    ExtraInfo {
                        xsize: shifted_xsize,
                        current_row: y,
                        group_x0: cur_rect.origin.0 >> in_dx,
                        out_extra_x,
                        start_of_row,
                        end_of_row,
                        image_height: shifted_ysize,
                        previous_call_was_previous_row: prev,
                    },
                    &input_data,
                    &mut outb[0][..],
                    data.local_states[i].as_deref_mut(),
                );
            }
        }
        Ok(())
    }

    // Renders *parts* of group's worth of data.
    // In particular, renders the sub-rectangle given in `image_area`, where (1, 1) refers to
    // the center of the group, and 0 and 2 include data from the neighbouring group (if any).
    #[instrument(skip(self, buffers))]
    pub(super) fn render_group(
        &self,
        data: &mut LowMemoryRenderPipelinePerThread,
        (gx, gy): (usize, usize),
        image_area: Rect,
        buffers: &mut [Option<OutputChannelRef>],
    ) -> Result<()> {
        let has_post_extend_ds = self.post_extend_downsample != (0, 0);
        let (pdx, pdy) = self.pre_extend_downsample;
        let pre_area = if has_post_extend_ds && (pdx, pdy) != (0, 0) {
            let px0 = (image_area.origin.0 >> pdx) << pdx;
            let px1 = (image_area.end().0.shrc(pdx) << pdx).min(self.shared.input_size.0);
            let py0 = (image_area.origin.1 >> pdy) << pdy;
            let py1 = (image_area.end().1.shrc(pdy) << pdy).min(self.shared.input_size.1);
            Rect {
                origin: (px0, py0),
                size: (px1 - px0, py1 - py0),
            }
        } else {
            image_area
        };

        let start_of_row = pre_area.origin.0 == 0;
        let end_of_row = pre_area.end().0 == self.shared.input_size.0;

        let Rect {
            origin: (x0, y0),
            size: (xsize, _),
        } = pre_area;

        let num_channels = self.shared.num_channels();
        let num_extra_rows = self.border_size.1;

        let max_dy = (0..num_channels)
            .map(|c| self.shared.channel_info[0][c].downsample.1 as usize)
            .chain(self.downsampling_for_stage.iter().map(|d| d.1))
            .max()
            .unwrap_or(0);
        let pre_fv_start = y0.saturating_sub(num_extra_rows + (1 << max_dy) - 1) as isize
            - num_extra_rows as isize;
        let pre_fv_end = pre_area.end().1 as isize;

        let extend_info = if has_post_extend_ds {
            let e_idx = self.shared.extend_stage_index.unwrap();
            let Stage::Extend(e) = &self.shared.stages[e_idx] else {
                unreachable!();
            };
            Some((
                e_idx,
                e,
                extend_canvas_rect(
                    image_area,
                    self.shared.input_size,
                    e.image_size,
                    e.frame_origin,
                    self.post_extend_downsample,
                ),
            ))
        } else {
            None
        };

        let (fv_start, fv_end) = if let Some((_, e, (canvas_rect, _, _))) = extend_info
            && canvas_rect.size.0 > 0
            && canvas_rect.size.1 > 0
        {
            (
                pre_fv_start.min(canvas_rect.origin.1 as isize - e.frame_origin.1),
                pre_fv_end.max(canvas_rect.end().1 as isize - e.frame_origin.1),
            )
        } else {
            (pre_fv_start, pre_fv_end)
        };

        let fillers: ChannelVec<_> = (0..num_channels)
            .map(|c| BufferFiller::new(self, c, (x0, xsize), (gx, gy), y0..pre_area.end().1))
            .collect();

        let mut previous_call_was_previous_row: SmallVec<bool, 32> =
            std::iter::repeat_n(false, self.shared.stages.len()).collect();

        for fv in fv_start..fv_end {
            // Step 1: read input channels.
            if fv >= pre_fv_start && fv < pre_fv_end {
                for c in 0..num_channels {
                    let Some(filler) = &fillers[c] else {
                        continue;
                    };
                    let (_, dy) = self.shared.channel_info[0][c].downsample;
                    let scaled_y_border = self.input_border_pixels[c].1 << dy;
                    let stage_vy = fv + scaled_y_border as isize;
                    if stage_vy % (1 << dy) != 0 {
                        continue;
                    }
                    let y = stage_vy >> dy;
                    if y < (y0 as isize - scaled_y_border as isize) >> dy {
                        continue;
                    }
                    if y < 0 || y >= self.shared.input_size.1.shrc(dy) as isize {
                        continue;
                    }
                    filler.fill(data, y as usize);
                }
            }

            let mut cur_fv = fv;
            let mut cur_rect = pre_area;
            let mut save_clip_rect = image_area;
            let mut space_size = self.shared.input_size;
            let mut post_extend_zero_copy = None;
            let mut cur_start_of_row = start_of_row;
            let mut cur_end_of_row = end_of_row;

            // Step 2: go through stages one by one.
            for (i, stage) in self.shared.stages.iter().enumerate() {
                if let Some((e_idx, _, _)) = extend_info
                    && i < e_idx
                    && !(fv >= pre_fv_start && fv < pre_fv_end)
                {
                    continue;
                }

                if let Stage::Extend(e) = stage {
                    if !has_post_extend_ds {
                        post_extend_zero_copy = Some((e.image_size, e.frame_origin));
                        continue;
                    }
                    let (_, _, (canvas_rect, (raw_cx0, raw_cx1), (raw_cy0, raw_cy1))) =
                        extend_info.unwrap();
                    if canvas_rect.size.0 == 0 || canvas_rect.size.1 == 0 {
                        break;
                    }
                    let cy_signed = fv + e.frame_origin.1;
                    if cy_signed < canvas_rect.origin.1 as isize
                        || cy_signed >= canvas_rect.end().1 as isize
                    {
                        break;
                    }
                    let cy = cy_signed as usize;
                    let cx0 = canvas_rect.origin.0;
                    let cx1 = canvas_rect.end().0;
                    let x0_off = RowBuffer::x0_offset::<f32>();
                    let (pre_bufs, ext_and_after) = data.row_buffers.split_at_mut(i + 1);
                    for c in 0..num_channels {
                        if !self.shared.channel_is_used[c] {
                            continue;
                        }
                        let dst = &mut ext_and_after[0][c].get_row_mut::<f32>(cy)[x0_off..];
                        if cy < raw_cy0 || cy >= raw_cy1 {
                            e.process_row_chunk((cx0, cy), cx1 - cx0, c, dst);
                        } else {
                            if cx0 < raw_cx0 {
                                e.process_row_chunk((cx0, cy), raw_cx0 - cx0, c, dst);
                            }
                            let (pre_si, pre_ci) = self.stage_input_buffer_index[i][c];
                            let fx_start = (raw_cx0 as isize - e.frame_origin.0) as usize;
                            let src_off = x0_off + (fx_start - pre_area.origin.0);
                            let copy_len = raw_cx1 - raw_cx0;
                            let src = &pre_bufs[pre_si][pre_ci].get_row::<f32>(fv as usize)
                                [src_off..src_off + copy_len];
                            dst[raw_cx0 - cx0..raw_cx1 - cx0].copy_from_slice(src);
                            if raw_cx1 < cx1 {
                                e.process_row_chunk(
                                    (raw_cx1, cy),
                                    cx1 - raw_cx1,
                                    c,
                                    &mut dst[raw_cx1 - cx0..],
                                );
                            }
                        }
                    }
                    cur_fv = cy as isize;
                    cur_rect = canvas_rect;
                    save_clip_rect = canvas_rect;
                    space_size = e.image_size;
                    cur_start_of_row = cx0 == 0;
                    cur_end_of_row = cx1 == e.image_size.0;
                    continue;
                }

                let y = if self.stage_after_downsample[i] {
                    if cur_fv < cur_rect.origin.1 as isize || cur_fv >= cur_rect.end().1 as isize {
                        continue;
                    }
                    let fv_u = cur_fv as usize;
                    let out_dy = self.stage_output_downsample[i].1;
                    if ((fv_u + 1) & ((1usize << out_dy) - 1)) != 0 && fv_u + 1 != space_size.1 {
                        continue;
                    }
                    fv_u >> out_dy
                } else {
                    let (_, dy) = self.downsampling_for_stage[i];
                    let scaled_y_border =
                        self.stage_output_border_pixels[i].1.shrc(stage.shift().1) << dy;
                    let stage_vy = cur_fv + scaled_y_border as isize;
                    if stage_vy % (1 << dy) != 0 {
                        continue;
                    }
                    let y = stage_vy >> dy;
                    if matches!(stage, Stage::Save(_)) {
                        if stage_vy - (cur_rect.origin.1 as isize) < -(scaled_y_border as isize) {
                            continue;
                        }
                    } else if y < (cur_rect.origin.1 as isize - scaled_y_border as isize) >> dy {
                        continue;
                    }
                    let shifted_ysize = space_size.1.shrc(dy);
                    if y < 0 || y >= shifted_ysize as isize {
                        continue;
                    }
                    y as usize
                };

                self.run_stage_on_row(
                    i,
                    stage,
                    data,
                    buffers,
                    &mut previous_call_was_previous_row,
                    cur_rect,
                    save_clip_rect,
                    space_size,
                    post_extend_zero_copy,
                    y,
                    cur_start_of_row,
                    cur_end_of_row,
                )?;
            }
        }
        Ok(())
    }

    // Renders a chunk of data outside the current frame.
    #[instrument(skip(self, buffers))]
    pub(super) fn render_outside_frame_internal(
        &self,
        data: &mut LowMemoryRenderPipelinePerThread,
        xrange: Range<usize>,
        yrange: Range<usize>,
        buffers: &mut [Option<OutputChannelRef>],
    ) -> Result<()> {
        let num_channels = self.shared.num_channels();
        let x0 = xrange.start;
        let y0 = yrange.start;
        let xsize = xrange.clone().count();
        let ysize = yrange.clone().count();
        let strip_rect = Rect {
            origin: (x0, y0),
            size: (xsize, ysize),
        };
        let extend_idx = self.shared.extend_stage_index.unwrap();
        let Stage::Extend(extend) = &self.shared.stages[extend_idx] else {
            unreachable!("extend stage is not an extend stage");
        };
        let canvas_size = extend.image_size;
        let start_of_row = x0 == 0;
        let end_of_row = xrange.end == canvas_size.0;
        let mut previous_call_was_previous_row: SmallVec<bool, 32> =
            std::iter::repeat_n(false, self.shared.stages.len()).collect();

        for y in yrange.clone() {
            for c in 0..num_channels {
                if !self.shared.channel_is_used[c] {
                    continue;
                }
                let (si, ci) = if self.post_extend_downsample != (0, 0) {
                    (extend_idx + 1, c)
                } else {
                    self.stage_input_buffer_index[extend_idx][c]
                };
                let row =
                    &mut data.row_buffers[si][ci].get_row_mut(y)[RowBuffer::x0_offset::<f32>()..];
                extend.process_row_chunk((x0, y), xsize, c, row);
            }
            for (i, stage) in self.shared.stages.iter().enumerate().skip(extend_idx + 1) {
                let stage_y = if self.stage_after_downsample[i] {
                    let out_dy = self.stage_output_downsample[i].1;
                    if ((y + 1) & ((1usize << out_dy) - 1)) != 0 && y + 1 != canvas_size.1 {
                        continue;
                    }
                    y >> out_dy
                } else {
                    y
                };
                self.run_stage_on_row(
                    i,
                    stage,
                    data,
                    buffers,
                    &mut previous_call_was_previous_row,
                    strip_rect,
                    strip_rect,
                    canvas_size,
                    None,
                    stage_y,
                    start_of_row,
                    end_of_row,
                )?;
            }
        }
        Ok(())
    }
}
