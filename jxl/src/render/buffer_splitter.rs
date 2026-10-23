// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};

use crate::api::JxlOutputBuffer;
use crate::headers::Orientation;
use crate::image::{Image, ImageDataType, Rect};
use crate::util::sync::Mutex;
use crate::util::{ChannelVec, ShiftRightCeil};

pub struct OutputChannelSplitter<'a> {
    // Safety invariant: all the currently-borrowed rects of are stored in
    // `borrowed_rects`.
    buffer: JxlOutputBuffer<'a>,
    borrowed_rects: Mutex<BTreeSet<Rect>>,
}

impl<'a> Drop for OutputChannelSplitter<'a> {
    fn drop(&mut self) {
        assert!(
            self.borrowed_rects
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        )
    }
}

impl<'a> OutputChannelSplitter<'a> {
    pub fn new(buffer: JxlOutputBuffer<'a>) -> Self {
        Self {
            buffer,
            borrowed_rects: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn from_image<T: ImageDataType>(image: &'a mut Image<T>) -> Self {
        let size = image.size();
        let raw = image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size,
            })
            .into_raw();
        Self::new(JxlOutputBuffer::from_image_rect_mut(raw))
    }

    pub fn borrow_typed_rect<T: ImageDataType>(&self, rect: Rect) -> OutputChannelRef<'a, '_> {
        self.borrow_rect(rect.to_byte_rect(T::DATA_TYPE_ID))
    }

    #[allow(unsafe_code)]
    pub fn borrow_rect(&self, rect: Rect) -> OutputChannelRef<'a, '_> {
        let mut rects = self
            .borrowed_rects
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for r in rects.iter() {
            assert!(!r.intersects(&rect));
        }
        rects.insert(rect);

        OutputChannelRef {
            s: self,
            r: rect,
            // SAFETY: we just checked that `self.borrowed_rects` does not contain
            // any rects that intersect with the new rect. Since by safety invariant
            // all the currently-borrowed rects are in `self.borrowed_rects`, the
            // new rect does not intersect the old ones.
            buf: Some(unsafe { self.buffer.rect(rect) }),
        }
    }
}

pub struct OutputChannelRef<'a, 'b> {
    s: &'a OutputChannelSplitter<'b>,
    buf: Option<JxlOutputBuffer<'b>>,
    // Safety invariant: `r` is the rect that was used to obtain `buf` from `s`.
    r: Rect,
}

impl<'a, 'b> Deref for OutputChannelRef<'a, 'b> {
    type Target = JxlOutputBuffer<'b>;
    fn deref(&self) -> &Self::Target {
        self.buf.as_ref().unwrap()
    }
}

impl<'a, 'b> DerefMut for OutputChannelRef<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.as_mut().unwrap()
    }
}

impl<'a, 'b> Drop for OutputChannelRef<'a, 'b> {
    fn drop(&mut self) {
        self.buf = None;
        let mut b = self
            .s
            .borrowed_rects
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Safety note: we just dropped the local buffer, and the
        // safety invariant of `self` says that this is the correct rect
        // to remove.
        assert!(b.remove(&self.r));
    }
}

// Information for splitting the output buffers.
#[derive(Debug)]
pub struct SaveStageBufferInfo {
    pub downsample: (u8, u8),
    pub orientation: Orientation,
    pub byte_size: usize,
    pub after_extend: bool,
    pub post_extend_downsample: (u8, u8),
}

/// Computes the canvas rectangle owned by `rect` (given in frame coordinates) after an `Extend`
/// stage followed by optional downsampling `ds`, along with the unexpanded visible-frame canvas
/// intervals `(raw_cx0, raw_cx1)` and `(raw_cy0, raw_cy1)`.
pub(crate) fn extend_canvas_rect(
    rect: Rect,
    frame_size: (usize, usize),
    full_image_size: (usize, usize),
    frame_origin: (isize, isize),
    ds: (u8, u8),
) -> (Rect, (usize, usize), (usize, usize)) {
    let rect = rect.clip(frame_size);
    let (cw, ch) = full_image_size;
    let vx0 = frame_origin.0.clamp(0, cw as isize) as usize;
    let vx1 = (frame_origin.0 + frame_size.0 as isize).clamp(0, cw as isize) as usize;
    let vy0 = frame_origin.1.clamp(0, ch as isize) as usize;
    let vy1 = (frame_origin.1 + frame_size.1 as isize).clamp(0, ch as isize) as usize;
    if rect.size.0 == 0 || rect.size.1 == 0 || vx0 >= vx1 || vy0 >= vy1 {
        return (
            Rect {
                origin: (0, 0),
                size: (0, 0),
            },
            (0, 0),
            (0, 0),
        );
    }
    let rx0 = rect.origin.0 as isize + frame_origin.0;
    let rx1 = rect.end().0 as isize + frame_origin.0;
    let ry0 = rect.origin.1 as isize + frame_origin.1;
    let ry1 = rect.end().1 as isize + frame_origin.1;
    let raw_cx0 = rx0.clamp(vx0 as isize, vx1 as isize) as usize;
    let raw_cx1 = rx1.clamp(vx0 as isize, vx1 as isize) as usize;
    let raw_cy0 = ry0.clamp(vy0 as isize, vy1 as isize) as usize;
    let raw_cy1 = ry1.clamp(vy0 as isize, vy1 as isize) as usize;
    if raw_cx0 >= raw_cx1 || raw_cy0 >= raw_cy1 {
        return (
            Rect {
                origin: (0, 0),
                size: (0, 0),
            },
            (0, 0),
            (0, 0),
        );
    }
    let avx0 = (vx0 >> ds.0) << ds.0;
    let avx1 = (vx1.shrc(ds.0) << ds.0).min(cw);
    let avy0 = (vy0 >> ds.1) << ds.1;
    let avy1 = (vy1.shrc(ds.1) << ds.1).min(ch);
    let cx0 = if raw_cx0 == vx0 { avx0 } else { raw_cx0 };
    let cx1 = if raw_cx1 == vx1 { avx1 } else { raw_cx1 };
    let cy0 = if raw_cy0 == vy0 { avy0 } else { raw_cy0 };
    let cy1 = if raw_cy1 == vy1 { avy1 } else { raw_cy1 };
    (
        Rect {
            origin: (cx0, cy0),
            size: (cx1 - cx0, cy1 - cy0),
        },
        (raw_cx0, raw_cx1),
        (raw_cy0, raw_cy1),
    )
}

/// Data structure responsible for handing out access to portions of the output buffers.
pub struct BufferSplitter<'a> {
    buffers: ChannelVec<Option<OutputChannelSplitter<'a>>>,
    requested_rects: Mutex<Vec<Rect>>,
}

impl<'a> BufferSplitter<'a> {
    pub fn new(bufs: &'a mut [Option<JxlOutputBuffer<'_>>]) -> Self {
        Self {
            requested_rects: Mutex::new(vec![]),
            buffers: bufs
                .iter_mut()
                .map(|x| {
                    x.as_mut()
                        .map(|x| OutputChannelSplitter::new(JxlOutputBuffer::reborrow(x)))
                })
                .collect(),
        }
    }

    pub fn get_local_buffers(
        &self,
        save_buffer_info: &[Option<SaveStageBufferInfo>],
        rect: Rect,
        outside_current_frame: bool,
        frame_size: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
    ) -> ChannelVec<Option<OutputChannelRef<'a, '_>>> {
        {
            let mut req = self.requested_rects.lock().unwrap();
            req.push(rect);
        }
        let rect = if !outside_current_frame {
            rect.clip(frame_size)
        } else {
            rect
        };

        save_buffer_info
            .iter()
            .zip(self.buffers.iter())
            .map(|(info, buf)| {
                let Some(bi) = info else {
                    // We never write to this buffer.
                    return None;
                };
                let buf = buf.as_ref()?;
                if outside_current_frame && !bi.after_extend {
                    // Before-extend stages do not write to rects outside the current frame.
                    return None;
                }
                let mut channel_rect = rect.downsample(bi.downsample);
                if !outside_current_frame {
                    let ds_frame_size = (
                        frame_size.0.shrc(bi.downsample.0),
                        frame_size.1.shrc(bi.downsample.1),
                    );
                    channel_rect = channel_rect.clip(ds_frame_size);
                    if bi.after_extend {
                        let (canvas_rect, _, _) = extend_canvas_rect(
                            rect,
                            frame_size,
                            full_image_size,
                            frame_origin,
                            bi.post_extend_downsample,
                        );
                        channel_rect = canvas_rect.downsample(bi.downsample);
                    }
                }
                if channel_rect.size.0 == 0 || channel_rect.size.1 == 0 {
                    // Buffer would be empty anyway.
                    return None;
                }
                let target_size = if bi.after_extend {
                    full_image_size
                } else {
                    frame_size
                };
                let ds_target_size = (
                    target_size.0.shrc(bi.downsample.0),
                    target_size.1.shrc(bi.downsample.1),
                );
                let channel_rect = bi.orientation.display_rect(channel_rect, ds_target_size);
                let channel_rect = channel_rect.to_byte_rect_sz(bi.byte_size);
                Some(buf.borrow_rect(channel_rect))
            })
            .collect()
    }

    pub fn into_changed_regions(self) -> Vec<Rect> {
        self.requested_rects.into_inner().unwrap()
    }

    #[cfg(test)]
    pub fn get_full_buffers(&self) -> ChannelVec<Option<OutputChannelRef<'a, '_>>> {
        self.buffers
            .iter()
            .map(|x| {
                x.as_ref().map(|x| {
                    x.borrow_rect(Rect {
                        origin: (0, 0),
                        size: x.buffer.byte_size(),
                    })
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::JxlOutputBuffer;
    use crate::headers::Orientation;
    use crate::image::Rect;

    #[test]
    fn test_buffer_splitter_basic() {
        let mut empty_bufs: [Option<JxlOutputBuffer>; 0] = [];
        let splitter = BufferSplitter::new(&mut empty_bufs);
        assert!(splitter.get_full_buffers().is_empty());

        let mut raw0 = vec![0u8; 100];
        let mut raw1 = vec![0u8; 200];
        {
            let buf0 = JxlOutputBuffer::new(&mut raw0, 10, 10);
            let buf1 = JxlOutputBuffer::new(&mut raw1, 10, 20);
            let mut bufs = [Some(buf0), Some(buf1), None];
            let splitter = BufferSplitter::new(&mut bufs);

            {
                let mut full = splitter.get_full_buffers();
                assert_eq!(full.len(), 3);
                assert!(full[2].is_none());

                full[0].as_mut().unwrap().row_mut(0)[0] = 42;
                full[1].as_mut().unwrap().row_mut(1)[2] = 99;
            }
        }

        assert_eq!(raw0[0], 42);
        assert_eq!(raw1[20 + 2], 99);
    }

    #[test]
    #[should_panic]
    fn test_buffer_splitter_overlapping_borrows_panic() {
        let mut raw0 = vec![0u8; 400];
        let buf0 = JxlOutputBuffer::new(&mut raw0, 20, 20);
        let mut bufs = [Some(buf0)];
        let splitter = BufferSplitter::new(&mut bufs);

        let info = vec![Some(SaveStageBufferInfo {
            downsample: (0, 0),
            orientation: Orientation::Identity,
            byte_size: 1,
            after_extend: false,
            post_extend_downsample: (0, 0),
        })];

        let _local1 = splitter.get_local_buffers(
            &info,
            Rect {
                origin: (0, 0),
                size: (10, 10),
            },
            false,
            (20, 20),
            (20, 20),
            (0, 0),
        );

        let _local2 = splitter.get_local_buffers(
            &info,
            Rect {
                origin: (5, 5),
                size: (10, 10),
            },
            false,
            (20, 20),
            (20, 20),
            (0, 0),
        );
    }

    #[cfg(not(any(target_family = "wasm", target_arch = "wasm32")))]
    #[test]
    fn test_buffer_splitter_multithreaded_arbtest() {
        arbtest::arbtest(|u| {
            let width = u.int_in_range(16..=128)?;
            let height = u.int_in_range(16..=128)?;
            let num_threads = u.int_in_range(2..=8)?;
            let num_channels = u.int_in_range(1..=4)?;

            let downsample_x: u8 = u.int_in_range(0..=3)?;
            let downsample_y: u8 = u.int_in_range(0..=3)?;
            let byte_size: usize = *u.choose(&[1, 2, 4])?;
            let orientation = *u.choose(&[
                Orientation::Identity,
                Orientation::FlipHorizontal,
                Orientation::Rotate180,
                Orientation::FlipVertical,
                Orientation::Transpose,
                Orientation::Rotate90Cw,
                Orientation::AntiTranspose,
                Orientation::Rotate90Ccw,
            ])?;

            let info: Vec<Option<SaveStageBufferInfo>> = (0..num_channels)
                .map(|ch| {
                    if ch % 2 == 0 {
                        Some(SaveStageBufferInfo {
                            downsample: (downsample_x, downsample_y),
                            orientation,
                            byte_size,
                            after_extend: false,
                            post_extend_downsample: (0, 0),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            let ds_size = (width.shrc(downsample_x), height.shrc(downsample_y));
            let (out_w, out_h) = orientation.map_size(ds_size);
            let mut raws: Vec<Vec<u8>> = (0..num_channels)
                .map(|_| vec![0u8; out_w * out_h * byte_size])
                .collect();
            let mut tiles = Vec::new();

            {
                let mut bufs: Vec<Option<JxlOutputBuffer>> = raws
                    .iter_mut()
                    .enumerate()
                    .map(|(i, raw)| {
                        if i == num_channels - 1 && num_channels > 1 {
                            None
                        } else {
                            Some(JxlOutputBuffer::new(raw, out_h, out_w * byte_size))
                        }
                    })
                    .collect();

                let splitter = BufferSplitter::new(&mut bufs);

                let align_x = 1usize << downsample_x;
                let align_y = 1usize << downsample_y;
                let tile_w = u.int_in_range(1..=4)? * align_x;
                let tile_h = u.int_in_range(1..=4)? * align_y;

                let mut tile_id = 1u8;

                for y in (0..height).step_by(tile_h) {
                    for x in (0..width).step_by(tile_w) {
                        let w = tile_w.min(width - x);
                        let h = tile_h.min(height - y);
                        let rect = Rect {
                            origin: (x, y),
                            size: (w, h),
                        };
                        tiles.push((rect, tile_id));
                        tile_id = if tile_id == u8::MAX { 1 } else { tile_id + 1 };
                    }
                }

                let chunk_size = tiles.len().div_ceil(num_threads);
                let tile_chunks: Vec<_> = tiles
                    .chunks(chunk_size.max(1))
                    .map(|c| c.to_vec())
                    .collect();

                std::thread::scope(|s| {
                    for chunk in tile_chunks {
                        let splitter_ref = &splitter;
                        let info_ref = &info;
                        s.spawn(move || {
                            for (rect, id) in chunk {
                                let mut local = splitter_ref.get_local_buffers(
                                    info_ref,
                                    rect,
                                    false,
                                    (width, height),
                                    (width, height),
                                    (0, 0),
                                );

                                for buf in local.iter_mut().flatten() {
                                    let (_, bh) = buf.byte_size();
                                    for ry in 0..bh {
                                        let row = buf.row_mut(ry);
                                        row.fill(id);
                                    }
                                }
                            }
                        });
                    }
                });

                assert_eq!(splitter.into_changed_regions().len(), tiles.len());
            }

            for (ch, raw) in raws.iter().enumerate() {
                if ch % 2 == 0 && !(ch == num_channels - 1 && num_channels > 1) {
                    assert!(
                        raw.iter().all(|&b| b != 0),
                        "unwritten byte in channel {ch} for {width}x{height} ds=({downsample_x},{downsample_y}) {orientation:?}"
                    );
                }
            }

            Ok(())
        });
    }
}
