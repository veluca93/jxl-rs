// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::{Endianness, JxlDataFormat, JxlOutputBuffer};
use crate::error::Result;
use crate::image::Image;
use crate::render::buffer_splitter::OutputChannelRef;
use crate::render::save::{SaveChannelType, SaveStage};
use crate::util::{DITHER_TABLE, f16};

impl SaveStage {
    pub(super) fn save_simple(
        &self,
        data: &[Image<f64>],
        buffers: &mut [Option<OutputChannelRef>],
    ) -> Result<()> {
        for i in self.channels.iter().skip(1) {
            assert_eq!(data[self.channels[0]].size(), data[*i].size());
        }
        let Some(buf) = buffers[self.output_buffer_index].as_mut() else {
            return Ok(());
        };
        let size = data[0].size();

        self.check_buffer_size(size, Some(buf))?;

        let output_channels = self.output_channels();
        let bps = self.data_format.bytes_per_sample();
        let pixel_bytes = output_channels * bps;
        // Align `px_buf` to 4 bytes so that the slice passed to `buf.write_pixel` is aligned to
        // `bps` (1, 2, or 4 bytes) as promised by `JxlOutputTransform`.
        #[repr(C, align(4))]
        struct AlignedPixel([u8; 16]);
        let mut px_buf = AlignedPixel([0u8; 16]);
        let px_buf = &mut px_buf.0[..pixel_bytes];

        // Pre-fill opaque alpha if needed (when RGBA requested but image has no alpha)
        if self.fill_opaque_alpha {
            let alpha_offset = self.channels.len() * bps;
            let opaque_bytes = self.data_format.opaque_alpha_bytes();
            px_buf[alpha_offset..alpha_offset + bps].copy_from_slice(&opaque_bytes);
        }

        for y in 0..size.1 {
            for x in 0..size.0 {
                let (dx, dy) = self.orientation.display_pixel((x, y), size);

                for (c, (&chan, &ch_ty)) in self
                    .channels
                    .iter()
                    .zip(self.channel_types.iter())
                    .enumerate()
                {
                    let px = data[chan].row(y)[x];

                    macro_rules! write_sample {
                        ($px: expr, $endianness: expr) => {{
                            let px = $px;
                            let sample_bytes = if $endianness == Endianness::LittleEndian {
                                px.to_le_bytes()
                            } else {
                                px.to_be_bytes()
                            };
                            px_buf[c * bps..(c + 1) * bps].copy_from_slice(&sample_bytes);
                        }};
                    }

                    match (ch_ty, self.data_format) {
                        (SaveChannelType::F32, JxlDataFormat::U8 { bit_depth }) => {
                            let max = ((1u32 << bit_depth) - 1) as f32;
                            let dither = DITHER_TABLE[(y + chan * 13) % 32][(x + chan * 23) % 32];
                            let v = ((px as f32) * max + dither).clamp(0.0, max).round() as u8;
                            write_sample!(v, Endianness::LittleEndian);
                        }
                        (
                            SaveChannelType::I16 { bit_depth: in_bd }
                            | SaveChannelType::I32 { bit_depth: in_bd },
                            JxlDataFormat::U8 { bit_depth: out_bd },
                        ) => {
                            let max = (1i32 << out_bd) - 1;
                            let scale = max / ((1i32 << in_bd) - 1);
                            let v = ((px as i32) * scale).clamp(0, max) as u8;
                            write_sample!(v, Endianness::LittleEndian);
                        }
                        (
                            SaveChannelType::F32,
                            JxlDataFormat::U16 {
                                endianness,
                                bit_depth,
                            },
                        ) => {
                            let max = ((1u32 << bit_depth) - 1) as f32;
                            let v = ((px as f32).clamp(0.0, 1.0) * max).round() as u16;
                            write_sample!(v, endianness);
                        }
                        (SaveChannelType::F32, JxlDataFormat::F32 { endianness }) => {
                            write_sample!(px as f32, endianness);
                        }
                        (SaveChannelType::F32, JxlDataFormat::F16 { endianness }) => {
                            write_sample!(f16::from_f64(px), endianness);
                        }
                        (
                            SaveChannelType::I16 { bit_depth: in_bd },
                            JxlDataFormat::F16 { endianness },
                        ) => {
                            let scale = 1.0 / ((1u64 << in_bd) - 1) as f32;
                            write_sample!(f16::from_f32((px as f32) * scale), endianness);
                        }
                        _ => unreachable!(),
                    }
                }

                buf.write_pixel(dy, dx, px_buf);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use test_log::test;

    use super::*;
    use crate::api::JxlColorType;
    use crate::headers::Orientation;
    use crate::image::Rect;
    use crate::render::buffer_splitter::OutputChannelSplitter;
    use crate::tests::assert_close;

    #[test]
    fn save_stage() -> Result<()> {
        let save_stage = SaveStage::new(
            &[0],
            Orientation::Identity,
            0,
            JxlColorType::Grayscale,
            JxlDataFormat::U8 { bit_depth: 8 },
            false,
        );
        let mut rng = XorShiftRng::seed_from_u64(0);
        let src = [Image::<f64>::new_random((128, 128), &mut rng)?];
        let mut dst = Image::<u8>::new_random((128, 128), &mut rng)?;

        {
            let r = Rect {
                size: (128, 128),
                origin: (0, 0),
            };
            let splitter = OutputChannelSplitter::new(JxlOutputBuffer::from_image_rect_mut(
                dst.get_rect_mut(r).into_raw(),
            ));
            save_stage.save_simple(&src, &mut [Some(splitter.borrow_rect(r))])?;
        }

        for y in 0..128 {
            for x in 0..128 {
                let dither = DITHER_TABLE[y % 32][x % 32];
                let expected = ((src[0].row(y)[x] as f32) * 255.0 + dither)
                    .clamp(0.0, 255.0)
                    .round() as u8;
                assert_eq!(expected, dst.row(y)[x]);
            }
        }

        Ok(())
    }

    fn do_test_orientation(
        orientation: Orientation,
        transform: impl Fn(usize, usize, usize, usize) -> (usize, usize),
    ) -> Result<()> {
        let (w, h) = (32, 16);
        let mut rng = XorShiftRng::seed_from_u64(0);
        let src = [Image::<f64>::new_random((w, h), &mut rng)?];

        let (ow, oh) = if orientation.is_transposing() {
            (h, w)
        } else {
            (w, h)
        };

        let save_stage = SaveStage::new(
            &[0],
            orientation,
            0,
            JxlColorType::Grayscale,
            JxlDataFormat::f32(),
            false,
        );

        let mut rng = XorShiftRng::seed_from_u64(0);
        let mut dst = Image::<f32>::new_random((ow, oh), &mut rng)?;

        {
            let r = Rect {
                size: (ow, oh),
                origin: (0, 0),
            };
            let splitter = OutputChannelSplitter::new(JxlOutputBuffer::from_image_rect_mut(
                dst.get_rect_mut(r).into_raw(),
            ));
            let byte_rect = r.to_byte_rect_sz(std::mem::size_of::<f32>());
            save_stage.save_simple(&src, &mut [Some(splitter.borrow_rect(byte_rect))])?;
        }

        // Iterate over the DESTINATION image pixels.
        for y_dest in 0..oh {
            for x_dest in 0..ow {
                // For each destination pixel, find its corresponding source pixel.
                let (src_x, src_y) = transform(x_dest, y_dest, w, h);
                assert_close!(
                    dst.row(y_dest)[x_dest],
                    src[0].row(src_y)[src_x] as f32,
                    1e-5,
                    rel: 1e-5,
                );
            }
        }

        Ok(())
    }

    #[test]
    fn orientation_identity() -> Result<()> {
        do_test_orientation(Orientation::Identity, |x, y, _, _| (x, y))
    }

    #[test]
    fn orientation_flip_horizontal() -> Result<()> {
        do_test_orientation(Orientation::FlipHorizontal, |x, y, w, _| (w - 1 - x, y))
    }

    #[test]
    fn orientation_flip_vertical() -> Result<()> {
        do_test_orientation(Orientation::FlipVertical, |x, y, _, h| (x, h - 1 - y))
    }

    #[test]
    fn orientation_rotate_180() -> Result<()> {
        do_test_orientation(Orientation::Rotate180, |x, y, w, h| (w - 1 - x, h - 1 - y))
    }

    // transposing orientations

    #[test]
    fn orientation_transpose() -> Result<()> {
        do_test_orientation(Orientation::Transpose, |x_dest, y_dest, _, _| {
            (y_dest, x_dest)
        })
    }

    #[test]
    fn orientation_rotate_90_cw() -> Result<()> {
        do_test_orientation(Orientation::Rotate90Cw, |x_dest, y_dest, _, h_src| {
            (y_dest, h_src - 1 - x_dest)
        })
    }

    #[test]
    fn orientation_anti_transpose() -> Result<()> {
        do_test_orientation(
            Orientation::AntiTranspose,
            |x_dest, y_dest, w_src, h_src| (w_src - 1 - y_dest, h_src - 1 - x_dest),
        )
    }

    #[test]
    fn orientation_rotate_90_ccw() -> Result<()> {
        do_test_orientation(Orientation::Rotate90Ccw, |x_dest, y_dest, w_src, _| {
            (w_src - 1 - y_dest, x_dest)
        })
    }
}
