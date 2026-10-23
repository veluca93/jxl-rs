// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdDescriptor, simd_function};

#[cfg(test)]
use crate::image::Image;
use crate::util::ShiftRightCeil;

/// Box downsampling stage reducing a single `f32` channel by `1 << shift` (`2x2`, `4x4`, or `8x8`).
///
/// Output dimensions are `(W.shrc(shift), H.shrc(shift))`. Partial boxes along the right and
/// bottom image edges average only the valid in-bounds pixels (`valid_cols * valid_rows`).
#[derive(Debug, Clone, Copy)]
pub struct DownsampleStage {
    channel: usize,
    shift: u8,
}

impl DownsampleStage {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(channel: usize, shift: u8) -> Self {
        assert!(
            (1..=3).contains(&shift),
            "unsupported downsample shift {shift}; expected 1 (2x2), 2 (4x4), or 3 (8x8)"
        );
        Self { channel, shift }
    }

    pub fn channel(&self) -> usize {
        self.channel
    }

    pub fn shift(&self) -> u8 {
        self.shift
    }

    pub fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    pub(in crate::render) fn process_row_chunk(
        &self,
        xsize: usize,
        input_rows: &[&[f32]],
        output_row: &mut [f32],
    ) {
        downsample_row_simd_dispatch(self.shift, xsize, input_rows, output_row);
    }

    #[cfg(test)]
    pub(in crate::render) fn run_simple(
        &self,
        chunk_size: usize,
        input: &Image<f64>,
        output: &mut Image<f64>,
    ) {
        let n = 1usize << self.shift;
        let (in_w, in_h) = input.size();
        let (out_w, out_h) = output.size();
        assert_eq!(
            (out_w, out_h),
            (in_w.shrc(self.shift), in_h.shrc(self.shift))
        );
        if out_w == 0 || out_h == 0 {
            return;
        }
        let mut in_f32: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; in_w]).collect();
        let mut out_f32 = vec![0.0f32; out_w];
        for oy in 0..out_h {
            let iy0 = oy << self.shift;
            let iy1 = (iy0 + n).min(in_h);
            let valid_rows = iy1 - iy0;
            for (r, dst) in (iy0..iy1).zip(in_f32.iter_mut()) {
                for (d, &s) in dst.iter_mut().zip(input.row(r)) {
                    *d = s as f32;
                }
            }
            for ox0 in (0..out_w).step_by(chunk_size) {
                let ox1 = (ox0 + chunk_size).min(out_w);
                let ix0 = ox0 << self.shift;
                let ix1 = (ox1 << self.shift).min(in_w);
                let mut refs: [&[f32]; 8] = [&[]; 8];
                for (r, slot) in in_f32[..valid_rows].iter().zip(refs.iter_mut()) {
                    *slot = &r[ix0..ix1];
                }
                self.process_row_chunk(ix1 - ix0, &refs[..valid_rows], &mut out_f32[ox0..ox1]);
            }
            for (d, &s) in output.row_mut(oy).iter_mut().zip(&out_f32) {
                *d = s as f64;
            }
        }
    }
}

impl std::fmt::Display for DownsampleStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let factor = 1usize << self.shift;
        write!(
            f,
            "{factor}x{factor} box downsampling of channel {}",
            self.channel
        )
    }
}

#[inline(always)]
fn hreduce_box<D: SimdDescriptor, const N: usize>(d: D, vsum: [D::F32Vec; N]) -> D::F32Vec {
    const { assert!(D::F32Vec::LEN <= 16) };
    let len = D::F32Vec::LEN;
    match N {
        2 => {
            let mut tmp = [0.0f32; 32];
            for (v, &vec) in vsum.iter().enumerate() {
                vec.store(&mut tmp[v * len..(v + 1) * len]);
            }
            let (c0, c1) = D::F32Vec::load_deinterleaved_2(d, &tmp[..2 * len]);
            c0 + c1
        }
        4 => {
            let mut tmp = [0.0f32; 64];
            for (v, &vec) in vsum.iter().enumerate() {
                vec.store(&mut tmp[v * len..(v + 1) * len]);
            }
            let (c0, c1, c2, c3) = D::F32Vec::load_deinterleaved_4(d, &tmp[..4 * len]);
            (c0 + c1) + (c2 + c3)
        }
        8 => {
            let mut tmp = [0.0f32; 128];
            for (v, &vec) in vsum.iter().enumerate() {
                vec.store(&mut tmp[v * len..(v + 1) * len]);
            }
            let mut pair = [0.0f32; 64];
            for p in 0..4 {
                let (a, b) =
                    D::F32Vec::load_deinterleaved_2(d, &tmp[2 * p * len..2 * (p + 1) * len]);
                (a + b).store(&mut pair[p * len..(p + 1) * len]);
            }
            let (q0, q1, q2, q3) = D::F32Vec::load_deinterleaved_4(d, &pair[..4 * len]);
            (q0 + q1) + (q2 + q3)
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn downsample_row_impl<D: SimdDescriptor, const N: usize, const SHIFT: u8>(
    d: D,
    xsize: usize,
    input_rows: &[&[f32]],
    output_row: &mut [f32],
) {
    let valid_rows = input_rows.len();
    assert!((1..=N).contains(&valid_rows));
    if xsize == 0 {
        return;
    }
    let out_xsize = xsize.shrc(SHIFT);
    assert!(output_row.len() >= out_xsize);
    for &row in input_rows {
        assert!(row.len() >= xsize);
    }

    let len = D::F32Vec::LEN;
    let full_blocks = (xsize >> SHIFT) / len;
    let inv_full = D::F32Vec::splat(d, 1.0f32 / ((N * valid_rows) as f32));

    for blk in 0..full_blocks {
        let in_base = blk * N * len;
        let first = input_rows[0];
        let mut vsum: [D::F32Vec; N] =
            std::array::from_fn(|v| D::F32Vec::load(d, &first[in_base + v * len..]));
        for &row in &input_rows[1..] {
            for (v, acc) in vsum.iter_mut().enumerate() {
                *acc += D::F32Vec::load(d, &row[in_base + v * len..]);
            }
        }
        (hreduce_box::<D, N>(d, vsum) * inv_full).store(&mut output_row[blk * len..]);
    }

    for (ox, out_px) in output_row[..out_xsize]
        .iter_mut()
        .enumerate()
        .skip(full_blocks * len)
    {
        let ix0 = ox << SHIFT;
        let ix1 = (ix0 + N).min(xsize);
        let mut sum = 0.0f32;
        for &row in input_rows {
            for &px in &row[ix0..ix1] {
                sum += px;
            }
        }
        *out_px = sum * (1.0f32 / (((ix1 - ix0) * valid_rows) as f32));
    }
}

simd_function!(
    downsample_row_simd_dispatch,
    d: D,
    fn downsample_row_simd(
        shift: u8,
        xsize: usize,
        input_rows: &[&[f32]],
        output_row: &mut [f32],
    ) {
        match shift {
            1 => downsample_row_impl::<D, 2, 1>(d, xsize, input_rows, output_row),
            2 => downsample_row_impl::<D, 4, 2>(d, xsize, input_rows, output_row),
            3 => downsample_row_impl::<D, 8, 3>(d, xsize, input_rows, output_row),
            _ => unreachable!(),
        }
    }
);

#[cfg(test)]
mod tests {
    use jxl_simd::{SimdDescriptor, test_all_instruction_sets};

    use super::{DownsampleStage, downsample_row_impl};
    use crate::util::ShiftRightCeil;

    fn test_downsample_row_simd<D: SimdDescriptor>(d: D) {
        for shift in 1..=3u8 {
            let stage = DownsampleStage::new(0, shift);
            assert_eq!(stage.channel(), 0);
            assert_eq!(stage.shift(), shift);
            assert!(stage.uses_channel(0));
            assert!(!stage.uses_channel(1));
            let n = 1usize << shift;
            for valid_rows in 1..=n {
                for xsize in (0..=35).chain([63, 64, 65, 127, 128, 131]) {
                    let rows_storage: Vec<Vec<f32>> = (0..valid_rows)
                        .map(|r| {
                            (0..xsize)
                                .map(|x| ((r * 137 + x * 53 + 11) % 997) as f32 * 0.001 - 0.3)
                                .collect()
                        })
                        .collect();
                    let row_refs: Vec<&[f32]> = rows_storage.iter().map(|r| r.as_slice()).collect();
                    let out_xsize = xsize.shrc(shift);
                    let mut actual = vec![-99.0f32; out_xsize + 3];
                    match shift {
                        1 => downsample_row_impl::<D, 2, 1>(d, xsize, &row_refs, &mut actual),
                        2 => downsample_row_impl::<D, 4, 2>(d, xsize, &row_refs, &mut actual),
                        3 => downsample_row_impl::<D, 8, 3>(d, xsize, &row_refs, &mut actual),
                        _ => unreachable!(),
                    }
                    assert_eq!(&actual[out_xsize..], &[-99.0; 3]);
                    if xsize == 35 {
                        let mut dispatched = vec![0.0f32; out_xsize];
                        stage.process_row_chunk(xsize, &row_refs, &mut dispatched);
                        for (a, b) in dispatched.iter().zip(&actual[..out_xsize]) {
                            assert!((a - b).abs() <= 1e-6);
                        }
                    }
                    for (ox, &got) in actual[..out_xsize].iter().enumerate() {
                        let ix0 = ox << shift;
                        let ix1 = (ix0 + n).min(xsize);
                        let sum: f32 = row_refs.iter().flat_map(|r| &r[ix0..ix1]).copied().sum();
                        let expected = sum / (((ix1 - ix0) * valid_rows) as f32);
                        assert!(
                            (got - expected).abs() <= 1e-6,
                            "shift={shift} valid_rows={valid_rows} xsize={xsize} ox={ox}: got {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    test_all_instruction_sets!(test_downsample_row_simd);

    use std::sync::Arc;

    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use crate::api::{Endianness, JxlColorType, JxlDataFormat, JxlOutputBuffer};
    use crate::error::{Error, Result};
    use crate::frame::ReferenceFrame;
    use crate::headers::Orientation;
    use crate::headers::frame_header::{BlendingInfo, BlendingMode};
    use crate::image::{BufferRecycler, DataTypeTag, Image};
    use crate::render::buffer_splitter::BufferSplitter;
    use crate::render::stages::{
        ExtendToImageDimensionsStage, GaborishStage, HorizontalChromaUpsample,
        PremultiplyAlphaStage, VerticalChromaUpsample,
    };
    use crate::render::{
        LowMemoryRenderPipeline, RenderPipeline, RenderPipelineBuilder, SimpleRenderPipeline,
    };

    fn run_pipeline<P: RenderPipeline>(
        build: impl FnOnce(RenderPipelineBuilder<P>) -> RenderPipelineBuilder<P>,
        inputs: &[Image<f32>],
        frame_size: (usize, usize),
        log_group_size: usize,
        out_specs: &[((usize, usize), usize)],
        reverse_groups: bool,
    ) -> Result<Vec<Vec<u8>>> {
        let ch_types = vec![(DataTypeTag::F32, None); inputs.len()];
        let recycler = Arc::new(BufferRecycler::new(1 << log_group_size));
        let builder =
            RenderPipelineBuilder::<P>::new(&ch_types, frame_size, 0, log_group_size, recycler);
        let mut pipeline = build(builder).build()?;

        let mut raw_bufs: Vec<Vec<u8>> = out_specs
            .iter()
            .map(|&((w, h), bpp)| vec![0xA5u8; w * h * bpp])
            .collect();
        {
            let mut jxl_bufs: Vec<Option<JxlOutputBuffer>> = raw_bufs
                .iter_mut()
                .zip(out_specs.iter())
                .map(|(raw, &((w, h), bpp))| Some(JxlOutputBuffer::new(raw, h, w * bpp)))
                .collect();
            pipeline.check_buffer_sizes(&mut jxl_bufs)?;
            let splitter = BufferSplitter::new(&mut jxl_bufs);

            let gx_count = frame_size.0.shrc(log_group_size);
            let gy_count = frame_size.1.shrc(log_group_size);
            let num_groups = gx_count * gy_count;
            let mut order: Vec<usize> = (0..num_groups).collect();
            if reverse_groups {
                order.reverse();
            }
            for g in order {
                let (gx, gy) = (g % gx_count, g / gx_count);
                for (c, img) in inputs.iter().enumerate() {
                    if !pipeline.used_channel_mask()[c] {
                        continue;
                    }
                    let mut buf = pipeline.get_buffer::<f32>(c)?;
                    let (iw, ih) = img.size();
                    let ds_x = usize::from(iw < frame_size.0);
                    let ds_y = usize::from(ih < frame_size.1);
                    let gsz_x = 1usize << (log_group_size - ds_x);
                    let gsz_y = 1usize << (log_group_size - ds_y);
                    let ox = gx * gsz_x;
                    let oy = gy * gsz_y;
                    let rw = (iw - ox).min(gsz_x);
                    let rh = (ih - oy).min(gsz_y);
                    for y in 0..rh {
                        buf.row_mut(y)[..rw].copy_from_slice(&img.row(oy + y)[ox..ox + rw]);
                    }
                    pipeline.set_buffer_for_group(c, g, true, buf, &splitter)?;
                }
            }
            pipeline.render_outside_frame(&splitter)?;
        }
        Ok(raw_bufs)
    }

    fn assert_f32_close(a: &[u8], b: &[u8], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: byte len mismatch");
        for (idx, (ca, cb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            let fa = f32::from_ne_bytes(ca.try_into().unwrap());
            let fb = f32::from_ne_bytes(cb.try_into().unwrap());
            assert!(
                (fa - fb).abs() <= tol,
                "{ctx} sample {idx}: lowmem={fa} vs simple={fb}"
            );
        }
    }

    #[test]
    fn downsample_stage_consistency() -> Result<()> {
        for shift in 1..=3 {
            crate::render::test::test_stage_consistency(
                || DownsampleStage::new(0, shift),
                (157, 131),
                3,
            )?;
        }
        Ok(())
    }

    #[test]
    fn downsample_pipeline_formats_and_orientations() -> Result<()> {
        let mut rng = XorShiftRng::seed_from_u64(42);
        let size = (515, 259);
        let inputs = [
            Image::<f32>::new_random(size, &mut rng)?,
            Image::<f32>::new_random(size, &mut rng)?,
            Image::<f32>::new_random(size, &mut rng)?,
        ];

        for shift in 1..=3u8 {
            let ds_size = (size.0.shrc(shift), size.1.shrc(shift));
            for orientation in [
                Orientation::Identity,
                Orientation::FlipHorizontal,
                Orientation::FlipVertical,
                Orientation::Rotate90Cw,
                Orientation::Transpose,
                Orientation::AntiTranspose,
            ] {
                let out_dim = orientation.map_size(ds_size);
                fn build_f32<P: RenderPipeline>(
                    p: RenderPipelineBuilder<P>,
                    shift: u8,
                    orientation: Orientation,
                ) -> RenderPipelineBuilder<P> {
                    p.add_downsample_stage(DownsampleStage::new(0, shift))
                        .add_downsample_stage(DownsampleStage::new(1, shift))
                        .add_downsample_stage(DownsampleStage::new(2, shift))
                        .add_save_stage(
                            &[0, 1, 2],
                            orientation,
                            0,
                            JxlColorType::Rgb,
                            JxlDataFormat::f32(),
                            false,
                        )
                }
                let simple = run_pipeline::<SimpleRenderPipeline>(
                    |p| build_f32(p, shift, orientation),
                    &inputs,
                    size,
                    8,
                    &[(out_dim, 12)],
                    false,
                )?;
                let lowmem = run_pipeline::<LowMemoryRenderPipeline>(
                    |p| build_f32(p, shift, orientation),
                    &inputs,
                    size,
                    8,
                    &[(out_dim, 12)],
                    true,
                )?;
                assert_f32_close(
                    &lowmem[0],
                    &simple[0],
                    1e-5,
                    &format!("shift={shift} {orientation:?}"),
                );
            }

            // U8 (dithered) and F16 output formats.
            for (fmt, bpp) in [
                (JxlDataFormat::U8 { bit_depth: 8 }, 4usize),
                (
                    JxlDataFormat::F16 {
                        endianness: Endianness::native(),
                    },
                    8usize,
                ),
            ] {
                fn build_fmt<P: RenderPipeline>(
                    p: RenderPipelineBuilder<P>,
                    shift: u8,
                    fmt: JxlDataFormat,
                ) -> RenderPipelineBuilder<P> {
                    p.add_downsample_stage(DownsampleStage::new(0, shift))
                        .add_downsample_stage(DownsampleStage::new(1, shift))
                        .add_downsample_stage(DownsampleStage::new(2, shift))
                        .add_save_stage(
                            &[0, 1, 2],
                            Orientation::Identity,
                            0,
                            JxlColorType::Rgba,
                            fmt,
                            true,
                        )
                }
                let simple = run_pipeline::<SimpleRenderPipeline>(
                    |p| build_fmt(p, shift, fmt),
                    &inputs,
                    size,
                    8,
                    &[(ds_size, bpp)],
                    false,
                )?;
                let lowmem = run_pipeline::<LowMemoryRenderPipeline>(
                    |p| build_fmt(p, shift, fmt),
                    &inputs,
                    size,
                    8,
                    &[(ds_size, bpp)],
                    true,
                )?;
                match fmt {
                    JxlDataFormat::U8 { .. } => {
                        assert_eq!(lowmem[0].len(), simple[0].len());
                        for (idx, (&a, &b)) in lowmem[0].iter().zip(simple[0].iter()).enumerate() {
                            assert!(
                                a.abs_diff(b) <= 1,
                                "shift={shift} U8 byte {idx}: lowmem={a} vs simple={b}"
                            );
                        }
                    }
                    JxlDataFormat::F16 { .. } => {
                        assert_eq!(lowmem[0].len(), simple[0].len());
                        for (idx, (ca, cb)) in lowmem[0]
                            .chunks_exact(2)
                            .zip(simple[0].chunks_exact(2))
                            .enumerate()
                        {
                            let ua = u16::from_ne_bytes(ca.try_into().unwrap());
                            let ub = u16::from_ne_bytes(cb.try_into().unwrap());
                            assert!(
                                ua.abs_diff(ub) <= 1,
                                "shift={shift} F16 sample {idx}: lowmem={ua:#06x} vs simple={ub:#06x}"
                            );
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }

    #[test]
    fn downsample_with_borders_chroma_and_chaining() -> Result<()> {
        let mut rng = XorShiftRng::seed_from_u64(7);
        // 257x513 exercises 1-pixel edge groups in both dimensions.
        let size = (257, 513);
        let inputs = [
            Image::<f32>::new_random(size, &mut rng)?,
            Image::<f32>::new_random((size.0.shrc(1), size.1.shrc(1)), &mut rng)?,
            Image::<f32>::new_random(size, &mut rng)?,
            Image::<f32>::new_random(size, &mut rng)?,
        ];
        let out_size = (size.0.shrc(2), size.1.shrc(2));
        fn build<P: RenderPipeline>(p: RenderPipelineBuilder<P>) -> RenderPipelineBuilder<P> {
            p.add_inout_stage(HorizontalChromaUpsample::new(1))
                .add_inout_stage(VerticalChromaUpsample::new(1))
                .add_inout_stage(GaborishStage::new([0.1, 0.1, 0.1], [0.05, 0.05, 0.05]))
                // Chain 2x -> 2x on channels 0..4, followed by InPlace PremultiplyAlphaStage.
                .add_downsample_stage(DownsampleStage::new(0, 1))
                .add_downsample_stage(DownsampleStage::new(1, 1))
                .add_downsample_stage(DownsampleStage::new(2, 1))
                .add_downsample_stage(DownsampleStage::new(3, 1))
                .add_downsample_stage(DownsampleStage::new(0, 1))
                .add_downsample_stage(DownsampleStage::new(1, 1))
                .add_downsample_stage(DownsampleStage::new(2, 1))
                .add_downsample_stage(DownsampleStage::new(3, 1))
                .add_inplace_stage(PremultiplyAlphaStage::new(0, 3, 3))
                .add_save_stage(
                    &[0, 1, 2, 3],
                    Orientation::Rotate90Cw,
                    0,
                    JxlColorType::Rgba,
                    JxlDataFormat::f32(),
                    false,
                )
        }
        let out_dim = Orientation::Rotate90Cw.map_size(out_size);
        let simple =
            run_pipeline::<SimpleRenderPipeline>(build, &inputs, size, 8, &[(out_dim, 16)], false)?;
        let lowmem = run_pipeline::<LowMemoryRenderPipeline>(
            build,
            &inputs,
            size,
            8,
            &[(out_dim, 16)],
            true,
        )?;
        assert_f32_close(&lowmem[0], &simple[0], 1e-5, "borders+chroma+chained");
        Ok(())
    }

    fn default_blending() -> BlendingInfo {
        BlendingInfo {
            mode: BlendingMode::Replace,
            alpha_channel: 0,
            clamp: false,
            source: 0,
        }
    }

    #[test]
    fn downsample_after_extend() -> Result<()> {
        let mut rng = XorShiftRng::seed_from_u64(99);
        let frame_size = (301, 277);
        let canvas_size = (419, 365);
        let inputs = [
            Image::<f32>::new_random(frame_size, &mut rng)?,
            Image::<f32>::new_random((frame_size.0.shrc(1), frame_size.1.shrc(1)), &mut rng)?,
            Image::<f32>::new_random(frame_size, &mut rng)?,
        ];
        let ref_frame = ReferenceFrame::random(&mut rng, canvas_size.0, canvas_size.1, 3, false)?;
        let refs = Arc::new([Some(ref_frame), None, None, None]);

        for frame_origin in [(27isize, 19isize), (-35, -17), (500, 500)] {
            for shift in [1u8, 2, 3] {
                let make_extend = || ExtendToImageDimensionsStage {
                    frame_origin,
                    image_size: canvas_size,
                    blending_info: default_blending(),
                    ec_blending_info: vec![],
                    reference_frames: refs.clone(),
                    zeros: vec![0.0; canvas_size.0],
                };
                let ds_canvas = (canvas_size.0.shrc(shift), canvas_size.1.shrc(shift));
                let specs = [
                    (frame_size, 12),
                    (Orientation::Rotate90Cw.map_size(ds_canvas), 12),
                ];
                fn build<P: RenderPipeline>(
                    p: RenderPipelineBuilder<P>,
                    extend: ExtendToImageDimensionsStage,
                    shift: u8,
                ) -> RenderPipelineBuilder<P> {
                    p.add_inout_stage(HorizontalChromaUpsample::new(1))
                        .add_inout_stage(VerticalChromaUpsample::new(1))
                        .add_inout_stage(GaborishStage::new([0.1, 0.1, 0.1], [0.05, 0.05, 0.05]))
                        .add_save_stage(
                            &[0, 1, 2],
                            Orientation::Identity,
                            0,
                            JxlColorType::Rgb,
                            JxlDataFormat::f32(),
                            false,
                        )
                        .add_extend_stage(extend)
                        .add_downsample_stage(DownsampleStage::new(0, shift))
                        .add_downsample_stage(DownsampleStage::new(1, shift))
                        .add_downsample_stage(DownsampleStage::new(2, shift))
                        .add_save_stage(
                            &[0, 1, 2],
                            Orientation::Rotate90Cw,
                            1,
                            JxlColorType::Rgb,
                            JxlDataFormat::f32(),
                            false,
                        )
                }
                let simple = run_pipeline::<SimpleRenderPipeline>(
                    |p| build(p, make_extend(), shift),
                    &inputs,
                    frame_size,
                    8,
                    &specs,
                    false,
                )?;
                let lowmem = run_pipeline::<LowMemoryRenderPipeline>(
                    |p| build(p, make_extend(), shift),
                    &inputs,
                    frame_size,
                    8,
                    &specs,
                    true,
                )?;
                assert_f32_close(
                    &lowmem[0],
                    &simple[0],
                    1e-5,
                    &format!("pre-extend origin={frame_origin:?} shift={shift}"),
                );
                assert_f32_close(
                    &lowmem[1],
                    &simple[1],
                    1e-5,
                    &format!("post-extend origin={frame_origin:?} shift={shift}"),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn downsample_builder_validation_errors() {
        let mk = || {
            RenderPipelineBuilder::<LowMemoryRenderPipeline>::new(
                &[(DataTypeTag::F32, None); 3],
                (64, 64),
                0,
                8,
                Arc::new(BufferRecycler::new(256)),
            )
        };

        // BORDER > 0 after DownsampleStage is rejected.
        let err = mk()
            .add_downsample_stage(DownsampleStage::new(0, 1))
            .add_downsample_stage(DownsampleStage::new(1, 1))
            .add_downsample_stage(DownsampleStage::new(2, 1))
            .add_inout_stage(GaborishStage::new([0.1; 3], [0.05; 3]))
            .add_save_stage(
                &[0, 1, 2],
                Orientation::Identity,
                0,
                JxlColorType::Rgb,
                JxlDataFormat::f32(),
                false,
            )
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, Error::PipelineInvalidStageAfterDownsample(_)));

        // SHIFT > 0 after DownsampleStage is rejected.
        let err = mk()
            .add_downsample_stage(DownsampleStage::new(0, 1))
            .add_inout_stage(HorizontalChromaUpsample::new(0))
            .add_save_stage(
                &[0],
                Orientation::Identity,
                0,
                JxlColorType::Grayscale,
                JxlDataFormat::f32(),
                false,
            )
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, Error::PipelineInvalidStageAfterDownsample(_)));

        // Extend after DownsampleStage is rejected.
        let extend = ExtendToImageDimensionsStage {
            frame_origin: (0, 0),
            image_size: (128, 128),
            blending_info: default_blending(),
            ec_blending_info: vec![],
            reference_frames: Arc::new([None, None, None, None]),
            zeros: vec![0.0; 128],
        };
        let err = mk()
            .add_downsample_stage(DownsampleStage::new(0, 1))
            .add_extend_stage(extend)
            .add_save_stage(
                &[0],
                Orientation::Identity,
                0,
                JxlColorType::Grayscale,
                JxlDataFormat::f32(),
                false,
            )
            .build()
            .err()
            .unwrap();
        assert!(matches!(
            err,
            Error::PipelineExtendRequiresFullResolution(0, (1, 1))
        ));

        // Mismatched channel downsamples at SaveStage is rejected.
        let err = mk()
            .add_downsample_stage(DownsampleStage::new(0, 1))
            .add_downsample_stage(DownsampleStage::new(1, 2))
            .add_save_stage(
                &[0, 1, 2],
                Orientation::Identity,
                0,
                JxlColorType::Rgb,
                JxlDataFormat::f32(),
                false,
            )
            .build()
            .err()
            .unwrap();
        assert!(matches!(
            err,
            Error::SaveDifferentDownsample((1, 1), (2, 2))
        ));
    }
}
