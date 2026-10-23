// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg_attr(not(test), allow(dead_code))]

use jxl_simd::{F32SimdVec, SimdDescriptor, simd_function};

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
}
