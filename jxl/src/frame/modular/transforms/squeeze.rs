// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{
    F32SimdVec, I16SimdVec, I16VecArray, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16,
    U32SimdVec, shl, shr, simd_function, simd_specialize,
};
use jxl_transforms::transpose::transpose_square_f32;

use crate::error::{Error, Result};
use crate::frame::modular::{ChannelInfo, ModularChannel, ModularStorage};
use crate::headers::modular::SqueezeParams;
use crate::image::{ImageRect, ImageRectMut, RawImageRect};
use crate::util::tracing_wrappers::*;

#[instrument(level = "trace", err)]
pub fn check_squeeze_params(
    channels: &[(usize, ChannelInfo)],
    params: &SqueezeParams,
) -> Result<()> {
    let end_channel = (params.begin_channel + params.num_channels) as usize;
    if end_channel > channels.len() {
        return Err(Error::InvalidChannelRange(
            params.begin_channel as usize,
            params.num_channels as usize,
            channels.len(),
        ));
    }
    if channels[params.begin_channel as usize].1.is_meta() != channels[end_channel - 1].1.is_meta()
    {
        return Err(Error::MixingDifferentChannels);
    }
    if channels[params.begin_channel as usize].1.is_meta() && !params.in_place {
        return Err(Error::MetaSqueezeRequiresInPlace);
    }
    for c in &channels[params.begin_channel as usize..end_channel] {
        if c.1.size.0 == 0 || c.1.size.1 == 0 {
            return Err(Error::TooManySqueezes);
        }
    }
    Ok(())
}

pub fn default_squeeze(data_channel_info: &[(usize, ChannelInfo)]) -> Vec<SqueezeParams> {
    let num_meta_channels = data_channel_info
        .iter()
        .take_while(|x| x.1.is_meta())
        .count();

    let mut w = data_channel_info[num_meta_channels].1.size.0;
    let mut h = data_channel_info[num_meta_channels].1.size.1;
    let nc = data_channel_info.len() - num_meta_channels;

    let mut params = vec![];

    if nc > 2 && data_channel_info[num_meta_channels + 1].1.size == (w, h) {
        // 420 previews
        let sp = SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_channel: num_meta_channels as u32 + 1,
            num_channels: 2,
        };
        if w > 1 {
            params.push(sp);
        }
        if h > 1 {
            params.push(SqueezeParams {
                horizontal: false,
                ..sp
            });
        }
    }

    const MAX_FIRST_PREVIEW_SIZE: usize = 8;

    let sp = SqueezeParams {
        begin_channel: num_meta_channels as u32,
        num_channels: nc as u32,
        in_place: true,
        horizontal: false,
    };

    // vertical first on tall images
    if w <= h && h > MAX_FIRST_PREVIEW_SIZE {
        params.push(SqueezeParams {
            horizontal: false,
            ..sp
        });
        h = h.div_ceil(2);
    }
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: true,
                ..sp
            });
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: false,
                ..sp
            });
            h = h.div_ceil(2);
        }
    }

    params
}

#[inline(always)]
fn smooth_tendency_impl<D: SimdDescriptor>(
    d: D,
    a: D::I32Vec,
    b: D::I32Vec,
    c: D::I32Vec,
) -> D::I32Vec {
    let a_b = a - b;
    let b_c = b - c;
    let a_c = a - c;
    let abs_a_b = a_b.abs();
    let abs_b_c = b_c.abs();
    let abs_a_c = a_c.abs();
    let non_monotonic = (a_b ^ b_c).lt_zero();
    let skip = a_b.eq_zero().andnot(non_monotonic);
    let skip = b_c.eq_zero().andnot(skip);

    let abs_a_b_3 = abs_a_b.mul_wide_take_high(D::I32Vec::splat(d, 0x55555556));

    let x = shr!(D::I32Vec::splat(d, 2) + abs_a_c + abs_a_b_3, 2);

    let abs_a_b_2_add_x = shl!(abs_a_b, 1) + (x & D::I32Vec::splat(d, 1));
    let x = x
        .gt(abs_a_b_2_add_x)
        .if_then_else_i32(shl!(abs_a_b, 1) + D::I32Vec::splat(d, 1), x);

    let abs_b_c_2 = shl!(abs_b_c, 1);
    let x = (x + (x & D::I32Vec::splat(d, 1)))
        .gt(abs_b_c_2)
        .if_then_else_i32(abs_b_c_2, x);

    let need_neg = a_c.lt_zero();
    let x = skip.maskz_i32(x);
    need_neg.if_then_else_i32(-x, x)
}

#[inline(always)]
fn smooth_tendency_impl_i16<D: SimdDescriptor>(
    d: D,
    a: D::I16Vec,
    b: D::I16Vec,
    c: D::I16Vec,
) -> D::I16Vec {
    let a_b = a - b;
    let b_c = b - c;
    let a_c = a - c;
    let abs_a_b = a_b.abs();
    let abs_b_c = b_c.abs();
    let abs_a_c = a_c.abs();
    let non_monotonic = (a_b ^ b_c).lt_zero();
    let skip = a_b.eq_zero().andnot(non_monotonic);
    let skip = b_c.eq_zero().andnot(skip);

    let abs_a_b_3 = abs_a_b.mul_wide_take_high(D::I16Vec::splat(d, 0x5556));

    let x = shr!(abs_a_c, 2)
        + shr!(
            D::I16Vec::splat(d, 2) + abs_a_b_3 + (abs_a_c & D::I16Vec::splat(d, 3)),
            2
        );

    let limit = D::I16Vec::splat(d, 16383);
    let safe_abs_a_b = abs_a_b.gt(limit).if_then_else_i16(limit, abs_a_b);
    let abs_a_b_2_add_x = shl!(safe_abs_a_b, 1) + (x & D::I16Vec::splat(d, 1));
    let x = x
        .gt(abs_a_b_2_add_x)
        .if_then_else_i16(shl!(safe_abs_a_b, 1) + D::I16Vec::splat(d, 1), x);

    let safe_abs_b_c = abs_b_c.gt(limit).if_then_else_i16(limit, abs_b_c);
    let abs_b_c_2 = shl!(safe_abs_b_c, 1);
    let x = (x + (x & D::I16Vec::splat(d, 1)))
        .gt(abs_b_c_2)
        .if_then_else_i16(abs_b_c_2, x);

    let need_neg = a_c.lt_zero();
    let x = skip.maskz_i16(x);
    need_neg.if_then_else_i16(-x, x)
}

#[inline(always)]
fn smooth_tendency_scalar(b: i64, a: i64, n: i64) -> i64 {
    let mut diff = 0;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        //      2c = a<<1 + diff - diff&1 <= 2b  so diff - diff&1 <= 2b - 2a
        //      2d = a<<1 - diff - diff&1 >= 2n  so diff + diff&1 <= 2a - 2n
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        //      2c = a<<1 + diff + diff&1 >= 2b  so diff + diff&1 >= 2b - 2a
        //      2d = a<<1 - diff + diff&1 <= 2n  so diff - diff&1 >= 2a - 2n
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

#[inline(always)]
fn unsqueeze_impl<D: SimdDescriptor>(
    d: D,
    avg: D::I32Vec,
    res: D::I32Vec,
    next_avg: D::I32Vec,
    prev: D::I32Vec,
) -> (D::I32Vec, D::I32Vec) {
    let tendency = smooth_tendency_impl(d, prev, avg, next_avg);
    let diff = res + tendency;
    let sign = shr!(diff.bitcast_to_u32(), 31).bitcast_to_i32();
    let diff_2 = shr!(diff + sign, 1);
    let a = avg + diff_2;
    let b = a - diff;
    (a, b)
}

#[inline(always)]
fn unsqueeze_impl_i16<D: SimdDescriptor>(
    d: D,
    avg: D::I16Vec,
    res: D::I16Vec,
    next_avg: D::I16Vec,
    prev: D::I16Vec,
) -> (D::I16Vec, D::I16Vec) {
    let tendency = smooth_tendency_impl_i16(d, prev, avg, next_avg);
    let diff = res + tendency;
    let sign = shr!(diff, 15);
    let diff_2 = shr!(diff - sign, 1);
    let a = avg + diff_2;
    let b = a - diff;
    (a, b)
}

#[inline(always)]
fn unsqueeze_scalar(avg: i32, res: i32, next_avg: i32, prev: i32) -> (i32, i32) {
    let tendency = smooth_tendency_scalar(prev as i64, avg as i64, next_avg as i64);
    let diff = (res as i64) + tendency;
    let a = (avg as i64) + (diff / 2);
    let b = a - diff;
    (a as i32, b as i32)
}

#[inline(always)]
fn hsqueeze_impl<D: SimdDescriptor>(
    d: D,
    y_start: usize,
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    out: &mut ImageRectMut<'_, i32>,
) {
    const {
        assert!(D::I32Vec::LEN <= 16);
        assert!(D::I32Vec::LEN.is_power_of_two());
    }

    let lanes = D::I32Vec::LEN;
    assert_eq!(y_start % lanes, 0);

    let (w, h) = in_res.size();
    if lanes == 1 {
        return hsqueeze_scalar(y_start, in_avg, in_res, in_next_avg, out_prev, out);
    }

    let has_tail = out.size().0 & 1 == 1;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.size().0 == 2 * w + 1);
    }

    let mask = !(lanes - 1);
    let y_limit = if w >= lanes { h & mask } else { y_start };

    let mut buf = [0f32; 512];
    for y in (y_start..y_limit).step_by(lanes) {
        for dy in 0..lanes {
            buf[dy] = f32::from_bits(in_avg.row(y + dy)[0] as u32);
            buf[lanes + dy] = f32::from_bits(in_res.row(y + dy)[0] as u32);
        }
        let mut avg_first = D::F32Vec::load(d, &buf).bitcast_to_i32();
        let mut res_first = D::F32Vec::load(d, &buf[lanes..]).bitcast_to_i32();

        let mut prev_b = match out_prev {
            None => avg_first,
            Some(lr) => {
                for (dy, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = f32::from_bits(lr.row(y + dy)[3] as u32);
                }
                D::F32Vec::load(d, &buf).bitcast_to_i32()
            }
        };

        let remainder_start = ((w - 1) & mask) + 1;
        let remainder_count = w - remainder_start;
        for x in (1..remainder_start).step_by(lanes) {
            let buf_arr = D::F32Vec::make_array_slice_mut(&mut buf);

            for dy in 0..lanes {
                let avg_row = &in_avg.row(y + dy)[x..][..lanes];
                let res_row = &in_res.row(y + dy)[x..][..lanes];
                let avg = D::I32Vec::load(d, avg_row);
                let res = D::I32Vec::load(d, res_row);
                avg.bitcast_to_f32().store_array(&mut buf_arr[2 * dy]);
                res.bitcast_to_f32().store_array(&mut buf_arr[2 * dy + 1]);
            }
            transpose_square_f32(d, buf_arr, 2);
            transpose_square_f32(d, &mut buf_arr[1..], 2);

            for idx in 0..lanes {
                let avg_next = D::F32Vec::load_array(d, &buf_arr[2 * idx]).bitcast_to_i32();
                let res_next = D::F32Vec::load_array(d, &buf_arr[2 * idx + 1]).bitcast_to_i32();
                let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_next, prev_b);
                a.bitcast_to_f32().store_array(&mut buf_arr[2 * idx]);
                b.bitcast_to_f32().store_array(&mut buf_arr[2 * idx + 1]);
                avg_first = avg_next;
                res_first = res_next;
                prev_b = b;
            }

            transpose_square_f32(d, buf_arr, 1);
            transpose_square_f32(d, &mut buf_arr[lanes..], 1);
            for dy in 0..lanes {
                let out_row = &mut out.row(y + dy)[2 * x - 2..][..2 * lanes];
                for group in 0..2 {
                    let v = D::F32Vec::load_array(d, &buf_arr[dy + group * lanes]).bitcast_to_i32();
                    v.store(&mut out_row[group * lanes..]);
                }
            }
        }

        let x = remainder_start;
        let has_next_in_avg = in_avg.size().0 > w;
        if remainder_count == 0 {
            let avg_last = if has_tail || has_next_in_avg {
                for (idx, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = f32::from_bits(in_avg.row(y + idx)[w] as u32);
                }
                D::F32Vec::load(d, &buf).bitcast_to_i32()
            } else if let Some(lr) = in_next_avg {
                for (idx, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = f32::from_bits(lr.row(y + idx)[0] as u32);
                }
                D::F32Vec::load(d, &buf).bitcast_to_i32()
            } else {
                avg_first
            };

            let buf_arr = D::F32Vec::make_array_slice_mut(&mut buf);
            let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_last, prev_b);
            a.bitcast_to_f32().store_array(&mut buf_arr[0]);
            b.bitcast_to_f32().store_array(&mut buf_arr[1]);
        } else {
            for dy in 0..lanes {
                let avg_row = in_avg.row(y + dy);
                let res_row = in_res.row(y + dy);
                for dx in 0..remainder_count {
                    buf[dx + lanes * 2 * dy] = f32::from_bits(avg_row[x + dx] as u32);
                    buf[dx + lanes * (2 * dy + 1)] = f32::from_bits(res_row[x + dx] as u32);
                }

                buf[remainder_count + lanes * 2 * dy] = if has_tail || has_next_in_avg {
                    f32::from_bits(avg_row[w] as u32)
                } else if let Some(lr) = in_next_avg {
                    f32::from_bits(lr.row(y + dy)[0] as u32)
                } else {
                    buf[remainder_count - 1 + lanes * 2 * dy]
                };
            }

            let buf_arr = D::F32Vec::make_array_slice_mut(&mut buf);
            transpose_square_f32(d, buf_arr, 2);
            transpose_square_f32(d, &mut buf_arr[1..], 2);

            for idx in 0..=remainder_count {
                let avg_next = D::F32Vec::load_array(d, &buf_arr[2 * idx]).bitcast_to_i32();
                let res_next = D::F32Vec::load_array(d, &buf_arr[2 * idx + 1]).bitcast_to_i32();
                let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_next, prev_b);
                a.bitcast_to_f32().store_array(&mut buf_arr[2 * idx]);
                b.bitcast_to_f32().store_array(&mut buf_arr[2 * idx + 1]);
                avg_first = avg_next;
                res_first = res_next;
                prev_b = b;
            }
        }

        let buf_arr = D::F32Vec::make_array_slice_mut(&mut buf);
        transpose_square_f32(d, buf_arr, 1);
        transpose_square_f32(d, &mut buf_arr[lanes..], 1);

        let x_limit = 2 * (remainder_count + 1);
        for dy in 0..lanes {
            let out_row = &mut out.row(y + dy)[2 * x - 2..];
            for (dx, out) in out_row[..x_limit].iter_mut().enumerate() {
                let group = dx / lanes;
                let group_x = dx % lanes;
                *out = buf[(dy + group * lanes) * lanes + group_x].to_bits() as i32;
            }
        }

        if has_tail {
            for dy in 0..lanes {
                out.row(y + dy)[2 * w] = in_avg.row(y + dy)[w];
            }
        }
    }

    let remainder_rows = h - y_limit;
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > 8 && remainder_rows >= 8 && w >= 8 {
        return hsqueeze_fwd(
            d.maybe_downgrade_256bit(),
            y_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }
    if lanes > 4 && remainder_rows >= 4 && w >= 4 {
        return hsqueeze_fwd(
            d.maybe_downgrade_128bit(),
            y_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }

    hsqueeze_scalar(y_limit, in_avg, in_res, in_next_avg, out_prev, out)
}

fn hsqueeze_scalar(
    y_start: usize,
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    out: &mut ImageRectMut<'_, i32>,
) {
    let (w, h) = in_res.size();

    debug_assert!(w >= 1);
    let has_tail = out.size().0 & 1 == 1;
    let has_next_in_avg = in_avg.size().0 > w;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.size().0 == 2 * w + 1);
    }

    for y in y_start..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let mut prev_b = match out_prev {
            None => avg_row[0],
            Some(lr) => lr.row(y)[3],
        };
        // Guarantee that `avg_row[x + 1]` is available.
        let x_end = if has_tail { w } else { w - 1 };
        for x in 0..x_end {
            let (a, b) = unsqueeze_scalar(avg_row[x], res_row[x], avg_row[x + 1], prev_b);
            out.row(y)[2 * x] = a;
            out.row(y)[2 * x + 1] = b;
            prev_b = b;
        }
        if !has_tail {
            let last_avg = if has_next_in_avg {
                avg_row[w]
            } else if let Some(lr) = in_next_avg {
                lr.row(y)[0]
            } else {
                avg_row[w - 1]
            };
            let (a, b) = unsqueeze_scalar(avg_row[w - 1], res_row[w - 1], last_avg, prev_b);
            out.row(y)[2 * w - 2] = a;
            out.row(y)[2 * w - 1] = b;
        } else {
            // 1 last pixel
            out.row(y)[2 * w] = in_avg.row(y)[w];
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hsqueeze_impl_i16<D: SimdDescriptor>(
    d: D,
    y_start: usize,
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    out: &mut ImageRectMut<'_, i16>,
    buf: &mut [i16; 2048],
) {
    const {
        assert!(D::I16Vec::LEN <= 32);
        assert!(D::I16Vec::LEN.is_power_of_two());
    }

    let lanes = D::I16Vec::LEN;
    assert_eq!(y_start % lanes, 0);

    let (w, h) = in_res.size();
    if lanes == 1 {
        return hsqueeze_scalar_i16(y_start, in_avg, in_res, in_next_avg, out_prev, out);
    }

    let has_tail = out.size().0 & 1 == 1;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.size().0 == 2 * w + 1);
    }

    let mask = !(lanes - 1);
    let y_limit = if w >= lanes { h & mask } else { y_start };

    for y in (y_start..y_limit).step_by(lanes) {
        for dy in 0..lanes {
            buf[dy] = in_avg.row(y + dy)[0];
            buf[lanes + dy] = in_res.row(y + dy)[0];
        }
        let mut avg_first = D::I16Vec::load(d, buf);
        let mut res_first = D::I16Vec::load(d, &buf[lanes..]);

        let mut prev_b = match out_prev {
            None => avg_first,
            Some(lr) => {
                for (dy, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = lr.row(y + dy)[3];
                }
                D::I16Vec::load(d, buf)
            }
        };

        let remainder_start = ((w - 1) & mask) + 1;
        let remainder_count = w - remainder_start;
        for x in (1..remainder_start).step_by(lanes) {
            let buf_arr = D::I16Vec::make_array_slice_mut(&mut buf[..2 * lanes * lanes]);

            for dy in 0..lanes {
                let avg_row = &in_avg.row(y + dy)[x..][..lanes];
                let res_row = &in_res.row(y + dy)[x..][..lanes];
                let avg = D::I16Vec::load(d, avg_row);
                let res = D::I16Vec::load(d, res_row);
                avg.store_array(&mut buf_arr[2 * dy]);
                res.store_array(&mut buf_arr[2 * dy + 1]);
            }
            transpose_square_i16(d, buf_arr, 2);
            transpose_square_i16(d, &mut buf_arr[1..], 2);

            for idx in 0..lanes {
                let avg_next = D::I16Vec::load_array(d, &buf_arr[2 * idx]);
                let res_next = D::I16Vec::load_array(d, &buf_arr[2 * idx + 1]);
                let (a, b) = unsqueeze_impl_i16(d, avg_first, res_first, avg_next, prev_b);
                a.store_array(&mut buf_arr[2 * idx]);
                b.store_array(&mut buf_arr[2 * idx + 1]);
                avg_first = avg_next;
                res_first = res_next;
                prev_b = b;
            }

            transpose_square_i16(d, buf_arr, 1);
            transpose_square_i16(d, &mut buf_arr[lanes..], 1);
            for dy in 0..lanes {
                let out_row = &mut out.row(y + dy)[2 * x - 2..][..2 * lanes];
                for group in 0..2 {
                    let v = D::I16Vec::load_array(d, &buf_arr[dy + group * lanes]);
                    v.store(&mut out_row[group * lanes..]);
                }
            }
        }

        let x = remainder_start;
        let has_next_in_avg = in_avg.size().0 > w;
        if remainder_count == 0 {
            let avg_last = if has_tail || has_next_in_avg {
                for (idx, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = in_avg.row(y + idx)[w];
                }
                D::I16Vec::load(d, buf)
            } else if let Some(lr) = in_next_avg {
                for (idx, out) in buf[..lanes].iter_mut().enumerate() {
                    *out = lr.row(y + idx)[0];
                }
                D::I16Vec::load(d, buf)
            } else {
                avg_first
            };

            let buf_arr = D::I16Vec::make_array_slice_mut(&mut buf[..2 * lanes * lanes]);
            let (a, b) = unsqueeze_impl_i16(d, avg_first, res_first, avg_last, prev_b);
            a.store_array(&mut buf_arr[0]);
            b.store_array(&mut buf_arr[1]);
        } else {
            for dy in 0..lanes {
                let avg_row = in_avg.row(y + dy);
                let res_row = in_res.row(y + dy);
                for dx in 0..remainder_count {
                    buf[dx + lanes * 2 * dy] = avg_row[x + dx];
                    buf[dx + lanes * (2 * dy + 1)] = res_row[x + dx];
                }

                buf[remainder_count + lanes * 2 * dy] = if has_tail || has_next_in_avg {
                    avg_row[w]
                } else if let Some(lr) = in_next_avg {
                    lr.row(y + dy)[0]
                } else {
                    buf[remainder_count - 1 + lanes * 2 * dy]
                };
            }

            let buf_arr = D::I16Vec::make_array_slice_mut(&mut buf[..2 * lanes * lanes]);
            transpose_square_i16(d, buf_arr, 2);
            transpose_square_i16(d, &mut buf_arr[1..], 2);

            for idx in 0..=remainder_count {
                let avg_next = D::I16Vec::load_array(d, &buf_arr[2 * idx]);
                let res_next = D::I16Vec::load_array(d, &buf_arr[2 * idx + 1]);
                let (a, b) = unsqueeze_impl_i16(d, avg_first, res_first, avg_next, prev_b);
                a.store_array(&mut buf_arr[2 * idx]);
                b.store_array(&mut buf_arr[2 * idx + 1]);
                avg_first = avg_next;
                res_first = res_next;
                prev_b = b;
            }
        }

        let buf_arr = D::I16Vec::make_array_slice_mut(&mut buf[..2 * lanes * lanes]);
        transpose_square_i16(d, buf_arr, 1);
        transpose_square_i16(d, &mut buf_arr[lanes..], 1);

        let x_limit = 2 * (remainder_count + 1);
        for dy in 0..lanes {
            let out_row = &mut out.row(y + dy)[2 * x - 2..];
            for (dx, out) in out_row[..x_limit].iter_mut().enumerate() {
                let group = dx / lanes;
                let group_x = dx % lanes;
                *out = buf[(dy + group * lanes) * lanes + group_x];
            }
        }

        if has_tail {
            for dy in 0..lanes {
                out.row(y + dy)[2 * w] = in_avg.row(y + dy)[w];
            }
        }
    }

    let remainder_rows = h - y_limit;
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > 16 && remainder_rows >= 16 && w >= 16 {
        return hsqueeze_fwd_i16(
            d.maybe_downgrade_256bit(),
            y_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
            buf,
        );
    }
    if lanes > 8 && remainder_rows >= 8 && w >= 8 {
        return hsqueeze_fwd_i16(
            d.maybe_downgrade_128bit(),
            y_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
            buf,
        );
    }

    hsqueeze_scalar_i16(y_limit, in_avg, in_res, in_next_avg, out_prev, out)
}

fn hsqueeze_scalar_i16(
    y_start: usize,
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    out: &mut ImageRectMut<'_, i16>,
) {
    let (w, h) = in_res.size();

    debug_assert!(w >= 1);
    let has_tail = out.size().0 & 1 == 1;
    let has_next_in_avg = in_avg.size().0 > w;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.size().0 == 2 * w + 1);
    }

    for y in y_start..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let mut prev_b = match out_prev {
            None => avg_row[0],
            Some(lr) => lr.row(y)[3],
        };
        // Guarantee that `avg_row[x + 1]` is available.
        let x_end = if has_tail { w } else { w - 1 };
        for x in 0..x_end {
            let (a, b) = unsqueeze_scalar(
                avg_row[x] as i32,
                res_row[x] as i32,
                avg_row[x + 1] as i32,
                prev_b as i32,
            );
            out.row(y)[2 * x] = a as i16;
            out.row(y)[2 * x + 1] = b as i16;
            prev_b = b as i16;
        }
        if !has_tail {
            let last_avg = if has_next_in_avg {
                avg_row[w]
            } else if let Some(lr) = in_next_avg {
                lr.row(y)[0]
            } else {
                avg_row[w - 1]
            };
            let (a, b) = unsqueeze_scalar(
                avg_row[w - 1] as i32,
                res_row[w - 1] as i32,
                last_avg as i32,
                prev_b as i32,
            );
            out.row(y)[2 * w - 2] = a as i16;
            out.row(y)[2 * w - 1] = b as i16;
        } else {
            // 1 last pixel
            out.row(y)[2 * w] = in_avg.row(y)[w];
        }
    }
}

simd_function!(
    hsqueeze,
    d: D,
    pub fn hsqueeze_fwd(
        y_start: usize,
        in_avg: &ImageRect<'_, i32>,
        in_res: &ImageRect<'_, i32>,
        in_next_avg: Option<&ImageRect<'_, i32>>,
        out_prev: Option<&ImageRect<'_, i32>>,
        out: &mut ImageRectMut<'_, i32>,
    ) {
        d.call(
            #[inline(always)]
            |d| hsqueeze_impl(d, y_start, in_avg, in_res, in_next_avg, out_prev, out),
        )
    }
);

simd_function!(
    hsqueeze_i16,
    d: D,
    #[allow(clippy::too_many_arguments)]
    pub fn hsqueeze_fwd_i16(
        y_start: usize,
        in_avg: &ImageRect<'_, i16>,
        in_res: &ImageRect<'_, i16>,
        in_next_avg: Option<&ImageRect<'_, i16>>,
        out_prev: Option<&ImageRect<'_, i16>>,
        out: &mut ImageRectMut<'_, i16>,
        buf: &mut [i16; 2048],
    ) {
        d.call(
            #[inline(always)]
            |d| hsqueeze_impl_i16(d, y_start, in_avg, in_res, in_next_avg, out_prev, out, buf),
        )
    }
);

#[inline(always)]
fn do_hsqueeze_step_i16(
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    buffers: &mut [&mut ModularChannel],
    scratch_i16: &mut [i16; 2048],
) {
    trace!("hsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<i16>::from_raw(out.data.as_rect_mut());
    // Shortcut: guarantees that row is at least 1px in the main loop
    if out_rect.size().0 == 0 || out_rect.size().1 == 0 {
        return;
    }

    let w = in_res.size().0;
    // Another shortcut: when output row has just 1px
    if w == 0 {
        let out_h = out_rect.size().1;
        for y in 0..out_h {
            out_rect.row(y)[0] = in_avg.row(y)[0];
        }
        return;
    }
    // Otherwise: 2 or more in in row
    hsqueeze_i16(
        0,
        in_avg,
        in_res,
        in_next_avg,
        out_prev,
        &mut out_rect,
        scratch_i16,
    );
}

#[inline(always)]
fn do_hsqueeze_step_i32(
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    buffers: &mut [&mut ModularChannel],
) {
    trace!("hsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
    // Shortcut: guarantees that row is at least 1px in the main loop
    if out_rect.size().0 == 0 || out_rect.size().1 == 0 {
        return;
    }

    let w = in_res.size().0;
    // Another shortcut: when output row has just 1px
    if w == 0 {
        let out_h = out_rect.size().1;
        for y in 0..out_h {
            out_rect.row(y)[0] = in_avg.row(y)[0];
        }
        return;
    }
    // Otherwise: 2 or more in in row
    hsqueeze(0, in_avg, in_res, in_next_avg, out_prev, &mut out_rect);
}

#[inline(always)]
pub fn do_hsqueeze_step(
    in_avg: &RawImageRect<'_>,
    in_res: &RawImageRect<'_>,
    in_next_avg: Option<&RawImageRect<'_>>,
    out_prev: Option<&RawImageRect<'_>>,
    buffers: &mut [&mut ModularChannel],
    storage: ModularStorage,
    scratch_i16: &mut [i16; 2048],
) {
    if storage == ModularStorage::I16 {
        do_hsqueeze_step_i16(
            &ImageRect::<i16>::from_raw(*in_avg),
            &ImageRect::<i16>::from_raw(*in_res),
            in_next_avg.map(|r| ImageRect::<i16>::from_raw(*r)).as_ref(),
            out_prev.map(|r| ImageRect::<i16>::from_raw(*r)).as_ref(),
            buffers,
            scratch_i16,
        );
    } else {
        do_hsqueeze_step_i32(
            &ImageRect::<i32>::from_raw(*in_avg),
            &ImageRect::<i32>::from_raw(*in_res),
            in_next_avg.map(|r| ImageRect::<i32>::from_raw(*r)).as_ref(),
            out_prev.map(|r| ImageRect::<i32>::from_raw(*r)).as_ref(),
            buffers,
        );
    }
}

#[inline(always)]
fn vsqueeze_impl<D: SimdDescriptor>(
    d: D,
    x_start: usize,
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    out: &mut ImageRectMut<'_, i32>,
) {
    const { assert!(D::I32Vec::LEN.is_power_of_two()) };

    let lanes = D::I32Vec::LEN;
    assert_eq!(x_start % lanes, 0);

    let (w, h) = in_res.size();
    if lanes == 1 {
        return vsqueeze_scalar(x_start, in_avg, in_res, in_next_avg, out_prev, out);
    }

    let has_tail = out.size().1 & 1 == 1;
    let has_next_in_avg = in_avg.size().1 > h;
    if has_tail {
        debug_assert!(in_avg.size().1 == h + 1);
        debug_assert!(out.size().1 == 2 * h + 1);
    }

    let mask = !(lanes - 1);
    let x_limit = if w >= lanes { w & mask } else { x_start };

    let prev_b_row = match out_prev {
        None => in_avg.row(0),
        Some(tb) => tb.row(3),
    };

    for x in (x_start..x_limit).step_by(lanes) {
        let mut prev_b = D::I32Vec::load(d, &prev_b_row[x..]);
        let mut avg_first = D::I32Vec::load(d, &in_avg.row(0)[x..]);
        let mut res_first = D::I32Vec::load(d, &in_res.row(0)[x..]);
        for y in 0..h - 1 {
            let avg_next = D::I32Vec::load(d, &in_avg.row(y + 1)[x..]);
            let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_next, prev_b);
            a.store(&mut out.row(2 * y)[x..]);
            b.store(&mut out.row(2 * y + 1)[x..]);
            prev_b = b;
            avg_first = avg_next;
            res_first = D::I32Vec::load(d, &in_res.row(y + 1)[x..]);
        }

        let avg_last = if has_tail || has_next_in_avg {
            D::I32Vec::load(d, &in_avg.row(h)[x..])
        } else if let Some(tb) = in_next_avg {
            D::I32Vec::load(d, &tb.row(0)[x..])
        } else {
            avg_first
        };
        let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_last, prev_b);
        a.store(&mut out.row(2 * h - 2)[x..]);
        b.store(&mut out.row(2 * h - 1)[x..]);

        if has_tail {
            avg_last.store(&mut out.row(2 * h)[x..]);
        }
    }

    let remainder_cols = w - x_limit;
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > 8 && remainder_cols >= 8 {
        return vsqueeze_fwd(
            d.maybe_downgrade_256bit(),
            x_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }
    if lanes > 4 && remainder_cols >= 4 {
        return vsqueeze_fwd(
            d.maybe_downgrade_128bit(),
            x_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }

    vsqueeze_scalar(x_limit, in_avg, in_res, in_next_avg, out_prev, out)
}

#[inline(always)]
fn vsqueeze_impl_i16<D: SimdDescriptor>(
    d: D,
    x_start: usize,
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    out: &mut ImageRectMut<'_, i16>,
) {
    const { assert!(D::I16Vec::LEN.is_power_of_two()) };

    let lanes = D::I16Vec::LEN;
    assert_eq!(x_start % lanes, 0);

    let (w, h) = in_res.size();
    if lanes == 1 {
        return vsqueeze_scalar_i16(x_start, in_avg, in_res, in_next_avg, out_prev, out);
    }

    let has_tail = out.size().1 & 1 == 1;
    let has_next_in_avg = in_avg.size().1 > h;
    if has_tail {
        debug_assert!(in_avg.size().1 == h + 1);
        debug_assert!(out.size().1 == 2 * h + 1);
    }

    let mask = !(lanes - 1);
    let x_limit = if w >= lanes { w & mask } else { x_start };

    let prev_b_row = match out_prev {
        None => in_avg.row(0),
        Some(tb) => tb.row(3),
    };

    for x in (x_start..x_limit).step_by(lanes) {
        let mut prev_b = D::I16Vec::load(d, &prev_b_row[x..]);
        let mut avg_first = D::I16Vec::load(d, &in_avg.row(0)[x..]);
        let mut res_first = D::I16Vec::load(d, &in_res.row(0)[x..]);
        for y in 0..h - 1 {
            let avg_next = D::I16Vec::load(d, &in_avg.row(y + 1)[x..]);
            let (a, b) = unsqueeze_impl_i16(d, avg_first, res_first, avg_next, prev_b);
            a.store(&mut out.row(2 * y)[x..]);
            b.store(&mut out.row(2 * y + 1)[x..]);
            prev_b = b;
            avg_first = avg_next;
            res_first = D::I16Vec::load(d, &in_res.row(y + 1)[x..]);
        }

        let avg_last = if has_tail || has_next_in_avg {
            D::I16Vec::load(d, &in_avg.row(h)[x..])
        } else if let Some(tb) = in_next_avg {
            D::I16Vec::load(d, &tb.row(0)[x..])
        } else {
            avg_first
        };
        let (a, b) = unsqueeze_impl_i16(d, avg_first, res_first, avg_last, prev_b);
        a.store(&mut out.row(2 * h - 2)[x..]);
        b.store(&mut out.row(2 * h - 1)[x..]);

        if has_tail {
            avg_last.store(&mut out.row(2 * h)[x..]);
        }
    }

    let remainder_cols = w - x_limit;
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > 16 && remainder_cols >= 16 {
        return vsqueeze_fwd_i16(
            d.maybe_downgrade_256bit(),
            x_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }
    if lanes > 8 && remainder_cols >= 8 {
        return vsqueeze_fwd_i16(
            d.maybe_downgrade_128bit(),
            x_limit,
            in_avg,
            in_res,
            in_next_avg,
            out_prev,
            out,
        );
    }

    vsqueeze_scalar_i16(x_limit, in_avg, in_res, in_next_avg, out_prev, out)
}

fn vsqueeze_scalar(
    x_start: usize,
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    out: &mut ImageRectMut<'_, i32>,
) {
    let (w, h) = in_res.size();

    let has_tail = out.size().1 & 1 == 1;
    let has_next_in_avg = in_avg.size().1 > h;
    if has_tail {
        debug_assert!(in_avg.size().1 == h + 1);
        debug_assert!(out.size().1 == 2 * h + 1);
    }

    {
        let prev_b_row = match out_prev {
            None => in_avg.row(0),
            Some(tb) => tb.row(3),
        };
        let avg_row = in_avg.row(0);
        let res_row = in_res.row(0);
        let avg_row_next = if !has_tail && (h == 1) {
            if has_next_in_avg {
                in_avg.row(1)
            } else if let Some(tb) = in_next_avg {
                tb.row(0)
            } else {
                in_avg.row(0)
            }
        } else {
            in_avg.row(1)
        };
        let [out_row0, out_row1] = out.distinct_rows_mut([0, 1]);
        for x in x_start..w {
            let (a, b) = unsqueeze_scalar(avg_row[x], res_row[x], avg_row_next[x], prev_b_row[x]);
            out_row0[x] = a;
            out_row1[x] = b;
        }
    }
    for y in 1..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let avg_row_next = if has_tail || y < h - 1 {
            in_avg.row(y + 1)
        } else if has_next_in_avg {
            in_avg.row(h)
        } else if let Some(tb) = in_next_avg {
            tb.row(0)
        } else {
            avg_row
        };
        let [prev_b_row, out_row0, out_row1] = out.distinct_rows_mut([2 * y - 1, 2 * y, 2 * y + 1]);
        for x in x_start..w {
            let (a, b) = unsqueeze_scalar(avg_row[x], res_row[x], avg_row_next[x], prev_b_row[x]);
            out_row0[x] = a;
            out_row1[x] = b;
        }
    }
    if has_tail {
        out.row(2 * h)[x_start..].copy_from_slice(&in_avg.row(h)[x_start..]);
    }
}

fn vsqueeze_scalar_i16(
    x_start: usize,
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    out: &mut ImageRectMut<'_, i16>,
) {
    let (w, h) = in_res.size();

    let has_tail = out.size().1 & 1 == 1;
    let has_next_in_avg = in_avg.size().1 > h;
    if has_tail {
        debug_assert!(in_avg.size().1 == h + 1);
        debug_assert!(out.size().1 == 2 * h + 1);
    }

    {
        let prev_b_row = match out_prev {
            None => in_avg.row(0),
            Some(tb) => tb.row(3),
        };
        let avg_row = in_avg.row(0);
        let res_row = in_res.row(0);
        let avg_row_next = if !has_tail && (h == 1) {
            if has_next_in_avg {
                in_avg.row(1)
            } else if let Some(tb) = in_next_avg {
                tb.row(0)
            } else {
                in_avg.row(0)
            }
        } else {
            in_avg.row(1)
        };
        let [out_row0, out_row1] = out.distinct_rows_mut([0, 1]);
        for x in x_start..w {
            let (a, b) = unsqueeze_scalar(
                avg_row[x] as i32,
                res_row[x] as i32,
                avg_row_next[x] as i32,
                prev_b_row[x] as i32,
            );
            out_row0[x] = a as i16;
            out_row1[x] = b as i16;
        }
    }
    for y in 1..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let avg_row_next = if has_tail || y < h - 1 {
            in_avg.row(y + 1)
        } else if has_next_in_avg {
            in_avg.row(h)
        } else if let Some(tb) = in_next_avg {
            tb.row(0)
        } else {
            avg_row
        };
        let [prev_b_row, out_row0, out_row1] = out.distinct_rows_mut([2 * y - 1, 2 * y, 2 * y + 1]);
        for x in x_start..w {
            let (a, b) = unsqueeze_scalar(
                avg_row[x] as i32,
                res_row[x] as i32,
                avg_row_next[x] as i32,
                prev_b_row[x] as i32,
            );
            out_row0[x] = a as i16;
            out_row1[x] = b as i16;
        }
    }
    if has_tail {
        out.row(2 * h)[x_start..].copy_from_slice(&in_avg.row(h)[x_start..]);
    }
}

simd_function!(
    vsqueeze,
    d: D,
    pub fn vsqueeze_fwd(
        x_start: usize,
        in_avg: &ImageRect<'_, i32>,
        in_res: &ImageRect<'_, i32>,
        in_next_avg: Option<&ImageRect<'_, i32>>,
        out_prev: Option<&ImageRect<'_, i32>>,
        out: &mut ImageRectMut<'_, i32>,
    ) {
        d.call(
            #[inline(always)]
            |d| vsqueeze_impl(d, x_start, in_avg, in_res, in_next_avg, out_prev, out),
        )
    }
);

simd_function!(
    vsqueeze_i16,
    d: D,
    pub fn vsqueeze_fwd_i16(
        x_start: usize,
        in_avg: &ImageRect<'_, i16>,
        in_res: &ImageRect<'_, i16>,
        in_next_avg: Option<&ImageRect<'_, i16>>,
        out_prev: Option<&ImageRect<'_, i16>>,
        out: &mut ImageRectMut<'_, i16>,
    ) {
        d.call(
            #[inline(always)]
            |d| vsqueeze_impl_i16(d, x_start, in_avg, in_res, in_next_avg, out_prev, out),
        )
    }
);

#[allow(dead_code)]
#[inline(always)]
fn do_vsqueeze_step_i16(
    in_avg: &ImageRect<'_, i16>,
    in_res: &ImageRect<'_, i16>,
    in_next_avg: Option<&ImageRect<'_, i16>>,
    out_prev: Option<&ImageRect<'_, i16>>,
    buffers: &mut [&mut ModularChannel],
) {
    trace!("vsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<i16>::from_raw(out.data.as_rect_mut());
    // Shortcut: guarantees that there at least 1 output row
    if out_rect.size().1 == 0 || out_rect.size().0 == 0 {
        return;
    }
    // Another shortcut: when there is one output row
    if in_res.size().1 == 0 {
        let w = in_avg.size().0;
        out_rect.row(0)[..w].copy_from_slice(in_avg.row(0));
        return;
    }
    // Otherwise: 2 or more rows

    vsqueeze_i16(0, in_avg, in_res, in_next_avg, out_prev, &mut out_rect);
}

#[inline(always)]
fn do_vsqueeze_step_i32(
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: Option<&ImageRect<'_, i32>>,
    out_prev: Option<&ImageRect<'_, i32>>,
    buffers: &mut [&mut ModularChannel],
) {
    trace!("vsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
    // Shortcut: guarantees that there at least 1 output row
    if out_rect.size().1 == 0 || out_rect.size().0 == 0 {
        return;
    }
    // Another shortcut: when there is one output row
    if in_res.size().1 == 0 {
        out_rect.row(0).copy_from_slice(in_avg.row(0));
        return;
    }
    // Otherwise: 2 or more rows

    vsqueeze(0, in_avg, in_res, in_next_avg, out_prev, &mut out_rect);
}

#[inline(always)]
pub fn do_vsqueeze_step(
    in_avg: &RawImageRect<'_>,
    in_res: &RawImageRect<'_>,
    in_next_avg: Option<&RawImageRect<'_>>,
    out_prev: Option<&RawImageRect<'_>>,
    buffers: &mut [&mut ModularChannel],
    storage: ModularStorage,
) {
    if storage == ModularStorage::I16 {
        do_vsqueeze_step_i16(
            &ImageRect::<i16>::from_raw(*in_avg),
            &ImageRect::<i16>::from_raw(*in_res),
            in_next_avg.map(|r| ImageRect::<i16>::from_raw(*r)).as_ref(),
            out_prev.map(|r| ImageRect::<i16>::from_raw(*r)).as_ref(),
            buffers,
        );
    } else {
        do_vsqueeze_step_i32(
            &ImageRect::<i32>::from_raw(*in_avg),
            &ImageRect::<i32>::from_raw(*in_res),
            in_next_avg.map(|r| ImageRect::<i32>::from_raw(*r)).as_ref(),
            out_prev.map(|r| ImageRect::<i32>::from_raw(*r)).as_ref(),
            buffers,
        );
    }
}

#[inline(always)]
fn transpose_i16_pow2<D: SimdDescriptor, const N: usize, const N4: usize>(
    d: D,
    data: &mut [I16VecArray<D>],
    stride: usize,
) {
    assert_eq!(N, D::I16Vec::LEN);
    assert_eq!(N4, 4 * N);
    assert!(data.len() > (N - 1) * stride);

    let mut cur = [D::I16Vec::zero(d); N];
    let mut nxt = [D::I16Vec::zero(d); N];
    let mut tmp = [0i16; N4];
    for i in 0..N {
        cur[i] = D::I16Vec::load_array(d, &data[i * stride]);
    }

    let mut rem = N;
    let q = N / 4;
    while rem >= 4 {
        for i in 0..q {
            D::I16Vec::store_interleaved_4(
                cur[i],
                cur[i + q],
                cur[i + 2 * q],
                cur[i + 3 * q],
                &mut tmp,
            );
            let tmp_arr = D::I16Vec::make_array_slice(&tmp);
            for k in 0..4 {
                nxt[4 * i + k] = D::I16Vec::load_array(d, &tmp_arr[k]);
            }
        }
        cur.copy_from_slice(&nxt);
        rem /= 4;
    }
    if rem == 2 {
        let h = N / 2;
        let tmp_half = &mut tmp[..2 * N];
        for i in 0..h {
            D::I16Vec::store_interleaved_2(cur[i], cur[i + h], tmp_half);
            let tmp_arr = D::I16Vec::make_array_slice(tmp_half);
            nxt[2 * i] = D::I16Vec::load_array(d, &tmp_arr[0]);
            nxt[2 * i + 1] = D::I16Vec::load_array(d, &tmp_arr[1]);
        }
        cur.copy_from_slice(&nxt);
    }

    for i in 0..N {
        cur[i].store_array(&mut data[i * stride]);
    }
}

#[inline(always)]
fn transpose_square_i16_fallback<D: SimdDescriptor>(
    d: D,
    data: &mut [I16VecArray<D>],
    stride: usize,
) {
    match D::I16Vec::LEN {
        0 | 1 => {}
        2 => transpose_i16_pow2::<D, 2, 8>(d, data, stride),
        4 => transpose_i16_pow2::<D, 4, 16>(d, data, stride),
        8 => transpose_i16_pow2::<D, 8, 32>(d, data, stride),
        16 => transpose_i16_pow2::<D, 16, 64>(d, data, stride),
        32 => transpose_i16_pow2::<D, 32, 128>(d, data, stride),
        64 => transpose_i16_pow2::<D, 64, 256>(d, data, stride),
        n => panic!("unsupported I16Vec::LEN = {n}"),
    }
}

#[inline(always)]
fn transpose_square_i16<D: SimdDescriptor>(d: D, data: &mut [I16VecArray<D>], stride: usize) {
    simd_specialize!(
        fn(d: D, data: &mut [I16VecArray<D>], stride: usize) {
            avx => {
                use std::arch::x86_64::*;

                use jxl_simd::I16VecAvx;

                assert!(data.len() > stride * 15);

                #[target_feature(enable = "avx2")]
                #[inline]
                fn transpose_8x8_inlane([r0, r1, r2, r3, r4, r5, r6, r7]: [__m256i; 8]) -> [__m256i; 8] {
                    let t0 = _mm256_unpacklo_epi16(r0, r1);
                    let t1 = _mm256_unpackhi_epi16(r0, r1);
                    let t2 = _mm256_unpacklo_epi16(r2, r3);
                    let t3 = _mm256_unpackhi_epi16(r2, r3);
                    let t4 = _mm256_unpacklo_epi16(r4, r5);
                    let t5 = _mm256_unpackhi_epi16(r4, r5);
                    let t6 = _mm256_unpacklo_epi16(r6, r7);
                    let t7 = _mm256_unpackhi_epi16(r6, r7);

                    let u0 = _mm256_unpacklo_epi32(t0, t2);
                    let u1 = _mm256_unpackhi_epi32(t0, t2);
                    let u2 = _mm256_unpacklo_epi32(t1, t3);
                    let u3 = _mm256_unpackhi_epi32(t1, t3);
                    let u4 = _mm256_unpacklo_epi32(t4, t6);
                    let u5 = _mm256_unpackhi_epi32(t4, t6);
                    let u6 = _mm256_unpacklo_epi32(t5, t7);
                    let u7 = _mm256_unpackhi_epi32(t5, t7);

                    let s0 = _mm256_unpacklo_epi64(u0, u4);
                    let s1 = _mm256_unpackhi_epi64(u0, u4);
                    let s2 = _mm256_unpacklo_epi64(u1, u5);
                    let s3 = _mm256_unpackhi_epi64(u1, u5);
                    let s4 = _mm256_unpacklo_epi64(u2, u6);
                    let s5 = _mm256_unpackhi_epi64(u2, u6);
                    let s6 = _mm256_unpacklo_epi64(u3, u7);
                    let s7 = _mm256_unpackhi_epi64(u3, u7);

                    [s0, s1, s2, s3, s4, s5, s6, s7]
                }

                let top = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx::load_array(d, &data[i * stride]).raw()
                }));
                let bot = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx::load_array(d, &data[(8 + i) * stride]).raw()
                }));

                for k in 0..8 {
                    I16VecAvx::from_raw(d, _mm256_permute2x128_si256::<0x20>(top[k], bot[k]))
                        .store_array(&mut data[k * stride]);
                    I16VecAvx::from_raw(d, _mm256_permute2x128_si256::<0x31>(top[k], bot[k]))
                        .store_array(&mut data[(8 + k) * stride]);
                }
            }
            avx512 => {
                use std::arch::x86_64::*;

                use jxl_simd::I16VecAvx512;

                assert!(data.len() > stride * 31);

                #[target_feature(enable = "avx512f,avx512bw")]
                #[inline]
                fn transpose_8x8_inlane([r0, r1, r2, r3, r4, r5, r6, r7]: [__m512i; 8]) -> [__m512i; 8] {
                    let t0 = _mm512_unpacklo_epi16(r0, r1);
                    let t1 = _mm512_unpackhi_epi16(r0, r1);
                    let t2 = _mm512_unpacklo_epi16(r2, r3);
                    let t3 = _mm512_unpackhi_epi16(r2, r3);
                    let t4 = _mm512_unpacklo_epi16(r4, r5);
                    let t5 = _mm512_unpackhi_epi16(r4, r5);
                    let t6 = _mm512_unpacklo_epi16(r6, r7);
                    let t7 = _mm512_unpackhi_epi16(r6, r7);

                    let u0 = _mm512_unpacklo_epi32(t0, t2);
                    let u1 = _mm512_unpackhi_epi32(t0, t2);
                    let u2 = _mm512_unpacklo_epi32(t1, t3);
                    let u3 = _mm512_unpackhi_epi32(t1, t3);
                    let u4 = _mm512_unpacklo_epi32(t4, t6);
                    let u5 = _mm512_unpackhi_epi32(t4, t6);
                    let u6 = _mm512_unpacklo_epi32(t5, t7);
                    let u7 = _mm512_unpackhi_epi32(t5, t7);

                    let s0 = _mm512_unpacklo_epi64(u0, u4);
                    let s1 = _mm512_unpackhi_epi64(u0, u4);
                    let s2 = _mm512_unpacklo_epi64(u1, u5);
                    let s3 = _mm512_unpackhi_epi64(u1, u5);
                    let s4 = _mm512_unpacklo_epi64(u2, u6);
                    let s5 = _mm512_unpackhi_epi64(u2, u6);
                    let s6 = _mm512_unpacklo_epi64(u3, u7);
                    let s7 = _mm512_unpackhi_epi64(u3, u7);

                    [s0, s1, s2, s3, s4, s5, s6, s7]
                }

                #[target_feature(enable = "avx512f,avx512bw")]
                #[inline]
                fn transpose_4x4_128(
                    v0: __m512i,
                    v1: __m512i,
                    v2: __m512i,
                    v3: __m512i,
                ) -> (__m512i, __m512i, __m512i, __m512i) {
                    let t0 = _mm512_shuffle_i64x2::<0x44>(v0, v1);
                    let t1 = _mm512_shuffle_i64x2::<0xEE>(v0, v1);
                    let t2 = _mm512_shuffle_i64x2::<0x44>(v2, v3);
                    let t3 = _mm512_shuffle_i64x2::<0xEE>(v2, v3);
                    (
                        _mm512_shuffle_i64x2::<0x88>(t0, t2),
                        _mm512_shuffle_i64x2::<0xDD>(t0, t2),
                        _mm512_shuffle_i64x2::<0x88>(t1, t3),
                        _mm512_shuffle_i64x2::<0xDD>(t1, t3),
                    )
                }

                let g0 = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx512::load_array(d, &data[i * stride]).raw()
                }));
                let g1 = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx512::load_array(d, &data[(8 + i) * stride]).raw()
                }));
                let g2 = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx512::load_array(d, &data[(16 + i) * stride]).raw()
                }));
                let g3 = transpose_8x8_inlane(std::array::from_fn(|i| {
                    I16VecAvx512::load_array(d, &data[(24 + i) * stride]).raw()
                }));

                for k in 0..8 {
                    let (out0, out1, out2, out3) = transpose_4x4_128(g0[k], g1[k], g2[k], g3[k]);
                    I16VecAvx512::from_raw(d, out0).store_array(&mut data[k * stride]);
                    I16VecAvx512::from_raw(d, out1).store_array(&mut data[(8 + k) * stride]);
                    I16VecAvx512::from_raw(d, out2).store_array(&mut data[(16 + k) * stride]);
                    I16VecAvx512::from_raw(d, out3).store_array(&mut data[(24 + k) * stride]);
                }
            }
            neon => {
                use std::arch::aarch64::*;

                use jxl_simd::I16VecNeon;

                assert!(data.len() > stride * 7);

                let r0 = I16VecNeon::load_array(d, &data[0]).raw();
                let r1 = I16VecNeon::load_array(d, &data[stride]).raw();
                let r2 = I16VecNeon::load_array(d, &data[2 * stride]).raw();
                let r3 = I16VecNeon::load_array(d, &data[3 * stride]).raw();
                let r4 = I16VecNeon::load_array(d, &data[4 * stride]).raw();
                let r5 = I16VecNeon::load_array(d, &data[5 * stride]).raw();
                let r6 = I16VecNeon::load_array(d, &data[6 * stride]).raw();
                let r7 = I16VecNeon::load_array(d, &data[7 * stride]).raw();

                // Stage 1: 16-bit zip
                let t0 = vzip1q_s16(r0, r1);
                let t1 = vzip2q_s16(r0, r1);
                let t2 = vzip1q_s16(r2, r3);
                let t3 = vzip2q_s16(r2, r3);
                let t4 = vzip1q_s16(r4, r5);
                let t5 = vzip2q_s16(r4, r5);
                let t6 = vzip1q_s16(r6, r7);
                let t7 = vzip2q_s16(r6, r7);

                // Stage 2: 32-bit zip
                let u0 = vreinterpretq_s16_s32(vzip1q_s32(
                    vreinterpretq_s32_s16(t0),
                    vreinterpretq_s32_s16(t2),
                ));
                let u1 = vreinterpretq_s16_s32(vzip2q_s32(
                    vreinterpretq_s32_s16(t0),
                    vreinterpretq_s32_s16(t2),
                ));
                let u2 = vreinterpretq_s16_s32(vzip1q_s32(
                    vreinterpretq_s32_s16(t1),
                    vreinterpretq_s32_s16(t3),
                ));
                let u3 = vreinterpretq_s16_s32(vzip2q_s32(
                    vreinterpretq_s32_s16(t1),
                    vreinterpretq_s32_s16(t3),
                ));
                let u4 = vreinterpretq_s16_s32(vzip1q_s32(
                    vreinterpretq_s32_s16(t4),
                    vreinterpretq_s32_s16(t6),
                ));
                let u5 = vreinterpretq_s16_s32(vzip2q_s32(
                    vreinterpretq_s32_s16(t4),
                    vreinterpretq_s32_s16(t6),
                ));
                let u6 = vreinterpretq_s16_s32(vzip1q_s32(
                    vreinterpretq_s32_s16(t5),
                    vreinterpretq_s32_s16(t7),
                ));
                let u7 = vreinterpretq_s16_s32(vzip2q_s32(
                    vreinterpretq_s32_s16(t5),
                    vreinterpretq_s32_s16(t7),
                ));

                // Stage 3: 64-bit zip
                let out0 = vreinterpretq_s16_s64(vzip1q_s64(
                    vreinterpretq_s64_s16(u0),
                    vreinterpretq_s64_s16(u4),
                ));
                let out1 = vreinterpretq_s16_s64(vzip2q_s64(
                    vreinterpretq_s64_s16(u0),
                    vreinterpretq_s64_s16(u4),
                ));
                let out2 = vreinterpretq_s16_s64(vzip1q_s64(
                    vreinterpretq_s64_s16(u1),
                    vreinterpretq_s64_s16(u5),
                ));
                let out3 = vreinterpretq_s16_s64(vzip2q_s64(
                    vreinterpretq_s64_s16(u1),
                    vreinterpretq_s64_s16(u5),
                ));
                let out4 = vreinterpretq_s16_s64(vzip1q_s64(
                    vreinterpretq_s64_s16(u2),
                    vreinterpretq_s64_s16(u6),
                ));
                let out5 = vreinterpretq_s16_s64(vzip2q_s64(
                    vreinterpretq_s64_s16(u2),
                    vreinterpretq_s64_s16(u6),
                ));
                let out6 = vreinterpretq_s16_s64(vzip1q_s64(
                    vreinterpretq_s64_s16(u3),
                    vreinterpretq_s64_s16(u7),
                ));
                let out7 = vreinterpretq_s16_s64(vzip2q_s64(
                    vreinterpretq_s64_s16(u3),
                    vreinterpretq_s64_s16(u7),
                ));

                I16VecNeon::from_raw(d, out0).store_array(&mut data[0]);
                I16VecNeon::from_raw(d, out1).store_array(&mut data[stride]);
                I16VecNeon::from_raw(d, out2).store_array(&mut data[2 * stride]);
                I16VecNeon::from_raw(d, out3).store_array(&mut data[3 * stride]);
                I16VecNeon::from_raw(d, out4).store_array(&mut data[4 * stride]);
                I16VecNeon::from_raw(d, out5).store_array(&mut data[5 * stride]);
                I16VecNeon::from_raw(d, out6).store_array(&mut data[6 * stride]);
                I16VecNeon::from_raw(d, out7).store_array(&mut data[7 * stride]);
            }
            v128 => {
                assert!(data.len() > 7 * stride);

                let r: [D::I16Vec; 8] =
                    std::array::from_fn(|i| D::I16Vec::load_array(d, &data[i * stride]));
                let mut mid = [D::I16Vec::zero(d); 8];
                let mut tmp = [[0i16; 8]; 4];

                D::I16Vec::store_interleaved_4(
                    r[0],
                    r[2],
                    r[4],
                    r[6],
                    tmp.as_flattened_mut(),
                );
                for k in 0..4 {
                    mid[k] = D::I16Vec::load_array(d, &tmp[k]);
                }

                D::I16Vec::store_interleaved_4(
                    r[1],
                    r[3],
                    r[5],
                    r[7],
                    tmp.as_flattened_mut(),
                );
                for k in 0..4 {
                    mid[4 + k] = D::I16Vec::load_array(d, &tmp[k]);
                }

                let mut tmp2 = [[0i16; 8]; 2];
                for i in 0..4 {
                    D::I16Vec::store_interleaved_2(
                        mid[i],
                        mid[i + 4],
                        tmp2.as_flattened_mut(),
                    );
                    D::I16Vec::load_array(d, &tmp2[0]).store_array(&mut data[2 * i * stride]);
                    D::I16Vec::load_array(d, &tmp2[1]).store_array(&mut data[(2 * i + 1) * stride]);
                }
            }
            _ => transpose_square_i16_fallback(d, data, stride),
        }
    );
}

#[cfg(test)]
mod tests {
    use jxl_simd::{I16SimdVec, SimdDescriptor, test_all_instruction_sets};

    use super::{transpose_square_i16, transpose_square_i16_fallback};

    fn check_transpose_i16_with<D: SimdDescriptor>(
        d: D,
        f: fn(D, &mut [<D::I16Vec as I16SimdVec>::UnderlyingArray], usize),
    ) {
        let len = D::I16Vec::LEN;
        let input: Vec<i16> = (0..len * len).map(|i| i as i16).collect();
        let mut output = input.clone();
        f(d, D::I16Vec::make_array_slice_mut(&mut output), 1);

        for i in 0..len {
            for j in 0..len {
                assert_eq!(output[i * len + j], input[j * len + i]);
            }
        }

        let stride = 2;
        let mut input_strided = vec![0i16; len * len * stride];
        for i in 0..len {
            for j in 0..len {
                input_strided[(i * stride) * len + j] = (i * len + j) as i16;
            }
        }
        let mut output_strided = input_strided.clone();
        f(
            d,
            D::I16Vec::make_array_slice_mut(&mut output_strided),
            stride,
        );

        for i in 0..len {
            for j in 0..len {
                assert_eq!(
                    output_strided[(i * stride) * len + j],
                    input_strided[(j * stride) * len + i]
                );
            }
        }
    }

    fn test_i16_transpose_square<D: SimdDescriptor>(d: D) {
        check_transpose_i16_with(d, transpose_square_i16);
    }
    test_all_instruction_sets!(test_i16_transpose_square);

    fn test_i16_transpose_square_fallback<D: SimdDescriptor>(d: D) {
        check_transpose_i16_with(d, transpose_square_i16_fallback);
    }
    test_all_instruction_sets!(test_i16_transpose_square_fallback);
}
