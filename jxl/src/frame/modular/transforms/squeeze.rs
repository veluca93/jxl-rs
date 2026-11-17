// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{
    I16Vec, I32Vec, IntElement, SignedElement, SimdDescriptor, SimdVec, shl, shr, simd_dispatch,
};

use crate::error::{Error, Result};
use crate::frame::modular::{ChannelInfo, ModularChannel, ModularStorage};
use crate::headers::modular::SqueezeParams;
use crate::image::{ImageDataType, ImageRect, ImageRectMut, RawImageRect};
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

trait SqueezeSample: IntElement + SignedElement + ImageDataType {
    const TWO: Self;
    const THREE: Self;
    const SAFE_LIMIT: Self;
    const OVERFLOW_SAFE_16: bool;

    fn to_i32(self) -> i32;
    fn from_i32(v: i32) -> Self;
    fn mul_high_third<D: SimdDescriptor>(d: D, v: SimdVec<Self, D>) -> SimdVec<Self, D>;
    fn sign_mask<D: SimdDescriptor>(v: SimdVec<Self, D>) -> SimdVec<Self, D>;
}

impl SqueezeSample for i32 {
    const TWO: Self = 2;
    const THREE: Self = 3;
    const SAFE_LIMIT: Self = 0;
    const OVERFLOW_SAFE_16: bool = false;

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v
    }

    #[inline(always)]
    fn mul_high_third<D: SimdDescriptor>(d: D, v: I32Vec<D>) -> I32Vec<D> {
        v.mul_wide_take_high(I32Vec::splat(d, 0x55555556))
    }

    #[inline(always)]
    fn sign_mask<D: SimdDescriptor>(v: I32Vec<D>) -> I32Vec<D> {
        shr!(v, 31)
    }
}

impl SqueezeSample for i16 {
    const TWO: Self = 2;
    const THREE: Self = 3;
    const SAFE_LIMIT: Self = 16383;
    const OVERFLOW_SAFE_16: bool = true;

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as i16
    }

    #[inline(always)]
    fn mul_high_third<D: SimdDescriptor>(d: D, v: I16Vec<D>) -> I16Vec<D> {
        v.mul_wide_take_high(I16Vec::splat(d, 0x5556))
    }

    #[inline(always)]
    fn sign_mask<D: SimdDescriptor>(v: I16Vec<D>) -> I16Vec<D> {
        shr!(v, 15)
    }
}

#[inline(always)]
fn smooth_tendency_impl<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    a: SimdVec<T, D>,
    b: SimdVec<T, D>,
    c: SimdVec<T, D>,
) -> SimdVec<T, D> {
    let a_b = a - b;
    let b_c = b - c;
    let a_c = a - c;
    let abs_a_b = a_b.abs();
    let abs_b_c = b_c.abs();
    let abs_a_c = a_c.abs();
    let non_monotonic = (a_b ^ b_c).lt_zero();
    let skip = a_b.eq_zero().andnot(non_monotonic);
    let skip = b_c.eq_zero().andnot(skip);

    let abs_a_b_3 = T::mul_high_third(d, abs_a_b);
    let one = SimdVec::splat(d, T::ONE);
    let two = SimdVec::splat(d, T::TWO);

    let x = if T::OVERFLOW_SAFE_16 {
        let three = SimdVec::splat(d, T::THREE);
        shr!(abs_a_c, 2) + shr!(two + abs_a_b_3 + (abs_a_c & three), 2)
    } else {
        shr!(two + abs_a_c + abs_a_b_3, 2)
    };

    let (safe_abs_a_b, safe_abs_b_c) = if T::OVERFLOW_SAFE_16 {
        let limit = SimdVec::splat(d, T::SAFE_LIMIT);
        (
            abs_a_b.gt(limit).if_then_else(limit, abs_a_b),
            abs_b_c.gt(limit).if_then_else(limit, abs_b_c),
        )
    } else {
        (abs_a_b, abs_b_c)
    };

    let abs_a_b_2_add_x = shl!(safe_abs_a_b, 1) + (x & one);
    let x = x
        .gt(abs_a_b_2_add_x)
        .if_then_else(shl!(safe_abs_a_b, 1) + one, x);

    let abs_b_c_2 = shl!(safe_abs_b_c, 1);
    let x = (x + (x & one)).gt(abs_b_c_2).if_then_else(abs_b_c_2, x);

    let need_neg = a_c.lt_zero();
    let x = skip.if_then_zero_else(x);
    need_neg.if_then_else(-x, x)
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
fn unsqueeze_impl<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    avg: SimdVec<T, D>,
    res: SimdVec<T, D>,
    next_avg: SimdVec<T, D>,
    prev: SimdVec<T, D>,
) -> (SimdVec<T, D>, SimdVec<T, D>) {
    let tendency = smooth_tendency_impl(d, prev, avg, next_avg);
    let diff = res + tendency;
    let sign = T::sign_mask(diff);
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
fn transpose_inlane<T: SqueezeSample, D: SimdDescriptor, const K: usize>(
    mut regs: [SimdVec<T, D>; K],
) -> [SimdVec<T, D>; K] {
    const { assert!(K == 4 || K == 8) };
    macro_rules! stage {
        ($half:literal) => {
            if K >= 2 * $half {
                for c in (0..K).step_by(2 * $half) {
                    for i in 0..$half {
                        let (a, b) = (regs[c + i], regs[c + $half + i]);
                        regs[c + i] = SimdVec::interleave_lo(a, b);
                        regs[c + $half + i] = SimdVec::interleave_hi(a, b);
                    }
                }
            }
        };
    }
    stage!(4);
    stage!(2);
    stage!(1);
    regs
}

#[inline(always)]
fn load_group_inlane<T: SqueezeSample, D: SimdDescriptor, const K: usize>(
    d: D,
    row_base: usize,
    load_row: &mut impl FnMut(usize) -> SimdVec<T, D>,
) -> [SimdVec<T, D>; K] {
    let mut regs = [SimdVec::zero(d); K];
    for (r, reg) in regs.iter_mut().enumerate() {
        *reg = load_row(row_base + r);
    }
    transpose_inlane::<T, D, K>(regs)
}

#[inline(always)]
fn load_transpose_tile<T: SqueezeSample, D: SimdDescriptor, const LEN: usize, const K: usize>(
    d: D,
    mut load_row: impl FnMut(usize) -> SimdVec<T, D>,
) -> [SimdVec<T, D>; LEN] {
    let mut out = [SimdVec::zero(d); LEN];
    if D::BYTES == 16 {
        let g0 = load_group_inlane::<T, D, K>(d, 0, &mut load_row);
        out[..K].copy_from_slice(&g0);
    } else if D::BYTES == 32 {
        let g0 = load_group_inlane::<T, D, K>(d, 0, &mut load_row);
        let g1 = load_group_inlane::<T, D, K>(d, K, &mut load_row);
        for c in 0..K {
            let (c0, c1) = g0[c].interleave_blocks_128(g1[c]);
            out[c] = c0;
            out[K + c] = c1;
        }
    } else if D::BYTES == 64 {
        let g0 = load_group_inlane::<T, D, K>(d, 0, &mut load_row);
        let g1 = load_group_inlane::<T, D, K>(d, K, &mut load_row);
        let g2 = load_group_inlane::<T, D, K>(d, 2 * K, &mut load_row);
        let g3 = load_group_inlane::<T, D, K>(d, 3 * K, &mut load_row);
        for c in 0..K {
            let (p0, p2) = g0[c].interleave_blocks_128(g1[c]);
            let (p1, p3) = g2[c].interleave_blocks_128(g3[c]);
            let (o0, o2) = p0.interleave_blocks_256(p1);
            let (o1, o3) = p2.interleave_blocks_256(p3);
            out[c] = o0;
            out[K + c] = o1;
            out[2 * K + c] = o2;
            out[3 * K + c] = o3;
        }
    }
    out
}

type StripPair<T, D, const K: usize> = ([SimdVec<T, D>; K], [SimdVec<T, D>; K]);

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn unsqueeze_k<T: SqueezeSample, D: SimdDescriptor, const LEN: usize, const K: usize>(
    d: D,
    avg_cols: &[SimdVec<T, D>; LEN],
    res_cols: &[SimdVec<T, D>; LEN],
    strip: usize,
    avg_first: &mut SimdVec<T, D>,
    res_first: &mut SimdVec<T, D>,
    prev_b: &mut SimdVec<T, D>,
) -> StripPair<T, D, K> {
    let mut a_strip = [SimdVec::zero(d); K];
    let mut b_strip = [SimdVec::zero(d); K];
    for c in 0..K {
        let avg_next = avg_cols[strip * K + c];
        let res_next = res_cols[strip * K + c];
        let (a, b) = unsqueeze_impl(d, *avg_first, *res_first, avg_next, *prev_b);
        *avg_first = avg_next;
        *res_first = res_next;
        *prev_b = b;
        a_strip[c] = a;
        b_strip[c] = b;
    }
    (
        transpose_inlane::<T, D, K>(a_strip),
        transpose_inlane::<T, D, K>(b_strip),
    )
}

#[inline(always)]
fn store_tile_vec<T: SqueezeSample, D: SimdDescriptor, const LEN: usize, const FULL: bool>(
    row: &mut [T],
    lane_tmp: &mut [T; LEN],
    out_x: usize,
    group: usize,
    steps: usize,
    v: SimdVec<T, D>,
) {
    if FULL || steps == LEN {
        v.store(&mut row[out_x + group * LEN..]);
    } else {
        let n = (2 * steps).saturating_sub(group * LEN).min(LEN);
        if n > 0 {
            v.store(lane_tmp);
            row[out_x + group * LEN..][..n].copy_from_slice(&lane_tmp[..n]);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn unsqueeze_and_store_tile<
    T: SqueezeSample,
    D: SimdDescriptor,
    const LEN: usize,
    const K: usize,
    const FULL: bool,
>(
    d: D,
    avg_cols: &[SimdVec<T, D>; LEN],
    res_cols: &[SimdVec<T, D>; LEN],
    avg_first: &mut SimdVec<T, D>,
    res_first: &mut SimdVec<T, D>,
    prev_b: &mut SimdVec<T, D>,
    out_rows: &mut [&mut [T]; LEN],
    lane_tmp: &mut [T; LEN],
    out_x: usize,
    steps: usize,
) {
    if D::BYTES == 16 {
        let (ra, rb) =
            unsqueeze_k::<T, D, LEN, K>(d, avg_cols, res_cols, 0, avg_first, res_first, prev_b);
        for r in 0..K {
            store_tile_vec::<T, D, LEN, FULL>(
                out_rows[r],
                lane_tmp,
                out_x,
                0,
                steps,
                SimdVec::interleave_lo(ra[r], rb[r]),
            );
            store_tile_vec::<T, D, LEN, FULL>(
                out_rows[r],
                lane_tmp,
                out_x,
                1,
                steps,
                SimdVec::interleave_hi(ra[r], rb[r]),
            );
        }
    } else if D::BYTES == 32 {
        for strip in 0..2 {
            let (ra, rb) = unsqueeze_k::<T, D, LEN, K>(
                d, avg_cols, res_cols, strip, avg_first, res_first, prev_b,
            );
            for r in 0..K {
                let lo = SimdVec::interleave_lo(ra[r], rb[r]);
                let hi = SimdVec::interleave_hi(ra[r], rb[r]);
                let (o0, o1) = lo.interleave_blocks_128(hi);
                store_tile_vec::<T, D, LEN, FULL>(out_rows[r], lane_tmp, out_x, strip, steps, o0);
                store_tile_vec::<T, D, LEN, FULL>(
                    out_rows[K + r],
                    lane_tmp,
                    out_x,
                    strip,
                    steps,
                    o1,
                );
            }
        }
    } else if D::BYTES == 64 {
        for pair in 0..2 {
            let (ra0, rb0) = unsqueeze_k::<T, D, LEN, K>(
                d,
                avg_cols,
                res_cols,
                2 * pair,
                avg_first,
                res_first,
                prev_b,
            );
            let mut s0_even = [SimdVec::zero(d); K];
            let mut s0_odd = [SimdVec::zero(d); K];
            for r in 0..K {
                let lo = SimdVec::interleave_lo(ra0[r], rb0[r]);
                let hi = SimdVec::interleave_hi(ra0[r], rb0[r]);
                let (even, odd) = lo.interleave_blocks_128(hi);
                s0_even[r] = even;
                s0_odd[r] = odd;
            }
            let (ra1, rb1) = unsqueeze_k::<T, D, LEN, K>(
                d,
                avg_cols,
                res_cols,
                2 * pair + 1,
                avg_first,
                res_first,
                prev_b,
            );
            for r in 0..K {
                let lo = SimdVec::interleave_lo(ra1[r], rb1[r]);
                let hi = SimdVec::interleave_hi(ra1[r], rb1[r]);
                let (s1_even, s1_odd) = lo.interleave_blocks_128(hi);
                let (row_0, row_2) = s0_even[r].interleave_blocks_256(s1_even);
                let (row_1, row_3) = s0_odd[r].interleave_blocks_256(s1_odd);
                store_tile_vec::<T, D, LEN, FULL>(out_rows[r], lane_tmp, out_x, pair, steps, row_0);
                store_tile_vec::<T, D, LEN, FULL>(
                    out_rows[K + r],
                    lane_tmp,
                    out_x,
                    pair,
                    steps,
                    row_1,
                );
                store_tile_vec::<T, D, LEN, FULL>(
                    out_rows[2 * K + r],
                    lane_tmp,
                    out_x,
                    pair,
                    steps,
                    row_2,
                );
                store_tile_vec::<T, D, LEN, FULL>(
                    out_rows[3 * K + r],
                    lane_tmp,
                    out_x,
                    pair,
                    steps,
                    row_3,
                );
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hsqueeze_bands<T: SqueezeSample, D: SimdDescriptor, const LEN: usize, const K: usize>(
    d: D,
    y_start: usize,
    y_limit: usize,
    has_tail: bool,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
) {
    let w = in_res.size().0;
    let has_next_in_avg = in_avg.size().0 > w;
    let mask = !(LEN - 1);
    let remainder_start = ((w - 1) & mask) + 1;
    let remainder_count = w - remainder_start;
    let mut lane_tmp = [T::default(); LEN];

    for y in (y_start..y_limit).step_by(LEN) {
        let avg_rows: [&[T]; LEN] = std::array::from_fn(|dy| in_avg.row(y + dy));
        let res_rows: [&[T]; LEN] = std::array::from_fn(|dy| in_res.row(y + dy));
        let mut out_rows: [&mut [T]; LEN] = out.distinct_rows_mut(std::array::from_fn(|dy| y + dy));

        for (dy, slot) in lane_tmp.iter_mut().enumerate() {
            *slot = avg_rows[dy][0];
        }
        let mut avg_first = SimdVec::load(d, &lane_tmp);
        for (dy, slot) in lane_tmp.iter_mut().enumerate() {
            *slot = res_rows[dy][0];
        }
        let mut res_first = SimdVec::load(d, &lane_tmp);

        let mut prev_b = match out_prev {
            None => avg_first,
            Some(lr) => {
                for (dy, slot) in lane_tmp.iter_mut().enumerate() {
                    *slot = lr.row(y + dy)[3];
                }
                SimdVec::load(d, &lane_tmp)
            }
        };

        for x in (1..remainder_start).step_by(LEN) {
            let avg_cols =
                load_transpose_tile::<T, D, LEN, K>(d, |dy| SimdVec::load(d, &avg_rows[dy][x..]));
            let res_cols =
                load_transpose_tile::<T, D, LEN, K>(d, |dy| SimdVec::load(d, &res_rows[dy][x..]));
            unsqueeze_and_store_tile::<T, D, LEN, K, true>(
                d,
                &avg_cols,
                &res_cols,
                &mut avg_first,
                &mut res_first,
                &mut prev_b,
                &mut out_rows,
                &mut lane_tmp,
                2 * x - 2,
                LEN,
            );
        }

        let x = remainder_start;
        if remainder_count == 0 {
            let avg_last = if has_tail || has_next_in_avg {
                for (dy, slot) in lane_tmp.iter_mut().enumerate() {
                    *slot = avg_rows[dy][w];
                }
                SimdVec::load(d, &lane_tmp)
            } else if let Some(lr) = in_next_avg {
                for (dy, slot) in lane_tmp.iter_mut().enumerate() {
                    *slot = lr.row(y + dy)[0];
                }
                SimdVec::load(d, &lane_tmp)
            } else {
                avg_first
            };

            let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_last, prev_b);
            let mut a_tmp = [T::default(); LEN];
            let mut b_tmp = [T::default(); LEN];
            a.store(&mut a_tmp);
            b.store(&mut b_tmp);
            for dy in 0..LEN {
                out_rows[dy][2 * x - 2] = a_tmp[dy];
                out_rows[dy][2 * x - 1] = b_tmp[dy];
            }
        } else {
            let avg_cols = load_transpose_tile::<T, D, LEN, K>(d, |dy| {
                lane_tmp[..remainder_count].copy_from_slice(&avg_rows[dy][x..x + remainder_count]);
                lane_tmp[remainder_count] = if has_tail || has_next_in_avg {
                    avg_rows[dy][w]
                } else if let Some(lr) = in_next_avg {
                    lr.row(y + dy)[0]
                } else {
                    lane_tmp[remainder_count - 1]
                };
                SimdVec::load(d, &lane_tmp)
            });
            let res_cols = load_transpose_tile::<T, D, LEN, K>(d, |dy| {
                lane_tmp[..remainder_count].copy_from_slice(&res_rows[dy][x..x + remainder_count]);
                SimdVec::load(d, &lane_tmp)
            });
            let steps = remainder_count + 1;
            unsqueeze_and_store_tile::<T, D, LEN, K, false>(
                d,
                &avg_cols,
                &res_cols,
                &mut avg_first,
                &mut res_first,
                &mut prev_b,
                &mut out_rows,
                &mut lane_tmp,
                2 * x - 2,
                steps,
            );
        }

        if has_tail {
            for dy in 0..LEN {
                out_rows[dy][2 * w] = avg_rows[dy][w];
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hsqueeze_impl<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    y_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
) {
    const {
        assert!(SimdVec::<T, D>::LEN <= 32);
        assert!(SimdVec::<T, D>::LEN.is_power_of_two());
    }

    let lanes = SimdVec::<T, D>::LEN;
    assert_eq!(y_start % lanes, 0);

    let (w, h) = in_res.size();
    if D::IS_EMULATED {
        return hsqueeze_scalar(y_start, in_avg, in_res, in_next_avg, out_prev, out);
    }

    let has_tail = out.size().0 & 1 == 1;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.size().0 == 2 * w + 1);
    }

    let mask = !(lanes - 1);
    let y_limit = if w >= lanes { h & mask } else { y_start };

    if y_limit > y_start {
        macro_rules! dispatch {
            ($(($len:literal, $k:literal)),+ $(,)?) => {
                match (lanes, 16 / std::mem::size_of::<T>()) {
                    $(($len, $k) => hsqueeze_bands::<T, D, $len, $k>(
                        d, y_start, y_limit, has_tail, in_avg, in_res, in_next_avg, out_prev, out,
                    ),)+
                    _ => unreachable!(),
                }
            };
        }
        dispatch!((4, 4), (8, 4), (16, 4), (8, 8), (16, 8), (32, 8));
    }

    let remainder_rows = h - y_limit;
    let lanes_256 = 32 / std::mem::size_of::<T>();
    let lanes_128 = 16 / std::mem::size_of::<T>();
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > lanes_256 && remainder_rows >= lanes_256 && w >= lanes_256 {
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
    if lanes > lanes_128 && remainder_rows >= lanes_128 && w >= lanes_128 {
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

fn hsqueeze_scalar<T: SqueezeSample>(
    y_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
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
                avg_row[x].to_i32(),
                res_row[x].to_i32(),
                avg_row[x + 1].to_i32(),
                prev_b.to_i32(),
            );
            out.row(y)[2 * x] = T::from_i32(a);
            out.row(y)[2 * x + 1] = T::from_i32(b);
            prev_b = T::from_i32(b);
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
                avg_row[w - 1].to_i32(),
                res_row[w - 1].to_i32(),
                last_avg.to_i32(),
                prev_b.to_i32(),
            );
            out.row(y)[2 * w - 2] = T::from_i32(a);
            out.row(y)[2 * w - 1] = T::from_i32(b);
        } else {
            // 1 last pixel
            out.row(y)[2 * w] = in_avg.row(y)[w];
        }
    }
}

#[simd_dispatch(hsqueeze)]
#[allow(clippy::too_many_arguments)]
fn hsqueeze_fwd<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    y_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
) {
    d.call(
        #[inline(always)]
        |d| hsqueeze_impl(d, y_start, in_avg, in_res, in_next_avg, out_prev, out),
    )
}

#[inline(always)]
fn do_hsqueeze_step_typed<T: SqueezeSample>(
    in_avg: &RawImageRect<'_>,
    in_res: &RawImageRect<'_>,
    in_next_avg: Option<&RawImageRect<'_>>,
    out_prev: Option<&RawImageRect<'_>>,
    buffers: &mut [&mut ModularChannel],
) {
    let in_avg = ImageRect::<T>::from_raw(*in_avg);
    let in_res = ImageRect::<T>::from_raw(*in_res);
    let in_next_avg = in_next_avg.map(|r| ImageRect::<T>::from_raw(*r));
    let out_prev = out_prev.map(|r| ImageRect::<T>::from_raw(*r));
    trace!("hsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<T>::from_raw(out.data.as_rect_mut());
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
    // Otherwise: 2 or more in row
    hsqueeze::<T>(
        0,
        &in_avg,
        &in_res,
        in_next_avg.as_ref(),
        out_prev.as_ref(),
        &mut out_rect,
    );
}

#[inline(always)]
pub fn do_hsqueeze_step(
    in_avg: &RawImageRect<'_>,
    in_res: &RawImageRect<'_>,
    in_next_avg: Option<&RawImageRect<'_>>,
    out_prev: Option<&RawImageRect<'_>>,
    buffers: &mut [&mut ModularChannel],
    storage: ModularStorage,
) {
    if storage == ModularStorage::I16 {
        do_hsqueeze_step_typed::<i16>(in_avg, in_res, in_next_avg, out_prev, buffers);
    } else {
        do_hsqueeze_step_typed::<i32>(in_avg, in_res, in_next_avg, out_prev, buffers);
    }
}

#[inline(always)]
fn vsqueeze_impl<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    x_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
) {
    const { assert!(SimdVec::<T, D>::LEN.is_power_of_two()) };

    let lanes = SimdVec::<T, D>::LEN;
    assert_eq!(x_start % lanes, 0);

    let (w, h) = in_res.size();
    if D::IS_EMULATED {
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
        let mut prev_b = SimdVec::load(d, &prev_b_row[x..]);
        let mut avg_first = SimdVec::load(d, &in_avg.row(0)[x..]);
        let mut res_first = SimdVec::load(d, &in_res.row(0)[x..]);
        for y in 0..h - 1 {
            let avg_next = SimdVec::load(d, &in_avg.row(y + 1)[x..]);
            let (a, b) = unsqueeze_impl(d, avg_first, res_first, avg_next, prev_b);
            a.store(&mut out.row(2 * y)[x..]);
            b.store(&mut out.row(2 * y + 1)[x..]);
            prev_b = b;
            avg_first = avg_next;
            res_first = SimdVec::load(d, &in_res.row(y + 1)[x..]);
        }

        let avg_last = if has_tail || has_next_in_avg {
            SimdVec::load(d, &in_avg.row(h)[x..])
        } else if let Some(tb) = in_next_avg {
            SimdVec::load(d, &tb.row(0)[x..])
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
    let lanes_256 = 32 / std::mem::size_of::<T>();
    let lanes_128 = 16 / std::mem::size_of::<T>();
    // We need `lanes > N` to convince the compiler that this function does not recurse
    if lanes > lanes_256 && remainder_cols >= lanes_256 {
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
    if lanes > lanes_128 && remainder_cols >= lanes_128 {
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

fn vsqueeze_scalar<T: SqueezeSample>(
    x_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
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
                avg_row[x].to_i32(),
                res_row[x].to_i32(),
                avg_row_next[x].to_i32(),
                prev_b_row[x].to_i32(),
            );
            out_row0[x] = T::from_i32(a);
            out_row1[x] = T::from_i32(b);
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
                avg_row[x].to_i32(),
                res_row[x].to_i32(),
                avg_row_next[x].to_i32(),
                prev_b_row[x].to_i32(),
            );
            out_row0[x] = T::from_i32(a);
            out_row1[x] = T::from_i32(b);
        }
    }
    if has_tail {
        out.row(2 * h)[x_start..].copy_from_slice(&in_avg.row(h)[x_start..]);
    }
}

#[simd_dispatch(vsqueeze)]
fn vsqueeze_fwd<T: SqueezeSample, D: SimdDescriptor>(
    d: D,
    x_start: usize,
    in_avg: &ImageRect<'_, T>,
    in_res: &ImageRect<'_, T>,
    in_next_avg: Option<&ImageRect<'_, T>>,
    out_prev: Option<&ImageRect<'_, T>>,
    out: &mut ImageRectMut<'_, T>,
) {
    d.call(
        #[inline(always)]
        |d| vsqueeze_impl(d, x_start, in_avg, in_res, in_next_avg, out_prev, out),
    )
}

#[inline(always)]
fn do_vsqueeze_step_typed<T: SqueezeSample>(
    in_avg: &RawImageRect<'_>,
    in_res: &RawImageRect<'_>,
    in_next_avg: Option<&RawImageRect<'_>>,
    out_prev: Option<&RawImageRect<'_>>,
    buffers: &mut [&mut ModularChannel],
) {
    let in_avg = ImageRect::<T>::from_raw(*in_avg);
    let in_res = ImageRect::<T>::from_raw(*in_res);
    let in_next_avg = in_next_avg.map(|r| ImageRect::<T>::from_raw(*r));
    let out_prev = out_prev.map(|r| ImageRect::<T>::from_raw(*r));
    trace!("vsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    let mut out_rect = ImageRectMut::<T>::from_raw(out.data.as_rect_mut());
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
    vsqueeze::<T>(
        0,
        &in_avg,
        &in_res,
        in_next_avg.as_ref(),
        out_prev.as_ref(),
        &mut out_rect,
    );
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
        do_vsqueeze_step_typed::<i16>(in_avg, in_res, in_next_avg, out_prev, buffers);
    } else {
        do_vsqueeze_step_typed::<i32>(in_avg, in_res, in_next_avg, out_prev, buffers);
    }
}
