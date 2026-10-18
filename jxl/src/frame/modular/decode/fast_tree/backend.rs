// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

#[allow(unused_imports)]
use super::nodes::PATH_LUT;

/// Architecture-specific SIMD primitives for fast decision tree evaluation
/// and multi-dimensional table lookup clamping.
pub trait TreeBackend: Copy + Default + Send + Sync + 'static {
    #[allow(dead_code)]
    const NAME: &'static str;

    /// Evaluates `count` thresholds (`1..=8`) against `val` in parallel,
    /// returning the number of thresholds where `val > threshold`.
    fn eval_range(val: i32, thresholds: &[i32; 8], count: usize) -> usize;

    /// Evaluates a depth-2 supernode (3 splits -> 4 children).
    /// `props`: `[p0, p1, p2]`, `splits`: `[s0, s1, s2]`.
    fn eval_d2(props: [i32; 3], splits: [i32; 3]) -> usize;

    /// Evaluates a depth-3 supernode (7 splits -> 8 children).
    /// `props`: `[p0..p6]`, `splits`: `[s0..s6]`.
    fn eval_d3(props: [i32; 7], splits: [i32; 7]) -> usize;

    /// Clamps and biases 4 properties for MDD:
    /// `result[i] = props[i].clamp(min[i], max[i]) + bias[i]`.
    #[allow(dead_code)]
    fn clamp_and_bias_4(
        props: [i32; 4],
        min: [i32; 4],
        max: [i32; 4],
        bias: [i32; 4],
    ) -> [i32; 4];

    /// Clamps and biases 6 properties for MDD:
    /// `result[i] = props[i].clamp(min[i], max[i]) + bias[i]`.
    #[allow(dead_code)]
    fn clamp_and_bias_6(
        props: [i32; 6],
        min: [i32; 6],
        max: [i32; 6],
        bias: [i32; 6],
    ) -> [i32; 6];
}

// ============================================================================
// Cross-Architecture Generic / Scalar Backend
// ============================================================================

#[derive(Clone, Copy, Default, Debug)]
pub struct ScalarBackend;

impl TreeBackend for ScalarBackend {
    const NAME: &'static str = "scalar";

    #[inline(always)]
    fn eval_range(val: i32, thresholds: &[i32; 8], count: usize) -> usize {
        let mut count_gt = 0;
        for &t in &thresholds[..count] {
            count_gt += (val > t) as usize;
        }
        count_gt
    }

    #[inline(always)]
    fn eval_d2(props: [i32; 3], splits: [i32; 3]) -> usize {
        let c0 = props[0] > splits[0];
        let c1 = if c0 {
            props[2] > splits[2]
        } else {
            props[1] > splits[1]
        };
        ((c0 as usize) << 1) | (c1 as usize)
    }

    #[inline(always)]
    fn eval_d3(props: [i32; 7], splits: [i32; 7]) -> usize {
        let c0 = props[0] > splits[0];
        let i1 = if c0 { 2 } else { 1 };
        let c1 = props[i1] > splits[i1];
        let i2 = if c0 {
            if c1 { 6 } else { 5 }
        } else {
            if c1 { 4 } else { 3 }
        };
        let c2 = props[i2] > splits[i2];
        ((c0 as usize) << 2) | ((c1 as usize) << 1) | (c2 as usize)
    }

    #[inline(always)]
    fn clamp_and_bias_4(
        props: [i32; 4],
        min: [i32; 4],
        max: [i32; 4],
        bias: [i32; 4],
    ) -> [i32; 4] {
        [
            props[0].clamp(min[0], max[0]) + bias[0],
            props[1].clamp(min[1], max[1]) + bias[1],
            props[2].clamp(min[2], max[2]) + bias[2],
            props[3].clamp(min[3], max[3]) + bias[3],
        ]
    }

    #[inline(always)]
    fn clamp_and_bias_6(
        props: [i32; 6],
        min: [i32; 6],
        max: [i32; 6],
        bias: [i32; 6],
    ) -> [i32; 6] {
        [
            props[0].clamp(min[0], max[0]) + bias[0],
            props[1].clamp(min[1], max[1]) + bias[1],
            props[2].clamp(min[2], max[2]) + bias[2],
            props[3].clamp(min[3], max[3]) + bias[3],
            props[4].clamp(min[4], max[4]) + bias[4],
            props[5].clamp(min[5], max[5]) + bias[5],
        ]
    }
}

// ============================================================================
// AVX2 Backend (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Default, Debug)]
pub struct Avx2Backend;

#[cfg(target_arch = "x86_64")]
impl TreeBackend for Avx2Backend {
    const NAME: &'static str = "avx2";

    #[inline(always)]
    fn eval_range(val: i32, thresholds: &[i32; 8], count: usize) -> usize {
        #[target_feature(enable = "avx2")]
        unsafe fn inner(val: i32, thresholds: &[i32; 8], count: usize) -> usize {
            use std::arch::x86_64::*;
            let v_val = _mm256_set1_epi32(val);
            let v_thresh = unsafe { _mm256_loadu_si256(thresholds.as_ptr() as *const __m256i) };
            let cmp = _mm256_cmpgt_epi32(v_val, v_thresh);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32;
            let valid_mask = (1u32 << count) - 1;
            (mask & valid_mask).count_ones() as usize
        }
        unsafe { inner(val, thresholds, count) }
    }

    #[inline(always)]
    fn eval_d2(props: [i32; 3], splits: [i32; 3]) -> usize {
        #[target_feature(enable = "avx2")]
        unsafe fn inner(props: [i32; 3], splits: [i32; 3]) -> usize {
            use std::arch::x86_64::*;
            let v_props = _mm_setr_epi32(props[0], props[1], props[2], 0);
            let v_splits = _mm_setr_epi32(splits[0], splits[1], splits[2], 0);
            let cmp = _mm_cmpgt_epi32(v_props, v_splits);
            let mask = _mm_movemask_ps(_mm_castsi128_ps(cmp)) as u32;
            let c0 = mask & 1;
            let c1 = (mask >> (1 + c0)) & 1;
            ((c0 as usize) << 1) | (c1 as usize)
        }
        unsafe { inner(props, splits) }
    }

    #[inline(always)]
    fn eval_d3(props: [i32; 7], splits: [i32; 7]) -> usize {
        #[target_feature(enable = "avx2")]
        unsafe fn inner(props: [i32; 7], splits: [i32; 7]) -> usize {
            use std::arch::x86_64::*;
            let v_props = _mm256_setr_epi32(
                props[0], props[1], props[2], props[3], props[4], props[5], props[6], 0,
            );
            let v_splits = _mm256_setr_epi32(
                splits[0], splits[1], splits[2], splits[3], splits[4], splits[5], splits[6], 0,
            );
            let cmp = _mm256_cmpgt_epi32(v_props, v_splits);
            let mask = (_mm256_movemask_ps(_mm256_castsi256_ps(cmp)) & 0x7f) as usize;
            PATH_LUT[mask] as usize
        }
        unsafe { inner(props, splits) }
    }

    #[inline(always)]
    fn clamp_and_bias_4(
        props: [i32; 4],
        min: [i32; 4],
        max: [i32; 4],
        bias: [i32; 4],
    ) -> [i32; 4] {
        [
            props[0].clamp(min[0], max[0]) + bias[0],
            props[1].clamp(min[1], max[1]) + bias[1],
            props[2].clamp(min[2], max[2]) + bias[2],
            props[3].clamp(min[3], max[3]) + bias[3],
        ]
    }

    #[inline(always)]
    fn clamp_and_bias_6(
        props: [i32; 6],
        min: [i32; 6],
        max: [i32; 6],
        bias: [i32; 6],
    ) -> [i32; 6] {
        [
            props[0].clamp(min[0], max[0]) + bias[0],
            props[1].clamp(min[1], max[1]) + bias[1],
            props[2].clamp(min[2], max[2]) + bias[2],
            props[3].clamp(min[3], max[3]) + bias[3],
            props[4].clamp(min[4], max[4]) + bias[4],
            props[5].clamp(min[5], max[5]) + bias[5],
        ]
    }
}

// ============================================================================
// NEON Backend (AArch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Default, Debug)]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
const NEON_LANE_MASKS: [[i32; 4]; 5] = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, -1, 0, 0],
    [-1, -1, -1, 0],
    [-1, -1, -1, -1],
];

#[cfg(target_arch = "aarch64")]
impl TreeBackend for NeonBackend {
    const NAME: &'static str = "neon";

    #[inline(always)]
    fn eval_range(val: i32, thresholds: &[i32; 8], count: usize) -> usize {
        #[target_feature(enable = "neon")]
        unsafe fn inner(val: i32, thresholds: &[i32; 8], count: usize) -> usize {
            use std::arch::aarch64::*;
            let v_val = vdupq_n_s32(val);
            let v_t0 = unsafe { vld1q_s32(thresholds.as_ptr()) };
            let cmp0 = vcgtq_s32(v_val, v_t0); // 0 or -1 (0xFFFFFFFF)
            let ones0 = vnegq_s32(vreinterpretq_s32_u32(cmp0)); // 0 or 1
            if count <= 4 {
                let mask = unsafe { vld1q_s32(NEON_LANE_MASKS[count].as_ptr()) };
                vaddvq_s32(vandq_s32(ones0, mask)) as usize
            } else {
                let v_t1 = unsafe { vld1q_s32(thresholds.as_ptr().add(4)) };
                let cmp1 = vcgtq_s32(v_val, v_t1);
                let ones1 = vnegq_s32(vreinterpretq_s32_u32(cmp1));
                let mask1 = unsafe { vld1q_s32(NEON_LANE_MASKS[count - 4].as_ptr()) };
                (vaddvq_s32(ones0) + vaddvq_s32(vandq_s32(ones1, mask1))) as usize
            }
        }
        unsafe { inner(val, thresholds, count) }
    }

    #[inline(always)]
    fn eval_d2(props: [i32; 3], splits: [i32; 3]) -> usize {
        // On AArch64, scalar comparisons compile to a branchless 2-cycle `cmp` + `csel` pair.
        let c0 = props[0] > splits[0];
        let c1 = if c0 {
            props[2] > splits[2]
        } else {
            props[1] > splits[1]
        };
        ((c0 as usize) << 1) | (c1 as usize)
    }

    #[inline(always)]
    fn eval_d3(props: [i32; 7], splits: [i32; 7]) -> usize {
        // 3-step decision tree compiling to 3 branchless `cmp` + `csel` pairs on AArch64 (3 cycles).
        let c0 = props[0] > splits[0];
        let i1 = if c0 { 2 } else { 1 };
        let c1 = props[i1] > splits[i1];
        let i2 = if c0 {
            if c1 { 6 } else { 5 }
        } else {
            if c1 { 4 } else { 3 }
        };
        let c2 = props[i2] > splits[i2];
        ((c0 as usize) << 2) | ((c1 as usize) << 1) | (c2 as usize)
    }

    #[inline(always)]
    fn clamp_and_bias_4(
        props: [i32; 4],
        min: [i32; 4],
        max: [i32; 4],
        bias: [i32; 4],
    ) -> [i32; 4] {
        #[target_feature(enable = "neon")]
        unsafe fn inner(
            props: [i32; 4],
            min: [i32; 4],
            max: [i32; 4],
            bias: [i32; 4],
        ) -> [i32; 4] {
            use std::arch::aarch64::*;
            let v_p = unsafe { vld1q_s32(props.as_ptr()) };
            let v_min = unsafe { vld1q_s32(min.as_ptr()) };
            let v_max = unsafe { vld1q_s32(max.as_ptr()) };
            let v_bias = unsafe { vld1q_s32(bias.as_ptr()) };
            let clamped = vminq_s32(vmaxq_s32(v_p, v_min), v_max);
            let biased = vaddq_s32(clamped, v_bias);
            let mut out = [0i32; 4];
            unsafe { vst1q_s32(out.as_mut_ptr(), biased) };
            out
        }
        unsafe { inner(props, min, max, bias) }
    }

    #[inline(always)]
    fn clamp_and_bias_6(
        props: [i32; 6],
        min: [i32; 6],
        max: [i32; 6],
        bias: [i32; 6],
    ) -> [i32; 6] {
        #[target_feature(enable = "neon")]
        unsafe fn inner(
            props: [i32; 6],
            min: [i32; 6],
            max: [i32; 6],
            bias: [i32; 6],
        ) -> [i32; 6] {
            use std::arch::aarch64::*;
            let v_p0 = unsafe { vld1q_s32(props.as_ptr()) };
            let v_min0 = unsafe { vld1q_s32(min.as_ptr()) };
            let v_max0 = unsafe { vld1q_s32(max.as_ptr()) };
            let v_bias0 = unsafe { vld1q_s32(bias.as_ptr()) };
            let clamped0 = vminq_s32(vmaxq_s32(v_p0, v_min0), v_max0);
            let biased0 = vaddq_s32(clamped0, v_bias0);

            let v_p1 = unsafe { vld1_s32(props.as_ptr().add(4)) };
            let v_min1 = unsafe { vld1_s32(min.as_ptr().add(4)) };
            let v_max1 = unsafe { vld1_s32(max.as_ptr().add(4)) };
            let v_bias1 = unsafe { vld1_s32(bias.as_ptr().add(4)) };
            let clamped1 = vmin_s32(vmax_s32(v_p1, v_min1), v_max1);
            let biased1 = vadd_s32(clamped1, v_bias1);

            let mut out = [0i32; 6];
            unsafe {
                vst1q_s32(out.as_mut_ptr(), biased0);
                vst1_s32(out.as_mut_ptr().add(4), biased1);
            }
            out
        }
        unsafe { inner(props, min, max, bias) }
    }
}
