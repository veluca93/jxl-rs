// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::x86_64::*;
use super::unaligned::{_mm_loadu_si128, _mm_storeu_si64, _mm_storeu_si128, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_storeu_ps, _mm256_storeu_si256};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use crate::x86_64::sse42::Sse42Descriptor;
use crate::{U32SimdVec, U64SimdVec, impl_f32_array_interface, impl_i16_array_interface};

/// Core 8x8 transpose algorithm for AVX2.
/// Takes 8 __m256 vectors representing rows and returns 8 transposed vectors.
/// Used by both store_interleaved_8 and transpose_square.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_8x8_core(
    r0: __m256,
    r1: __m256,
    r2: __m256,
    r3: __m256,
    r4: __m256,
    r5: __m256,
    r6: __m256,
    r7: __m256,
) -> (
    __m256,
    __m256,
    __m256,
    __m256,
    __m256,
    __m256,
    __m256,
    __m256,
) {
    // Stage 1: Unpack low/high pairs
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);

    // Stage 2: Shuffle to group 32-bit elements using 64-bit unpacks
    let s0 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t0),
        _mm256_castps_pd(t2),
    ));
    let s1 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t0),
        _mm256_castps_pd(t2),
    ));
    let s2 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t1),
        _mm256_castps_pd(t3),
    ));
    let s3 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t1),
        _mm256_castps_pd(t3),
    ));
    let s4 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t4),
        _mm256_castps_pd(t6),
    ));
    let s5 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t4),
        _mm256_castps_pd(t6),
    ));
    let s6 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t5),
        _mm256_castps_pd(t7),
    ));
    let s7 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t5),
        _mm256_castps_pd(t7),
    ));

    // Stage 3: 128-bit permute to finalize transpose
    let c0 = _mm256_permute2f128_ps::<0x20>(s0, s4);
    let c1 = _mm256_permute2f128_ps::<0x20>(s1, s5);
    let c2 = _mm256_permute2f128_ps::<0x20>(s2, s6);
    let c3 = _mm256_permute2f128_ps::<0x20>(s3, s7);
    let c4 = _mm256_permute2f128_ps::<0x31>(s0, s4);
    let c5 = _mm256_permute2f128_ps::<0x31>(s1, s5);
    let c6 = _mm256_permute2f128_ps::<0x31>(s2, s6);
    let c7 = _mm256_permute2f128_ps::<0x31>(s3, s7);

    (c0, c1, c2, c3, c4, c5, c6, c7)
}

// Safety invariant: this type is only ever constructed if avx2, fma, and f16c are available.
#[target_feature(enable = "avx2,fma,f16c")]
#[derive(Clone, Copy, Debug)]
pub struct AvxDescriptor(());

impl AvxDescriptor {
    /// # Safety
    /// The caller must guarantee that the "avx2", "fma", and "f16c" target features are available.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn new_unchecked() -> Self {
        Self(())
    }

    #[target_feature(from_args)]
    pub fn as_sse42(&self) -> Sse42Descriptor {
        Sse42Descriptor::new_unchecked()
    }
}

/// Prepared 8-entry lookup table for AVX2.
/// For AVX2, vpermps is both fast and exact, so we just store f32 values directly.
#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Avx(__m256, AvxDescriptor);

impl SimdDescriptor for AvxDescriptor {
    type F32Vec = F32VecAvx;
    type I32Vec = I32VecAvx;
    type I16Vec = I16VecAvx;
    type U64Vec = U64VecAvx;
    type U32Vec = U32VecAvx;
    type U8Vec = U8VecAvx;
    type U16Vec = U16VecAvx;
    type Mask = MaskAvx;
    type Mask16 = Mask16Avx;
    type Bf16Table8 = Bf16Table8Avx;

    type Descriptor256 = Self;
    type Descriptor128 = Sse42Descriptor;

    #[target_feature(from_args)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    #[target_feature(from_args)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self.as_sse42()
    }

    #[allow(unsafe_code)]
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            // SAFETY: we just checked avx2, fma, and f16c.
            Some(unsafe { Self(()) })
        } else {
            None
        }
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecAvx(__m256, AvxDescriptor);

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskAvx(__m256, AvxDescriptor);

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Avx(__m256i, AvxDescriptor);

impl F32SimdVec for F32VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 8;

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        Self(_mm256_loadu_ps(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [f32]) {
        _mm256_storeu_ps(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 2 * F32VecAvx::LEN);
        // a = [a0, a1, a2, a3, a4, a5, a6, a7], b = [b0, b1, b2, b3, b4, b5, b6, b7]
        // Output: [a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7]
        let lo = _mm256_unpacklo_ps(a.0, b.0); // [a0, b0, a1, b1, a4, b4, a5, b5]
        let hi = _mm256_unpackhi_ps(a.0, b.0); // [a2, b2, a3, b3, a6, b6, a7, b7]
        // Need to permute to get correct order
        let out0 = _mm256_permute2f128_ps::<0x20>(lo, hi); // lower halves: [a0,b0,a1,b1, a2,b2,a3,b3]
        let out1 = _mm256_permute2f128_ps::<0x31>(lo, hi); // upper halves: [a4,b4,a5,b5, a6,b6,a7,b7]
        let [d0, d1, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm256_storeu_ps(d0, out0);
        _mm256_storeu_ps(d1, out1);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 3 * F32VecAvx::LEN);

        let idx_a0 = _mm256_setr_epi32(0, 0, 0, 1, 0, 0, 2, 0);
        let idx_b0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 2);
        let idx_c0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0);

        let two = _mm256_set1_epi32(2);
        let three = _mm256_set1_epi32(3);
        let five = _mm256_set1_epi32(5);
        let six = _mm256_set1_epi32(6);

        let a0 = _mm256_permutevar8x32_ps(a.0, idx_a0);
        let b0 = _mm256_permutevar8x32_ps(b.0, idx_b0);
        let c0 = _mm256_permutevar8x32_ps(c.0, idx_c0);
        let out0 = _mm256_blend_ps::<0b10010010>(a0, b0);
        let out0 = _mm256_blend_ps::<0b00100100>(out0, c0);

        let a1 = _mm256_permutevar8x32_ps(a.0, _mm256_add_epi32(idx_b0, three));
        let b1 = _mm256_permutevar8x32_ps(b.0, _mm256_add_epi32(idx_c0, three));
        let c1 = _mm256_permutevar8x32_ps(c.0, _mm256_add_epi32(idx_a0, two));
        let out1 = _mm256_blend_ps::<0b00100100>(a1, b1);
        let out1 = _mm256_blend_ps::<0b01001001>(out1, c1);

        let a2 = _mm256_permutevar8x32_ps(a.0, _mm256_add_epi32(idx_c0, six));
        let b2 = _mm256_permutevar8x32_ps(b.0, _mm256_add_epi32(idx_a0, five));
        let c2 = _mm256_permutevar8x32_ps(c.0, _mm256_add_epi32(idx_b0, five));
        let out2 = _mm256_blend_ps::<0b01001001>(a2, b2);
        let out2 = _mm256_blend_ps::<0b10010010>(out2, c2);

        let [d0, d1, d2, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm256_storeu_ps(d0, out0);
        _mm256_storeu_ps(d1, out1);
        _mm256_storeu_ps(d2, out2);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 4 * F32VecAvx::LEN);
        // First interleave pairs
        let ab_lo = _mm256_unpacklo_ps(a.0, b.0);
        let ab_hi = _mm256_unpackhi_ps(a.0, b.0);
        let cd_lo = _mm256_unpacklo_ps(c.0, d.0);
        let cd_hi = _mm256_unpackhi_ps(c.0, d.0);

        // Cast to pd for 64-bit interleave
        let abcd_0 = _mm256_castpd_ps(_mm256_unpacklo_pd(
            _mm256_castps_pd(ab_lo),
            _mm256_castps_pd(cd_lo),
        ));
        let abcd_1 = _mm256_castpd_ps(_mm256_unpackhi_pd(
            _mm256_castps_pd(ab_lo),
            _mm256_castps_pd(cd_lo),
        ));
        let abcd_2 = _mm256_castpd_ps(_mm256_unpacklo_pd(
            _mm256_castps_pd(ab_hi),
            _mm256_castps_pd(cd_hi),
        ));
        let abcd_3 = _mm256_castpd_ps(_mm256_unpackhi_pd(
            _mm256_castps_pd(ab_hi),
            _mm256_castps_pd(cd_hi),
        ));

        // Permute to get correct order across lanes
        let out0 = _mm256_permute2f128_ps::<0x20>(abcd_0, abcd_1);
        let out1 = _mm256_permute2f128_ps::<0x20>(abcd_2, abcd_3);
        let out2 = _mm256_permute2f128_ps::<0x31>(abcd_0, abcd_1);
        let out3 = _mm256_permute2f128_ps::<0x31>(abcd_2, abcd_3);

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm256_storeu_ps(d0, out0);
        _mm256_storeu_ps(d1, out1);
        _mm256_storeu_ps(d2, out2);
        _mm256_storeu_ps(d3, out3);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        assert!(dest.len() >= 8 * F32VecAvx::LEN);
        // This is essentially an 8x8 transpose, same algorithm as transpose_square
        let (c0, c1, c2, c3, c4, c5, c6, c7) =
            transpose_8x8_core(a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0);

        let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm256_storeu_ps(d0, c0);
        _mm256_storeu_ps(d1, c1);
        _mm256_storeu_ps(d2, c2);
        _mm256_storeu_ps(d3, c3);
        _mm256_storeu_ps(d4, c4);
        _mm256_storeu_ps(d5, c5);
        _mm256_storeu_ps(d6, c6);
        _mm256_storeu_ps(d7, c7);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        assert!(src.len() >= 2 * F32VecAvx::LEN);
        // Input: [a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7]
        // Output: a = [a0, a1, a2, a3, a4, a5, a6, a7], b = [b0, b1, b2, b3, b4, b5, b6, b7]
        let [s0, s1, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1) = (_mm256_loadu_ps(s0), _mm256_loadu_ps(s1));

        // Reverse the store_interleaved_2 operation
        // First, undo the permute2f128
        let lo = _mm256_permute2f128_ps::<0x20>(in0, in1); // [a0,b0,a1,b1, a4,b4,a5,b5]
        let hi = _mm256_permute2f128_ps::<0x31>(in0, in1); // [a2,b2,a3,b3, a6,b6,a7,b7]

        // Then undo the unpack - use shuffle to separate a and b
        let a_lo = _mm256_shuffle_ps::<0x88>(lo, hi); // [a0, a1, a2, a3, a4, a5, a6, a7]
        let b_lo = _mm256_shuffle_ps::<0xDD>(lo, hi); // [b0, b1, b2, b3, b4, b5, b6, b7]

        (Self(a_lo, d), Self(b_lo, d))
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        assert!(src.len() >= 3 * F32VecAvx::LEN);
        // Input layout (24 floats):
        // in0: [a0, b0, c0, a1, b1, c1, a2, b2]
        // in1: [c2, a3, b3, c3, a4, b4, c4, a5]
        // in2: [b5, c5, a6, b6, c6, a7, b7, c7]
        // Output: a = [a0..a7], b = [b0..b7], c = [c0..c7]

        let [s0, s1, s2, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1, in2) = (_mm256_loadu_ps(s0), _mm256_loadu_ps(s1), _mm256_loadu_ps(s2));

        // Use permutevar8x32 to gather elements into correct positions, then blend.
        // permutevar8x32(src, idx): output[i] = src[idx[i]]

        // a: a0=in0[0], a1=in0[3], a2=in0[6], a3=in1[1], a4=in1[4], a5=in1[7], a6=in2[2], a7=in2[5]
        let perm_a0 = _mm256_setr_epi32(0, 3, 6, 0, 0, 0, 0, 0);
        let perm_a1 = _mm256_setr_epi32(0, 0, 0, 1, 4, 7, 0, 0);
        let perm_a2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 2, 5);
        let a0 = _mm256_permutevar8x32_ps(in0, perm_a0);
        let a1 = _mm256_permutevar8x32_ps(in1, perm_a1);
        let a2 = _mm256_permutevar8x32_ps(in2, perm_a2);
        let a_out = _mm256_blend_ps::<0b00111000>(a0, a1); // positions 3,4,5 from a1
        let a_out = _mm256_blend_ps::<0b11000000>(a_out, a2); // positions 6,7 from a2

        // b: b0=in0[1], b1=in0[4], b2=in0[7], b3=in1[2], b4=in1[5], b5=in2[0], b6=in2[3], b7=in2[6]
        let perm_b0 = _mm256_setr_epi32(1, 4, 7, 0, 0, 0, 0, 0);
        let perm_b1 = _mm256_setr_epi32(0, 0, 0, 2, 5, 0, 0, 0);
        let perm_b2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 3, 6);
        let b0 = _mm256_permutevar8x32_ps(in0, perm_b0);
        let b1 = _mm256_permutevar8x32_ps(in1, perm_b1);
        let b2 = _mm256_permutevar8x32_ps(in2, perm_b2);
        let b_out = _mm256_blend_ps::<0b00011000>(b0, b1); // positions 3,4 from b1
        let b_out = _mm256_blend_ps::<0b11100000>(b_out, b2); // positions 5,6,7 from b2

        // c: c0=in0[2], c1=in0[5], c2=in1[0], c3=in1[3], c4=in1[6], c5=in2[1], c6=in2[4], c7=in2[7]
        let perm_c0 = _mm256_setr_epi32(2, 5, 0, 0, 0, 0, 0, 0);
        let perm_c1 = _mm256_setr_epi32(0, 0, 0, 3, 6, 0, 0, 0);
        let perm_c2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 4, 7);
        let c0 = _mm256_permutevar8x32_ps(in0, perm_c0);
        let c1 = _mm256_permutevar8x32_ps(in1, perm_c1);
        let c2 = _mm256_permutevar8x32_ps(in2, perm_c2);
        let c_out = _mm256_blend_ps::<0b00011100>(c0, c1); // positions 2,3,4 from c1
        let c_out = _mm256_blend_ps::<0b11100000>(c_out, c2); // positions 5,6,7 from c2

        (Self(a_out, d), Self(b_out, d), Self(c_out, d))
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        assert!(src.len() >= 4 * F32VecAvx::LEN);
        // Input: [a0,b0,c0,d0, a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3, ...]
        // Output: a = [a0..a7], b = [b0..b7], c = [c0..c7], d = [d0..d7]
        let [s0, s1, s2, s3, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1, in2, in3) = (_mm256_loadu_ps(s0), _mm256_loadu_ps(s1), _mm256_loadu_ps(s2), _mm256_loadu_ps(s3));

        // This is essentially an 8x4 to 4x8 transpose
        // First, unpack pairs
        let t0 = _mm256_unpacklo_ps(in0, in1); // [a0,a2,b0,b2, a1,a3,b1,b3]
        let t1 = _mm256_unpackhi_ps(in0, in1); // [c0,c2,d0,d2, c1,c3,d1,d3]
        let t2 = _mm256_unpacklo_ps(in2, in3); // [a4,a6,b4,b6, a5,a7,b5,b7]
        let t3 = _mm256_unpackhi_ps(in2, in3); // [c4,c6,d4,d6, c5,c7,d5,d7]

        // Second level unpack
        let u0 = _mm256_unpacklo_ps(t0, t2); // [a0,a4,a2,a6, a1,a5,a3,a7]
        let u1 = _mm256_unpackhi_ps(t0, t2); // [b0,b4,b2,b6, b1,b5,b3,b7]
        let u2 = _mm256_unpacklo_ps(t1, t3); // [c0,c4,c2,c6, c1,c5,c3,c7]
        let u3 = _mm256_unpackhi_ps(t1, t3); // [d0,d4,d2,d6, d1,d5,d3,d7]

        // Permute to get correct order
        let perm = _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7);
        let a = _mm256_permutevar8x32_ps(u0, perm);
        let b = _mm256_permutevar8x32_ps(u1, perm);
        let c = _mm256_permutevar8x32_ps(u2, perm);
        let dv = _mm256_permutevar8x32_ps(u3, perm);

        (Self(a, d), Self(b, d), Self(c, d), Self(dv, d))
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_add(self, mul: F32VecAvx, add: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_fmadd_ps(self.0, mul.0, add.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn neg_mul_add(self, mul: F32VecAvx, add: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_fnmadd_ps(self.0, mul.0, add.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(_mm256_set1_ps(v), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(_mm256_setzero_ps(), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn abs(self) -> F32VecAvx {
        F32VecAvx(_mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn floor(self) -> F32VecAvx {
        F32VecAvx(_mm256_floor_ps(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn sqrt(self) -> F32VecAvx {
        F32VecAvx(_mm256_sqrt_ps(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn neg(self) -> F32VecAvx {
        F32VecAvx(_mm256_xor_ps(_mm256_set1_ps(-0.0), self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn copysign(self, sign: F32VecAvx) -> F32VecAvx {
        let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        F32VecAvx(
            _mm256_or_ps(
                _mm256_andnot_ps(sign_mask, self.0),
                _mm256_and_ps(sign_mask, sign.0),
            ),
            self.1,
        )
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn max(self, other: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_max_ps(self.0, other.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn min(self, other: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_min_ps(self.0, other.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn gt(self, other: F32VecAvx) -> MaskAvx {
        MaskAvx(_mm256_cmp_ps::<{ _CMP_GT_OQ }>(self.0, other.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn as_i32(self) -> I32VecAvx {
        I32VecAvx(_mm256_cvttps_epi32(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_to_i32(self) -> I32VecAvx {
        I32VecAvx(_mm256_castps_si256(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn prepare_table_bf16_8(d: AvxDescriptor, table: &[f32; 8]) -> Bf16Table8Avx {
        // For AVX2, vpermps is exact and fast, so we just load the table as-is
        // SAFETY: `table` has 8 elements, exactly as many as we load.
        Bf16Table8Avx(_mm256_loadu_ps(table), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn table_lookup_bf16_8(d: AvxDescriptor, table: Bf16Table8Avx, indices: I32VecAvx) -> Self {
        F32VecAvx(_mm256_permutevar8x32_ps(table.0, indices.0), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn round_store_u8(self, dest: &mut [u8]) {
        // Round to nearest integer
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0);
        // Convert to i32
        let i32s = _mm256_cvtps_epi32(rounded);
        // Extract 128-bit halves and pack
        let lo = _mm256_castsi256_si128(i32s);
        let hi = _mm256_extracti128_si256::<1>(i32s);
        // Pack 4+4 i32s to 8 u16s
        let u16s = _mm_packus_epi32(lo, hi);
        // Pack 8 u16s to 8 u8s (use same vector twice, take lower half)
        let u8s = _mm_packus_epi16(u16s, u16s);
        // Store lower 8 bytes
        _mm_storeu_si64(dest.first_chunk_mut::<8>().unwrap(), u8s);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn round_store_u16(self, dest: &mut [u16]) {
        // Round to nearest integer
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0);
        // Convert to i32
        let i32s = _mm256_cvtps_epi32(rounded);
        // Extract 128-bit halves and pack
        let lo = _mm256_castsi256_si128(i32s);
        let hi = _mm256_extracti128_si256::<1>(i32s);
        // Pack 4+4 i32s to 8 u16s
        let u16s = _mm_packus_epi32(lo, hi);
        // Store 8 u16s (16 bytes)
        _mm_storeu_si128(dest.first_chunk_mut().unwrap(), u16s);
    }

    impl_f32_array_interface!();

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        let bits = _mm_loadu_si128(mem.first_chunk::<8>().unwrap());
        F32VecAvx(_mm256_cvtph_ps(bits), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        let bits = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(self.0);
        _mm_storeu_si128(dest.first_chunk_mut::<8>().unwrap(), bits);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 7);

        let r0 = F32VecAvx::load_array(d, &data[0]).0;
        let r1 = F32VecAvx::load_array(d, &data[1 * stride]).0;
        let r2 = F32VecAvx::load_array(d, &data[2 * stride]).0;
        let r3 = F32VecAvx::load_array(d, &data[3 * stride]).0;
        let r4 = F32VecAvx::load_array(d, &data[4 * stride]).0;
        let r5 = F32VecAvx::load_array(d, &data[5 * stride]).0;
        let r6 = F32VecAvx::load_array(d, &data[6 * stride]).0;
        let r7 = F32VecAvx::load_array(d, &data[7 * stride]).0;

        let (c0, c1, c2, c3, c4, c5, c6, c7) = transpose_8x8_core(r0, r1, r2, r3, r4, r5, r6, r7);

        F32VecAvx(c0, d).store_array(&mut data[0]);
        F32VecAvx(c1, d).store_array(&mut data[1 * stride]);
        F32VecAvx(c2, d).store_array(&mut data[2 * stride]);
        F32VecAvx(c3, d).store_array(&mut data[3 * stride]);
        F32VecAvx(c4, d).store_array(&mut data[4 * stride]);
        F32VecAvx(c5, d).store_array(&mut data[5 * stride]);
        F32VecAvx(c6, d).store_array(&mut data[6 * stride]);
        F32VecAvx(c7, d).store_array(&mut data[7 * stride]);
    }
}

impl Add<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn add(self, rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_add_ps(self.0, rhs.0), self.1)
    }
}

impl Sub<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub(self, rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_sub_ps(self.0, rhs.0), self.1)
    }
}

impl Mul<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul(self, rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_mul_ps(self.0, rhs.0), self.1)
    }
}

impl Div<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn div(self, rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_div_ps(self.0, rhs.0), self.1)
    }
}

impl AddAssign<F32VecAvx> for F32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn add_assign(&mut self, rhs: F32VecAvx) {
        self.0 = _mm256_add_ps(self.0, rhs.0);
    }
}

impl SubAssign<F32VecAvx> for F32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub_assign(&mut self, rhs: F32VecAvx) {
        self.0 = _mm256_sub_ps(self.0, rhs.0);
    }
}

impl MulAssign<F32VecAvx> for F32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_assign(&mut self, rhs: F32VecAvx) {
        self.0 = _mm256_mul_ps(self.0, rhs.0);
    }
}

impl DivAssign<F32VecAvx> for F32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn div_assign(&mut self, rhs: F32VecAvx) {
        self.0 = _mm256_div_ps(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecAvx(__m256i, AvxDescriptor);

impl I32SimdVec for I32VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 8;

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        Self(_mm256_loadu_si256(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_from_i16(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(_mm256_cvtepi16_epi32(_mm_loadu_si128(mem.first_chunk::<8>().unwrap())), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load_from_u16(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(_mm256_cvtepu16_epi32(_mm_loadu_si128(mem.first_chunk::<8>().unwrap())), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [i32]) {
        _mm256_storeu_si256(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(_mm256_set1_epi32(v), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn as_f32(self) -> F32VecAvx {
        F32VecAvx(_mm256_cvtepi32_ps(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_to_f32(self) -> F32VecAvx {
        F32VecAvx(_mm256_castsi256_ps(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_to_u32(self) -> U32VecAvx {
        U32VecAvx(self.0, self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn abs(self) -> I32VecAvx {
        I32VecAvx(_mm256_abs_epi32(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn gt(self, rhs: I32VecAvx) -> MaskAvx {
        MaskAvx(
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(self.0, rhs.0)),
            self.1,
        )
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn lt_zero(self) -> MaskAvx {
        I32VecAvx(_mm256_setzero_si256(), self.1).gt(self)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn eq(self, rhs: I32VecAvx) -> MaskAvx {
        MaskAvx(
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(self.0, rhs.0)),
            self.1,
        )
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn eq_zero(self) -> MaskAvx {
        self.eq(I32VecAvx(_mm256_setzero_si256(), self.1))
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        I32VecAvx(_mm256_slli_epi32::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        I32VecAvx(_mm256_srai_epi32::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_wide_take_high(self, rhs: I32VecAvx) -> I32VecAvx {
        let l = _mm256_mul_epi32(self.0, rhs.0);
        let h = _mm256_mul_epi32(
            _mm256_srli_epi64::<32>(self.0),
            _mm256_srli_epi64::<32>(rhs.0),
        );
        let p0 = _mm256_unpacklo_epi32(l, h);
        let p1 = _mm256_unpackhi_epi32(l, h);
        I32VecAvx(_mm256_unpackhi_epi64(p0, p1), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_u16(self, dest: &mut [u16]) {
        let tmp = _mm256_shuffle_epi8(
            self.0,
            _mm256_setr_epi8(
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, //
                0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
            ),
        );
        let tmp = _mm256_permute4x64_epi64(tmp, 0xD8);
        _mm_storeu_si128(dest.first_chunk_mut::<8>().unwrap(), _mm256_extracti128_si256::<0>(tmp));
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_u8(self, dest: &mut [u8]) {
        let tmp = _mm256_shuffle_epi8(
            self.0,
            _mm256_setr_epi8(
                0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
                0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ),
        );
        let lo = _mm256_castsi256_si128(tmp);
        let hi = _mm256_extracti128_si256::<1>(tmp);
        let packed = _mm_unpacklo_epi32(lo, hi);
        _mm_storeu_si64(dest.first_chunk_mut::<8>().unwrap(), packed);
    }
}

impl Add<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn add(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_add_epi32(self.0, rhs.0), self.1)
    }
}

impl Sub<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_sub_epi32(self.0, rhs.0), self.1)
    }
}

impl Mul<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_mullo_epi32(self.0, rhs.0), self.1)
    }
}

impl Shl<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn shl(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_sllv_epi32(self.0, rhs.0), self.1)
    }
}

impl Shr<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn shr(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_srav_epi32(self.0, rhs.0), self.1)
    }
}

impl Neg for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn neg(self) -> I32VecAvx {
        I32VecAvx(_mm256_setzero_si256(), self.1) - self
    }
}

impl BitAnd<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_and_si256(self.0, rhs.0), self.1)
    }
}

impl BitOr<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_or_si256(self.0, rhs.0), self.1)
    }
}

impl BitXor<I32VecAvx> for I32VecAvx {
    type Output = I32VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor(self, rhs: I32VecAvx) -> I32VecAvx {
        I32VecAvx(_mm256_xor_si256(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn add_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_add_epi32(self.0, rhs.0);
    }
}

impl SubAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_sub_epi32(self.0, rhs.0);
    }
}

impl MulAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_mullo_epi32(self.0, rhs.0);
    }
}

impl ShlAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn shl_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_sllv_epi32(self.0, rhs.0);
    }
}

impl ShrAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn shr_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_srav_epi32(self.0, rhs.0);
    }
}

impl BitAndAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_and_si256(self.0, rhs.0);
    }
}

impl BitOrAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_or_si256(self.0, rhs.0);
    }
}

impl BitXorAssign<I32VecAvx> for I32VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor_assign(&mut self, rhs: I32VecAvx) {
        self.0 = _mm256_xor_si256(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecAvx(__m256i, AvxDescriptor);

impl I16SimdVec for I16VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 16;

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(_mm256_loadu_si256(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [i16]) {
        _mm256_storeu_si256(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: i16) -> Self {
        Self(_mm256_set1_epi16(v), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(_mm256_setzero_si256(), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn abs(self) -> I16VecAvx {
        I16VecAvx(_mm256_abs_epi16(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn gt(self, rhs: I16VecAvx) -> Mask16Avx {
        Mask16Avx(_mm256_cmpgt_epi16(self.0, rhs.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn lt_zero(self) -> Mask16Avx {
        I16VecAvx(_mm256_setzero_si256(), self.1).gt(self)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn eq(self, rhs: I16VecAvx) -> Mask16Avx {
        Mask16Avx(_mm256_cmpeq_epi16(self.0, rhs.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn eq_zero(self) -> Mask16Avx {
        self.eq(I16VecAvx(_mm256_setzero_si256(), self.1))
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm256_slli_epi16::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm256_srai_epi16::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_wide_take_high(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_mulhi_epi16(self.0, rhs.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_u16(self) -> U16VecAvx {
        U16VecAvx(self.0, self.1)
    }

    impl_i16_array_interface!();

    #[inline(always)]
    #[target_feature(from_args)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 15);

        #[target_feature(enable = "avx2")]
        #[inline]
        fn transpose_8x8_inlane(
            r0: __m256i,
            r1: __m256i,
            r2: __m256i,
            r3: __m256i,
            r4: __m256i,
            r5: __m256i,
            r6: __m256i,
            r7: __m256i,
        ) -> (
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
        ) {
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

            (s0, s1, s2, s3, s4, s5, s6, s7)
        }

        let r0 = I16VecAvx::load_array(d, &data[0]).0;
        let r1 = I16VecAvx::load_array(d, &data[1 * stride]).0;
        let r2 = I16VecAvx::load_array(d, &data[2 * stride]).0;
        let r3 = I16VecAvx::load_array(d, &data[3 * stride]).0;
        let r4 = I16VecAvx::load_array(d, &data[4 * stride]).0;
        let r5 = I16VecAvx::load_array(d, &data[5 * stride]).0;
        let r6 = I16VecAvx::load_array(d, &data[6 * stride]).0;
        let r7 = I16VecAvx::load_array(d, &data[7 * stride]).0;
        let r8 = I16VecAvx::load_array(d, &data[8 * stride]).0;
        let r9 = I16VecAvx::load_array(d, &data[9 * stride]).0;
        let r10 = I16VecAvx::load_array(d, &data[10 * stride]).0;
        let r11 = I16VecAvx::load_array(d, &data[11 * stride]).0;
        let r12 = I16VecAvx::load_array(d, &data[12 * stride]).0;
        let r13 = I16VecAvx::load_array(d, &data[13 * stride]).0;
        let r14 = I16VecAvx::load_array(d, &data[14 * stride]).0;
        let r15 = I16VecAvx::load_array(d, &data[15 * stride]).0;

        let (t0, t1, t2, t3, t4, t5, t6, t7) = transpose_8x8_inlane(r0, r1, r2, r3, r4, r5, r6, r7);
        let (b0, b1, b2, b3, b4, b5, b6, b7) =
            transpose_8x8_inlane(r8, r9, r10, r11, r12, r13, r14, r15);

        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t0, b0), d).store_array(&mut data[0]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t1, b1), d).store_array(&mut data[1 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t2, b2), d).store_array(&mut data[2 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t3, b3), d).store_array(&mut data[3 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t4, b4), d).store_array(&mut data[4 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t5, b5), d).store_array(&mut data[5 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t6, b6), d).store_array(&mut data[6 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x20>(t7, b7), d).store_array(&mut data[7 * stride]);

        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t0, b0), d).store_array(&mut data[8 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t1, b1), d).store_array(&mut data[9 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t2, b2), d).store_array(&mut data[10 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t3, b3), d).store_array(&mut data[11 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t4, b4), d).store_array(&mut data[12 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t5, b5), d).store_array(&mut data[13 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t6, b6), d).store_array(&mut data[14 * stride]);
        I16VecAvx(_mm256_permute2x128_si256::<0x31>(t7, b7), d).store_array(&mut data[15 * stride]);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_u8(self, dest: &mut [u8]) {
        let shuffle_mask = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, //
            0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let tmp = _mm256_shuffle_epi8(self.0, shuffle_mask);
        let lo = _mm256_castsi256_si128(tmp);
        let hi = _mm256_extracti128_si256::<1>(tmp);
        let packed = _mm_unpacklo_epi64(lo, hi);
        _mm_storeu_si128(dest.first_chunk_mut().unwrap(), packed);
    }
}

impl Add<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn add(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_add_epi16(self.0, rhs.0), self.1)
    }
}

impl Sub<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_sub_epi16(self.0, rhs.0), self.1)
    }
}

impl Neg for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn neg(self) -> I16VecAvx {
        I16VecAvx(_mm256_sub_epi16(_mm256_setzero_si256(), self.0), self.1)
    }
}

impl BitAnd<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_and_si256(self.0, rhs.0), self.1)
    }
}

impl BitOr<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_or_si256(self.0, rhs.0), self.1)
    }
}

impl BitXor<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_xor_si256(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn add_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_add_epi16(self.0, rhs.0);
    }
}

impl SubAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn sub_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_sub_epi16(self.0, rhs.0);
    }
}

impl BitAndAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_and_si256(self.0, rhs.0);
    }
}

impl BitOrAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_or_si256(self.0, rhs.0);
    }
}

impl BitXorAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_xor_si256(self.0, rhs.0);
    }
}

impl Mul<I16VecAvx> for I16VecAvx {
    type Output = I16VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul(self, rhs: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_mullo_epi16(self.0, rhs.0), self.1)
    }
}

impl MulAssign<I16VecAvx> for I16VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn mul_assign(&mut self, rhs: I16VecAvx) {
        self.0 = _mm256_mullo_epi16(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecAvx(__m256i, AvxDescriptor);

impl U32SimdVec for U32VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 8;

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_to_i32(self) -> I32VecAvx {
        I32VecAvx(self.0, self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm256_srli_epi32::<AMOUNT_I>(self.0), self.1)
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecAvx(pub(crate) __m256i, pub(crate) AvxDescriptor);

impl U64SimdVec for U64VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: u64) -> Self {
        Self(_mm256_set1_epi64x(v as i64), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[u64]) -> Self {
        Self(_mm256_loadu_si256(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [u64]) {
        _mm256_storeu_si256(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm256_slli_epi64::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm256_srli_epi64::<AMOUNT_I>(self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_to_u32(self) -> U32VecAvx {
        U32VecAvx(self.0, self.1)
    }
}

impl Add<U64VecAvx> for U64VecAvx {
    type Output = U64VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn add(self, rhs: U64VecAvx) -> U64VecAvx {
        U64VecAvx(_mm256_add_epi64(self.0, rhs.0), self.1)
    }
}

impl AddAssign<U64VecAvx> for U64VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn add_assign(&mut self, rhs: U64VecAvx) {
        self.0 = _mm256_add_epi64(self.0, rhs.0);
    }
}

impl BitAnd<U64VecAvx> for U64VecAvx {
    type Output = U64VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand(self, rhs: U64VecAvx) -> U64VecAvx {
        U64VecAvx(_mm256_and_si256(self.0, rhs.0), self.1)
    }
}

impl BitAndAssign<U64VecAvx> for U64VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand_assign(&mut self, rhs: U64VecAvx) {
        self.0 = _mm256_and_si256(self.0, rhs.0);
    }
}

impl BitOr<U64VecAvx> for U64VecAvx {
    type Output = U64VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor(self, rhs: U64VecAvx) -> U64VecAvx {
        U64VecAvx(_mm256_or_si256(self.0, rhs.0), self.1)
    }
}

impl BitOrAssign<U64VecAvx> for U64VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor_assign(&mut self, rhs: U64VecAvx) {
        self.0 = _mm256_or_si256(self.0, rhs.0);
    }
}

impl BitXor<U64VecAvx> for U64VecAvx {
    type Output = U64VecAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor(self, rhs: U64VecAvx) -> U64VecAvx {
        U64VecAvx(_mm256_xor_si256(self.0, rhs.0), self.1)
    }
}

impl BitXorAssign<U64VecAvx> for U64VecAvx {
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitxor_assign(&mut self, rhs: U64VecAvx) {
        self.0 = _mm256_xor_si256(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecAvx(__m256i, AvxDescriptor);

impl U8SimdVec for U8VecAvx {
    type Descriptor = AvxDescriptor;
    const LEN: usize = 32;

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        Self(_mm256_loadu_si256(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(_mm256_set1_epi8(v as i8), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [u8]) {
        _mm256_storeu_si256(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 2 * U8VecAvx::LEN);
        // a = [A0..A15 | A16..A31]
        // b = [B0..B15 | B16..B31]
        let lo = _mm256_unpacklo_epi8(a.0, b.0); // [A0 B0..A7 B7 | A16 B16..A23 B23]
        let hi = _mm256_unpackhi_epi8(a.0, b.0); // [A8 B8..A15 B15 | A24 B24..A31 B31]

        // R0 = [A0 B0..A7 B7 | A8 B8..A15 B15]
        let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
        // R1 = [A16 B16..A23 B23 | A24 B24..A31 B31]
        let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);

        let [d0, d1, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 3 * U8VecAvx::LEN);

        // U8 Masks
        let mask_a0 = _mm256_setr_epi8(
            0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7, -1, -1,
            8, -1, -1, 9, -1, -1, 10, -1,
        );
        let mask_a1 = _mm256_setr_epi8(
            -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1, 0, -1, -1, 1, -1, -1,
            2, -1, -1, 3, -1, -1, 4, -1, -1, 5,
        );
        let mask_a2 = _mm256_setr_epi8(
            -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1, -1, 11, -1, -1, 12, -1, -1,
            13, -1, -1, 14, -1, -1, 15, -1, -1,
        );
        let mask_b0 = _mm256_setr_epi8(
            -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7, -1,
            -1, 8, -1, -1, 9, -1, -1, 10,
        );
        let mask_b1 = _mm256_setr_epi8(
            -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1, 0, -1, -1, 1, -1,
            -1, 2, -1, -1, 3, -1, -1, 4, -1, -1,
        );
        let mask_b2 = _mm256_setr_epi8(
            5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1, -1, 11, -1, -1, 12, -1,
            -1, 13, -1, -1, 14, -1, -1, 15, -1,
        );
        let mask_c0 = _mm256_setr_epi8(
            -1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, 7,
            -1, -1, 8, -1, -1, 9, -1, -1,
        );
        let mask_c1 = _mm256_setr_epi8(
            10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1, 0, -1, -1, 1,
            -1, -1, 2, -1, -1, 3, -1, -1, 4, -1,
        );
        let mask_c2 = _mm256_setr_epi8(
            -1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1, -1, 11, -1, -1, 12,
            -1, -1, 13, -1, -1, 14, -1, -1, 15,
        );

        // Create duplicated vectors for lane swizzling
        let a_dup_lo = _mm256_permute2x128_si256::<0x00>(a.0, a.0);
        let b_dup_lo = _mm256_permute2x128_si256::<0x00>(b.0, b.0);
        let c_dup_lo = _mm256_permute2x128_si256::<0x00>(c.0, c.0);

        let a_dup_hi = _mm256_permute2x128_si256::<0x11>(a.0, a.0);
        let b_dup_hi = _mm256_permute2x128_si256::<0x11>(b.0, b.0);
        let c_dup_hi = _mm256_permute2x128_si256::<0x11>(c.0, c.0);

        let out0 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a_dup_lo, mask_a0),
                _mm256_shuffle_epi8(b_dup_lo, mask_b0),
            ),
            _mm256_shuffle_epi8(c_dup_lo, mask_c0),
        );

        let out1 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a.0, mask_a1),
                _mm256_shuffle_epi8(b.0, mask_b1),
            ),
            _mm256_shuffle_epi8(c.0, mask_c1),
        );

        let out2 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a_dup_hi, mask_a2),
                _mm256_shuffle_epi8(b_dup_hi, mask_b2),
            ),
            _mm256_shuffle_epi8(c_dup_hi, mask_c2),
        );

        let [d0, d1, d2, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
        _mm256_storeu_si256(d2, out2);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 4 * U8VecAvx::LEN);
        // First interleave pairs: ab and cd
        let ab_lo = _mm256_unpacklo_epi8(a.0, b.0);
        let ab_hi = _mm256_unpackhi_epi8(a.0, b.0);
        let cd_lo = _mm256_unpacklo_epi8(c.0, d.0);
        let cd_hi = _mm256_unpackhi_epi8(c.0, d.0);

        // Then interleave the pairs to get 4-byte chunks
        let out0_p = _mm256_unpacklo_epi16(ab_lo, cd_lo);
        let out1_p = _mm256_unpackhi_epi16(ab_lo, cd_lo);
        let out2_p = _mm256_unpacklo_epi16(ab_hi, cd_hi);
        let out3_p = _mm256_unpackhi_epi16(ab_hi, cd_hi);

        // Reorder lanes
        let out0 = _mm256_permute2x128_si256::<0x20>(out0_p, out1_p);
        let out1 = _mm256_permute2x128_si256::<0x20>(out2_p, out3_p);
        let out2 = _mm256_permute2x128_si256::<0x31>(out0_p, out1_p);
        let out3 = _mm256_permute2x128_si256::<0x31>(out2_p, out3_p);

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
        _mm256_storeu_si256(d2, out2);
        _mm256_storeu_si256(d3, out3);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecAvx(__m256i, AvxDescriptor);

impl U16SimdVec for U16VecAvx {
    type Descriptor = AvxDescriptor;
    const LEN: usize = 16;

    #[inline(always)]
    #[target_feature(from_args)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(_mm256_loadu_si256(mem.first_chunk().unwrap()), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(_mm256_set1_epi16(v as i16), d)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store(&self, mem: &mut [u16]) {
        _mm256_storeu_si256(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 2 * U16VecAvx::LEN);
        // a = [A0..A7 | A8..A15]
        // b = [B0..B7 | B8..B15]
        let lo = _mm256_unpacklo_epi16(a.0, b.0); // [A0 B0..A3 B3 | A8 B8..A11 B11]
        let hi = _mm256_unpackhi_epi16(a.0, b.0); // [A4 B4..A7 B7 | A12 B12..A15 B15]

        // R0 = [A0 B0..A7 B7]
        let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
        // R1 = [A8 B8..A15 B15]
        let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);

        let [d0, d1, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 3 * U16VecAvx::LEN);

        // U16 Masks
        let mask_a0 = _mm256_setr_epi8(
            0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1,
            8, 9, -1, -1, -1, -1, 10, 11,
        );
        let mask_a1 = _mm256_setr_epi8(
            -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1,
            2, 3, -1, -1, -1, -1, 4, 5, -1, -1,
        );
        let mask_a2 = _mm256_setr_epi8(
            -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1,
            -1, -1, -1, 14, 15, -1, -1, -1, -1,
        );
        let mask_b0 = _mm256_setr_epi8(
            -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 6, 7, -1, -1,
            -1, -1, 8, 9, -1, -1, -1, -1,
        );
        let mask_b1 = _mm256_setr_epi8(
            10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, 0, 1, -1, -1,
            -1, -1, 2, 3, -1, -1, -1, -1, 4, 5,
        );
        let mask_b2 = _mm256_setr_epi8(
            -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11, -1, -1, -1, -1, 12,
            13, -1, -1, -1, -1, 14, 15, -1, -1,
        );
        let mask_c0 = _mm256_setr_epi8(
            -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 6, 7,
            -1, -1, -1, -1, 8, 9, -1, -1,
        );
        let mask_c1 = _mm256_setr_epi8(
            -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, 0, 1,
            -1, -1, -1, -1, 2, 3, -1, -1, -1, -1,
        );
        let mask_c2 = _mm256_setr_epi8(
            4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11, -1, -1, -1,
            -1, 12, 13, -1, -1, -1, -1, 14, 15,
        );

        // Create duplicated vectors for lane swizzling
        let a_dup_lo = _mm256_permute2x128_si256::<0x00>(a.0, a.0);
        let b_dup_lo = _mm256_permute2x128_si256::<0x00>(b.0, b.0);
        let c_dup_lo = _mm256_permute2x128_si256::<0x00>(c.0, c.0);

        let a_dup_hi = _mm256_permute2x128_si256::<0x11>(a.0, a.0);
        let b_dup_hi = _mm256_permute2x128_si256::<0x11>(b.0, b.0);
        let c_dup_hi = _mm256_permute2x128_si256::<0x11>(c.0, c.0);

        let out0 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a_dup_lo, mask_a0),
                _mm256_shuffle_epi8(b_dup_lo, mask_b0),
            ),
            _mm256_shuffle_epi8(c_dup_lo, mask_c0),
        );

        let out1 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a.0, mask_a1),
                _mm256_shuffle_epi8(b.0, mask_b1),
            ),
            _mm256_shuffle_epi8(c.0, mask_c1),
        );

        let out2 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(a_dup_hi, mask_a2),
                _mm256_shuffle_epi8(b_dup_hi, mask_b2),
            ),
            _mm256_shuffle_epi8(c_dup_hi, mask_c2),
        );

        let [d0, d1, d2, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
        _mm256_storeu_si256(d2, out2);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 4 * U16VecAvx::LEN);
        // First interleave pairs: ab and cd
        let ab_lo = _mm256_unpacklo_epi16(a.0, b.0);
        let ab_hi = _mm256_unpackhi_epi16(a.0, b.0);
        let cd_lo = _mm256_unpacklo_epi16(c.0, d.0);
        let cd_hi = _mm256_unpackhi_epi16(c.0, d.0);

        // Then interleave the pairs to get 4-u16 chunks (8 bytes)
        let out0_p = _mm256_unpacklo_epi32(ab_lo, cd_lo);
        let out1_p = _mm256_unpackhi_epi32(ab_lo, cd_lo);
        let out2_p = _mm256_unpacklo_epi32(ab_hi, cd_hi);
        let out3_p = _mm256_unpackhi_epi32(ab_hi, cd_hi);

        // Reorder lanes
        let out0 = _mm256_permute2x128_si256::<0x20>(out0_p, out1_p);
        let out1 = _mm256_permute2x128_si256::<0x20>(out2_p, out3_p);
        let out2 = _mm256_permute2x128_si256::<0x31>(out0_p, out1_p);
        let out3 = _mm256_permute2x128_si256::<0x31>(out2_p, out3_p);

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm256_storeu_si256(d0, out0);
        _mm256_storeu_si256(d1, out1);
        _mm256_storeu_si256(d2, out2);
        _mm256_storeu_si256(d3, out3);
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn bitcast_i16(self) -> I16VecAvx {
        I16VecAvx(self.0, self.1)
    }
}

impl SimdMask for MaskAvx {
    type Descriptor = AvxDescriptor;

    #[inline(always)]
    #[target_feature(from_args)]
    fn if_then_else_f32(self, if_true: F32VecAvx, if_false: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_blendv_ps(if_false.0, if_true.0, self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn if_then_else_i32(self, if_true: I32VecAvx, if_false: I32VecAvx) -> I32VecAvx {
        I32VecAvx(
            _mm256_blendv_epi8(if_false.0, if_true.0, _mm256_castps_si256(self.0)),
            self.1,
        )
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn maskz_i32(self, v: I32VecAvx) -> I32VecAvx {
        I32VecAvx(
            _mm256_andnot_si256(_mm256_castps_si256(self.0), v.0),
            self.1,
        )
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn all(self) -> bool {
        _mm256_movemask_ps(self.0) == 0b11111111
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn andnot(self, rhs: MaskAvx) -> MaskAvx {
        MaskAvx(_mm256_andnot_ps(self.0, rhs.0), self.1)
    }
}

impl BitAnd<MaskAvx> for MaskAvx {
    type Output = MaskAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand(self, rhs: MaskAvx) -> MaskAvx {
        MaskAvx(_mm256_and_ps(self.0, rhs.0), self.1)
    }
}

impl BitOr<MaskAvx> for MaskAvx {
    type Output = MaskAvx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor(self, rhs: MaskAvx) -> MaskAvx {
        MaskAvx(_mm256_or_ps(self.0, rhs.0), self.1)
    }
}

impl SimdMask16 for Mask16Avx {
    type Descriptor = AvxDescriptor;

    #[inline(always)]
    #[target_feature(from_args)]
    fn if_then_else_i16(self, if_true: I16VecAvx, if_false: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_blendv_epi8(if_false.0, if_true.0, self.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn maskz_i16(self, v: I16VecAvx) -> I16VecAvx {
        I16VecAvx(_mm256_andnot_si256(self.0, v.0), self.1)
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn all(self) -> bool {
        _mm256_movemask_epi8(self.0) as u32 == 0xFFFF_FFFF
    }

    #[inline(always)]
    #[target_feature(from_args)]
    fn andnot(self, rhs: Mask16Avx) -> Mask16Avx {
        Mask16Avx(_mm256_andnot_si256(self.0, rhs.0), self.1)
    }
}

impl BitAnd<Mask16Avx> for Mask16Avx {
    type Output = Mask16Avx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitand(self, rhs: Mask16Avx) -> Mask16Avx {
        Mask16Avx(_mm256_and_si256(self.0, rhs.0), self.1)
    }
}

impl BitOr<Mask16Avx> for Mask16Avx {
    type Output = Mask16Avx;
    #[inline(always)]
    #[target_feature(from_args)]
    fn bitor(self, rhs: Mask16Avx) -> Mask16Avx {
        Mask16Avx(_mm256_or_si256(self.0, rhs.0), self.1)
    }
}
