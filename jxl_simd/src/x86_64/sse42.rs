// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::x86_64::*;
use super::unaligned::{_mm_loadl_epi64, _mm_loadu_ps, _mm_loadu_si128, _mm_storeu_ps, _mm_storeu_si32, _mm_storeu_si64, _mm_storeu_si128};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use crate::{U32SimdVec, U64SimdVec, impl_f32_array_interface, impl_i16_array_interface};

// Safety invariant: this type is only ever constructed if sse4.2 is available.
#[target_feature(enable = "sse4.2")]
#[derive(Clone, Copy, Debug)]
pub struct Sse42Descriptor(());

impl Sse42Descriptor {
    #[target_feature(enable = "sse4.2")]
    #[inline(always)]
    pub fn new_unchecked() -> Self {
        Self(())
    }
}

impl SimdDescriptor for Sse42Descriptor {
    type F32Vec = F32VecSse42;
    type I32Vec = I32VecSse42;
    type I16Vec = I16VecSse42;
    type U64Vec = U64VecSse42;
    type U32Vec = U32VecSse42;
    type U16Vec = U16VecSse42;
    type U8Vec = U8VecSse42;
    type Mask = MaskSse42;
    type Mask16 = Mask16Sse42;
    type Bf16Table8 = Bf16Table8Sse42;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[target_feature(from_args)]
    #[inline(always)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    #[allow(unsafe_code)]
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: we just checked sse4.2.
            Some(unsafe { Self(()) })
        } else {
            None
        }
    }
}

/// Prepared 8-entry BF16 lookup table for SSE4.2.
#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Sse42(__m128i, Sse42Descriptor);

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecSse42(__m128, Sse42Descriptor);

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskSse42(__m128, Sse42Descriptor);

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Sse42(__m128i, Sse42Descriptor);

impl F32SimdVec for F32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        Self(_mm_loadu_ps(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        _mm_storeu_ps(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 2 * Self::LEN);
        // a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
        // lo = [a0, b0, a1, b1], hi = [a2, b2, a3, b3]
        let lo = _mm_unpacklo_ps(a.0, b.0);
        let hi = _mm_unpackhi_ps(a.0, b.0);
        let [d0, d1, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm_storeu_ps(d0, lo);
        _mm_storeu_ps(d1, hi);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 3 * Self::LEN);
        // Input vectors:
        // a = [a0, a1, a2, a3]
        // b = [b0, b1, b2, b3]
        // c = [c0, c1, c2, c3]

        // Desired interleaved output stored in 3 __m128 registers:
        // out0 = [a0, b0, c0, a1]
        // out1 = [b1, c1, a2, b2]
        // out2 = [c2, a3, b3, c3]

        // Intermediate interleavings of input pairs
        let p_ab_lo = _mm_unpacklo_ps(a.0, b.0); // [a0, b0, a1, b1]
        let p_ab_hi = _mm_unpackhi_ps(a.0, b.0); // [a2, b2, a3, b3]

        let p_ca_lo = _mm_unpacklo_ps(c.0, a.0); // [c0, a0, c1, a1]
        let p_ca_hi = _mm_unpackhi_ps(c.0, a.0); // [c2, a2, c3, a3]

        let p_bc_hi = _mm_unpackhi_ps(b.0, c.0); // [b2, c2, b3, c3]

        // Construct out0 = [a0, b0, c0, a1]
        let out0 = _mm_shuffle_ps::<0xC4>(p_ab_lo, p_ca_lo);

        // Construct out1 = [b1, c1, a2, b2]
        let out1_tmp1 = _mm_shuffle_ps::<0xAF>(p_ab_lo, p_ca_lo); // [b1, b1, c1, c1]
        let out1 = _mm_shuffle_ps::<0x48>(out1_tmp1, p_ab_hi);

        // Construct out2 = [c2, a3, b3, c3]
        let out2 = _mm_shuffle_ps::<0xEC>(p_ca_hi, p_bc_hi);

        // Store the results
        let [d0, d1, d2, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm_storeu_ps(d0, out0);
        _mm_storeu_ps(d1, out1);
        _mm_storeu_ps(d2, out2);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 4 * Self::LEN);
        // First interleave pairs: ab and cd
        let ab_lo = _mm_unpacklo_ps(a.0, b.0); // [a0, b0, a1, b1]
        let ab_hi = _mm_unpackhi_ps(a.0, b.0); // [a2, b2, a3, b3]
        let cd_lo = _mm_unpacklo_ps(c.0, d.0); // [c0, d0, c1, d1]
        let cd_hi = _mm_unpackhi_ps(c.0, d.0); // [c2, d2, c3, d3]

        // Then interleave the pairs to get final layout
        let out0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo))); // [a0, b0, c0, d0]
        let out1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo))); // [a1, b1, c1, d1]
        let out2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi))); // [a2, b2, c2, d2]
        let out3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi))); // [a3, b3, c3, d3]

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm_storeu_ps(d0, out0);
        _mm_storeu_ps(d1, out1);
        _mm_storeu_ps(d2, out2);
        _mm_storeu_ps(d3, out3);
    }

    #[target_feature(from_args)]
    #[inline(always)]
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
        assert!(dest.len() >= 8 * Self::LEN);
        // For 4-wide vectors storing 8 interleaved, we need 32 elements output
        // Output: [a0,b0,c0,d0,e0,f0,g0,h0, a1,b1,c1,d1,e1,f1,g1,h1, ...]
        let ab_lo = _mm_unpacklo_ps(a.0, b.0);
        let ab_hi = _mm_unpackhi_ps(a.0, b.0);
        let cd_lo = _mm_unpacklo_ps(c.0, d.0);
        let cd_hi = _mm_unpackhi_ps(c.0, d.0);
        let ef_lo = _mm_unpacklo_ps(e.0, f.0);
        let ef_hi = _mm_unpackhi_ps(e.0, f.0);
        let gh_lo = _mm_unpacklo_ps(g.0, h.0);
        let gh_hi = _mm_unpackhi_ps(g.0, h.0);

        let abcd_0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
        let abcd_1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
        let abcd_2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));
        let abcd_3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));
        let efgh_0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ef_lo), _mm_castps_pd(gh_lo)));
        let efgh_1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ef_lo), _mm_castps_pd(gh_lo)));
        let efgh_2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ef_hi), _mm_castps_pd(gh_hi)));
        let efgh_3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ef_hi), _mm_castps_pd(gh_hi)));

        let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut().0 else { unreachable!() };
        _mm_storeu_ps(d0, abcd_0);
        _mm_storeu_ps(d1, efgh_0);
        _mm_storeu_ps(d2, abcd_1);
        _mm_storeu_ps(d3, efgh_1);
        _mm_storeu_ps(d4, abcd_2);
        _mm_storeu_ps(d5, efgh_2);
        _mm_storeu_ps(d6, abcd_3);
        _mm_storeu_ps(d7, efgh_3);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        assert!(src.len() >= 2 * Self::LEN);
        // Input: [a0, b0, a1, b1, a2, b2, a3, b3]
        // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
        let [s0, s1, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1) = (_mm_loadu_ps(s0), _mm_loadu_ps(s1));

        // Shuffle to separate a and b components
        let a = _mm_shuffle_ps::<0x88>(in0, in1); // [a0, a1, a2, a3]
        let b = _mm_shuffle_ps::<0xDD>(in0, in1); // [b0, b1, b2, b3]

        (Self(a, d), Self(b, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        assert!(src.len() >= 3 * Self::LEN);
        // Input: [a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3]
        // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3]

        let [s0, s1, s2, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1, in2) = (_mm_loadu_ps(s0), _mm_loadu_ps(s1), _mm_loadu_ps(s2));

        // Extract using shuffles.
        // _mm_shuffle_ps(a, b, imm8): result[0:1] from a, result[2:3] from b
        // imm8 bits: [1:0]=A, [3:2]=B select from a; [5:4]=C, [7:6]=D select from b
        //
        // Element positions in input:
        // a: a0=in0[0], a1=in0[3], a2=in1[2], a3=in2[1]
        // b: b0=in0[1], b1=in1[0], b2=in1[3], b3=in2[2]
        // c: c0=in0[2], c1=in1[1], c2=in2[0], c3=in2[3]

        // Channel a: gather a0,a1 and a2,a3, then combine
        let a_lo = _mm_shuffle_ps::<0xC0>(in0, in0); // [a0, a0, a0, a1]
        let a_hi = _mm_shuffle_ps::<0x98>(in1, in2); // [b1, a2, a3, b3]
        let a = _mm_shuffle_ps::<0x9C>(a_lo, a_hi); // [a0, a1, a2, a3]

        // Channel b: gather b0,b1 and b2,b3, then combine
        let b_lo = _mm_shuffle_ps::<0x01>(in0, in1); // [b0, a0, b1, b1]
        let b_hi = _mm_shuffle_ps::<0x2C>(in1, in2); // [b1, b2, b3, c2]
        let b = _mm_shuffle_ps::<0x98>(b_lo, b_hi); // [b0, b1, b2, b3]

        // Channel c: gather c0,c1 and c2,c3, then combine
        let c_lo = _mm_shuffle_ps::<0x12>(in0, in1); // [c0, a0, c1, b1]
        let c_hi = _mm_shuffle_ps::<0x30>(in2, in2); // [c2, c2, c3, c2]
        let c = _mm_shuffle_ps::<0x98>(c_lo, c_hi); // [c0, c1, c2, c3]

        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        assert!(src.len() >= 4 * Self::LEN);
        // Input: [a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3]
        // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3], d = [d0, d1, d2, d3]
        let [s0, s1, s2, s3, ..] = src.as_chunks().0 else { unreachable!() };
        let (in0, in1, in2, in3) = (_mm_loadu_ps(s0), _mm_loadu_ps(s1), _mm_loadu_ps(s2), _mm_loadu_ps(s3));

        // This is effectively a 4x4 matrix transpose
        // First interleave pairs
        let t0 = _mm_unpacklo_ps(in0, in1); // [a0, a1, b0, b1]
        let t1 = _mm_unpackhi_ps(in0, in1); // [c0, c1, d0, d1]
        let t2 = _mm_unpacklo_ps(in2, in3); // [a2, a3, b2, b3]
        let t3 = _mm_unpackhi_ps(in2, in3); // [c2, c3, d2, d3]

        // Then combine
        let a = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(t0), _mm_castps_pd(t2))); // [a0, a1, a2, a3]
        let b = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(t0), _mm_castps_pd(t2))); // [b0, b1, b2, b3]
        let c = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(t1), _mm_castps_pd(t3))); // [c0, c1, c2, c3]
        let dv = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(t1), _mm_castps_pd(t3))); // [d0, d1, d2, d3]

        (Self(a, d), Self(b, d), Self(c, d), Self(dv, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        self * mul + add
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        add - self * mul
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(_mm_set1_ps(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(_mm_setzero_ps(), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> Self {
        Self(
            _mm_castsi128_ps(_mm_andnot_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(self.0),
            )),
            self.1,
        )
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn floor(self) -> Self {
        Self(_mm_floor_ps(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn sqrt(self) -> Self {
        Self(_mm_sqrt_ps(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> Self {
        Self(
            _mm_castsi128_ps(_mm_xor_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(self.0),
            )),
            self.1,
        )
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
        Self(
            _mm_or_ps(
                _mm_andnot_ps(sign_mask, self.0),
                _mm_and_ps(sign_mask, sign.0),
            ),
            self.1,
        )
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(_mm_max_ps(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self(_mm_min_ps(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, other: Self) -> MaskSse42 {
        MaskSse42(_mm_cmpgt_ps(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn as_i32(self) -> I32VecSse42 {
        I32VecSse42(_mm_cvttps_epi32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSse42 {
        I32VecSse42(_mm_castps_si128(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn prepare_table_bf16_8(d: Sse42Descriptor, table: &[f32; 8]) -> Bf16Table8Sse42 {
        // Convert f32 table to BF16 packed in 128 bits (16 bytes for 8 entries)
        // BF16 is the high 16 bits of f32
        let ([t0, t1], _) = table.as_chunks() else {
            unreachable!()
        };
        let (table_lo, table_hi) = (_mm_loadu_ps(t0), _mm_loadu_ps(t1));
        let table_lo_i32 = _mm_castps_si128(table_lo);
        let table_hi_i32 = _mm_castps_si128(table_hi);

        // Extract high 16 bits (bf16) from each f32 using shuffle
        // f32 bytes: [b0, b1, b2, b3] -> bf16 bytes: [b2, b3]
        let bf16_extract =
            _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
        let bf16_lo = _mm_shuffle_epi8(table_lo_i32, bf16_extract);
        let bf16_hi = _mm_shuffle_epi8(table_hi_i32, bf16_extract);
        // Combine: bf16_lo has bytes 0-7, bf16_hi has bytes 0-7
        // Result: [bf16_0..bf16_3, bf16_4..bf16_7]
        Bf16Table8Sse42(_mm_unpacklo_epi64(bf16_lo, bf16_hi), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn table_lookup_bf16_8(
        d: Sse42Descriptor,
        table: Bf16Table8Sse42,
        indices: I32VecSse42,
    ) -> Self {
        // Build shuffle mask efficiently using arithmetic on 32-bit indices.
        // For each index i (0-7), we need to select bytes [2*i, 2*i+1] from bf16_table
        // and place them in the high 16 bits of each 32-bit f32 lane (bytes 2,3),
        // with bytes 0,1 set to zero (using 0x80 which gives 0 in pshufb).
        //
        // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
        // As a 32-bit value: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
        //                  = 0x8080 | (i << 17) | (i << 25) | (1 << 24)
        //                  = (i << 17) | (i << 25) | 0x01008080
        let shl17 = _mm_slli_epi32::<17>(indices.0);
        let shl25 = _mm_slli_epi32::<25>(indices.0);
        let base = _mm_set1_epi32(0x01008080u32 as i32);
        let shuffle_mask = _mm_or_si128(_mm_or_si128(shl17, shl25), base);

        // Shuffle the bf16 table to get the values
        let result = _mm_shuffle_epi8(table.0, shuffle_mask);

        // Result has bf16 in high 16 bits of each 32-bit lane = valid f32
        Self(_mm_castsi128_ps(result), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        // Round to nearest integer
        let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0);
        // Convert to i32
        let i32s = _mm_cvtps_epi32(rounded);
        // Pack i32 -> u16 -> u8 (use same vector twice, take lower half each time)
        let u16s = _mm_packus_epi32(i32s, i32s);
        let u8s = _mm_packus_epi16(u16s, u16s);
        // Store lower 4 bytes
        _mm_storeu_si32(dest.first_chunk_mut::<4>().unwrap(), u8s);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        // Round to nearest integer
        let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0);
        // Convert to i32
        let i32s = _mm_cvtps_epi32(rounded);
        // Pack i32 -> u16 (use same vector twice, take lower half)
        let u16s = _mm_packus_epi32(i32s, i32s);
        // Store lower 8 bytes (4 u16s)
        _mm_storeu_si64(dest.first_chunk_mut::<4>().unwrap(), u16s);
    }

    impl_f32_array_interface!();

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        // TODO: use SIMD bit manipulation instead of scalar loop (see wasm32/simd128.rs)
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            result[i] = crate::f16::from_bits(mem[i]).to_f32();
        }
        Self::load(d, &result)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        // TODO: use SIMD bit manipulation instead of scalar loop (see wasm32/simd128.rs)
        let mut tmp = [0.0f32; 4];
        self.store(&mut tmp);
        for i in 0..4 {
            dest[i] = crate::f16::from_f32(tmp[i]).to_bits();
        }
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 3);

        let p0 = Self::load_array(d, &data[0]).0;
        let p1 = Self::load_array(d, &data[1 * stride]).0;
        let p2 = Self::load_array(d, &data[2 * stride]).0;
        let p3 = Self::load_array(d, &data[3 * stride]).0;

        let q0 = _mm_unpacklo_ps(p0, p2);
        let q1 = _mm_unpacklo_ps(p1, p3);
        let q2 = _mm_unpackhi_ps(p0, p2);
        let q3 = _mm_unpackhi_ps(p1, p3);

        let r0 = _mm_unpacklo_ps(q0, q1);
        let r1 = _mm_unpackhi_ps(q0, q1);
        let r2 = _mm_unpacklo_ps(q2, q3);
        let r3 = _mm_unpackhi_ps(q2, q3);

        Self(r0, d).store_array(&mut data[0]);
        Self(r1, d).store_array(&mut data[1 * stride]);
        Self(r2, d).store_array(&mut data[2 * stride]);
        Self(r3, d).store_array(&mut data[3 * stride]);
    }
}

impl Add<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_add_ps(self.0, rhs.0), self.1)
    }
}

impl Sub<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_sub_ps(self.0, rhs.0), self.1)
    }
}

impl Mul<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_mul_ps(self.0, rhs.0), self.1)
    }
}

impl Div<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn div(self, rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_div_ps(self.0, rhs.0), self.1)
    }
}

impl AddAssign<F32VecSse42> for F32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: F32VecSse42) {
        self.0 = _mm_add_ps(self.0, rhs.0);
    }
}

impl SubAssign<F32VecSse42> for F32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: F32VecSse42) {
        self.0 = _mm_sub_ps(self.0, rhs.0);
    }
}

impl MulAssign<F32VecSse42> for F32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: F32VecSse42) {
        self.0 = _mm_mul_ps(self.0, rhs.0);
    }
}

impl DivAssign<F32VecSse42> for F32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn div_assign(&mut self, rhs: F32VecSse42) {
        self.0 = _mm_div_ps(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecSse42(__m128i, Sse42Descriptor);

impl I32SimdVec for I32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        Self(_mm_loadu_si128(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_from_i16(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(_mm_cvtepi16_epi32(_mm_loadl_epi64(mem.first_chunk::<4>().unwrap())), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_from_u16(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(_mm_cvtepu16_epi32(_mm_loadl_epi64(mem.first_chunk::<4>().unwrap())), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        _mm_storeu_si128(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(_mm_set1_epi32(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn as_f32(self) -> F32VecSse42 {
        F32VecSse42(_mm_cvtepi32_ps(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_f32(self) -> F32VecSse42 {
        F32VecSse42(_mm_castsi128_ps(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSse42 {
        U32VecSse42(self.0, self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> Self {
        Self(_mm_abs_epi32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, rhs: Self) -> MaskSse42 {
        MaskSse42(_mm_castsi128_ps(_mm_cmpgt_epi32(self.0, rhs.0)), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn lt_zero(self) -> MaskSse42 {
        Self(_mm_setzero_si128(), self.1).gt(self)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq(self, rhs: Self) -> MaskSse42 {
        MaskSse42(_mm_castsi128_ps(_mm_cmpeq_epi32(self.0, rhs.0)), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq_zero(self) -> MaskSse42 {
        self.eq(Self(_mm_setzero_si128(), self.1))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_slli_epi32::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_srai_epi32::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        let l = _mm_mul_epi32(self.0, rhs.0);
        let h = _mm_mul_epi32(_mm_srli_epi64::<32>(self.0), _mm_srli_epi64::<32>(rhs.0));
        let p0 = _mm_unpacklo_epi32(l, h);
        let p1 = _mm_unpackhi_epi32(l, h);
        Self(_mm_unpackhi_epi64(p0, p1), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        // Truncate i32 -> u16 using shuffle
        let shuffle_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
        let u16s = _mm_shuffle_epi8(self.0, shuffle_mask);
        _mm_storeu_si64(dest.first_chunk_mut::<4>().unwrap(), u16s);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        // Truncate i32 -> u8 using shuffle
        let shuffle_mask =
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let u8s = _mm_shuffle_epi8(self.0, shuffle_mask);
        // Store lower 4 bytes
        _mm_storeu_si32(dest.first_chunk_mut::<4>().unwrap(), u8s);
    }
}

impl Add<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_add_epi32(self.0, rhs.0), self.1)
    }
}

impl Sub<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_sub_epi32(self.0, rhs.0), self.1)
    }
}

impl Mul<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_mullo_epi32(self.0, rhs.0), self.1)
    }
}

impl Neg for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> I32VecSse42 {
        I32VecSse42(_mm_setzero_si128(), self.1) - self
    }
}

impl BitAnd<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_and_si128(self.0, rhs.0), self.1)
    }
}

impl BitOr<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_or_si128(self.0, rhs.0), self.1)
    }
}

impl BitXor<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_xor_si128(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_add_epi32(self.0, rhs.0);
    }
}

impl SubAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_sub_epi32(self.0, rhs.0);
    }
}

impl MulAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_mullo_epi32(self.0, rhs.0);
    }
}

impl BitAndAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_and_si128(self.0, rhs.0);
    }
}

impl BitOrAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_or_si128(self.0, rhs.0);
    }
}

impl BitXorAssign<I32VecSse42> for I32VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I32VecSse42) {
        self.0 = _mm_xor_si128(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecSse42(__m128i, Sse42Descriptor);

impl I16SimdVec for I16VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 8;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(_mm_loadu_si128(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [i16]) {
        _mm_storeu_si128(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i16) -> Self {
        Self(_mm_set1_epi16(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(_mm_setzero_si128(), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> Self {
        Self(_mm_abs_epi16(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, rhs: Self) -> Mask16Sse42 {
        Mask16Sse42(_mm_cmpgt_epi16(self.0, rhs.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn lt_zero(self) -> Mask16Sse42 {
        Self(_mm_setzero_si128(), self.1).gt(self)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq(self, rhs: Self) -> Mask16Sse42 {
        Mask16Sse42(_mm_cmpeq_epi16(self.0, rhs.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq_zero(self) -> Mask16Sse42 {
        self.eq(Self(_mm_setzero_si128(), self.1))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_slli_epi16::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_srai_epi16::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        Self(_mm_mulhi_epi16(self.0, rhs.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_u16(self) -> U16VecSse42 {
        U16VecSse42(self.0, self.1)
    }

    impl_i16_array_interface!();

    #[target_feature(from_args)]
    #[inline(always)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 7);

        let r0 = Self::load_array(d, &data[0]).0;
        let r1 = Self::load_array(d, &data[1 * stride]).0;
        let r2 = Self::load_array(d, &data[2 * stride]).0;
        let r3 = Self::load_array(d, &data[3 * stride]).0;
        let r4 = Self::load_array(d, &data[4 * stride]).0;
        let r5 = Self::load_array(d, &data[5 * stride]).0;
        let r6 = Self::load_array(d, &data[6 * stride]).0;
        let r7 = Self::load_array(d, &data[7 * stride]).0;

        // Stage 1: 16-bit unpack
        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpackhi_epi16(r0, r1);
        let t2 = _mm_unpacklo_epi16(r2, r3);
        let t3 = _mm_unpackhi_epi16(r2, r3);
        let t4 = _mm_unpacklo_epi16(r4, r5);
        let t5 = _mm_unpackhi_epi16(r4, r5);
        let t6 = _mm_unpacklo_epi16(r6, r7);
        let t7 = _mm_unpackhi_epi16(r6, r7);

        // Stage 2: 32-bit unpack
        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);

        // Stage 3: 64-bit unpack
        let out0 = _mm_unpacklo_epi64(u0, u4);
        let out1 = _mm_unpackhi_epi64(u0, u4);
        let out2 = _mm_unpacklo_epi64(u1, u5);
        let out3 = _mm_unpackhi_epi64(u1, u5);
        let out4 = _mm_unpacklo_epi64(u2, u6);
        let out5 = _mm_unpackhi_epi64(u2, u6);
        let out6 = _mm_unpacklo_epi64(u3, u7);
        let out7 = _mm_unpackhi_epi64(u3, u7);

        Self(out0, d).store_array(&mut data[0]);
        Self(out1, d).store_array(&mut data[1 * stride]);
        Self(out2, d).store_array(&mut data[2 * stride]);
        Self(out3, d).store_array(&mut data[3 * stride]);
        Self(out4, d).store_array(&mut data[4 * stride]);
        Self(out5, d).store_array(&mut data[5 * stride]);
        Self(out6, d).store_array(&mut data[6 * stride]);
        Self(out7, d).store_array(&mut data[7 * stride]);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        let mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuffled = _mm_shuffle_epi8(self.0, mask);
        let val = _mm_cvtsi128_si64(shuffled);
        let bytes = val.to_ne_bytes();
        dest[..8].copy_from_slice(&bytes);
    }
}

impl Add<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_add_epi16(self.0, rhs.0), self.1)
    }
}

impl Sub<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_sub_epi16(self.0, rhs.0), self.1)
    }
}

impl Neg for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> I16VecSse42 {
        I16VecSse42(_mm_sub_epi16(_mm_setzero_si128(), self.0), self.1)
    }
}

impl BitAnd<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_and_si128(self.0, rhs.0), self.1)
    }
}

impl BitOr<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_or_si128(self.0, rhs.0), self.1)
    }
}

impl BitXor<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_xor_si128(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_add_epi16(self.0, rhs.0);
    }
}

impl SubAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_sub_epi16(self.0, rhs.0);
    }
}

impl BitAndAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_and_si128(self.0, rhs.0);
    }
}

impl BitOrAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_or_si128(self.0, rhs.0);
    }
}

impl BitXorAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_xor_si128(self.0, rhs.0);
    }
}

impl Mul<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_mullo_epi16(self.0, rhs.0), self.1)
    }
}

impl MulAssign<I16VecSse42> for I16VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I16VecSse42) {
        self.0 = _mm_mullo_epi16(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecSse42(__m128i, Sse42Descriptor);

impl U32SimdVec for U32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSse42 {
        I32VecSse42(self.0, self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_srli_epi32::<AMOUNT_I>(self.0), self.1)
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecSse42(__m128i, Sse42Descriptor);

impl U64SimdVec for U64VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 2;

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u64) -> Self {
        Self(_mm_set1_epi64x(v as i64), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u64]) -> Self {
        Self(_mm_loadu_si128(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u64]) {
        _mm_storeu_si128(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_slli_epi64::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(_mm_srli_epi64::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSse42 {
        U32VecSse42(self.0, self.1)
    }
}

impl Add<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_add_epi64(self.0, rhs.0), self.1)
    }
}

impl AddAssign<U64VecSse42> for U64VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: U64VecSse42) {
        self.0 = _mm_add_epi64(self.0, rhs.0);
    }
}

impl BitAnd<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_and_si128(self.0, rhs.0), self.1)
    }
}

impl BitAndAssign<U64VecSse42> for U64VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: U64VecSse42) {
        self.0 = _mm_and_si128(self.0, rhs.0);
    }
}

impl BitOr<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_or_si128(self.0, rhs.0), self.1)
    }
}

impl BitOrAssign<U64VecSse42> for U64VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: U64VecSse42) {
        self.0 = _mm_or_si128(self.0, rhs.0);
    }
}

impl BitXor<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_xor_si128(self.0, rhs.0), self.1)
    }
}

impl BitXorAssign<U64VecSse42> for U64VecSse42 {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: U64VecSse42) {
        self.0 = _mm_xor_si128(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecSse42(__m128i, Sse42Descriptor);

impl U8SimdVec for U8VecSse42 {
    type Descriptor = Sse42Descriptor;
    const LEN: usize = 16;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        Self(_mm_loadu_si128(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(_mm_set1_epi8(v as i8), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        _mm_storeu_si128(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 2 * Self::LEN);
        let lo = _mm_unpacklo_epi8(a.0, b.0);
        let hi = _mm_unpackhi_epi8(a.0, b.0);
        let [d0, d1, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, lo);
        _mm_storeu_si128(d1, hi);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 3 * Self::LEN);

        // Masks for out0
        let mask_a0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
        let mask_b0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
        let mask_c0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);

        // Masks for out1
        let mask_a1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
        let mask_b1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
        let mask_c1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);

        // Masks for out2
        let mask_a2 = _mm_setr_epi8(
            -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
        );
        let mask_b2 = _mm_setr_epi8(
            -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
        );
        let mask_c2 = _mm_setr_epi8(
            10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
        );

        let out0 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a0),
                _mm_shuffle_epi8(b.0, mask_b0),
            ),
            _mm_shuffle_epi8(c.0, mask_c0),
        );
        let out1 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a1),
                _mm_shuffle_epi8(b.0, mask_b1),
            ),
            _mm_shuffle_epi8(c.0, mask_c1),
        );
        let out2 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a2),
                _mm_shuffle_epi8(b.0, mask_b2),
            ),
            _mm_shuffle_epi8(c.0, mask_c2),
        );

        let [d0, d1, d2, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, out0);
        _mm_storeu_si128(d1, out1);
        _mm_storeu_si128(d2, out2);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 4 * Self::LEN);
        // First interleave pairs: ab and cd
        let ab_lo = _mm_unpacklo_epi8(a.0, b.0);
        let ab_hi = _mm_unpackhi_epi8(a.0, b.0);
        let cd_lo = _mm_unpacklo_epi8(c.0, d.0);
        let cd_hi = _mm_unpackhi_epi8(c.0, d.0);

        // Then interleave the pairs to get final layout
        let out0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
        let out1 = _mm_unpackhi_epi16(ab_lo, cd_lo);
        let out2 = _mm_unpacklo_epi16(ab_hi, cd_hi);
        let out3 = _mm_unpackhi_epi16(ab_hi, cd_hi);

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, out0);
        _mm_storeu_si128(d1, out1);
        _mm_storeu_si128(d2, out2);
        _mm_storeu_si128(d3, out3);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecSse42(__m128i, Sse42Descriptor);

impl U16SimdVec for U16VecSse42 {
    type Descriptor = Sse42Descriptor;
    const LEN: usize = 8;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(_mm_loadu_si128(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(_mm_set1_epi16(v as i16), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        _mm_storeu_si128(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 2 * Self::LEN);
        let lo = _mm_unpacklo_epi16(a.0, b.0);
        let hi = _mm_unpackhi_epi16(a.0, b.0);
        let [d0, d1, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, lo);
        _mm_storeu_si128(d1, hi);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 3 * Self::LEN);

        // Masks for out0
        let mask_a0 = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
        let mask_b0 = _mm_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
        let mask_c0 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);

        // Masks for out1
        let mask_a1 = _mm_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
        let mask_b1 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
        let mask_c1 = _mm_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);

        // Masks for out2
        let mask_a2 = _mm_setr_epi8(
            -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
        );
        let mask_b2 = _mm_setr_epi8(
            10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
        );
        let mask_c2 = _mm_setr_epi8(
            -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
        );

        let out0 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a0),
                _mm_shuffle_epi8(b.0, mask_b0),
            ),
            _mm_shuffle_epi8(c.0, mask_c0),
        );
        let out1 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a1),
                _mm_shuffle_epi8(b.0, mask_b1),
            ),
            _mm_shuffle_epi8(c.0, mask_c1),
        );
        let out2 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(a.0, mask_a2),
                _mm_shuffle_epi8(b.0, mask_b2),
            ),
            _mm_shuffle_epi8(c.0, mask_c2),
        );

        let [d0, d1, d2, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, out0);
        _mm_storeu_si128(d1, out1);
        _mm_storeu_si128(d2, out2);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 4 * Self::LEN);
        // First interleave pairs: ab and cd
        let ab_lo = _mm_unpacklo_epi16(a.0, b.0);
        let ab_hi = _mm_unpackhi_epi16(a.0, b.0);
        let cd_lo = _mm_unpacklo_epi16(c.0, d.0);
        let cd_hi = _mm_unpackhi_epi16(c.0, d.0);

        // Then interleave the pairs to get final layout
        let out0 = _mm_unpacklo_epi32(ab_lo, cd_lo);
        let out1 = _mm_unpackhi_epi32(ab_lo, cd_lo);
        let out2 = _mm_unpacklo_epi32(ab_hi, cd_hi);
        let out3 = _mm_unpackhi_epi32(ab_hi, cd_hi);

        let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<{ Self::LEN }>().0 else { unreachable!() };
        _mm_storeu_si128(d0, out0);
        _mm_storeu_si128(d1, out1);
        _mm_storeu_si128(d2, out2);
        _mm_storeu_si128(d3, out3);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_i16(self) -> I16VecSse42 {
        I16VecSse42(self.0, self.1)
    }
}

impl SimdMask for MaskSse42 {
    type Descriptor = Sse42Descriptor;

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_f32(self, if_true: F32VecSse42, if_false: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_blendv_ps(if_false.0, if_true.0, self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_i32(self, if_true: I32VecSse42, if_false: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(
            _mm_blendv_epi8(if_false.0, if_true.0, _mm_castps_si128(self.0)),
            self.1,
        )
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn maskz_i32(self, v: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_andnot_si128(_mm_castps_si128(self.0), v.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn all(self) -> bool {
        _mm_movemask_ps(self.0) == 0b1111
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn andnot(self, rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_andnot_ps(self.0, rhs.0), self.1)
    }
}

impl BitAnd<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_and_ps(self.0, rhs.0), self.1)
    }
}

impl BitOr<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_or_ps(self.0, rhs.0), self.1)
    }
}

impl SimdMask16 for Mask16Sse42 {
    type Descriptor = Sse42Descriptor;

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_i16(self, if_true: I16VecSse42, if_false: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_blendv_epi8(if_false.0, if_true.0, self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn maskz_i16(self, v: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_andnot_si128(self.0, v.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn all(self) -> bool {
        _mm_movemask_epi8(self.0) == 0xFFFF
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn andnot(self, rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_andnot_si128(self.0, rhs.0), self.1)
    }
}

impl BitAnd<Mask16Sse42> for Mask16Sse42 {
    type Output = Mask16Sse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_and_si128(self.0, rhs.0), self.1)
    }
}

impl BitOr<Mask16Sse42> for Mask16Sse42 {
    type Output = Mask16Sse42;
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_or_si128(self.0, rhs.0), self.1)
    }
}
