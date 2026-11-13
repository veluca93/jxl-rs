// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use super::super::{
    AvxDescriptor, F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16,
    U8SimdVec, U16SimdVec,
};
use crate::unaligned::x86_64 as unaligned;
use crate::{
    Sse42Descriptor, U32SimdVec, U64SimdVec, impl_f32_array_interface, impl_i16_array_interface,
};

// Safety invariant: this type is only ever constructed if avx512f and avx512bw are available.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Descriptor(());

#[allow(unused)]
impl Avx512Descriptor {
    /// # Safety
    /// The caller must guarantee that "avx512f" and "avx512bw" target features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
    pub fn downgrade_avx(self) -> AvxDescriptor {
        // SAFETY: the safety invariant on `self` guarantees avx512f is available, which implies
        // avx2 and fma.
        unsafe { AvxDescriptor::new_unchecked() }
    }
}

/// Prepared 8-entry lookup table for AVX512.
/// For AVX512, vpermutexvar_ps is both fast and exact, so we store f32 values
/// duplicated to fill a 512-bit register.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Avx512(__m512);

impl crate::SimdDescriptor512 for Avx512Descriptor {}

impl SimdDescriptor for Avx512Descriptor {
    #[inline(always)]
    fn as_avx512(self) -> Option<Self> {
        Some(self)
    }

    type F32Vec = F32VecAvx512;
    type I32Vec = I32VecAvx512;
    type I16Vec = I16VecAvx512;
    type U64Vec = U64VecAvx512;
    type U32Vec = U32VecAvx512;
    type U8Vec = U8VecAvx512;
    type U16Vec = U16VecAvx512;
    type Mask = MaskAvx512;
    type Mask16 = Mask16Avx512;
    type Bf16Table8 = Bf16Table8Avx512;

    type Descriptor256 = AvxDescriptor;
    type Descriptor128 = Sse42Descriptor;

    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self.downgrade_avx()
    }

    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self.downgrade_avx().downgrade_sse42()
    }

    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: we just checked for avx512f and avx512bw.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx512f,avx512bw")]
        #[inline(never)]
        unsafe fn inner<R>(d: Avx512Descriptor, f: impl FnOnce(Avx512Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees avx512f and avx512bw.
        unsafe { inner(self, f) }
    }
}

// TODO(veluca): retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_avx {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident$(<$(const $cg:ident: $cg_ty:ty),* $(,)?>)?($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name$(<$(const $cg: $cg_ty),*>)?(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "avx512f,avx512bw")]
            #[inline]
            fn inner$(<$(const $cg: $cg_ty),*>)?($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff avx512f and avx512bw are available.
            unsafe { inner$(::<$($cg),*>)?(self, $($arg),*) }
        }
    };
    (
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name($($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "avx512f,avx512bw")]
            #[inline]
            fn inner($($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: the descriptor or vector argument witnesses that avx512f and avx512bw are available.
            unsafe { inner($($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecAvx512(__m512, Avx512Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskAvx512(__mmask16, Avx512Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Avx512(__mmask32, Avx512Descriptor);

impl F32SimdVec for F32VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, F32VecAvx512);
    crate::impl_isa_array_conversions!(avx512, f32, 16);

    const LEN: usize = 16;

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[f32]) -> F32VecAvx512 {
            F32VecAvx512(unaligned::_mm512_loadu_ps(mem.first_chunk().unwrap()), d)
        }
    );

    fn_avx!(this: &F32VecAvx512, fn store(mem: &mut [f32]) {
        unaligned::_mm512_storeu_ps(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(
        fn store_interleaved_2(a: F32VecAvx512, b: F32VecAvx512, dest: &mut [f32]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            // a = [a0..a15], b = [b0..b15]
            // Output: [a0, b0, a1, b1, ..., a15, b15]
            // unpacklo within each 128-bit lane: lane0=[a0,b0,a1,b1], lane1=[a4,b4,a5,b5], etc.
            let lo = _mm512_unpacklo_ps(a.0, b.0);
            // unpackhi within each 128-bit lane: lane0=[a2,b2,a3,b3], lane1=[a6,b6,a7,b7], etc.
            let hi = _mm512_unpackhi_ps(a.0, b.0);

            // Permute to interleave 128-bit chunks from lo and hi
            // out0 needs: lo lanes 0,1 interleaved with hi lanes 0,1
            let idx_lo = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
            // out1 needs: lo lanes 2,3 interleaved with hi lanes 2,3
            let idx_hi =
                _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);

            let out0 = _mm512_permutex2var_ps(lo, idx_lo, hi);
            let out1 = _mm512_permutex2var_ps(lo, idx_hi, hi);

            unaligned::_mm512_storeu_ps(d0, out0);
            unaligned::_mm512_storeu_ps(d1, out1);
        }
    );

    fn_avx!(
        fn store_interleaved_3(
            a: F32VecAvx512,
            b: F32VecAvx512,
            c: F32VecAvx512,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };

            let idx_ab0 = _mm512_setr_epi32(0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5);
            let idx_c0 = _mm512_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0);

            let idx_ab1 = _mm512_setr_epi32(21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26);
            let idx_c1 = _mm512_setr_epi32(0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0);

            let idx_ab2 =
                _mm512_setr_epi32(0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0);
            let idx_c2 = _mm512_setr_epi32(10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15);

            let out0 = _mm512_permutex2var_ps(a.0, idx_ab0, b.0);
            let out0 = _mm512_mask_permutexvar_ps(out0, 0b0100100100100100, idx_c0, c.0);

            let out1 = _mm512_permutex2var_ps(a.0, idx_ab1, b.0);
            let out1 = _mm512_mask_permutexvar_ps(out1, 0b0010010010010010, idx_c1, c.0);

            let out2 = _mm512_permutex2var_ps(a.0, idx_ab2, b.0);
            let out2 = _mm512_mask_permutexvar_ps(out2, 0b1001001001001001, idx_c2, c.0);

            unaligned::_mm512_storeu_ps(d0, out0);
            unaligned::_mm512_storeu_ps(d1, out1);
            unaligned::_mm512_storeu_ps(d2, out2);
        }
    );

    fn_avx!(
        fn store_interleaved_4(
            a: F32VecAvx512,
            b: F32VecAvx512,
            c: F32VecAvx512,
            d: F32VecAvx512,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            // a = [a0..a15], b = [b0..b15], c = [c0..c15], d = [d0..d15]
            // Output: [a0,b0,c0,d0, a1,b1,c1,d1, ..., a15,b15,c15,d15]

            // Stage 1: Interleave pairs within 128-bit lanes
            // ab_lo lane k: [a[4k], b[4k], a[4k+1], b[4k+1]]
            let ab_lo = _mm512_unpacklo_ps(a.0, b.0);
            // ab_hi lane k: [a[4k+2], b[4k+2], a[4k+3], b[4k+3]]
            let ab_hi = _mm512_unpackhi_ps(a.0, b.0);
            let cd_lo = _mm512_unpacklo_ps(c.0, d.0);
            let cd_hi = _mm512_unpackhi_ps(c.0, d.0);

            // Stage 2: 64-bit interleave to get 4 elements together
            // abcd_0 lane k: [a[4k], b[4k], c[4k], d[4k]]
            let abcd_0 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ab_lo),
                _mm512_castps_pd(cd_lo),
            ));
            // abcd_1 lane k: [a[4k+1], b[4k+1], c[4k+1], d[4k+1]]
            let abcd_1 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ab_lo),
                _mm512_castps_pd(cd_lo),
            ));
            // abcd_2 lane k: [a[4k+2], b[4k+2], c[4k+2], d[4k+2]]
            let abcd_2 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ab_hi),
                _mm512_castps_pd(cd_hi),
            ));
            // abcd_3 lane k: [a[4k+3], b[4k+3], c[4k+3], d[4k+3]]
            let abcd_3 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ab_hi),
                _mm512_castps_pd(cd_hi),
            ));

            // Stage 3: We need to output where each output vector collects same-index
            // elements from all 4 lanes. This is essentially a 4x4 transpose of 128-bit blocks.
            // out0 = [abcd_0 lane 0, abcd_1 lane 0, abcd_2 lane 0, abcd_3 lane 0]
            // out1 = [abcd_0 lane 1, abcd_1 lane 1, abcd_2 lane 1, abcd_3 lane 1]
            // etc.

            // Step 3a: First combine pairs (0,1) and (2,3) selecting same lane from each
            // pair01_lane0 = [abcd_0 lane 0, abcd_1 lane 0, abcd_0 lane 2, abcd_1 lane 2]
            let idx_even =
                _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
            // pair01_lane1 = [abcd_0 lane 1, abcd_1 lane 1, abcd_0 lane 3, abcd_1 lane 3]
            let idx_odd =
                _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);

            let pair01_02 = _mm512_permutex2var_ps(abcd_0, idx_even, abcd_1);
            let pair01_13 = _mm512_permutex2var_ps(abcd_0, idx_odd, abcd_1);
            let pair23_02 = _mm512_permutex2var_ps(abcd_2, idx_even, abcd_3);
            let pair23_13 = _mm512_permutex2var_ps(abcd_2, idx_odd, abcd_3);

            // Step 3b: Now combine pairs to get final output
            // out0 needs lanes 0 from pair01_02 and pair23_02
            // out1 needs lanes 1 from pair01_13 and pair23_13
            // But pair01_02 has: [abcd_0 lane 0, abcd_1 lane 0, abcd_0 lane 2, abcd_1 lane 2]
            // And pair23_02 has: [abcd_2 lane 0, abcd_3 lane 0, abcd_2 lane 2, abcd_3 lane 2]
            // out0 = [abcd_0 lane 0, abcd_1 lane 0, abcd_2 lane 0, abcd_3 lane 0]
            //      = [pair01_02 lane 0, pair01_02 lane 1, pair23_02 lane 0, pair23_02 lane 1]
            let idx_0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            let idx_1 =
                _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

            let out0 = _mm512_permutex2var_ps(pair01_02, idx_0, pair23_02);
            let out2 = _mm512_permutex2var_ps(pair01_02, idx_1, pair23_02);
            let out1 = _mm512_permutex2var_ps(pair01_13, idx_0, pair23_13);
            let out3 = _mm512_permutex2var_ps(pair01_13, idx_1, pair23_13);

            unaligned::_mm512_storeu_ps(d0, out0);
            unaligned::_mm512_storeu_ps(d1, out1);
            unaligned::_mm512_storeu_ps(d2, out2);
            unaligned::_mm512_storeu_ps(d3, out3);
        }
    );

    fn_avx!(
        fn store_interleaved_8(
            a: F32VecAvx512,
            b: F32VecAvx512,
            c: F32VecAvx512,
            d: F32VecAvx512,
            e: F32VecAvx512,
            f: F32VecAvx512,
            g: F32VecAvx512,
            h: F32VecAvx512,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            // a..h each have 16 elements. Output is 128 elements interleaved:
            // [a0,b0,c0,d0,e0,f0,g0,h0, a1,b1,c1,d1,e1,f1,g1,h1, ..., a15,b15,...,h15]
            // Each output vector is 16 floats = 2 groups of 8.

            // Stage 1: Unpack pairs within 128-bit lanes
            // ab_lo lane k: [a[4k], b[4k], a[4k+1], b[4k+1]]
            let ab_lo = _mm512_unpacklo_ps(a.0, b.0);
            let ab_hi = _mm512_unpackhi_ps(a.0, b.0);
            let cd_lo = _mm512_unpacklo_ps(c.0, d.0);
            let cd_hi = _mm512_unpackhi_ps(c.0, d.0);
            let ef_lo = _mm512_unpacklo_ps(e.0, f.0);
            let ef_hi = _mm512_unpackhi_ps(e.0, f.0);
            let gh_lo = _mm512_unpacklo_ps(g.0, h.0);
            let gh_hi = _mm512_unpackhi_ps(g.0, h.0);

            // Stage 2: 64-bit interleave to get 4-element groups
            // abcd_0 lane k: [a[4k], b[4k], c[4k], d[4k]]
            let abcd_0 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ab_lo),
                _mm512_castps_pd(cd_lo),
            ));
            let abcd_1 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ab_lo),
                _mm512_castps_pd(cd_lo),
            ));
            let abcd_2 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ab_hi),
                _mm512_castps_pd(cd_hi),
            ));
            let abcd_3 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ab_hi),
                _mm512_castps_pd(cd_hi),
            ));
            let efgh_0 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ef_lo),
                _mm512_castps_pd(gh_lo),
            ));
            let efgh_1 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ef_lo),
                _mm512_castps_pd(gh_lo),
            ));
            let efgh_2 = _mm512_castpd_ps(_mm512_unpacklo_pd(
                _mm512_castps_pd(ef_hi),
                _mm512_castps_pd(gh_hi),
            ));
            let efgh_3 = _mm512_castpd_ps(_mm512_unpackhi_pd(
                _mm512_castps_pd(ef_hi),
                _mm512_castps_pd(gh_hi),
            ));

            // Stage 3: Combine abcd_i with efgh_i to get 8-element groups per lane
            // full_0 = [abcd_0 lane 0, efgh_0 lane 0, abcd_0 lane 1, efgh_0 lane 1,
            //           abcd_0 lane 2, efgh_0 lane 2, abcd_0 lane 3, efgh_0 lane 3]
            // But we need output like:
            // out0 = [all channels at index 0, all channels at index 1]
            //      = [abcd_0 lane 0 ++ efgh_0 lane 0, abcd_1 lane 0 ++ efgh_1 lane 0]

            // Interleave 128-bit blocks from abcd and efgh within each vector
            let idx_02 =
                _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
            let idx_13 =
                _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);

            // full_0_02 = [abcd_0 lane 0, efgh_0 lane 0, abcd_0 lane 2, efgh_0 lane 2]
            let full_0_02 = _mm512_permutex2var_ps(abcd_0, idx_02, efgh_0);
            let full_0_13 = _mm512_permutex2var_ps(abcd_0, idx_13, efgh_0);
            let full_1_02 = _mm512_permutex2var_ps(abcd_1, idx_02, efgh_1);
            let full_1_13 = _mm512_permutex2var_ps(abcd_1, idx_13, efgh_1);
            let full_2_02 = _mm512_permutex2var_ps(abcd_2, idx_02, efgh_2);
            let full_2_13 = _mm512_permutex2var_ps(abcd_2, idx_13, efgh_2);
            let full_3_02 = _mm512_permutex2var_ps(abcd_3, idx_02, efgh_3);
            let full_3_13 = _mm512_permutex2var_ps(abcd_3, idx_13, efgh_3);

            // Stage 4: Now we need to combine across the _0/_1/_2/_3 indices
            // full_i_02 has: [idx 4*lane, idx 4*lane+1 at (abcd,efgh) for lanes 0,2]
            // We need output vectors that have consecutive indices from all channels

            // out0 = [idx 0 all 8 ch, idx 1 all 8 ch] = [full_0_02 lanes 0,1, full_1_02 lanes 0,1]
            // out1 = [idx 2 all 8 ch, idx 3 all 8 ch] = [full_2_02 lanes 0,1, full_3_02 lanes 0,1]
            // out2 = [idx 4 all 8 ch, idx 5 all 8 ch] = [full_0_13 lanes 0,1, full_1_13 lanes 0,1]
            // out3 = [idx 6 all 8 ch, idx 7 all 8 ch] = [full_2_13 lanes 0,1, full_3_13 lanes 0,1]
            // out4 = [idx 8 all 8 ch, idx 9 all 8 ch] = [full_0_02 lanes 2,3, full_1_02 lanes 2,3]
            // etc.

            let idx_lo = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            let idx_hi =
                _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);

            let out0 = _mm512_permutex2var_ps(full_0_02, idx_lo, full_1_02);
            let out1 = _mm512_permutex2var_ps(full_2_02, idx_lo, full_3_02);
            let out2 = _mm512_permutex2var_ps(full_0_13, idx_lo, full_1_13);
            let out3 = _mm512_permutex2var_ps(full_2_13, idx_lo, full_3_13);
            let out4 = _mm512_permutex2var_ps(full_0_02, idx_hi, full_1_02);
            let out5 = _mm512_permutex2var_ps(full_2_02, idx_hi, full_3_02);
            let out6 = _mm512_permutex2var_ps(full_0_13, idx_hi, full_1_13);
            let out7 = _mm512_permutex2var_ps(full_2_13, idx_hi, full_3_13);

            unaligned::_mm512_storeu_ps(d0, out0);
            unaligned::_mm512_storeu_ps(d1, out1);
            unaligned::_mm512_storeu_ps(d2, out2);
            unaligned::_mm512_storeu_ps(d3, out3);
            unaligned::_mm512_storeu_ps(d4, out4);
            unaligned::_mm512_storeu_ps(d5, out5);
            unaligned::_mm512_storeu_ps(d6, out6);
            unaligned::_mm512_storeu_ps(d7, out7);
        }
    );

    fn_avx!(
        fn load_deinterleaved_2(d: Avx512Descriptor, src: &[f32]) -> (F32VecAvx512, F32VecAvx512) {
            let [s0, s1, ..] = src.as_chunks::<16>().0 else {
                panic!("src too short");
            };
            // Input: [a0,b0,a1,b1,...,a15,b15]
            // Output: a = [a0..a15], b = [b0..b15]
            let in0 = unaligned::_mm512_loadu_ps(s0);
            let in1 = unaligned::_mm512_loadu_ps(s1);

            // Use permutex2var to gather even/odd indices
            let idx_a =
                _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            let idx_b =
                _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

            let a = _mm512_permutex2var_ps(in0, idx_a, in1);
            let b = _mm512_permutex2var_ps(in0, idx_b, in1);

            (F32VecAvx512(a, d), F32VecAvx512(b, d))
        }
    );

    fn_avx!(
        fn load_deinterleaved_3(
            d: Avx512Descriptor,
            src: &[f32],
        ) -> (F32VecAvx512, F32VecAvx512, F32VecAvx512) {
            let [s0, s1, s2, ..] = src.as_chunks::<16>().0 else {
                panic!("src too short");
            };
            // Input layout (48 floats in 3x16-float vectors):
            // in0: [a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5]
            // in1: [b5,c5,a6,b6,c6,a7,b7,c7,a8,b8,c8,a9,b9,c9,a10,b10]
            // in2: [c10,a11,b11,c11,a12,b12,c12,a13,b13,c13,a14,b14,c14,a15,b15,c15]
            // Output: a = [a0..a15], b = [b0..b15], c = [c0..c15]
            let in0 = unaligned::_mm512_loadu_ps(s0);
            let in1 = unaligned::_mm512_loadu_ps(s1);
            let in2 = unaligned::_mm512_loadu_ps(s2);

            // Use permutex2var to gather elements from pairs of vectors, then blend.
            // For 'a': positions 0,3,6,9,12,15 from in0; 2,5,8,11,14 from in1; 1,4,7,10,13 from in2
            // a[0..5] from in0, a[6..10] from in1, a[11..15] from in2

            // Gather indices for each channel from in0+in1 (first 32 elements)
            let idx_a_01 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
            let idx_b_01 =
                _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
            let idx_c_01 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);

            // Gather indices for remaining elements from in1+in2 (last 32 elements)
            let idx_a_12 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 20, 23, 26, 29);
            let idx_b_12 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 21, 24, 27, 30);
            let idx_c_12 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 19, 22, 25, 28, 31);

            // Gather from in0+in1 for first ~11 elements, in1+in2 for last ~5
            let a_01 = _mm512_permutex2var_ps(in0, idx_a_01, in1);
            let a_12 = _mm512_permutex2var_ps(in1, idx_a_12, in2);
            let a = _mm512_mask_blend_ps(0xF800, a_01, a_12); // positions 11-15 from a_12

            let b_01 = _mm512_permutex2var_ps(in0, idx_b_01, in1);
            let b_12 = _mm512_permutex2var_ps(in1, idx_b_12, in2);
            let b = _mm512_mask_blend_ps(0xF800, b_01, b_12); // positions 11-15 from b_12

            let c_01 = _mm512_permutex2var_ps(in0, idx_c_01, in1);
            let c_12 = _mm512_permutex2var_ps(in1, idx_c_12, in2);
            let c = _mm512_mask_blend_ps(0xFC00, c_01, c_12); // positions 10-15 from c_12

            (F32VecAvx512(a, d), F32VecAvx512(b, d), F32VecAvx512(c, d))
        }
    );

    fn_avx!(
        fn load_deinterleaved_4(
            d: Avx512Descriptor,
            src: &[f32],
        ) -> (F32VecAvx512, F32VecAvx512, F32VecAvx512, F32VecAvx512) {
            let [s0, s1, s2, s3, ..] = src.as_chunks::<16>().0 else {
                panic!("src too short");
            };
            // Input: [a0,b0,c0,d0,a1,b1,c1,d1,...] (64 floats)
            // Output: a = [a0..a15], b = [b0..b15], c = [c0..c15], d = [d0..d15]
            let in0 = unaligned::_mm512_loadu_ps(s0);
            let in1 = unaligned::_mm512_loadu_ps(s1);
            let in2 = unaligned::_mm512_loadu_ps(s2);
            let in3 = unaligned::_mm512_loadu_ps(s3);

            // Use permutex2var to gather every 4th element
            let idx_a = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
            let idx_b = _mm512_setr_epi32(1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29);
            let idx_c =
                _mm512_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30);
            let idx_d =
                _mm512_setr_epi32(3, 7, 11, 15, 19, 23, 27, 31, 3, 7, 11, 15, 19, 23, 27, 31);

            // Gather from in0+in1 for first 8 elements, in2+in3 for last 8
            let a01 = _mm512_permutex2var_ps(in0, idx_a, in1);
            let a23 = _mm512_permutex2var_ps(in2, idx_a, in3);
            let a = _mm512_mask_blend_ps(0xFF00, a01, a23);

            let b01 = _mm512_permutex2var_ps(in0, idx_b, in1);
            let b23 = _mm512_permutex2var_ps(in2, idx_b, in3);
            let b = _mm512_mask_blend_ps(0xFF00, b01, b23);

            let c01 = _mm512_permutex2var_ps(in0, idx_c, in1);
            let c23 = _mm512_permutex2var_ps(in2, idx_c, in3);
            let c = _mm512_mask_blend_ps(0xFF00, c01, c23);

            let d01 = _mm512_permutex2var_ps(in0, idx_d, in1);
            let d23 = _mm512_permutex2var_ps(in2, idx_d, in3);
            let dv = _mm512_mask_blend_ps(0xFF00, d01, d23);

            (
                F32VecAvx512(a, d),
                F32VecAvx512(b, d),
                F32VecAvx512(c, d),
                F32VecAvx512(dv, d),
            )
        }
    );

    fn_avx!(this: F32VecAvx512, fn mul_add(mul: F32VecAvx512, add: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_fmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn neg_mul_add(mul: F32VecAvx512, add: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_fnmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: f32) -> F32VecAvx512 {
            F32VecAvx512(_mm512_set1_ps(v), d)
        }
    );

    fn_avx!(
        fn zero(d: Avx512Descriptor) -> F32VecAvx512 {
            F32VecAvx512(_mm512_setzero_ps(), d)
        }
    );

    fn_avx!(this: F32VecAvx512, fn abs() -> F32VecAvx512 {
        F32VecAvx512(_mm512_abs_ps(this.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn floor() -> F32VecAvx512 {
        F32VecAvx512(_mm512_roundscale_ps::<{ _MM_FROUND_FLOOR }>(this.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn sqrt() -> F32VecAvx512 {
        F32VecAvx512(_mm512_sqrt_ps(this.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn neg() -> F32VecAvx512 {
        F32VecAvx512(
            _mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_set1_epi32(i32::MIN),
                _mm512_castps_si512(this.0),
            )),
            this.1,
        )
    });

    fn_avx!(this: F32VecAvx512, fn copysign(sign: F32VecAvx512) -> F32VecAvx512 {
        let sign_mask = _mm512_set1_epi32(i32::MIN);
        F32VecAvx512(
            _mm512_castsi512_ps(_mm512_or_si512(
                _mm512_andnot_si512(sign_mask, _mm512_castps_si512(this.0)),
                _mm512_and_si512(sign_mask, _mm512_castps_si512(sign.0)),
            )),
            this.1,
        )
    });

    fn_avx!(this: F32VecAvx512, fn max(other: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_max_ps(this.0, other.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn min(other: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_min_ps(this.0, other.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn gt(other: F32VecAvx512) -> MaskAvx512 {
        MaskAvx512(_mm512_cmp_ps_mask::<{_CMP_GT_OQ}>(this.0, other.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn as_i32() -> I32VecAvx512 {
        I32VecAvx512(_mm512_cvttps_epi32(this.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn bitcast_to_i32() -> I32VecAvx512 {
        I32VecAvx512(_mm512_castps_si512(this.0), this.1)
    });

    fn_avx!(
        fn prepare_table_bf16_8(_d: Avx512Descriptor, table: &[f32; 8]) -> Bf16Table8Avx512 {
            let table_256 = unaligned::_mm256_loadu_ps(table);
            // Zero-extend to 512-bit; vpermutexvar with indices 0-7 only reads first 256 bits
            Bf16Table8Avx512(_mm512_zextps256_ps512(table_256))
        }
    );

    fn_avx!(
        fn table_lookup_bf16_8(
            d: Avx512Descriptor,
            table: Bf16Table8Avx512,
            indices: I32VecAvx512,
        ) -> F32VecAvx512 {
            F32VecAvx512(_mm512_permutexvar_ps(indices.0, table.0), d)
        }
    );

    fn_avx!(this: F32VecAvx512, fn round_store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<16>().unwrap();
        // Round to nearest integer
        let rounded = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(this.0);
        // Convert to i32
        let i32s = _mm512_cvtps_epi32(rounded);
        // Use pmovusdb: saturating conversion from 32-bit to 8-bit unsigned
        let u8s = _mm512_cvtusepi32_epi8(i32s);
        // Store 16 bytes
        unaligned::_mm_storeu_si128_u8(out, u8s);
    });

    fn_avx!(this: F32VecAvx512, fn round_store_u16(dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<16>().unwrap();
        // Round to nearest integer
        let rounded = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(this.0);
        // Convert to i32
        let i32s = _mm512_cvtps_epi32(rounded);
        // Use pmovusdw: saturating conversion from 32-bit to 16-bit unsigned
        let u16s = _mm512_cvtusepi32_epi16(i32s);
        // Store 16 u16s (32 bytes)
        unaligned::_mm256_storeu_si256_u16(out, u16s);
    });

    impl_f32_array_interface!();

    fn_avx!(
        fn load_f16_bits(d: Avx512Descriptor, mem: &[u16]) -> F32VecAvx512 {
            let bits = unaligned::_mm256_loadu_si256_u16(mem.first_chunk().unwrap());
            F32VecAvx512(_mm512_cvtph_ps(bits), d)
        }
    );

    fn_avx!(this: F32VecAvx512, fn store_f16_bits(dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<16>().unwrap();
        let bits = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(this.0);
        unaligned::_mm256_storeu_si256_u16(out, bits);
    });
}

impl Add<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn add(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn sub(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn mul(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn div(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn add_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn sub_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn mul_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn div_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_div_ps(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecAvx512(__m512i, Avx512Descriptor);

impl I32SimdVec for I32VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, I32VecAvx512);

    const LEN: usize = 16;

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[i32]) -> I32VecAvx512 {
            I32VecAvx512(
                unaligned::_mm512_loadu_si512_i32(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_avx!(
        fn load_from_i16(d: Avx512Descriptor, mem: &[i16]) -> I32VecAvx512 {
            I32VecAvx512(
                _mm512_cvtepi16_epi32(unaligned::_mm256_loadu_si256_i16(
                    mem.first_chunk::<16>().unwrap(),
                )),
                d,
            )
        }
    );

    fn_avx!(
        fn load_from_u16(d: Avx512Descriptor, mem: &[u16]) -> I32VecAvx512 {
            I32VecAvx512(
                _mm512_cvtepu16_epi32(unaligned::_mm256_loadu_si256_u16(
                    mem.first_chunk::<16>().unwrap(),
                )),
                d,
            )
        }
    );

    fn_avx!(this: &I32VecAvx512, fn store(mem: &mut [i32]) {
        unaligned::_mm512_storeu_si512_i32(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: i32) -> I32VecAvx512 {
            I32VecAvx512(_mm512_set1_epi32(v), d)
        }
    );

    fn_avx!(this: I32VecAvx512, fn as_f32() -> F32VecAvx512 {
         F32VecAvx512(_mm512_cvtepi32_ps(this.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn bitcast_to_f32() -> F32VecAvx512 {
         F32VecAvx512(_mm512_castsi512_ps(this.0), this.1)
    });

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecAvx512 {
        U32VecAvx512(self.0, self.1)
    }

    fn_avx!(this: I32VecAvx512, fn abs() -> I32VecAvx512 {
        I32VecAvx512(_mm512_abs_epi32(this.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn gt(rhs: I32VecAvx512) -> MaskAvx512 {
        MaskAvx512(_mm512_cmpgt_epi32_mask(this.0, rhs.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn lt_zero() -> MaskAvx512 {
        I32VecAvx512(_mm512_setzero_epi32(), this.1).gt(this)
    });

    fn_avx!(this: I32VecAvx512, fn eq(rhs: I32VecAvx512) -> MaskAvx512 {
        MaskAvx512(_mm512_cmpeq_epi32_mask(this.0, rhs.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn eq_zero() -> MaskAvx512 {
        I32VecAvx512(_mm512_setzero_epi32(), this.1).eq(this)
    });

    fn_avx!(this: I32VecAvx512, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecAvx512 {
        I32VecAvx512(_mm512_slli_epi32::<AMOUNT_U>(this.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecAvx512 {
        I32VecAvx512(_mm512_srai_epi32::<AMOUNT_U>(this.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn mul_wide_take_high(rhs: I32VecAvx512) -> I32VecAvx512 {
        let l = _mm512_mul_epi32(this.0, rhs.0);
        let h = _mm512_mul_epi32(_mm512_srli_epi64::<32>(this.0), _mm512_srli_epi64::<32>(rhs.0));
        let idx = _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
        I32VecAvx512(_mm512_permutex2var_epi32(l, idx, h), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn store_u16(dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<16>().unwrap();
        let tmp = _mm512_cvtepi32_epi16(this.0);
        unaligned::_mm256_storeu_si256_u16(out, tmp);
    });

    fn_avx!(this: I32VecAvx512, fn store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<16>().unwrap();
        let tmp_vec = _mm512_cvtepi32_epi8(this.0);
        unaligned::_mm_storeu_si128_u8(out, tmp_vec);
    });
}

impl Add<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn add(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_add_epi32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn sub(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_sub_epi32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn mul(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_mullo_epi32(this.0, rhs.0), this.1)
    });
}

impl Neg for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn neg() -> I32VecAvx512 {
        I32VecAvx512(_mm512_setzero_epi32(), this.1) - this
    });
}

impl Shl<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn shl(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_sllv_epi32(this.0, rhs.0), this.1)
    });
}

impl Shr<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn shr(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_srav_epi32(this.0, rhs.0), this.1)
    });
}

impl BitAnd<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn bitand(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_and_si512(this.0, rhs.0), this.1)
    });
}

impl BitOr<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn bitor(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_or_si512(this.0, rhs.0), this.1)
    });
}

impl BitXor<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn bitxor(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_xor_si512(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn add_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_add_epi32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn sub_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_sub_epi32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn mul_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_mullo_epi32(this.0, rhs.0)
    });
}

impl ShlAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn shl_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_sllv_epi32(this.0, rhs.0)
    });
}

impl ShrAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn shr_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_srav_epi32(this.0, rhs.0)
    });
}

impl BitAndAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn bitand_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_and_si512(this.0, rhs.0)
    });
}

impl BitOrAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn bitor_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_or_si512(this.0, rhs.0)
    });
}

impl BitXorAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn bitxor_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_xor_si512(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecAvx512(__m512i, Avx512Descriptor);

impl I16SimdVec for I16VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, I16VecAvx512);
    crate::impl_isa_array_conversions!(avx512, i16, 32);

    const LEN: usize = 32;

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[i16]) -> I16VecAvx512 {
            I16VecAvx512(
                unaligned::_mm512_loadu_si512_i16(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_avx!(this: &I16VecAvx512, fn store(mem: &mut [i16]) {
        unaligned::_mm512_storeu_si512_i16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: i16) -> I16VecAvx512 {
            I16VecAvx512(_mm512_set1_epi16(v), d)
        }
    );

    fn_avx!(
        fn zero(d: Avx512Descriptor) -> I16VecAvx512 {
            I16VecAvx512(_mm512_setzero_si512(), d)
        }
    );

    fn_avx!(this: I16VecAvx512, fn abs() -> I16VecAvx512 {
        I16VecAvx512(_mm512_abs_epi16(this.0), this.1)
    });

    fn_avx!(this: I16VecAvx512, fn gt(rhs: I16VecAvx512) -> Mask16Avx512 {
        Mask16Avx512(_mm512_cmpgt_epi16_mask(this.0, rhs.0), this.1)
    });

    fn_avx!(this: I16VecAvx512, fn lt_zero() -> Mask16Avx512 {
        I16VecAvx512(_mm512_setzero_si512(), this.1).gt(this)
    });

    fn_avx!(this: I16VecAvx512, fn eq(rhs: I16VecAvx512) -> Mask16Avx512 {
        Mask16Avx512(_mm512_cmpeq_epi16_mask(this.0, rhs.0), this.1)
    });

    fn_avx!(this: I16VecAvx512, fn eq_zero() -> Mask16Avx512 {
        this.eq(I16VecAvx512(_mm512_setzero_si512(), this.1))
    });

    fn_avx!(this: I16VecAvx512, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecAvx512 {
        I16VecAvx512(_mm512_slli_epi16::<AMOUNT_U>(this.0), this.1)
    });

    fn_avx!(this: I16VecAvx512, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecAvx512 {
        I16VecAvx512(_mm512_srai_epi16::<AMOUNT_U>(this.0), this.1)
    });

    fn_avx!(this: I16VecAvx512, fn mul_wide_take_high(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_mulhi_epi16(this.0, rhs.0), this.1)
    });

    #[inline(always)]
    fn bitcast_u16(self) -> U16VecAvx512 {
        U16VecAvx512(self.0, self.1)
    }

    impl_i16_array_interface!();

    fn_avx!(this: I16VecAvx512, fn store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<32>().unwrap();
        let tmp_vec = _mm512_cvtepi16_epi8(this.0);
        unaligned::_mm256_storeu_si256_u8(out, tmp_vec);
    });
}

impl Add<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn add(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_add_epi16(this.0, rhs.0), this.1)
    });
}

impl Sub<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn sub(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_sub_epi16(this.0, rhs.0), this.1)
    });
}

impl Neg for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn neg() -> I16VecAvx512 {
        I16VecAvx512(_mm512_sub_epi16(_mm512_setzero_si512(), this.0), this.1)
    });
}

impl BitAnd<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn bitand(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_and_si512(this.0, rhs.0), this.1)
    });
}

impl BitOr<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn bitor(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_or_si512(this.0, rhs.0), this.1)
    });
}

impl BitXor<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn bitxor(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_xor_si512(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn add_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_add_epi16(this.0, rhs.0);
    });
}

impl SubAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn sub_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_sub_epi16(this.0, rhs.0);
    });
}

impl BitAndAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn bitand_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_and_si512(this.0, rhs.0);
    });
}

impl BitOrAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn bitor_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_or_si512(this.0, rhs.0);
    });
}

impl BitXorAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn bitxor_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_xor_si512(this.0, rhs.0);
    });
}

impl Mul<I16VecAvx512> for I16VecAvx512 {
    type Output = I16VecAvx512;
    fn_avx!(this: I16VecAvx512, fn mul(rhs: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_mullo_epi16(this.0, rhs.0), this.1)
    });
}

impl MulAssign<I16VecAvx512> for I16VecAvx512 {
    fn_avx!(this: &mut I16VecAvx512, fn mul_assign(rhs: I16VecAvx512) {
        this.0 = _mm512_mullo_epi16(this.0, rhs.0);
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecAvx512(__m512i, Avx512Descriptor);

impl U32SimdVec for U32VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, U32VecAvx512);

    const LEN: usize = 16;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecAvx512 {
        I32VecAvx512(self.0, self.1)
    }

    fn_avx!(this: U32VecAvx512, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U32VecAvx512 {
        U32VecAvx512(_mm512_srli_epi32::<AMOUNT_U>(this.0), this.1)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecAvx512(__m512i, Avx512Descriptor);

impl U64SimdVec for U64VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, U64VecAvx512);

    const LEN: usize = 8;

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: u64) -> U64VecAvx512 {
            U64VecAvx512(_mm512_set1_epi64(v as i64), d)
        }
    );

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[u64]) -> U64VecAvx512 {
            U64VecAvx512(
                unaligned::_mm512_loadu_si512_u64(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_avx!(this: &U64VecAvx512, fn store(mem: &mut [u64]) {
        unaligned::_mm512_storeu_si512_u64(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(this: U64VecAvx512, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecAvx512 {
        U64VecAvx512(_mm512_slli_epi64::<AMOUNT_U>(this.0), this.1)
    });

    fn_avx!(this: U64VecAvx512, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecAvx512 {
        U64VecAvx512(_mm512_srli_epi64::<AMOUNT_U>(this.0), this.1)
    });

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecAvx512 {
        U32VecAvx512(self.0, self.1)
    }
}

impl Add<U64VecAvx512> for U64VecAvx512 {
    type Output = U64VecAvx512;
    fn_avx!(this: U64VecAvx512, fn add(rhs: U64VecAvx512) -> U64VecAvx512 {
        U64VecAvx512(_mm512_add_epi64(this.0, rhs.0), this.1)
    });
}

impl AddAssign<U64VecAvx512> for U64VecAvx512 {
    fn_avx!(this: &mut U64VecAvx512, fn add_assign(rhs: U64VecAvx512) {
        this.0 = _mm512_add_epi64(this.0, rhs.0);
    });
}

impl BitAnd<U64VecAvx512> for U64VecAvx512 {
    type Output = U64VecAvx512;
    fn_avx!(this: U64VecAvx512, fn bitand(rhs: U64VecAvx512) -> U64VecAvx512 {
        U64VecAvx512(_mm512_and_si512(this.0, rhs.0), this.1)
    });
}

impl BitAndAssign<U64VecAvx512> for U64VecAvx512 {
    fn_avx!(this: &mut U64VecAvx512, fn bitand_assign(rhs: U64VecAvx512) {
        this.0 = _mm512_and_si512(this.0, rhs.0);
    });
}

impl BitOr<U64VecAvx512> for U64VecAvx512 {
    type Output = U64VecAvx512;
    fn_avx!(this: U64VecAvx512, fn bitor(rhs: U64VecAvx512) -> U64VecAvx512 {
        U64VecAvx512(_mm512_or_si512(this.0, rhs.0), this.1)
    });
}

impl BitOrAssign<U64VecAvx512> for U64VecAvx512 {
    fn_avx!(this: &mut U64VecAvx512, fn bitor_assign(rhs: U64VecAvx512) {
        this.0 = _mm512_or_si512(this.0, rhs.0);
    });
}

impl BitXor<U64VecAvx512> for U64VecAvx512 {
    type Output = U64VecAvx512;
    fn_avx!(this: U64VecAvx512, fn bitxor(rhs: U64VecAvx512) -> U64VecAvx512 {
        U64VecAvx512(_mm512_xor_si512(this.0, rhs.0), this.1)
    });
}

impl BitXorAssign<U64VecAvx512> for U64VecAvx512 {
    fn_avx!(this: &mut U64VecAvx512, fn bitxor_assign(rhs: U64VecAvx512) {
        this.0 = _mm512_xor_si512(this.0, rhs.0);
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecAvx512(__m512i, Avx512Descriptor);

impl U8SimdVec for U8VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, U8VecAvx512);
    const LEN: usize = 64;

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[u8]) -> U8VecAvx512 {
            U8VecAvx512(
                unaligned::_mm512_loadu_si512_u8(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: u8) -> U8VecAvx512 {
            U8VecAvx512(_mm512_set1_epi8(v as i8), d)
        }
    );

    fn_avx!(this: &U8VecAvx512, fn store(mem: &mut [u8]) {
        unaligned::_mm512_storeu_si512_u8(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(
        fn store_interleaved_2(a: U8VecAvx512, b: U8VecAvx512, dest: &mut [u8]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<64>().0 else {
                panic!("dest too short");
            };
            let lo = _mm512_unpacklo_epi8(a.0, b.0);
            let hi = _mm512_unpackhi_epi8(a.0, b.0);
            let idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
            let idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
            let out0 = _mm512_permutex2var_epi64(lo, idx0, hi);
            let out1 = _mm512_permutex2var_epi64(lo, idx1, hi);

            unaligned::_mm512_storeu_si512_u8(d0, out0);
            unaligned::_mm512_storeu_si512_u8(d1, out1);
        }
    );

    fn_avx!(
        fn store_interleaved_3(a: U8VecAvx512, b: U8VecAvx512, c: U8VecAvx512, dest: &mut [u8]) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<64>().0 else {
                panic!("dest too short");
            };

            let mask_a0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5,
            ));
            let mask_b0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1,
            ));
            let mask_c0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1,
            ));

            let mask_a1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1,
            ));
            let mask_b1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10,
            ));
            let mask_c1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1,
            ));

            let mask_a2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
            ));
            let mask_b2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
            ));
            let mask_c2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
            ));

            let res0 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a0),
                    _mm512_shuffle_epi8(b.0, mask_b0),
                ),
                _mm512_shuffle_epi8(c.0, mask_c0),
            );
            let res1 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a1),
                    _mm512_shuffle_epi8(b.0, mask_b1),
                ),
                _mm512_shuffle_epi8(c.0, mask_c1),
            );
            let res2 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a2),
                    _mm512_shuffle_epi8(b.0, mask_b2),
                ),
                _mm512_shuffle_epi8(c.0, mask_c2),
            );
            let idx_a0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 0, 1);
            let part_a0 = _mm512_permutex2var_epi64(res0, idx_a0, res1);
            let idx_f0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 4, 5);
            let final0 = _mm512_permutex2var_epi64(part_a0, idx_f0, res2);
            let idx_a1 = _mm512_setr_epi64(2, 3, 10, 11, 4, 5, 0, 1);
            let part_a1 = _mm512_permutex2var_epi64(res1, idx_a1, res2);
            let idx_f1 = _mm512_setr_epi64(0, 1, 2, 3, 12, 13, 4, 5);
            let final1 = _mm512_permutex2var_epi64(part_a1, idx_f1, res0);
            let idx_a2 = _mm512_setr_epi64(4, 5, 14, 15, 6, 7, 0, 1);
            let part_a2 = _mm512_permutex2var_epi64(res2, idx_a2, res0);
            let idx_f2 = _mm512_setr_epi64(0, 1, 2, 3, 14, 15, 4, 5);
            let final2 = _mm512_permutex2var_epi64(part_a2, idx_f2, res1);

            unaligned::_mm512_storeu_si512_u8(d0, final0);
            unaligned::_mm512_storeu_si512_u8(d1, final1);
            unaligned::_mm512_storeu_si512_u8(d2, final2);
        }
    );

    fn_avx!(
        fn store_interleaved_4(
            a: U8VecAvx512,
            b: U8VecAvx512,
            c: U8VecAvx512,
            d: U8VecAvx512,
            dest: &mut [u8],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<64>().0 else {
                panic!("dest too short");
            };
            let ab_lo = _mm512_unpacklo_epi8(a.0, b.0);
            let ab_hi = _mm512_unpackhi_epi8(a.0, b.0);
            let cd_lo = _mm512_unpacklo_epi8(c.0, d.0);
            let cd_hi = _mm512_unpackhi_epi8(c.0, d.0);

            let abcd_0 = _mm512_unpacklo_epi16(ab_lo, cd_lo);
            let abcd_1 = _mm512_unpackhi_epi16(ab_lo, cd_lo);
            let abcd_2 = _mm512_unpacklo_epi16(ab_hi, cd_hi);
            let abcd_3 = _mm512_unpackhi_epi16(ab_hi, cd_hi);

            let idx_even = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
            let idx_odd = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

            let pair01_02 = _mm512_permutex2var_epi64(abcd_0, idx_even, abcd_1);
            let pair01_13 = _mm512_permutex2var_epi64(abcd_0, idx_odd, abcd_1);
            let pair23_02 = _mm512_permutex2var_epi64(abcd_2, idx_even, abcd_3);
            let pair23_13 = _mm512_permutex2var_epi64(abcd_2, idx_odd, abcd_3);

            let idx_0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
            let idx_1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

            let out0 = _mm512_permutex2var_epi64(pair01_02, idx_0, pair23_02);
            let out1 = _mm512_permutex2var_epi64(pair01_02, idx_1, pair23_02);
            let out2 = _mm512_permutex2var_epi64(pair01_13, idx_0, pair23_13);
            let out3 = _mm512_permutex2var_epi64(pair01_13, idx_1, pair23_13);

            unaligned::_mm512_storeu_si512_u8(d0, out0);
            unaligned::_mm512_storeu_si512_u8(d1, out1);
            unaligned::_mm512_storeu_si512_u8(d2, out2);
            unaligned::_mm512_storeu_si512_u8(d3, out3);
        }
    );
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecAvx512(__m512i, Avx512Descriptor);

impl U16SimdVec for U16VecAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, U16VecAvx512);
    const LEN: usize = 32;

    fn_avx!(
        fn load(d: Avx512Descriptor, mem: &[u16]) -> U16VecAvx512 {
            U16VecAvx512(
                unaligned::_mm512_loadu_si512_u16(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_avx!(
        fn splat(d: Avx512Descriptor, v: u16) -> U16VecAvx512 {
            U16VecAvx512(_mm512_set1_epi16(v as i16), d)
        }
    );

    fn_avx!(this: &U16VecAvx512, fn store(mem: &mut [u16]) {
        unaligned::_mm512_storeu_si512_u16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_avx!(
        fn store_interleaved_2(a: U16VecAvx512, b: U16VecAvx512, dest: &mut [u16]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<32>().0 else {
                panic!("dest too short");
            };
            let lo = _mm512_unpacklo_epi16(a.0, b.0);
            let hi = _mm512_unpackhi_epi16(a.0, b.0);
            let idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
            let idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
            let out0 = _mm512_permutex2var_epi64(lo, idx0, hi);
            let out1 = _mm512_permutex2var_epi64(lo, idx1, hi);

            unaligned::_mm512_storeu_si512_u16(d0, out0);
            unaligned::_mm512_storeu_si512_u16(d1, out1);
        }
    );

    fn_avx!(
        fn store_interleaved_3(
            a: U16VecAvx512,
            b: U16VecAvx512,
            c: U16VecAvx512,
            dest: &mut [u16],
        ) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<32>().0 else {
                panic!("dest too short");
            };

            let mask_a0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1,
            ));
            let mask_b0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5,
            ));
            let mask_c0 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1,
            ));

            let mask_a1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11,
            ));
            let mask_b1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1,
            ));
            let mask_c1 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1,
            ));

            let mask_a2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
            ));
            let mask_b2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
            ));
            let mask_c2 = _mm512_broadcast_i32x4(_mm_setr_epi8(
                -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
            ));

            let res0 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a0),
                    _mm512_shuffle_epi8(b.0, mask_b0),
                ),
                _mm512_shuffle_epi8(c.0, mask_c0),
            );
            let res1 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a1),
                    _mm512_shuffle_epi8(b.0, mask_b1),
                ),
                _mm512_shuffle_epi8(c.0, mask_c1),
            );
            let res2 = _mm512_or_si512(
                _mm512_or_si512(
                    _mm512_shuffle_epi8(a.0, mask_a2),
                    _mm512_shuffle_epi8(b.0, mask_b2),
                ),
                _mm512_shuffle_epi8(c.0, mask_c2),
            );

            let idx_a0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 0, 1);
            let part_a0 = _mm512_permutex2var_epi64(res0, idx_a0, res1);
            let idx_f0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 4, 5);
            let final0 = _mm512_permutex2var_epi64(part_a0, idx_f0, res2);

            let idx_a1 = _mm512_setr_epi64(2, 3, 10, 11, 4, 5, 0, 1);
            let part_a1 = _mm512_permutex2var_epi64(res1, idx_a1, res2);
            let idx_f1 = _mm512_setr_epi64(0, 1, 2, 3, 12, 13, 4, 5);
            let final1 = _mm512_permutex2var_epi64(part_a1, idx_f1, res0);

            let idx_a2 = _mm512_setr_epi64(4, 5, 14, 15, 6, 7, 0, 1);
            let part_a2 = _mm512_permutex2var_epi64(res2, idx_a2, res0);
            let idx_f2 = _mm512_setr_epi64(0, 1, 2, 3, 14, 15, 4, 5);
            let final2 = _mm512_permutex2var_epi64(part_a2, idx_f2, res1);

            unaligned::_mm512_storeu_si512_u16(d0, final0);
            unaligned::_mm512_storeu_si512_u16(d1, final1);
            unaligned::_mm512_storeu_si512_u16(d2, final2);
        }
    );

    fn_avx!(
        fn store_interleaved_4(
            a: U16VecAvx512,
            b: U16VecAvx512,
            c: U16VecAvx512,
            d: U16VecAvx512,
            dest: &mut [u16],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<32>().0 else {
                panic!("dest too short");
            };
            let ab_lo = _mm512_unpacklo_epi16(a.0, b.0);
            let ab_hi = _mm512_unpackhi_epi16(a.0, b.0);
            let cd_lo = _mm512_unpacklo_epi16(c.0, d.0);
            let cd_hi = _mm512_unpackhi_epi16(c.0, d.0);

            let abcd_0 = _mm512_unpacklo_epi32(ab_lo, cd_lo);
            let abcd_1 = _mm512_unpackhi_epi32(ab_lo, cd_lo);
            let abcd_2 = _mm512_unpacklo_epi32(ab_hi, cd_hi);
            let abcd_3 = _mm512_unpackhi_epi32(ab_hi, cd_hi);

            // Transpose 4x4 of 128-bit lanes (same as u8)
            let idx_even = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
            let idx_odd = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

            let pair01_02 = _mm512_permutex2var_epi64(abcd_0, idx_even, abcd_1);
            let pair01_13 = _mm512_permutex2var_epi64(abcd_0, idx_odd, abcd_1);
            let pair23_02 = _mm512_permutex2var_epi64(abcd_2, idx_even, abcd_3);
            let pair23_13 = _mm512_permutex2var_epi64(abcd_2, idx_odd, abcd_3);

            let idx_0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
            let idx_1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

            let out0 = _mm512_permutex2var_epi64(pair01_02, idx_0, pair23_02);
            let out1 = _mm512_permutex2var_epi64(pair01_02, idx_1, pair23_02);
            let out2 = _mm512_permutex2var_epi64(pair01_13, idx_0, pair23_13);
            let out3 = _mm512_permutex2var_epi64(pair01_13, idx_1, pair23_13);

            unaligned::_mm512_storeu_si512_u16(d0, out0);
            unaligned::_mm512_storeu_si512_u16(d1, out1);
            unaligned::_mm512_storeu_si512_u16(d2, out2);
            unaligned::_mm512_storeu_si512_u16(d3, out3);
        }
    );

    #[inline(always)]
    fn bitcast_i16(self) -> I16VecAvx512 {
        I16VecAvx512(self.0, self.1)
    }
}

impl SimdMask for MaskAvx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, MaskAvx512);

    fn_avx!(this: MaskAvx512, fn if_then_else_f32(if_true: F32VecAvx512, if_false: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_mask_blend_ps(this.0, if_false.0, if_true.0), this.1)
    });

    fn_avx!(this: MaskAvx512, fn if_then_else_i32(if_true: I32VecAvx512, if_false: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_mask_blend_epi32(this.0, if_false.0, if_true.0), this.1)
    });

    fn_avx!(this: MaskAvx512, fn maskz_i32(v: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_mask_set1_epi32(v.0, this.0, 0), this.1)
    });

    fn_avx!(this: MaskAvx512, fn all() -> bool {
        this.0 == 0b1111111111111111
    });

    fn_avx!(this: MaskAvx512, fn andnot(rhs: MaskAvx512) -> MaskAvx512 {
        MaskAvx512((!this.0) & rhs.0, this.1)
    });
}

impl BitAnd<MaskAvx512> for MaskAvx512 {
    type Output = MaskAvx512;
    fn_avx!(this: MaskAvx512, fn bitand(rhs: MaskAvx512) -> MaskAvx512 {
        MaskAvx512(this.0 & rhs.0, this.1)
    });
}

impl BitOr<MaskAvx512> for MaskAvx512 {
    type Output = MaskAvx512;
    fn_avx!(this: MaskAvx512, fn bitor(rhs: MaskAvx512) -> MaskAvx512 {
        MaskAvx512(this.0 | rhs.0, this.1)
    });
}

impl SimdMask16 for Mask16Avx512 {
    type Descriptor = Avx512Descriptor;
    crate::impl_isa_conversions!(avx512, Mask16Avx512);

    fn_avx!(this: Mask16Avx512, fn if_then_else_i16(if_true: I16VecAvx512, if_false: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_mask_blend_epi16(this.0, if_false.0, if_true.0), this.1)
    });

    fn_avx!(this: Mask16Avx512, fn maskz_i16(v: I16VecAvx512) -> I16VecAvx512 {
        I16VecAvx512(_mm512_mask_set1_epi16(v.0, this.0, 0), this.1)
    });

    fn_avx!(this: Mask16Avx512, fn all() -> bool {
        this.0 == 0xFFFF_FFFF
    });

    fn_avx!(this: Mask16Avx512, fn andnot(rhs: Mask16Avx512) -> Mask16Avx512 {
        Mask16Avx512((!this.0) & rhs.0, this.1)
    });
}

impl BitAnd<Mask16Avx512> for Mask16Avx512 {
    type Output = Mask16Avx512;
    fn_avx!(this: Mask16Avx512, fn bitand(rhs: Mask16Avx512) -> Mask16Avx512 {
        Mask16Avx512(this.0 & rhs.0, this.1)
    });
}

impl BitOr<Mask16Avx512> for Mask16Avx512 {
    type Output = Mask16Avx512;
    fn_avx!(this: Mask16Avx512, fn bitor(rhs: Mask16Avx512) -> Mask16Avx512 {
        Mask16Avx512(this.0 | rhs.0, this.1)
    });
}

crate::impl_raw_accessors!(F32VecAvx512, __m512, Avx512Descriptor);
crate::impl_raw_accessors!(I32VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(I16VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(U32VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(U64VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(U8VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(U16VecAvx512, __m512i, Avx512Descriptor);
crate::impl_raw_accessors!(MaskAvx512, __mmask16, Avx512Descriptor);
crate::impl_raw_accessors!(Mask16Avx512, __mmask32, Avx512Descriptor);
