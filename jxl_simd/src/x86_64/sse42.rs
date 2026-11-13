// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use crate::unaligned::x86_64 as unaligned;
use crate::{U32SimdVec, U64SimdVec, impl_f32_array_interface, impl_i16_array_interface};

// Safety invariant: this type is only ever constructed if sse4.2 is available.
#[derive(Clone, Copy, Debug)]
pub struct Sse42Descriptor(());

impl Sse42Descriptor {
    /// # Safety
    /// The caller must guarantee that the sse4.2 target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl crate::SimdDescriptor128 for Sse42Descriptor {}

impl SimdDescriptor for Sse42Descriptor {
    #[inline(always)]
    fn as_sse42(self) -> Option<Self> {
        Some(self)
    }

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

    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    fn new() -> Option<Self> {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: we just checked sse4.2.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "sse4.2")]
        #[inline(never)]
        unsafe fn inner<R>(d: Sse42Descriptor, f: impl FnOnce(Sse42Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees sse4.2.
        unsafe { inner(self, f) }
    }
}

// TODO(veluca): retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_sse42 {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident$(<$(const $cg:ident: $cg_ty:ty),* $(,)?>)?($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name$(<$(const $cg: $cg_ty),*>)?(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "sse4.2")]
            #[inline]
            fn inner$(<$(const $cg: $cg_ty),*>)?($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff sse4.2 is available.
            unsafe { inner$(::<$($cg),*>)?(self, $($arg),*) }
        }
    };
    (
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name($($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "sse4.2")]
            #[inline]
            fn inner($($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: the descriptor or vector argument witnesses that sse4.2 is available.
            unsafe { inner($($arg),*) }
        }
    };
}

/// Prepared 8-entry BF16 lookup table for SSE4.2.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Sse42(__m128i);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecSse42(__m128, Sse42Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskSse42(__m128, Sse42Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Sse42(__m128i, Sse42Descriptor);

impl F32SimdVec for F32VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, F32VecSse42);
    crate::impl_isa_array_conversions!(sse42, f32, 4);

    const LEN: usize = 4;

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[f32]) -> F32VecSse42 {
            F32VecSse42(unaligned::_mm_loadu_ps(mem.first_chunk().unwrap()), d)
        }
    );

    fn_sse42!(this: &F32VecSse42, fn store(mem: &mut [f32]) {
        unaligned::_mm_storeu_ps(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(
        fn store_interleaved_2(a: F32VecSse42, b: F32VecSse42, dest: &mut [f32]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
            // a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            // lo = [a0, b0, a1, b1], hi = [a2, b2, a3, b3]
            let lo = _mm_unpacklo_ps(a.0, b.0);
            let hi = _mm_unpackhi_ps(a.0, b.0);
            unaligned::_mm_storeu_ps(d0, lo);
            unaligned::_mm_storeu_ps(d1, hi);
        }
    );

    fn_sse42!(
        fn store_interleaved_3(a: F32VecSse42, b: F32VecSse42, c: F32VecSse42, dest: &mut [f32]) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
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

            unaligned::_mm_storeu_ps(d0, out0);
            unaligned::_mm_storeu_ps(d1, out1);
            unaligned::_mm_storeu_ps(d2, out2);
        }
    );

    fn_sse42!(
        fn store_interleaved_4(
            a: F32VecSse42,
            b: F32VecSse42,
            c: F32VecSse42,
            d: F32VecSse42,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
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

            unaligned::_mm_storeu_ps(d0, out0);
            unaligned::_mm_storeu_ps(d1, out1);
            unaligned::_mm_storeu_ps(d2, out2);
            unaligned::_mm_storeu_ps(d3, out3);
        }
    );

    fn_sse42!(
        fn store_interleaved_8(
            a: F32VecSse42,
            b: F32VecSse42,
            c: F32VecSse42,
            d: F32VecSse42,
            e: F32VecSse42,
            f: F32VecSse42,
            g: F32VecSse42,
            h: F32VecSse42,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
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

            unaligned::_mm_storeu_ps(d0, abcd_0);
            unaligned::_mm_storeu_ps(d1, efgh_0);
            unaligned::_mm_storeu_ps(d2, abcd_1);
            unaligned::_mm_storeu_ps(d3, efgh_1);
            unaligned::_mm_storeu_ps(d4, abcd_2);
            unaligned::_mm_storeu_ps(d5, efgh_2);
            unaligned::_mm_storeu_ps(d6, abcd_3);
            unaligned::_mm_storeu_ps(d7, efgh_3);
        }
    );

    fn_sse42!(
        fn load_deinterleaved_2(d: Sse42Descriptor, src: &[f32]) -> (F32VecSse42, F32VecSse42) {
            let [s0, s1, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // Input: [a0, b0, a1, b1, a2, b2, a3, b3]
            // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            let in0 = unaligned::_mm_loadu_ps(s0); // [a0, b0, a1, b1]
            let in1 = unaligned::_mm_loadu_ps(s1); // [a2, b2, a3, b3]

            // Shuffle to separate a and b components
            let a = _mm_shuffle_ps::<0x88>(in0, in1); // [a0, a1, a2, a3]
            let b = _mm_shuffle_ps::<0xDD>(in0, in1); // [b0, b1, b2, b3]

            (F32VecSse42(a, d), F32VecSse42(b, d))
        }
    );

    fn_sse42!(
        fn load_deinterleaved_3(
            d: Sse42Descriptor,
            src: &[f32],
        ) -> (F32VecSse42, F32VecSse42, F32VecSse42) {
            let [s0, s1, s2, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // Input: [a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3]
            // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3]
            let in0 = unaligned::_mm_loadu_ps(s0); // [a0, b0, c0, a1]
            let in1 = unaligned::_mm_loadu_ps(s1); // [b1, c1, a2, b2]
            let in2 = unaligned::_mm_loadu_ps(s2); // [c2, a3, b3, c3]

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

            (F32VecSse42(a, d), F32VecSse42(b, d), F32VecSse42(c, d))
        }
    );

    fn_sse42!(
        fn load_deinterleaved_4(
            d: Sse42Descriptor,
            src: &[f32],
        ) -> (F32VecSse42, F32VecSse42, F32VecSse42, F32VecSse42) {
            let [s0, s1, s2, s3, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // Input: [a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3]
            // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3], d = [d0, d1, d2, d3]
            let in0 = unaligned::_mm_loadu_ps(s0); // [a0, b0, c0, d0]
            let in1 = unaligned::_mm_loadu_ps(s1); // [a1, b1, c1, d1]
            let in2 = unaligned::_mm_loadu_ps(s2); // [a2, b2, c2, d2]
            let in3 = unaligned::_mm_loadu_ps(s3); // [a3, b3, c3, d3]

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

            (
                F32VecSse42(a, d),
                F32VecSse42(b, d),
                F32VecSse42(c, d),
                F32VecSse42(dv, d),
            )
        }
    );

    fn_sse42!(this: F32VecSse42, fn mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        this * mul + add
    });

    fn_sse42!(this: F32VecSse42, fn neg_mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        add - this * mul
    });

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: f32) -> F32VecSse42 {
            F32VecSse42(_mm_set1_ps(v), d)
        }
    );

    fn_sse42!(
        fn zero(d: Sse42Descriptor) -> F32VecSse42 {
            F32VecSse42(_mm_setzero_ps(), d)
        }
    );

    fn_sse42!(this: F32VecSse42, fn abs() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_andnot_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn floor() -> F32VecSse42 {
        F32VecSse42(_mm_floor_ps(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn sqrt() -> F32VecSse42 {
        F32VecSse42(_mm_sqrt_ps(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn neg() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_xor_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn copysign(sign: F32VecSse42) -> F32VecSse42 {
        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
        F32VecSse42(
            _mm_or_ps(
                _mm_andnot_ps(sign_mask, this.0),
                _mm_and_ps(sign_mask, sign.0),
            ),
            this.1,
        )
    });

    fn_sse42!(this: F32VecSse42, fn max(other: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_max_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn min(other: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_min_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn gt(other: F32VecSse42) -> MaskSse42 {
        MaskSse42(_mm_cmpgt_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn as_i32() -> I32VecSse42 {
        I32VecSse42(_mm_cvttps_epi32(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn bitcast_to_i32() -> I32VecSse42 {
        I32VecSse42(_mm_castps_si128(this.0), this.1)
    });

    fn_sse42!(
        fn prepare_table_bf16_8(_d: Sse42Descriptor, table: &[f32; 8]) -> Bf16Table8Sse42 {
            // Convert f32 table to BF16 packed in 128 bits (16 bytes for 8 entries)
            // BF16 is the high 16 bits of f32
            let ([t0, t1], []) = table.as_chunks::<4>() else {
                unreachable!();
            };
            let table_lo = unaligned::_mm_loadu_ps(t0);
            let table_hi = unaligned::_mm_loadu_ps(t1);
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
            Bf16Table8Sse42(_mm_unpacklo_epi64(bf16_lo, bf16_hi))
        }
    );

    fn_sse42!(
        fn table_lookup_bf16_8(
            d: Sse42Descriptor,
            table: Bf16Table8Sse42,
            indices: I32VecSse42,
        ) -> F32VecSse42 {
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
            F32VecSse42(_mm_castsi128_ps(result), d)
        }
    );

    fn_sse42!(this: F32VecSse42, fn round_store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        // Round to nearest integer
        let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(this.0);
        // Convert to i32
        let i32s = _mm_cvtps_epi32(rounded);
        // Pack i32 -> u16 -> u8 (use same vector twice, take lower half each time)
        let u16s = _mm_packus_epi32(i32s, i32s);
        let u8s = _mm_packus_epi16(u16s, u16s);
        // Store lower 4 bytes
        unaligned::_mm_storeu_si32_u8(out, u8s);
    });

    fn_sse42!(this: F32VecSse42, fn round_store_u16(dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        // Round to nearest integer
        let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(this.0);
        // Convert to i32
        let i32s = _mm_cvtps_epi32(rounded);
        // Pack i32 -> u16 (use same vector twice, take lower half)
        let u16s = _mm_packus_epi32(i32s, i32s);
        // Store lower 8 bytes (4 u16s)
        unaligned::_mm_storeu_si64_u16(out, u16s);
    });

    impl_f32_array_interface!();

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // TODO: use SIMD bit manipulation instead of scalar loop (see wasm32/simd128.rs)
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            result[i] = crate::f16::from_bits(mem[i]).to_f32();
        }
        Self::load(d, &result)
    }

    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        // TODO: use SIMD bit manipulation instead of scalar loop (see wasm32/simd128.rs)
        let mut tmp = [0.0f32; 4];
        self.store(&mut tmp);
        for i in 0..4 {
            dest[i] = crate::f16::from_f32(tmp[i]).to_bits();
        }
    }
}

impl Add<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn add(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn sub(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn mul(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn div(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn add_assign(rhs: F32VecSse42) {
        this.0 = _mm_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn sub_assign(rhs: F32VecSse42) {
        this.0 = _mm_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn mul_assign(rhs: F32VecSse42) {
        this.0 = _mm_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn div_assign(rhs: F32VecSse42) {
        this.0 = _mm_div_ps(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecSse42(__m128i, Sse42Descriptor);

impl I32SimdVec for I32VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, I32VecSse42);

    const LEN: usize = 4;

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[i32]) -> I32VecSse42 {
            I32VecSse42(
                unaligned::_mm_loadu_si128_i32(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_sse42!(
        fn load_from_i16(d: Sse42Descriptor, mem: &[i16]) -> I32VecSse42 {
            I32VecSse42(
                _mm_cvtepi16_epi32(unaligned::_mm_loadu_si64_i16(mem.first_chunk().unwrap())),
                d,
            )
        }
    );

    fn_sse42!(
        fn load_from_u16(d: Sse42Descriptor, mem: &[u16]) -> I32VecSse42 {
            I32VecSse42(
                _mm_cvtepu16_epi32(unaligned::_mm_loadu_si64_u16(mem.first_chunk().unwrap())),
                d,
            )
        }
    );

    fn_sse42!(this: &I32VecSse42, fn store(mem: &mut [i32]) {
        unaligned::_mm_storeu_si128_i32(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: i32) -> I32VecSse42 {
            I32VecSse42(_mm_set1_epi32(v), d)
        }
    );

    fn_sse42!(this: I32VecSse42, fn as_f32() -> F32VecSse42 {
        F32VecSse42(_mm_cvtepi32_ps(this.0), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn bitcast_to_f32() -> F32VecSse42 {
        F32VecSse42(_mm_castsi128_ps(this.0), this.1)
    });

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSse42 {
        U32VecSse42(self.0, self.1)
    }

    fn_sse42!(this: I32VecSse42, fn abs() -> I32VecSse42 {
        I32VecSse42(
            _mm_abs_epi32(
                this.0,
            ),
            this.1)
    });

    fn_sse42!(this: I32VecSse42, fn gt(rhs: I32VecSse42) -> MaskSse42 {
        MaskSse42(
            _mm_castsi128_ps(_mm_cmpgt_epi32(this.0, rhs.0)),
            this.1,
        )
    });

    fn_sse42!(this: I32VecSse42, fn lt_zero() -> MaskSse42 {
        I32VecSse42(_mm_setzero_si128(), this.1).gt(this)
    });

    fn_sse42!(this: I32VecSse42, fn eq(rhs: I32VecSse42) -> MaskSse42 {
        MaskSse42(
            _mm_castsi128_ps(_mm_cmpeq_epi32(this.0, rhs.0)),
            this.1,
        )
    });

    fn_sse42!(this: I32VecSse42, fn eq_zero() -> MaskSse42 {
        this.eq(I32VecSse42(_mm_setzero_si128(), this.1))
    });

    fn_sse42!(this: I32VecSse42, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecSse42 {
        I32VecSse42(_mm_slli_epi32::<AMOUNT_I>(this.0), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecSse42 {
        I32VecSse42(_mm_srai_epi32::<AMOUNT_I>(this.0), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn mul_wide_take_high(rhs: I32VecSse42) -> I32VecSse42 {
        let l = _mm_mul_epi32(this.0, rhs.0);
        let h = _mm_mul_epi32(_mm_srli_epi64::<32>(this.0), _mm_srli_epi64::<32>(rhs.0));
        let p0 = _mm_unpacklo_epi32(l, h);
        let p1 = _mm_unpackhi_epi32(l, h);
        I32VecSse42(_mm_unpackhi_epi64(p0, p1), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn store_u16(dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        // Truncate i32 -> u16 using shuffle
        let shuffle_mask =
            _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
        let u16s = _mm_shuffle_epi8(this.0, shuffle_mask);
        unaligned::_mm_storeu_si64_u16(out, u16s);
    });

    fn_sse42!(this: I32VecSse42, fn store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        // Truncate i32 -> u8 using shuffle
        let shuffle_mask =
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let u8s = _mm_shuffle_epi8(this.0, shuffle_mask);
        // Store lower 4 bytes
        unaligned::_mm_storeu_si32_u8(out, u8s);
    });
}

impl Add<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn add(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_add_epi32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn sub(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_sub_epi32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn mul(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_mullo_epi32(this.0, rhs.0), this.1)
    });
}

impl Neg for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn neg() -> I32VecSse42 {
        I32VecSse42(_mm_setzero_si128(), this.1) - this
    });
}

impl BitAnd<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitand(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_and_si128(this.0, rhs.0), this.1)
    });
}

impl BitOr<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitor(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_or_si128(this.0, rhs.0), this.1)
    });
}

impl BitXor<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitxor(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_xor_si128(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn add_assign(rhs: I32VecSse42) {
        this.0 = _mm_add_epi32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn sub_assign(rhs: I32VecSse42) {
        this.0 = _mm_sub_epi32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn mul_assign(rhs: I32VecSse42) {
        this.0 = _mm_mullo_epi32(this.0, rhs.0)
    });
}

impl BitAndAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitand_assign(rhs: I32VecSse42) {
        this.0 = _mm_and_si128(this.0, rhs.0)
    });
}

impl BitOrAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitor_assign(rhs: I32VecSse42) {
        this.0 = _mm_or_si128(this.0, rhs.0)
    });
}

impl BitXorAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitxor_assign(rhs: I32VecSse42) {
        this.0 = _mm_xor_si128(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecSse42(__m128i, Sse42Descriptor);

impl I16SimdVec for I16VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, I16VecSse42);
    crate::impl_isa_array_conversions!(sse42, i16, 8);

    const LEN: usize = 8;

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[i16]) -> I16VecSse42 {
            I16VecSse42(
                unaligned::_mm_loadu_si128_i16(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_sse42!(this: &I16VecSse42, fn store(mem: &mut [i16]) {
        unaligned::_mm_storeu_si128_i16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: i16) -> I16VecSse42 {
            I16VecSse42(_mm_set1_epi16(v), d)
        }
    );

    fn_sse42!(
        fn zero(d: Sse42Descriptor) -> I16VecSse42 {
            I16VecSse42(_mm_setzero_si128(), d)
        }
    );

    fn_sse42!(this: I16VecSse42, fn abs() -> I16VecSse42 {
        I16VecSse42(_mm_abs_epi16(this.0), this.1)
    });

    fn_sse42!(this: I16VecSse42, fn gt(rhs: I16VecSse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_cmpgt_epi16(this.0, rhs.0), this.1)
    });

    fn_sse42!(this: I16VecSse42, fn lt_zero() -> Mask16Sse42 {
        I16VecSse42(_mm_setzero_si128(), this.1).gt(this)
    });

    fn_sse42!(this: I16VecSse42, fn eq(rhs: I16VecSse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_cmpeq_epi16(this.0, rhs.0), this.1)
    });

    fn_sse42!(this: I16VecSse42, fn eq_zero() -> Mask16Sse42 {
        this.eq(I16VecSse42(_mm_setzero_si128(), this.1))
    });

    fn_sse42!(this: I16VecSse42, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecSse42 {
        I16VecSse42(_mm_slli_epi16::<AMOUNT_I>(this.0), this.1)
    });

    fn_sse42!(this: I16VecSse42, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecSse42 {
        I16VecSse42(_mm_srai_epi16::<AMOUNT_I>(this.0), this.1)
    });

    fn_sse42!(this: I16VecSse42, fn mul_wide_take_high(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_mulhi_epi16(this.0, rhs.0), this.1)
    });

    #[inline(always)]
    fn bitcast_u16(self) -> U16VecSse42 {
        U16VecSse42(self.0, self.1)
    }

    impl_i16_array_interface!();

    fn_sse42!(this: I16VecSse42, fn store_u8(dest: &mut [u8]) {
        let out = dest.first_chunk_mut::<8>().unwrap();
        let mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        let shuffled = _mm_shuffle_epi8(this.0, mask);
        unaligned::_mm_storeu_si64_u8(out, shuffled);
    });
}

impl Add<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn add(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_add_epi16(this.0, rhs.0), this.1)
    });
}

impl Sub<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn sub(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_sub_epi16(this.0, rhs.0), this.1)
    });
}

impl Neg for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn neg() -> I16VecSse42 {
        I16VecSse42(_mm_sub_epi16(_mm_setzero_si128(), this.0), this.1)
    });
}

impl BitAnd<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn bitand(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_and_si128(this.0, rhs.0), this.1)
    });
}

impl BitOr<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn bitor(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_or_si128(this.0, rhs.0), this.1)
    });
}

impl BitXor<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn bitxor(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_xor_si128(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn add_assign(rhs: I16VecSse42) {
        this.0 = _mm_add_epi16(this.0, rhs.0)
    });
}

impl SubAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn sub_assign(rhs: I16VecSse42) {
        this.0 = _mm_sub_epi16(this.0, rhs.0)
    });
}

impl BitAndAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn bitand_assign(rhs: I16VecSse42) {
        this.0 = _mm_and_si128(this.0, rhs.0)
    });
}

impl BitOrAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn bitor_assign(rhs: I16VecSse42) {
        this.0 = _mm_or_si128(this.0, rhs.0)
    });
}

impl BitXorAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn bitxor_assign(rhs: I16VecSse42) {
        this.0 = _mm_xor_si128(this.0, rhs.0)
    });
}

impl Mul<I16VecSse42> for I16VecSse42 {
    type Output = I16VecSse42;
    fn_sse42!(this: I16VecSse42, fn mul(rhs: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_mullo_epi16(this.0, rhs.0), this.1)
    });
}

impl MulAssign<I16VecSse42> for I16VecSse42 {
    fn_sse42!(this: &mut I16VecSse42, fn mul_assign(rhs: I16VecSse42) {
        this.0 = _mm_mullo_epi16(this.0, rhs.0);
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecSse42(__m128i, Sse42Descriptor);

impl U32SimdVec for U32VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, U32VecSse42);

    const LEN: usize = 4;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSse42 {
        I32VecSse42(self.0, self.1)
    }

    fn_sse42!(this: U32VecSse42, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U32VecSse42 {
        U32VecSse42(_mm_srli_epi32::<AMOUNT_I>(this.0), this.1)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecSse42(__m128i, Sse42Descriptor);

impl U64SimdVec for U64VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, U64VecSse42);

    const LEN: usize = 2;

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: u64) -> U64VecSse42 {
            U64VecSse42(_mm_set1_epi64x(v as i64), d)
        }
    );

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[u64]) -> U64VecSse42 {
            U64VecSse42(
                unaligned::_mm_loadu_si128_u64(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_sse42!(this: &U64VecSse42, fn store(mem: &mut [u64]) {
        unaligned::_mm_storeu_si128_u64(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(this: U64VecSse42, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecSse42 {
        U64VecSse42(_mm_slli_epi64::<AMOUNT_I>(this.0), this.1)
    });

    fn_sse42!(this: U64VecSse42, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecSse42 {
        U64VecSse42(_mm_srli_epi64::<AMOUNT_I>(this.0), this.1)
    });

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSse42 {
        U32VecSse42(self.0, self.1)
    }
}

impl Add<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    fn_sse42!(this: U64VecSse42, fn add(rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_add_epi64(this.0, rhs.0), this.1)
    });
}

impl AddAssign<U64VecSse42> for U64VecSse42 {
    fn_sse42!(this: &mut U64VecSse42, fn add_assign(rhs: U64VecSse42) {
        this.0 = _mm_add_epi64(this.0, rhs.0)
    });
}

impl BitAnd<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    fn_sse42!(this: U64VecSse42, fn bitand(rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_and_si128(this.0, rhs.0), this.1)
    });
}

impl BitAndAssign<U64VecSse42> for U64VecSse42 {
    fn_sse42!(this: &mut U64VecSse42, fn bitand_assign(rhs: U64VecSse42) {
        this.0 = _mm_and_si128(this.0, rhs.0)
    });
}

impl BitOr<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    fn_sse42!(this: U64VecSse42, fn bitor(rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_or_si128(this.0, rhs.0), this.1)
    });
}

impl BitOrAssign<U64VecSse42> for U64VecSse42 {
    fn_sse42!(this: &mut U64VecSse42, fn bitor_assign(rhs: U64VecSse42) {
        this.0 = _mm_or_si128(this.0, rhs.0)
    });
}

impl BitXor<U64VecSse42> for U64VecSse42 {
    type Output = U64VecSse42;
    fn_sse42!(this: U64VecSse42, fn bitxor(rhs: U64VecSse42) -> U64VecSse42 {
        U64VecSse42(_mm_xor_si128(this.0, rhs.0), this.1)
    });
}

impl BitXorAssign<U64VecSse42> for U64VecSse42 {
    fn_sse42!(this: &mut U64VecSse42, fn bitxor_assign(rhs: U64VecSse42) {
        this.0 = _mm_xor_si128(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecSse42(__m128i, Sse42Descriptor);

impl U8SimdVec for U8VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, U8VecSse42);
    const LEN: usize = 16;

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[u8]) -> U8VecSse42 {
            U8VecSse42(unaligned::_mm_loadu_si128_u8(mem.first_chunk().unwrap()), d)
        }
    );

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: u8) -> U8VecSse42 {
            U8VecSse42(_mm_set1_epi8(v as i8), d)
        }
    );

    fn_sse42!(this: &U8VecSse42, fn store(mem: &mut [u8]) {
        unaligned::_mm_storeu_si128_u8(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(
        fn store_interleaved_2(a: U8VecSse42, b: U8VecSse42, dest: &mut [u8]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            let lo = _mm_unpacklo_epi8(a.0, b.0);
            let hi = _mm_unpackhi_epi8(a.0, b.0);
            unaligned::_mm_storeu_si128_u8(d0, lo);
            unaligned::_mm_storeu_si128_u8(d1, hi);
        }
    );

    fn_sse42!(
        fn store_interleaved_3(a: U8VecSse42, b: U8VecSse42, c: U8VecSse42, dest: &mut [u8]) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };

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

            unaligned::_mm_storeu_si128_u8(d0, out0);
            unaligned::_mm_storeu_si128_u8(d1, out1);
            unaligned::_mm_storeu_si128_u8(d2, out2);
        }
    );

    fn_sse42!(
        fn store_interleaved_4(
            a: U8VecSse42,
            b: U8VecSse42,
            c: U8VecSse42,
            d: U8VecSse42,
            dest: &mut [u8],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
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

            unaligned::_mm_storeu_si128_u8(d0, out0);
            unaligned::_mm_storeu_si128_u8(d1, out1);
            unaligned::_mm_storeu_si128_u8(d2, out2);
            unaligned::_mm_storeu_si128_u8(d3, out3);
        }
    );
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecSse42(__m128i, Sse42Descriptor);

impl U16SimdVec for U16VecSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, U16VecSse42);
    const LEN: usize = 8;

    fn_sse42!(
        fn load(d: Sse42Descriptor, mem: &[u16]) -> U16VecSse42 {
            U16VecSse42(
                unaligned::_mm_loadu_si128_u16(mem.first_chunk().unwrap()),
                d,
            )
        }
    );

    fn_sse42!(
        fn splat(d: Sse42Descriptor, v: u16) -> U16VecSse42 {
            U16VecSse42(_mm_set1_epi16(v as i16), d)
        }
    );

    fn_sse42!(this: &U16VecSse42, fn store(mem: &mut [u16]) {
        unaligned::_mm_storeu_si128_u16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_sse42!(
        fn store_interleaved_2(a: U16VecSse42, b: U16VecSse42, dest: &mut [u16]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };
            let lo = _mm_unpacklo_epi16(a.0, b.0);
            let hi = _mm_unpackhi_epi16(a.0, b.0);
            unaligned::_mm_storeu_si128_u16(d0, lo);
            unaligned::_mm_storeu_si128_u16(d1, hi);
        }
    );

    fn_sse42!(
        fn store_interleaved_3(a: U16VecSse42, b: U16VecSse42, c: U16VecSse42, dest: &mut [u16]) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };

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

            unaligned::_mm_storeu_si128_u16(d0, out0);
            unaligned::_mm_storeu_si128_u16(d1, out1);
            unaligned::_mm_storeu_si128_u16(d2, out2);
        }
    );

    fn_sse42!(
        fn store_interleaved_4(
            a: U16VecSse42,
            b: U16VecSse42,
            c: U16VecSse42,
            d: U16VecSse42,
            dest: &mut [u16],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };
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

            unaligned::_mm_storeu_si128_u16(d0, out0);
            unaligned::_mm_storeu_si128_u16(d1, out1);
            unaligned::_mm_storeu_si128_u16(d2, out2);
            unaligned::_mm_storeu_si128_u16(d3, out3);
        }
    );

    #[inline(always)]
    fn bitcast_i16(self) -> I16VecSse42 {
        I16VecSse42(self.0, self.1)
    }
}

impl SimdMask for MaskSse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, MaskSse42);

    fn_sse42!(this: MaskSse42, fn if_then_else_f32(if_true: F32VecSse42, if_false: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_blendv_ps(if_false.0, if_true.0, this.0), this.1)
    });

    fn_sse42!(this: MaskSse42, fn if_then_else_i32(if_true: I32VecSse42, if_false: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_blendv_epi8(if_false.0, if_true.0, _mm_castps_si128(this.0)), this.1)
    });

    fn_sse42!(this: MaskSse42, fn maskz_i32(v: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_andnot_si128(_mm_castps_si128(this.0), v.0), this.1)
    });

    fn_sse42!(this: MaskSse42, fn all() -> bool {
        _mm_movemask_ps(this.0) == 0b1111
    });

    fn_sse42!(this: MaskSse42, fn andnot(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_andnot_ps(this.0, rhs.0), this.1)
    });
}

impl BitAnd<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    fn_sse42!(this: MaskSse42, fn bitand(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_and_ps(this.0, rhs.0), this.1)
    });
}

impl BitOr<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    fn_sse42!(this: MaskSse42, fn bitor(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_or_ps(this.0, rhs.0), this.1)
    });
}

impl SimdMask16 for Mask16Sse42 {
    type Descriptor = Sse42Descriptor;
    crate::impl_isa_conversions!(sse42, Mask16Sse42);

    fn_sse42!(this: Mask16Sse42, fn if_then_else_i16(if_true: I16VecSse42, if_false: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_blendv_epi8(if_false.0, if_true.0, this.0), this.1)
    });

    fn_sse42!(this: Mask16Sse42, fn maskz_i16(v: I16VecSse42) -> I16VecSse42 {
        I16VecSse42(_mm_andnot_si128(this.0, v.0), this.1)
    });

    fn_sse42!(this: Mask16Sse42, fn all() -> bool {
        _mm_movemask_epi8(this.0) == 0xFFFF
    });

    fn_sse42!(this: Mask16Sse42, fn andnot(rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_andnot_si128(this.0, rhs.0), this.1)
    });
}

impl BitAnd<Mask16Sse42> for Mask16Sse42 {
    type Output = Mask16Sse42;
    fn_sse42!(this: Mask16Sse42, fn bitand(rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_and_si128(this.0, rhs.0), this.1)
    });
}

impl BitOr<Mask16Sse42> for Mask16Sse42 {
    type Output = Mask16Sse42;
    fn_sse42!(this: Mask16Sse42, fn bitor(rhs: Mask16Sse42) -> Mask16Sse42 {
        Mask16Sse42(_mm_or_si128(this.0, rhs.0), this.1)
    });
}

crate::impl_raw_accessors!(F32VecSse42, __m128, Sse42Descriptor);
crate::impl_raw_accessors!(I32VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(I16VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(U32VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(U64VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(U8VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(U16VecSse42, __m128i, Sse42Descriptor);
crate::impl_raw_accessors!(MaskSse42, __m128, Sse42Descriptor);
crate::impl_raw_accessors!(Mask16Sse42, __m128i, Sse42Descriptor);
