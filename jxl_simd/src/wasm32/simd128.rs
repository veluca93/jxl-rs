// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::wasm32::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use crate::unaligned::wasm32 as unaligned;
use crate::{U32SimdVec, U64SimdVec, impl_i16_array_interface};

// Safety invariant: this type is only ever constructed if simd128 is available.
#[derive(Clone, Copy, Debug)]
pub struct Simd128Descriptor(());

impl Simd128Descriptor {
    /// # Safety
    /// The caller must guarantee that the "simd128" target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

macro_rules! fn_simd128 {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "simd128")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff simd128 is available.
            #[allow(unused_unsafe)]
            unsafe { inner(self, $($arg),*) }
        }
    };
    (
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        fn $name($($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "simd128")]
            #[inline]
            fn inner($($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: the descriptor or vector argument witnesses that simd128 is available.
            #[allow(unused_unsafe)]
            unsafe { inner($($arg),*) }
        }
    };
}

/// Prepared 8-entry BF16 lookup table for WASM SIMD128.
/// Contains 8 BF16 values packed into 16 bytes (v128).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Simd128(v128);

impl crate::SimdDescriptor128 for Simd128Descriptor {}

impl SimdDescriptor for Simd128Descriptor {
    #[inline(always)]
    fn as_simd128(self) -> Option<Self> {
        Some(self)
    }

    type F32Vec = F32VecSimd128;

    type I32Vec = I32VecSimd128;

    type I16Vec = I16VecSimd128;

    type U64Vec = U64VecSimd128;

    type U32Vec = U32VecSimd128;

    type U16Vec = U16VecSimd128;

    type U8Vec = U8VecSimd128;

    type Mask = MaskSimd128;
    type Mask16 = Mask16Simd128;
    type Bf16Table8 = Bf16Table8Simd128;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn new() -> Option<Self> {
        if cfg!(target_feature = "simd128") {
            // SAFETY: we just checked that simd128 is enabled at compile time.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    fn maybe_downgrade_256bit(self) -> Self {
        self
    }

    fn maybe_downgrade_128bit(self) -> Self {
        self
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecSimd128(v128, Simd128Descriptor);

impl F32SimdVec for F32VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, F32VecSimd128);
    crate::impl_isa_array_conversions!(simd128, f32, 4);

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(f32x4_splat(v), d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(f32x4_splat(0.0), d)
    }

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[f32]) -> F32VecSimd128 {
            F32VecSimd128(unaligned::v128_load_f32(mem.first_chunk().unwrap()), d)
        }
    );

    fn_simd128!(this: &F32VecSimd128, fn store(mem: &mut [f32]) {
        unaligned::v128_store_f32(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_simd128!(
        fn store_interleaved_2(a: F32VecSimd128, b: F32VecSimd128, dest: &mut [f32]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
            // a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            // out0 = [a0, b0, a1, b1], out1 = [a2, b2, a3, b3]
            let lo = i32x4_shuffle::<0, 4, 1, 5>(a.0, b.0);
            let hi = i32x4_shuffle::<2, 6, 3, 7>(a.0, b.0);
            unaligned::v128_store_f32(d0, lo);
            unaligned::v128_store_f32(d1, hi);
        }
    );

    fn_simd128!(
        fn store_interleaved_3(
            a: F32VecSimd128,
            b: F32VecSimd128,
            c: F32VecSimd128,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
            // a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3]
            // out0 = [a0, b0, c0, a1], out1 = [b1, c1, a2, b2], out2 = [c2, a3, b3, c3]
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a.0, b.0); // [a0, b0, a1, b1]
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a.0, b.0); // [a2, b2, a3, b3]
            let out0 = i32x4_shuffle::<0, 1, 4, 2>(ab_lo, c.0); // [a0, b0, c0, a1]
            let tmp1 = i32x4_shuffle::<3, 5, 0, 0>(ab_lo, c.0); // [b1, c1, ?, ?]
            let out1 = i32x4_shuffle::<0, 1, 4, 5>(tmp1, ab_hi); // [b1, c1, a2, b2]
            let out2 = i32x4_shuffle::<6, 2, 3, 7>(ab_hi, c.0); // [c2, a3, b3, c3]

            unaligned::v128_store_f32(d0, out0);
            unaligned::v128_store_f32(d1, out1);
            unaligned::v128_store_f32(d2, out2);
        }
    );

    fn_simd128!(
        fn store_interleaved_4(
            a: F32VecSimd128,
            b: F32VecSimd128,
            c: F32VecSimd128,
            d: F32VecSimd128,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };
            // a = [a0,a1,a2,a3], b = [b0,b1,b2,b3], c = [c0,c1,c2,c3], d = [d0,d1,d2,d3]
            // out = [a0,b0,c0,d0, a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3]
            // Stage 1: interleave pairs
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a.0, b.0); // [a0, b0, a1, b1]
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a.0, b.0); // [a2, b2, a3, b3]
            let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c.0, d.0); // [c0, d0, c1, d1]
            let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c.0, d.0); // [c2, d2, c3, d3]

            // Stage 2: interleave pairs of pairs (using 64-bit granularity)
            let out0 = i64x2_shuffle::<0, 2>(ab_lo, cd_lo); // [a0, b0, c0, d0]
            let out1 = i64x2_shuffle::<1, 3>(ab_lo, cd_lo); // [a1, b1, c1, d1]
            let out2 = i64x2_shuffle::<0, 2>(ab_hi, cd_hi); // [a2, b2, c2, d2]
            let out3 = i64x2_shuffle::<1, 3>(ab_hi, cd_hi); // [a3, b3, c3, d3]

            unaligned::v128_store_f32(d0, out0);
            unaligned::v128_store_f32(d1, out1);
            unaligned::v128_store_f32(d2, out2);
            unaligned::v128_store_f32(d3, out3);
        }
    );

    fn_simd128!(
        fn store_interleaved_8(
            a: F32VecSimd128,
            b: F32VecSimd128,
            c: F32VecSimd128,
            d: F32VecSimd128,
            e: F32VecSimd128,
            f: F32VecSimd128,
            g: F32VecSimd128,
            h: F32VecSimd128,
            dest: &mut [f32],
        ) {
            let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut::<4>().0 else {
                panic!("dest too short");
            };

            // Interleave adjacent pairs
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a.0, b.0);
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a.0, b.0);
            let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c.0, d.0);
            let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c.0, d.0);
            let ef_lo = i32x4_shuffle::<0, 4, 1, 5>(e.0, f.0);
            let ef_hi = i32x4_shuffle::<2, 6, 3, 7>(e.0, f.0);
            let gh_lo = i32x4_shuffle::<0, 4, 1, 5>(g.0, h.0);
            let gh_hi = i32x4_shuffle::<2, 6, 3, 7>(g.0, h.0);

            // Combine into 4-wide groups
            let abcd_0 = i64x2_shuffle::<0, 2>(ab_lo, cd_lo);
            let abcd_1 = i64x2_shuffle::<1, 3>(ab_lo, cd_lo);
            let abcd_2 = i64x2_shuffle::<0, 2>(ab_hi, cd_hi);
            let abcd_3 = i64x2_shuffle::<1, 3>(ab_hi, cd_hi);
            let efgh_0 = i64x2_shuffle::<0, 2>(ef_lo, gh_lo);
            let efgh_1 = i64x2_shuffle::<1, 3>(ef_lo, gh_lo);
            let efgh_2 = i64x2_shuffle::<0, 2>(ef_hi, gh_hi);
            let efgh_3 = i64x2_shuffle::<1, 3>(ef_hi, gh_hi);

            unaligned::v128_store_f32(d0, abcd_0);
            unaligned::v128_store_f32(d1, efgh_0);
            unaligned::v128_store_f32(d2, abcd_1);
            unaligned::v128_store_f32(d3, efgh_1);
            unaligned::v128_store_f32(d4, abcd_2);
            unaligned::v128_store_f32(d5, efgh_2);
            unaligned::v128_store_f32(d6, abcd_3);
            unaligned::v128_store_f32(d7, efgh_3);
        }
    );

    fn_simd128!(
        fn load_deinterleaved_2(
            d: Simd128Descriptor,
            src: &[f32],
        ) -> (F32VecSimd128, F32VecSimd128) {
            let [s0, s1, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // src = [a0, b0, a1, b1, a2, b2, a3, b3]
            let lo = unaligned::v128_load_f32(s0); // [a0, b0, a1, b1]
            let hi = unaligned::v128_load_f32(s1); // [a2, b2, a3, b3]
            let a = i32x4_shuffle::<0, 2, 4, 6>(lo, hi); // [a0, a1, a2, a3]
            let b = i32x4_shuffle::<1, 3, 5, 7>(lo, hi); // [b0, b1, b2, b3]
            (F32VecSimd128(a, d), F32VecSimd128(b, d))
        }
    );

    fn_simd128!(
        fn load_deinterleaved_3(
            d: Simd128Descriptor,
            src: &[f32],
        ) -> (F32VecSimd128, F32VecSimd128, F32VecSimd128) {
            let [s0, s1, s2, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // src = [a0,b0,c0, a1,b1,c1, a2,b2,c2, a3,b3,c3]
            // v0 = [a0, b0, c0, a1], v1 = [b1, c1, a2, b2], v2 = [c2, a3, b3, c3]
            let v0 = unaligned::v128_load_f32(s0);
            let v1 = unaligned::v128_load_f32(s1);
            let v2 = unaligned::v128_load_f32(s2);

            // a = [a0, a1, a2, a3] = v0[0], v0[3], v1[2], v2[1]
            let a01 = i32x4_shuffle::<0, 3, 0, 0>(v0, v0); // [a0, a1, ?, ?]
            let a23 = i32x4_shuffle::<2, 5, 0, 0>(v1, v2); // [a2, a3, ?, ?]
            let a = i64x2_shuffle::<0, 2>(a01, a23);

            // b = [b0, b1, b2, b3] = v0[1], v1[0], v1[3], v2[2]
            let b01 = i32x4_shuffle::<1, 4, 0, 0>(v0, v1); // [b0, b1, ?, ?]
            let b23 = i32x4_shuffle::<3, 6, 0, 0>(v1, v2); // [b2, b3, ?, ?]
            let b = i64x2_shuffle::<0, 2>(b01, b23);

            // c = [c0, c1, c2, c3] = v0[2], v1[1], v2[0], v2[3]
            let c01 = i32x4_shuffle::<2, 5, 0, 0>(v0, v1); // [c0, c1, ?, ?]
            let c23 = i32x4_shuffle::<0, 3, 0, 0>(v2, v2); // [c2, c3, ?, ?]
            let c = i64x2_shuffle::<0, 2>(c01, c23);

            (
                F32VecSimd128(a, d),
                F32VecSimd128(b, d),
                F32VecSimd128(c, d),
            )
        }
    );

    fn_simd128!(
        fn load_deinterleaved_4(
            d: Simd128Descriptor,
            src: &[f32],
        ) -> (F32VecSimd128, F32VecSimd128, F32VecSimd128, F32VecSimd128) {
            let [s0, s1, s2, s3, ..] = src.as_chunks::<4>().0 else {
                panic!("src too short");
            };
            // src = [a0,b0,c0,d0, a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3]
            let v0 = unaligned::v128_load_f32(s0); // [a0, b0, c0, d0]
            let v1 = unaligned::v128_load_f32(s1); // [a1, b1, c1, d1]
            let v2 = unaligned::v128_load_f32(s2); // [a2, b2, c2, d2]
            let v3 = unaligned::v128_load_f32(s3); // [a3, b3, c3, d3]

            // Stage 1: interleave pairs
            let ac02 = i32x4_shuffle::<0, 4, 2, 6>(v0, v1); // [a0, a1, c0, c1]
            let bd02 = i32x4_shuffle::<1, 5, 3, 7>(v0, v1); // [b0, b1, d0, d1]
            let ac13 = i32x4_shuffle::<0, 4, 2, 6>(v2, v3); // [a2, a3, c2, c3]
            let bd13 = i32x4_shuffle::<1, 5, 3, 7>(v2, v3); // [b2, b3, d2, d3]

            // Stage 2: combine
            let a = i64x2_shuffle::<0, 2>(ac02, ac13); // [a0, a1, a2, a3]
            let b = i64x2_shuffle::<0, 2>(bd02, bd13); // [b0, b1, b2, b3]
            let c = i64x2_shuffle::<1, 3>(ac02, ac13); // [c0, c1, c2, c3]
            let dd = i64x2_shuffle::<1, 3>(bd02, bd13); // [d0, d1, d2, d3]
            (
                F32VecSimd128(a, d),
                F32VecSimd128(b, d),
                F32VecSimd128(c, d),
                F32VecSimd128(dd, d),
            )
        }
    );

    crate::impl_f32_array_interface!();

    #[inline(always)]
    fn mul_add(self, mul: F32VecSimd128, add: F32VecSimd128) -> F32VecSimd128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            F32VecSimd128(f32x4_relaxed_madd(self.0, mul.0, add.0), self.1)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            F32VecSimd128(f32x4_add(f32x4_mul(self.0, mul.0), add.0), self.1)
        }
    }

    #[inline(always)]
    fn neg_mul_add(self, mul: F32VecSimd128, add: F32VecSimd128) -> F32VecSimd128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            F32VecSimd128(f32x4_relaxed_nmadd(self.0, mul.0, add.0), self.1)
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            F32VecSimd128(f32x4_sub(add.0, f32x4_mul(self.0, mul.0)), self.1)
        }
    }

    #[inline(always)]
    fn abs(self) -> F32VecSimd128 {
        F32VecSimd128(f32x4_abs(self.0), self.1)
    }

    #[inline(always)]
    fn floor(self) -> F32VecSimd128 {
        F32VecSimd128(f32x4_floor(self.0), self.1)
    }

    #[inline(always)]
    fn sqrt(self) -> F32VecSimd128 {
        F32VecSimd128(f32x4_sqrt(self.0), self.1)
    }

    #[inline(always)]
    fn neg(self) -> F32VecSimd128 {
        F32VecSimd128(f32x4_neg(self.0), self.1)
    }

    #[inline(always)]
    fn copysign(self, sign: F32VecSimd128) -> F32VecSimd128 {
        // Select sign bit from `sign`, magnitude from `self`
        let sign_mask = u32x4_splat(0x8000_0000);
        F32VecSimd128(v128_bitselect(sign.0, self.0, sign_mask), self.1)
    }

    #[inline(always)]
    fn max(self, other: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_max(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn min(self, other: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_min(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn gt(self, other: F32VecSimd128) -> MaskSimd128 {
        MaskSimd128(f32x4_gt(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn as_i32(self) -> I32VecSimd128 {
        I32VecSimd128(i32x4_trunc_sat_f32x4(self.0), self.1)
    }

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSimd128 {
        // v128 is untyped; no conversion needed, just reinterpret.
        I32VecSimd128(self.0, self.1)
    }

    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        let rounded = f32x4_nearest(self.0);
        let i32s = i32x4_trunc_sat_f32x4(rounded);
        // Saturate i32 -> i16 (signed narrow, preserving sign for next stage)
        let i16s = i16x8_narrow_i32x4(i32s, i32s);
        // Saturate i16 -> u8 (signed-to-unsigned narrow, clamping to [0, 255])
        let u8s = u8x16_narrow_i16x8(i16s, i16s);
        *dest.first_chunk_mut::<4>().unwrap() = u32x4_extract_lane::<0>(u8s).to_ne_bytes();
    }

    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        let rounded = f32x4_nearest(self.0);
        let i32s = i32x4_trunc_sat_f32x4(rounded);
        // Saturate i32 -> u16 (narrow to 16-bit)
        let u16s = u16x8_narrow_i32x4(i32s, i32s);
        // Store lower 8 bytes (4 u16s)
        let bytes = u64x2_extract_lane::<0>(u16s).to_ne_bytes();
        for (dst, chunk) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *dst = u16::from_ne_bytes(*chunk);
        }
    }

    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        let abs = f32x4_abs(self.0);
        let sign = v128_and(u32x4_shr(self.0, 16), u32x4_splat(0x8000));

        // Round to nearest even
        let abs_shifted = u32x4_shr(abs, 13);
        let lsb = v128_and(abs_shifted, u32x4_splat(1));
        let rounded = i32x4_add(abs, i32x4_add(lsb, u32x4_splat(0x0FFF)));

        // Normal: shift exp+mant into f16 position, rebias, clamp to inf (0x7C00)
        let normal = i32x4_min(
            i32x4_sub(u32x4_shr(rounded, 13), u32x4_splat(112 << 10)),
            i32x4_splat(0x7C00),
        );

        // Subnormal (|x| < 2^-14): magic number trick to denormalize
        // Adding 2^-14 normalizes the mantissa via float rounding,
        // then integer-subtract the bias and shift to get the 10-bit mantissa.
        let magic = u32x4_splat(113 << 23);
        let subnorm = u32x4_shr(i32x4_sub(f32x4_add(abs, magic), magic), 13);

        // Inf/NaN (exp==255): set f16 exp=31, keep top 10 mantissa bits
        let infnan = v128_or(
            u32x4_splat(0x7C00),
            v128_and(abs_shifted, u32x4_splat(0x03FF)),
        );

        let is_subnorm = i32x4_lt(abs, u32x4_splat(0x38800000));
        let is_infnan = i32x4_gt(abs, u32x4_splat(0x7F7FFFFF));
        let result = v128_bitselect(subnorm, normal, is_subnorm);
        let result = v128_bitselect(infnan, result, is_infnan);
        let h = v128_or(sign, result);

        // Narrow u32->u16 and store
        let packed = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0>(h, h);
        let bytes = u64x2_extract_lane::<0>(packed).to_ne_bytes();
        for (dst, chunk) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *dst = u16::from_ne_bytes(*chunk);
        }
    }

    fn_simd128!(
        fn load_f16_bits(d: Simd128Descriptor, mem: &[u16]) -> F32VecSimd128 {
            let lo = unaligned::v128_load64_splat_u16(mem.first_chunk().unwrap());
            let zero = i32x4_splat(0);
            let wide =
                i8x16_shuffle::<0, 1, 16, 16, 2, 3, 16, 16, 4, 5, 16, 16, 6, 7, 16, 16>(lo, zero);
            let sign = i32x4_shl(v128_and(wide, u32x4_splat(0x8000)), 16);
            let mag = v128_and(wide, u32x4_splat(0x7FFF));
            let mag_shifted = i32x4_shl(mag, 13);

            // Normal: shift exp+mant into f32 position, rebias exponent (+112)
            let normal = i32x4_add(mag_shifted, u32x4_splat(112 << 23));

            // Subnormal (exp==0): magic number trick to normalize
            let magic = u32x4_splat(113 << 23);
            let subnorm = f32x4_sub(v128_or(magic, mag_shifted), magic);

            // Inf/NaN (exp==31): normal path gives f32 exp=143, need 255; add 112<<23
            let infnan = i32x4_add(normal, u32x4_splat(112 << 23));

            let is_subnorm = i32x4_lt(mag, u32x4_splat(0x0400));
            let is_infnan = i32x4_gt(mag, u32x4_splat(0x7BFF));
            let result = v128_bitselect(subnorm, normal, is_subnorm);
            let result = v128_bitselect(infnan, result, is_infnan);

            F32VecSimd128(v128_or(sign, result), d)
        }
    );

    fn_simd128!(
        fn prepare_table_bf16_8(_d: Simd128Descriptor, table: &[f32; 8]) -> Bf16Table8Simd128 {
            // Convert f32 table to BF16 packed in 128 bits (16 bytes for 8 entries)
            // BF16 is the high 16 bits of f32
            let ([lo, hi], []) = table.as_chunks::<4>() else {
                unreachable!()
            };
            let table_lo = unaligned::v128_load_f32(lo);
            let table_hi = unaligned::v128_load_f32(hi);

            // Shift right by 16 to get BF16 values, then narrow to 16-bit
            let lo_shifted = u32x4_shr(table_lo, 16);
            let hi_shifted = u32x4_shr(table_hi, 16);

            // Narrow u32 -> u16, packing both halves
            Bf16Table8Simd128(u16x8_narrow_i32x4(lo_shifted, hi_shifted))
        }
    );

    #[inline(always)]
    fn table_lookup_bf16_8(
        d: Simd128Descriptor,
        table: Bf16Table8Simd128,
        indices: I32VecSimd128,
    ) -> Self {
        // Build shuffle mask efficiently using arithmetic on 32-bit indices.
        // For each index i (0-7), we need to select bytes [2*i, 2*i+1] from bf16_table
        // and place them in the high 16 bits of each 32-bit f32 lane (bytes 2,3),
        // with bytes 0,1 set to zero (using out-of-range index which gives 0 in swizzle).
        //
        // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
        // As a 32-bit value: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
        //                  = (i << 17) | (i << 25) | 0x01008080
        let shl17 = i32x4_shl(indices.0, 17);
        let shl25 = i32x4_shl(indices.0, 25);
        let base = u32x4_splat(0x01008080);
        let shuffle_mask = v128_or(v128_or(shl17, shl25), base);

        // Perform the table lookup (out of range indices give 0)
        F32VecSimd128(i8x16_swizzle(table.0, shuffle_mask), d)
    }
}

impl Add<F32VecSimd128> for F32VecSimd128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_add(self.0, rhs.0), self.1)
    }
}

impl Sub<F32VecSimd128> for F32VecSimd128 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_sub(self.0, rhs.0), self.1)
    }
}

impl Mul<F32VecSimd128> for F32VecSimd128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_mul(self.0, rhs.0), self.1)
    }
}

impl Div<F32VecSimd128> for F32VecSimd128 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: F32VecSimd128) -> F32VecSimd128 {
        F32VecSimd128(f32x4_div(self.0, rhs.0), self.1)
    }
}

impl AddAssign<F32VecSimd128> for F32VecSimd128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: F32VecSimd128) {
        self.0 = f32x4_add(self.0, rhs.0);
    }
}

impl SubAssign<F32VecSimd128> for F32VecSimd128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: F32VecSimd128) {
        self.0 = f32x4_sub(self.0, rhs.0);
    }
}

impl MulAssign<F32VecSimd128> for F32VecSimd128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: F32VecSimd128) {
        self.0 = f32x4_mul(self.0, rhs.0);
    }
}

impl DivAssign<F32VecSimd128> for F32VecSimd128 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: F32VecSimd128) {
        self.0 = f32x4_div(self.0, rhs.0);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecSimd128(v128, Simd128Descriptor);

impl I32SimdVec for I32VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, I32VecSimd128);

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(i32x4_splat(v), d)
    }

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[i32]) -> I32VecSimd128 {
            I32VecSimd128(unaligned::v128_load_i32(mem.first_chunk().unwrap()), d)
        }
    );

    fn_simd128!(
        fn load_from_i16(d: Simd128Descriptor, mem: &[i16]) -> I32VecSimd128 {
            I32VecSimd128(
                i32x4_extend_low_i16x8(unaligned::v128_load64_zero_i16(mem.first_chunk().unwrap())),
                d,
            )
        }
    );

    fn_simd128!(
        fn load_from_u16(d: Simd128Descriptor, mem: &[u16]) -> I32VecSimd128 {
            I32VecSimd128(
                u32x4_extend_low_u16x8(unaligned::v128_load64_zero_u16(mem.first_chunk().unwrap())),
                d,
            )
        }
    );

    fn_simd128!(this: &I32VecSimd128, fn store(mem: &mut [i32]) {
        unaligned::v128_store_i32(mem.first_chunk_mut().unwrap(), this.0);
    });

    #[inline(always)]
    fn abs(self) -> I32VecSimd128 {
        I32VecSimd128(i32x4_abs(self.0), self.1)
    }

    #[inline(always)]
    fn as_f32(self) -> F32VecSimd128 {
        F32VecSimd128(f32x4_convert_i32x4(self.0), self.1)
    }

    #[inline(always)]
    fn bitcast_to_f32(self) -> F32VecSimd128 {
        // v128 is untyped; no conversion needed.
        F32VecSimd128(self.0, self.1)
    }

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSimd128 {
        // v128 is untyped; no conversion needed.
        U32VecSimd128(self.0, self.1)
    }

    #[inline(always)]
    fn gt(self, other: I32VecSimd128) -> MaskSimd128 {
        MaskSimd128(i32x4_gt(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn lt_zero(self) -> MaskSimd128 {
        MaskSimd128(i32x4_lt(self.0, i32x4_splat(0)), self.1)
    }

    #[inline(always)]
    fn eq(self, other: I32VecSimd128) -> MaskSimd128 {
        MaskSimd128(i32x4_eq(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn eq_zero(self) -> MaskSimd128 {
        MaskSimd128(i32x4_eq(self.0, i32x4_splat(0)), self.1)
    }

    #[inline(always)]
    fn mul_wide_take_high(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        // Multiply pairs and take the high 32 bits of each 64-bit result
        let lo = i64x2_extmul_low_i32x4(self.0, rhs.0); // [a0*b0, a1*b1] as i64
        let hi = i64x2_extmul_high_i32x4(self.0, rhs.0); // [a2*b2, a3*b3] as i64
        // Extract high 32 bits: shift right by 32, then shuffle to pack
        // After shift: lo = [hi(a0*b0), 0, hi(a1*b1), 0] as i32 view
        // We want: [hi(a0*b0), hi(a1*b1), hi(a2*b2), hi(a3*b3)]
        let lo_shifted = i64x2_shr(lo, 32);
        let hi_shifted = i64x2_shr(hi, 32);
        // Pack: take lanes 0,2 from lo_shifted and 0,2 from hi_shifted
        // lo_shifted as i32x4: [hi0, 0, hi1, 0]
        // hi_shifted as i32x4: [hi2, 0, hi3, 0]
        I32VecSimd128(i32x4_shuffle::<0, 2, 4, 6>(lo_shifted, hi_shifted), self.1)
    }

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i32x4_shl(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i32x4_shr(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        let out = dest.first_chunk_mut::<4>().unwrap();
        // Truncating narrow: take the lower 16 bits of each 32-bit lane via shuffle.
        // Input bytes: [l0,l1,h0,h1, l2,l3,h2,h3, l4,l5,h4,h5, l6,l7,h6,h7]
        // Output bytes (lower 8): [l0,l1,l2,l3,l4,l5,l6,l7]
        let packed =
            i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0>(self.0, self.0);
        let bytes = u64x2_extract_lane::<0>(packed).to_ne_bytes();
        for (dst, chunk) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *dst = u16::from_ne_bytes(*chunk);
        }
    }

    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        // Truncating narrow i32 -> u8: take the lowest byte of each 32-bit lane.
        // Input bytes: [b0,_,_,_, b1,_,_,_, b2,_,_,_, b3,_,_,_]
        // Output bytes (lower 4): [b0, b1, b2, b3]
        let packed =
            i8x16_shuffle::<0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(self.0, self.0);
        *dest.first_chunk_mut::<4>().unwrap() = u32x4_extract_lane::<0>(packed).to_ne_bytes();
    }
}

impl Add<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn add(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(i32x4_add(self.0, rhs.0), self.1)
    }
}

impl Sub<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn sub(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(i32x4_sub(self.0, rhs.0), self.1)
    }
}

impl Mul<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn mul(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(i32x4_mul(self.0, rhs.0), self.1)
    }
}

impl Neg for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn neg(self) -> I32VecSimd128 {
        I32VecSimd128(i32x4_neg(self.0), self.1)
    }
}

impl BitAnd<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn bitand(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(v128_and(self.0, rhs.0), self.1)
    }
}

impl BitOr<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn bitor(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(v128_or(self.0, rhs.0), self.1)
    }
}

impl BitXor<I32VecSimd128> for I32VecSimd128 {
    type Output = I32VecSimd128;
    #[inline(always)]
    fn bitxor(self, rhs: I32VecSimd128) -> I32VecSimd128 {
        I32VecSimd128(v128_xor(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = i32x4_add(self.0, rhs.0)
    }
}

impl SubAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = i32x4_sub(self.0, rhs.0)
    }
}

impl MulAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = i32x4_mul(self.0, rhs.0)
    }
}

impl BitAndAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = v128_and(self.0, rhs.0);
    }
}

impl BitOrAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = v128_or(self.0, rhs.0);
    }
}

impl BitXorAssign<I32VecSimd128> for I32VecSimd128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I32VecSimd128) {
        self.0 = v128_xor(self.0, rhs.0);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecSimd128(v128, Simd128Descriptor);

impl I16SimdVec for I16VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, I16VecSimd128);
    crate::impl_isa_array_conversions!(simd128, i16, 8);

    const LEN: usize = 8;

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[i16]) -> I16VecSimd128 {
            I16VecSimd128(unaligned::v128_load_i16(mem.first_chunk().unwrap()), d)
        }
    );

    fn_simd128!(this: &I16VecSimd128, fn store(mem: &mut [i16]) {
        unaligned::v128_store_i16(mem.first_chunk_mut().unwrap(), this.0);
    });

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i16) -> Self {
        Self(i16x8_splat(v), d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(i16x8_splat(0), d)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(i16x8_abs(self.0), self.1)
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask16Simd128 {
        Mask16Simd128(i16x8_gt(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn lt_zero(self) -> Mask16Simd128 {
        Mask16Simd128(i16x8_lt(self.0, i16x8_splat(0)), self.1)
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask16Simd128 {
        Mask16Simd128(i16x8_eq(self.0, other.0), self.1)
    }

    #[inline(always)]
    fn eq_zero(self) -> Mask16Simd128 {
        Mask16Simd128(i16x8_eq(self.0, i16x8_splat(0)), self.1)
    }

    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        let lo = i32x4_extmul_low_i16x8(self.0, rhs.0);
        let hi = i32x4_extmul_high_i16x8(self.0, rhs.0);
        Self(
            i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(lo, hi),
            self.1,
        )
    }

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i16x8_shl(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i16x8_shr(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn bitcast_u16(self) -> U16VecSimd128 {
        U16VecSimd128(self.0, self.1)
    }

    impl_i16_array_interface!();

    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::LEN);
        let packed =
            i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0>(self.0, self.0);
        let val = u64x2_extract_lane::<0>(packed);
        dest[..8].copy_from_slice(&val.to_ne_bytes());
    }
}

impl Add<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn add(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(i16x8_add(self.0, rhs.0), self.1)
    }
}

impl Sub<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn sub(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(i16x8_sub(self.0, rhs.0), self.1)
    }
}

impl Neg for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn neg(self) -> I16VecSimd128 {
        I16VecSimd128(i16x8_neg(self.0), self.1)
    }
}

impl BitAnd<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn bitand(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(v128_and(self.0, rhs.0), self.1)
    }
}

impl BitOr<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn bitor(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(v128_or(self.0, rhs.0), self.1)
    }
}

impl BitXor<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn bitxor(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(v128_xor(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = i16x8_add(self.0, rhs.0);
    }
}

impl SubAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = i16x8_sub(self.0, rhs.0);
    }
}

impl BitAndAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = v128_and(self.0, rhs.0);
    }
}

impl BitOrAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = v128_or(self.0, rhs.0);
    }
}

impl BitXorAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = v128_xor(self.0, rhs.0);
    }
}

impl Mul<I16VecSimd128> for I16VecSimd128 {
    type Output = I16VecSimd128;
    #[inline(always)]
    fn mul(self, rhs: I16VecSimd128) -> I16VecSimd128 {
        I16VecSimd128(i16x8_mul(self.0, rhs.0), self.1)
    }
}

impl MulAssign<I16VecSimd128> for I16VecSimd128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I16VecSimd128) {
        self.0 = i16x8_mul(self.0, rhs.0);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecSimd128(v128, Simd128Descriptor);

impl U32SimdVec for U32VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, U32VecSimd128);

    const LEN: usize = 4;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSimd128 {
        // v128 is untyped; no conversion needed.
        I32VecSimd128(self.0, self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(u32x4_shr(self.0, AMOUNT_U), self.1)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecSimd128(v128, Simd128Descriptor);

impl U64SimdVec for U64VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, U64VecSimd128);

    const LEN: usize = 2;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u64) -> Self {
        Self(i64x2_splat(v as i64), d)
    }

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[u64]) -> U64VecSimd128 {
            U64VecSimd128(unaligned::v128_load_u64(mem.first_chunk().unwrap()), d)
        }
    );

    fn_simd128!(this: &U64VecSimd128, fn store(mem: &mut [u64]) {
        unaligned::v128_store_u64(mem.first_chunk_mut().unwrap(), this.0);
    });

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i64x2_shl(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(u64x2_shr(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSimd128 {
        U32VecSimd128(self.0, self.1)
    }
}

impl Add<U64VecSimd128> for U64VecSimd128 {
    type Output = U64VecSimd128;
    #[inline(always)]
    fn add(self, rhs: U64VecSimd128) -> U64VecSimd128 {
        U64VecSimd128(i64x2_add(self.0, rhs.0), self.1)
    }
}

impl AddAssign<U64VecSimd128> for U64VecSimd128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: U64VecSimd128) {
        self.0 = i64x2_add(self.0, rhs.0);
    }
}

impl BitAnd<U64VecSimd128> for U64VecSimd128 {
    type Output = U64VecSimd128;
    #[inline(always)]
    fn bitand(self, rhs: U64VecSimd128) -> U64VecSimd128 {
        U64VecSimd128(v128_and(self.0, rhs.0), self.1)
    }
}

impl BitAndAssign<U64VecSimd128> for U64VecSimd128 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: U64VecSimd128) {
        self.0 = v128_and(self.0, rhs.0);
    }
}

impl BitOr<U64VecSimd128> for U64VecSimd128 {
    type Output = U64VecSimd128;
    #[inline(always)]
    fn bitor(self, rhs: U64VecSimd128) -> U64VecSimd128 {
        U64VecSimd128(v128_or(self.0, rhs.0), self.1)
    }
}

impl BitOrAssign<U64VecSimd128> for U64VecSimd128 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: U64VecSimd128) {
        self.0 = v128_or(self.0, rhs.0);
    }
}

impl BitXor<U64VecSimd128> for U64VecSimd128 {
    type Output = U64VecSimd128;
    #[inline(always)]
    fn bitxor(self, rhs: U64VecSimd128) -> U64VecSimd128 {
        U64VecSimd128(v128_xor(self.0, rhs.0), self.1)
    }
}

impl BitXorAssign<U64VecSimd128> for U64VecSimd128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: U64VecSimd128) {
        self.0 = v128_xor(self.0, rhs.0);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecSimd128(v128, Simd128Descriptor);

impl U8SimdVec for U8VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, U8VecSimd128);
    const LEN: usize = 16;

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[u8]) -> U8VecSimd128 {
            U8VecSimd128(unaligned::v128_load_u8(mem.first_chunk().unwrap()), d)
        }
    );

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(u8x16_splat(v), d)
    }

    fn_simd128!(this: &U8VecSimd128, fn store(mem: &mut [u8]) {
        unaligned::v128_store_u8(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_simd128!(
        fn store_interleaved_2(a: U8VecSimd128, b: U8VecSimd128, dest: &mut [u8]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            let lo =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a.0, b.0);
            let hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                a.0, b.0,
            );
            unaligned::v128_store_u8(d0, lo);
            unaligned::v128_store_u8(d1, hi);
        }
    );

    fn_simd128!(
        fn store_interleaved_3(a: U8VecSimd128, b: U8VecSimd128, c: U8VecSimd128, dest: &mut [u8]) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            // 3-way byte interleave using 2 shuffles per output vector.
            let ab0 =
                i8x16_shuffle::<0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5>(a.0, b.0);
            let out0 =
                i8x16_shuffle::<0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15>(ab0, c.0);

            let ab1 =
                i8x16_shuffle::<21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26>(a.0, b.0);
            let out1 =
                i8x16_shuffle::<0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15>(ab1, c.0);

            let ab2 =
                i8x16_shuffle::<0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0>(a.0, b.0);
            let out2 =
                i8x16_shuffle::<26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31>(ab2, c.0);

            unaligned::v128_store_u8(d0, out0);
            unaligned::v128_store_u8(d1, out1);
            unaligned::v128_store_u8(d2, out2);
        }
    );

    fn_simd128!(
        fn store_interleaved_4(
            a: U8VecSimd128,
            b: U8VecSimd128,
            c: U8VecSimd128,
            d: U8VecSimd128,
            dest: &mut [u8],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<16>().0 else {
                panic!("dest too short");
            };
            // Stage 1: interleave a,b and c,d
            let ab_lo =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a.0, b.0);
            let ab_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                a.0, b.0,
            );
            let cd_lo =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(c.0, d.0);
            let cd_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                c.0, d.0,
            );

            // Stage 2: interleave ab and cd at 16-bit granularity
            let out0 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                ab_lo, cd_lo,
            );
            let out1 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                ab_lo, cd_lo,
            );
            let out2 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                ab_hi, cd_hi,
            );
            let out3 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                ab_hi, cd_hi,
            );

            unaligned::v128_store_u8(d0, out0);
            unaligned::v128_store_u8(d1, out1);
            unaligned::v128_store_u8(d2, out2);
            unaligned::v128_store_u8(d3, out3);
        }
    );
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecSimd128(v128, Simd128Descriptor);

impl U16SimdVec for U16VecSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, U16VecSimd128);
    const LEN: usize = 8;

    fn_simd128!(
        fn load(d: Simd128Descriptor, mem: &[u16]) -> U16VecSimd128 {
            U16VecSimd128(unaligned::v128_load_u16(mem.first_chunk().unwrap()), d)
        }
    );

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(u16x8_splat(v), d)
    }

    fn_simd128!(this: &U16VecSimd128, fn store(mem: &mut [u16]) {
        unaligned::v128_store_u16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_simd128!(
        fn store_interleaved_2(a: U16VecSimd128, b: U16VecSimd128, dest: &mut [u16]) {
            let [d0, d1, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };
            let lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0);
            let hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0);
            unaligned::v128_store_u16(d0, lo);
            unaligned::v128_store_u16(d1, hi);
        }
    );

    fn_simd128!(
        fn store_interleaved_3(
            a: U16VecSimd128,
            b: U16VecSimd128,
            c: U16VecSimd128,
            dest: &mut [u16],
        ) {
            let [d0, d1, d2, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };
            // 3-way u16 interleave using 2 shuffles per output vector.
            let ab0 = i16x8_shuffle::<0, 8, 0, 1, 9, 0, 2, 10>(a.0, b.0);
            let out0 = i16x8_shuffle::<0, 1, 8, 3, 4, 9, 6, 7>(ab0, c.0);

            let ab1 = i16x8_shuffle::<0, 3, 11, 0, 4, 12, 0, 5>(a.0, b.0);
            let out1 = i16x8_shuffle::<10, 1, 2, 11, 4, 5, 12, 7>(ab1, c.0);

            let ab2 = i16x8_shuffle::<13, 0, 6, 14, 0, 7, 15, 0>(a.0, b.0);
            let out2 = i16x8_shuffle::<0, 13, 2, 3, 14, 5, 6, 15>(ab2, c.0);

            unaligned::v128_store_u16(d0, out0);
            unaligned::v128_store_u16(d1, out1);
            unaligned::v128_store_u16(d2, out2);
        }
    );

    fn_simd128!(
        fn store_interleaved_4(
            a: U16VecSimd128,
            b: U16VecSimd128,
            c: U16VecSimd128,
            d: U16VecSimd128,
            dest: &mut [u16],
        ) {
            let [d0, d1, d2, d3, ..] = dest.as_chunks_mut::<8>().0 else {
                panic!("dest too short");
            };
            // Stage 1: interleave a,b and c,d at 16-bit level
            let ab_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0);
            let ab_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0);
            let cd_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(c.0, d.0);
            let cd_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(c.0, d.0);

            // Stage 2: interleave at 32-bit level
            let out0 = i32x4_shuffle::<0, 4, 1, 5>(ab_lo, cd_lo);
            let out1 = i32x4_shuffle::<2, 6, 3, 7>(ab_lo, cd_lo);
            let out2 = i32x4_shuffle::<0, 4, 1, 5>(ab_hi, cd_hi);
            let out3 = i32x4_shuffle::<2, 6, 3, 7>(ab_hi, cd_hi);

            unaligned::v128_store_u16(d0, out0);
            unaligned::v128_store_u16(d1, out1);
            unaligned::v128_store_u16(d2, out2);
            unaligned::v128_store_u16(d3, out3);
        }
    );

    #[inline(always)]
    fn bitcast_i16(self) -> I16VecSimd128 {
        I16VecSimd128(self.0, self.1)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskSimd128(v128, Simd128Descriptor);

impl SimdMask for MaskSimd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, MaskSimd128);

    #[inline(always)]
    fn if_then_else_f32(self, if_true: F32VecSimd128, if_false: F32VecSimd128) -> F32VecSimd128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            F32VecSimd128(
                i32x4_relaxed_laneselect(if_true.0, if_false.0, self.0),
                self.1,
            )
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            F32VecSimd128(v128_bitselect(if_true.0, if_false.0, self.0), self.1)
        }
    }

    #[inline(always)]
    fn if_then_else_i32(self, if_true: I32VecSimd128, if_false: I32VecSimd128) -> I32VecSimd128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            I32VecSimd128(
                i32x4_relaxed_laneselect(if_true.0, if_false.0, self.0),
                self.1,
            )
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            I32VecSimd128(v128_bitselect(if_true.0, if_false.0, self.0), self.1)
        }
    }

    #[inline(always)]
    fn maskz_i32(self, v: I32VecSimd128) -> I32VecSimd128 {
        // Zero out lanes where mask is true: v AND NOT mask
        I32VecSimd128(v128_andnot(v.0, self.0), self.1)
    }

    #[inline(always)]
    fn andnot(self, rhs: MaskSimd128) -> MaskSimd128 {
        // !self & rhs
        MaskSimd128(v128_andnot(rhs.0, self.0), self.1)
    }

    #[inline(always)]
    fn all(self) -> bool {
        u32x4_all_true(self.0)
    }
}

impl BitAnd<MaskSimd128> for MaskSimd128 {
    type Output = MaskSimd128;
    #[inline(always)]
    fn bitand(self, rhs: MaskSimd128) -> MaskSimd128 {
        MaskSimd128(v128_and(self.0, rhs.0), self.1)
    }
}

impl BitOr<MaskSimd128> for MaskSimd128 {
    type Output = MaskSimd128;
    #[inline(always)]
    fn bitor(self, rhs: MaskSimd128) -> MaskSimd128 {
        MaskSimd128(v128_or(self.0, rhs.0), self.1)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Simd128(v128, Simd128Descriptor);

impl SimdMask16 for Mask16Simd128 {
    type Descriptor = Simd128Descriptor;
    crate::impl_isa_conversions!(simd128, Mask16Simd128);

    #[inline(always)]
    fn if_then_else_i16(self, if_true: I16VecSimd128, if_false: I16VecSimd128) -> I16VecSimd128 {
        #[cfg(target_feature = "relaxed-simd")]
        {
            I16VecSimd128(
                i16x8_relaxed_laneselect(if_true.0, if_false.0, self.0),
                self.1,
            )
        }
        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            I16VecSimd128(v128_bitselect(if_true.0, if_false.0, self.0), self.1)
        }
    }

    #[inline(always)]
    fn maskz_i16(self, v: I16VecSimd128) -> I16VecSimd128 {
        // Zero out lanes where mask is true: v AND NOT mask
        I16VecSimd128(v128_andnot(v.0, self.0), self.1)
    }

    #[inline(always)]
    fn andnot(self, rhs: Mask16Simd128) -> Mask16Simd128 {
        // !self & rhs
        Mask16Simd128(v128_andnot(rhs.0, self.0), self.1)
    }

    #[inline(always)]
    fn all(self) -> bool {
        i16x8_all_true(self.0)
    }
}

impl BitAnd<Mask16Simd128> for Mask16Simd128 {
    type Output = Mask16Simd128;
    #[inline(always)]
    fn bitand(self, rhs: Mask16Simd128) -> Mask16Simd128 {
        Mask16Simd128(v128_and(self.0, rhs.0), self.1)
    }
}

impl BitOr<Mask16Simd128> for Mask16Simd128 {
    type Output = Mask16Simd128;
    #[inline(always)]
    fn bitor(self, rhs: Mask16Simd128) -> Mask16Simd128 {
        Mask16Simd128(v128_or(self.0, rhs.0), self.1)
    }
}

crate::impl_raw_accessors!(F32VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(I32VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(I16VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(U32VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(U64VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(U8VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(U16VecSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(MaskSimd128, v128, Simd128Descriptor);
crate::impl_raw_accessors!(Mask16Simd128, v128, Simd128Descriptor);
