// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
mod wasm32;

pub mod float16;
pub mod scalar;
pub mod unaligned;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub use aarch64::neon::{
    Bf16Table8Neon, F32VecNeon, I16VecNeon, I32VecNeon, Mask16Neon, MaskNeon, NeonDescriptor,
    U8VecNeon, U16VecNeon, U32VecNeon, U64VecNeon,
};
pub use float16::f16;
pub use scalar::ScalarDescriptor;
#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
pub use wasm32::simd128::{
    Bf16Table8Simd128, F32VecSimd128, I16VecSimd128, I32VecSimd128, Mask16Simd128, MaskSimd128,
    Simd128Descriptor, U8VecSimd128, U16VecSimd128, U32VecSimd128, U64VecSimd128,
};
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub use x86_64::avx::{
    AvxDescriptor, Bf16Table8Avx, F32VecAvx, I16VecAvx, I32VecAvx, Mask16Avx, MaskAvx, U8VecAvx,
    U16VecAvx, U32VecAvx, U64VecAvx,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use x86_64::avx512::{
    Avx512Descriptor, Bf16Table8Avx512, F32VecAvx512, I16VecAvx512, I32VecAvx512, Mask16Avx512,
    MaskAvx512, U8VecAvx512, U16VecAvx512, U32VecAvx512, U64VecAvx512,
};
#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
pub use x86_64::sse42::{
    Bf16Table8Sse42, F32VecSse42, I16VecSse42, I32VecSse42, Mask16Sse42, MaskSse42,
    Sse42Descriptor, U8VecSse42, U16VecSse42, U32VecSse42, U64VecSse42,
};

macro_rules! decl_isa_conversions {
    ($scalar_ty:ty, $sse42_ty:ident, $avx_ty:ident, $avx512_ty:ident, $neon_ty:ident, $simd128_ty:ident) => {
        #[inline(always)]
        fn as_scalar(self) -> Option<$scalar_ty> {
            None
        }

        #[inline(always)]
        fn from_scalar(_v: $scalar_ty) -> Option<Self> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
        #[inline(always)]
        fn as_sse42(self) -> Option<$sse42_ty> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
        #[inline(always)]
        fn from_sse42(_v: $sse42_ty) -> Option<Self> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        #[inline(always)]
        fn as_avx(self) -> Option<$avx_ty> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        #[inline(always)]
        fn from_avx(_v: $avx_ty) -> Option<Self> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[inline(always)]
        fn as_avx512(self) -> Option<$avx512_ty> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[inline(always)]
        fn from_avx512(_v: $avx512_ty) -> Option<Self> {
            None
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        #[inline(always)]
        fn as_neon(self) -> Option<$neon_ty> {
            None
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        #[inline(always)]
        fn from_neon(_v: $neon_ty) -> Option<Self> {
            None
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
        #[inline(always)]
        fn as_simd128(self) -> Option<$simd128_ty> {
            None
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
        #[inline(always)]
        fn from_simd128(_v: $simd128_ty) -> Option<Self> {
            None
        }
    };
}

macro_rules! decl_isa_array_conversions {
    ($elem:ty, $scalar_len:literal, $sse42_len:literal, $avx_len:literal, $avx512_len:literal, $neon_len:literal, $simd128_len:literal) => {
        #[inline(always)]
        fn array_slice_as_scalar(
            _slice: &[Self::UnderlyingArray],
        ) -> Option<&[[$elem; $scalar_len]]> {
            None
        }

        #[inline(always)]
        fn array_slice_mut_as_scalar(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $scalar_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
        #[inline(always)]
        fn array_slice_as_sse42(
            _slice: &[Self::UnderlyingArray],
        ) -> Option<&[[$elem; $sse42_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
        #[inline(always)]
        fn array_slice_mut_as_sse42(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $sse42_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        #[inline(always)]
        fn array_slice_as_avx(_slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $avx_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        #[inline(always)]
        fn array_slice_mut_as_avx(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $avx_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[inline(always)]
        fn array_slice_as_avx512(
            _slice: &[Self::UnderlyingArray],
        ) -> Option<&[[$elem; $avx512_len]]> {
            None
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[inline(always)]
        fn array_slice_mut_as_avx512(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $avx512_len]]> {
            None
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        #[inline(always)]
        fn array_slice_as_neon(_slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $neon_len]]> {
            None
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        #[inline(always)]
        fn array_slice_mut_as_neon(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $neon_len]]> {
            None
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
        #[inline(always)]
        fn array_slice_as_simd128(
            _slice: &[Self::UnderlyingArray],
        ) -> Option<&[[$elem; $simd128_len]]> {
            None
        }

        #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
        #[inline(always)]
        fn array_slice_mut_as_simd128(
            _slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $simd128_len]]> {
            None
        }
    };
}

#[allow(unused_macros)]
macro_rules! impl_raw_accessors {
    ($vec:ty, $raw:ty, $desc:ty) => {
        impl $vec {
            #[inline(always)]
            pub fn raw(self) -> $raw {
                self.0
            }

            #[inline(always)]
            pub fn descriptor(self) -> $desc {
                self.1
            }

            #[inline(always)]
            pub fn from_raw(d: $desc, raw: $raw) -> Self {
                Self(raw, d)
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use impl_raw_accessors;

macro_rules! impl_isa_conversions {
    (scalar, $ty:ty) => {
        #[inline(always)]
        fn as_scalar(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_scalar(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
    (sse42, $ty:ty) => {
        #[inline(always)]
        fn as_sse42(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_sse42(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
    (avx, $ty:ty) => {
        #[inline(always)]
        fn as_avx(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_avx(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
    (avx512, $ty:ty) => {
        #[inline(always)]
        fn as_avx512(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_avx512(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
    (neon, $ty:ty) => {
        #[inline(always)]
        fn as_neon(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_neon(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
    (simd128, $ty:ty) => {
        #[inline(always)]
        fn as_simd128(self) -> Option<$ty> {
            Some(self)
        }

        #[inline(always)]
        fn from_simd128(v: $ty) -> Option<Self> {
            Some(v)
        }
    };
}

pub(crate) use impl_isa_conversions;

macro_rules! impl_isa_array_conversions {
    (scalar, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_scalar(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_scalar(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
    (sse42, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_sse42(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_sse42(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
    (avx, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_avx(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_avx(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
    (avx512, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_avx512(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_avx512(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
    (neon, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_neon(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_neon(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
    (simd128, $elem:ty, $len:literal) => {
        #[inline(always)]
        fn array_slice_as_simd128(slice: &[Self::UnderlyingArray]) -> Option<&[[$elem; $len]]> {
            Some(slice)
        }

        #[inline(always)]
        fn array_slice_mut_as_simd128(
            slice: &mut [Self::UnderlyingArray],
        ) -> Option<&mut [[$elem; $len]]> {
            Some(slice)
        }
    };
}

pub(crate) use impl_isa_array_conversions;

pub trait SimdDescriptor: Sized + Copy + Debug + Send + Sync {
    type F32Vec: F32SimdVec<Descriptor = Self>;

    type I32Vec: I32SimdVec<Descriptor = Self>;

    type I16Vec: I16SimdVec<Descriptor = Self>;

    type U64Vec: U64SimdVec<Descriptor = Self>;

    type U32Vec: U32SimdVec<Descriptor = Self>;

    type U16Vec: U16SimdVec<Descriptor = Self>;

    type U8Vec: U8SimdVec<Descriptor = Self>;

    type Mask: SimdMask<Descriptor = Self>;

    type Mask16: SimdMask16<Descriptor = Self>;

    /// Prepared 8-entry BF16 lookup table for fast approximate lookups.
    /// Use `F32SimdVec::prepare_table_bf16_8` to create and
    /// `F32SimdVec::table_lookup_bf16_8` to use.
    type Bf16Table8: Copy;

    type Descriptor256: SimdDescriptor<Descriptor256 = Self::Descriptor256>;
    type Descriptor128: SimdDescriptor<Descriptor128 = Self::Descriptor128>;

    fn new() -> Option<Self>;

    /// Returns a vector descriptor suitable for operations on vectors of length 256 (Self if the
    /// current vector type is suitable). Note that it might still be beneficial to use `Self` for
    /// .call(), as the compiler could make use of features from more advanced instruction sets.
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256;

    /// Same as Self::maybe_downgrade_256bit, but for 128 bits.
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128;

    /// Calls the given closure within a target feature context.
    /// This enables establishing an unbroken chain of inline functions from the feature-annotated
    /// gateway up to the closure, allowing SIMD intrinsics to be used safely.
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R;

    #[inline(always)]
    fn as_scalar(self) -> Option<ScalarDescriptor> {
        None
    }

    #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
    #[inline(always)]
    fn as_sse42(self) -> Option<Sse42Descriptor> {
        None
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[inline(always)]
    fn as_avx(self) -> Option<AvxDescriptor> {
        None
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[inline(always)]
    fn as_avx512(self) -> Option<Avx512Descriptor> {
        None
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[inline(always)]
    fn as_neon(self) -> Option<NeonDescriptor> {
        None
    }

    #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
    #[inline(always)]
    fn as_simd128(self) -> Option<Simd128Descriptor> {
        None
    }
}

/// Subtrait for 128-bit SIMD descriptors (`Sse42Descriptor`, `NeonDescriptor`, `Simd128Descriptor`).
pub trait SimdDescriptor128:
    SimdDescriptor<
        F32Vec: F32SimdVec<Descriptor = Self, UnderlyingArray = [f32; 4]>,
        I16Vec: I16SimdVec<Descriptor = Self, UnderlyingArray = [i16; 8]>,
    >
{
}

/// Subtrait for 256-bit SIMD descriptors (`AvxDescriptor`).
pub trait SimdDescriptor256:
    SimdDescriptor<
        F32Vec: F32SimdVec<Descriptor = Self, UnderlyingArray = [f32; 8]>,
        I16Vec: I16SimdVec<Descriptor = Self, UnderlyingArray = [i16; 16]>,
    >
{
}

/// Subtrait for 512-bit SIMD descriptors (`Avx512Descriptor`).
pub trait SimdDescriptor512:
    SimdDescriptor<
        F32Vec: F32SimdVec<Descriptor = Self, UnderlyingArray = [f32; 16]>,
        I16Vec: I16SimdVec<Descriptor = Self, UnderlyingArray = [i16; 32]>,
    >
{
}

pub type F32Vec<D> = <D as SimdDescriptor>::F32Vec;
pub type I32Vec<D> = <D as SimdDescriptor>::I32Vec;
pub type I16Vec<D> = <D as SimdDescriptor>::I16Vec;
pub type U64Vec<D> = <D as SimdDescriptor>::U64Vec;
pub type U32Vec<D> = <D as SimdDescriptor>::U32Vec;
pub type U16Vec<D> = <D as SimdDescriptor>::U16Vec;
pub type U8Vec<D> = <D as SimdDescriptor>::U8Vec;
pub type Mask<D> = <D as SimdDescriptor>::Mask;
pub type Mask16<D> = <D as SimdDescriptor>::Mask16;
pub type F32VecArray<D> = <<D as SimdDescriptor>::F32Vec as F32SimdVec>::UnderlyingArray;
pub type I16VecArray<D> = <<D as SimdDescriptor>::I16Vec as I16SimdVec>::UnderlyingArray;

pub trait F32SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
    + DivAssign<Self>
{
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    /// An array of f32 of length Self::LEN.
    type UnderlyingArray: Copy
        + Default
        + Debug
        + AsRef<[f32]>
        + AsMut<[f32]>
        + std::ops::Index<usize, Output = f32>
        + std::ops::IndexMut<usize, Output = f32>;

    /// Converts v to an array of v.
    fn splat(d: Self::Descriptor, v: f32) -> Self;

    fn zero(d: Self::Descriptor) -> Self;

    fn mul_add(self, mul: Self, add: Self) -> Self;

    /// Computes `add - self * mul`, equivalent to `self * (-mul) + add`.
    /// Uses fused multiply-add with negation when available (FMA3 fnmadd).
    fn neg_mul_add(self, mul: Self, add: Self) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self;

    fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [f32]);

    fn store_array(&self, mem: &mut Self::UnderlyingArray);

    /// Stores two vectors interleaved: [a0, b0, a1, b1, a2, b2, ...].
    /// Requires `dest.len() >= 2 * Self::LEN` or it will panic.
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]);

    /// Stores three vectors interleaved: [a0, b0, c0, a1, b1, c1, ...].
    /// Requires `dest.len() >= 3 * Self::LEN` or it will panic.
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]);

    /// Stores four vectors interleaved: [a0, b0, c0, d0, a1, b1, c1, d1, ...].
    /// Requires `dest.len() >= 4 * Self::LEN` or it will panic.
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]);

    /// Stores eight vectors interleaved: [a0, b0, c0, d0, e0, f0, g0, h0, a1, ...].
    /// Requires `dest.len() >= 8 * Self::LEN` or it will panic.
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
    );

    /// Loads two vectors from interleaved data: [a0, b0, a1, b1, a2, b2, ...].
    /// Returns (a, b) where a = [a0, a1, a2, ...] and b = [b0, b1, b2, ...].
    /// Requires `src.len() >= 2 * Self::LEN` or it will panic.
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self);

    /// Loads three vectors from interleaved data: [a0, b0, c0, a1, b1, c1, ...].
    /// Returns (a, b, c) where a = [a0, a1, ...], b = [b0, b1, ...], c = [c0, c1, ...].
    /// Requires `src.len() >= 3 * Self::LEN` or it will panic.
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self);

    /// Loads four vectors from interleaved data: [a0, b0, c0, d0, a1, b1, c1, d1, ...].
    /// Returns (a, b, c, d) where each vector contains the deinterleaved components.
    /// Requires `src.len() >= 4 * Self::LEN` or it will panic.
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self);

    /// Rounds to nearest integer and stores as u8.
    /// Behavior is unspecified if values would overflow u8.
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn round_store_u8(self, dest: &mut [u8]);

    /// Rounds to nearest integer and stores as u16.
    /// Behavior is unspecified if values would overflow u16.
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn round_store_u16(self, dest: &mut [u16]);

    fn abs(self) -> Self;

    fn floor(self) -> Self;

    fn sqrt(self) -> Self;

    /// Negates all elements. Currently unused but kept for API completeness.
    #[allow(dead_code)]
    fn neg(self) -> Self;

    fn copysign(self, sign: Self) -> Self;

    fn max(self, other: Self) -> Self;

    fn min(self, other: Self) -> Self;

    fn gt(self, other: Self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::Mask;

    fn as_i32(self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::I32Vec;

    fn bitcast_to_i32(self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::I32Vec;

    /// Prepares an 8-entry f32 table for fast approximate lookups.
    /// Values are converted to BF16 format (loses lower 16 mantissa bits).
    ///
    /// Use this when you need to perform multiple lookups with the same table.
    /// The prepared table can be reused with [`table_lookup_bf16_8`][Self::table_lookup_bf16_8].
    fn prepare_table_bf16_8(
        d: Self::Descriptor,
        table: &[f32; 8],
    ) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::Bf16Table8;

    /// Performs fast approximate table lookup using a prepared BF16 table.
    ///
    /// This is the fastest lookup method when the same table is used multiple times.
    /// Use [`prepare_table_bf16_8`][Self::prepare_table_bf16_8] to create the prepared table.
    ///
    /// # Panics
    /// May panic or produce undefined results if indices contain values outside 0..8 range.
    fn table_lookup_bf16_8(
        d: Self::Descriptor,
        table: <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::Bf16Table8,
        indices: <<Self as F32SimdVec>::Descriptor as SimdDescriptor>::I32Vec,
    ) -> Self;

    /// Converts a slice of f32 into a slice of Self::UnderlyingArray. If slice.len() is not a
    /// multiple of `Self::LEN` this will panic.
    fn make_array_slice(slice: &[f32]) -> &[Self::UnderlyingArray];

    /// Converts a mut slice of f32 into a slice of Self::UnderlyingArray. If slice.len() is not a
    /// multiple of `Self::LEN` this will panic.
    fn make_array_slice_mut(slice: &mut [f32]) -> &mut [Self::UnderlyingArray];

    /// Loads f16 values (stored as u16 bit patterns) and converts them to f32.
    /// Uses hardware conversion instructions when available (F16C on x86, NEON fp16 on ARM).
    /// Requires `mem.len() >= Self::LEN` or it will panic.
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self;

    /// Converts f32 values to f16 and stores as u16 bit patterns.
    /// Uses hardware conversion instructions when available (F16C on x86, NEON fp16 on ARM).
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn store_f16_bits(self, dest: &mut [u16]);

    decl_isa_conversions!(
        f32,
        F32VecSse42,
        F32VecAvx,
        F32VecAvx512,
        F32VecNeon,
        F32VecSimd128
    );
    decl_isa_array_conversions!(f32, 1, 4, 8, 16, 4, 4);
}

pub trait I32SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;

    #[allow(dead_code)]
    const LEN: usize;

    /// Converts v to an array of v.
    fn splat(d: Self::Descriptor, v: i32) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load_from_i16(d: Self::Descriptor, mem: &[i16]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load_from_u16(d: Self::Descriptor, mem: &[u16]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [i32]);

    fn abs(self) -> Self;

    fn as_f32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::F32Vec;

    fn bitcast_to_f32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::F32Vec;

    fn bitcast_to_u32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::U32Vec;

    fn gt(self, other: Self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::Mask;

    fn lt_zero(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::Mask;

    fn eq(self, other: Self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::Mask;

    fn eq_zero(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::Mask;

    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn mul_wide_take_high(self, rhs: Self) -> Self;

    /// Stores the lower 16 bits of each i32 lane as u16 values.
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn store_u16(self, dest: &mut [u16]);

    /// Stores two vectors interleaved: [a0, b0, a1, b1, a2, b2, ...].
    /// Requires `dest.len() >= 2 * Self::LEN` or it will panic.
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i32]) {
        let a_f32 = a.bitcast_to_f32();
        let b_f32 = b.bitcast_to_f32();
        // SAFETY: i32 and f32 have the same size and alignment, and both are bag-of-bits types,
        // so the implicit transmute is safe.
        let dest_f32 =
            unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<f32>(), dest.len()) };
        <<Self as I32SimdVec>::Descriptor as SimdDescriptor>::F32Vec::store_interleaved_2(
            a_f32, b_f32, dest_f32,
        );
    }

    /// Stores the lower 8 bits of each i32 lane as u8 values.
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn store_u8(self, dest: &mut [u8]);

    decl_isa_conversions!(
        std::num::Wrapping<i32>,
        I32VecSse42,
        I32VecAvx,
        I32VecAvx512,
        I32VecNeon,
        I32VecSimd128
    );
}

pub trait I16SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Neg<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    /// An array of i16 of length Self::LEN.
    type UnderlyingArray: Copy
        + Default
        + Debug
        + AsRef<[i16]>
        + AsMut<[i16]>
        + std::ops::Index<usize, Output = i16>
        + std::ops::IndexMut<usize, Output = i16>;

    /// Converts v to an array of v.
    fn splat(d: Self::Descriptor, v: i16) -> Self;

    fn zero(d: Self::Descriptor) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[i16]) -> Self;

    fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [i16]);

    fn store_array(&self, mem: &mut Self::UnderlyingArray);

    /// Converts a slice of i16 into a slice of Self::UnderlyingArray. If slice.len() is not a
    /// multiple of `Self::LEN` this will panic.
    fn make_array_slice(slice: &[i16]) -> &[Self::UnderlyingArray];

    /// Converts a mut slice of i16 into a slice of Self::UnderlyingArray. If slice.len() is not a
    /// multiple of `Self::LEN` this will panic.
    fn make_array_slice_mut(slice: &mut [i16]) -> &mut [Self::UnderlyingArray];

    fn abs(self) -> Self;

    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn mul_wide_take_high(self, rhs: Self) -> Self;

    fn gt(self, other: Self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::Mask16;

    fn lt_zero(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::Mask16;

    fn eq(self, other: Self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::Mask16;

    fn eq_zero(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::Mask16;

    fn bitcast_u16(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::U16Vec;

    /// Stores two vectors interleaved: [a0, b0, a1, b1, a2, b2, ...].
    /// Requires `dest.len() >= 2 * Self::LEN` or it will panic.
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i16]) {
        assert!(dest.len() >= 2 * Self::LEN);
        // SAFETY: i16 and u16 have the same size and alignment, and both are bag-of-bits types,
        // so the implicit transmute is safe.
        let dest_u16 =
            unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<u16>(), dest.len()) };
        <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::U16Vec::store_interleaved_2(
            a.bitcast_u16(),
            b.bitcast_u16(),
            dest_u16,
        );
    }

    /// Stores three vectors interleaved: [a0, b0, c0, a1, b1, c1, ...].
    /// Requires `dest.len() >= 3 * Self::LEN` or it will panic.
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [i16]) {
        assert!(dest.len() >= 3 * Self::LEN);
        // SAFETY: i16 and u16 have the same size and alignment, and both are bag-of-bits types,
        // so the implicit transmute is safe.
        let dest_u16 =
            unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<u16>(), dest.len()) };
        <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::U16Vec::store_interleaved_3(
            a.bitcast_u16(),
            b.bitcast_u16(),
            c.bitcast_u16(),
            dest_u16,
        );
    }

    /// Stores four vectors interleaved: [a0, b0, c0, d0, a1, b1, c1, d1, ...].
    /// Requires `dest.len() >= 4 * Self::LEN` or it will panic.
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [i16]) {
        assert!(dest.len() >= 4 * Self::LEN);
        // SAFETY: i16 and u16 have the same size and alignment, and both are bag-of-bits types,
        // so the implicit transmute is safe.
        let dest_u16 =
            unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<u16>(), dest.len()) };
        <<Self as I16SimdVec>::Descriptor as SimdDescriptor>::U16Vec::store_interleaved_4(
            a.bitcast_u16(),
            b.bitcast_u16(),
            c.bitcast_u16(),
            d.bitcast_u16(),
            dest_u16,
        );
    }

    /// Stores the lower 8 bits of each lane into `dest`.
    /// Requires `dest.len() >= Self::LEN` or it will panic.
    fn store_u8(self, dest: &mut [u8]);

    decl_isa_conversions!(
        std::num::Wrapping<i16>,
        I16VecSse42,
        I16VecAvx,
        I16VecAvx512,
        I16VecNeon,
        I16VecSimd128
    );
    decl_isa_array_conversions!(i16, 1, 8, 16, 32, 8, 8);
}

pub trait U64SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;

    #[allow(dead_code)]
    const LEN: usize;

    fn splat(d: Self::Descriptor, v: u64) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[u64]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [u64]);

    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    fn bitcast_to_u32(self) -> <<Self as U64SimdVec>::Descriptor as SimdDescriptor>::U32Vec;

    decl_isa_conversions!(
        std::num::Wrapping<u64>,
        U64VecSse42,
        U64VecAvx,
        U64VecAvx512,
        U64VecNeon,
        U64VecSimd128
    );
}

pub trait U32SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;

    #[allow(dead_code)]
    const LEN: usize;

    fn bitcast_to_i32(self) -> <<Self as U32SimdVec>::Descriptor as SimdDescriptor>::I32Vec;

    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self;

    decl_isa_conversions!(
        std::num::Wrapping<u32>,
        U32VecSse42,
        U32VecAvx,
        U32VecAvx512,
        U32VecNeon,
        U32VecSimd128
    );
}

pub trait U8SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    fn load(d: Self::Descriptor, mem: &[u8]) -> Self;
    fn splat(d: Self::Descriptor, v: u8) -> Self;
    fn store(&self, mem: &mut [u8]);

    /// Stores two vectors interleaved: [a0, b0, a1, b1, a2, b2, ...].
    /// Requires `dest.len() >= 2 * Self::LEN` or it will panic.
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]);

    /// Stores three vectors interleaved: [a0, b0, c0, a1, b1, c1, ...].
    /// Requires `dest.len() >= 3 * Self::LEN` or it will panic.
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]);

    /// Stores four vectors interleaved: [a0, b0, c0, d0, a1, b1, c1, d1, ...].
    /// Requires `dest.len() >= 4 * Self::LEN` or it will panic.
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]);

    decl_isa_conversions!(
        u8,
        U8VecSse42,
        U8VecAvx,
        U8VecAvx512,
        U8VecNeon,
        U8VecSimd128
    );
}

pub trait U16SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    fn load(d: Self::Descriptor, mem: &[u16]) -> Self;
    fn splat(d: Self::Descriptor, v: u16) -> Self;
    fn store(&self, mem: &mut [u16]);

    /// Stores two vectors interleaved: [a0, b0, a1, b1, a2, b2, ...].
    /// Requires `dest.len() >= 2 * Self::LEN` or it will panic.
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]);

    /// Stores three vectors interleaved: [a0, b0, c0, a1, b1, c1, ...].
    /// Requires `dest.len() >= 3 * Self::LEN` or it will panic.
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]);

    /// Stores four vectors interleaved: [a0, b0, c0, d0, a1, b1, c1, d1, ...].
    /// Requires `dest.len() >= 4 * Self::LEN` or it will panic.
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]);

    fn bitcast_i16(self) -> <<Self as U16SimdVec>::Descriptor as SimdDescriptor>::I16Vec;

    decl_isa_conversions!(
        u16,
        U16VecSse42,
        U16VecAvx,
        U16VecAvx512,
        U16VecNeon,
        U16VecSimd128
    );
}

#[macro_export]
macro_rules! shl {
    ($val: expr, $amount: literal) => {
        $val.shl::<{ $amount as u32 }, { $amount as i32 }>()
    };
}

#[macro_export]
macro_rules! shr {
    ($val: expr, $amount: literal) => {
        $val.shr::<{ $amount as u32 }, { $amount as i32 }>()
    };
}

pub trait SimdMask:
    Sized + Copy + Debug + Send + Sync + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self>
{
    type Descriptor: SimdDescriptor;

    fn if_then_else_f32(
        self,
        if_true: <<Self as SimdMask>::Descriptor as SimdDescriptor>::F32Vec,
        if_false: <<Self as SimdMask>::Descriptor as SimdDescriptor>::F32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptor>::F32Vec;

    fn if_then_else_i32(
        self,
        if_true: <<Self as SimdMask>::Descriptor as SimdDescriptor>::I32Vec,
        if_false: <<Self as SimdMask>::Descriptor as SimdDescriptor>::I32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptor>::I32Vec;

    fn maskz_i32(
        self,
        v: <<Self as SimdMask>::Descriptor as SimdDescriptor>::I32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptor>::I32Vec;

    fn all(self) -> bool;

    // !self & rhs
    fn andnot(self, rhs: Self) -> Self;

    decl_isa_conversions!(bool, MaskSse42, MaskAvx, MaskAvx512, MaskNeon, MaskSimd128);
}

pub trait SimdMask16:
    Sized + Copy + Debug + Send + Sync + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self>
{
    type Descriptor: SimdDescriptor;

    fn if_then_else_i16(
        self,
        if_true: <<Self as SimdMask16>::Descriptor as SimdDescriptor>::I16Vec,
        if_false: <<Self as SimdMask16>::Descriptor as SimdDescriptor>::I16Vec,
    ) -> <<Self as SimdMask16>::Descriptor as SimdDescriptor>::I16Vec;

    fn maskz_i16(
        self,
        v: <<Self as SimdMask16>::Descriptor as SimdDescriptor>::I16Vec,
    ) -> <<Self as SimdMask16>::Descriptor as SimdDescriptor>::I16Vec;

    fn all(self) -> bool;

    // !self & rhs
    fn andnot(self, rhs: Self) -> Self;

    decl_isa_conversions!(
        bool,
        Mask16Sse42,
        Mask16Avx,
        Mask16Avx512,
        Mask16Neon,
        Mask16Simd128
    );
}

macro_rules! impl_f32_array_interface {
    () => {
        type UnderlyingArray = [f32; Self::LEN];

        #[inline(always)]
        fn make_array_slice(slice: &[f32]) -> &[Self::UnderlyingArray] {
            let (ret, rem) = slice.as_chunks();
            assert!(rem.is_empty());
            ret
        }

        #[inline(always)]
        fn make_array_slice_mut(slice: &mut [f32]) -> &mut [Self::UnderlyingArray] {
            let (ret, rem) = slice.as_chunks_mut();
            assert!(rem.is_empty());
            ret
        }

        #[inline(always)]
        fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self {
            Self::load(d, mem)
        }

        #[inline(always)]
        fn store_array(&self, mem: &mut Self::UnderlyingArray) {
            self.store(mem);
        }
    };
}

pub(crate) use impl_f32_array_interface;

macro_rules! impl_i16_array_interface {
    () => {
        type UnderlyingArray = [i16; Self::LEN];

        #[inline(always)]
        fn make_array_slice(slice: &[i16]) -> &[Self::UnderlyingArray] {
            let (ret, rem) = slice.as_chunks();
            assert!(rem.is_empty());
            ret
        }

        #[inline(always)]
        fn make_array_slice_mut(slice: &mut [i16]) -> &mut [Self::UnderlyingArray] {
            let (ret, rem) = slice.as_chunks_mut();
            assert!(rem.is_empty());
            ret
        }

        #[inline(always)]
        fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self {
            Self::load(d, mem)
        }

        #[inline(always)]
        fn store_array(&self, mem: &mut Self::UnderlyingArray) {
            self.store(mem);
        }
    };
}

pub(crate) use impl_i16_array_interface;

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_isa_type {
    (scalar, Desc) => {
        $crate::ScalarDescriptor
    };
    (scalar, F32Vec) => {
        f32
    };
    (scalar, I32Vec) => {
        ::std::num::Wrapping<i32>
    };
    (scalar, I16Vec) => {
        ::std::num::Wrapping<i16>
    };
    (scalar, U64Vec) => {
        ::std::num::Wrapping<u64>
    };
    (scalar, U32Vec) => {
        ::std::num::Wrapping<u32>
    };
    (scalar, U16Vec) => {
        u16
    };
    (scalar, U8Vec) => {
        u8
    };
    (scalar, Mask) => {
        bool
    };
    (scalar, Mask16) => {
        bool
    };
    (scalar, F32ArrRef) => {
        &[[f32; 1]]
    };
    (scalar, F32ArrMut) => {
        &mut [[f32; 1]]
    };
    (scalar, I16ArrRef) => {
        &[[i16; 1]]
    };
    (scalar, I16ArrMut) => {
        &mut [[i16; 1]]
    };

    (sse42, Desc) => {
        $crate::Sse42Descriptor
    };
    (sse42, F32Vec) => {
        $crate::F32VecSse42
    };
    (sse42, I32Vec) => {
        $crate::I32VecSse42
    };
    (sse42, I16Vec) => {
        $crate::I16VecSse42
    };
    (sse42, U64Vec) => {
        $crate::U64VecSse42
    };
    (sse42, U32Vec) => {
        $crate::U32VecSse42
    };
    (sse42, U16Vec) => {
        $crate::U16VecSse42
    };
    (sse42, U8Vec) => {
        $crate::U8VecSse42
    };
    (sse42, Mask) => {
        $crate::MaskSse42
    };
    (sse42, Mask16) => {
        $crate::Mask16Sse42
    };
    (sse42, F32ArrRef) => {
        &[[f32; 4]]
    };
    (sse42, F32ArrMut) => {
        &mut [[f32; 4]]
    };
    (sse42, I16ArrRef) => {
        &[[i16; 8]]
    };
    (sse42, I16ArrMut) => {
        &mut [[i16; 8]]
    };

    (avx, Desc) => {
        $crate::AvxDescriptor
    };
    (avx, F32Vec) => {
        $crate::F32VecAvx
    };
    (avx, I32Vec) => {
        $crate::I32VecAvx
    };
    (avx, I16Vec) => {
        $crate::I16VecAvx
    };
    (avx, U64Vec) => {
        $crate::U64VecAvx
    };
    (avx, U32Vec) => {
        $crate::U32VecAvx
    };
    (avx, U16Vec) => {
        $crate::U16VecAvx
    };
    (avx, U8Vec) => {
        $crate::U8VecAvx
    };
    (avx, Mask) => {
        $crate::MaskAvx
    };
    (avx, Mask16) => {
        $crate::Mask16Avx
    };
    (avx, F32ArrRef) => {
        &[[f32; 8]]
    };
    (avx, F32ArrMut) => {
        &mut [[f32; 8]]
    };
    (avx, I16ArrRef) => {
        &[[i16; 16]]
    };
    (avx, I16ArrMut) => {
        &mut [[i16; 16]]
    };

    (avx512, Desc) => {
        $crate::Avx512Descriptor
    };
    (avx512, F32Vec) => {
        $crate::F32VecAvx512
    };
    (avx512, I32Vec) => {
        $crate::I32VecAvx512
    };
    (avx512, I16Vec) => {
        $crate::I16VecAvx512
    };
    (avx512, U64Vec) => {
        $crate::U64VecAvx512
    };
    (avx512, U32Vec) => {
        $crate::U32VecAvx512
    };
    (avx512, U16Vec) => {
        $crate::U16VecAvx512
    };
    (avx512, U8Vec) => {
        $crate::U8VecAvx512
    };
    (avx512, Mask) => {
        $crate::MaskAvx512
    };
    (avx512, Mask16) => {
        $crate::Mask16Avx512
    };
    (avx512, F32ArrRef) => {
        &[[f32; 16]]
    };
    (avx512, F32ArrMut) => {
        &mut [[f32; 16]]
    };
    (avx512, I16ArrRef) => {
        &[[i16; 32]]
    };
    (avx512, I16ArrMut) => {
        &mut [[i16; 32]]
    };

    (neon, Desc) => {
        $crate::NeonDescriptor
    };
    (neon, F32Vec) => {
        $crate::F32VecNeon
    };
    (neon, I32Vec) => {
        $crate::I32VecNeon
    };
    (neon, I16Vec) => {
        $crate::I16VecNeon
    };
    (neon, U64Vec) => {
        $crate::U64VecNeon
    };
    (neon, U32Vec) => {
        $crate::U32VecNeon
    };
    (neon, U16Vec) => {
        $crate::U16VecNeon
    };
    (neon, U8Vec) => {
        $crate::U8VecNeon
    };
    (neon, Mask) => {
        $crate::MaskNeon
    };
    (neon, Mask16) => {
        $crate::Mask16Neon
    };
    (neon, F32ArrRef) => {
        &[[f32; 4]]
    };
    (neon, F32ArrMut) => {
        &mut [[f32; 4]]
    };
    (neon, I16ArrRef) => {
        &[[i16; 8]]
    };
    (neon, I16ArrMut) => {
        &mut [[i16; 8]]
    };

    (simd128, Desc) => {
        $crate::Simd128Descriptor
    };
    (simd128, F32Vec) => {
        $crate::F32VecSimd128
    };
    (simd128, I32Vec) => {
        $crate::I32VecSimd128
    };
    (simd128, I16Vec) => {
        $crate::I16VecSimd128
    };
    (simd128, U64Vec) => {
        $crate::U64VecSimd128
    };
    (simd128, U32Vec) => {
        $crate::U32VecSimd128
    };
    (simd128, U16Vec) => {
        $crate::U16VecSimd128
    };
    (simd128, U8Vec) => {
        $crate::U8VecSimd128
    };
    (simd128, Mask) => {
        $crate::MaskSimd128
    };
    (simd128, Mask16) => {
        $crate::Mask16Simd128
    };
    (simd128, F32ArrRef) => {
        &[[f32; 4]]
    };
    (simd128, F32ArrMut) => {
        &mut [[f32; 4]]
    };
    (simd128, I16ArrRef) => {
        &[[i16; 8]]
    };
    (simd128, I16ArrMut) => {
        &mut [[i16; 8]]
    };

    (v128, F32ArrRef) => {
        &[[f32; 4]]
    };
    (v128, F32ArrMut) => {
        &mut [[f32; 4]]
    };
    (v128, I16ArrRef) => {
        &[[i16; 8]]
    };
    (v128, I16ArrMut) => {
        &mut [[i16; 8]]
    };

    (v256, F32ArrRef) => {
        &[[f32; 8]]
    };
    (v256, F32ArrMut) => {
        &mut [[f32; 8]]
    };
    (v256, I16ArrRef) => {
        &[[i16; 16]]
    };
    (v256, I16ArrMut) => {
        &mut [[i16; 16]]
    };

    (v512, F32ArrRef) => {
        &[[f32; 16]]
    };
    (v512, F32ArrMut) => {
        &mut [[f32; 16]]
    };
    (v512, I16ArrRef) => {
        &[[i16; 32]]
    };
    (v512, I16ArrMut) => {
        &mut [[i16; 32]]
    };

    ((@width $w:ident $D:ident), F32ArrRef) => {
        $crate::__simd_specialize_isa_type!($w, F32ArrRef)
    };
    ((@width $w:ident $D:ident), F32ArrMut) => {
        $crate::__simd_specialize_isa_type!($w, F32ArrMut)
    };
    ((@width $w:ident $D:ident), I16ArrRef) => {
        $crate::__simd_specialize_isa_type!($w, I16ArrRef)
    };
    ((@width $w:ident $D:ident), I16ArrMut) => {
        $crate::__simd_specialize_isa_type!($w, I16ArrMut)
    };
    ((@width $w:ident $D:ident), $assoc:ident) => {
        <$D as $crate::SimdDescriptor>::$assoc
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arg_ty {
    ($isa:tt, $D2:ident :: F32Vec) => {
        $crate::__simd_specialize_isa_type!($isa, F32Vec)
    };
    ($isa:tt, F32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, F32Vec)
    };
    ($isa:tt, $p:ident :: F32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, F32Vec)
    };
    ($isa:tt, $D2:ident :: I32Vec) => {
        $crate::__simd_specialize_isa_type!($isa, I32Vec)
    };
    ($isa:tt, I32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, I32Vec)
    };
    ($isa:tt, $p:ident :: I32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, I32Vec)
    };
    ($isa:tt, $D2:ident :: I16Vec) => {
        $crate::__simd_specialize_isa_type!($isa, I16Vec)
    };
    ($isa:tt, I16Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, I16Vec)
    };
    ($isa:tt, $p:ident :: I16Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, I16Vec)
    };
    ($isa:tt, $D2:ident :: U64Vec) => {
        $crate::__simd_specialize_isa_type!($isa, U64Vec)
    };
    ($isa:tt, U64Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U64Vec)
    };
    ($isa:tt, $p:ident :: U64Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U64Vec)
    };
    ($isa:tt, $D2:ident :: U32Vec) => {
        $crate::__simd_specialize_isa_type!($isa, U32Vec)
    };
    ($isa:tt, U32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U32Vec)
    };
    ($isa:tt, $p:ident :: U32Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U32Vec)
    };
    ($isa:tt, $D2:ident :: U16Vec) => {
        $crate::__simd_specialize_isa_type!($isa, U16Vec)
    };
    ($isa:tt, U16Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U16Vec)
    };
    ($isa:tt, $p:ident :: U16Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U16Vec)
    };
    ($isa:tt, $D2:ident :: U8Vec) => {
        $crate::__simd_specialize_isa_type!($isa, U8Vec)
    };
    ($isa:tt, U8Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U8Vec)
    };
    ($isa:tt, $p:ident :: U8Vec < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, U8Vec)
    };
    ($isa:tt, $D2:ident :: Mask) => {
        $crate::__simd_specialize_isa_type!($isa, Mask)
    };
    ($isa:tt, Mask < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, Mask)
    };
    ($isa:tt, $p:ident :: Mask < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, Mask)
    };
    ($isa:tt, $D2:ident :: Mask16) => {
        $crate::__simd_specialize_isa_type!($isa, Mask16)
    };
    ($isa:tt, Mask16 < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, Mask16)
    };
    ($isa:tt, $p:ident :: Mask16 < $D2:ty >) => {
        $crate::__simd_specialize_isa_type!($isa, Mask16)
    };
    ($isa:tt, & [ F32VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrRef)
    };
    ($isa:tt, & [ $p:ident :: F32VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrRef)
    };
    ($isa:tt, & mut [ F32VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrMut)
    };
    ($isa:tt, & mut [ $p:ident :: F32VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrMut)
    };
    ($isa:tt, & [ I16VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrRef)
    };
    ($isa:tt, & [ $p:ident :: I16VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrRef)
    };
    ($isa:tt, & mut [ I16VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrMut)
    };
    ($isa:tt, & mut [ $p:ident :: I16VecArray < $D2:ty > ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrMut)
    };
    ($isa:tt, & [ < $D2:ident :: F32Vec as F32SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrRef)
    };
    ($isa:tt, & [ < $D2:ident :: F32Vec as $p:ident :: F32SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrRef)
    };
    ($isa:tt, & mut [ < $D2:ident :: F32Vec as F32SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrMut)
    };
    ($isa:tt, & mut [ < $D2:ident :: F32Vec as $p:ident :: F32SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, F32ArrMut)
    };
    ($isa:tt, & [ < $D2:ident :: I16Vec as I16SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrRef)
    };
    ($isa:tt, & [ < $D2:ident :: I16Vec as $p:ident :: I16SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrRef)
    };
    ($isa:tt, & mut [ < $D2:ident :: I16Vec as I16SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrMut)
    };
    ($isa:tt, & mut [ < $D2:ident :: I16Vec as $p:ident :: I16SimdVec > :: UnderlyingArray ]) => {
        $crate::__simd_specialize_isa_type!($isa, I16ArrMut)
    };
    ($isa:tt, $($other:tt)+) => {
        $($other)+
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arg_conv {
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: F32Vec) => {
        <$D::F32Vec as $crate::F32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, F32Vec < $D2:ty >) => {
        <$D::F32Vec as $crate::F32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: F32Vec < $D2:ty >) => {
        <$D::F32Vec as $crate::F32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: I32Vec) => {
        <$D::I32Vec as $crate::I32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, I32Vec < $D2:ty >) => {
        <$D::I32Vec as $crate::I32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: I32Vec < $D2:ty >) => {
        <$D::I32Vec as $crate::I32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: I16Vec) => {
        <$D::I16Vec as $crate::I16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, I16Vec < $D2:ty >) => {
        <$D::I16Vec as $crate::I16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: I16Vec < $D2:ty >) => {
        <$D::I16Vec as $crate::I16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: U64Vec) => {
        <$D::U64Vec as $crate::U64SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, U64Vec < $D2:ty >) => {
        <$D::U64Vec as $crate::U64SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: U64Vec < $D2:ty >) => {
        <$D::U64Vec as $crate::U64SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: U32Vec) => {
        <$D::U32Vec as $crate::U32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, U32Vec < $D2:ty >) => {
        <$D::U32Vec as $crate::U32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: U32Vec < $D2:ty >) => {
        <$D::U32Vec as $crate::U32SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: U16Vec) => {
        <$D::U16Vec as $crate::U16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, U16Vec < $D2:ty >) => {
        <$D::U16Vec as $crate::U16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: U16Vec < $D2:ty >) => {
        <$D::U16Vec as $crate::U16SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: U8Vec) => {
        <$D::U8Vec as $crate::U8SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, U8Vec < $D2:ty >) => {
        <$D::U8Vec as $crate::U8SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: U8Vec < $D2:ty >) => {
        <$D::U8Vec as $crate::U8SimdVec>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: Mask) => {
        <$D::Mask as $crate::SimdMask>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, Mask < $D2:ty >) => {
        <$D::Mask as $crate::SimdMask>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: Mask < $D2:ty >) => {
        <$D::Mask as $crate::SimdMask>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $D2:ident :: Mask16) => {
        <$D::Mask16 as $crate::SimdMask16>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, Mask16 < $D2:ty >) => {
        <$D::Mask16 as $crate::SimdMask16>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $p:ident :: Mask16 < $D2:ty >) => {
        <$D::Mask16 as $crate::SimdMask16>::$as_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ F32VecArray < $D2:ty > ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ $p:ident :: F32VecArray < $D2:ty > ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ F32VecArray < $D2:ty > ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ $p:ident :: F32VecArray < $D2:ty > ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ I16VecArray < $D2:ty > ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ $p:ident :: I16VecArray < $D2:ty > ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ I16VecArray < $D2:ty > ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ $p:ident :: I16VecArray < $D2:ty > ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ < $D2:ident :: F32Vec as F32SimdVec > :: UnderlyingArray ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ < $D2:ident :: F32Vec as $p:ident :: F32SimdVec > :: UnderlyingArray ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ < $D2:ident :: F32Vec as F32SimdVec > :: UnderlyingArray ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ < $D2:ident :: F32Vec as $p:ident :: F32SimdVec > :: UnderlyingArray ]) => {
        <$D::F32Vec as $crate::F32SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ < $D2:ident :: I16Vec as I16SimdVec > :: UnderlyingArray ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & [ < $D2:ident :: I16Vec as $p:ident :: I16SimdVec > :: UnderlyingArray ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ < $D2:ident :: I16Vec as I16SimdVec > :: UnderlyingArray ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, & mut [ < $D2:ident :: I16Vec as $p:ident :: I16SimdVec > :: UnderlyingArray ]) => {
        <$D::I16Vec as $crate::I16SimdVec>::$arr_mut_fn($arg).unwrap()
    };
    ($as_fn:ident, $arr_fn:ident, $arr_mut_fn:ident, $D:ident, $arg:ident, $($other:tt)+) => {
        $arg
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_ret_ty {
    ($isa:tt, ()) => {
        ()
    };
    ($isa:tt, $($other:tt)+) => {
        $crate::__simd_specialize_arg_ty!($isa, $($other)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_ret_conv {
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: F32Vec) => {
        <$D::F32Vec as $crate::F32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, F32Vec < $D2:ty >) => {
        <$D::F32Vec as $crate::F32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: F32Vec < $D2:ty >) => {
        <$D::F32Vec as $crate::F32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: I32Vec) => {
        <$D::I32Vec as $crate::I32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, I32Vec < $D2:ty >) => {
        <$D::I32Vec as $crate::I32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: I32Vec < $D2:ty >) => {
        <$D::I32Vec as $crate::I32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: I16Vec) => {
        <$D::I16Vec as $crate::I16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, I16Vec < $D2:ty >) => {
        <$D::I16Vec as $crate::I16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: I16Vec < $D2:ty >) => {
        <$D::I16Vec as $crate::I16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: U64Vec) => {
        <$D::U64Vec as $crate::U64SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, U64Vec < $D2:ty >) => {
        <$D::U64Vec as $crate::U64SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: U64Vec < $D2:ty >) => {
        <$D::U64Vec as $crate::U64SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: U32Vec) => {
        <$D::U32Vec as $crate::U32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, U32Vec < $D2:ty >) => {
        <$D::U32Vec as $crate::U32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: U32Vec < $D2:ty >) => {
        <$D::U32Vec as $crate::U32SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: U16Vec) => {
        <$D::U16Vec as $crate::U16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, U16Vec < $D2:ty >) => {
        <$D::U16Vec as $crate::U16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: U16Vec < $D2:ty >) => {
        <$D::U16Vec as $crate::U16SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: U8Vec) => {
        <$D::U8Vec as $crate::U8SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, U8Vec < $D2:ty >) => {
        <$D::U8Vec as $crate::U8SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: U8Vec < $D2:ty >) => {
        <$D::U8Vec as $crate::U8SimdVec>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: Mask) => {
        <$D::Mask as $crate::SimdMask>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, Mask < $D2:ty >) => {
        <$D::Mask as $crate::SimdMask>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: Mask < $D2:ty >) => {
        <$D::Mask as $crate::SimdMask>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $D2:ident :: Mask16) => {
        <$D::Mask16 as $crate::SimdMask16>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, Mask16 < $D2:ty >) => {
        <$D::Mask16 as $crate::SimdMask16>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $p:ident :: Mask16 < $D2:ty >) => {
        <$D::Mask16 as $crate::SimdMask16>::$from_fn($val).unwrap()
    };
    ($from_fn:ident, $D:ident, $val:expr, $($other:tt)+) => {
        $val
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_scalar {
    ($d:ident : $D:ident, [$(($arg:ident : [$($arg_ty:tt)+]))*], [$($ret:tt)+], $body:block, $else:block) => {
        if let ::core::option::Option::Some(__simd_d) = <$D as $crate::SimdDescriptor>::as_scalar($d) {
            #[inline]
            fn __simd_arm(
                $d: $crate::ScalarDescriptor,
                $(
                    $arg: $crate::__simd_specialize_arg_ty!(scalar, $($arg_ty)+),
                )*
            ) -> $crate::__simd_specialize_ret_ty!(scalar, $($ret)+) {
                let _ = (&$d, $(&$arg,)*);
                $body
            }
            let __simd_ret = __simd_arm(
                __simd_d,
                $(
                    $crate::__simd_specialize_arg_conv!(as_scalar, array_slice_as_scalar, array_slice_mut_as_scalar, $D, $arg, $($arg_ty)+),
                )*
            );
            $crate::__simd_specialize_ret_conv!(from_scalar, $D, __simd_ret, $($ret)+)
        } else $else
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_common {
    (
        $isa:ident,
        $desc:ident,
        $feat:literal,
        $as_desc:ident,
        $as_vec:ident,
        $from_vec:ident,
        $arr:ident,
        $arr_mut:ident;
        $d:ident : $D:ident,
        [$(($arg:ident : [$($arg_ty:tt)+]))*],
        [$($ret:tt)+],
        $body:block,
        $else:block
    ) => {
        if let ::core::option::Option::Some(__simd_d) = <$D as $crate::SimdDescriptor>::$as_desc($d) {
            #[target_feature(enable = $feat)]
            #[inline]
            fn __simd_arm(
                $d: $crate::$desc,
                $(
                    $arg: $crate::__simd_specialize_arg_ty!($isa, $($arg_ty)+),
                )*
            ) -> $crate::__simd_specialize_ret_ty!($isa, $($ret)+) {
                let _ = (&$d, $(&$arg,)*);
                $body
            }
            // SAFETY: The `as_*` descriptor conversion witnesses that the target feature is enabled.
            #[allow(unsafe_code, unused_unsafe, clippy::undocumented_unsafe_blocks)]
            let __simd_ret = unsafe {
                __simd_arm(
                    __simd_d,
                    $(
                        $crate::__simd_specialize_arg_conv!($as_vec, $arr, $arr_mut, $D, $arg, $($arg_ty)+),
                    )*
                )
            };
            $crate::__simd_specialize_ret_conv!($from_vec, $D, __simd_ret, $($ret)+)
        } else $else
    };
}

#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_sse42 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_common!(
            sse42,
            Sse42Descriptor,
            "sse4.2",
            as_sse42,
            as_sse42,
            from_sse42,
            array_slice_as_sse42,
            array_slice_mut_as_sse42;
            $($tt)*
        )
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "sse42")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_sse42 {
    ($d:ident : $D:ident, $params:tt, $ret:tt, $body:block, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_avx {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_common!(
            avx,
            AvxDescriptor,
            "avx2,fma,f16c",
            as_avx,
            as_avx,
            from_avx,
            array_slice_as_avx,
            array_slice_mut_as_avx;
            $($tt)*
        )
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_avx {
    ($d:ident : $D:ident, $params:tt, $ret:tt, $body:block, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_avx512 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_common!(
            avx512,
            Avx512Descriptor,
            "avx512f,avx512bw",
            as_avx512,
            as_avx512,
            from_avx512,
            array_slice_as_avx512,
            array_slice_mut_as_avx512;
            $($tt)*
        )
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_avx512 {
    ($d:ident : $D:ident, $params:tt, $ret:tt, $body:block, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_neon {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_common!(
            neon,
            NeonDescriptor,
            "neon",
            as_neon,
            as_neon,
            from_neon,
            array_slice_as_neon,
            array_slice_mut_as_neon;
            $($tt)*
        )
    };
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_neon {
    ($d:ident : $D:ident, $params:tt, $ret:tt, $body:block, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_simd128 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_common!(
            simd128,
            Simd128Descriptor,
            "simd128",
            as_simd128,
            as_simd128,
            from_simd128,
            array_slice_as_simd128,
            array_slice_mut_as_simd128;
            $($tt)*
        )
    };
}

#[cfg(not(all(target_arch = "wasm32", feature = "simd128")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_simd128 {
    ($d:ident : $D:ident, $params:tt, $ret:tt, $body:block, $else:block) => {
        $else
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_width_dispatch_backend {
    (
        $isa_arm:ident,
        $desc:ident,
        $feat:literal,
        $as_desc:ident,
        $as_vec:ident,
        $from_vec:ident,
        $arr:ident,
        $arr_mut:ident;
        $arm_fn:ident,
        $d:ident : $D:ident,
        [$(($arg:ident : [$($arg_ty:tt)+]))*],
        [$($ret:tt)+],
        $else:block
    ) => {
        if let ::core::option::Option::Some(__simd_d) = <$D as $crate::SimdDescriptor>::$as_desc($d) {
            #[target_feature(enable = $feat)]
            #[inline]
            fn __simd_trampoline(
                $d: $crate::$desc,
                $(
                    $arg: $crate::__simd_specialize_arg_ty!($isa_arm, $($arg_ty)+),
                )*
            ) -> $crate::__simd_specialize_ret_ty!($isa_arm, $($ret)+) {
                $arm_fn::<$crate::$desc>($d, $($arg,)*)
            }
            // SAFETY: The `as_*` descriptor conversion witnesses that the target feature is enabled.
            #[allow(unsafe_code, unused_unsafe, clippy::undocumented_unsafe_blocks)]
            let __simd_ret = unsafe {
                __simd_trampoline(
                    __simd_d,
                    $(
                        $crate::__simd_specialize_arg_conv!($as_vec, $arr, $arr_mut, $D, $arg, $($arg_ty)+),
                    )*
                )
            };
            $crate::__simd_specialize_ret_conv!($from_vec, $D, __simd_ret, $($ret)+)
        } else $else
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_width_common {
    (
        $width:ident,
        $trait_name:ident,
        $call_macro:ident;
        $d:ident : $D:ident,
        [$(($arg:ident : [$($arg_ty:tt)+]))*],
        [$($ret:tt)+],
        $body:block,
        $else:block
    ) => {
        {
            #[allow(dead_code)]
            #[inline(always)]
            fn __simd_arm<$D: $crate::$trait_name>(
                $d: $D,
                $(
                    $arg: $crate::__simd_specialize_arg_ty!((@width $width $D), $($arg_ty)+),
                )*
            ) -> $crate::__simd_specialize_ret_ty!((@width $width $D), $($ret)+) {
                let _ = (&$d, $(&$arg,)*);
                $body
            }
            $crate::$call_macro!(
                __simd_arm,
                $d : $D,
                [$(($arg : [$($arg_ty)+]))*],
                [$($ret)+],
                $else
            )
        }
    };
}

#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v128 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $crate::__simd_specialize_width_dispatch_backend!(
            sse42,
            Sse42Descriptor,
            "sse4.2",
            as_sse42,
            as_sse42,
            from_sse42,
            array_slice_as_sse42,
            array_slice_mut_as_sse42;
            $arm_fn,
            $d : $D,
            $params,
            $ret,
            $else
        )
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v128 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $crate::__simd_specialize_width_dispatch_backend!(
            neon,
            NeonDescriptor,
            "neon",
            as_neon,
            as_neon,
            from_neon,
            array_slice_as_neon,
            array_slice_mut_as_neon;
            $arm_fn,
            $d : $D,
            $params,
            $ret,
            $else
        )
    };
}

#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v128 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $crate::__simd_specialize_width_dispatch_backend!(
            simd128,
            Simd128Descriptor,
            "simd128",
            as_simd128,
            as_simd128,
            from_simd128,
            array_slice_as_simd128,
            array_slice_mut_as_simd128;
            $arm_fn,
            $d : $D,
            $params,
            $ret,
            $else
        )
    };
}

#[cfg(not(any(
    all(target_arch = "x86_64", feature = "sse42"),
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "simd128")
)))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v128 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v256 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $crate::__simd_specialize_width_dispatch_backend!(
            avx,
            AvxDescriptor,
            "avx2,fma,f16c",
            as_avx,
            as_avx,
            from_avx,
            array_slice_as_avx,
            array_slice_mut_as_avx;
            $arm_fn,
            $d : $D,
            $params,
            $ret,
            $else
        )
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v256 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $else
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v512 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $crate::__simd_specialize_width_dispatch_backend!(
            avx512,
            Avx512Descriptor,
            "avx512f,avx512bw",
            as_avx512,
            as_avx512,
            from_avx512,
            array_slice_as_avx512,
            array_slice_mut_as_avx512;
            $arm_fn,
            $d : $D,
            $params,
            $ret,
            $else
        )
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_call_v512 {
    ($arm_fn:ident, $d:ident : $D:ident, $params:tt, $ret:tt, $else:block) => {
        $else
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_v128 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_width_common!(v128, SimdDescriptor128, __simd_specialize_call_v128; $($tt)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_v256 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_width_common!(v256, SimdDescriptor256, __simd_specialize_call_v256; $($tt)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_arm_v512 {
    ($($tt:tt)*) => {
        $crate::__simd_specialize_arm_width_common!(v512, SimdDescriptor512, __simd_specialize_call_v512; $($tt)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_dispatch {
    (scalar, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_scalar!($($rest)*)
    };
    (sse42, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_sse42!($($rest)*)
    };
    (avx, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_avx!($($rest)*)
    };
    (avx512, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_avx512!($($rest)*)
    };
    (neon, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_neon!($($rest)*)
    };
    (simd128, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_simd128!($($rest)*)
    };
    (v128, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_v128!($($rest)*)
    };
    (v256, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_v256!($($rest)*)
    };
    (v512, $($rest:tt)*) => {
        $crate::__simd_specialize_arm_v512!($($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_emit {
    (@chain [$d:ident : $D:ident] $params:tt $ret:tt [] $fallback:block) => {
        $fallback
    };
    (@chain [$d:ident : $D:ident] $params:tt $ret:tt [($isa:ident => $body:block) $($rest:tt)*] $fallback:block) => {
        $crate::__simd_specialize_dispatch!(
            $isa,
            $d : $D,
            $params,
            $ret,
            $body,
            {
                $crate::__simd_specialize_emit!(@chain [$d : $D] $params $ret [$($rest)*] $fallback)
            }
        )
    };
    ([$d:ident : $D:ident] $params:tt $ret:tt $arms:tt $fallback:block) => {
        {
            let _ = &$d;
            $crate::__simd_specialize_emit!(@chain [$d : $D] $params $ret $arms $fallback)
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_parse_arms {
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($acc:tt)*] _ => $fallback:block $(,)?) => {
        $crate::__simd_specialize_emit!([$d : $D] [$($params)*] [$($ret)+] [$($acc)*] $fallback)
    };
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($acc:tt)*] _ => $fallback:expr $(,)?) => {
        $crate::__simd_specialize_emit!([$d : $D] [$($params)*] [$($ret)+] [$($acc)*] { $fallback })
    };
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($acc:tt)*] $isa:ident => $body:block , $($rest:tt)+) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($params)*] [$($ret)+] [$($acc)* ($isa => $body)] $($rest)+)
    };
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($acc:tt)*] $isa:ident => $body:block $($rest:tt)+) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($params)*] [$($ret)+] [$($acc)* ($isa => $body)] $($rest)+)
    };
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($acc:tt)*] $isa:ident => $body:expr , $($rest:tt)+) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($params)*] [$($ret)+] [$($acc)* ($isa => { $body })] $($rest)+)
    };
    ([$d:ident : $D:ident] [$($params:tt)*] [$($ret:tt)+] [$($arms:tt)+]) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($params)*] [$($ret)+] [] $($arms)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_parse_params {
    (@init [$d:ident : $D:ident] [$($ret:tt)+] [$($arms:tt)*]) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [] [$($ret)+] [$($arms)*])
    };
    (@init [$d:ident : $D:ident] [$($ret:tt)+] [$($arms:tt)*] $arg:ident : $($rest:tt)+) => {
        $crate::__simd_specialize_parse_params!(@ty [$d : $D] [] [$arg] [] [$($ret)+] [$($arms)*] $($rest)+)
    };
    (@ty [$d:ident : $D:ident] [$($parsed:tt)*] [$cur_arg:ident] [$($cur_ty:tt)+] [$($ret:tt)+] [$($arms:tt)*] , $next:ident : $($rest:tt)+) => {
        $crate::__simd_specialize_parse_params!(@ty [$d : $D] [$($parsed)* ($cur_arg : [$($cur_ty)+])] [$next] [] [$($ret)+] [$($arms)*] $($rest)+)
    };
    (@ty [$d:ident : $D:ident] [$($parsed:tt)*] [$cur_arg:ident] [$($cur_ty:tt)+] [$($ret:tt)+] [$($arms:tt)*] ,) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($parsed)* ($cur_arg : [$($cur_ty)+])] [$($ret)+] [$($arms)*])
    };
    (@ty [$d:ident : $D:ident] [$($parsed:tt)*] [$cur_arg:ident] [$($cur_ty:tt)+] [$($ret:tt)+] [$($arms:tt)*]) => {
        $crate::__simd_specialize_parse_arms!([$d : $D] [$($parsed)* ($cur_arg : [$($cur_ty)+])] [$($ret)+] [$($arms)*])
    };
    (@ty [$d:ident : $D:ident] [$($parsed:tt)*] [$cur_arg:ident] [$($cur_ty:tt)*] [$($ret:tt)+] [$($arms:tt)*] $t:tt $($rest:tt)*) => {
        $crate::__simd_specialize_parse_params!(@ty [$d : $D] [$($parsed)*] [$cur_arg] [$($cur_ty)* $t] [$($ret)+] [$($arms)*] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_specialize_split_ret {
    (@ret [$($ret:tt)+] [$d:ident : $D:ident] [$($rest_params:tt)*] { $($arms:tt)* }) => {
        $crate::__simd_specialize_parse_params!(@init [$d : $D] [$($ret)+] [$($arms)*] $($rest_params)*)
    };
    (@ret [$($ret:tt)*] [$d:ident : $D:ident] [$($rest_params:tt)*] $t:tt $($tail:tt)+) => {
        $crate::__simd_specialize_split_ret!(@ret [$($ret)* $t] [$d : $D] [$($rest_params)*] $($tail)+)
    };
}

/// Dispatches to a platform-specific SIMD implementation when `d` matches an arm's ISA, falling
/// back to `_ => { ... }` otherwise.
#[macro_export]
macro_rules! simd_specialize {
    (
        fn($d:ident : $D:ident $(, $($rest_params:tt)*)?) -> $($after:tt)+
    ) => {
        $crate::__simd_specialize_split_ret!(@ret [] [$d : $D] [$($($rest_params)*)?] $($after)+)
    };
    (
        fn($d:ident : $D:ident $(, $($rest_params:tt)*)?) { $($arms:tt)* }
    ) => {
        $crate::__simd_specialize_parse_params!(@init [$d : $D] [()] [$($arms)*] $($($rest_params)*)?)
    };
}

#[cfg(test)]
mod test {
    use arbtest::arbitrary::Unstructured;

    use crate::{
        F32SimdVec, I16SimdVec, I32SimdVec, ScalarDescriptor, SimdDescriptor, SimdMask16,
        U8SimdVec, U16SimdVec, U64SimdVec, test_all_instruction_sets,
    };

    enum Distribution {
        Floats,
        NonZeroFloats,
    }

    fn arb_vec<D: SimdDescriptor>(_: D, u: &mut Unstructured, dist: Distribution) -> Vec<f32> {
        let mut res = vec![0.0; D::F32Vec::LEN];
        for v in res.iter_mut() {
            match dist {
                Distribution::Floats => {
                    *v = u.arbitrary::<i32>().unwrap() as f32
                        / (1.0 + u.arbitrary::<u32>().unwrap() as f32)
                }
                Distribution::NonZeroFloats => {
                    let sign = if u.arbitrary::<bool>().unwrap() {
                        1.0
                    } else {
                        -1.0
                    };
                    *v = sign * (1.0 + u.arbitrary::<u32>().unwrap() as f32)
                        / (1.0 + u.arbitrary::<u32>().unwrap() as f32);
                }
            }
        }
        res
    }

    fn compare_scalar_simd(scalar: f32, simd: f32, max_abs: f32, max_rel: f32) {
        let abs = (simd - scalar).abs();
        let max = simd.abs().max(scalar.abs());
        let rel = abs / max;
        assert!(
            abs < max_abs || rel < max_rel,
            "simd {simd}, scalar {scalar}, abs {abs:?} rel {rel:?}",
        );
    }

    macro_rules! test_instruction {
        ($name:ident, |$a:ident: $a_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(D::F32Vec::load(d, &a[idx..])).store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a);
                    let simd_res = compute(d, &a);
                    for (scalar, simd) in scalar_res.iter().zip(simd_res.iter()) {
                        compare_scalar_simd(*scalar, *simd, 1e-6, 1e-6);
                    }
                    Ok(())
                })
                .size_min(64);
            }
            test_all_instruction_sets!($name);
        };
        ($name:ident, |$a:ident: $a_dist:ident, $b:ident: $b_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec, $b: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(D::F32Vec::load(d, &a[idx..]), D::F32Vec::load(d, &b[idx..]))
                            .store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let b = arb_vec(d, u, Distribution::$b_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a, &b);
                    let simd_res = compute(d, &a, &b);
                    for (scalar, simd) in scalar_res.iter().zip(simd_res.iter()) {
                        compare_scalar_simd(*scalar, *simd, 1e-6, 1e-6);
                    }
                    Ok(())
                })
                .size_min(128);
            }
            test_all_instruction_sets!($name);
        };
        ($name:ident, |$a:ident: $a_dist:ident, $b:ident: $b_dist:ident, $c:ident: $c_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec, $b: D::F32Vec, $c: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(
                            D::F32Vec::load(d, &a[idx..]),
                            D::F32Vec::load(d, &b[idx..]),
                            D::F32Vec::load(d, &c[idx..]),
                        )
                        .store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let b = arb_vec(d, u, Distribution::$b_dist);
                    let c = arb_vec(d, u, Distribution::$c_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a, &b, &c);
                    let simd_res = compute(d, &a, &b, &c);
                    for (scalar, simd) in scalar_res.iter().zip(simd_res.iter()) {
                        // Less strict requirements because of fma.
                        compare_scalar_simd(*scalar, *simd, 3e-5, 3e-5);
                    }
                    Ok(())
                })
                .size_min(172);
            }
            test_all_instruction_sets!($name);
        };
    }

    test_instruction!(add, |a: Floats, b: Floats| { a + b });
    test_instruction!(mul, |a: Floats, b: Floats| { a * b });
    test_instruction!(sub, |a: Floats, b: Floats| { a - b });
    test_instruction!(div, |a: Floats, b: NonZeroFloats| { a / b });

    test_instruction!(add_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res += b;
        res
    });
    test_instruction!(mul_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res *= b;
        res
    });
    test_instruction!(sub_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res -= b;
        res
    });
    test_instruction!(div_assign, |a: Floats, b: NonZeroFloats| {
        let mut res = a;
        res /= b;
        res
    });

    test_instruction!(mul_add, |a: Floats, b: Floats, c: Floats| {
        a.mul_add(b, c)
    });

    test_instruction!(neg_mul_add, |a: Floats, b: Floats, c: Floats| {
        a.neg_mul_add(b, c)
    });

    // Validate that neg_mul_add computes c - a * b correctly
    fn test_neg_mul_add_correctness<D: SimdDescriptor>(d: D) {
        let a_vals = [
            2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 2.5, 3.5, 4.5, 5.5, 1.0, 2.0, 3.0, 4.0,
        ];
        let b_vals = [
            1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 4.5, 0.25, 0.75, 1.25, 1.75,
        ];
        let c_vals = [
            10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0, 42.0, 6.0, 16.0, 26.0,
            36.0,
        ];

        let a = D::F32Vec::load(d, &a_vals[..D::F32Vec::LEN]);
        let b = D::F32Vec::load(d, &b_vals[..D::F32Vec::LEN]);
        let c = D::F32Vec::load(d, &c_vals[..D::F32Vec::LEN]);

        let result = a.neg_mul_add(b, c);
        let expected = c - a * b;

        let mut result_vals = [0.0; 16];
        let mut expected_vals = [0.0; 16];
        result.store(&mut result_vals[..D::F32Vec::LEN]);
        expected.store(&mut expected_vals[..D::F32Vec::LEN]);

        for i in 0..D::F32Vec::LEN {
            assert!(
                (result_vals[i] - expected_vals[i]).abs() < 1e-5,
                "neg_mul_add correctness failed at index {}: got {}, expected {}",
                i,
                result_vals[i],
                expected_vals[i]
            );
        }
    }

    test_all_instruction_sets!(test_neg_mul_add_correctness);

    test_instruction!(abs, |a: Floats| { a.abs() });
    test_instruction!(max, |a: Floats, b: Floats| { a.max(b) });
    test_instruction!(min, |a: Floats, b: Floats| { a.min(b) });

    // Test that the call method works, compiles, and can capture arguments
    fn test_call<D: SimdDescriptor>(d: D) {
        // Test basic call functionality
        let result = d.call(|_d| 42);
        assert_eq!(result, 42);

        // Test with capturing variables
        let multiplier = 3.0f32;
        let addend = 5.0f32;

        // Test SIMD operations inside call with captures
        let input = vec![1.0f32; D::F32Vec::LEN * 4];
        let mut output = vec![0.0f32; D::F32Vec::LEN * 4];

        d.call(|d| {
            let mult_vec = D::F32Vec::splat(d, multiplier);
            let add_vec = D::F32Vec::splat(d, addend);

            for idx in (0..input.len()).step_by(D::F32Vec::LEN) {
                let vec = D::F32Vec::load(d, &input[idx..]);
                let result = vec * mult_vec + add_vec;
                result.store(&mut output[idx..]);
            }
        });

        // Verify results
        for &val in &output {
            assert_eq!(val, 1.0 * multiplier + addend);
        }
    }
    test_all_instruction_sets!(test_call);

    fn test_neg<D: SimdDescriptor>(d: D) {
        // Test negation operation with enough elements for any SIMD size
        let len = D::F32Vec::LEN * 2; // Ensure we have at least 2 full vectors
        let input: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
            .collect();
        let expected: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
            .collect();
        let mut output = vec![0.0f32; input.len()];

        for idx in (0..input.len()).step_by(D::F32Vec::LEN) {
            let vec = D::F32Vec::load(d, &input[idx..]);
            let negated = vec.neg();
            negated.store(&mut output[idx..]);
        }

        for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                out, exp,
                "Mismatch at index {}: expected {}, got {}",
                i, exp, out
            );
        }
    }
    test_all_instruction_sets!(test_neg);

    fn test_store_interleaved_2<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let mut output = vec![0.0f32; 2 * len];

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);
        D::F32Vec::store_interleaved_2(a_vec, b_vec, &mut output);

        // Verify interleaved output: [a0, b0, a1, b1, ...]
        for i in 0..len {
            assert_eq!(
                output[2 * i],
                a[i],
                "store_interleaved_2 failed at position {}: expected a[{}]={}, got {}",
                2 * i,
                i,
                a[i],
                output[2 * i]
            );
            assert_eq!(
                output[2 * i + 1],
                b[i],
                "store_interleaved_2 failed at position {}: expected b[{}]={}, got {}",
                2 * i + 1,
                i,
                b[i],
                output[2 * i + 1]
            );
        }
    }
    test_all_instruction_sets!(test_store_interleaved_2);

    fn test_store_interleaved_3<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let c: Vec<f32> = (0..len).map(|i| (i + 200) as f32).collect();
        let mut output = vec![0.0f32; 3 * len];

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);
        let c_vec = D::F32Vec::load(d, &c);
        D::F32Vec::store_interleaved_3(a_vec, b_vec, c_vec, &mut output);

        // Verify interleaved output: [a0, b0, c0, a1, b1, c1, ...]
        for i in 0..len {
            assert_eq!(
                output[3 * i],
                a[i],
                "store_interleaved_3 failed at position {}: expected a[{}]={}, got {}",
                3 * i,
                i,
                a[i],
                output[3 * i]
            );
            assert_eq!(
                output[3 * i + 1],
                b[i],
                "store_interleaved_3 failed at position {}: expected b[{}]={}, got {}",
                3 * i + 1,
                i,
                b[i],
                output[3 * i + 1]
            );
            assert_eq!(
                output[3 * i + 2],
                c[i],
                "store_interleaved_3 failed at position {}: expected c[{}]={}, got {}",
                3 * i + 2,
                i,
                c[i],
                output[3 * i + 2]
            );
        }
    }
    test_all_instruction_sets!(test_store_interleaved_3);

    fn test_store_interleaved_4<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let c: Vec<f32> = (0..len).map(|i| (i + 200) as f32).collect();
        let e: Vec<f32> = (0..len).map(|i| (i + 300) as f32).collect();
        let mut output = vec![0.0f32; 4 * len];

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);
        let c_vec = D::F32Vec::load(d, &c);
        let d_vec = D::F32Vec::load(d, &e);
        D::F32Vec::store_interleaved_4(a_vec, b_vec, c_vec, d_vec, &mut output);

        // Verify interleaved output: [a0, b0, c0, d0, a1, b1, c1, d1, ...]
        for i in 0..len {
            assert_eq!(
                output[4 * i],
                a[i],
                "store_interleaved_4 failed at position {}: expected a[{}]={}, got {}",
                4 * i,
                i,
                a[i],
                output[4 * i]
            );
            assert_eq!(
                output[4 * i + 1],
                b[i],
                "store_interleaved_4 failed at position {}: expected b[{}]={}, got {}",
                4 * i + 1,
                i,
                b[i],
                output[4 * i + 1]
            );
            assert_eq!(
                output[4 * i + 2],
                c[i],
                "store_interleaved_4 failed at position {}: expected c[{}]={}, got {}",
                4 * i + 2,
                i,
                c[i],
                output[4 * i + 2]
            );
            assert_eq!(
                output[4 * i + 3],
                e[i],
                "store_interleaved_4 failed at position {}: expected d[{}]={}, got {}",
                4 * i + 3,
                i,
                e[i],
                output[4 * i + 3]
            );
        }
    }
    test_all_instruction_sets!(test_store_interleaved_4);

    fn test_store_interleaved_8<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let arr_a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let arr_b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let arr_c: Vec<f32> = (0..len).map(|i| (i + 200) as f32).collect();
        let arr_d: Vec<f32> = (0..len).map(|i| (i + 300) as f32).collect();
        let arr_e: Vec<f32> = (0..len).map(|i| (i + 400) as f32).collect();
        let arr_f: Vec<f32> = (0..len).map(|i| (i + 500) as f32).collect();
        let arr_g: Vec<f32> = (0..len).map(|i| (i + 600) as f32).collect();
        let arr_h: Vec<f32> = (0..len).map(|i| (i + 700) as f32).collect();
        let mut output = vec![0.0f32; 8 * len];

        let a = D::F32Vec::load(d, &arr_a);
        let b = D::F32Vec::load(d, &arr_b);
        let c = D::F32Vec::load(d, &arr_c);
        let dv = D::F32Vec::load(d, &arr_d);
        let e = D::F32Vec::load(d, &arr_e);
        let f = D::F32Vec::load(d, &arr_f);
        let g = D::F32Vec::load(d, &arr_g);
        let h = D::F32Vec::load(d, &arr_h);
        D::F32Vec::store_interleaved_8(a, b, c, dv, e, f, g, h, &mut output);

        // Verify interleaved output: [a0, b0, c0, d0, e0, f0, g0, h0, a1, ...]
        let arrays = [
            &arr_a, &arr_b, &arr_c, &arr_d, &arr_e, &arr_f, &arr_g, &arr_h,
        ];
        for i in 0..len {
            for (j, arr) in arrays.iter().enumerate() {
                assert_eq!(
                    output[8 * i + j],
                    arr[i],
                    "store_interleaved_8 failed at position {}: expected {}[{}]={}, got {}",
                    8 * i + j,
                    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][j],
                    i,
                    arr[i],
                    output[8 * i + j]
                );
            }
        }
    }
    test_all_instruction_sets!(test_store_interleaved_8);

    fn test_load_deinterleaved_2<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        // Create interleaved input: [a0, b0, a1, b1, ...]
        let mut interleaved = vec![0.0f32; 2 * len];
        let expected_a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let expected_b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        for i in 0..len {
            interleaved[2 * i] = expected_a[i];
            interleaved[2 * i + 1] = expected_b[i];
        }

        let (a_vec, b_vec) = D::F32Vec::load_deinterleaved_2(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        a_vec.store(&mut out_a);
        b_vec.store(&mut out_b);

        for i in 0..len {
            assert_eq!(
                out_a[i], expected_a[i],
                "load_deinterleaved_2 failed for channel a at {}: expected {}, got {}",
                i, expected_a[i], out_a[i]
            );
            assert_eq!(
                out_b[i], expected_b[i],
                "load_deinterleaved_2 failed for channel b at {}: expected {}, got {}",
                i, expected_b[i], out_b[i]
            );
        }
    }
    test_all_instruction_sets!(test_load_deinterleaved_2);

    fn test_load_deinterleaved_3<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        // Create interleaved input: [a0, b0, c0, a1, b1, c1, ...]
        let mut interleaved = vec![0.0f32; 3 * len];
        let expected_a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let expected_b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let expected_c: Vec<f32> = (0..len).map(|i| (i + 200) as f32).collect();
        for i in 0..len {
            interleaved[3 * i] = expected_a[i];
            interleaved[3 * i + 1] = expected_b[i];
            interleaved[3 * i + 2] = expected_c[i];
        }

        let (a_vec, b_vec, c_vec) = D::F32Vec::load_deinterleaved_3(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        let mut out_c = vec![0.0f32; len];
        a_vec.store(&mut out_a);
        b_vec.store(&mut out_b);
        c_vec.store(&mut out_c);

        for i in 0..len {
            assert_eq!(
                out_a[i], expected_a[i],
                "load_deinterleaved_3 failed for channel a at {}: expected {}, got {}",
                i, expected_a[i], out_a[i]
            );
            assert_eq!(
                out_b[i], expected_b[i],
                "load_deinterleaved_3 failed for channel b at {}: expected {}, got {}",
                i, expected_b[i], out_b[i]
            );
            assert_eq!(
                out_c[i], expected_c[i],
                "load_deinterleaved_3 failed for channel c at {}: expected {}, got {}",
                i, expected_c[i], out_c[i]
            );
        }
    }
    test_all_instruction_sets!(test_load_deinterleaved_3);

    fn test_load_deinterleaved_4<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        // Create interleaved input: [a0, b0, c0, d0, a1, b1, c1, d1, ...]
        let mut interleaved = vec![0.0f32; 4 * len];
        let expected_a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let expected_b: Vec<f32> = (0..len).map(|i| (i + 100) as f32).collect();
        let expected_c: Vec<f32> = (0..len).map(|i| (i + 200) as f32).collect();
        let expected_d: Vec<f32> = (0..len).map(|i| (i + 300) as f32).collect();
        for i in 0..len {
            interleaved[4 * i] = expected_a[i];
            interleaved[4 * i + 1] = expected_b[i];
            interleaved[4 * i + 2] = expected_c[i];
            interleaved[4 * i + 3] = expected_d[i];
        }

        let (a_vec, b_vec, c_vec, d_vec) = D::F32Vec::load_deinterleaved_4(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        let mut out_c = vec![0.0f32; len];
        let mut out_d = vec![0.0f32; len];
        a_vec.store(&mut out_a);
        b_vec.store(&mut out_b);
        c_vec.store(&mut out_c);
        d_vec.store(&mut out_d);

        for i in 0..len {
            assert_eq!(
                out_a[i], expected_a[i],
                "load_deinterleaved_4 failed for channel a at {}: expected {}, got {}",
                i, expected_a[i], out_a[i]
            );
            assert_eq!(
                out_b[i], expected_b[i],
                "load_deinterleaved_4 failed for channel b at {}: expected {}, got {}",
                i, expected_b[i], out_b[i]
            );
            assert_eq!(
                out_c[i], expected_c[i],
                "load_deinterleaved_4 failed for channel c at {}: expected {}, got {}",
                i, expected_c[i], out_c[i]
            );
            assert_eq!(
                out_d[i], expected_d[i],
                "load_deinterleaved_4 failed for channel d at {}: expected {}, got {}",
                i, expected_d[i], out_d[i]
            );
        }
    }
    test_all_instruction_sets!(test_load_deinterleaved_4);

    // Roundtrip tests: verify store_interleaved + load_deinterleaved returns original data
    fn test_interleave_roundtrip_2<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| (i * 7 + 3) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i * 11 + 5) as f32).collect();

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);

        let mut interleaved = vec![0.0f32; 2 * len];
        D::F32Vec::store_interleaved_2(a_vec, b_vec, &mut interleaved);

        let (a_out, b_out) = D::F32Vec::load_deinterleaved_2(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        a_out.store(&mut out_a);
        b_out.store(&mut out_b);

        assert_eq!(out_a, a, "interleave_roundtrip_2 failed for channel a");
        assert_eq!(out_b, b, "interleave_roundtrip_2 failed for channel b");
    }
    test_all_instruction_sets!(test_interleave_roundtrip_2);

    fn test_interleave_roundtrip_3<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| (i * 7 + 3) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i * 11 + 5) as f32).collect();
        let c: Vec<f32> = (0..len).map(|i| (i * 13 + 9) as f32).collect();

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);
        let c_vec = D::F32Vec::load(d, &c);

        let mut interleaved = vec![0.0f32; 3 * len];
        D::F32Vec::store_interleaved_3(a_vec, b_vec, c_vec, &mut interleaved);

        let (a_out, b_out, c_out) = D::F32Vec::load_deinterleaved_3(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        let mut out_c = vec![0.0f32; len];
        a_out.store(&mut out_a);
        b_out.store(&mut out_b);
        c_out.store(&mut out_c);

        assert_eq!(out_a, a, "interleave_roundtrip_3 failed for channel a");
        assert_eq!(out_b, b, "interleave_roundtrip_3 failed for channel b");
        assert_eq!(out_c, c, "interleave_roundtrip_3 failed for channel c");
    }
    test_all_instruction_sets!(test_interleave_roundtrip_3);

    fn test_interleave_roundtrip_4<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let a: Vec<f32> = (0..len).map(|i| (i * 7 + 3) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (i * 11 + 5) as f32).collect();
        let c: Vec<f32> = (0..len).map(|i| (i * 13 + 9) as f32).collect();
        let e: Vec<f32> = (0..len).map(|i| (i * 17 + 1) as f32).collect();

        let a_vec = D::F32Vec::load(d, &a);
        let b_vec = D::F32Vec::load(d, &b);
        let c_vec = D::F32Vec::load(d, &c);
        let d_vec = D::F32Vec::load(d, &e);

        let mut interleaved = vec![0.0f32; 4 * len];
        D::F32Vec::store_interleaved_4(a_vec, b_vec, c_vec, d_vec, &mut interleaved);

        let (a_out, b_out, c_out, d_out) = D::F32Vec::load_deinterleaved_4(d, &interleaved);

        let mut out_a = vec![0.0f32; len];
        let mut out_b = vec![0.0f32; len];
        let mut out_c = vec![0.0f32; len];
        let mut out_d = vec![0.0f32; len];
        a_out.store(&mut out_a);
        b_out.store(&mut out_b);
        c_out.store(&mut out_c);
        d_out.store(&mut out_d);

        assert_eq!(out_a, a, "interleave_roundtrip_4 failed for channel a");
        assert_eq!(out_b, b, "interleave_roundtrip_4 failed for channel b");
        assert_eq!(out_c, c, "interleave_roundtrip_4 failed for channel c");
        assert_eq!(out_d, e, "interleave_roundtrip_4 failed for channel d");
    }
    test_all_instruction_sets!(test_interleave_roundtrip_4);

    fn test_prepare_table_bf16_8<D: SimdDescriptor>(d: D) {
        // Create an 8-entry lookup table with known values
        // Use integer values that are exactly representable in BF16
        let lut: [f32; 8] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let len = D::F32Vec::LEN;

        // Prepare the table once
        let prepared = D::F32Vec::prepare_table_bf16_8(d, &lut);

        // Create indices that are valid for the LUT (0..8)
        let indices: Vec<i32> = (0..len).map(|i| (i % 8) as i32).collect();
        let expected: Vec<f32> = indices.iter().map(|&i| lut[i as usize]).collect();

        // Perform table lookup with prepared table
        let indices_vec = D::I32Vec::load(d, &indices);
        let result = D::F32Vec::table_lookup_bf16_8(d, prepared, indices_vec);

        let mut output = vec![0.0f32; len];
        result.store(&mut output);

        // Verify results - prepared lookup may have BF16 precision loss
        // BF16 has ~0.4% relative error for typical values
        for i in 0..len {
            let tolerance = if expected[i] == 0.0 {
                0.01
            } else {
                expected[i].abs() * 0.01
            };
            assert!(
                (output[i] - expected[i]).abs() < tolerance,
                "table_lookup_bf16_8 failed at position {}: expected {}, got {}",
                i,
                expected[i],
                output[i]
            );
        }
    }
    test_all_instruction_sets!(test_prepare_table_bf16_8);

    /// Test that I32 multiplication operates on all elements, not just alternating lanes.
    /// This catches the bug where _mm_mul_epi32 was used instead of _mm_mullo_epi32.
    fn test_i32_mul_all_elements<D: SimdDescriptor>(d: D) {
        let len = D::I32Vec::LEN;

        // Create input vectors where each lane has a unique value
        let mut a_data = vec![0i32; len];
        let mut b_data = vec![0i32; len];
        for i in 0..len {
            a_data[i] = (i + 1) as i32; // [1, 2, 3, 4, ...]
            b_data[i] = (i + 10) as i32; // [10, 11, 12, 13, ...]
        }

        let a = D::I32Vec::load(d, &a_data);
        let b = D::I32Vec::load(d, &b_data);
        let result = a * b;

        let mut result_data = vec![0i32; len];
        result.store(&mut result_data);

        // Verify EVERY element was multiplied correctly
        for i in 0..len {
            let expected = a_data[i] * b_data[i];
            assert_eq!(
                result_data[i], expected,
                "I32 mul failed at index {}: {} * {} = {}, got {}",
                i, a_data[i], b_data[i], expected, result_data[i]
            );
        }
    }
    test_all_instruction_sets!(test_i32_mul_all_elements);

    fn test_store_u16<D: SimdDescriptor>(d: D) {
        let data = [
            0xbabau32 as i32,
            0x1234u32 as i32,
            0xdeadbabau32 as i32,
            0xdead1234u32 as i32,
            0x1111babau32 as i32,
            0x11111234u32 as i32,
            0x76543210u32 as i32,
            0x01234567u32 as i32,
            0x00000000u32 as i32,
            0xffffffffu32 as i32,
            0x23949289u32 as i32,
            0xf9371913u32 as i32,
            0xdeadbeefu32 as i32,
            0xbeefdeadu32 as i32,
            0xaaaaaaaau32 as i32,
            0xbbbbbbbbu32 as i32,
        ];
        let mut output = [0u16; 16];
        for i in (0..16).step_by(D::I32Vec::LEN) {
            let vec = D::I32Vec::load(d, &data[i..]);
            vec.store_u16(&mut output[i..]);
        }

        for i in 0..16 {
            let expected = data[i] as u16;
            assert_eq!(
                output[i], expected,
                "store_u16 failed at index {}: expected {}, got {}",
                i, expected, output[i]
            );
        }
    }
    test_all_instruction_sets!(test_store_u16);

    fn test_store_interleaved_2_u8<D: SimdDescriptor>(d: D) {
        let len = D::U8Vec::LEN;
        let a: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i + 100) as u8).collect();
        let mut output = vec![0u8; 2 * len];

        let a_vec = D::U8Vec::load(d, &a);
        let b_vec = D::U8Vec::load(d, &b);
        D::U8Vec::store_interleaved_2(a_vec, b_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[2 * i], a[i]);
            assert_eq!(output[2 * i + 1], b[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_2_u8);

    fn test_store_interleaved_3_u8<D: SimdDescriptor>(d: D) {
        let len = D::U8Vec::LEN;
        let a: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i + 100) as u8).collect();
        let c: Vec<u8> = (0..len).map(|i| (i + 50) as u8).collect();
        let mut output = vec![0u8; 3 * len];

        let a_vec = D::U8Vec::load(d, &a);
        let b_vec = D::U8Vec::load(d, &b);
        let c_vec = D::U8Vec::load(d, &c);
        D::U8Vec::store_interleaved_3(a_vec, b_vec, c_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[3 * i], a[i]);
            assert_eq!(output[3 * i + 1], b[i]);
            assert_eq!(output[3 * i + 2], c[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_3_u8);

    fn test_store_interleaved_4_u8<D: SimdDescriptor>(d: D) {
        let len = D::U8Vec::LEN;
        let a: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i + 100) as u8).collect();
        let c: Vec<u8> = (0..len).map(|i| (i + 50) as u8).collect();
        let e: Vec<u8> = (0..len).map(|i| (i + 200) as u8).collect();
        let mut output = vec![0u8; 4 * len];

        let a_vec = D::U8Vec::load(d, &a);
        let b_vec = D::U8Vec::load(d, &b);
        let c_vec = D::U8Vec::load(d, &c);
        let d_vec = D::U8Vec::load(d, &e);
        D::U8Vec::store_interleaved_4(a_vec, b_vec, c_vec, d_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[4 * i], a[i]);
            assert_eq!(output[4 * i + 1], b[i]);
            assert_eq!(output[4 * i + 2], c[i]);
            assert_eq!(output[4 * i + 3], e[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_4_u8);

    fn test_store_interleaved_2_u16<D: SimdDescriptor>(d: D) {
        let len = D::U16Vec::LEN;
        let a: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let b: Vec<u16> = (0..len).map(|i| (i + 1000) as u16).collect();
        let mut output = vec![0u16; 2 * len];

        let a_vec = D::U16Vec::load(d, &a);
        let b_vec = D::U16Vec::load(d, &b);
        D::U16Vec::store_interleaved_2(a_vec, b_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[2 * i], a[i]);
            assert_eq!(output[2 * i + 1], b[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_2_u16);

    fn test_store_interleaved_3_u16<D: SimdDescriptor>(d: D) {
        let len = D::U16Vec::LEN;
        let a: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let b: Vec<u16> = (0..len).map(|i| (i + 1000) as u16).collect();
        let c: Vec<u16> = (0..len).map(|i| (i + 2000) as u16).collect();
        let mut output = vec![0u16; 3 * len];

        let a_vec = D::U16Vec::load(d, &a);
        let b_vec = D::U16Vec::load(d, &b);
        let c_vec = D::U16Vec::load(d, &c);
        D::U16Vec::store_interleaved_3(a_vec, b_vec, c_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[3 * i], a[i]);
            assert_eq!(output[3 * i + 1], b[i]);
            assert_eq!(output[3 * i + 2], c[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_3_u16);

    fn test_store_interleaved_4_u16<D: SimdDescriptor>(d: D) {
        let len = D::U16Vec::LEN;
        let a: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let b: Vec<u16> = (0..len).map(|i| (i + 1000) as u16).collect();
        let c: Vec<u16> = (0..len).map(|i| (i + 2000) as u16).collect();
        let e: Vec<u16> = (0..len).map(|i| (i + 3000) as u16).collect();
        let mut output = vec![0u16; 4 * len];

        let a_vec = D::U16Vec::load(d, &a);
        let b_vec = D::U16Vec::load(d, &b);
        let c_vec = D::U16Vec::load(d, &c);
        let d_vec = D::U16Vec::load(d, &e);
        D::U16Vec::store_interleaved_4(a_vec, b_vec, c_vec, d_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[4 * i], a[i]);
            assert_eq!(output[4 * i + 1], b[i]);
            assert_eq!(output[4 * i + 2], c[i]);
            assert_eq!(output[4 * i + 3], e[i]);
        }
    }
    test_all_instruction_sets!(test_store_interleaved_4_u16);

    fn test_store_u8<D: SimdDescriptor>(d: D) {
        let data = [
            0xba_i32,
            0x12345678_i32,
            0xdeadbabeu32 as i32,
            0x76543210_i32,
            0x11111111_i32,
            0x00000000_i32,
            0xffffffffu32 as i32,
            0x12345678_i32,
            0x87654321u32 as i32,
            0xabcdef01u32 as i32,
            0x10203040_i32,
            0x50607080_i32,
            0x01020304_i32,
            0x05060708_i32,
            0x090a0b0c_i32,
            0x0d0e0f00_i32,
        ];
        let mut output = [0u8; 16];
        for i in (0..16).step_by(D::I32Vec::LEN) {
            let vec = D::I32Vec::load(d, &data[i..]);
            vec.store_u8(&mut output[i..]);
        }

        for i in 0..16 {
            let expected = (data[i] & 0xff) as u8;
            assert_eq!(
                output[i], expected,
                "store_u8 failed at index {}: expected 0x{:02x}, got 0x{:02x}",
                i, expected, output[i]
            );
        }
    }
    test_all_instruction_sets!(test_store_u8);

    fn test_f16_load_roundtrip<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let values: &[f32] = &[0.0, 1.0, -1.0, 0.5, 100.0, -42.5, 65504.0, 0.001];
        for chunk in values.chunks(len) {
            let mut input = vec![0.0f32; len];
            input[..chunk.len()].copy_from_slice(chunk);
            let v = D::F32Vec::load(d, &input);
            let mut f16_buf = vec![0u16; len];
            v.store_f16_bits(&mut f16_buf);
            let back = D::F32Vec::load_f16_bits(d, &f16_buf);
            let mut output = vec![0.0f32; len];
            back.store(&mut output);
            for i in 0..chunk.len() {
                let expected = input[i];
                if expected == 0.0 {
                    assert_eq!(
                        output[i].to_bits() & 0x7FFF_FFFF,
                        0,
                        "zero not preserved at {i}"
                    );
                } else {
                    let rel_err = ((expected - output[i]) / expected).abs();
                    assert!(
                        rel_err < 0.001,
                        "roundtrip failed at {i}: {expected} -> {}",
                        output[i]
                    );
                }
            }
        }
    }
    test_all_instruction_sets!(test_f16_load_roundtrip);

    fn test_f16_load_special_values<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let cases: &[(u16, f32)] = &[
            (0x0001, 5.960_464_5e-8),    // smallest positive subnormal
            (0x03FF, 6.097_555e-5),      // largest positive subnormal
            (0x8001, -5.960_464_5e-8),   // smallest negative subnormal
            (0x7C00, f32::INFINITY),     // +inf
            (0xFC00, f32::NEG_INFINITY), // -inf
            (0x7C01, f32::NAN),          // NaN
            (0x0000, 0.0),               // +zero
        ];
        for chunk in cases.chunks(len) {
            let mut bits = vec![0u16; len];
            for (i, &(b, _)) in chunk.iter().enumerate() {
                bits[i] = b;
            }
            let v = D::F32Vec::load_f16_bits(d, &bits);
            let mut output = vec![0.0f32; len];
            v.store(&mut output);
            for (i, &(_, expected)) in chunk.iter().enumerate() {
                if expected.is_nan() {
                    assert!(output[i].is_nan(), "expected NaN at {i}, got {}", output[i]);
                } else if expected.is_infinite() {
                    assert_eq!(output[i], expected, "inf mismatch at {i}");
                } else if expected == 0.0 {
                    assert_eq!(output[i], 0.0, "zero mismatch at {i}");
                } else {
                    assert!(
                        (output[i] - expected).abs() < expected.abs() * 1e-3,
                        "subnormal mismatch at {i}: expected {expected}, got {}",
                        output[i],
                    );
                }
            }
        }
    }
    test_all_instruction_sets!(test_f16_load_special_values);

    fn test_f16_store_roundtrip<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let known: &[(f32, u16)] = &[(1.0, 0x3C00), (-2.0, 0xC000), (0.5, 0x3800), (0.0, 0x0000)];
        let mut input = vec![0.0f32; len];
        let mut expected_bits = vec![0u16; len];
        for i in 0..len.min(known.len()) {
            input[i] = known[i].0;
            expected_bits[i] = known[i].1;
        }
        let v = D::F32Vec::load(d, &input);
        let mut output = vec![0u16; len];
        v.store_f16_bits(&mut output);
        for i in 0..len.min(known.len()) {
            assert_eq!(
                output[i], expected_bits[i],
                "store_f16_bits wrong at {i}: input {}, expected 0x{:04X}, got 0x{:04X}",
                input[i], expected_bits[i], output[i],
            );
        }
    }
    test_all_instruction_sets!(test_f16_store_roundtrip);

    fn test_f16_store_special_values<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        // (f32 input, expected f16 bits)
        let cases: &[(f32, u16)] = &[
            (5.960_464_5e-8, 0x0001),    // smallest positive subnormal
            (6.097_555e-5, 0x03FF),      // largest positive subnormal
            (-5.960_464_5e-8, 0x8001),   // smallest negative subnormal
            (f32::INFINITY, 0x7C00),     // +inf
            (f32::NEG_INFINITY, 0xFC00), // -inf
            (100000.0, 0x7C00),          // overflow to inf
            (65504.0, 0x7BFF),           // max normal f16
            (f32::NAN, 0x7E00),          // NaN preserved
        ];
        for chunk in cases.chunks(len) {
            let mut input = vec![0.0f32; len];
            for (i, &(v, _)) in chunk.iter().enumerate() {
                input[i] = v;
            }
            let v = D::F32Vec::load(d, &input);
            let mut output = vec![0u16; len];
            v.store_f16_bits(&mut output);
            for (i, &(inp, expected)) in chunk.iter().enumerate() {
                assert_eq!(
                    output[i], expected,
                    "store_f16_bits special at {i}: input {inp}, expected 0x{expected:04X}, got 0x{:04X}",
                    output[i],
                );
            }
        }
    }
    test_all_instruction_sets!(test_f16_store_special_values);

    fn test_f16_underflow_to_zero<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let mut input = vec![0.0f32; len];
        input[0] = 1e-8;
        if len > 1 {
            input[1] = -1e-8;
        }
        if len > 2 {
            input[2] = 1e-10;
        }
        let v = D::F32Vec::load(d, &input);
        let mut f16_buf = vec![0u16; len];
        v.store_f16_bits(&mut f16_buf);
        // Magnitude bits should be zero (sign may be preserved)
        for (i, &bits) in f16_buf.iter().enumerate().take(3.min(len)) {
            assert_eq!(
                bits & 0x7FFF,
                0,
                "underflow not flushed to zero at {i}: got 0x{bits:04X}",
            );
        }
    }
    test_all_instruction_sets!(test_f16_underflow_to_zero);

    fn compare_f16_bits(s: u16, m: u16) {
        let (s_exp, m_exp) = ((s >> 10) & 0x1F, (m >> 10) & 0x1F);
        if s_exp == 0x1F || m_exp == 0x1F {
            assert_eq!(s_exp, m_exp, "inf/nan: scalar 0x{s:04X}, simd 0x{m:04X}");
            return;
        }
        let (s_mag, m_mag) = (s & 0x7FFF, m & 0x7FFF);
        let diff_mag = (s_mag as i32 - m_mag as i32).unsigned_abs();
        assert!(diff_mag <= 1, "magnitude: scalar 0x{s:04X}, simd 0x{m:04X}");
        if s_mag != 0 || m_mag != 0 {
            let (s_sign, m_sign) = (s & 0x8000, m & 0x8000);
            assert_eq!(s_sign, m_sign, "sign: scalar 0x{s:04X}, simd 0x{m:04X}");
        }
    }

    fn test_f16_store_scalar_equivalent<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        arbtest::arbtest(|u| {
            let mut input = vec![0.0f32; len];
            for v in input.iter_mut() {
                *v = f32::from_bits(u.arbitrary::<u32>()?);
            }
            let mut simd_bits = vec![0u16; len];
            D::F32Vec::load(d, &input).store_f16_bits(&mut simd_bits);
            for i in 0..len {
                compare_f16_bits(crate::f16::from_f32(input[i]).to_bits(), simd_bits[i]);
            }
            Ok(())
        });
    }
    test_all_instruction_sets!(test_f16_store_scalar_equivalent);

    fn test_f16_load_scalar_equivalent<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        arbtest::arbtest(|u| {
            let mut input = vec![0u16; len];
            for v in input.iter_mut() {
                *v = u.arbitrary::<u16>()?;
            }
            let mut simd_f32 = vec![0.0f32; len];
            D::F32Vec::load_f16_bits(d, &input).store(&mut simd_f32);
            for i in 0..len {
                let (s, m) = (crate::f16::from_bits(input[i]).to_f32(), simd_f32[i]);
                if s.is_nan() {
                    assert!(m.is_nan(), "expected NaN for 0x{:04X}, got {m}", input[i]);
                    continue;
                }
                let (s_b, m_b) = (s.to_bits(), m.to_bits());
                assert_eq!(s_b, m_b, "load 0x{:04X}: scalar {s}, simd {m}", input[i]);
            }
            Ok(())
        });
    }
    test_all_instruction_sets!(test_f16_load_scalar_equivalent);

    fn test_i32_store_interleaved_2<D: SimdDescriptor>(d: D) {
        let len = D::I32Vec::LEN;
        let mut a_input = vec![0i32; len];
        let mut b_input = vec![0i32; len];
        for i in 0..len {
            a_input[i] = i as i32 * 2;
            b_input[i] = i as i32 * 2 + 1;
        }
        let a = D::I32Vec::load(d, &a_input);
        let b = D::I32Vec::load(d, &b_input);
        let mut dest = vec![0i32; 2 * len];
        D::I32Vec::store_interleaved_2(a, b, &mut dest);
        for i in 0..len {
            assert_eq!(dest[2 * i], i as i32 * 2);
            assert_eq!(dest[2 * i + 1], i as i32 * 2 + 1);
        }
    }
    test_all_instruction_sets!(test_i32_store_interleaved_2);

    fn test_as_i32_scalar_equivalent<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        arbtest::arbtest(|u| {
            let mut input = vec![0.0f32; len];
            for v in input.iter_mut() {
                // Generate floats within i32 range
                *v = u.arbitrary::<i16>()? as f32 + (u.arbitrary::<u8>()? as f32 / 256.0);
            }
            let simd_i32 = D::F32Vec::load(d, &input).as_i32();
            let mut output = vec![0i32; len];
            simd_i32.store(&mut output);
            for i in 0..len {
                let expected = input[i] as i32;
                assert_eq!(
                    output[i], expected,
                    "as_i32 mismatch for input {}: expected {expected}, got {}",
                    input[i], output[i]
                );
            }
            Ok(())
        });
    }
    test_all_instruction_sets!(test_as_i32_scalar_equivalent);

    fn test_i32_load_from_i16<D: SimdDescriptor>(d: D) {
        let len = D::I32Vec::LEN;
        arbtest::arbtest(|u| {
            let mut input = vec![0i16; len];
            for v in input.iter_mut() {
                *v = u.arbitrary::<i16>()?;
            }
            let mut output = vec![0i32; len];
            D::I32Vec::load_from_i16(d, &input).store(&mut output);
            for i in 0..len {
                assert_eq!(
                    output[i], input[i] as i32,
                    "mismatch at index {i} for input {}",
                    input[i]
                );
            }
            Ok(())
        });
    }
    test_all_instruction_sets!(test_i32_load_from_i16);

    fn test_i32_load_from_u16<D: SimdDescriptor>(d: D) {
        let len = D::I32Vec::LEN;
        arbtest::arbtest(|u| {
            let mut input = vec![0u16; len];
            for v in input.iter_mut() {
                *v = u.arbitrary::<u16>()?;
            }
            let mut output = vec![0i32; len];
            D::I32Vec::load_from_u16(d, &input).store(&mut output);
            for i in 0..len {
                assert_eq!(
                    output[i], input[i] as i32,
                    "mismatch at index {i} for input {}",
                    input[i]
                );
            }
            Ok(())
        });
    }
    test_all_instruction_sets!(test_i32_load_from_u16);

    fn test_u64_operations<D: SimdDescriptor>(d: D) {
        let len = D::U64Vec::LEN;
        arbtest::arbtest(|u| {
            let mut a = vec![0u64; len];
            let mut b = vec![0u64; len];
            for i in 0..len {
                a[i] = u.arbitrary::<u64>()?;
                b[i] = u.arbitrary::<u64>()?;
            }
            let va = D::U64Vec::load(d, &a);
            let vb = D::U64Vec::load(d, &b);

            // Addition
            let mut res_add = vec![0u64; len];
            (va + vb).store(&mut res_add);
            for i in 0..len {
                assert_eq!(res_add[i], a[i].wrapping_add(b[i]));
            }

            // Bitwise XOR
            let mut res_xor = vec![0u64; len];
            (va ^ vb).store(&mut res_xor);
            for i in 0..len {
                assert_eq!(res_xor[i], a[i] ^ b[i]);
            }

            // Bitwise AND
            let mut res_and = vec![0u64; len];
            (va & vb).store(&mut res_and);
            for i in 0..len {
                assert_eq!(res_and[i], a[i] & b[i]);
            }

            // Bitwise OR
            let mut res_or = vec![0u64; len];
            (va | vb).store(&mut res_or);
            for i in 0..len {
                assert_eq!(res_or[i], a[i] | b[i]);
            }

            // Shift left by 23
            let mut res_shl = vec![0u64; len];
            crate::shl!(va, 23).store(&mut res_shl);
            for i in 0..len {
                assert_eq!(res_shl[i], a[i] << 23);
            }

            // Shift right by 18
            let mut res_shr18 = vec![0u64; len];
            crate::shr!(va, 18).store(&mut res_shr18);
            for i in 0..len {
                assert_eq!(res_shr18[i], a[i] >> 18);
            }

            // Shift right by 5
            let mut res_shr5 = vec![0u64; len];
            crate::shr!(va, 5).store(&mut res_shr5);
            for i in 0..len {
                assert_eq!(res_shr5[i], a[i] >> 5);
            }

            Ok(())
        });
    }
    test_all_instruction_sets!(test_u64_operations);

    fn test_i16_operations<D: SimdDescriptor>(d: D) {
        let len = D::I16Vec::LEN;
        arbtest::arbtest(|u| {
            let mut a = vec![0i16; len];
            let mut b = vec![0i16; len];
            for i in 0..len {
                a[i] = u.arbitrary::<i16>()?;
                b[i] = u.arbitrary::<i16>()?;
            }
            let va = D::I16Vec::load(d, &a);
            let vb = D::I16Vec::load(d, &b);

            // Addition
            let mut res_add = vec![0i16; len];
            (va + vb).store(&mut res_add);
            for i in 0..len {
                assert_eq!(res_add[i], a[i].wrapping_add(b[i]));
            }

            // Subtraction
            let mut res_sub = vec![0i16; len];
            (va - vb).store(&mut res_sub);
            for i in 0..len {
                assert_eq!(res_sub[i], a[i].wrapping_sub(b[i]));
            }

            // Multiplication
            let mut res_mul = vec![0i16; len];
            (va * vb).store(&mut res_mul);
            for i in 0..len {
                assert_eq!(res_mul[i], a[i].wrapping_mul(b[i]));
            }

            // store_u8
            let mut res_u8 = vec![0u8; len];
            va.store_u8(&mut res_u8);
            for i in 0..len {
                assert_eq!(res_u8[i], (a[i] & 0xFF) as u8);
            }

            // Negation
            let mut res_neg = vec![0i16; len];
            (-va).store(&mut res_neg);
            for i in 0..len {
                assert_eq!(res_neg[i], a[i].wrapping_neg());
            }

            // Bitwise XOR
            let mut res_xor = vec![0i16; len];
            (va ^ vb).store(&mut res_xor);
            for i in 0..len {
                assert_eq!(res_xor[i], a[i] ^ b[i]);
            }

            // Bitwise AND
            let mut res_and = vec![0i16; len];
            (va & vb).store(&mut res_and);
            for i in 0..len {
                assert_eq!(res_and[i], a[i] & b[i]);
            }

            // Bitwise OR
            let mut res_or = vec![0i16; len];
            (va | vb).store(&mut res_or);
            for i in 0..len {
                assert_eq!(res_or[i], a[i] | b[i]);
            }

            // Shift left by 5
            let mut res_shl = vec![0i16; len];
            crate::shl!(va, 5).store(&mut res_shl);
            for i in 0..len {
                assert_eq!(res_shl[i], (a[i] as u16).wrapping_shl(5) as i16);
            }

            // Shift right by 3
            let mut res_shr = vec![0i16; len];
            crate::shr!(va, 3).store(&mut res_shr);
            for i in 0..len {
                assert_eq!(res_shr[i], a[i] >> 3);
            }

            // abs
            let mut res_abs = vec![0i16; len];
            va.abs().store(&mut res_abs);
            for i in 0..len {
                assert_eq!(res_abs[i], a[i].wrapping_abs());
            }

            // mul_wide_take_high
            let mut res_mulhi = vec![0i16; len];
            va.mul_wide_take_high(vb).store(&mut res_mulhi);
            for i in 0..len {
                let expected = ((a[i] as i32 * b[i] as i32) >> 16) as i16;
                assert_eq!(res_mulhi[i], expected);
            }

            // gt
            let mask_gt = va.gt(vb);
            let mut res_gt = vec![0i16; len];
            mask_gt.if_then_else_i16(va, vb).store(&mut res_gt);
            for i in 0..len {
                assert_eq!(res_gt[i], if a[i] > b[i] { a[i] } else { b[i] });
            }

            // lt_zero
            let mask_ltz = va.lt_zero();
            let mut res_ltz = vec![0i16; len];
            mask_ltz.if_then_else_i16(va, vb).store(&mut res_ltz);
            for i in 0..len {
                assert_eq!(res_ltz[i], if a[i] < 0 { a[i] } else { b[i] });
            }

            // eq
            let mask_eq = va.eq(vb);
            let mut res_eq = vec![0i16; len];
            mask_eq
                .if_then_else_i16(
                    <D as SimdDescriptor>::I16Vec::splat(d, 1),
                    <D as SimdDescriptor>::I16Vec::splat(d, 0),
                )
                .store(&mut res_eq);
            for i in 0..len {
                assert_eq!(res_eq[i], if a[i] == b[i] { 1 } else { 0 });
            }

            // eq_zero
            let mask_eqz = va.eq_zero();
            let mut res_eqz = vec![0i16; len];
            mask_eqz.if_then_else_i16(va, vb).store(&mut res_eqz);
            for i in 0..len {
                assert_eq!(res_eqz[i], if a[i] == 0 { a[i] } else { b[i] });
            }

            // maskz_i16
            let mut res_maskz = vec![0i16; len];
            mask_gt.maskz_i16(va).store(&mut res_maskz);
            for i in 0..len {
                assert_eq!(res_maskz[i], if a[i] > b[i] { 0 } else { a[i] });
            }

            // all
            assert_eq!(mask_eqz.all(), a.iter().all(|&x| x == 0));

            // andnot
            let mask_andnot = mask_gt.andnot(mask_ltz);
            let mut res_andnot = vec![0i16; len];
            mask_andnot.if_then_else_i16(va, vb).store(&mut res_andnot);
            for i in 0..len {
                let expected_mask = !(a[i] > b[i]) && (a[i] < 0);
                assert_eq!(res_andnot[i], if expected_mask { a[i] } else { b[i] });
            }

            Ok(())
        });
    }
    test_all_instruction_sets!(test_i16_operations);

    fn test_i16_store_interleaved_2<D: SimdDescriptor>(d: D) {
        let len = D::I16Vec::LEN;
        let a: Vec<i16> = (0..len).map(|i| i as i16).collect();
        let b: Vec<i16> = (0..len).map(|i| (i + 100) as i16).collect();
        let mut output = vec![0i16; 2 * len];

        let a_vec = D::I16Vec::load(d, &a);
        let b_vec = D::I16Vec::load(d, &b);
        D::I16Vec::store_interleaved_2(a_vec, b_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[2 * i], a[i]);
            assert_eq!(output[2 * i + 1], b[i]);
        }
    }
    test_all_instruction_sets!(test_i16_store_interleaved_2);

    fn test_i16_store_interleaved_3<D: SimdDescriptor>(d: D) {
        let len = D::I16Vec::LEN;
        let a: Vec<i16> = (0..len).map(|i| i as i16).collect();
        let b: Vec<i16> = (0..len).map(|i| (i + 100) as i16).collect();
        let c: Vec<i16> = (0..len).map(|i| (i + 200) as i16).collect();
        let mut output = vec![0i16; 3 * len];

        let a_vec = D::I16Vec::load(d, &a);
        let b_vec = D::I16Vec::load(d, &b);
        let c_vec = D::I16Vec::load(d, &c);
        D::I16Vec::store_interleaved_3(a_vec, b_vec, c_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[3 * i], a[i]);
            assert_eq!(output[3 * i + 1], b[i]);
            assert_eq!(output[3 * i + 2], c[i]);
        }
    }
    test_all_instruction_sets!(test_i16_store_interleaved_3);

    fn test_i16_store_interleaved_4<D: SimdDescriptor>(d: D) {
        let len = D::I16Vec::LEN;
        let a: Vec<i16> = (0..len).map(|i| i as i16).collect();
        let b: Vec<i16> = (0..len).map(|i| (i + 100) as i16).collect();
        let c: Vec<i16> = (0..len).map(|i| (i + 200) as i16).collect();
        let e: Vec<i16> = (0..len).map(|i| (i + 300) as i16).collect();
        let mut output = vec![0i16; 4 * len];

        let a_vec = D::I16Vec::load(d, &a);
        let b_vec = D::I16Vec::load(d, &b);
        let c_vec = D::I16Vec::load(d, &c);
        let e_vec = D::I16Vec::load(d, &e);
        D::I16Vec::store_interleaved_4(a_vec, b_vec, c_vec, e_vec, &mut output);

        for i in 0..len {
            assert_eq!(output[4 * i], a[i]);
            assert_eq!(output[4 * i + 1], b[i]);
            assert_eq!(output[4 * i + 2], c[i]);
            assert_eq!(output[4 * i + 3], e[i]);
        }
    }
    test_all_instruction_sets!(test_i16_store_interleaved_4);

    fn test_as_isa_roundtrip<D: SimdDescriptor>(d: D) {
        let len = D::F32Vec::LEN;
        let input: Vec<f32> = (0..len).map(|i| i as f32 + 0.5).collect();
        let v = D::F32Vec::load(d, &input);

        if let Some(sd) = d.as_scalar() {
            let sv = v.as_scalar().unwrap();
            assert_eq!(sv, input[0]);
            let back = D::F32Vec::from_scalar(sv).unwrap();
            let mut out = vec![0.0; len];
            back.store(&mut out);
            assert_eq!(out, input);
            assert!(sd.as_scalar().is_some());
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
        if let Some(sd) = d.as_sse42() {
            let sv = v.as_sse42().unwrap();
            let raw = sv.raw();
            let back = crate::F32VecSse42::from_raw(sd, raw);
            let gen_back = D::F32Vec::from_sse42(back).unwrap();
            let mut out = vec![0.0; len];
            gen_back.store(&mut out);
            assert_eq!(out, input);
            assert!(sv.descriptor().as_sse42().is_some());
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if let Some(ad) = d.as_avx() {
            let av = v.as_avx().unwrap();
            let raw = av.raw();
            let back = crate::F32VecAvx::from_raw(ad, raw);
            let gen_back = D::F32Vec::from_avx(back).unwrap();
            let mut out = vec![0.0; len];
            gen_back.store(&mut out);
            assert_eq!(out, input);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(zd) = d.as_avx512() {
            let zv = v.as_avx512().unwrap();
            let raw = zv.raw();
            let back = crate::F32VecAvx512::from_raw(zd, raw);
            let gen_back = D::F32Vec::from_avx512(back).unwrap();
            let mut out = vec![0.0; len];
            gen_back.store(&mut out);
            assert_eq!(out, input);
        }
    }
    test_all_instruction_sets!(test_as_isa_roundtrip);

    #[deny(unsafe_code)]
    fn test_simd_specialize<D: SimdDescriptor>(d: D) {
        use crate::{F32Vec, F32VecArray, I16VecArray};

        let len = D::F32Vec::LEN;
        let a_data: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
        let b_data: Vec<f32> = (0..len).map(|i| (i * 2) as f32).collect();
        let a = D::F32Vec::load(d, &a_data);
        let b = D::F32Vec::load(d, &b_data);

        // 1. Vector arguments (both D::F32Vec and F32Vec<D>), partial ISA arms, vector return.
        #[deny(unsafe_code)]
        fn scale_and_add_bias<D: SimdDescriptor>(
            d: D,
            a: D::F32Vec,
            b: F32Vec<D>,
            bias: f32,
        ) -> F32Vec<D> {
            crate::simd_specialize!(
                fn(d: D, a: D::F32Vec, b: F32Vec<D>, bias: f32) -> F32Vec<D> {
                    avx => {
                        let raw = std::arch::x86_64::_mm256_add_ps(
                            a.raw(),
                            crate::F32VecAvx::splat(d, bias).raw(),
                        );
                        crate::F32VecAvx::from_raw(d, raw) + b
                    }
                    sse42 => a + crate::F32VecSse42::splat(d, bias) + b,
                    _ => a + D::F32Vec::splat(d, bias) + b,
                }
            )
        }
        let mut out = vec![0.0f32; len];
        let res = scale_and_add_bias(d, a, b, 3.0);
        res.store(&mut out);
        for i in 0..len {
            assert_eq!(out[i], a_data[i] + 3.0 + b_data[i]);
        }

        // 2. Mutable F32VecArray and I16VecArray slices + scalar arg.
        #[deny(unsafe_code)]
        fn set_f32_ends<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], val: f32) {
            crate::simd_specialize!(
                fn(d: D, data: &mut [F32VecArray<D>], val: f32) {
                    avx => {
                        data[0][0] = val;
                        data[1][7] = val * 2.0;
                    }
                    sse42 => {
                        data[0][0] = val;
                        data[1][3] = val * 2.0;
                    }
                    _ => {
                        data[0][0] = val;
                        data[1][D::F32Vec::LEN - 1] = val * 2.0;
                    }
                }
            );
        }

        let mut arr_f32 = vec![0.0f32; len * 2];
        set_f32_ends(d, D::F32Vec::make_array_slice_mut(&mut arr_f32), 7.0);
        assert_eq!(arr_f32[0], 7.0);
        assert_eq!(arr_f32[len * 2 - 1], 14.0);

        #[deny(unsafe_code)]
        fn set_i16_last<D: SimdDescriptor>(d: D, data: &mut [I16VecArray<D>], val: i16) {
            crate::simd_specialize!(
                fn(d: D, data: &mut [I16VecArray<D>], val: i16) {
                    avx => {
                        data[0][15] = val;
                    }
                    _ => {
                        data[0][D::I16Vec::LEN - 1] = val;
                    }
                }
            );
        }

        let i16_len = D::I16Vec::LEN;
        let mut arr_i16 = vec![0i16; i16_len];
        set_i16_last(d, D::I16Vec::make_array_slice_mut(&mut arr_i16), 42);
        assert_eq!(arr_i16[i16_len - 1], 42);

        // 3. Width arms (`v128`, `v256`, `v512`) and mixed ISA + width arms under `#[deny(unsafe_code)]`.
        #[deny(unsafe_code)]
        fn width_specialize<D: SimdDescriptor>(
            d: D,
            a: D::F32Vec,
            data: &mut [F32VecArray<D>],
            bias: f32,
        ) -> F32Vec<D> {
            crate::simd_specialize!(
                fn(d: D, a: D::F32Vec, data: &mut [F32VecArray<D>], bias: f32) -> F32Vec<D> {
                    v512 => {
                        data[0][15] = bias * 4.0;
                        a + D::F32Vec::splat(d, bias)
                    }
                    v256 => {
                        data[0][7] = bias * 3.0;
                        a + D::F32Vec::splat(d, bias)
                    }
                    neon => {
                        data[0][3] = bias * 2.0;
                        a + crate::F32VecNeon::splat(d, bias)
                    }
                    v128 => {
                        data[0][3] = bias * 2.0;
                        a + D::F32Vec::splat(d, bias)
                    }
                    _ => {
                        data[0][D::F32Vec::LEN - 1] = bias;
                        a + D::F32Vec::splat(d, bias)
                    }
                }
            )
        }

        let mut arr_width = vec![0.0f32; len];
        let res_w = width_specialize(d, a, D::F32Vec::make_array_slice_mut(&mut arr_width), 5.0);
        let mut out_w = vec![0.0f32; len];
        res_w.store(&mut out_w);
        for i in 0..len {
            assert_eq!(out_w[i], a_data[i] + 5.0);
        }
        let expected_factor = match len {
            16 => 4.0,
            8 => 3.0,
            4 => 2.0,
            _ => 1.0,
        };
        assert_eq!(arr_width[len - 1], 5.0 * expected_factor);
    }
    test_all_instruction_sets!(test_simd_specialize);
}
