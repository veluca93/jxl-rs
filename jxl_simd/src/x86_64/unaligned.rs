// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe wrappers around unaligned x86_64 SIMD load/store intrinsics,
//! modeled after the `safe_unaligned_simd` crate.

#![allow(unsafe_code)]
#![allow(dead_code)]

use std::arch::x86_64::{__m128, __m128i, __m256, __m256i, __m512, __m512i};

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for types that represent 4 unaligned bytes (`[T; N]` of size 4).
pub trait Is4BytesUnaligned: sealed::Sealed {}

/// Marker trait for types that represent 8 unaligned bytes (`[T; N]` of size 8).
pub trait Is8BytesUnaligned: sealed::Sealed {}

/// Marker trait for types that represent 16 unaligned bytes (`[T; N]` of size 16).
pub trait Is16BytesUnaligned: sealed::Sealed {}

/// Marker trait for types that represent 32 unaligned bytes (`[T; N]` of size 32).
pub trait Is32BytesUnaligned: sealed::Sealed {}

/// Marker trait for types that represent 64 unaligned bytes (`[T; N]` of size 64).
pub trait Is64BytesUnaligned: sealed::Sealed {}

macro_rules! impl_unaligned_size {
    ($trait:ident, $bytes:expr, [$($ty:ty : $len:expr),* $(,)?]) => {
        $(
            const _: () = assert!(core::mem::size_of::<[$ty; $len]>() == $bytes);
            impl sealed::Sealed for [$ty; $len] {}
            impl $trait for [$ty; $len] {}
        )*
    };
}

impl_unaligned_size!(
    Is4BytesUnaligned,
    4,
    [u8: 4, i8: 4, u16: 2, i16: 2, u32: 1, i32: 1, f32: 1]
);

impl_unaligned_size!(
    Is8BytesUnaligned,
    8,
    [u8: 8, i8: 8, u16: 4, i16: 4, u32: 2, i32: 2, u64: 1, i64: 1, f32: 2, f64: 1]
);

impl_unaligned_size!(
    Is16BytesUnaligned,
    16,
    [u8: 16, i8: 16, u16: 8, i16: 8, u32: 4, i32: 4, u64: 2, i64: 2, f32: 4, f64: 2]
);

impl_unaligned_size!(
    Is32BytesUnaligned,
    32,
    [u8: 32, i8: 32, u16: 16, i16: 16, u32: 8, i32: 8, u64: 4, i64: 4, f32: 8, f64: 4]
);

impl_unaligned_size!(
    Is64BytesUnaligned,
    64,
    [u8: 64, i8: 64, u16: 32, i16: 32, u32: 16, i32: 16, u64: 8, i64: 8, f32: 16, f64: 8]
);

#[target_feature(enable = "sse")]
#[inline(always)]
pub fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128 {
    // SAFETY: `mem_addr` is a valid reference to 4 `f32`s and `_mm_loadu_ps` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm_loadu_ps(mem_addr.as_ptr()) }
}

#[target_feature(enable = "sse")]
#[inline(always)]
pub fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) {
    // SAFETY: `mem_addr` is a valid mutable reference to 4 `f32`s and `_mm_storeu_ps` supports unaligned stores.
    unsafe { std::arch::x86_64::_mm_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_loadu_si128<T: Is16BytesUnaligned>(mem_addr: &T) -> __m128i {
    // SAFETY: `T` is 16 bytes and `_mm_loadu_si128` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm_loadu_si128(core::ptr::from_ref(mem_addr).cast::<__m128i>()) }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_storeu_si128<T: Is16BytesUnaligned>(mem_addr: &mut T, a: __m128i) {
    // SAFETY: `T` is 16 bytes and `_mm_storeu_si128` supports unaligned stores.
    unsafe {
        std::arch::x86_64::_mm_storeu_si128(core::ptr::from_mut(mem_addr).cast::<__m128i>(), a)
    }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_loadl_epi64<T: Is8BytesUnaligned>(mem_addr: &T) -> __m128i {
    // SAFETY: `T` is 8 bytes and `_mm_loadl_epi64` loads 64 bits without alignment requirements.
    unsafe { std::arch::x86_64::_mm_loadl_epi64(core::ptr::from_ref(mem_addr).cast::<__m128i>()) }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_storel_epi64<T: Is8BytesUnaligned>(mem_addr: &mut T, a: __m128i) {
    // SAFETY: `T` is 8 bytes and `_mm_storel_epi64` stores 64 bits without alignment requirements.
    unsafe {
        std::arch::x86_64::_mm_storel_epi64(core::ptr::from_mut(mem_addr).cast::<__m128i>(), a)
    }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_storeu_si32<T: Is4BytesUnaligned>(mem_addr: &mut T, a: __m128i) {
    // SAFETY: `T` is 4 bytes and `_mm_storeu_si32` stores 32 bits without alignment requirements.
    unsafe { std::arch::x86_64::_mm_storeu_si32(core::ptr::from_mut(mem_addr).cast::<u8>(), a) }
}

#[target_feature(enable = "sse2")]
#[inline(always)]
pub fn _mm_storeu_si64<T: Is8BytesUnaligned>(mem_addr: &mut T, a: __m128i) {
    // SAFETY: `T` is 8 bytes and `_mm_storeu_si64` stores 64 bits without alignment requirements.
    unsafe { std::arch::x86_64::_mm_storeu_si64(core::ptr::from_mut(mem_addr).cast::<u8>(), a) }
}

#[target_feature(enable = "avx")]
#[inline(always)]
pub fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256 {
    // SAFETY: `mem_addr` is a valid reference to 8 `f32`s and `_mm256_loadu_ps` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm256_loadu_ps(mem_addr.as_ptr()) }
}

#[target_feature(enable = "avx")]
#[inline(always)]
pub fn _mm256_storeu_ps(mem_addr: &mut [f32; 8], a: __m256) {
    // SAFETY: `mem_addr` is a valid mutable reference to 8 `f32`s and `_mm256_storeu_ps` supports unaligned stores.
    unsafe { std::arch::x86_64::_mm256_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[target_feature(enable = "avx")]
#[inline(always)]
pub fn _mm256_loadu_si256<T: Is32BytesUnaligned>(mem_addr: &T) -> __m256i {
    // SAFETY: `T` is 32 bytes and `_mm256_loadu_si256` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm256_loadu_si256(core::ptr::from_ref(mem_addr).cast::<__m256i>()) }
}

#[target_feature(enable = "avx")]
#[inline(always)]
pub fn _mm256_storeu_si256<T: Is32BytesUnaligned>(mem_addr: &mut T, a: __m256i) {
    // SAFETY: `T` is 32 bytes and `_mm256_storeu_si256` supports unaligned stores.
    unsafe {
        std::arch::x86_64::_mm256_storeu_si256(core::ptr::from_mut(mem_addr).cast::<__m256i>(), a)
    }
}

#[target_feature(enable = "avx512f")]
#[inline(always)]
pub fn _mm512_loadu_ps(mem_addr: &[f32; 16]) -> __m512 {
    // SAFETY: `mem_addr` is a valid reference to 16 `f32`s and `_mm512_loadu_ps` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm512_loadu_ps(mem_addr.as_ptr()) }
}

#[target_feature(enable = "avx512f")]
#[inline(always)]
pub fn _mm512_storeu_ps(mem_addr: &mut [f32; 16], a: __m512) {
    // SAFETY: `mem_addr` is a valid mutable reference to 16 `f32`s and `_mm512_storeu_ps` supports unaligned stores.
    unsafe { std::arch::x86_64::_mm512_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[target_feature(enable = "avx512f")]
#[inline(always)]
pub fn _mm512_loadu_si512<T: Is64BytesUnaligned>(mem_addr: &T) -> __m512i {
    // SAFETY: `T` is 64 bytes and `_mm512_loadu_si512` supports unaligned loads.
    unsafe { std::arch::x86_64::_mm512_loadu_si512(core::ptr::from_ref(mem_addr).cast::<__m512i>()) }
}

#[target_feature(enable = "avx512f")]
#[inline(always)]
pub fn _mm512_storeu_si512<T: Is64BytesUnaligned>(mem_addr: &mut T, a: __m512i) {
    // SAFETY: `T` is 64 bytes and `_mm512_storeu_si512` supports unaligned stores.
    unsafe {
        std::arch::x86_64::_mm512_storeu_si512(core::ptr::from_mut(mem_addr).cast::<__m512i>(), a)
    }
}
