// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe wrappers around `unsafe` x86_64 SIMD pointer load/store intrinsics.

#![allow(unsafe_code)]
#![allow(dead_code)]
// Clippy treats safe `#[target_feature]` functions as requiring `# Safety` docs.
#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::{
    self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
};

#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128 {
    // SAFETY: `mem_addr` is valid for 4 `f32`s and `_mm_loadu_ps` supports unaligned loads.
    unsafe { arch::_mm_loadu_ps(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) {
    // SAFETY: `mem_addr` is valid for 4 `f32`s and `_mm_storeu_ps` supports unaligned stores.
    unsafe { arch::_mm_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_pd(mem_addr: &[f64; 2]) -> __m128d {
    // SAFETY: `mem_addr` is valid for 2 `f64`s and `_mm_loadu_pd` supports unaligned loads.
    unsafe { arch::_mm_loadu_pd(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_pd(mem_addr: &mut [f64; 2], a: __m128d) {
    // SAFETY: `mem_addr` is valid for 2 `f64`s and `_mm_storeu_pd` supports unaligned stores.
    unsafe { arch::_mm_storeu_pd(mem_addr.as_mut_ptr(), a) }
}

#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256 {
    // SAFETY: `mem_addr` is valid for 8 `f32`s and `_mm256_loadu_ps` supports unaligned loads.
    unsafe { arch::_mm256_loadu_ps(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_ps(mem_addr: &mut [f32; 8], a: __m256) {
    // SAFETY: `mem_addr` is valid for 8 `f32`s and `_mm256_storeu_ps` supports unaligned stores.
    unsafe { arch::_mm256_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_pd(mem_addr: &[f64; 4]) -> __m256d {
    // SAFETY: `mem_addr` is valid for 4 `f64`s and `_mm256_loadu_pd` supports unaligned loads.
    unsafe { arch::_mm256_loadu_pd(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_pd(mem_addr: &mut [f64; 4], a: __m256d) {
    // SAFETY: `mem_addr` is valid for 4 `f64`s and `_mm256_storeu_pd` supports unaligned stores.
    unsafe { arch::_mm256_storeu_pd(mem_addr.as_mut_ptr(), a) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_ps(mem_addr: &[f32; 16]) -> __m512 {
    // SAFETY: `mem_addr` is valid for 16 `f32`s and `_mm512_loadu_ps` supports unaligned loads.
    unsafe { arch::_mm512_loadu_ps(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_ps(mem_addr: &mut [f32; 16], a: __m512) {
    // SAFETY: `mem_addr` is valid for 16 `f32`s and `_mm512_storeu_ps` supports unaligned stores.
    unsafe { arch::_mm512_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_pd(mem_addr: &[f64; 8]) -> __m512d {
    // SAFETY: `mem_addr` is valid for 8 `f64`s and `_mm512_loadu_pd` supports unaligned loads.
    unsafe { arch::_mm512_loadu_pd(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_pd(mem_addr: &mut [f64; 8], a: __m512d) {
    // SAFETY: `mem_addr` is valid for 8 `f64`s and `_mm512_storeu_pd` supports unaligned stores.
    unsafe { arch::_mm512_storeu_pd(mem_addr.as_mut_ptr(), a) }
}

macro_rules! x86_si_load_store {
    (
        feature = $feat:literal,
        vec = $vec:ty,
        bytes = $bytes:expr,
        load_intrinsic = $load_intrinsic:ident,
        store_intrinsic = $store_intrinsic:ident,
        $(($load_fn:ident, $store_fn:ident, [$elem:ty; $len:expr])),* $(,)?
    ) => {
        $(
            #[inline]
            #[target_feature(enable = $feat)]
            pub fn $load_fn(mem_addr: &[$elem; $len]) -> $vec {
                const { assert!(std::mem::size_of::<[$elem; $len]>() == $bytes); }
                // SAFETY: `mem_addr` is a fixed-size primitive array of `$bytes` bytes and
                // the target intrinsic supports unaligned loads.
                unsafe { arch::$load_intrinsic(mem_addr.as_ptr().cast()) }
            }

            #[inline]
            #[target_feature(enable = $feat)]
            pub fn $store_fn(mem_addr: &mut [$elem; $len], a: $vec) {
                const { assert!(std::mem::size_of::<[$elem; $len]>() == $bytes); }
                // SAFETY: `mem_addr` is a fixed-size primitive array of `$bytes` bytes and
                // the target intrinsic supports unaligned stores.
                unsafe { arch::$store_intrinsic(mem_addr.as_mut_ptr().cast(), a) }
            }
        )*
    };
}

x86_si_load_store!(
    feature = "sse2",
    vec = __m128i,
    bytes = 16,
    load_intrinsic = _mm_loadu_si128,
    store_intrinsic = _mm_storeu_si128,
    (_mm_loadu_si128_u8, _mm_storeu_si128_u8, [u8; 16]),
    (_mm_loadu_si128_i8, _mm_storeu_si128_i8, [i8; 16]),
    (_mm_loadu_si128_u16, _mm_storeu_si128_u16, [u16; 8]),
    (_mm_loadu_si128_i16, _mm_storeu_si128_i16, [i16; 8]),
    (_mm_loadu_si128_u32, _mm_storeu_si128_u32, [u32; 4]),
    (_mm_loadu_si128_i32, _mm_storeu_si128_i32, [i32; 4]),
    (_mm_loadu_si128_u64, _mm_storeu_si128_u64, [u64; 2]),
    (_mm_loadu_si128_i64, _mm_storeu_si128_i64, [i64; 2]),
);

x86_si_load_store!(
    feature = "sse2",
    vec = __m128i,
    bytes = 8,
    load_intrinsic = _mm_loadu_si64,
    store_intrinsic = _mm_storeu_si64,
    (_mm_loadu_si64_u8, _mm_storeu_si64_u8, [u8; 8]),
    (_mm_loadu_si64_i8, _mm_storeu_si64_i8, [i8; 8]),
    (_mm_loadu_si64_u16, _mm_storeu_si64_u16, [u16; 4]),
    (_mm_loadu_si64_i16, _mm_storeu_si64_i16, [i16; 4]),
    (_mm_loadu_si64_u32, _mm_storeu_si64_u32, [u32; 2]),
    (_mm_loadu_si64_i32, _mm_storeu_si64_i32, [i32; 2]),
    (_mm_loadu_si64_u64, _mm_storeu_si64_u64, [u64; 1]),
    (_mm_loadu_si64_i64, _mm_storeu_si64_i64, [i64; 1]),
);

x86_si_load_store!(
    feature = "sse2",
    vec = __m128i,
    bytes = 4,
    load_intrinsic = _mm_loadu_si32,
    store_intrinsic = _mm_storeu_si32,
    (_mm_loadu_si32_u8, _mm_storeu_si32_u8, [u8; 4]),
    (_mm_loadu_si32_i8, _mm_storeu_si32_i8, [i8; 4]),
    (_mm_loadu_si32_u16, _mm_storeu_si32_u16, [u16; 2]),
    (_mm_loadu_si32_i16, _mm_storeu_si32_i16, [i16; 2]),
    (_mm_loadu_si32_u32, _mm_storeu_si32_u32, [u32; 1]),
    (_mm_loadu_si32_i32, _mm_storeu_si32_i32, [i32; 1]),
);

x86_si_load_store!(
    feature = "sse2",
    vec = __m128i,
    bytes = 2,
    load_intrinsic = _mm_loadu_si16,
    store_intrinsic = _mm_storeu_si16,
    (_mm_loadu_si16_u8, _mm_storeu_si16_u8, [u8; 2]),
    (_mm_loadu_si16_i8, _mm_storeu_si16_i8, [i8; 2]),
    (_mm_loadu_si16_u16, _mm_storeu_si16_u16, [u16; 1]),
    (_mm_loadu_si16_i16, _mm_storeu_si16_i16, [i16; 1]),
);

x86_si_load_store!(
    feature = "avx",
    vec = __m256i,
    bytes = 32,
    load_intrinsic = _mm256_loadu_si256,
    store_intrinsic = _mm256_storeu_si256,
    (_mm256_loadu_si256_u8, _mm256_storeu_si256_u8, [u8; 32]),
    (_mm256_loadu_si256_i8, _mm256_storeu_si256_i8, [i8; 32]),
    (_mm256_loadu_si256_u16, _mm256_storeu_si256_u16, [u16; 16]),
    (_mm256_loadu_si256_i16, _mm256_storeu_si256_i16, [i16; 16]),
    (_mm256_loadu_si256_u32, _mm256_storeu_si256_u32, [u32; 8]),
    (_mm256_loadu_si256_i32, _mm256_storeu_si256_i32, [i32; 8]),
    (_mm256_loadu_si256_u64, _mm256_storeu_si256_u64, [u64; 4]),
    (_mm256_loadu_si256_i64, _mm256_storeu_si256_i64, [i64; 4]),
);

x86_si_load_store!(
    feature = "avx512f",
    vec = __m512i,
    bytes = 64,
    load_intrinsic = _mm512_loadu_si512,
    store_intrinsic = _mm512_storeu_si512,
    (_mm512_loadu_si512_u8, _mm512_storeu_si512_u8, [u8; 64]),
    (_mm512_loadu_si512_i8, _mm512_storeu_si512_i8, [i8; 64]),
    (_mm512_loadu_si512_u16, _mm512_storeu_si512_u16, [u16; 32]),
    (_mm512_loadu_si512_i16, _mm512_storeu_si512_i16, [i16; 32]),
    (_mm512_loadu_si512_u32, _mm512_storeu_si512_u32, [u32; 16]),
    (_mm512_loadu_si512_i32, _mm512_storeu_si512_i32, [i32; 16]),
    (_mm512_loadu_si512_u64, _mm512_storeu_si512_u64, [u64; 8]),
    (_mm512_loadu_si512_i64, _mm512_storeu_si512_i64, [i64; 8]),
);

#[inline]
#[target_feature(enable = "sse")]
pub fn prefetch_t0<T: ?Sized>(r: &T) {
    #[cfg(miri)]
    let _ = r;
    #[cfg(not(miri))]
    {
        let base = (r as *const T).cast::<i8>();
        for i in (0..core::mem::size_of_val(r)).step_by(64) {
            arch::_mm_prefetch::<{ arch::_MM_HINT_T0 }>(base.wrapping_add(i));
        }
    }
}

#[inline]
#[target_feature(enable = "sse")]
pub fn prefetch_et0<T: ?Sized>(r: &T) {
    #[cfg(miri)]
    let _ = r;
    #[cfg(not(miri))]
    {
        let base = (r as *const T).cast::<i8>();
        for i in (0..core::mem::size_of_val(r)).step_by(64) {
            arch::_mm_prefetch::<{ arch::_MM_HINT_ET0 }>(base.wrapping_add(i));
        }
    }
}
