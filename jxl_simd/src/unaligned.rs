// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe wrappers for unaligned SIMD load and store intrinsics.

#![allow(unsafe_code)]
#![allow(dead_code)]
// Clippy treats safe `#[target_feature]` functions as requiring `# Safety` docs.
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::x86_64::{self as arch, __m128, __m128i, __m256, __m256i, __m512, __m512i};

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
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use std::arch::aarch64::{self as arch, *};

    macro_rules! neon_load {
        ($($fn_name:ident(&[$elem:ty; $len:literal]) -> $ret:ty),* $(,)?) => {
            $(
                #[inline]
                #[target_feature(enable = "neon")]
                pub fn $fn_name(from: &[$elem; $len]) -> $ret {
                    // SAFETY: `from` is valid and aligned for `$elem`, and `neon` is enabled.
                    unsafe { arch::$fn_name(from.as_ptr()) }
                }
            )*
        };
    }

    macro_rules! neon_store {
        ($($fn_name:ident(&mut [$elem:ty; $len:literal], $val_ty:ty)),* $(,)?) => {
            $(
                #[inline]
                #[target_feature(enable = "neon")]
                pub fn $fn_name(into: &mut [$elem; $len], val: $val_ty) {
                    // SAFETY: `into` is valid and aligned for `$elem`, and `neon` is enabled.
                    unsafe { arch::$fn_name(into.as_mut_ptr(), val) }
                }
            )*
        };
    }

    neon_load!(
        vld1_u8(&[u8; 8]) -> uint8x8_t,
        vld1_u16(&[u16; 4]) -> uint16x4_t,
        vld1_s16(&[i16; 4]) -> int16x4_t,
        vld1q_f32(&[f32; 4]) -> float32x4_t,
        vld1q_s32(&[i32; 4]) -> int32x4_t,
        vld1q_u32(&[u32; 4]) -> uint32x4_t,
        vld1q_s16(&[i16; 8]) -> int16x8_t,
        vld1q_u16(&[u16; 8]) -> uint16x8_t,
        vld1q_u8(&[u8; 16]) -> uint8x16_t,
        vld1q_u64(&[u64; 2]) -> uint64x2_t,
        vld2q_f32(&[f32; 8]) -> float32x4x2_t,
        vld3q_f32(&[f32; 12]) -> float32x4x3_t,
        vld4q_f32(&[f32; 16]) -> float32x4x4_t,
    );

    neon_store!(
        vst1_u8(&mut [u8; 8], uint8x8_t),
        vst1_u16(&mut [u16; 4], uint16x4_t),
        vst1_s16(&mut [i16; 4], int16x4_t),
        vst1q_f32(&mut [f32; 4], float32x4_t),
        vst1q_s32(&mut [i32; 4], int32x4_t),
        vst1q_u32(&mut [u32; 4], uint32x4_t),
        vst1q_s16(&mut [i16; 8], int16x8_t),
        vst1q_u16(&mut [u16; 8], uint16x8_t),
        vst1q_u8(&mut [u8; 16], uint8x16_t),
        vst1q_u64(&mut [u64; 2], uint64x2_t),
        vst2q_f32(&mut [f32; 8], float32x4x2_t),
        vst3q_f32(&mut [f32; 12], float32x4x3_t),
        vst4q_f32(&mut [f32; 16], float32x4x4_t),
        vst2q_u8(&mut [u8; 32], uint8x16x2_t),
        vst3q_u8(&mut [u8; 48], uint8x16x3_t),
        vst4q_u8(&mut [u8; 64], uint8x16x4_t),
        vst2q_u16(&mut [u16; 16], uint16x8x2_t),
        vst3q_u16(&mut [u16; 24], uint16x8x3_t),
        vst4q_u16(&mut [u16; 32], uint16x8x4_t),
    );
}

#[cfg(target_arch = "wasm32")]
pub mod wasm32 {
    use std::arch::wasm32::{self as arch, v128};

    macro_rules! wasm_load {
        (
            bytes = $bytes:expr,
            intrinsic = $intrinsic:ident,
            $(($fn_name:ident, [$elem:ty; $len:expr])),* $(,)?
        ) => {
            $(
                #[inline]
                #[target_feature(enable = "simd128")]
                pub fn $fn_name(mem_addr: &[$elem; $len]) -> v128 {
                    const { assert!(std::mem::size_of::<[$elem; $len]>() == $bytes); }
                    // SAFETY: `mem_addr` is a fixed-size primitive array of `$bytes` bytes and
                    // the wasm32 simd128 load intrinsic supports unaligned loads.
                    unsafe { arch::$intrinsic(mem_addr.as_ptr().cast()) }
                }
            )*
        };
    }

    macro_rules! wasm_store {
        (
            bytes = $bytes:expr,
            intrinsic = $intrinsic:ident,
            $(($fn_name:ident, [$elem:ty; $len:expr])),* $(,)?
        ) => {
            $(
                #[inline]
                #[target_feature(enable = "simd128")]
                pub fn $fn_name(mem_addr: &mut [$elem; $len], a: v128) {
                    const { assert!(std::mem::size_of::<[$elem; $len]>() == $bytes); }
                    // SAFETY: `mem_addr` is a fixed-size primitive array of `$bytes` bytes and
                    // the wasm32 simd128 store intrinsic supports unaligned stores.
                    unsafe { arch::$intrinsic(mem_addr.as_mut_ptr().cast(), a) }
                }
            )*
        };
    }

    wasm_load!(
        bytes = 16,
        intrinsic = v128_load,
        (v128_load_f32, [f32; 4]),
        (v128_load_f64, [f64; 2]),
        (v128_load_u8, [u8; 16]),
        (v128_load_i8, [i8; 16]),
        (v128_load_u16, [u16; 8]),
        (v128_load_i16, [i16; 8]),
        (v128_load_u32, [u32; 4]),
        (v128_load_i32, [i32; 4]),
        (v128_load_u64, [u64; 2]),
        (v128_load_i64, [i64; 2]),
    );

    wasm_store!(
        bytes = 16,
        intrinsic = v128_store,
        (v128_store_f32, [f32; 4]),
        (v128_store_f64, [f64; 2]),
        (v128_store_u8, [u8; 16]),
        (v128_store_i8, [i8; 16]),
        (v128_store_u16, [u16; 8]),
        (v128_store_i16, [i16; 8]),
        (v128_store_u32, [u32; 4]),
        (v128_store_i32, [i32; 4]),
        (v128_store_u64, [u64; 2]),
        (v128_store_i64, [i64; 2]),
    );

    wasm_load!(
        bytes = 8,
        intrinsic = v128_load64_splat,
        (v128_load64_splat_u8, [u8; 8]),
        (v128_load64_splat_i8, [i8; 8]),
        (v128_load64_splat_u16, [u16; 4]),
        (v128_load64_splat_i16, [i16; 4]),
        (v128_load64_splat_u32, [u32; 2]),
        (v128_load64_splat_i32, [i32; 2]),
        (v128_load64_splat_u64, [u64; 1]),
        (v128_load64_splat_i64, [i64; 1]),
        (v128_load64_splat_f32, [f32; 2]),
        (v128_load64_splat_f64, [f64; 1]),
    );

    wasm_load!(
        bytes = 8,
        intrinsic = v128_load64_zero,
        (v128_load64_zero_u8, [u8; 8]),
        (v128_load64_zero_i8, [i8; 8]),
        (v128_load64_zero_u16, [u16; 4]),
        (v128_load64_zero_i16, [i16; 4]),
        (v128_load64_zero_u32, [u32; 2]),
        (v128_load64_zero_i32, [i32; 2]),
        (v128_load64_zero_u64, [u64; 1]),
        (v128_load64_zero_i64, [i64; 1]),
        (v128_load64_zero_f32, [f32; 2]),
        (v128_load64_zero_f64, [f64; 1]),
    );

    wasm_load!(
        bytes = 4,
        intrinsic = v128_load32_zero,
        (v128_load32_zero_u8, [u8; 4]),
        (v128_load32_zero_i8, [i8; 4]),
        (v128_load32_zero_u16, [u16; 2]),
        (v128_load32_zero_i16, [i16; 2]),
        (v128_load32_zero_u32, [u32; 1]),
        (v128_load32_zero_i32, [i32; 1]),
        (v128_load32_zero_f32, [f32; 1]),
    );

    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn i32x4_load_extend_i16x4(mem_addr: &[i16; 4]) -> v128 {
        // SAFETY: `mem_addr` is valid for 4 `i16`s and `i32x4_load_extend_i16x4` supports unaligned loads.
        unsafe { arch::i32x4_load_extend_i16x4(mem_addr.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn u32x4_load_extend_u16x4(mem_addr: &[u16; 4]) -> v128 {
        // SAFETY: `mem_addr` is valid for 4 `u16`s and `u32x4_load_extend_u16x4` supports unaligned loads.
        unsafe { arch::u32x4_load_extend_u16x4(mem_addr.as_ptr()) }
    }
}
