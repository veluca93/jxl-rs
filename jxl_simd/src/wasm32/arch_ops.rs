// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe wrappers around `unsafe` Wasm `simd128` pointer load/store intrinsics.

#![allow(unsafe_code)]
#![allow(dead_code)]
// Clippy treats safe `#[target_feature]` functions as requiring `# Safety` docs.
#![allow(clippy::missing_safety_doc)]

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
pub fn v128_store64_lane0_u8(mem_addr: &mut [u8; 8], a: v128) {
    // SAFETY: `mem_addr` is valid for 8 bytes and `v128_store64_lane` supports unaligned stores.
    unsafe { arch::v128_store64_lane::<0>(a, mem_addr.as_mut_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn v128_store64_lane0_u16(mem_addr: &mut [u16; 4], a: v128) {
    // SAFETY: `mem_addr` is valid for 8 bytes and `v128_store64_lane` supports unaligned stores.
    unsafe { arch::v128_store64_lane::<0>(a, mem_addr.as_mut_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn v128_store64_lane0_i16(mem_addr: &mut [i16; 4], a: v128) {
    // SAFETY: `mem_addr` is valid for 8 bytes and `v128_store64_lane` supports unaligned stores.
    unsafe { arch::v128_store64_lane::<0>(a, mem_addr.as_mut_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn v128_store32_lane0_u8(mem_addr: &mut [u8; 4], a: v128) {
    // SAFETY: `mem_addr` is valid for 4 bytes and `v128_store32_lane` supports unaligned stores.
    unsafe { arch::v128_store32_lane::<0>(a, mem_addr.as_mut_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn i16x8_load_extend_i8x8(mem_addr: &[i8; 8]) -> v128 {
    // SAFETY: `mem_addr` is valid for 8 `i8`s and supports unaligned loads.
    unsafe { arch::i16x8_load_extend_i8x8(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn u16x8_load_extend_u8x8(mem_addr: &[u8; 8]) -> v128 {
    // SAFETY: `mem_addr` is valid for 8 `u8`s and supports unaligned loads.
    unsafe { arch::u16x8_load_extend_u8x8(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn i32x4_load_extend_i16x4(mem_addr: &[i16; 4]) -> v128 {
    // SAFETY: `mem_addr` is valid for 4 `i16`s and supports unaligned loads.
    unsafe { arch::i32x4_load_extend_i16x4(mem_addr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn u32x4_load_extend_u16x4(mem_addr: &[u16; 4]) -> v128 {
    // SAFETY: `mem_addr` is valid for 4 `u16`s and supports unaligned loads.
    unsafe { arch::u32x4_load_extend_u16x4(mem_addr.as_ptr()) }
}
