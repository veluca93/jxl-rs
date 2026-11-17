// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe wrappers around `unsafe` AArch64 NEON pointer load/store intrinsics and inline assembly.

#![allow(unsafe_code)]
#![allow(dead_code)]
// Clippy treats safe `#[target_feature]` functions as requiring `# Safety` docs.
#![allow(clippy::missing_safety_doc)]

use std::arch::aarch64::{self as arch, *};

macro_rules! neon_load {
    ($($fn_name:ident(&[$elem:ty; $len:literal]) -> $ret:ty),* $(,)?) => {
        $(
            #[inline]
            #[target_feature(enable = "neon")]
            pub fn $fn_name(from: &[$elem; $len]) -> $ret {
                // SAFETY: `from` is valid for reads of `$len` elements (`vld*` allows unaligned addresses), and `neon` is enabled.
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
                // SAFETY: `into` is valid for writes of `$len` elements (`vst*` allows unaligned addresses), and `neon` is enabled.
                unsafe { arch::$fn_name(into.as_mut_ptr(), val) }
            }
        )*
    };
}

neon_load!(
    vld1_u8(&[u8; 8]) -> uint8x8_t,
    vld1_u16(&[u16; 4]) -> uint16x4_t,
    vld1_s16(&[i16; 4]) -> int16x4_t,
    vld1_u32(&[u32; 2]) -> uint32x2_t,
    vld1_s32(&[i32; 2]) -> int32x2_t,
    vld1_f32(&[f32; 2]) -> float32x2_t,
    vld1q_f32(&[f32; 4]) -> float32x4_t,
    vld1q_u8(&[u8; 16]) -> uint8x16_t,
    vld1q_s16(&[i16; 8]) -> int16x8_t,
    vld1q_u16(&[u16; 8]) -> uint16x8_t,
    vld1q_s32(&[i32; 4]) -> int32x4_t,
    vld1q_u32(&[u32; 4]) -> uint32x4_t,
    vld1q_u64(&[u64; 2]) -> uint64x2_t,
);

neon_store!(
    vst1_u8(&mut [u8; 8], uint8x8_t),
    vst1_u16(&mut [u16; 4], uint16x4_t),
    vst1_s16(&mut [i16; 4], int16x4_t),
    vst1_u32(&mut [u32; 2], uint32x2_t),
    vst1_s32(&mut [i32; 2], int32x2_t),
    vst1_f32(&mut [f32; 2], float32x2_t),
    vst1q_f32(&mut [f32; 4], float32x4_t),
    vst1q_u8(&mut [u8; 16], uint8x16_t),
    vst1q_s16(&mut [i16; 8], int16x8_t),
    vst1q_u16(&mut [u16; 8], uint16x8_t),
    vst1q_s32(&mut [i32; 4], int32x4_t),
    vst1q_u32(&mut [u32; 4], uint32x4_t),
    vst1q_u64(&mut [u64; 2], uint64x2_t),
    vst2q_f32(&mut [f32; 8], float32x4x2_t),
    vst3q_f32(&mut [f32; 12], float32x4x3_t),
    vst4q_f32(&mut [f32; 16], float32x4x4_t),
    vst2q_u8(&mut [u8; 32], uint8x16x2_t),
    vst3q_u8(&mut [u8; 48], uint8x16x3_t),
    vst4q_u8(&mut [u8; 64], uint8x16x4_t),
    vst2q_s16(&mut [i16; 16], int16x8x2_t),
    vst3q_s16(&mut [i16; 24], int16x8x3_t),
    vst4q_s16(&mut [i16; 32], int16x8x4_t),
    vst2q_u16(&mut [u16; 16], uint16x8x2_t),
    vst3q_u16(&mut [u16; 24], uint16x8x3_t),
    vst4q_u16(&mut [u16; 32], uint16x8x4_t),
    vst2q_s32(&mut [i32; 8], int32x4x2_t),
    vst3q_s32(&mut [i32; 12], int32x4x3_t),
    vst4q_s32(&mut [i32; 16], int32x4x4_t),
    vst2q_u32(&mut [u32; 8], uint32x4x2_t),
    vst3q_u32(&mut [u32; 12], uint32x4x3_t),
    vst4q_u32(&mut [u32; 16], uint32x4x4_t),
    vst2q_u64(&mut [u64; 4], uint64x2x2_t),
    vst3q_u64(&mut [u64; 6], uint64x2x3_t),
    vst4q_u64(&mut [u64; 8], uint64x2x4_t),
);

/// Converts 4 IEEE 754 `f16` bit patterns into `f32` lanes using baseline AArch64 `fcvtl`.
#[inline]
#[target_feature(enable = "neon")]
pub fn fcvtl_f32_f16(f16_bits: uint16x4_t) -> float32x4_t {
    let mut result: float32x4_t;
    // SAFETY: `fcvtl` is part of baseline AArch64 NEON (ARMv8.0-A).
    unsafe {
        std::arch::asm!(
            "fcvtl {out:v}.4s, {inp:v}.4h",
            inp = in(vreg) f16_bits,
            out = lateout(vreg) result,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Converts 4 `f32` lanes into IEEE 754 `f16` bit patterns using baseline AArch64 `fcvtn`.
#[inline]
#[target_feature(enable = "neon")]
pub fn fcvtn_f16_f32(f32_vals: float32x4_t) -> uint16x4_t {
    let mut f16_bits: uint16x4_t;
    // SAFETY: `fcvtn` is part of baseline AArch64 NEON (ARMv8.0-A).
    unsafe {
        std::arch::asm!(
            "fcvtn {out:v}.4h, {inp:v}.4s",
            inp = in(vreg) f32_vals,
            out = lateout(vreg) f16_bits,
            options(pure, nomem, nostack),
        );
    }
    f16_bits
}
