// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::too_many_arguments)]

use jxl_simd::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec,
    U16SimdVec, simd_function,
};

#[inline(always)]
fn convert_f32_vec<D: SimdDescriptor>(
    d: D,
    val: D::F32Vec,
    scale: D::F32Vec,
    zero: D::F32Vec,
    dither_row: &[f32; 64],
    dither_x: usize,
) -> D::F32Vec {
    let dither = D::F32Vec::load(d, &dither_row[dither_x..]);
    let scaled = val * scale;
    let dithered = scaled + dither;
    dithered.max(zero).min(scale)
}

#[inline(always)]
fn convert_i16_vec<D: SimdDescriptor>(
    _d: D,
    val: D::I16Vec,
    scale: D::I16Vec,
    zero: D::I16Vec,
    max_vec: D::I16Vec,
) -> D::I16Vec {
    let scaled = val * scale;
    let zeroclip = scaled.lt_zero().if_then_else_i16(zero, scaled);
    scaled.gt(max_vec).if_then_else_i16(max_vec, zeroclip)
}

// --- Channel Readers ---

pub(crate) trait ChannelReaderU8<D: SimdDescriptor> {
    fn check_len(&self, len: usize);
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec;
    fn read_u8_vec_tail(&self, d: D, base_x: usize, rem: usize, stack: &mut [u8; 64]) -> D::U8Vec;
}

pub(crate) trait ChannelReaderU16<D: SimdDescriptor> {
    fn check_len(&self, len: usize);
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec;
    fn read_u16_vec_tail(
        &self,
        d: D,
        base_x: usize,
        rem: usize,
        stack: &mut [u16; 64],
    ) -> D::U16Vec;
}

pub(crate) trait ChannelReaderF32<D: SimdDescriptor> {
    fn check_len(&self, len: usize);
    fn read_f32_vec(&self, d: D, base_x: usize, stack: &mut [f32; 64]) -> D::F32Vec;
    fn read_f32_vec_tail(
        &self,
        d: D,
        base_x: usize,
        rem: usize,
        stack: &mut [f32; 64],
    ) -> D::F32Vec;
}

struct F32ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    scale_vec: D::F32Vec,
    zero: D::F32Vec,
    dither_row: &'static [f32; 64],
    x0: usize,
    dither_channel: usize,
}

impl<'a, D: SimdDescriptor> F32ToU8Reader<'a, D> {
    #[inline(always)]
    fn new(
        d: D,
        slice: &'a [f32],
        max: f32,
        position: (usize, usize),
        dither_channel: usize,
    ) -> Self {
        let (x0, y0) = position;
        let dither_y = (y0 + dither_channel * 13) % 32;
        Self {
            slice,
            scale_vec: D::F32Vec::splat(d, max),
            zero: D::F32Vec::splat(d, 0.0),
            dither_row: &crate::util::DITHER_TABLE[dither_y],
            x0,
            dither_channel,
        }
    }

    #[inline(always)]
    fn convert_chunk(
        &self,
        d: D,
        in_chunk: &[f32],
        base_x: usize,
        stack: &mut [u8; 64],
    ) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u8_len.checked_div(f32_len).unwrap_or(1);
        assert!(in_chunk.len() >= u8_len);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let val = D::F32Vec::load(d, &in_chunk[k * f32_len..]);
            let dx = (self.x0 + x + self.dither_channel * 23) % 32;
            let clamped = convert_f32_vec(d, val, self.scale_vec, self.zero, self.dither_row, dx);
            clamped.round_store_u8(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for F32ToU8Reader<'a, D> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        self.convert_chunk(
            d,
            &self.slice[base_x..base_x + D::U8Vec::LEN],
            base_x,
            stack,
        )
    }

    #[inline(always)]
    fn read_u8_vec_tail(&self, d: D, base_x: usize, rem: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let mut in_buf = [0.0f32; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        self.convert_chunk(d, &in_buf, base_x, stack)
    }
}

struct I16ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [i16],
    scale: D::I16Vec,
    zero: D::I16Vec,
    max_vec: D::I16Vec,
}

impl<'a, D: SimdDescriptor> I16ToU8Reader<'a, D> {
    #[inline(always)]
    fn new(d: D, slice: &'a [i16], mult: i16, max: i16) -> Self {
        Self {
            slice,
            scale: D::I16Vec::splat(d, mult),
            zero: D::I16Vec::splat(d, 0),
            max_vec: D::I16Vec::splat(d, max),
        }
    }

    #[inline(always)]
    fn convert_chunk(&self, d: D, in_chunk: &[i16], stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let i16_len = D::I16Vec::LEN;
        let ratio = u8_len.checked_div(i16_len).unwrap_or(1);
        assert!(in_chunk.len() >= u8_len);
        for k in 0..ratio {
            let val = D::I16Vec::load(d, &in_chunk[k * i16_len..]);
            let clip = convert_i16_vec(d, val, self.scale, self.zero, self.max_vec);
            clip.store_u8(&mut stack[k * i16_len..(k + 1) * i16_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for I16ToU8Reader<'a, D> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        self.convert_chunk(d, &self.slice[base_x..base_x + D::U8Vec::LEN], stack)
    }

    #[inline(always)]
    fn read_u8_vec_tail(&self, d: D, base_x: usize, rem: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let mut in_buf = [0i16; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        self.convert_chunk(d, &in_buf, stack)
    }
}

struct I32ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [i32],
    scale: D::I32Vec,
    zero: D::I32Vec,
    max: D::I32Vec,
}

impl<'a, D: SimdDescriptor> I32ToU8Reader<'a, D> {
    #[inline(always)]
    fn new(d: D, slice: &'a [i32], scale: i32, max: i32) -> Self {
        Self {
            slice,
            scale: D::I32Vec::splat(d, scale),
            zero: D::I32Vec::splat(d, 0),
            max: D::I32Vec::splat(d, max),
        }
    }

    #[inline(always)]
    fn convert_chunk(&self, d: D, in_chunk: &[i32], stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let i32_len = D::I32Vec::LEN;
        let ratio = u8_len.checked_div(i32_len).unwrap_or(1);
        assert!(in_chunk.len() >= u8_len);
        for k in 0..ratio {
            let val = D::I32Vec::load(d, &in_chunk[k * i32_len..]);
            let scaled = val * self.scale;
            let zeroclip = scaled.lt_zero().if_then_else_i32(self.zero, scaled);
            let clip = scaled.gt(self.max).if_then_else_i32(self.max, zeroclip);
            clip.store_u8(&mut stack[k * i32_len..(k + 1) * i32_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for I32ToU8Reader<'a, D> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        self.convert_chunk(d, &self.slice[base_x..base_x + D::U8Vec::LEN], stack)
    }

    #[inline(always)]
    fn read_u8_vec_tail(&self, d: D, base_x: usize, rem: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let mut in_buf = [0i32; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        self.convert_chunk(d, &in_buf, stack)
    }
}

struct OpaqueU8AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U8Vec,
}

impl<D: SimdDescriptor> OpaqueU8AlphaReader<D> {
    #[inline(always)]
    fn new(d: D) -> Self {
        Self {
            alpha_vec: D::U8Vec::splat(d, 255),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU8<D> for OpaqueU8AlphaReader<D> {
    #[inline(always)]
    fn check_len(&self, _len: usize) {}

    #[inline(always)]
    fn read_u8_vec(&self, _d: D, _base_x: usize, _stack: &mut [u8; 64]) -> D::U8Vec {
        self.alpha_vec
    }

    #[inline(always)]
    fn read_u8_vec_tail(
        &self,
        _d: D,
        _base_x: usize,
        _rem: usize,
        _stack: &mut [u8; 64],
    ) -> D::U8Vec {
        self.alpha_vec
    }
}

struct F32ToF16Reader<'a> {
    slice: &'a [f32],
}

impl<'a> F32ToF16Reader<'a> {
    #[inline(always)]
    fn new(slice: &'a [f32]) -> Self {
        Self { slice }
    }

    #[inline(always)]
    fn convert_chunk<D: SimdDescriptor>(
        &self,
        d: D,
        in_chunk: &[f32],
        stack: &mut [u16; 64],
    ) -> D::U16Vec {
        let u16_len = D::U16Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u16_len.checked_div(f32_len).unwrap_or(1);
        assert!(in_chunk.len() >= u16_len);
        for k in 0..ratio {
            let val = D::F32Vec::load(d, &in_chunk[k * f32_len..]);
            val.store_f16_bits(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U16Vec::load(d, &stack[..u16_len])
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for F32ToF16Reader<'a> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec {
        self.convert_chunk(d, &self.slice[base_x..base_x + D::U16Vec::LEN], stack)
    }

    #[inline(always)]
    fn read_u16_vec_tail(
        &self,
        d: D,
        base_x: usize,
        rem: usize,
        stack: &mut [u16; 64],
    ) -> D::U16Vec {
        let mut in_buf = [0.0f32; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        self.convert_chunk(d, &in_buf, stack)
    }
}

struct F32ToU16Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    scale_vec: D::F32Vec,
    zero: D::F32Vec,
}

impl<'a, D: SimdDescriptor> F32ToU16Reader<'a, D> {
    #[inline(always)]
    fn new(d: D, slice: &'a [f32], max: f32) -> Self {
        Self {
            slice,
            scale_vec: D::F32Vec::splat(d, max),
            zero: D::F32Vec::splat(d, 0.0),
        }
    }

    #[inline(always)]
    fn convert_chunk(&self, d: D, in_chunk: &[f32], stack: &mut [u16; 64]) -> D::U16Vec {
        let u16_len = D::U16Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u16_len.checked_div(f32_len).unwrap_or(1);
        assert!(in_chunk.len() >= u16_len);
        for k in 0..ratio {
            let val = D::F32Vec::load(d, &in_chunk[k * f32_len..]);
            let clamped = (val * self.scale_vec).max(self.zero).min(self.scale_vec);
            clamped.round_store_u16(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U16Vec::load(d, &stack[..u16_len])
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for F32ToU16Reader<'a, D> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec {
        self.convert_chunk(d, &self.slice[base_x..base_x + D::U16Vec::LEN], stack)
    }

    #[inline(always)]
    fn read_u16_vec_tail(
        &self,
        d: D,
        base_x: usize,
        rem: usize,
        stack: &mut [u16; 64],
    ) -> D::U16Vec {
        let mut in_buf = [0.0f32; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        self.convert_chunk(d, &in_buf, stack)
    }
}

struct OpaqueU16AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U16Vec,
}

impl<D: SimdDescriptor> OpaqueU16AlphaReader<D> {
    #[inline(always)]
    fn new(d: D, val: u16) -> Self {
        Self {
            alpha_vec: D::U16Vec::splat(d, val),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU16<D> for OpaqueU16AlphaReader<D> {
    #[inline(always)]
    fn check_len(&self, _len: usize) {}

    #[inline(always)]
    fn read_u16_vec(&self, _d: D, _base_x: usize, _stack: &mut [u16; 64]) -> D::U16Vec {
        self.alpha_vec
    }

    #[inline(always)]
    fn read_u16_vec_tail(
        &self,
        _d: D,
        _base_x: usize,
        _rem: usize,
        _stack: &mut [u16; 64],
    ) -> D::U16Vec {
        self.alpha_vec
    }
}

struct DirectF32Reader<'a> {
    slice: &'a [f32],
}

impl<'a> DirectF32Reader<'a> {
    #[inline(always)]
    fn new(slice: &'a [f32]) -> Self {
        Self { slice }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderF32<D> for DirectF32Reader<'a> {
    #[inline(always)]
    fn check_len(&self, len: usize) {
        assert!(self.slice.len() >= len);
    }

    #[inline(always)]
    fn read_f32_vec(&self, d: D, base_x: usize, _stack: &mut [f32; 64]) -> D::F32Vec {
        D::F32Vec::load(d, &self.slice[base_x..base_x + D::F32Vec::LEN])
    }

    #[inline(always)]
    fn read_f32_vec_tail(
        &self,
        d: D,
        base_x: usize,
        rem: usize,
        _stack: &mut [f32; 64],
    ) -> D::F32Vec {
        let mut in_buf = [0.0f32; 64];
        in_buf[..rem].copy_from_slice(&self.slice[base_x..base_x + rem]);
        D::F32Vec::load(d, &in_buf)
    }
}

struct OpaqueF32AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::F32Vec,
}

impl<D: SimdDescriptor> OpaqueF32AlphaReader<D> {
    #[inline(always)]
    fn new(d: D) -> Self {
        Self {
            alpha_vec: D::F32Vec::splat(d, 1.0),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderF32<D> for OpaqueF32AlphaReader<D> {
    #[inline(always)]
    fn check_len(&self, _len: usize) {}

    #[inline(always)]
    fn read_f32_vec(&self, _d: D, _base_x: usize, _stack: &mut [f32; 64]) -> D::F32Vec {
        self.alpha_vec
    }

    #[inline(always)]
    fn read_f32_vec_tail(
        &self,
        _d: D,
        _base_x: usize,
        _rem: usize,
        _stack: &mut [f32; 64],
    ) -> D::F32Vec {
        self.alpha_vec
    }
}

#[derive(Clone, Copy)]
pub(super) enum ChannelSourceU8<'a> {
    F32 {
        slice: &'a [f32],
        max: f32,
        dither_channel: usize,
    },
    I16 {
        slice: &'a [i16],
        mult: i16,
        max: i16,
    },
    I32 {
        slice: &'a [i32],
        mult: i32,
        max: i32,
    },
}

// --- Fused Execution Loops ---

macro_rules! define_run_fused_1 {
    ($name:ident, $trait_name:ident, $ty:ty, $vec_trait:ident, $read_fn:ident, $read_tail_fn:ident) => {
        #[inline(always)]
        fn $name<D: SimdDescriptor, R: $trait_name<D>>(d: D, r: &R, output: &mut [$ty]) {
            let vec_len = D::$vec_trait::LEN;
            let limit = output.len();
            r.check_len(limit);
            let num_blocks = limit / vec_len;
            let mut stack = [0 as $ty; 64];

            for block in 0..num_blocks {
                let base_x = block * vec_len;
                let v = r.$read_fn(d, base_x, &mut stack);
                v.store(&mut output[base_x..base_x + vec_len]);
            }

            let n = num_blocks * vec_len;
            if n < limit {
                let v = r.$read_tail_fn(d, n, limit - n, &mut stack);
                let mut tail = [0 as $ty; 64];
                v.store(&mut tail[..vec_len]);
                output[n..limit].copy_from_slice(&tail[..limit - n]);
            }
        }
    };
}

macro_rules! define_run_fused {
    (
        $name:ident,
        $trait_name:ident,
        $ty:ty,
        $vec_trait:ident,
        $store_fn:ident,
        $read_fn:ident,
        $read_tail_fn:ident,
        $cnt:expr,
        [$($R:ident),+],
        $(($r:ident : $r_ty:ident)),+
    ) => {
        #[inline(always)]
        fn $name<
            D: SimdDescriptor,
            $($R: $trait_name<D>,)+
        >(
            d: D,
            $($r: &$r_ty,)+
            output: &mut [$ty],
        ) {
            let vec_len = D::$vec_trait::LEN;
            let limit = output.len() / $cnt;
            $(
                $r.check_len(limit);
            )+
            let num_blocks = limit / vec_len;
            let mut stack = [0 as $ty; 64];

            for block in 0..num_blocks {
                let base_x = block * vec_len;
                $(
                    let $r = $r.$read_fn(d, base_x, &mut stack);
                )+
                D::$vec_trait::$store_fn(
                    $($r,)+
                    &mut output[base_x * $cnt..(base_x + vec_len) * $cnt],
                );
            }

            let n = num_blocks * vec_len;
            if n < limit {
                let rem = limit - n;
                $(
                    let $r = $r.$read_tail_fn(d, n, rem, &mut stack);
                )+
                let mut tail = [0 as $ty; 64 * $cnt];
                D::$vec_trait::$store_fn(
                    $($r,)+
                    &mut tail[..vec_len * $cnt],
                );
                output[n * $cnt..limit * $cnt].copy_from_slice(&tail[..rem * $cnt]);
            }
        }
    };
}

define_run_fused_1!(
    run_fused_1_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    read_u8_vec,
    read_u8_vec_tail
);
define_run_fused!(
    run_fused_2_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_2,
    read_u8_vec,
    read_u8_vec_tail,
    2,
    [R, RA],
    (r0: R),
    (r1: RA)
);
define_run_fused!(
    run_fused_3_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_3,
    read_u8_vec,
    read_u8_vec_tail,
    3,
    [R],
    (r0: R),
    (r1: R),
    (r2: R)
);
define_run_fused!(
    run_fused_4_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_4,
    read_u8_vec,
    read_u8_vec_tail,
    4,
    [R, RA],
    (r0: R),
    (r1: R),
    (r2: R),
    (r3: RA)
);

define_run_fused_1!(
    run_fused_1_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    read_u16_vec,
    read_u16_vec_tail
);
define_run_fused!(
    run_fused_2_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_2,
    read_u16_vec,
    read_u16_vec_tail,
    2,
    [R, RA],
    (r0: R),
    (r1: RA)
);
define_run_fused!(
    run_fused_3_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_3,
    read_u16_vec,
    read_u16_vec_tail,
    3,
    [R],
    (r0: R),
    (r1: R),
    (r2: R)
);
define_run_fused!(
    run_fused_4_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_4,
    read_u16_vec,
    read_u16_vec_tail,
    4,
    [R, RA],
    (r0: R),
    (r1: R),
    (r2: R),
    (r3: RA)
);

define_run_fused_1!(
    run_fused_1_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    read_f32_vec,
    read_f32_vec_tail
);
define_run_fused!(
    run_fused_2_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_2,
    read_f32_vec,
    read_f32_vec_tail,
    2,
    [R, RA],
    (r0: R),
    (r1: RA)
);
define_run_fused!(
    run_fused_3_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_3,
    read_f32_vec,
    read_f32_vec_tail,
    3,
    [R],
    (r0: R),
    (r1: R),
    (r2: R)
);
define_run_fused!(
    run_fused_4_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_4,
    read_f32_vec,
    read_f32_vec_tail,
    4,
    [R, RA],
    (r0: R),
    (r1: R),
    (r2: R),
    (r3: RA)
);

// --- SIMD Functions ---

macro_rules! with_u8_reader {
    ($d:expr, $pos:expr, $src:expr, |$r:ident| $body:expr) => {
        match $src {
            ChannelSourceU8::F32 {
                slice,
                max,
                dither_channel,
            } => {
                let $r = F32ToU8Reader::new($d, slice, max, $pos, dither_channel);
                $body
            }
            ChannelSourceU8::I16 { slice, mult, max } => {
                let $r = I16ToU8Reader::new($d, slice, mult, max);
                $body
            }
            ChannelSourceU8::I32 { slice, mult, max } => {
                let $r = I32ToU8Reader::new($d, slice, mult, max);
                $body
            }
        }
    };
}

macro_rules! with_u8_color3_readers {
    ($d:expr, $pos:expr, ($c0:expr, $c1:expr, $c2:expr), |$r0:ident, $r1:ident, $r2:ident| $body:expr) => {
        match ($c0, $c1, $c2) {
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    max,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                    ..
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                    ..
                },
            ) => {
                let $r0 = F32ToU8Reader::new($d, s0, max, $pos, dc0);
                let $r1 = F32ToU8Reader::new($d, s1, max, $pos, dc1);
                let $r2 = F32ToU8Reader::new($d, s2, max, $pos, dc2);
                $body
            }
            (
                ChannelSourceU8::I16 {
                    slice: s0,
                    mult,
                    max,
                },
                ChannelSourceU8::I16 { slice: s1, .. },
                ChannelSourceU8::I16 { slice: s2, .. },
            ) => {
                let $r0 = I16ToU8Reader::new($d, s0, mult, max);
                let $r1 = I16ToU8Reader::new($d, s1, mult, max);
                let $r2 = I16ToU8Reader::new($d, s2, mult, max);
                $body
            }
            (
                ChannelSourceU8::I32 {
                    slice: s0,
                    mult,
                    max,
                },
                ChannelSourceU8::I32 { slice: s1, .. },
                ChannelSourceU8::I32 { slice: s2, .. },
            ) => {
                let $r0 = I32ToU8Reader::new($d, s0, mult, max);
                let $r1 = I32ToU8Reader::new($d, s1, mult, max);
                let $r2 = I32ToU8Reader::new($d, s2, mult, max);
                $body
            }
            _ => unreachable!("color channels must share the same ChannelSourceU8 variant"),
        }
    };
}

simd_function!(
    store_fused_u8,
    d: D,
    pub(super) fn store_fused_u8_impl(
        inputs: &[ChannelSourceU8<'_>],
        fill_opaque_alpha: bool,
        position: (usize, usize),
        output: &mut [u8],
    ) {
        match (inputs, fill_opaque_alpha) {
            (&[c0], false) => {
                with_u8_reader!(d, position, c0, |r0| run_fused_1_u8(d, &r0, output));
            }
            (&[c0], true) => {
                let r1 = OpaqueU8AlphaReader::new(d);
                with_u8_reader!(d, position, c0, |r0| run_fused_2_u8(d, &r0, &r1, output));
            }
            (&[c0, c1], false) => {
                with_u8_reader!(d, position, c0, |r0| {
                    with_u8_reader!(d, position, c1, |r1| run_fused_2_u8(d, &r0, &r1, output))
                });
            }
            (&[c0, c1, c2], false) => {
                with_u8_color3_readers!(d, position, (c0, c1, c2), |r0, r1, r2| {
                    run_fused_3_u8(d, &r0, &r1, &r2, output)
                });
            }
            (&[c0, c1, c2], true) => {
                let r3 = OpaqueU8AlphaReader::new(d);
                with_u8_color3_readers!(d, position, (c0, c1, c2), |r0, r1, r2| {
                    run_fused_4_u8(d, &r0, &r1, &r2, &r3, output)
                });
            }
            (&[c0, c1, c2, c3], false) => {
                with_u8_color3_readers!(d, position, (c0, c1, c2), |r0, r1, r2| {
                    with_u8_reader!(d, position, c3, |r3| {
                        run_fused_4_u8(d, &r0, &r1, &r2, &r3, output)
                    })
                });
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_u16,
    d: D,
    pub(super) fn store_fused_u16_impl(
        inputs: &[&[f32]],
        max: f32,
        fill_opaque_alpha: bool,
        output: &mut [u16],
    ) {
        match (inputs, fill_opaque_alpha) {
            (&[s0], false) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                run_fused_1_u16(d, &r0, output);
            }
            (&[s0], true) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                let r1 = OpaqueU16AlphaReader::new(d, 0xFFFF);
                run_fused_2_u16(d, &r0, &r1, output);
            }
            (&[s0, s1], false) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                let r1 = F32ToU16Reader::new(d, s1, max);
                run_fused_2_u16(d, &r0, &r1, output);
            }
            (&[s0, s1, s2], false) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                let r1 = F32ToU16Reader::new(d, s1, max);
                let r2 = F32ToU16Reader::new(d, s2, max);
                run_fused_3_u16(d, &r0, &r1, &r2, output);
            }
            (&[s0, s1, s2], true) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                let r1 = F32ToU16Reader::new(d, s1, max);
                let r2 = F32ToU16Reader::new(d, s2, max);
                let r3 = OpaqueU16AlphaReader::new(d, 0xFFFF);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            (&[s0, s1, s2, s3], false) => {
                let r0 = F32ToU16Reader::new(d, s0, max);
                let r1 = F32ToU16Reader::new(d, s1, max);
                let r2 = F32ToU16Reader::new(d, s2, max);
                let r3 = F32ToU16Reader::new(d, s3, max);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_f16,
    d: D,
    pub(super) fn store_fused_f16_impl(
        inputs: &[&[f32]],
        fill_opaque_alpha: bool,
        output: &mut [u16],
    ) {
        match (inputs, fill_opaque_alpha) {
            (&[s0], false) => {
                let r0 = F32ToF16Reader::new(s0);
                run_fused_1_u16(d, &r0, output);
            }
            (&[s0], true) => {
                let r0 = F32ToF16Reader::new(s0);
                let r1 = OpaqueU16AlphaReader::new(d, 0x3C00);
                run_fused_2_u16(d, &r0, &r1, output);
            }
            (&[s0, s1], false) => {
                let r0 = F32ToF16Reader::new(s0);
                let r1 = F32ToF16Reader::new(s1);
                run_fused_2_u16(d, &r0, &r1, output);
            }
            (&[s0, s1, s2], false) => {
                let r0 = F32ToF16Reader::new(s0);
                let r1 = F32ToF16Reader::new(s1);
                let r2 = F32ToF16Reader::new(s2);
                run_fused_3_u16(d, &r0, &r1, &r2, output);
            }
            (&[s0, s1, s2], true) => {
                let r0 = F32ToF16Reader::new(s0);
                let r1 = F32ToF16Reader::new(s1);
                let r2 = F32ToF16Reader::new(s2);
                let r3 = OpaqueU16AlphaReader::new(d, 0x3C00);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            (&[s0, s1, s2, s3], false) => {
                let r0 = F32ToF16Reader::new(s0);
                let r1 = F32ToF16Reader::new(s1);
                let r2 = F32ToF16Reader::new(s2);
                let r3 = F32ToF16Reader::new(s3);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_f32,
    d: D,
    pub(super) fn store_fused_f32_impl(
        inputs: &[&[f32]],
        fill_opaque_alpha: bool,
        output: &mut [f32],
    ) {
        match (inputs, fill_opaque_alpha) {
            (&[s0], false) => {
                let r0 = DirectF32Reader::new(s0);
                run_fused_1_f32(d, &r0, output);
            }
            (&[s0], true) => {
                let r0 = DirectF32Reader::new(s0);
                let r1 = OpaqueF32AlphaReader::new(d);
                run_fused_2_f32(d, &r0, &r1, output);
            }
            (&[s0, s1], false) => {
                let r0 = DirectF32Reader::new(s0);
                let r1 = DirectF32Reader::new(s1);
                run_fused_2_f32(d, &r0, &r1, output);
            }
            (&[s0, s1, s2], false) => {
                let r0 = DirectF32Reader::new(s0);
                let r1 = DirectF32Reader::new(s1);
                let r2 = DirectF32Reader::new(s2);
                run_fused_3_f32(d, &r0, &r1, &r2, output);
            }
            (&[s0, s1, s2], true) => {
                let r0 = DirectF32Reader::new(s0);
                let r1 = DirectF32Reader::new(s1);
                let r2 = DirectF32Reader::new(s2);
                let r3 = OpaqueF32AlphaReader::new(d);
                run_fused_4_f32(d, &r0, &r1, &r2, &r3, output);
            }
            (&[s0, s1, s2, s3], false) => {
                let r0 = DirectF32Reader::new(s0);
                let r1 = DirectF32Reader::new(s1);
                let r2 = DirectF32Reader::new(s2);
                let r3 = DirectF32Reader::new(s3);
                run_fused_4_f32(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);
