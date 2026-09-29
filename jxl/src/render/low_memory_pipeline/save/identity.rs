// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::too_many_arguments)]

use jxl_simd::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec,
    U16SimdVec, simd_function,
};

use crate::image::ImageDataType;

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

#[inline(always)]
fn load_or_padded_f32<D: SimdDescriptor>(d: D, slice: &[f32], x: usize) -> D::F32Vec {
    if x + D::F32Vec::LEN <= slice.len() {
        D::F32Vec::load(d, &slice[x..])
    } else {
        let mut buf = [0.0f32; 64];
        let avail = slice.len().saturating_sub(x);
        buf[..avail].copy_from_slice(&slice[x..]);
        D::F32Vec::load(d, &buf)
    }
}

#[inline(always)]
fn load_or_padded_u8<D: SimdDescriptor>(d: D, slice: &[u8], x: usize) -> D::U8Vec {
    if x + D::U8Vec::LEN <= slice.len() {
        D::U8Vec::load(d, &slice[x..])
    } else {
        let mut buf = [0u8; 64];
        let avail = slice.len().saturating_sub(x);
        buf[..avail].copy_from_slice(&slice[x..]);
        D::U8Vec::load(d, &buf)
    }
}

#[inline(always)]
fn load_or_padded_u16<D: SimdDescriptor>(d: D, slice: &[u16], x: usize) -> D::U16Vec {
    if x + D::U16Vec::LEN <= slice.len() {
        D::U16Vec::load(d, &slice[x..])
    } else {
        let mut buf = [0u16; 64];
        let avail = slice.len().saturating_sub(x);
        buf[..avail].copy_from_slice(&slice[x..]);
        D::U16Vec::load(d, &buf)
    }
}

#[inline(always)]
fn load_or_padded_i16<D: SimdDescriptor>(d: D, slice: &[i16], x: usize) -> D::I16Vec {
    if x + D::I16Vec::LEN <= slice.len() {
        D::I16Vec::load(d, &slice[x..])
    } else {
        let mut buf = [0i16; 64];
        let avail = slice.len().saturating_sub(x);
        buf[..avail].copy_from_slice(&slice[x..]);
        D::I16Vec::load(d, &buf)
    }
}

#[inline(always)]
fn load_or_padded_i32<D: SimdDescriptor>(d: D, slice: &[i32], x: usize) -> D::I32Vec {
    if x + D::I32Vec::LEN <= slice.len() {
        D::I32Vec::load(d, &slice[x..])
    } else {
        let mut buf = [0i32; 64];
        let avail = slice.len().saturating_sub(x);
        buf[..avail].copy_from_slice(&slice[x..]);
        D::I32Vec::load(d, &buf)
    }
}

// --- Channel Readers ---

pub(crate) trait ChannelReaderU8<D: SimdDescriptor> {
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec;
}

pub(crate) trait ChannelReaderU16<D: SimdDescriptor> {
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec;
}

pub(crate) trait ChannelReaderF32<D: SimdDescriptor> {
    fn read_f32_vec(&self, d: D, base_x: usize, stack: &mut [f32; 64]) -> D::F32Vec;
}

pub(crate) struct DirectSliceReader<'a, T> {
    slice: &'a [T],
}

impl<'a, T> DirectSliceReader<'a, T> {
    #[inline(always)]
    pub(crate) fn new(slice: &'a [T]) -> Self {
        Self { slice }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for DirectSliceReader<'a, u8> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, _stack: &mut [u8; 64]) -> D::U8Vec {
        load_or_padded_u8(d, self.slice, base_x)
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for DirectSliceReader<'a, u16> {
    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, _stack: &mut [u16; 64]) -> D::U16Vec {
        load_or_padded_u16(d, self.slice, base_x)
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderF32<D> for DirectSliceReader<'a, f32> {
    #[inline(always)]
    fn read_f32_vec(&self, d: D, base_x: usize, _stack: &mut [f32; 64]) -> D::F32Vec {
        load_or_padded_f32(d, self.slice, base_x)
    }
}

pub(crate) struct F32ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    scale_vec: D::F32Vec,
    zero: D::F32Vec,
    dither_row: &'static [f32; 64],
    x0: usize,
    dither_channel: usize,
}

impl<'a, D: SimdDescriptor> F32ToU8Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(
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
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for F32ToU8Reader<'a, D> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u8_len.checked_div(f32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let val = load_or_padded_f32(d, self.slice, x);
            let dx = (self.x0 + x + self.dither_channel * 23) % 32;
            let clamped = convert_f32_vec(d, val, self.scale_vec, self.zero, self.dither_row, dx);
            clamped.round_store_u8(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

pub(crate) struct I16ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [i16],
    scale: D::I16Vec,
    zero: D::I16Vec,
    max_vec: D::I16Vec,
}

impl<'a, D: SimdDescriptor> I16ToU8Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(d: D, slice: &'a [i16], mult: i16, max: i16) -> Self {
        Self {
            slice,
            scale: D::I16Vec::splat(d, mult),
            zero: D::I16Vec::splat(d, 0),
            max_vec: D::I16Vec::splat(d, max),
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for I16ToU8Reader<'a, D> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let i16_len = D::I16Vec::LEN;
        let ratio = u8_len.checked_div(i16_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * i16_len;
            let val = load_or_padded_i16(d, self.slice, x);
            let clip = convert_i16_vec(d, val, self.scale, self.zero, self.max_vec);
            clip.store_u8(&mut stack[k * i16_len..(k + 1) * i16_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

pub(crate) struct I32ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [i32],
    scale: D::I32Vec,
    zero: D::I32Vec,
    max: D::I32Vec,
}

impl<'a, D: SimdDescriptor> I32ToU8Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(d: D, slice: &'a [i32], scale: i32, max: i32) -> Self {
        Self {
            slice,
            scale: D::I32Vec::splat(d, scale),
            zero: D::I32Vec::splat(d, 0),
            max: D::I32Vec::splat(d, max),
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for I32ToU8Reader<'a, D> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let i32_len = D::I32Vec::LEN;
        let ratio = u8_len.checked_div(i32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * i32_len;
            let val = load_or_padded_i32(d, self.slice, x);
            let scaled = val * self.scale;
            let zeroclip = scaled.lt_zero().if_then_else_i32(self.zero, scaled);
            let clip = scaled.gt(self.max).if_then_else_i32(self.max, zeroclip);
            clip.store_u8(&mut stack[k * i32_len..(k + 1) * i32_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }
}

pub(crate) struct OpaqueU8AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U8Vec,
}

impl<D: SimdDescriptor> OpaqueU8AlphaReader<D> {
    #[inline(always)]
    pub(crate) fn new(d: D) -> Self {
        Self {
            alpha_vec: D::U8Vec::splat(d, 255),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU8<D> for OpaqueU8AlphaReader<D> {
    #[inline(always)]
    fn read_u8_vec(&self, _d: D, _base_x: usize, _stack: &mut [u8; 64]) -> D::U8Vec {
        self.alpha_vec
    }
}

pub(crate) struct F32ToF16Reader<'a> {
    slice: &'a [f32],
}

impl<'a> F32ToF16Reader<'a> {
    #[inline(always)]
    pub(crate) fn new(slice: &'a [f32]) -> Self {
        Self { slice }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for F32ToF16Reader<'a> {
    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec {
        let u16_len = D::U16Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u16_len.checked_div(f32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let val = load_or_padded_f32(d, self.slice, x);
            val.store_f16_bits(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U16Vec::load(d, &stack[..u16_len])
    }
}

pub(crate) struct F32ToU16Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    scale_vec: D::F32Vec,
    zero: D::F32Vec,
}

impl<'a, D: SimdDescriptor> F32ToU16Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(d: D, slice: &'a [f32], max: f32) -> Self {
        Self {
            slice,
            scale_vec: D::F32Vec::splat(d, max),
            zero: D::F32Vec::splat(d, 0.0),
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for F32ToU16Reader<'a, D> {
    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec {
        let u16_len = D::U16Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u16_len.checked_div(f32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let val = load_or_padded_f32(d, self.slice, x);
            let clamped = (val * self.scale_vec).max(self.zero).min(self.scale_vec);
            clamped.round_store_u16(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U16Vec::load(d, &stack[..u16_len])
    }
}

pub(crate) struct OpaqueF16AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U16Vec,
}

impl<D: SimdDescriptor> OpaqueF16AlphaReader<D> {
    #[inline(always)]
    pub(crate) fn new(d: D) -> Self {
        Self {
            alpha_vec: D::U16Vec::splat(d, 0x3C00),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU16<D> for OpaqueF16AlphaReader<D> {
    #[inline(always)]
    fn read_u16_vec(&self, _d: D, _base_x: usize, _stack: &mut [u16; 64]) -> D::U16Vec {
        self.alpha_vec
    }
}

#[derive(Clone, Copy)]
pub(super) enum ChannelSourceU8<'a> {
    F32 {
        slice: &'a [f32],
        dither_channel: usize,
    },
    I16 {
        slice: &'a [i16],
    },
    OpaqueAlpha,
}

#[derive(Clone, Copy)]
pub(super) enum ChannelSourceU16<'a> {
    F32 {
        slice: &'a [f32],
    },
    OpaqueAlpha,
}

// --- Fused Execution Loops ---

macro_rules! define_run_fused_1 {
    ($name:ident, $trait_name:ident, $ty:ty, $vec_trait:ident, $read_fn:ident) => {
        #[inline(always)]
        fn $name<D: SimdDescriptor, R: $trait_name<D>>(
            d: D,
            r: &R,
            output: &mut [$ty],
        ) {
            let vec_len = D::$vec_trait::LEN;
            let limit = output.len();
            let num_blocks = limit / vec_len;
            let mut stack = [0 as $ty; 64];

            for block in 0..num_blocks {
                let base_x = block * vec_len;
                let v = r.$read_fn(d, base_x, &mut stack);
                v.store(&mut output[base_x..base_x + vec_len]);
            }

            let n = num_blocks * vec_len;
            if n < limit {
                let v = r.$read_fn(d, n, &mut stack);
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
        $cnt:expr,
        $(($r:ident, $R:ident)),+
    ) => {
        #[inline(always)]
        fn $name<
            D: SimdDescriptor,
            $($R: $trait_name<D>,)+
        >(
            d: D,
            $($r: &$R,)+
            output: &mut [$ty],
        ) {
            let vec_len = D::$vec_trait::LEN;
            let limit = output.len() / $cnt;
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
                $(
                    let $r = $r.$read_fn(d, n, &mut stack);
                )+
                let mut tail = [0 as $ty; 64 * $cnt];
                D::$vec_trait::$store_fn(
                    $($r,)+
                    &mut tail[..vec_len * $cnt],
                );
                output[n * $cnt..limit * $cnt].copy_from_slice(&tail[..(limit - n) * $cnt]);
            }
        }
    };
}

define_run_fused_1!(run_fused_1_u8, ChannelReaderU8, u8, U8Vec, read_u8_vec);
define_run_fused!(
    run_fused_2_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_2,
    read_u8_vec,
    2,
    (r0, R0),
    (r1, R1)
);
define_run_fused!(
    run_fused_3_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_3,
    read_u8_vec,
    3,
    (r0, R0),
    (r1, R1),
    (r2, R2)
);
define_run_fused!(
    run_fused_4_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_4,
    read_u8_vec,
    4,
    (r0, R0),
    (r1, R1),
    (r2, R2),
    (r3, R3)
);

define_run_fused_1!(run_fused_1_u16, ChannelReaderU16, u16, U16Vec, read_u16_vec);
define_run_fused!(
    run_fused_2_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_2,
    read_u16_vec,
    2,
    (r0, R0),
    (r1, R1)
);
define_run_fused!(
    run_fused_3_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_3,
    read_u16_vec,
    3,
    (r0, R0),
    (r1, R1),
    (r2, R2)
);
define_run_fused!(
    run_fused_4_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_4,
    read_u16_vec,
    4,
    (r0, R0),
    (r1, R1),
    (r2, R2),
    (r3, R3)
);

define_run_fused_1!(run_fused_1_f32, ChannelReaderF32, f32, F32Vec, read_f32_vec);
define_run_fused!(
    run_fused_2_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_2,
    read_f32_vec,
    2,
    (r0, R0),
    (r1, R1)
);
define_run_fused!(
    run_fused_3_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_3,
    read_f32_vec,
    3,
    (r0, R0),
    (r1, R1),
    (r2, R2)
);
define_run_fused!(
    run_fused_4_f32,
    ChannelReaderF32,
    f32,
    F32Vec,
    store_interleaved_4,
    read_f32_vec,
    4,
    (r0, R0),
    (r1, R1),
    (r2, R2),
    (r3, R3)
);

// --- SIMD Functions ---

simd_function!(
    store_interleaved_u8,
    d: D,
    fn store_interleaved_impl_u8(inputs: &[&[u8]], output: &mut [u8]) {
        match inputs.len() {
            1 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                run_fused_1_u8(d, &r0, output);
            }
            2 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                run_fused_2_u8(d, &r0, &r1, output);
            }
            3 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                run_fused_3_u8(d, &r0, &r1, &r2, output);
            }
            4 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                let r3 = DirectSliceReader::new(inputs[3]);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_interleaved_u16,
    d: D,
    fn store_interleaved_impl_u16(inputs: &[&[u16]], output: &mut [u16]) {
        match inputs.len() {
            1 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                run_fused_1_u16(d, &r0, output);
            }
            2 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                run_fused_2_u16(d, &r0, &r1, output);
            }
            3 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                run_fused_3_u16(d, &r0, &r1, &r2, output);
            }
            4 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                let r3 = DirectSliceReader::new(inputs[3]);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_interleaved_f32,
    d: D,
    fn store_interleaved_impl_f32(inputs: &[&[f32]], output: &mut [f32]) {
        match inputs.len() {
            1 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                run_fused_1_f32(d, &r0, output);
            }
            2 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                run_fused_2_f32(d, &r0, &r1, output);
            }
            3 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                run_fused_3_f32(d, &r0, &r1, &r2, output);
            }
            4 => {
                let r0 = DirectSliceReader::new(inputs[0]);
                let r1 = DirectSliceReader::new(inputs[1]);
                let r2 = DirectSliceReader::new(inputs[2]);
                let r3 = DirectSliceReader::new(inputs[3]);
                run_fused_4_f32(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

pub(super) fn store_u8(slices: &[&[u8]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out = &mut output_buf[..xsize * channels];
    store_interleaved_u8(slices, out);
    xsize
}

pub(super) fn store_u16(slices: &[&[u16]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out_bytes = &mut output_buf[..xsize * channels * 2];
    let ptr = out_bytes.as_mut_ptr();
    if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
        let out_u16 = u16::cast_slice_mut(out_bytes);
        store_interleaved_u16(slices, out_u16);
        xsize
    } else {
        0
    }
}

pub(super) fn store_f32(slices: &[&[f32]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out_bytes = &mut output_buf[..xsize * channels * 4];
    let ptr = out_bytes.as_mut_ptr();
    if ptr.align_offset(std::mem::align_of::<f32>()) == 0 {
        let out_f32 = f32::cast_slice_mut(out_bytes);
        store_interleaved_f32(slices, out_f32);
        xsize
    } else {
        0
    }
}

simd_function!(
    f32_to_u8_simd,
    d: D,
    pub(super) fn f32_to_u8_simd_impl(
        input: &[f32],
        output: &mut [u8],
        max: f32,
        position: (usize, usize),
        dither_channel: usize,
    ) {
        let r = F32ToU8Reader::new(d, input, max, position, dither_channel);
        run_fused_1_u8(d, &r, output);
    }
);

simd_function!(
    i16_to_u8_simd,
    d: D,
    pub(super) fn i16_to_u8_simd_impl(
        input: &[i16],
        output: &mut [u8],
        mult: i16,
        max: i16,
    ) {
        let r = I16ToU8Reader::new(d, input, mult, max);
        run_fused_1_u8(d, &r, output);
    }
);

simd_function!(
    i32_to_u8_simd_dispatch,
    d: D,
    pub(crate) fn i32_to_u8_simd(
        input: &[i32],
        output: &mut [u8],
        scale: i32,
        max: i32,
    ) {
        let r = I32ToU8Reader::new(d, input, scale, max);
        run_fused_1_u8(d, &r, output);
    }
);

simd_function!(
    f32_to_u16_simd_dispatch,
    d: D,
    pub(crate) fn f32_to_u16_simd(
        input: &[f32],
        output: &mut [u16],
        max: f32,
    ) {
        let r = F32ToU16Reader::new(d, input, max);
        run_fused_1_u16(d, &r, output);
    }
);

simd_function!(
    f32_to_f16_simd_dispatch,
    d: D,
    pub(crate) fn f32_to_f16_simd(
        input: &[f32],
        output: &mut [u16],
    ) {
        let r = F32ToF16Reader::new(input);
        run_fused_1_u16(d, &r, output);
    }
);

simd_function!(
    store_fused_3_u8,
    d: D,
    pub(super) fn store_fused_3_u8_impl(
        c0: ChannelSourceU8<'_>,
        c1: ChannelSourceU8<'_>,
        c2: ChannelSourceU8<'_>,
        output: &mut [u8],
        max: f32,
        mult: i16,
        max_i16: i16,
        position: (usize, usize),
    ) {
        match (c0, c1, c2) {
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                run_fused_3_u8(d, &r0, &r1, &r2, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                run_fused_3_u8(d, &r0, &r1, &r2, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_4_u8,
    d: D,
    pub(super) fn store_fused_4_u8_impl(
        c0: ChannelSourceU8<'_>,
        c1: ChannelSourceU8<'_>,
        c2: ChannelSourceU8<'_>,
        c3: ChannelSourceU8<'_>,
        output: &mut [u8],
        max: f32,
        mult: i16,
        max_i16: i16,
        position: (usize, usize),
    ) {
        match (c0, c1, c2, c3) {
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::F32 {
                    slice: s3,
                    dither_channel: dc3,
                },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = F32ToU8Reader::new(d, s3, max, position, dc3);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::I16 { slice: s3 },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = I16ToU8Reader::new(d, s3, mult, max_i16);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::OpaqueAlpha,
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = OpaqueU8AlphaReader::new(d);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
                ChannelSourceU8::I16 { slice: s3 },
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                let r3 = I16ToU8Reader::new(d, s3, mult, max_i16);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
                ChannelSourceU8::OpaqueAlpha,
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                let r3 = OpaqueU8AlphaReader::new(d);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_3_f16,
    d: D,
    pub(super) fn store_fused_3_f16_impl(
        s0: &[f32],
        s1: &[f32],
        s2: &[f32],
        output: &mut [u16],
    ) {
        let r0 = F32ToF16Reader::new(s0);
        let r1 = F32ToF16Reader::new(s1);
        let r2 = F32ToF16Reader::new(s2);
        run_fused_3_u16(d, &r0, &r1, &r2, output);
    }
);

simd_function!(
    store_fused_4_f16,
    d: D,
    pub(super) fn store_fused_4_f16_impl(
        s0: &[f32],
        s1: &[f32],
        s2: &[f32],
        s3: ChannelSourceU16<'_>,
        output: &mut [u16],
    ) {
        let r0 = F32ToF16Reader::new(s0);
        let r1 = F32ToF16Reader::new(s1);
        let r2 = F32ToF16Reader::new(s2);
        match s3 {
            ChannelSourceU16::F32 { slice: s3 } => {
                let r3 = F32ToF16Reader::new(s3);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            ChannelSourceU16::OpaqueAlpha => {
                let r3 = OpaqueF16AlphaReader::new(d);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
        }
    }
);
