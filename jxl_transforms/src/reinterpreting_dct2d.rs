// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32Vec, SimdDescriptor, SimdDescriptor128};

use crate::*;

#[inline(always)]
fn reinterpreting_dct2d_1_2_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 2, "Data length mismatch");
    assert!(output.len() > 1);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 8];
    for i in 0..2 {
        buf[i * 4] = data[i];
    }
    {
        let buf = F32Vec::<D>::make_array_slice_mut(&mut buf);
        do_reinterpreting_dct_2(d, buf, 1);
    }
    for x in 0..2 {
        output[x] = buf[x * 4];
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_1_2<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_1_2_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_2_1_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 2, "Data length mismatch");
    assert!(output.len() > 1);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 8];
    for i in 0..2 {
        buf[i * 4] = data[i];
    }
    {
        let buf = F32Vec::<D>::make_array_slice_mut(&mut buf);
        do_reinterpreting_dct_2(d, buf, 1);
    }
    for x in 0..2 {
        output[x] = buf[x * 4];
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_2_1<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_2_1_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_1_4_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 3);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    for i in 0..4 {
        buf[i * 4] = data[i];
    }
    {
        let buf = F32Vec::<D>::make_array_slice_mut(&mut buf);
        do_reinterpreting_dct_4(d, buf, 1);
    }
    for x in 0..4 {
        output[x] = buf[x * 4];
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_1_4<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_1_4_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_4_1_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 3);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    for i in 0..4 {
        buf[i * 4] = data[i];
    }
    {
        let buf = F32Vec::<D>::make_array_slice_mut(&mut buf);
        do_reinterpreting_dct_4(d, buf, 1);
    }
    for x in 0..4 {
        output[x] = buf[x * 4];
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_1<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_4_1_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_2_2_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 17);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    for r in 0..2 {
        buf[r * 4..r * 4 + 2].copy_from_slice(&data[r * 2..r * 2 + 2]);
    }
    do_reinterpreting_dct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    crate::transpose::transpose_in_place(d, &mut buf, 4);
    do_reinterpreting_dct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    for y in 0..2 {
        for x in 0..2 {
            output[y * 16 + x] = buf[y * 4 + x];
        }
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_2_2<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_2_2_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_2_4_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 8, "Data length mismatch");
    assert!(output.len() > 35);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    for r in 0..2 {
        buf[r * 4..r * 4 + 4].copy_from_slice(&data[r * 4..r * 4 + 4]);
    }
    crate::transpose::transpose_in_place(d, &mut buf, 4);
    do_reinterpreting_dct_4(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    crate::transpose::transpose_in_place(d, &mut buf, 4);
    do_reinterpreting_dct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    for y in 0..2 {
        for x in 0..4 {
            output[y * 32 + x] = buf[y * 4 + x];
        }
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_2_4<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_2_4_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
fn reinterpreting_dct2d_4_2_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 8, "Data length mismatch");
    assert!(output.len() > 35);
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    for r in 0..4 {
        buf[r * 4..r * 4 + 2].copy_from_slice(&data[r * 2..r * 2 + 2]);
    }
    do_reinterpreting_dct_4(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    crate::transpose::transpose_in_place(d, &mut buf, 4);
    do_reinterpreting_dct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    for y in 0..2 {
        for x in 0..4 {
            output[y * 32 + x] = buf[y * 4 + x];
        }
    }
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_2<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    reinterpreting_dct2d_4_2_impl(d.maybe_downgrade_128bit(), data, output)
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_4<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 16, "Data length mismatch");
    assert!(output.len() > 99);
    col_reinterpreting_dct_4(d.maybe_downgrade_128bit(), data, 4, 4);
    crate::transpose::transpose_into_strided(d, data, output, 4, 4, 32);
    col_reinterpreting_dct_4(d.maybe_downgrade_128bit(), output, 4, 32);
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_8<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    assert!(output.len() > 199);
    col_reinterpreting_dct_4(d.maybe_downgrade_256bit(), data, 8, 8);
    let mut tmp = [0.0f32; 32];
    crate::transpose::transpose(d, data, &mut tmp, 4, 8);
    col_reinterpreting_dct_8(d.maybe_downgrade_128bit(), &mut tmp, 4, 4);
    crate::transpose::transpose_into_strided(d, &tmp, output, 8, 4, 64);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_4<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    assert!(output.len() > 199);
    col_reinterpreting_dct_8(d.maybe_downgrade_128bit(), data, 4, 4);
    crate::transpose::transpose_into_strided(d, data, output, 8, 4, 64);
    col_reinterpreting_dct_4(d.maybe_downgrade_256bit(), output, 8, 64);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_8<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 64, "Data length mismatch");
    assert!(output.len() > 455);
    col_reinterpreting_dct_8(d.maybe_downgrade_256bit(), data, 8, 8);
    crate::transpose::transpose_into_strided(d, data, output, 8, 8, 64);
    col_reinterpreting_dct_8(d.maybe_downgrade_256bit(), output, 8, 64);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_16<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    assert!(output.len() > 911);
    col_reinterpreting_dct_8(d, data, 16, 16);
    let mut tmp = [0.0f32; 128];
    crate::transpose::transpose(d, data, &mut tmp, 8, 16);
    col_reinterpreting_dct_16(d.maybe_downgrade_256bit(), &mut tmp, 8, 8);
    crate::transpose::transpose_into_strided(d, &tmp, output, 16, 8, 128);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_8<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    assert!(output.len() > 911);
    col_reinterpreting_dct_16(d.maybe_downgrade_256bit(), data, 8, 8);
    crate::transpose::transpose_into_strided(d, data, output, 16, 8, 128);
    col_reinterpreting_dct_8(d, output, 16, 128);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_16<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 256, "Data length mismatch");
    assert!(output.len() > 1935);
    col_reinterpreting_dct_16(d, data, 16, 16);
    crate::transpose::transpose_into_strided(d, data, output, 16, 16, 128);
    col_reinterpreting_dct_16(d, output, 16, 128);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_32<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    assert!(output.len() > 3871);
    col_reinterpreting_dct_16(d, data, 32, 32);
    let mut tmp = [0.0f32; 512];
    crate::transpose::transpose(d, data, &mut tmp, 16, 32);
    col_reinterpreting_dct_32(d, &mut tmp, 16, 16);
    crate::transpose::transpose_into_strided(d, &tmp, output, 32, 16, 256);
}

#[inline(always)]
pub fn reinterpreting_dct2d_32_16<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    assert!(output.len() > 3871);
    col_reinterpreting_dct_32(d, data, 16, 16);
    crate::transpose::transpose_into_strided(d, data, output, 32, 16, 256);
    col_reinterpreting_dct_16(d, output, 32, 256);
}

#[inline(always)]
pub fn reinterpreting_dct2d_32_32<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 1024, "Data length mismatch");
    assert!(output.len() > 7967);
    col_reinterpreting_dct_32(d, data, 32, 32);
    let chunks = 32 / F32Vec::<D>::LEN;
    for i in 0..chunks {
        crate::transpose::transpose_in_place_step(d, data, 32, i);
        let data_arr = F32Vec::<D>::make_array_slice_mut(&mut data[..1024]);
        do_reinterpreting_dct_32(d, &mut data_arr[i..], chunks);
    }
    let data = &data[..1024];
    let output = &mut output[..31 * 256 + 32];
    let len = F32Vec::<D>::LEN;
    for y in 0..32 {
        for x in (0..32).step_by(len) {
            F32Vec::load(d, &data[y * 32 + x..]).store(&mut output[y * 256 + x..]);
        }
    }
}
