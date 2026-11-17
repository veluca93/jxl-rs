// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32Vec, SimdDescriptor, SimdDescriptor128};

use crate::*;

#[inline(always)]
fn idct2d_2_2_impl<D: SimdDescriptor128>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    const { assert!(F32Vec::<D>::LEN == 4) };
    let mut buf = [0.0f32; 16];
    buf[0] = data[0];
    buf[1] = data[1];
    buf[4] = data[2];
    buf[5] = data[3];
    do_idct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    crate::transpose::transpose_in_place(d, &mut buf, 4);
    do_idct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);
    data[0] = buf[0];
    data[1] = buf[1];
    data[2] = buf[4];
    data[3] = buf[5];
}

#[inline(always)]
pub fn idct2d_2_2<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    idct2d_2_2_impl(d.maybe_downgrade_128bit(), data)
}

#[inline(always)]
pub fn idct2d_4_4<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 16, "Data length mismatch");
    let d = d.maybe_downgrade_128bit();
    let d_col = d;
    col_idct_4(d_col, data, 4);
    crate::transpose::transpose_in_place(d, data, 4);
    col_idct_4(d_col, data, 4);
}

#[inline(always)]
pub fn idct2d_4_8<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    let d = d.maybe_downgrade_128bit();
    let mut tmp = [0.0f32; 32];
    crate::transpose::transpose(d, data, &mut tmp, 4, 8);
    col_idct_8(d, &mut tmp, 4);
    crate::transpose::transpose(d, &tmp, data, 8, 4);
    col_idct_4(d, data, 8);
}

#[inline(always)]
pub fn idct2d_8_4<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    let d = d.maybe_downgrade_128bit();
    col_idct_4(d, data, 8);
    let mut tmp = [0.0f32; 32];
    crate::transpose::transpose(d, data, &mut tmp, 4, 8);
    col_idct_8_inout(d, &tmp, data, 4);
}

#[inline(always)]
pub fn idct2d_8_8<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 64, "Data length mismatch");
    let d_col = d.maybe_downgrade_256bit();
    col_idct_8(d_col, data, 8);
    crate::transpose::transpose_in_place(d, data, 8);
    col_idct_8(d_col, data, 8);
}

#[inline(always)]
pub fn idct2d_8_16<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    let mut tmp = [0.0f32; 128];
    crate::transpose::transpose(d, data, &mut tmp, 8, 16);
    col_idct_16(d.maybe_downgrade_256bit(), &mut tmp, 8);
    crate::transpose::transpose(d, &tmp, data, 16, 8);
    col_idct_8(d, data, 16);
}

#[inline(always)]
pub fn idct2d_8_32<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 256, "Data length mismatch");
    let mut tmp = [0.0f32; 256];
    crate::transpose::transpose(d, data, &mut tmp, 8, 32);
    col_idct_32(d.maybe_downgrade_256bit(), &mut tmp, 8);
    crate::transpose::transpose(d, &tmp, data, 32, 8);
    col_idct_8(d, data, 32);
}

#[inline(always)]
pub fn idct2d_16_8<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    col_idct_8(d, data, 16);
    let mut tmp = [0.0f32; 128];
    crate::transpose::transpose(d, data, &mut tmp, 8, 16);
    col_idct_16_inout(d.maybe_downgrade_256bit(), &tmp, data, 8);
}

#[inline(always)]
pub fn idct2d_16_16<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 256, "Data length mismatch");
    let d_col = d;
    col_idct_16(d_col, data, 16);
    crate::transpose::transpose_in_place(d, data, 16);
    col_idct_16(d_col, data, 16);
}

#[inline(always)]
pub fn idct2d_16_32<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    let data = &mut data[..512];
    let len = F32Vec::<D>::LEN;
    let column_chunks = 32 / len;
    let row_chunks = 16 / len;
    for i in 0..row_chunks {
        for j in 0..column_chunks {
            crate::transpose::transpose_in_place_strided(
                d,
                &mut data[(i * 32 + j) * len..],
                len,
                32,
            );
        }
        let data_arr = F32Vec::<D>::make_array_slice_mut(data);
        do_idct_32_rowblock(d, &mut data_arr[i * 32..]);
    }
    for i in 0..column_chunks {
        for j in 0..row_chunks {
            crate::transpose::transpose_in_place_strided(
                d,
                &mut data[(j * 32 + i) * len..],
                len,
                32,
            );
        }
        let data_arr = F32Vec::<D>::make_array_slice_mut(data);
        do_idct_16(d, &mut data_arr[i..], column_chunks);
    }
}

#[inline(always)]
pub fn idct2d_32_8<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 256, "Data length mismatch");
    col_idct_8(d, data, 32);
    let mut tmp = [0.0f32; 256];
    crate::transpose::transpose(d, data, &mut tmp, 8, 32);
    col_idct_32_inout(d.maybe_downgrade_256bit(), &tmp, data, 8);
}

#[inline(always)]
pub fn idct2d_32_16<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    col_idct_16(d, data, 32);
    let mut tmp = [0.0f32; 512];
    crate::transpose::transpose(d, data, &mut tmp, 16, 32);
    col_idct_32_inout(d, &tmp, data, 16);
}

#[inline(always)]
pub fn idct2d_32_32<D: SimdDescriptor>(d: D, data: &mut [f32]) {
    assert_eq!(data.len(), 1024, "Data length mismatch");
    let d_col = d;
    col_idct_32(d_col, data, 32);
    crate::transpose::transpose_in_place(d, data, 32);
    col_idct_32(d_col, data, 32);
}
