// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unused)]
#![allow(clippy::type_complexity)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
use jxl_simd::{F32Vec, F32VecArray, SimdDescriptor};

#[allow(clippy::too_many_arguments)]
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub(super) fn reinterpreting_dct_4<D: SimdDescriptor>(
    d: D,
    v0: F32Vec<D>,
    v1: F32Vec<D>,
    v2: F32Vec<D>,
    v3: F32Vec<D>,
) -> (F32Vec<D>, F32Vec<D>, F32Vec<D>, F32Vec<D>) {
    let v4 = v0 + v3;
    let v5 = v1 + v2;
    let v6 = v4 + v5;
    let v7 = v4 - v5;
    let v8 = v0 - v3;
    let v9 = v1 - v2;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let v10 = v8 * mul;
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let v11 = v9 * mul;
    let v12 = v10 + v11;
    let v13 = v10 - v11;
    let v14 = v12.mul_add(F32Vec::splat(d, std::f32::consts::SQRT_2), v13);
    (
        v6 * F32Vec::splat(d, 0.250000),
        v14 * F32Vec::splat(d, 0.256440),
        v7 * F32Vec::splat(d, 0.277234),
        v13 * F32Vec::splat(d, 0.317640),
    )
}

#[inline(always)]
pub(super) fn do_reinterpreting_dct_4<D: SimdDescriptor>(
    d: D,
    data: &mut [F32VecArray<D>],
    stride: usize,
) {
    assert!(data.len() > 3 * stride);
    let mut v0 = F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = F32Vec::load_array(d, &data[1 * stride]);
    let mut v2 = F32Vec::load_array(d, &data[2 * stride]);
    let mut v3 = F32Vec::load_array(d, &data[3 * stride]);
    (v0, v1, v2, v3) = reinterpreting_dct_4(d, v0, v1, v2, v3);
    v0.store_array(&mut data[0 * stride]);
    v1.store_array(&mut data[1 * stride]);
    v2.store_array(&mut data[2 * stride]);
    v3.store_array(&mut data[3 * stride]);
}

#[inline(always)]
pub(super) fn col_reinterpreting_dct_4<D: SimdDescriptor>(
    d: D,
    data: &mut [f32],
    cols: usize,
    row_stride: usize,
) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    assert!(row_stride.is_multiple_of(F32Vec::<D>::LEN));
    let col_chunks = cols / F32Vec::<D>::LEN;
    let stride_chunks = row_stride / F32Vec::<D>::LEN;
    let len = 3 * row_stride + cols;
    let data = F32Vec::<D>::make_array_slice_mut(&mut data[..len]);
    for i in 0..col_chunks {
        do_reinterpreting_dct_4(d, &mut data[i..], stride_chunks);
    }
}
