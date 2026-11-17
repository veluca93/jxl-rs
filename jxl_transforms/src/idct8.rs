// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unused)]
#![allow(clippy::type_complexity)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
use jxl_simd::{F32Vec, F32VecArray, SimdDescriptor};

use crate::*;

#[allow(clippy::too_many_arguments)]
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub(super) fn idct_8<D: SimdDescriptor>(
    d: D,
    mut v0: F32Vec<D>,
    mut v1: F32Vec<D>,
    mut v2: F32Vec<D>,
    mut v3: F32Vec<D>,
    mut v4: F32Vec<D>,
    mut v5: F32Vec<D>,
    mut v6: F32Vec<D>,
    mut v7: F32Vec<D>,
) -> (
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
) {
    let mut v8 = v0 + v4;
    let mut v9 = v0 - v4;
    let mut v10 = v2 + v6;
    let mut v11 = v2 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v12 = v11 + v10;
    let mut v13 = v11 - v10;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v14 = v12.mul_add(mul, v8);
    let mut v15 = v12.neg_mul_add(mul, v8);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v16 = v13.mul_add(mul, v9);
    let mut v17 = v13.neg_mul_add(mul, v9);
    let mut v18 = v1 + v3;
    let mut v19 = v3 + v5;
    let mut v20 = v5 + v7;
    let mut v21 = v1 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v22 = v21 + v19;
    let mut v23 = v21 - v19;
    let mut v24 = v18 + v20;
    let mut v25 = v18 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v26 = v25 + v24;
    let mut v27 = v25 - v24;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v28 = v26.mul_add(mul, v22);
    let mut v29 = v26.neg_mul_add(mul, v22);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v30 = v27.mul_add(mul, v23);
    let mut v31 = v27.neg_mul_add(mul, v23);
    let mul = F32Vec::splat(d, 0.5097955791041592);
    let mut v32 = v28.mul_add(mul, v14);
    let mut v33 = v28.neg_mul_add(mul, v14);
    let mul = F32Vec::splat(d, 0.6013448869350453);
    let mut v34 = v30.mul_add(mul, v16);
    let mut v35 = v30.neg_mul_add(mul, v16);
    let mul = F32Vec::splat(d, 0.8999762231364156);
    let mut v36 = v31.mul_add(mul, v17);
    let mut v37 = v31.neg_mul_add(mul, v17);
    let mul = F32Vec::splat(d, 2.5629154477415055);
    let mut v38 = v29.mul_add(mul, v15);
    let mut v39 = v29.neg_mul_add(mul, v15);
    (v32, v34, v36, v38, v39, v37, v35, v33)
}

#[inline(always)]
pub(super) fn do_idct_8<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {
    assert!(data.len() > 7 * stride);
    let mut v0 = F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = F32Vec::load_array(d, &data[1 * stride]);
    let mut v2 = F32Vec::load_array(d, &data[2 * stride]);
    let mut v3 = F32Vec::load_array(d, &data[3 * stride]);
    let mut v4 = F32Vec::load_array(d, &data[4 * stride]);
    let mut v5 = F32Vec::load_array(d, &data[5 * stride]);
    let mut v6 = F32Vec::load_array(d, &data[6 * stride]);
    let mut v7 = F32Vec::load_array(d, &data[7 * stride]);
    (v0, v1, v2, v3, v4, v5, v6, v7) = idct_8(d, v0, v1, v2, v3, v4, v5, v6, v7);
    v0.store_array(&mut data[0 * stride]);
    v1.store_array(&mut data[1 * stride]);
    v2.store_array(&mut data[2 * stride]);
    v3.store_array(&mut data[3 * stride]);
    v4.store_array(&mut data[4 * stride]);
    v5.store_array(&mut data[5 * stride]);
    v6.store_array(&mut data[6 * stride]);
    v7.store_array(&mut data[7 * stride]);
}

#[inline(always)]
pub(super) fn do_idct_8_inout<D: SimdDescriptor>(
    d: D,
    src: &[F32VecArray<D>],
    src_stride: usize,
    dst: &mut [F32VecArray<D>],
    dst_stride: usize,
) {
    assert!(src.len() > 7 * src_stride);
    assert!(dst.len() > 7 * dst_stride);
    let mut v0 = F32Vec::load_array(d, &src[0 * src_stride]);
    let mut v1 = F32Vec::load_array(d, &src[1 * src_stride]);
    let mut v2 = F32Vec::load_array(d, &src[2 * src_stride]);
    let mut v3 = F32Vec::load_array(d, &src[3 * src_stride]);
    let mut v4 = F32Vec::load_array(d, &src[4 * src_stride]);
    let mut v5 = F32Vec::load_array(d, &src[5 * src_stride]);
    let mut v6 = F32Vec::load_array(d, &src[6 * src_stride]);
    let mut v7 = F32Vec::load_array(d, &src[7 * src_stride]);
    (v0, v1, v2, v3, v4, v5, v6, v7) = idct_8(d, v0, v1, v2, v3, v4, v5, v6, v7);
    v0.store_array(&mut dst[0 * dst_stride]);
    v1.store_array(&mut dst[1 * dst_stride]);
    v2.store_array(&mut dst[2 * dst_stride]);
    v3.store_array(&mut dst[3 * dst_stride]);
    v4.store_array(&mut dst[4 * dst_stride]);
    v5.store_array(&mut dst[5 * dst_stride]);
    v6.store_array(&mut dst[6 * dst_stride]);
    v7.store_array(&mut dst[7 * dst_stride]);
}

#[inline(always)]
pub(super) fn col_idct_8<D: SimdDescriptor>(d: D, data: &mut [f32], cols: usize) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let data = F32Vec::<D>::make_array_slice_mut(&mut data[..8 * cols]);
    for i in 0..chunks {
        do_idct_8(d, &mut data[i..], chunks);
    }
}

#[inline(always)]
pub(super) fn col_idct_8_inout<D: SimdDescriptor>(d: D, src: &[f32], dst: &mut [f32], cols: usize) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let src = F32Vec::<D>::make_array_slice(&src[..8 * cols]);
    let dst = F32Vec::<D>::make_array_slice_mut(&mut dst[..8 * cols]);
    for i in 0..chunks {
        do_idct_8_inout(d, &src[i..], chunks, &mut dst[i..], chunks);
    }
}
