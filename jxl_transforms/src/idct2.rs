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
pub(super) fn idct_2<D: SimdDescriptor>(
    d: D,
    mut v0: F32Vec<D>,
    mut v1: F32Vec<D>,
) -> (F32Vec<D>, F32Vec<D>) {
    let mut v2 = v0 + v1;
    let mut v3 = v0 - v1;
    (v2, v3)
}

#[inline(always)]
pub(super) fn do_idct_2<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {
    assert!(data.len() > 1 * stride);
    let mut v0 = F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = F32Vec::load_array(d, &data[1 * stride]);
    (v0, v1) = idct_2(d, v0, v1);
    v0.store_array(&mut data[0 * stride]);
    v1.store_array(&mut data[1 * stride]);
}

#[inline(always)]
pub(super) fn do_idct_2_inout<D: SimdDescriptor>(
    d: D,
    src: &[F32VecArray<D>],
    src_stride: usize,
    dst: &mut [F32VecArray<D>],
    dst_stride: usize,
) {
    assert!(src.len() > 1 * src_stride);
    assert!(dst.len() > 1 * dst_stride);
    let mut v0 = F32Vec::load_array(d, &src[0 * src_stride]);
    let mut v1 = F32Vec::load_array(d, &src[1 * src_stride]);
    (v0, v1) = idct_2(d, v0, v1);
    v0.store_array(&mut dst[0 * dst_stride]);
    v1.store_array(&mut dst[1 * dst_stride]);
}

#[inline(always)]
pub(super) fn col_idct_2<D: SimdDescriptor>(d: D, data: &mut [f32], cols: usize) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let data = F32Vec::<D>::make_array_slice_mut(&mut data[..2 * cols]);
    for i in 0..chunks {
        do_idct_2(d, &mut data[i..], chunks);
    }
}

#[inline(always)]
pub(super) fn col_idct_2_inout<D: SimdDescriptor>(d: D, src: &[f32], dst: &mut [f32], cols: usize) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let src = F32Vec::<D>::make_array_slice(&src[..2 * cols]);
    let dst = F32Vec::<D>::make_array_slice_mut(&mut dst[..2 * cols]);
    for i in 0..chunks {
        do_idct_2_inout(d, &src[i..], chunks, &mut dst[i..], chunks);
    }
}
