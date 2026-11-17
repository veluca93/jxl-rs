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
pub(super) fn reinterpreting_dct_8<D: SimdDescriptor>(
    d: D,
    v0: F32Vec<D>,
    v1: F32Vec<D>,
    v2: F32Vec<D>,
    v3: F32Vec<D>,
    v4: F32Vec<D>,
    v5: F32Vec<D>,
    v6: F32Vec<D>,
    v7: F32Vec<D>,
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
    let v8 = v0 + v7;
    let v9 = v1 + v6;
    let v10 = v2 + v5;
    let v11 = v3 + v4;
    let v12 = v8 + v11;
    let v13 = v9 + v10;
    let v14 = v12 + v13;
    let v15 = v12 - v13;
    let v16 = v8 - v11;
    let v17 = v9 - v10;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let v18 = v16 * mul;
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let v19 = v17 * mul;
    let v20 = v18 + v19;
    let v21 = v18 - v19;
    let v22 = v20.mul_add(F32Vec::splat(d, std::f32::consts::SQRT_2), v21);
    let v23 = v0 - v7;
    let v24 = v1 - v6;
    let v25 = v2 - v5;
    let v26 = v3 - v4;
    let mul = F32Vec::splat(d, 0.5097955791041592);
    let v27 = v23 * mul;
    let mul = F32Vec::splat(d, 0.6013448869350453);
    let v28 = v24 * mul;
    let mul = F32Vec::splat(d, 0.8999762231364156);
    let v29 = v25 * mul;
    let mul = F32Vec::splat(d, 2.5629154477415055);
    let v30 = v26 * mul;
    let v31 = v27 + v30;
    let v32 = v28 + v29;
    let v33 = v31 + v32;
    let v34 = v31 - v32;
    let v35 = v27 - v30;
    let v36 = v28 - v29;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let v37 = v35 * mul;
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let v38 = v36 * mul;
    let v39 = v37 + v38;
    let v40 = v37 - v38;
    let v41 = v39.mul_add(F32Vec::splat(d, std::f32::consts::SQRT_2), v40);
    let v42 = v33.mul_add(F32Vec::splat(d, std::f32::consts::SQRT_2), v41);
    let v43 = v41 + v34;
    let v44 = v34 + v40;
    (
        v14 * F32Vec::splat(d, 0.125000),
        v42 * F32Vec::splat(d, 0.125794),
        v22 * F32Vec::splat(d, 0.128220),
        v43 * F32Vec::splat(d, 0.132413),
        v15 * F32Vec::splat(d, 0.138617),
        v44 * F32Vec::splat(d, 0.147222),
        v21 * F32Vec::splat(d, 0.158820),
        v40 * F32Vec::splat(d, 0.174311),
    )
}

#[inline(always)]
pub(super) fn do_reinterpreting_dct_8<D: SimdDescriptor>(
    d: D,
    data: &mut [F32VecArray<D>],
    stride: usize,
) {
    assert!(data.len() > 7 * stride);
    let mut v0 = F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = F32Vec::load_array(d, &data[1 * stride]);
    let mut v2 = F32Vec::load_array(d, &data[2 * stride]);
    let mut v3 = F32Vec::load_array(d, &data[3 * stride]);
    let mut v4 = F32Vec::load_array(d, &data[4 * stride]);
    let mut v5 = F32Vec::load_array(d, &data[5 * stride]);
    let mut v6 = F32Vec::load_array(d, &data[6 * stride]);
    let mut v7 = F32Vec::load_array(d, &data[7 * stride]);
    (v0, v1, v2, v3, v4, v5, v6, v7) = reinterpreting_dct_8(d, v0, v1, v2, v3, v4, v5, v6, v7);
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
pub(super) fn col_reinterpreting_dct_8<D: SimdDescriptor>(
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
    let len = 7 * row_stride + cols;
    let data = F32Vec::<D>::make_array_slice_mut(&mut data[..len]);
    for i in 0..col_chunks {
        do_reinterpreting_dct_8(d, &mut data[i..], stride_chunks);
    }
}
