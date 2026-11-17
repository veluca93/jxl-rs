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
pub(super) fn idct_16<D: SimdDescriptor>(
    d: D,
    mut v0: F32Vec<D>,
    mut v1: F32Vec<D>,
    mut v2: F32Vec<D>,
    mut v3: F32Vec<D>,
    mut v4: F32Vec<D>,
    mut v5: F32Vec<D>,
    mut v6: F32Vec<D>,
    mut v7: F32Vec<D>,
    mut v8: F32Vec<D>,
    mut v9: F32Vec<D>,
    mut v10: F32Vec<D>,
    mut v11: F32Vec<D>,
    mut v12: F32Vec<D>,
    mut v13: F32Vec<D>,
    mut v14: F32Vec<D>,
    mut v15: F32Vec<D>,
) -> (
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
    F32Vec<D>,
) {
    let mut v16 = v0 + v8;
    let mut v17 = v0 - v8;
    let mut v18 = v4 + v12;
    let mut v19 = v4 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v20 = v19 + v18;
    let mut v21 = v19 - v18;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v22 = v20.mul_add(mul, v16);
    let mut v23 = v20.neg_mul_add(mul, v16);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v24 = v21.mul_add(mul, v17);
    let mut v25 = v21.neg_mul_add(mul, v17);
    let mut v26 = v2 + v6;
    let mut v27 = v6 + v10;
    let mut v28 = v10 + v14;
    let mut v29 = v2 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v30 = v29 + v27;
    let mut v31 = v29 - v27;
    let mut v32 = v26 + v28;
    let mut v33 = v26 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v34 = v33 + v32;
    let mut v35 = v33 - v32;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v36 = v34.mul_add(mul, v30);
    let mut v37 = v34.neg_mul_add(mul, v30);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v38 = v35.mul_add(mul, v31);
    let mut v39 = v35.neg_mul_add(mul, v31);
    let mul = F32Vec::splat(d, 0.5097955791041592);
    let mut v40 = v36.mul_add(mul, v22);
    let mut v41 = v36.neg_mul_add(mul, v22);
    let mul = F32Vec::splat(d, 0.6013448869350453);
    let mut v42 = v38.mul_add(mul, v24);
    let mut v43 = v38.neg_mul_add(mul, v24);
    let mul = F32Vec::splat(d, 0.8999762231364156);
    let mut v44 = v39.mul_add(mul, v25);
    let mut v45 = v39.neg_mul_add(mul, v25);
    let mul = F32Vec::splat(d, 2.5629154477415055);
    let mut v46 = v37.mul_add(mul, v23);
    let mut v47 = v37.neg_mul_add(mul, v23);
    let mut v48 = v1 + v3;
    let mut v49 = v3 + v5;
    let mut v50 = v5 + v7;
    let mut v51 = v7 + v9;
    let mut v52 = v9 + v11;
    let mut v53 = v11 + v13;
    let mut v54 = v13 + v15;
    let mut v55 = v1 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v56 = v55 + v51;
    let mut v57 = v55 - v51;
    let mut v58 = v49 + v53;
    let mut v59 = v49 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v60 = v59 + v58;
    let mut v61 = v59 - v58;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v62 = v60.mul_add(mul, v56);
    let mut v63 = v60.neg_mul_add(mul, v56);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v64 = v61.mul_add(mul, v57);
    let mut v65 = v61.neg_mul_add(mul, v57);
    let mut v66 = v48 + v50;
    let mut v67 = v50 + v52;
    let mut v68 = v52 + v54;
    let mut v69 = v48 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v70 = v69 + v67;
    let mut v71 = v69 - v67;
    let mut v72 = v66 + v68;
    let mut v73 = v66 * F32Vec::splat(d, std::f32::consts::SQRT_2);
    let mut v74 = v73 + v72;
    let mut v75 = v73 - v72;
    let mul = F32Vec::splat(d, 0.5411961001461970);
    let mut v76 = v74.mul_add(mul, v70);
    let mut v77 = v74.neg_mul_add(mul, v70);
    let mul = F32Vec::splat(d, 1.3065629648763764);
    let mut v78 = v75.mul_add(mul, v71);
    let mut v79 = v75.neg_mul_add(mul, v71);
    let mul = F32Vec::splat(d, 0.5097955791041592);
    let mut v80 = v76.mul_add(mul, v62);
    let mut v81 = v76.neg_mul_add(mul, v62);
    let mul = F32Vec::splat(d, 0.6013448869350453);
    let mut v82 = v78.mul_add(mul, v64);
    let mut v83 = v78.neg_mul_add(mul, v64);
    let mul = F32Vec::splat(d, 0.8999762231364156);
    let mut v84 = v79.mul_add(mul, v65);
    let mut v85 = v79.neg_mul_add(mul, v65);
    let mul = F32Vec::splat(d, 2.5629154477415055);
    let mut v86 = v77.mul_add(mul, v63);
    let mut v87 = v77.neg_mul_add(mul, v63);
    let mul = F32Vec::splat(d, 0.5024192861881557);
    let mut v88 = v80.mul_add(mul, v40);
    let mut v89 = v80.neg_mul_add(mul, v40);
    let mul = F32Vec::splat(d, 0.5224986149396889);
    let mut v90 = v82.mul_add(mul, v42);
    let mut v91 = v82.neg_mul_add(mul, v42);
    let mul = F32Vec::splat(d, 0.5669440348163577);
    let mut v92 = v84.mul_add(mul, v44);
    let mut v93 = v84.neg_mul_add(mul, v44);
    let mul = F32Vec::splat(d, 0.6468217833599901);
    let mut v94 = v86.mul_add(mul, v46);
    let mut v95 = v86.neg_mul_add(mul, v46);
    let mul = F32Vec::splat(d, 0.7881546234512502);
    let mut v96 = v87.mul_add(mul, v47);
    let mut v97 = v87.neg_mul_add(mul, v47);
    let mul = F32Vec::splat(d, 1.0606776859903471);
    let mut v98 = v85.mul_add(mul, v45);
    let mut v99 = v85.neg_mul_add(mul, v45);
    let mul = F32Vec::splat(d, 1.7224470982383342);
    let mut v100 = v83.mul_add(mul, v43);
    let mut v101 = v83.neg_mul_add(mul, v43);
    let mul = F32Vec::splat(d, 5.1011486186891553);
    let mut v102 = v81.mul_add(mul, v41);
    let mut v103 = v81.neg_mul_add(mul, v41);
    (
        v88, v90, v92, v94, v96, v98, v100, v102, v103, v101, v99, v97, v95, v93, v91, v89,
    )
}

#[inline(always)]
pub(super) fn do_idct_16<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {
    assert!(data.len() > 15 * stride);
    let mut v0 = F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = F32Vec::load_array(d, &data[1 * stride]);
    let mut v2 = F32Vec::load_array(d, &data[2 * stride]);
    let mut v3 = F32Vec::load_array(d, &data[3 * stride]);
    let mut v4 = F32Vec::load_array(d, &data[4 * stride]);
    let mut v5 = F32Vec::load_array(d, &data[5 * stride]);
    let mut v6 = F32Vec::load_array(d, &data[6 * stride]);
    let mut v7 = F32Vec::load_array(d, &data[7 * stride]);
    let mut v8 = F32Vec::load_array(d, &data[8 * stride]);
    let mut v9 = F32Vec::load_array(d, &data[9 * stride]);
    let mut v10 = F32Vec::load_array(d, &data[10 * stride]);
    let mut v11 = F32Vec::load_array(d, &data[11 * stride]);
    let mut v12 = F32Vec::load_array(d, &data[12 * stride]);
    let mut v13 = F32Vec::load_array(d, &data[13 * stride]);
    let mut v14 = F32Vec::load_array(d, &data[14 * stride]);
    let mut v15 = F32Vec::load_array(d, &data[15 * stride]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ) = idct_16(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    );
    v0.store_array(&mut data[0 * stride]);
    v1.store_array(&mut data[1 * stride]);
    v2.store_array(&mut data[2 * stride]);
    v3.store_array(&mut data[3 * stride]);
    v4.store_array(&mut data[4 * stride]);
    v5.store_array(&mut data[5 * stride]);
    v6.store_array(&mut data[6 * stride]);
    v7.store_array(&mut data[7 * stride]);
    v8.store_array(&mut data[8 * stride]);
    v9.store_array(&mut data[9 * stride]);
    v10.store_array(&mut data[10 * stride]);
    v11.store_array(&mut data[11 * stride]);
    v12.store_array(&mut data[12 * stride]);
    v13.store_array(&mut data[13 * stride]);
    v14.store_array(&mut data[14 * stride]);
    v15.store_array(&mut data[15 * stride]);
}

#[inline(always)]
pub(super) fn do_idct_16_inout<D: SimdDescriptor>(
    d: D,
    src: &[F32VecArray<D>],
    src_stride: usize,
    dst: &mut [F32VecArray<D>],
    dst_stride: usize,
) {
    assert!(src.len() > 15 * src_stride);
    assert!(dst.len() > 15 * dst_stride);
    let mut v0 = F32Vec::load_array(d, &src[0 * src_stride]);
    let mut v1 = F32Vec::load_array(d, &src[1 * src_stride]);
    let mut v2 = F32Vec::load_array(d, &src[2 * src_stride]);
    let mut v3 = F32Vec::load_array(d, &src[3 * src_stride]);
    let mut v4 = F32Vec::load_array(d, &src[4 * src_stride]);
    let mut v5 = F32Vec::load_array(d, &src[5 * src_stride]);
    let mut v6 = F32Vec::load_array(d, &src[6 * src_stride]);
    let mut v7 = F32Vec::load_array(d, &src[7 * src_stride]);
    let mut v8 = F32Vec::load_array(d, &src[8 * src_stride]);
    let mut v9 = F32Vec::load_array(d, &src[9 * src_stride]);
    let mut v10 = F32Vec::load_array(d, &src[10 * src_stride]);
    let mut v11 = F32Vec::load_array(d, &src[11 * src_stride]);
    let mut v12 = F32Vec::load_array(d, &src[12 * src_stride]);
    let mut v13 = F32Vec::load_array(d, &src[13 * src_stride]);
    let mut v14 = F32Vec::load_array(d, &src[14 * src_stride]);
    let mut v15 = F32Vec::load_array(d, &src[15 * src_stride]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    ) = idct_16(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    );
    v0.store_array(&mut dst[0 * dst_stride]);
    v1.store_array(&mut dst[1 * dst_stride]);
    v2.store_array(&mut dst[2 * dst_stride]);
    v3.store_array(&mut dst[3 * dst_stride]);
    v4.store_array(&mut dst[4 * dst_stride]);
    v5.store_array(&mut dst[5 * dst_stride]);
    v6.store_array(&mut dst[6 * dst_stride]);
    v7.store_array(&mut dst[7 * dst_stride]);
    v8.store_array(&mut dst[8 * dst_stride]);
    v9.store_array(&mut dst[9 * dst_stride]);
    v10.store_array(&mut dst[10 * dst_stride]);
    v11.store_array(&mut dst[11 * dst_stride]);
    v12.store_array(&mut dst[12 * dst_stride]);
    v13.store_array(&mut dst[13 * dst_stride]);
    v14.store_array(&mut dst[14 * dst_stride]);
    v15.store_array(&mut dst[15 * dst_stride]);
}

#[inline(always)]
pub(super) fn col_idct_16<D: SimdDescriptor>(d: D, data: &mut [f32], cols: usize) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let data = F32Vec::<D>::make_array_slice_mut(&mut data[..16 * cols]);
    for i in 0..chunks {
        do_idct_16(d, &mut data[i..], chunks);
    }
}

#[inline(always)]
pub(super) fn col_idct_16_inout<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    dst: &mut [f32],
    cols: usize,
) {
    const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };
    assert!(cols.is_multiple_of(F32Vec::<D>::LEN));
    let chunks = cols / F32Vec::<D>::LEN;
    let src = F32Vec::<D>::make_array_slice(&src[..16 * cols]);
    let dst = F32Vec::<D>::make_array_slice_mut(&mut dst[..16 * cols]);
    for i in 0..chunks {
        do_idct_16_inout(d, &src[i..], chunks, &mut dst[i..], chunks);
    }
}
