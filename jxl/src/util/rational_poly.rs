// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32Vec, SimdDescriptor};

/// Computes `(p0 + p1 x + p2 x^2 + ...) / (q0 + q1 x + q2 x^2 + ...)`.
#[inline(always)]
pub fn eval_rational_poly_simd<D: SimdDescriptor, const P: usize, const Q: usize>(
    d: D,
    x: F32Vec<D>,
    p: [f32; P],
    q: [f32; Q],
) -> F32Vec<D> {
    let mut yp = F32Vec::splat(d, p[P - 1]);
    for i in (0..P - 1).rev() {
        yp = yp.mul_add(x, F32Vec::splat(d, p[i]));
    }
    let mut yq = F32Vec::splat(d, q[Q - 1]);
    for i in (0..Q - 1).rev() {
        yq = yq.mul_add(x, F32Vec::splat(d, q[i]));
    }
    yp / yq
}
