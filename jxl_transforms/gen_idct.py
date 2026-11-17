#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys
import math

n = int(sys.argv[1])

assert n > 1
assert n <= 256
assert n & (n - 1) == 0

print(
    """\
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
"""
)

print("#![allow(unused)]")
print("#![allow(clippy::type_complexity)]")
print("#![allow(clippy::erasing_op)]")
print("#![allow(clippy::identity_op)]")

print("use jxl_simd::{F32Vec, F32VecArray, SimdDescriptor};")
print()
print("use crate::*;")

print()

variables = list(range(n))

next_var = n


def next():
    global next_var
    next_var += 1
    return next_var - 1


def wc_multiplier(i, N):
    return 1.0 / (2 * math.cos((i + 0.5) * math.pi / N))


def forward_even_odd(variables):
    n = len(variables)
    ret = [0] * len(variables)
    for i in range(n // 2):
        ret[i] = variables[2 * i]
    for i in range(n // 2, n):
        ret[i] = variables[2 * (i - n // 2) + 1]
    return ret


def b_transpose(variables):
    n = len(variables)
    ret = [0] * len(variables)
    for i in range(1, n):
        ret[i] = next()
        print("let mut v%d = v%d + v%d;" % (ret[i], variables[i - 1], variables[i]))
    ret[0] = next()
    print(
        "let mut v%d = v%d * F32Vec::splat(d, std::f32::consts::SQRT_2);"
        % (ret[0], variables[0])
    )
    return ret


def multiply_and_add(variables):
    n = len(variables)
    ret = [0] * len(variables)
    for i in range(n // 2):
        print("let mul = F32Vec::splat(d, %.16f);" % wc_multiplier(i, n))
        ret[i] = next()
        print(
            "let mut v%d = v%d.mul_add(mul, v%d);"
            % (ret[i], variables[n // 2 + i], variables[i])
        )
        ret[n - i - 1] = next()
        print(
            "let mut v%d = v%d.neg_mul_add(mul, v%d);"
            % (ret[n - i - 1], variables[n // 2 + i], variables[i])
        )
    return ret


def call_idct(variables):
    n = len(variables)
    print("(")
    for i in variables:
        print("v%d," % i)
    print(") = idct_%d(d," % n)
    for i in variables:
        print("v%d," % i)
    print(");")


def d_call_idct(variables):
    print("d.call(#[inline(always)] |_| {")
    call_idct(variables)
    print("});")


def idct(variables):
    n = len(variables)
    ret = [0] * len(variables)
    if n == 2:
        ret[0] = next()
        print("let mut v%d = v%d + v%d;" % (ret[0], variables[0], variables[1]))
        ret[1] = next()
        print("let mut v%d = v%d - v%d;" % (ret[1], variables[0], variables[1]))
    else:
        ret = forward_even_odd(variables)
        first_half = ret[: n // 2]
        second_half = ret[n // 2 :]
        first_half = idct(first_half)
        second_half = b_transpose(second_half)
        second_half = idct(second_half)
        ret = first_half + second_half
        ret = multiply_and_add(ret)

    return ret


print("#[allow(clippy::too_many_arguments)]")
print("#[allow(clippy::excessive_precision)]")
print("#[inline(always)]")
print("pub(super) fn idct_%d<D: SimdDescriptor>(d: D," % n)
for v in variables:
    print("mut v%d: F32Vec<D>," % v)
print(") -> (")
for _ in range(n):
    print("F32Vec<D>,")
print(") {")

variables = idct(variables)

print("(")

for v in variables:
    print("v%d," % v)

print(")")

print("}")


print()
print("#[inline(always)]")
print(
    "pub(super) fn do_idct_%d<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {"
    % n
)

print("assert!(data.len() > %d * stride);" % (n - 1))
for i in range(n):
    print("let mut v%d = F32Vec::load_array(d, &data[%d*stride]);" % (i, i))

call_idct(list(range(n)))

for i in range(n):
    print("v%d.store_array(&mut data[%d*stride]);" % (i, i))

print("}")

print()
print("#[inline(always)]")
print(
    "pub(super) fn do_idct_%d_inout<D: SimdDescriptor>("
    "d: D, src: &[F32VecArray<D>], src_stride: usize, dst: &mut [F32VecArray<D>], dst_stride: usize) {"
    % n
)

print("assert!(src.len() > %d * src_stride);" % (n - 1))
print("assert!(dst.len() > %d * dst_stride);" % (n - 1))
for i in range(n):
    print("let mut v%d = F32Vec::load_array(d, &src[%d*src_stride]);" % (i, i))

call_idct(list(range(n)))

for i in range(n):
    print("v%d.store_array(&mut dst[%d*dst_stride]);" % (i, i))

print("}")

print()
print("#[inline(always)]")
print(
    "pub(super) fn col_idct_%d<D: SimdDescriptor>(d: D, data: &mut [f32], cols: usize) {"
    % n
)
print("const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };")
print("assert!(cols.is_multiple_of(F32Vec::<D>::LEN));")
print("let chunks = cols / F32Vec::<D>::LEN;")
print("let data = F32Vec::<D>::make_array_slice_mut(&mut data[..%d * cols]);" % n)
print("for i in 0..chunks {")
print("do_idct_%d(d, &mut data[i..], chunks);" % n)
print("}")
print("}")

print()
print("#[inline(always)]")
print(
    "pub(super) fn col_idct_%d_inout<D: SimdDescriptor>("
    "d: D, src: &[f32], dst: &mut [f32], cols: usize) {"
    % n
)
print("const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };")
print("assert!(cols.is_multiple_of(F32Vec::<D>::LEN));")
print("let chunks = cols / F32Vec::<D>::LEN;")
print("let src = F32Vec::<D>::make_array_slice(&src[..%d * cols]);" % n)
print("let dst = F32Vec::<D>::make_array_slice_mut(&mut dst[..%d * cols]);" % n)
print("for i in 0..chunks {")
print("do_idct_%d_inout(d, &src[i..], chunks, &mut dst[i..], chunks);" % n)
print("}")
print("}")

if n == 32:
    print()
    print("#[inline(always)]")
    print(
        "pub(super) fn do_idct_32_rowblock<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>]) {"
    )
    print("assert!(data.len() >= 32);")
    print("const { assert!(32usize.is_multiple_of(F32Vec::<D>::LEN)) };")
    print("let row_stride = 32 / F32Vec::<D>::LEN;")
    for i in range(32):
        print(
            "let mut v%d = F32Vec::load_array(d, &data[row_stride * (%d %% F32Vec::<D>::LEN) + (%d / F32Vec::<D>::LEN)]);"
            % (i, i, i)
        )
    call_idct(list(range(32)))
    for i in range(32):
        print(
            "v%d.store_array(&mut data[row_stride * (%d %% F32Vec::<D>::LEN) + (%d / F32Vec::<D>::LEN)]);"
            % (i, i, i)
        )
    print("}")
