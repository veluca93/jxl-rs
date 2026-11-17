#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys
import math

n = int(sys.argv[1])

assert n > 1
assert n <= 32
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

variables = list(range(n))

next_var = n


def next():
    global next_var
    next_var += 1
    return next_var - 1


def wc_multiplier(i, N):
    return 1.0 / (2 * math.cos((i + 0.5) * math.pi / N))


def resampling_scale(i, N):
    return (
        math.cos(i / (16 * N) * math.pi)
        * math.cos(i / (8 * N) * math.pi)
        * math.cos(i / (4 * N) * math.pi)
        * N
    )


def inverse_even_odd(variables):
    n = len(variables)
    ret = [0] * len(variables)
    for i in range(n // 2):
        ret[2 * i] = variables[i]
    for i in range(n // 2):
        ret[2 * i + 1] = variables[n // 2 + i]
    return ret


def addsub_reverse(variables, op):
    n = len(variables)
    ret = [0] * (n // 2)
    for i in range(n // 2):
        ret[i] = next()
        print(
            "let v%d = v%d %s v%d;" % (ret[i], variables[i], op, variables[n - i - 1])
        )
    return ret


def b(variables):
    n = len(variables)
    ret = [0] * len(variables)
    ret[0] = next()
    print(
        "let v%d = v%d.mul_add(F32Vec::splat(d, std::f32::consts::SQRT_2), v%d);"
        % (ret[0], variables[0], variables[1])
    )
    for i in range(1, n - 1):
        ret[i] = next()
        print("let v%d = v%d + v%d;" % (ret[i], variables[i], variables[i + 1]))
    ret[n - 1] = variables[n - 1]
    return ret


def multiply(variables):
    n = len(variables)
    ret = [x for x in variables]
    for i in range(n):
        print("let mul = F32Vec::splat(d, %.16f);" % wc_multiplier(i, 2 * n))
        ret[i] = next()
        print("let v%d = v%d * mul;" % (ret[i], variables[i]))
    return ret


def call_reinterpreting_dct(variables):
    n = len(variables)
    print("(")
    for i in variables:
        print("v%d," % i)
    print(") = reinterpreting_dct_%d(d," % n)
    for i in variables:
        print("v%d," % i)
    print(");")


def dct(variables):
    n = len(variables)
    ret = [0] * len(variables)
    if n == 2:
        ret[0] = next()
        print("let v%d = v%d + v%d;" % (ret[0], variables[0], variables[1]))
        ret[1] = next()
        print("let v%d = v%d - v%d;" % (ret[1], variables[0], variables[1]))
    else:
        first_half = addsub_reverse(variables, "+")
        first_half = dct(first_half)
        second_half = addsub_reverse(variables, "-")
        second_half = multiply(second_half)
        second_half = dct(second_half)
        second_half = b(second_half)
        ret = first_half + second_half
        ret = inverse_even_odd(ret)

    return ret


print("#[allow(clippy::too_many_arguments)]")
print("#[allow(clippy::excessive_precision)]")
print("#[inline(always)]")
print("pub(super) fn reinterpreting_dct_%d<D: SimdDescriptor>(d: D," % n)
for v in variables:
    print("v%d: F32Vec<D>," % v)
print(") -> (")
for _ in range(n):
    print("F32Vec<D>,")
print(") {")

variables = dct(variables)

print("(")

for i, v in enumerate(variables):
    print("v%d * F32Vec::splat(d, %f)," % (v, 1.0 / resampling_scale(i, n)))

print(")")

print("}")


print()
print("#[inline(always)]")
print(
    "pub(super) fn do_reinterpreting_dct_%d<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {"
    % n
)

print("assert!(data.len() > %d * stride);" % (n - 1))
for i in range(n):
    print("let mut v%d = F32Vec::load_array(d, &data[%d*stride]);" % (i, i))

call_reinterpreting_dct(list(range(n)))

for i in range(n):
    print("v%d.store_array(&mut data[%d*stride]);" % (i, i))

print("}")

print()
print("#[inline(always)]")
print(
    "pub(super) fn col_reinterpreting_dct_%d<D: SimdDescriptor>("
    "d: D, data: &mut [f32], cols: usize, row_stride: usize) {"
    % n
)
print("const { assert!(F32Vec::<D>::LEN.is_power_of_two()) };")
print("assert!(cols.is_multiple_of(F32Vec::<D>::LEN));")
print("assert!(row_stride.is_multiple_of(F32Vec::<D>::LEN));")
print("let col_chunks = cols / F32Vec::<D>::LEN;")
print("let stride_chunks = row_stride / F32Vec::<D>::LEN;")
print("let len = %d * row_stride + cols;" % (n - 1))
print("let data = F32Vec::<D>::make_array_slice_mut(&mut data[..len]);")
print("for i in 0..col_chunks {")
print("do_reinterpreting_dct_%d(d, &mut data[i..], stride_chunks);" % n)
print("}")
print("}")
