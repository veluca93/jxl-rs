#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

HEADER = """\
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
"""

SIZES = [
    (1, 2),
    (2, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
]

print(HEADER)

print("use jxl_simd::{F32Vec, SimdDescriptor, SimdDescriptor128};")
print()
print("use crate::*;")


def descr_for_width(w):
    if w < 8:
        return "d.maybe_downgrade_128bit()"
    if w < 16:
        return "d.maybe_downgrade_256bit()"
    return "d"


for ROWS, COLS in SIZES:
    print()
    SZ = ROWS * COLS
    L = min(ROWS, COLS)
    S = max(ROWS, COLS)
    OUT_STRIDE = S * 8
    if ROWS < 4 or COLS < 4:
        print("#[inline(always)]")
        print(
            "fn reinterpreting_dct2d_%d_%d_impl<D: SimdDescriptor128>(d: D, data: &mut [f32], output: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
        print("assert!(output.len() > %d);" % ((L - 1) * OUT_STRIDE + S - 1))
        print("const { assert!(F32Vec::<D>::LEN == 4) };")
        if ROWS == 1 or COLS == 1:
            print("let mut buf = [0.0f32; %d];" % (S * 4))
            print("for i in 0..%d {" % S)
            print("buf[i * 4] = data[i];")
            print("}")
            print("{")
            print("let buf = F32Vec::<D>::make_array_slice_mut(&mut buf);")
            print("do_reinterpreting_dct_%d(d, buf, 1);" % S)
            print("}")
            print("for x in 0..%d {" % S)
            print("output[x] = buf[x * 4];")
            print("}")
        else:
            print("let mut buf = [0.0f32; 16];")
            print("for r in 0..%d {" % ROWS)
            print(
                "buf[r * 4..r * 4 + %d].copy_from_slice(&data[r * %d..r * %d + %d]);"
                % (COLS, COLS, COLS, COLS)
            )
            print("}")
            if ROWS < COLS:
                print("crate::transpose::transpose_in_place(d, &mut buf, 4);")
                print("do_reinterpreting_dct_%d(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);" % COLS)
                print("crate::transpose::transpose_in_place(d, &mut buf, 4);")
                print("do_reinterpreting_dct_%d(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);" % ROWS)
            else:
                print("do_reinterpreting_dct_%d(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);" % ROWS)
                print("crate::transpose::transpose_in_place(d, &mut buf, 4);")
                print("do_reinterpreting_dct_%d(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);" % COLS)
            print("for y in 0..%d {" % L)
            print("for x in 0..%d {" % S)
            print("output[y * %d + x] = buf[y * 4 + x];" % OUT_STRIDE)
            print("}")
            print("}")
        print("}")
        print()
        print("#[inline(always)]")
        print(
            "pub fn reinterpreting_dct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print(
            "reinterpreting_dct2d_%d_%d_impl(d.maybe_downgrade_128bit(), data, output)"
            % (ROWS, COLS)
        )
        print("}")
    else:
        print("#[inline(always)]")
        print(
            "pub fn reinterpreting_dct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut [f32], output: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
        print("assert!(output.len() > %d);" % ((L - 1) * OUT_STRIDE + S - 1))
        if ROWS == 32 and COLS == 32:
            print("col_reinterpreting_dct_32(d, data, 32, 32);")
            print("let chunks = 32 / F32Vec::<D>::LEN;")
            print("for i in 0..chunks {")
            print("crate::transpose::transpose_in_place_step(d, data, 32, i);")
            print("let data_arr = F32Vec::<D>::make_array_slice_mut(&mut data[..1024]);")
            print("do_reinterpreting_dct_32(d, &mut data_arr[i..], chunks);")
            print("}")
            print("let data = &data[..1024];")
            print("let output = &mut output[..31 * 256 + 32];")
            print("let len = F32Vec::<D>::LEN;")
            print("for y in 0..32 {")
            print("for x in (0..32).step_by(len) {")
            print(
                "F32Vec::load(d, &data[y * 32 + x..]).store(&mut output[y * 256 + x..]);"
            )
            print("}")
            print("}")
        elif ROWS >= COLS:
            print(
                "col_reinterpreting_dct_%d(%s, data, %d, %d);"
                % (ROWS, descr_for_width(COLS), COLS, COLS)
            )
            print(
                "crate::transpose::transpose_into_strided(d, data, output, %d, %d, %d);"
                % (ROWS, COLS, OUT_STRIDE)
            )
            print(
                "col_reinterpreting_dct_%d(%s, output, %d, %d);"
                % (COLS, descr_for_width(ROWS), ROWS, OUT_STRIDE)
            )
        else:
            print(
                "col_reinterpreting_dct_%d(%s, data, %d, %d);"
                % (ROWS, descr_for_width(COLS), COLS, COLS)
            )
            print("let mut tmp = [0.0f32; %d];" % SZ)
            print(
                "crate::transpose::transpose(d, data, &mut tmp, %d, %d);"
                % (ROWS, COLS)
            )
            print(
                "col_reinterpreting_dct_%d(%s, &mut tmp, %d, %d);"
                % (COLS, descr_for_width(ROWS), ROWS, ROWS)
            )
            print(
                "crate::transpose::transpose_into_strided(d, &tmp, output, %d, %d, %d);"
                % (COLS, ROWS, OUT_STRIDE)
            )
        print("}")
