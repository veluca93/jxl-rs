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
    (2, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 8),
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
    if ROWS < 4 or COLS < 4:
        assert ROWS == 2 and COLS == 2
        print("#[inline(always)]")
        print(
            "fn idct2d_%d_%d_impl<D: SimdDescriptor128>(d: D, data: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
        print("const { assert!(F32Vec::<D>::LEN == 4) };")
        print("let mut buf = [0.0f32; 16];")
        print("buf[0] = data[0];")
        print("buf[1] = data[1];")
        print("buf[4] = data[2];")
        print("buf[5] = data[3];")
        print("do_idct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);")
        print("crate::transpose::transpose_in_place(d, &mut buf, 4);")
        print("do_idct_2(d, F32Vec::<D>::make_array_slice_mut(&mut buf), 1);")
        print("data[0] = buf[0];")
        print("data[1] = buf[1];")
        print("data[2] = buf[4];")
        print("data[3] = buf[5];")
        print("}")
        print()
        print("#[inline(always)]")
        print(
            "pub fn idct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print("idct2d_%d_%d_impl(d.maybe_downgrade_128bit(), data)" % (ROWS, COLS))
        print("}")
    else:
        already_128 = ROWS < 8 or COLS < 8
        d_for = lambda w: "d" if already_128 else descr_for_width(w)
        print("#[inline(always)]")
        print(
            "pub fn idct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut [f32]) {"
            % (ROWS, COLS)
        )
        print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
        if already_128:
            print("let d = d.maybe_downgrade_128bit();")
        if ROWS == COLS:
            d_n = d_for(ROWS)
            print("let d_col = %s;" % d_n)
            print("col_idct_%d(d_col, data, %d);" % (ROWS, ROWS))
            print("crate::transpose::transpose_in_place(d, data, %d);" % ROWS)
            print("col_idct_%d(d_col, data, %d);" % (ROWS, ROWS))
        elif ROWS == 16 and COLS == 32:
            print("let data = &mut data[..512];")
            print("let len = F32Vec::<D>::LEN;")
            print("let column_chunks = 32 / len;")
            print("let row_chunks = 16 / len;")
            print("for i in 0..row_chunks {")
            print("for j in 0..column_chunks {")
            print(
                "crate::transpose::transpose_in_place_strided(d, &mut data[(i * 32 + j) * len..], len, 32);"
            )
            print("}")
            print("let data_arr = F32Vec::<D>::make_array_slice_mut(data);")
            print("do_idct_32_rowblock(d, &mut data_arr[i * 32..]);")
            print("}")
            print("for i in 0..column_chunks {")
            print("for j in 0..row_chunks {")
            print(
                "crate::transpose::transpose_in_place_strided(d, &mut data[(j * 32 + i) * len..], len, 32);"
            )
            print("}")
            print("let data_arr = F32Vec::<D>::make_array_slice_mut(data);")
            print("do_idct_16(d, &mut data_arr[i..], column_chunks);")
            print("}")
        elif ROWS < COLS:
            print("let mut tmp = [0.0f32; %d];" % SZ)
            print("crate::transpose::transpose(d, data, &mut tmp, %d, %d);" % (ROWS, COLS))
            print("col_idct_%d(%s, &mut tmp, %d);" % (COLS, d_for(ROWS), ROWS))
            print("crate::transpose::transpose(d, &tmp, data, %d, %d);" % (COLS, ROWS))
            print("col_idct_%d(%s, data, %d);" % (ROWS, d_for(COLS), COLS))
        else:
            # Note: when ROWS > COLS, the input is stored as COLS rows of ROWS columns.
            print("col_idct_%d(%s, data, %d);" % (COLS, d_for(ROWS), ROWS))
            print("let mut tmp = [0.0f32; %d];" % SZ)
            print("crate::transpose::transpose(d, data, &mut tmp, %d, %d);" % (COLS, ROWS))
            print(
                "col_idct_%d_inout(%s, &tmp, data, %d);"
                % (ROWS, d_for(COLS), COLS)
            )
        print("}")
