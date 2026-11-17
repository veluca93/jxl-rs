// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![forbid(unsafe_code)]

use jxl_simd::{F32Vec, SimdDescriptor, SimdDescriptor128, SimdElement, SimdVec};

#[inline(always)]
fn check_transpose_bounds(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    dst_stride: usize,
) {
    assert!(src.len() >= rows * cols);
    assert!(dst_stride >= rows);
    if rows > 0 && cols > 0 {
        assert!(dst.len() >= (cols - 1) * dst_stride + rows);
    }
}

#[inline(always)]
fn transpose_rect_scalar(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    const BLOCK: usize = 16;
    if rows >= BLOCK && cols >= BLOCK && (rows > 64 || cols > 64) {
        for cb in (0..cols).step_by(BLOCK) {
            let c_end = (cb + BLOCK).min(cols);
            for rb in (0..rows).step_by(BLOCK) {
                let r_end = (rb + BLOCK).min(rows);
                for r in rb..r_end {
                    let src_row = &src[r * src_stride + cb..][..c_end - cb];
                    for (dc, &v) in src_row.iter().enumerate() {
                        dst[(cb + dc) * dst_stride + r] = v;
                    }
                }
            }
        }
    } else {
        for r in 0..rows {
            let src_row = &src[r * src_stride..][..cols];
            for (c, &v) in src_row.iter().enumerate() {
                dst[c * dst_stride + r] = v;
            }
        }
    }
}

#[inline(always)]
fn transpose_4_inlane<T: SimdElement, D: SimdDescriptor>(
    r: [SimdVec<T, D>; 4],
) -> [SimdVec<T, D>; 4] {
    let [r0, r1, r2, r3] = r;
    let t0 = SimdVec::interleave_lo(r0, r2);
    let t2 = SimdVec::interleave_hi(r0, r2);
    let t1 = SimdVec::interleave_lo(r1, r3);
    let t3 = SimdVec::interleave_hi(r1, r3);
    [
        SimdVec::interleave_lo(t0, t1),
        SimdVec::interleave_hi(t0, t1),
        SimdVec::interleave_lo(t2, t3),
        SimdVec::interleave_hi(t2, t3),
    ]
}

macro_rules! def_tile_io {
    ($load:ident, $store:ident, $k:literal, [$($idx:literal),+]) => {
        #[inline(always)]
        fn $load<T: SimdElement, D: SimdDescriptor>(
            d: D,
            src: &[T],
            stride: usize,
        ) -> [SimdVec<T, D>; $k] {
            let len = SimdVec::<T, D>::LEN;
            assert!(stride <= isize::MAX as usize / (std::mem::size_of::<T>() * len).max(1));
            let s = &src[..($k - 1) * stride + len];
            [$(SimdVec::load(d, &s[$idx * stride..])),+]
        }

        #[inline(always)]
        fn $store<T: SimdElement, D: SimdDescriptor>(
            dst: &mut [T],
            stride: usize,
            cols: [SimdVec<T, D>; $k],
        ) {
            let len = SimdVec::<T, D>::LEN;
            assert!(stride <= isize::MAX as usize / (std::mem::size_of::<T>() * len).max(1));
            let dst = &mut dst[..($k - 1) * stride + len];
            $(cols[$idx].store(&mut dst[$idx * stride..]);)+
        }
    };
}

def_tile_io!(load_tile_4, store_tile_4, 4, [0, 1, 2, 3]);

#[inline(always)]
fn load_regs<T: SimdElement, D: SimdDescriptor, const LEN: usize>(
    d: D,
    src: &[T],
    stride: usize,
) -> [SimdVec<T, D>; LEN] {
    assert!(stride <= isize::MAX as usize / (std::mem::size_of::<T>() * LEN).max(1));
    let s = &src[..(LEN - 1) * stride + LEN];
    let mut regs = [SimdVec::zero(d); LEN];
    for (i, reg) in regs.iter_mut().enumerate() {
        *reg = SimdVec::load(d, &s[i * stride..]);
    }
    regs
}

#[inline(always)]
fn store_regs<T: SimdElement, D: SimdDescriptor, const LEN: usize>(
    dst: &mut [T],
    stride: usize,
    regs: [SimdVec<T, D>; LEN],
) {
    assert!(stride <= isize::MAX as usize / (std::mem::size_of::<T>() * LEN).max(1));
    let d_out = &mut dst[..(LEN - 1) * stride + LEN];
    for (i, reg) in regs.into_iter().enumerate() {
        reg.store(&mut d_out[i * stride..]);
    }
}

#[inline(always)]
fn transpose_regs<T: SimdElement, D: SimdDescriptor, const LEN: usize, const K: usize>(
    mut regs: [SimdVec<T, D>; LEN],
) -> [SimdVec<T, D>; LEN] {
    macro_rules! stage {
        ($half:literal) => {
            if K >= 2 * $half {
                for g in (0..LEN).step_by(K) {
                    for c in (0..K).step_by(2 * $half) {
                        for i in 0..$half {
                            let (a, b) = (regs[g + c + i], regs[g + c + $half + i]);
                            regs[g + c + i] = SimdVec::interleave_lo(a, b);
                            regs[g + c + $half + i] = SimdVec::interleave_hi(a, b);
                        }
                    }
                }
            }
        };
    }
    stage!(8);
    stage!(4);
    stage!(2);
    stage!(1);
    if D::BYTES == 32 {
        for i in 0..K {
            let (o0, o1) = regs[i].interleave_blocks_128(regs[K + i]);
            regs[i] = o0;
            regs[K + i] = o1;
        }
    } else if D::BYTES == 64 {
        for i in 0..K {
            let (p0, p2) = regs[i].interleave_blocks_128(regs[K + i]);
            let (p1, p3) = regs[2 * K + i].interleave_blocks_128(regs[3 * K + i]);
            let (o0, o2) = p0.interleave_blocks_256(p1);
            let (o1, o3) = p2.interleave_blocks_256(p3);
            regs[i] = o0;
            regs[K + i] = o1;
            regs[2 * K + i] = o2;
            regs[3 * K + i] = o3;
        }
    }
    regs
}

macro_rules! dispatch_len_k {
    ($t:ty, $d:ty, |$l:ident, $k:ident| $body:block) => {{
        match (SimdVec::<$t, $d>::LEN, 16 / std::mem::size_of::<$t>()) {
            (2, 2) => {
                const $l: usize = 2;
                const $k: usize = 2;
                $body
            }
            (4, 2) => {
                const $l: usize = 4;
                const $k: usize = 2;
                $body
            }
            (4, 4) => {
                const $l: usize = 4;
                const $k: usize = 4;
                $body
            }
            (8, 2) => {
                const $l: usize = 8;
                const $k: usize = 2;
                $body
            }
            (8, 4) => {
                const $l: usize = 8;
                const $k: usize = 4;
                $body
            }
            (8, 8) => {
                const $l: usize = 8;
                const $k: usize = 8;
                $body
            }
            (16, 4) => {
                const $l: usize = 16;
                const $k: usize = 4;
                $body
            }
            (16, 8) => {
                const $l: usize = 16;
                const $k: usize = 8;
                $body
            }
            (16, 16) => {
                const $l: usize = 16;
                const $k: usize = 16;
                $body
            }
            (32, 8) => {
                const $l: usize = 32;
                const $k: usize = 8;
                $body
            }
            (32, 16) => {
                const $l: usize = 32;
                const $k: usize = 16;
                $body
            }
            (64, 16) => {
                const $l: usize = 64;
                const $k: usize = 16;
                $body
            }
            _ => unreachable!(),
        }
    }};
}

#[inline(always)]
fn transpose_vec_tile<T: SimdElement, D: SimdDescriptor>(
    d: D,
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
) {
    dispatch_len_k!(T, D, |LEN, K| {
        let tr = transpose_regs::<T, D, LEN, K>(load_regs::<T, D, LEN>(d, src, src_stride));
        store_regs(dst, dst_stride, tr);
    });
}

#[inline(always)]
fn transpose_vec_diag<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    stride: usize,
    tile_idx: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    let off = tile_idx * len * stride + tile_idx * len;
    dispatch_len_k!(T, D, |LEN, K| {
        let tr = transpose_regs::<T, D, LEN, K>(load_regs::<T, D, LEN>(d, &data[off..], stride));
        store_regs(&mut data[off..], stride, tr);
    });
}

#[inline(always)]
fn transpose_vec_swap<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    stride: usize,
    ti: usize,
    tj: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    let r_base = ti * len;
    let c_base = tj * len;
    let off_a = r_base * stride + c_base;
    let off_b = c_base * stride + r_base;
    dispatch_len_k!(T, D, |LEN, K| {
        let a_tr =
            transpose_regs::<T, D, LEN, K>(load_regs::<T, D, LEN>(d, &data[off_a..], stride));
        let b_regs = load_regs::<T, D, LEN>(d, &data[off_b..], stride);
        store_regs(&mut data[off_b..], stride, a_tr);
        store_regs(
            &mut data[off_a..],
            stride,
            transpose_regs::<T, D, LEN, K>(b_regs),
        );
    });
}

#[inline(always)]
fn transpose_rect_tiles<D: SimdDescriptor, const K: usize, const BLOCK: usize>(
    d: D,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    r_aligned: usize,
    c_aligned: usize,
) {
    if r_aligned >= BLOCK && c_aligned >= BLOCK {
        for cb in (0..c_aligned).step_by(BLOCK) {
            let c_end = (cb + BLOCK).min(c_aligned);
            for rb in (0..r_aligned).step_by(BLOCK) {
                let r_end = (rb + BLOCK).min(r_aligned);
                for r in (rb..r_end).step_by(K) {
                    for c in (cb..c_end).step_by(K) {
                        transpose_vec_tile::<f32, D>(
                            d,
                            &src[r * src_stride + c..],
                            src_stride,
                            &mut dst[c * dst_stride + r..],
                            dst_stride,
                        );
                    }
                }
            }
        }
    } else {
        for c in (0..c_aligned).step_by(K) {
            for r in (0..r_aligned).step_by(K) {
                transpose_vec_tile::<f32, D>(
                    d,
                    &src[r * src_stride + c..],
                    src_stride,
                    &mut dst[c * dst_stride + r..],
                    dst_stride,
                );
            }
        }
    }
}

#[inline(always)]
fn transpose_in_place_tiled<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    n: usize,
    stride: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    let block_tiles_per_dim: usize = (16 / len).max(1);

    let n_aligned = n & !(len - 1);
    let total_tiles = n / len;
    let full_blocks = total_tiles / block_tiles_per_dim;
    let block_tiles = full_blocks * block_tiles_per_dim;

    for bi in 0..full_blocks {
        let r_tile_base = bi * block_tiles_per_dim;
        for li in 0..block_tiles_per_dim {
            let ti = r_tile_base + li;
            transpose_vec_diag::<T, D>(d, data, stride, ti);
            for lj in li + 1..block_tiles_per_dim {
                transpose_vec_swap::<T, D>(d, data, stride, ti, r_tile_base + lj);
            }
        }
        for bj in bi + 1..full_blocks {
            let c_tile_base = bj * block_tiles_per_dim;
            for li in 0..block_tiles_per_dim {
                for lj in 0..block_tiles_per_dim {
                    transpose_vec_swap::<T, D>(d, data, stride, r_tile_base + li, c_tile_base + lj);
                }
            }
        }
    }

    for ti in block_tiles..total_tiles {
        transpose_vec_diag::<T, D>(d, data, stride, ti);
        for tj in 0..ti {
            transpose_vec_swap::<T, D>(d, data, stride, tj, ti);
        }
    }

    for i in n_aligned..n {
        for j in 0..i {
            data.swap(i * stride + j, j * stride + i);
        }
    }
}

#[inline(always)]
fn transpose_rect_w128<D: SimdDescriptor128>(
    d: D,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    let r4 = rows & !3;
    let c4 = cols & !3;
    transpose_rect_tiles::<D, 4, 16>(d, src, src_stride, dst, dst_stride, r4, c4);
    if c4 < cols && r4 > 0 {
        transpose_rect_scalar(
            &src[c4..],
            src_stride,
            &mut dst[c4 * dst_stride..],
            dst_stride,
            r4,
            cols - c4,
        );
    }
    if r4 < rows {
        transpose_rect_scalar(
            &src[r4 * src_stride..],
            src_stride,
            &mut dst[r4..],
            dst_stride,
            rows - r4,
            cols,
        );
    }
}

#[inline(always)]
fn transpose_into_strided_w128<D: SimdDescriptor128>(
    d: D,
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    dst_stride: usize,
) {
    check_transpose_bounds(src, dst, rows, cols, dst_stride);
    if rows == 0 || cols == 0 {
        return;
    }
    if rows <= 32 && cols <= 32 {
        if (rows & 3) == 0 && (cols & 3) == 0 && rows * cols <= 128 {
            let mut c = 0;
            while c < cols {
                let mut r = 0;
                while r < rows {
                    transpose_vec_tile::<f32, D>(
                        d,
                        &src[r * cols + c..],
                        cols,
                        &mut dst[c * dst_stride + r..],
                        dst_stride,
                    );
                    r += 4;
                }
                c += 4;
            }
            return;
        }
        return transpose_rect_w128(d, src, cols, dst, dst_stride, rows, cols);
    }
    d.call(
        #[inline(always)]
        |d| transpose_rect_w128(d, src, cols, dst, dst_stride, rows, cols),
    );
}

#[inline(always)]
fn transpose_in_place_w128<T: SimdElement, D: SimdDescriptor128>(
    d: D,
    data: &mut [T],
    n: usize,
    stride: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    if n < len {
        for i in 1..n {
            for j in 0..i {
                data.swap(i * stride + j, j * stride + i);
            }
        }
        return;
    }
    if n == len {
        return transpose_vec_diag::<T, D>(d, data, stride, 0);
    }
    if n == 2 * len {
        transpose_vec_diag::<T, D>(d, data, stride, 0);
        transpose_vec_diag::<T, D>(d, data, stride, 1);
        return transpose_vec_swap::<T, D>(d, data, stride, 0, 1);
    }
    if n <= 8 * len {
        return transpose_in_place_tiled::<T, D>(d, data, n, stride);
    }
    d.call(
        #[inline(always)]
        |d| transpose_in_place_tiled::<T, D>(d, data, n, stride),
    );
}

#[inline(always)]
fn transpose_rect_w256<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    let d128 = d.maybe_downgrade_128bit();
    if rows < 8 && cols < 8 {
        return transpose_rect_w128(d128, src, src_stride, dst, dst_stride, rows, cols);
    }
    let r8 = rows & !7;
    let c8 = cols & !7;
    transpose_rect_tiles::<D, 8, 32>(d, src, src_stride, dst, dst_stride, r8, c8);
    if c8 < cols && r8 > 0 {
        transpose_rect_w128(
            d128,
            &src[c8..],
            src_stride,
            &mut dst[c8 * dst_stride..],
            dst_stride,
            r8,
            cols - c8,
        );
    }
    if r8 < rows {
        transpose_rect_w128(
            d128,
            &src[r8 * src_stride..],
            src_stride,
            &mut dst[r8..],
            dst_stride,
            rows - r8,
            cols,
        );
    }
}

#[inline(always)]
fn transpose_into_strided_w256<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    dst_stride: usize,
) {
    if rows < 8 && cols < 8 {
        return transpose_into_strided_w128(
            d.maybe_downgrade_128bit(),
            src,
            dst,
            rows,
            cols,
            dst_stride,
        );
    }
    check_transpose_bounds(src, dst, rows, cols, dst_stride);
    if rows == 0 || cols == 0 {
        return;
    }
    if rows <= 32 && cols <= 32 {
        if (rows & 7) == 0 && (cols & 7) == 0 {
            if rows * cols <= 512 {
                macro_rules! tile8 {
                    ($r:literal, $c:literal) => {
                        if $r < rows && $c < cols {
                            transpose_vec_tile::<f32, D>(
                                d,
                                &src[$r * cols + $c..],
                                cols,
                                &mut dst[$c * dst_stride + $r..],
                                dst_stride,
                            );
                        }
                    };
                }
                tile8!(0, 0);
                tile8!(8, 0);
                tile8!(16, 0);
                tile8!(24, 0);
                tile8!(0, 8);
                tile8!(8, 8);
                tile8!(16, 8);
                tile8!(24, 8);
                tile8!(0, 16);
                tile8!(8, 16);
                tile8!(0, 24);
                tile8!(8, 24);
                return;
            }
            let mut c = 0;
            while c < cols {
                let mut r = 0;
                while r < rows {
                    transpose_vec_tile::<f32, D>(
                        d,
                        &src[r * cols + c..],
                        cols,
                        &mut dst[c * dst_stride + r..],
                        dst_stride,
                    );
                    r += 8;
                }
                c += 8;
            }
            return;
        }
        if rows == 8 && cols == 4 {
            let d128 = d.maybe_downgrade_128bit();
            let s = &src[..32];
            let mut tmp = [0.0f32; 32];
            for r in 0..4 {
                F32Vec::load(d128, &s[4 * r..]).store(&mut tmp[8 * r..]);
                F32Vec::load(d128, &s[4 * (r + 4)..]).store(&mut tmp[8 * r + 4..]);
            }
            let tr = transpose_4_inlane(load_tile_4(d, &tmp, 8));
            return store_tile_4(dst, dst_stride, tr);
        }
        if rows == 4 && cols == 8 {
            let tr = transpose_4_inlane(load_tile_4(d, src, 8));
            if dst_stride == 4 {
                let [u0, u1, u2, u3] = tr;
                let (d0, d2) = u0.interleave_blocks_128(u1);
                let (d1, d3) = u2.interleave_blocks_128(u3);
                return store_tile_4(dst, 8, [d0, d1, d2, d3]);
            }
            let d128 = d.maybe_downgrade_128bit();
            let mut tmp = [0.0f32; 32];
            store_tile_4(&mut tmp, 8, tr);
            let dst = &mut dst[..7 * dst_stride + 4];
            for r in 0..4 {
                F32Vec::load(d128, &tmp[8 * r..]).store(&mut dst[r * dst_stride..]);
                F32Vec::load(d128, &tmp[8 * r + 4..]).store(&mut dst[(r + 4) * dst_stride..]);
            }
            return;
        }
        if (rows < 8 || cols < 8) && (rows & 3) == 0 && (cols & 3) == 0 {
            return transpose_into_strided_w128(
                d.maybe_downgrade_128bit(),
                src,
                dst,
                rows,
                cols,
                dst_stride,
            );
        }
    }
    d.call(
        #[inline(always)]
        |d| transpose_rect_w256(d, src, cols, dst, dst_stride, rows, cols),
    );
}

#[inline(always)]
fn transpose_in_place_w256<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    n: usize,
    stride: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    if n < len {
        return transpose_in_place_w128(d.maybe_downgrade_128bit(), data, n, stride);
    }
    if n == len {
        return transpose_vec_diag::<T, D>(d, data, stride, 0);
    }
    if n == 2 * len {
        transpose_vec_diag::<T, D>(d, data, stride, 0);
        transpose_vec_diag::<T, D>(d, data, stride, 1);
        return transpose_vec_swap::<T, D>(d, data, stride, 0, 1);
    }
    if n <= 4 * len {
        return transpose_in_place_tiled::<T, D>(d, data, n, stride);
    }
    d.call(
        #[inline(always)]
        |d| transpose_in_place_tiled::<T, D>(d, data, n, stride),
    );
}

#[inline(always)]
fn transpose_rect_w512<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    let d256 = d.maybe_downgrade_256bit();
    if rows < 16 && cols < 16 {
        return transpose_rect_w256(d256, src, src_stride, dst, dst_stride, rows, cols);
    }
    let r16 = rows & !15;
    let c16 = cols & !15;
    transpose_rect_tiles::<D, 16, 32>(d, src, src_stride, dst, dst_stride, r16, c16);
    if c16 < cols && r16 > 0 {
        transpose_rect_w256(
            d256,
            &src[c16..],
            src_stride,
            &mut dst[c16 * dst_stride..],
            dst_stride,
            r16,
            cols - c16,
        );
    }
    if r16 < rows {
        transpose_rect_w256(
            d256,
            &src[r16 * src_stride..],
            src_stride,
            &mut dst[r16..],
            dst_stride,
            rows - r16,
            cols,
        );
    }
}

#[inline(always)]
fn transpose_into_strided_w512<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    dst_stride: usize,
) {
    if rows <= 32 && cols <= 32 && (rows < 16 || cols < 16) {
        return transpose_into_strided_w256(
            d.maybe_downgrade_256bit(),
            src,
            dst,
            rows,
            cols,
            dst_stride,
        );
    }
    check_transpose_bounds(src, dst, rows, cols, dst_stride);
    if rows == 0 || cols == 0 {
        return;
    }
    if rows <= 32 && cols <= 32 && (rows & 15) == 0 && (cols & 15) == 0 {
        let mut c = 0;
        while c < cols {
            let mut r = 0;
            while r < rows {
                transpose_vec_tile::<f32, D>(
                    d,
                    &src[r * cols + c..],
                    cols,
                    &mut dst[c * dst_stride + r..],
                    dst_stride,
                );
                r += 16;
            }
            c += 16;
        }
        return;
    }
    d.call(
        #[inline(always)]
        |d| transpose_rect_w512(d, src, cols, dst, dst_stride, rows, cols),
    );
}

#[inline(always)]
fn transpose_in_place_w512<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    n: usize,
    stride: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    if n < len {
        return transpose_in_place_w256(d.maybe_downgrade_256bit(), data, n, stride);
    }
    if n == len {
        return transpose_vec_diag::<T, D>(d, data, stride, 0);
    }
    if n == 2 * len {
        transpose_vec_diag::<T, D>(d, data, stride, 0);
        transpose_vec_diag::<T, D>(d, data, stride, 1);
        return transpose_vec_swap::<T, D>(d, data, stride, 0, 1);
    }
    d.call(
        #[inline(always)]
        |d| transpose_in_place_tiled::<T, D>(d, data, n, stride),
    );
}

/// Transposes a `rows x cols` contiguous row-major matrix in `src` into a `cols x rows` matrix in
/// `dst` with destination row stride `dst_stride` (`dst_stride >= rows`).
#[inline(always)]
pub(crate) fn transpose_into_strided<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    dst_stride: usize,
) {
    const { assert!(D::BYTES == 16 || D::BYTES == 32 || D::BYTES == 64) };
    if D::IS_EMULATED {
        check_transpose_bounds(src, dst, rows, cols, dst_stride);
        if rows > 0 && cols > 0 {
            transpose_rect_scalar(src, cols, dst, dst_stride, rows, cols);
        }
    } else {
        match D::BYTES {
            64 => transpose_into_strided_w512(d, src, dst, rows, cols, dst_stride),
            32 => transpose_into_strided_w256(d, src, dst, rows, cols, dst_stride),
            16 => transpose_into_strided_w128(
                d.maybe_downgrade_128bit(),
                src,
                dst,
                rows,
                cols,
                dst_stride,
            ),
            _ => unreachable!(),
        }
    }
}

/// Transposes a `rows x cols` contiguous row-major matrix in `src` into a `cols x rows` contiguous
/// row-major matrix in `dst`.
#[inline(always)]
pub fn transpose<D: SimdDescriptor>(d: D, src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    transpose_into_strided(d, src, dst, rows, cols, rows);
}

/// Transposes tile-row/tile-column `tile_idx` (`K = SimdVec::<T, D>::LEN` rows/cols) of an
/// `n x n` row-major matrix in `data` in place, transposing diagonal tile `(tile_idx, tile_idx)`
/// and transposing+swapping off-diagonal tile pairs `(tile_idx, j)` and `(j, tile_idx)` for
/// `j > tile_idx` in registers.
#[inline(always)]
pub(crate) fn transpose_in_place_step<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    n: usize,
    tile_idx: usize,
) {
    const { assert!(D::BYTES == 16 || D::BYTES == 32 || D::BYTES == 64) };
    let k = SimdVec::<T, D>::LEN;
    assert!(data.len() >= n * n);
    assert!(n.is_multiple_of(k));
    let chunks = n / k;
    assert!(tile_idx < chunks);
    if D::IS_EMULATED {
        let r_base = tile_idx * k;
        for dr in 0..k {
            for dc in (dr + 1)..k {
                data.swap(
                    (r_base + dr) * n + r_base + dc,
                    (r_base + dc) * n + r_base + dr,
                );
            }
        }
        for j in tile_idx + 1..chunks {
            let c_base = j * k;
            for dr in 0..k {
                for dc in 0..k {
                    data.swap(
                        (r_base + dr) * n + c_base + dc,
                        (c_base + dc) * n + r_base + dr,
                    );
                }
            }
        }
        return;
    }
    transpose_vec_diag::<T, D>(d, data, n, tile_idx);
    for j in tile_idx + 1..chunks {
        transpose_vec_swap::<T, D>(d, data, n, tile_idx, j);
    }
}

/// Transposes an `n x n` row-major matrix in `data` with row stride `stride` (`stride >= n`) in
/// place.
#[inline(always)]
pub fn transpose_in_place_strided<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [T],
    n: usize,
    stride: usize,
) {
    const { assert!(D::BYTES == 16 || D::BYTES == 32 || D::BYTES == 64) };
    assert!(stride >= n);
    if n == 0 {
        return;
    }
    assert!(data.len() >= (n - 1) * stride + n);
    if D::IS_EMULATED {
        for i in 1..n {
            for j in 0..i {
                data.swap(i * stride + j, j * stride + i);
            }
        }
    } else {
        match D::BYTES {
            64 => transpose_in_place_w512(d, data, n, stride),
            32 => transpose_in_place_w256(d, data, n, stride),
            16 => transpose_in_place_w128(d.maybe_downgrade_128bit(), data, n, stride),
            _ => unreachable!(),
        }
    }
}

/// Transposes an `n x n` contiguous row-major matrix in `data` in place.
#[inline(always)]
pub fn transpose_in_place<T: SimdElement, D: SimdDescriptor>(d: D, data: &mut [T], n: usize) {
    transpose_in_place_strided(d, data, n, n);
}

#[cfg(test)]
mod tests {
    use jxl_simd::{simd_test, F32Vec, SimdDescriptor, SimdElement, SimdVec};

    use super::*;

    fn check_transpose_in_place_generic<T: SimdElement + PartialEq, D: SimdDescriptor>(
        d: D,
        gen_val: impl Fn(usize) -> T,
    ) {
        let len = SimdVec::<T, D>::LEN;
        for n in [1, len, 2 * len, len + 3] {
            for stride in [n, 2 * n, 2 * n + 5] {
                let mut mat = vec![T::default(); (n - 1) * stride + n];
                for r in 0..n {
                    for c in 0..n {
                        mat[r * stride + c] = gen_val(r * 100 + c + stride);
                    }
                }
                let orig = mat.clone();
                if stride == n {
                    transpose_in_place(d, &mut mat, n);
                } else {
                    transpose_in_place_strided(d, &mut mat, n, stride);
                }
                for r in 0..n {
                    for c in 0..n {
                        assert_eq!(
                            mat[r * stride + c],
                            orig[c * stride + r],
                            "n={n} stride={stride} r={r} c={c}"
                        );
                    }
                }
            }
        }
    }

    #[simd_test]
    fn test_transpose_in_place_generic<D: SimdDescriptor>(d: D) {
        check_transpose_in_place_generic::<u64, D>(d, |i| (i * 13 + 7) as u64);
        check_transpose_in_place_generic::<f32, D>(d, |i| i as f32 * 0.5);
        check_transpose_in_place_generic::<i32, D>(d, |i| i as i32 - 50);
        check_transpose_in_place_generic::<i16, D>(d, |i| (i as i32 - 500) as i16);
        check_transpose_in_place_generic::<u16, D>(d, |i| (i * 11 + 5) as u16);
        check_transpose_in_place_generic::<u8, D>(d, |i| (i * 7 + 3) as u8);
    }

    #[simd_test]
    fn test_transpose_rectangular<D: SimdDescriptor>(d: D) {
        let sizes = [
            (0, 0),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (8, 32),
            (32, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 128),
            (128, 64),
            (128, 256),
            (256, 128),
            (256, 256),
            (13, 29),
            (29, 13),
            (24, 24),
        ];

        for (rows, cols) in sizes {
            let src: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
            let mut dst = vec![-1.0f32; rows * cols];
            transpose(d, &src, &mut dst, rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(dst[c * rows + r], src[r * cols + c], "({rows}, {cols})");
                }
            }

            let dst_stride = rows + 11;
            let mut strided = vec![-9.0f32; cols * dst_stride];
            transpose_into_strided(d, &src, &mut strided, rows, cols, dst_stride);
            for c in 0..cols {
                for r in 0..rows {
                    assert_eq!(
                        strided[c * dst_stride + r],
                        src[r * cols + c],
                        "strided ({rows}, {cols})"
                    );
                }
                for pad in rows..dst_stride {
                    assert_eq!(strided[c * dst_stride + pad], -9.0f32);
                }
            }
        }
    }

    #[simd_test]
    fn test_transpose_in_place_and_step<D: SimdDescriptor>(d: D) {
        for n in [1, 2, 4, 7, 8, 12, 16, 20, 24, 32, 48, 64, 68, 72, 128] {
            let orig: Vec<f32> = (0..n * n).map(|i| (i + 1) as f32).collect();
            let mut mat = orig.clone();
            transpose_in_place(d, &mut mat, n);
            for r in 0..n {
                for c in 0..n {
                    assert_eq!(mat[r * n + c], orig[c * n + r], "in_place n={n}");
                }
            }

            if n.is_multiple_of(F32Vec::<D>::LEN) {
                let mut stepped = orig.clone();
                for tile_idx in 0..(n / F32Vec::<D>::LEN) {
                    transpose_in_place_step(d, &mut stepped, n, tile_idx);
                }
                assert_eq!(stepped, mat, "step n={n}");
            }
        }
    }
}
