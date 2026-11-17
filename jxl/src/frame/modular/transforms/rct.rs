// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{I16Vec, I32Vec, IntElement, SimdDescriptor, SimdVec, shr, simd_dispatch};

use crate::frame::modular::transforms::{RctOp, RctPermutation};
use crate::frame::modular::{ModularChannel, ModularStorage};
use crate::image::{ImageDataType, ImageRectMut};
use crate::util::tracing_wrappers::*;

trait RctSample: IntElement + ImageDataType {
    fn avg<D: SimdDescriptor>(d: D, a: SimdVec<Self, D>, b: SimdVec<Self, D>) -> SimdVec<Self, D>;
}

impl RctSample for i32 {
    #[inline(always)]
    fn avg<D: SimdDescriptor>(_: D, a: I32Vec<D>, b: I32Vec<D>) -> I32Vec<D> {
        shr!(a + b, 1)
    }
}

impl RctSample for i16 {
    #[inline(always)]
    fn avg<D: SimdDescriptor>(d: D, a: I16Vec<D>, b: I16Vec<D>) -> I16Vec<D> {
        shr!(a, 1) + shr!(b, 1) + (a & b & I16Vec::splat(d, 1))
    }
}

#[inline(always)]
fn rct_impl<T: RctSample, D: SimdDescriptor, const OP: u32>(
    d: D,
    v0: SimdVec<T, D>,
    v1: SimdVec<T, D>,
    v2: SimdVec<T, D>,
) -> (SimdVec<T, D>, SimdVec<T, D>, SimdVec<T, D>) {
    const { assert!(OP <= 6) };

    match OP {
        0 => (v0, v1, v2),
        1 => (v0, v1, v2 + v0),
        2 => (v0, v1 + v0, v2),
        3 => (v0, v1 + v0, v2 + v0),
        4 => {
            let avg = T::avg(d, v0, v2);
            (v0, v1 + avg, v2)
        }
        5 => {
            let v2 = v0 + v2;
            let avg = T::avg(d, v0, v2);
            (v0, v1 + avg, v2)
        }
        6 => {
            let (y, co, cg) = (v0, v1, v2);
            let y = y - shr!(cg, 1);
            let g = cg + y;
            let y = y - shr!(co, 1);
            let r = y + co;
            (r, g, y)
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn rct_row_impl<T: RctSample, D: SimdDescriptor, const OP: u32>(
    d: D,
    rgb: [&mut [T]; 3],
) -> [&mut [T]; 3] {
    const { assert!(OP <= 6) };

    let [mut it_row_r, mut it_row_g, mut it_row_b] =
        rgb.map(|x| x.chunks_exact_mut(SimdVec::<T, D>::LEN));
    let it = (&mut it_row_r).zip(&mut it_row_g).zip(&mut it_row_b);

    for ((r, g), b) in it {
        let v0 = SimdVec::load(d, r);
        let v1 = SimdVec::load(d, g);
        let v2 = SimdVec::load(d, b);
        let (w0, w1, w2) = rct_impl::<T, D, OP>(d, v0, v1, v2);
        w0.store(r);
        w1.store(g);
        w2.store(b);
    }

    [
        it_row_r.into_remainder(),
        it_row_g.into_remainder(),
        it_row_b.into_remainder(),
    ]
}

#[inline(always)]
fn rct_loop_impl<T: RctSample, D: SimdDescriptor, const OP: u32>(
    d: D,
    r: &mut ImageRectMut<'_, T>,
    g: &mut ImageRectMut<'_, T>,
    b: &mut ImageRectMut<'_, T>,
) {
    const { assert!(OP <= 6) };

    let h = r.size().1;

    for pos_y in 0..h {
        let mut rgb = [r.row(pos_y), g.row(pos_y), b.row(pos_y)];

        rgb = rct_row_impl::<T, D, OP>(d, rgb);
        if D::BYTES > 32 {
            rgb = rct_row_impl::<T, _, OP>(d.maybe_downgrade_256bit(), rgb);
        }
        if D::BYTES > 16 {
            rgb = rct_row_impl::<T, _, OP>(d.maybe_downgrade_128bit(), rgb);
        }
        let [r, g, b] = rgb;
        if !r.is_empty() {
            let d128 = d.maybe_downgrade_128bit();
            let v0 = SimdVec::load_partial(d128, r);
            let v1 = SimdVec::load_partial(d128, g);
            let v2 = SimdVec::load_partial(d128, b);
            let (w0, w1, w2) = rct_impl::<T, _, OP>(d128, v0, v1, v2);
            w0.store_partial(r);
            w1.store_partial(g);
            w2.store_partial(b);
        }
    }
}

#[simd_dispatch(rct_loop)]
fn rct_loop_fwd<T: RctSample, D: SimdDescriptor>(
    d: D,
    r: &mut ImageRectMut<'_, T>,
    g: &mut ImageRectMut<'_, T>,
    b: &mut ImageRectMut<'_, T>,
    op: RctOp,
) {
    match op {
        RctOp::Noop => {}
        RctOp::AddFirstToThird => rct_loop_impl::<T, D, 1>(d, r, g, b),
        RctOp::AddFirstToSecond => rct_loop_impl::<T, D, 2>(d, r, g, b),
        RctOp::AddFirstToSecondAndThird => rct_loop_impl::<T, D, 3>(d, r, g, b),
        RctOp::AddAvgToSecond => rct_loop_impl::<T, D, 4>(d, r, g, b),
        RctOp::AddFirstToThirdAndAvgToSecond => rct_loop_impl::<T, D, 5>(d, r, g, b),
        RctOp::YCoCg => rct_loop_impl::<T, D, 6>(d, r, g, b),
    }
}

fn apply_rct_permutation(buffers: &mut [&mut ModularChannel], perm: RctPermutation) {
    let [r, g, b] = buffers else {
        unreachable!("incorrect buffer count for RCT");
    };

    // Note: Gbr and Brg use the *inverse* permutation compared to libjxl, because we *first* write
    // to the buffers and then permute them, while in libjxl the buffers to be written to are
    // permuted first.
    // The same is true for Rbg/Grb/Bgr, but since those are involutions it doesn't change
    // anything.
    match perm {
        RctPermutation::Rgb => {}
        RctPermutation::Gbr => {
            // out[1, 2, 0] = in[0, 1, 2]
            std::mem::swap(&mut g.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Brg => {
            // out[2, 0, 1] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Rbg => {
            // out[0, 2, 1] = in[0, 1, 2]
            std::mem::swap(&mut b.data, &mut g.data);
        }
        RctPermutation::Grb => {
            // out[1, 0, 2] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Bgr => {
            // out[2, 1, 0] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data);
        }
    }
}

fn do_rct_step_typed<T: RctSample>(buffers: &mut [&mut ModularChannel], op: RctOp) {
    let [r, g, b] = buffers else {
        unreachable!("incorrect buffer count for RCT");
    };

    if op != RctOp::Noop {
        let mut r_rect = ImageRectMut::<T>::from_raw(r.data.as_rect_mut());
        let mut g_rect = ImageRectMut::<T>::from_raw(g.data.as_rect_mut());
        let mut b_rect = ImageRectMut::<T>::from_raw(b.data.as_rect_mut());
        rct_loop::<T>(&mut r_rect, &mut g_rect, &mut b_rect, op);
    }
}

// Applies a RCT in-place to the given buffers.
#[instrument(level = "debug", skip(buffers), ret)]
pub fn do_rct_step(
    buffers: &mut [&mut ModularChannel],
    storage: ModularStorage,
    op: RctOp,
    perm: RctPermutation,
) {
    match storage {
        ModularStorage::I16 => do_rct_step_typed::<i16>(buffers, op),
        ModularStorage::I32 => do_rct_step_typed::<i32>(buffers, op),
    }

    apply_rct_permutation(buffers, perm);
}
