// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::type_complexity, dead_code)]

use std::fmt::Debug;

use super::sealed::{
    ComparableElement, CompareBackend, Element, IntBitBackend, IntElement, LaneBackend,
    U8ExtraBackend,
};
use super::{
    F32Vec, I32Vec, SimdDescriptor, SimdDescriptor128, SimdVec, U8Vec, U16Vec, U32Vec, U64Vec,
};

#[inline(always)]
pub(crate) fn chunks<T, const N: usize>(slice: &[T]) -> &[[T; N]] {
    let (ret, rem) = slice.as_chunks::<N>();
    assert!(rem.is_empty());
    ret
}

#[inline(always)]
pub(crate) fn chunks_mut<T, const N: usize>(slice: &mut [T]) -> &mut [[T; N]] {
    let (ret, rem) = slice.as_chunks_mut::<N>();
    assert!(rem.is_empty());
    ret
}

#[inline(always)]
pub(crate) fn store_interleaved_2<T: Copy + Default + Debug + 'static, D>(
    d: D,
    a: <D as LaneBackend<T>>::Raw,
    b: <D as LaneBackend<T>>::Raw,
    dest: &mut [T],
) where
    D: SimdDescriptor + LaneBackend<T>,
{
    let len = D::BYTES / std::mem::size_of::<T>();
    assert!(dest.len() >= 2 * len);
    let t0 = <D as LaneBackend<T>>::interleave_lo(d, a, b);
    let t1 = <D as LaneBackend<T>>::interleave_hi(d, a, b);
    let b0 = <D as LaneBackend<T>>::to_raw_bytes(d, t0);
    let b1 = <D as LaneBackend<T>>::to_raw_bytes(d, t1);
    let (out0, out1) = if D::BYTES == 16 {
        (b0, b1)
    } else if D::BYTES == 32 {
        d.interleave_128_halves(b0, b1)
    } else {
        let (p0, p1) = d.interleave_128_halves(b0, b1);
        d.interleave_256_halves(p0, p1)
    };
    let out0 = <D as LaneBackend<T>>::from_raw_bytes(d, out0);
    let out1 = <D as LaneBackend<T>>::from_raw_bytes(d, out1);
    <D as LaneBackend<T>>::store(d, out0, dest);
    <D as LaneBackend<T>>::store(d, out1, &mut dest[len..]);
}

#[inline(always)]
pub(crate) fn store_interleaved_4<T: Copy + Default + Debug + 'static, D>(
    d: D,
    a: <D as LaneBackend<T>>::Raw,
    b: <D as LaneBackend<T>>::Raw,
    c: <D as LaneBackend<T>>::Raw,
    e: <D as LaneBackend<T>>::Raw,
    dest: &mut [T],
) where
    D: SimdDescriptor + LaneBackend<T>,
{
    let len = D::BYTES / std::mem::size_of::<T>();
    assert!(dest.len() >= 4 * len);
    let ac_0 = <D as LaneBackend<T>>::interleave_lo(d, a, c);
    let ac_1 = <D as LaneBackend<T>>::interleave_hi(d, a, c);
    let bd_0 = <D as LaneBackend<T>>::interleave_lo(d, b, e);
    let bd_1 = <D as LaneBackend<T>>::interleave_hi(d, b, e);

    let r0 =
        <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::interleave_lo(d, ac_0, bd_0));
    let r1 =
        <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::interleave_hi(d, ac_0, bd_0));
    let r2 =
        <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::interleave_lo(d, ac_1, bd_1));
    let r3 =
        <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::interleave_hi(d, ac_1, bd_1));

    let [out0, out1, out2, out3] = if D::BYTES == 16 {
        [r0, r1, r2, r3]
    } else {
        let (p0, p2) = d.interleave_128_halves(r0, r1);
        let (p1, p3) = d.interleave_128_halves(r2, r3);
        if D::BYTES == 32 {
            [p0, p1, p2, p3]
        } else {
            let (o0, o2) = d.interleave_256_halves(p0, p1);
            let (o1, o3) = d.interleave_256_halves(p2, p3);
            [o0, o1, o2, o3]
        }
    };

    <D as LaneBackend<T>>::store(d, <D as LaneBackend<T>>::from_raw_bytes(d, out0), dest);
    <D as LaneBackend<T>>::store(
        d,
        <D as LaneBackend<T>>::from_raw_bytes(d, out1),
        &mut dest[len..],
    );
    <D as LaneBackend<T>>::store(
        d,
        <D as LaneBackend<T>>::from_raw_bytes(d, out2),
        &mut dest[2 * len..],
    );
    <D as LaneBackend<T>>::store(
        d,
        <D as LaneBackend<T>>::from_raw_bytes(d, out3),
        &mut dest[3 * len..],
    );
}

/// Software IEEE 754 `f16` bits -> `f32` conversion for ISAs without native hardware conversion.
#[inline(always)]
pub(crate) fn load_f16_bits_sw<D: SimdDescriptor>(d: D, mem: &[u16]) -> F32Vec<D> {
    let wide = I32Vec::load_from_u16(d, mem);
    let sign = (wide & I32Vec::splat(d, 0x8000)).shl::<16>();
    let mag = wide & I32Vec::splat(d, 0x7FFF);
    let mag_shifted = mag.shl::<13>();

    let normal = mag_shifted + I32Vec::splat(d, 112 << 23);
    let magic = I32Vec::splat(d, 113 << 23).bitcast::<f32>();
    let subnorm =
        ((magic.bitcast::<i32>() | mag_shifted).bitcast::<f32>() - magic).bitcast::<i32>();
    let infnan = normal + I32Vec::splat(d, 112 << 23);

    let is_subnorm = I32Vec::splat(d, 0x0400).gt(mag);
    let is_infnan = mag.gt(I32Vec::splat(d, 0x7BFF));
    let res = is_subnorm.if_then_else(subnorm, normal);
    let res = is_infnan.if_then_else(infnan, res);
    (sign | res).bitcast::<f32>()
}

/// Software `f32` -> IEEE 754 `f16` bits conversion (round-to-nearest-ties-to-even).
#[inline(always)]
pub(crate) fn store_f16_bits_sw<D: SimdDescriptor>(v: F32Vec<D>, dest: &mut [u16]) {
    let d = v.descriptor();
    let bits = v.bitcast::<i32>();
    let abs = bits & I32Vec::splat(d, 0x7FFF_FFFF);
    let sign = bits.bitcast::<u32>().shr::<16>().bitcast::<i32>() & I32Vec::splat(d, 0x8000);

    let abs_shifted = abs.bitcast::<u32>().shr::<13>().bitcast::<i32>();
    let lsb = abs_shifted & I32Vec::splat(d, 1);
    let rounded = abs + lsb + I32Vec::splat(d, 0x0FFF);

    let normal = (rounded.bitcast::<u32>().shr::<13>().bitcast::<i32>()
        - I32Vec::splat(d, 112 << 10))
    .min(I32Vec::splat(d, 0x7C00));

    let magic = I32Vec::splat(d, 126 << 23);
    let subnorm = (abs.bitcast::<f32>() + magic.bitcast::<f32>()).bitcast::<i32>() - magic;

    let qnan = I32Vec::splat(d, 0x7E00) | (abs_shifted & I32Vec::splat(d, 0x03FF));

    let is_subnorm = I32Vec::splat(d, 0x3880_0000).gt(abs);
    let is_nan = abs.gt(I32Vec::splat(d, 0x7F80_0000));
    let res = is_subnorm.if_then_else(subnorm, normal);
    let res = is_nan.if_then_else(qnan, res);
    (sign | res).store_u16(dest);
}

const fn swizzle_mask_u64s(m: [u8; 16]) -> (u64, u64) {
    (
        u64::from_le_bytes([m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]]),
        u64::from_le_bytes([m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]]),
    )
}

#[inline(always)]
fn splat_swizzle_u64s<D: SimdDescriptor>(d: D, (lo, hi): (u64, u64)) -> U8Vec<D> {
    U64Vec::splat(d, lo)
        .interleave_lo(U64Vec::splat(d, hi))
        .bitcast::<u8>()
}

const fn interleave3_u64s<T>(out_vec: usize, ch: usize) -> (u64, u64) {
    let elem_bytes = std::mem::size_of::<T>();
    let mut m = [0x80u8; 16];
    let mut i = 0;
    while i < 16 {
        let pos = out_vec * 16 + i;
        let q = pos / elem_bytes;
        let r = pos % elem_bytes;
        if q % 3 == ch {
            m[i] = ((q / 3) * elem_bytes + r) as u8;
        }
        i += 1;
    }
    swizzle_mask_u64s(m)
}

#[inline(always)]
pub(crate) fn store_interleaved_3<T: Copy + Default + Debug + 'static, D>(
    d: D,
    a: <D as LaneBackend<T>>::Raw,
    b: <D as LaneBackend<T>>::Raw,
    c: <D as LaneBackend<T>>::Raw,
    dest: &mut [T],
) where
    D: SimdDescriptor + LaneBackend<T>,
{
    let len = D::BYTES / std::mem::size_of::<T>();
    assert!(dest.len() >= 3 * len);

    if D::IS_EMULATED {
        let mut aa = <D as LaneBackend<T>>::zero_array();
        let mut ba = <D as LaneBackend<T>>::zero_array();
        let mut ca = <D as LaneBackend<T>>::zero_array();
        <D as LaneBackend<T>>::store_array(d, a, &mut aa);
        <D as LaneBackend<T>>::store_array(d, b, &mut ba);
        <D as LaneBackend<T>>::store_array(d, c, &mut ca);
        let (sa, sb, sc) = (aa.as_ref(), ba.as_ref(), ca.as_ref());
        for i in 0..len {
            dest[3 * i] = sa[i];
            dest[3 * i + 1] = sb[i];
            dest[3 * i + 2] = sc[i];
        }
        return;
    }

    let au = SimdVec(
        <u8 as Element>::from_raw_bytes(d, <D as LaneBackend<T>>::to_raw_bytes(d, a)),
        d,
    );
    let bu = SimdVec(
        <u8 as Element>::from_raw_bytes(d, <D as LaneBackend<T>>::to_raw_bytes(d, b)),
        d,
    );
    let cu = SimdVec(
        <u8 as Element>::from_raw_bytes(d, <D as LaneBackend<T>>::to_raw_bytes(d, c)),
        d,
    );

    macro_rules! step {
        ($ov:literal) => {{
            let m0 = splat_swizzle_u64s(d, const { interleave3_u64s::<T>($ov, 0) });
            let m1 = splat_swizzle_u64s(d, const { interleave3_u64s::<T>($ov, 1) });
            let m2 = splat_swizzle_u64s(d, const { interleave3_u64s::<T>($ov, 2) });
            let v = au.table_lookup_bytes_or_zero(m0)
                | bu.table_lookup_bytes_or_zero(m1)
                | cu.table_lookup_bytes_or_zero(m2);
            <u8 as Element>::to_raw_bytes(d, v.0)
        }};
    }
    let (t0, t1, t2) = (step!(0), step!(1), step!(2));

    let [out0, out1, out2] = if D::BYTES == 16 {
        [t0, t1, t2]
    } else {
        let (u0, p3) = d.interleave_128_halves(t0, t1);
        let (u1, _) = d.interleave_128_halves(t2, p3);
        let (_, u2) = d.interleave_128_halves(t1, t2);
        if D::BYTES == 32 {
            [u0, u1, u2]
        } else {
            let (out0, _) = d.interleave_256_halves(u0, u1);
            let (_, out2) = d.interleave_256_halves(u1, u2);
            let (_, u0_hi) = d.interleave_256_halves(u0, u0);
            let (out1, _) = d.interleave_256_halves(u2, u0_hi);
            [out0, out1, out2]
        }
    };

    <D as LaneBackend<T>>::store(d, <D as LaneBackend<T>>::from_raw_bytes(d, out0), dest);
    <D as LaneBackend<T>>::store(
        d,
        <D as LaneBackend<T>>::from_raw_bytes(d, out1),
        &mut dest[len..],
    );
    <D as LaneBackend<T>>::store(
        d,
        <D as LaneBackend<T>>::from_raw_bytes(d, out2),
        &mut dest[2 * len..],
    );
}

#[inline(always)]
fn transpose_4x4_f32_in_128<D: SimdDescriptor>(
    x0: F32Vec<D>,
    x1: F32Vec<D>,
    x2: F32Vec<D>,
    x3: F32Vec<D>,
) -> [F32Vec<D>; 4] {
    let lo01 = x0.interleave_lo(x1).bitcast::<u64>();
    let hi01 = x0.interleave_hi(x1).bitcast::<u64>();
    let lo23 = x2.interleave_lo(x3).bitcast::<u64>();
    let hi23 = x2.interleave_hi(x3).bitcast::<u64>();
    [
        lo01.interleave_lo(lo23).bitcast::<f32>(),
        lo01.interleave_hi(lo23).bitcast::<f32>(),
        hi01.interleave_lo(hi23).bitcast::<f32>(),
        hi01.interleave_hi(hi23).bitcast::<f32>(),
    ]
}

#[inline(always)]
pub(crate) fn f32_store_interleaved_8<D: SimdDescriptor>(
    d: D,
    a: <D as LaneBackend<f32>>::Raw,
    b: <D as LaneBackend<f32>>::Raw,
    c: <D as LaneBackend<f32>>::Raw,
    dv: <D as LaneBackend<f32>>::Raw,
    e: <D as LaneBackend<f32>>::Raw,
    f: <D as LaneBackend<f32>>::Raw,
    g: <D as LaneBackend<f32>>::Raw,
    h: <D as LaneBackend<f32>>::Raw,
    dest: &mut [f32],
) {
    let len = F32Vec::<D>::LEN;
    assert!(dest.len() >= 8 * len);

    if D::IS_EMULATED {
        let mut bufs = [<D as LaneBackend<f32>>::zero_array(); 8];
        for (buf, raw) in bufs.iter_mut().zip([a, b, c, dv, e, f, g, h]) {
            SimdVec::<f32, D>(raw, d).store_array(buf);
        }
        for i in 0..len {
            for (ch, buf) in bufs.iter().enumerate() {
                dest[8 * i + ch] = buf.as_ref()[i];
            }
        }
        return;
    }

    let [abcd_0, abcd_1, abcd_2, abcd_3] =
        transpose_4x4_f32_in_128(SimdVec(a, d), SimdVec(b, d), SimdVec(c, d), SimdVec(dv, d));
    let [efgh_0, efgh_1, efgh_2, efgh_3] =
        transpose_4x4_f32_in_128(SimdVec(e, d), SimdVec(f, d), SimdVec(g, d), SimdVec(h, d));

    let outs = if D::BYTES == 16 {
        [
            abcd_0, efgh_0, abcd_1, efgh_1, abcd_2, efgh_2, abcd_3, efgh_3,
        ]
    } else if D::BYTES == 32 {
        let (out0, out4) = abcd_0.interleave_blocks_128(efgh_0);
        let (out1, out5) = abcd_1.interleave_blocks_128(efgh_1);
        let (out2, out6) = abcd_2.interleave_blocks_128(efgh_2);
        let (out3, out7) = abcd_3.interleave_blocks_128(efgh_3);
        [out0, out1, out2, out3, out4, out5, out6, out7]
    } else {
        let (p0_02, p0_13) = abcd_0.interleave_blocks_128(efgh_0);
        let (p1_02, p1_13) = abcd_1.interleave_blocks_128(efgh_1);
        let (p2_02, p2_13) = abcd_2.interleave_blocks_128(efgh_2);
        let (p3_02, p3_13) = abcd_3.interleave_blocks_128(efgh_3);
        let (out0, out4) = p0_02.interleave_blocks_256(p1_02);
        let (out1, out5) = p2_02.interleave_blocks_256(p3_02);
        let (out2, out6) = p0_13.interleave_blocks_256(p1_13);
        let (out3, out7) = p2_13.interleave_blocks_256(p3_13);
        [out0, out1, out2, out3, out4, out5, out6, out7]
    };

    for (k, out) in outs.into_iter().enumerate() {
        out.store(&mut dest[k * len..]);
    }
}

#[inline(always)]
fn deinterleave_2_vecs<T: Copy + Default + Debug + 'static, D: SimdDescriptor + LaneBackend<T>>(
    d: D,
    v0: <D as LaneBackend<T>>::Raw,
    v1: <D as LaneBackend<T>>::Raw,
) -> (<D as LaneBackend<T>>::Raw, <D as LaneBackend<T>>::Raw) {
    let (mut w0, mut w1) = if D::BYTES == 16 {
        (v0, v1)
    } else {
        let b0 = <D as LaneBackend<T>>::to_raw_bytes(d, v0);
        let b1 = <D as LaneBackend<T>>::to_raw_bytes(d, v1);
        let (u0, u1) = if D::BYTES == 64 {
            let (x, y) = d.interleave_256_halves(b0, b1);
            d.interleave_128_halves(x, y)
        } else {
            d.interleave_128_halves(b0, b1)
        };
        (
            <D as LaneBackend<T>>::from_raw_bytes(d, u0),
            <D as LaneBackend<T>>::from_raw_bytes(d, u1),
        )
    };
    let steps = (16 / std::mem::size_of::<T>()).ilog2();
    for _ in 0..steps {
        let lo = <D as LaneBackend<T>>::interleave_lo(d, w0, w1);
        let hi = <D as LaneBackend<T>>::interleave_hi(d, w0, w1);
        w0 = lo;
        w1 = hi;
    }
    (w0, w1)
}

#[inline(always)]
pub(crate) fn load_deinterleaved_2<
    T: Copy + Default + Debug + 'static,
    D: SimdDescriptor + LaneBackend<T>,
>(
    d: D,
    src: &[T],
) -> (<D as LaneBackend<T>>::Raw, <D as LaneBackend<T>>::Raw) {
    let len = D::BYTES / std::mem::size_of::<T>();
    assert!(src.len() >= 2 * len);
    if D::IS_EMULATED {
        let mut a = <D as LaneBackend<T>>::zero_array();
        let mut b = <D as LaneBackend<T>>::zero_array();
        for i in 0..len {
            a.as_mut()[i] = src[2 * i];
            b.as_mut()[i] = src[2 * i + 1];
        }
        return (
            <D as LaneBackend<T>>::load_array(d, &a),
            <D as LaneBackend<T>>::load_array(d, &b),
        );
    }
    deinterleave_2_vecs::<T, D>(
        d,
        <D as LaneBackend<T>>::load(d, src),
        <D as LaneBackend<T>>::load(d, &src[len..]),
    )
}

const fn deinterleave3_u64s<T>(ch: usize, in_vec: usize) -> (u64, u64) {
    let elem_bytes = std::mem::size_of::<T>();
    let mut m = [0x80u8; 16];
    let mut i = 0;
    while i < 16 {
        let lane = i / elem_bytes;
        let r = i % elem_bytes;
        let src_pos = (lane * 3 + ch) * elem_bytes + r;
        if src_pos / 16 == in_vec {
            m[i] = (src_pos % 16) as u8;
        }
        i += 1;
    }
    swizzle_mask_u64s(m)
}

#[inline(always)]
pub(crate) fn load_deinterleaved_3<
    T: Copy + Default + Debug + 'static,
    D: SimdDescriptor + LaneBackend<T>,
>(
    d: D,
    src: &[T],
) -> (
    <D as LaneBackend<T>>::Raw,
    <D as LaneBackend<T>>::Raw,
    <D as LaneBackend<T>>::Raw,
) {
    let elem_bytes = std::mem::size_of::<T>();
    let len = D::BYTES / elem_bytes;
    assert!(src.len() >= 3 * len);
    if D::IS_EMULATED {
        let mut a = <D as LaneBackend<T>>::zero_array();
        let mut b = <D as LaneBackend<T>>::zero_array();
        let mut c = <D as LaneBackend<T>>::zero_array();
        for i in 0..len {
            a.as_mut()[i] = src[3 * i];
            b.as_mut()[i] = src[3 * i + 1];
            c.as_mut()[i] = src[3 * i + 2];
        }
        return (
            <D as LaneBackend<T>>::load_array(d, &a),
            <D as LaneBackend<T>>::load_array(d, &b),
            <D as LaneBackend<T>>::load_array(d, &c),
        );
    }

    let in0 = <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::load(d, src));
    let in1 = <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::load(d, &src[len..]));
    let in2 =
        <D as LaneBackend<T>>::to_raw_bytes(d, <D as LaneBackend<T>>::load(d, &src[2 * len..]));

    let (u0, u1, u2) = if D::BYTES == 64 {
        let (_, hi1) = d.interleave_256_halves(in1, in1);
        let (u0, _) = d.interleave_256_halves(in0, hi1);
        let (lo2, hi2) = d.interleave_256_halves(in2, in2);
        let (_, u1) = d.interleave_256_halves(in0, lo2);
        let (u2, _) = d.interleave_256_halves(in1, hi2);
        (u0, u1, u2)
    } else {
        (in0, in1, in2)
    };

    let (t0, t1, t2) = if D::BYTES >= 32 {
        let (a, b) = d.interleave_128_halves(u0, u1);
        let (t1, _) = d.interleave_128_halves(b, u2);
        let (_, t2) = d.interleave_128_halves(a, u2);
        let (_, b3) = d.interleave_128_halves(b, b);
        let (t0, _) = d.interleave_128_halves(a, b3);
        (t0, t1, t2)
    } else {
        (u0, u1, u2)
    };

    let t0u = <D as LaneBackend<u8>>::from_raw_bytes(d, t0);
    let t1u = <D as LaneBackend<u8>>::from_raw_bytes(d, t1);
    let t2u = <D as LaneBackend<u8>>::from_raw_bytes(d, t2);
    macro_rules! extract {
        ($ch:literal) => {{
            let m0 = splat_swizzle_u64s(d, const { deinterleave3_u64s::<T>($ch, 0) }).0;
            let m1 = splat_swizzle_u64s(d, const { deinterleave3_u64s::<T>($ch, 1) }).0;
            let m2 = splat_swizzle_u64s(d, const { deinterleave3_u64s::<T>($ch, 2) }).0;
            let s0 = <D as U8ExtraBackend>::u8_swizzle_128_or_zero(d, t0u, m0);
            let s1 = <D as U8ExtraBackend>::u8_swizzle_128_or_zero(d, t1u, m1);
            let s2 = <D as U8ExtraBackend>::u8_swizzle_128_or_zero(d, t2u, m2);
            let out = <D as IntBitBackend<u8>>::bit_or(
                d,
                <D as IntBitBackend<u8>>::bit_or(d, s0, s1),
                s2,
            );
            <D as LaneBackend<T>>::from_raw_bytes(d, <D as LaneBackend<u8>>::to_raw_bytes(d, out))
        }};
    }
    (extract!(0), extract!(1), extract!(2))
}

#[inline(always)]
pub(crate) fn load_deinterleaved_4<
    T: Copy + Default + Debug + 'static,
    D: SimdDescriptor + LaneBackend<T>,
>(
    d: D,
    src: &[T],
) -> (
    <D as LaneBackend<T>>::Raw,
    <D as LaneBackend<T>>::Raw,
    <D as LaneBackend<T>>::Raw,
    <D as LaneBackend<T>>::Raw,
) {
    let len = D::BYTES / std::mem::size_of::<T>();
    assert!(src.len() >= 4 * len);
    if D::IS_EMULATED {
        let mut a = <D as LaneBackend<T>>::zero_array();
        let mut b = <D as LaneBackend<T>>::zero_array();
        let mut c = <D as LaneBackend<T>>::zero_array();
        let mut e = <D as LaneBackend<T>>::zero_array();
        for i in 0..len {
            a.as_mut()[i] = src[4 * i];
            b.as_mut()[i] = src[4 * i + 1];
            c.as_mut()[i] = src[4 * i + 2];
            e.as_mut()[i] = src[4 * i + 3];
        }
        return (
            <D as LaneBackend<T>>::load_array(d, &a),
            <D as LaneBackend<T>>::load_array(d, &b),
            <D as LaneBackend<T>>::load_array(d, &c),
            <D as LaneBackend<T>>::load_array(d, &e),
        );
    }
    let (e01, o01) = deinterleave_2_vecs::<T, D>(
        d,
        <D as LaneBackend<T>>::load(d, src),
        <D as LaneBackend<T>>::load(d, &src[len..]),
    );
    let (e23, o23) = deinterleave_2_vecs::<T, D>(
        d,
        <D as LaneBackend<T>>::load(d, &src[2 * len..]),
        <D as LaneBackend<T>>::load(d, &src[3 * len..]),
    );
    let (c0, c2) = deinterleave_2_vecs::<T, D>(d, e01, e23);
    let (c1, c3) = deinterleave_2_vecs::<T, D>(d, o01, o23);
    (c0, c1, c2, c3)
}

#[inline(always)]
pub(crate) fn shl_vec_default<T, D>(
    d: D,
    mut v: <D as LaneBackend<T>>::Raw,
    counts: <D as LaneBackend<T>>::Raw,
) -> <D as LaneBackend<T>>::Raw
where
    T: IntElement,
    D: SimdDescriptor + IntBitBackend<T>,
{
    macro_rules! step {
        ($b:literal) => {
            if T::BITS > $b {
                let c = <D as LaneBackend<T>>::splat(d, T::from_u8($b));
                let m = <D as CompareBackend<T>>::eq_zero(
                    d,
                    <D as IntBitBackend<T>>::bit_and(d, counts, c),
                );
                v = <D as CompareBackend<T>>::mask_select(
                    d,
                    m,
                    v,
                    <D as IntBitBackend<T>>::shl::<$b>(d, v),
                );
            }
        };
    }
    step!(1);
    step!(2);
    step!(4);
    step!(8);
    step!(16);
    step!(32);
    v
}

#[inline(always)]
pub(crate) fn shr_vec_default<T, D>(
    d: D,
    mut v: <D as LaneBackend<T>>::Raw,
    counts: <D as LaneBackend<T>>::Raw,
) -> <D as LaneBackend<T>>::Raw
where
    T: IntElement,
    D: SimdDescriptor + IntBitBackend<T>,
{
    macro_rules! step {
        ($b:literal) => {
            if T::BITS > $b {
                let c = <D as LaneBackend<T>>::splat(d, T::from_u8($b));
                let m = <D as CompareBackend<T>>::eq_zero(
                    d,
                    <D as IntBitBackend<T>>::bit_and(d, counts, c),
                );
                v = <D as CompareBackend<T>>::mask_select(
                    d,
                    m,
                    v,
                    <D as IntBitBackend<T>>::shr::<$b>(d, v),
                );
            }
        };
    }
    step!(1);
    step!(2);
    step!(4);
    step!(8);
    step!(16);
    step!(32);
    v
}

const POPCNT_LUT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
const CLZ_LUT: [u8; 16] = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline(always)]
fn u8_count_ones<D: SimdDescriptor>(d: D, v: U8Vec<D>) -> U8Vec<D> {
    let lut = splat_swizzle_u64s(d, const { swizzle_mask_u64s(POPCNT_LUT) });
    let lo = lut.table_lookup_bytes_or_zero(v & U8Vec::splat(d, 0x0F));
    let hi = lut.table_lookup_bytes_or_zero(crate::shr!(v, 4));
    lo + hi
}

#[inline(always)]
fn u8_leading_zeros<D: SimdDescriptor>(d: D, v: U8Vec<D>) -> U8Vec<D> {
    let lut = splat_swizzle_u64s(d, const { swizzle_mask_u64s(CLZ_LUT) });
    let hi_nib = crate::shr!(v, 4);
    let hi_clz = lut.table_lookup_bytes_or_zero(hi_nib);
    let lo_clz = lut.table_lookup_bytes_or_zero(v & U8Vec::splat(d, 0x0F)) + U8Vec::splat(d, 4);
    hi_nib.eq_zero().if_then_else(lo_clz, hi_clz)
}

#[inline(always)]
pub(crate) fn int_count_ones<T: crate::IntElement, D: SimdDescriptor>(
    v: SimdVec<T, D>,
) -> SimdVec<T, D> {
    let d = v.descriptor();
    let p8 = u8_count_ones(d, v.bitcast::<u8>());
    match std::mem::size_of::<T>() {
        1 => p8.bitcast::<T>(),
        2 => {
            let p = p8.bitcast::<u16>();
            ((p + crate::shr!(p, 8)) & U16Vec::splat(d, 0x00FF)).bitcast::<T>()
        }
        4 => {
            let p = p8.bitcast::<u32>();
            let p = (p + crate::shr!(p, 8)) & U32Vec::splat(d, 0x00FF_00FF);
            ((p + crate::shr!(p, 16)) & U32Vec::splat(d, 0x0000_FFFF)).bitcast::<T>()
        }
        8 => {
            let p = p8.bitcast::<u64>();
            let p = (p + crate::shr!(p, 8)) & U64Vec::splat(d, 0x00FF_00FF_00FF_00FF);
            let p = (p + crate::shr!(p, 16)) & U64Vec::splat(d, 0x0000_FFFF_0000_FFFF);
            ((p + crate::shr!(p, 32)) & U64Vec::splat(d, 0x0000_0000_FFFF_FFFF)).bitcast::<T>()
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
pub(crate) fn int_leading_zeros<T: crate::IntElement, D: SimdDescriptor>(
    v: SimdVec<T, D>,
) -> SimdVec<T, D> {
    let d = v.descriptor();
    let clz8 = u8_leading_zeros(d, v.bitcast::<u8>());
    if std::mem::size_of::<T>() == 1 {
        return clz8.bitcast::<T>();
    }
    let u16v = v.bitcast::<u16>();
    let c16 = clz8.bitcast::<u16>();
    let clz16 = crate::shr!(u16v, 8).eq_zero().if_then_else(
        (c16 & U16Vec::splat(d, 0xFF)) + U16Vec::splat(d, 8),
        crate::shr!(c16, 8),
    );
    if std::mem::size_of::<T>() == 2 {
        return clz16.bitcast::<T>();
    }
    let u32v = v.bitcast::<u32>();
    let c32 = clz16.bitcast::<u32>();
    let clz32 = crate::shr!(u32v, 16).eq_zero().if_then_else(
        (c32 & U32Vec::splat(d, 0xFFFF)) + U32Vec::splat(d, 16),
        crate::shr!(c32, 16),
    );
    if std::mem::size_of::<T>() == 4 {
        return clz32.bitcast::<T>();
    }
    let u64v = v.bitcast::<u64>();
    let c64 = clz32.bitcast::<u64>();
    crate::shr!(u64v, 32)
        .eq_zero()
        .if_then_else(
            (c64 & U64Vec::splat(d, 0xFFFF_FFFF)) + U64Vec::splat(d, 32),
            crate::shr!(c64, 32),
        )
        .bitcast::<T>()
}

#[inline(always)]
pub(crate) fn int_trailing_zeros<T: crate::IntElement, D: SimdDescriptor>(
    v: SimdVec<T, D>,
) -> SimdVec<T, D> {
    let d = v.descriptor();
    (!v & (v - SimdVec::splat(d, <T as IntElement>::ONE))).count_ones()
}

#[inline(always)]
fn promote_int_pair<S: IntElement, W: Element, D: SimdDescriptor>(
    d: D,
    a: S::Raw<D>,
) -> (W::Raw<D>, W::Raw<D>) {
    let ext = if S::IS_SIGNED {
        let neg_one = S::sub(d, S::zero(d), S::splat(d, S::ONE));
        S::mask_select_zero(d, S::gt(d, S::zero(d), a), neg_one)
    } else {
        S::zero(d)
    };
    let p = S::to_raw_bytes(d, S::interleave_lo(d, a, ext));
    let q = S::to_raw_bytes(d, S::interleave_hi(d, a, ext));
    let (lo, hi) = if D::BYTES == 16 {
        (p, q)
    } else if D::BYTES == 32 {
        d.interleave_128_halves(p, q)
    } else {
        let (r0, r1) = d.interleave_128_halves(p, q);
        d.interleave_256_halves(r0, r1)
    };
    (W::from_raw_bytes(d, lo), W::from_raw_bytes(d, hi))
}

#[inline(always)]
pub(crate) fn promote_int_lo<S: IntElement, W: Element, D: SimdDescriptor>(
    d: D,
    a: S::Raw<D>,
    as_w: fn(S) -> W,
) -> W::Raw<D> {
    if D::IS_EMULATED {
        let mut src = S::zero_array::<D>();
        let mut dst = W::zero_array::<D>();
        S::store_array(d, a, &mut src);
        for (o, &x) in dst.as_mut().iter_mut().zip(src.as_ref().iter()) {
            *o = as_w(x);
        }
        return W::load_array(d, &dst);
    }
    promote_int_pair::<S, W, D>(d, a).0
}

#[inline(always)]
pub(crate) fn promote_int_hi<S: IntElement, W: Element, D: SimdDescriptor>(
    d: D,
    a: S::Raw<D>,
    as_w: fn(S) -> W,
) -> W::Raw<D> {
    if D::IS_EMULATED {
        let mut src = S::zero_array::<D>();
        let mut dst = W::zero_array::<D>();
        S::store_array(d, a, &mut src);
        let half = src.as_ref().len() / 2;
        for (o, &x) in dst.as_mut().iter_mut().zip(src.as_ref()[half..].iter()) {
            *o = as_w(x);
        }
        return W::load_array(d, &dst);
    }
    promote_int_pair::<S, W, D>(d, a).1
}

#[inline(always)]
pub(crate) fn demote_int<W: ComparableElement, N: Element, D: SimdDescriptor + LaneBackend<N>>(
    d: D,
    lo: W::Raw<D>,
    hi: W::Raw<D>,
    lo_bound: W,
    hi_bound: W,
    clamp_n: fn(W) -> N,
) -> D::RawBytes {
    if D::IS_EMULATED {
        let mut a = W::zero_array::<D>();
        let mut b = W::zero_array::<D>();
        let mut dst = N::zero_array::<D>();
        W::store_array(d, lo, &mut a);
        W::store_array(d, hi, &mut b);
        let half = a.as_ref().len();
        for (i, &x) in a.as_ref().iter().enumerate() {
            dst.as_mut()[i] = clamp_n(x);
        }
        for (i, &x) in b.as_ref().iter().enumerate() {
            dst.as_mut()[half + i] = clamp_n(x);
        }
        return N::to_raw_bytes(d, N::load_array(d, &dst));
    }
    let lb = W::splat(d, lo_bound);
    let hb = W::splat(d, hi_bound);
    let c0 = W::min(d, W::max(d, lo, lb), hb);
    let c1 = W::min(d, W::max(d, hi, lb), hb);
    let b0 = <D as LaneBackend<N>>::from_raw_bytes(d, W::to_raw_bytes(d, c0));
    let b1 = <D as LaneBackend<N>>::from_raw_bytes(d, W::to_raw_bytes(d, c1));
    <D as LaneBackend<N>>::to_raw_bytes(d, deinterleave_2_vecs::<N, D>(d, b0, b1).0)
}

#[inline(always)]
pub(crate) fn f64_as_i64_sw<D: SimdDescriptor>(
    d: D,
    a: <D as LaneBackend<f64>>::Raw,
) -> <D as LaneBackend<i64>>::Raw {
    let mut src = <D as LaneBackend<f64>>::zero_array();
    let mut dst = <D as LaneBackend<i64>>::zero_array();
    <D as LaneBackend<f64>>::store_array(d, a, &mut src);
    for (o, &x) in dst.as_mut().iter_mut().zip(src.as_ref().iter()) {
        *o = x as i64;
    }
    <D as LaneBackend<i64>>::load_array(d, &dst)
}

#[inline(always)]
pub(crate) fn i64_as_f64_sw<D: SimdDescriptor>(
    d: D,
    a: <D as LaneBackend<i64>>::Raw,
) -> <D as LaneBackend<f64>>::Raw {
    let mut src = <D as LaneBackend<i64>>::zero_array();
    let mut dst = <D as LaneBackend<f64>>::zero_array();
    <D as LaneBackend<i64>>::store_array(d, a, &mut src);
    for (o, &x) in dst.as_mut().iter_mut().zip(src.as_ref().iter()) {
        *o = x as f64;
    }
    <D as LaneBackend<f64>>::load_array(d, &dst)
}

#[inline(always)]
pub(crate) fn prepare_table_bf16_8_128<D: SimdDescriptor128>(d: D, table: &[f32; 8]) -> U8Vec<D> {
    let bf16 = table.map(|v| (v.to_bits() >> 16) as u16);
    U16Vec::load(d, &bf16).bitcast::<u8>()
}

#[inline(always)]
pub(crate) fn table_lookup_bf16_8_128<D: SimdDescriptor>(
    d: D,
    table_bytes: U8Vec<D>,
    indices: I32Vec<D>,
) -> F32Vec<D> {
    let idx = indices.bitcast::<u32>();
    let byte_idx =
        (idx.shl::<17>() | idx.shl::<25>() | U32Vec::splat(d, 0x0100_8080)).bitcast::<u8>();
    table_bytes
        .table_lookup_bytes_or_zero(byte_idx)
        .bitcast::<f32>()
}
