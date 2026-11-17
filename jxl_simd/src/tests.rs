// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    ComparableElement, Emu128Descriptor, F32Vec, FloatElement, I16Vec, I32Vec, IntElement,
    MaskWidth, MulElement, SignedElement, SimdDescriptor, SimdElement, SimdMask, SimdVec, U8Vec,
    U64Vec, W8, W16, W32, W64, simd_test,
};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn fill_inputs<T: Copy>(
    seed: u64,
    n: usize,
    boundary: &[T],
    mut from_rng: impl FnMut(&mut Rng) -> T,
) -> Vec<T> {
    let mut rng = Rng(seed.max(1));
    let total = n.max(boundary.len()).next_multiple_of(64);
    (0..total)
        .map(|i| {
            if i < boundary.len() {
                boundary[i]
            } else {
                from_rng(&mut rng)
            }
        })
        .collect()
}

trait TestScalar: SimdElement + PartialEq {
    fn boundary() -> Vec<Self>;
    fn from_rng(rng: &mut Rng) -> Self;
    fn check_eq(expected: Self, actual: Self, ctx: &str) {
        assert_eq!(expected, actual, "{ctx}");
    }
}

impl TestScalar for f32 {
    fn boundary() -> Vec<Self> {
        vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -1.5, -2.5, 123.75, -123.75, 1e-4, -1e4,
        ]
    }
    fn from_rng(rng: &mut Rng) -> Self {
        let m = (rng.next_u64() & 0xFFFF) as f32 / 1024.0;
        let s = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        s * (0.25 + m)
    }
    fn check_eq(e: Self, a: Self, ctx: &str) {
        if e == 0.0 && a == 0.0 {
            return;
        }
        assert!(
            (e - a).abs() <= 1e-6 * (1.0 + e.abs().max(a.abs())),
            "{ctx}: expected {e}, got {a}"
        );
    }
}

macro_rules! impl_test_scalar_int {
    ($($t:ty),*) => {
        $(impl TestScalar for $t {
            fn boundary() -> Vec<Self> {
                vec![0, 1, <$t>::MIN, <$t>::MAX, <$t>::MIN.wrapping_add(1), <$t>::MAX.wrapping_sub(1), 7 as $t, (!0u64) as $t, 0x55u64 as $t, 0xAAu64 as $t]
            }
            fn from_rng(rng: &mut Rng) -> Self {
                rng.next_u64() as $t
            }
        })*
    };
}
impl_test_scalar_int!(u8, i16, u16, i32, u32, u64);

macro_rules! check_against_emu {
    ($d:expr, $t:ty, $d_ty:ty, $a:expr, $b:expr, $c:expr, |$darg:ident, $va:ident, $vb:ident, $vc:ident| $body:expr, $ctx:expr) => {{
        let ed = Emu128Descriptor::new().unwrap();
        let elen = SimdVec::<$t, Emu128Descriptor>::LEN;
        let dlen = SimdVec::<$t, $d_ty>::LEN;
        let mut out_d = vec![<$t>::default(); dlen];
        let mut out_e = vec![<$t>::default(); dlen];
        for off in (0..$a.len().min($b.len()).min($c.len()) - dlen + 1).step_by(dlen) {
            let va = SimdVec::<$t, $d_ty>::load($d, &$a[off..]);
            let vb = SimdVec::<$t, $d_ty>::load($d, &$b[off..]);
            let vc = SimdVec::<$t, $d_ty>::load($d, &$c[off..]);
            let res_d = {
                let ($darg, $va, $vb, $vc) = ($d, va, vb, vc);
                $body
            };
            res_d.store(&mut out_d);
            for chunk in (0..dlen).step_by(elen) {
                let eva = SimdVec::<$t, Emu128Descriptor>::load(ed, &$a[off + chunk..]);
                let evb = SimdVec::<$t, Emu128Descriptor>::load(ed, &$b[off + chunk..]);
                let evc = SimdVec::<$t, Emu128Descriptor>::load(ed, &$c[off + chunk..]);
                let res_e = {
                    let ($darg, $va, $vb, $vc) = (ed, eva, evb, evc);
                    $body
                };
                res_e.store(&mut out_e[chunk..]);
            }
            for i in 0..dlen {
                TestScalar::check_eq(out_e[i], out_d[i], $ctx);
            }
        }
    }};
}

fn test_lane_arith_cmp<T: TestScalar + ComparableElement + MulElement, D: SimdDescriptor>(
    d: D,
    seed: u64,
) {
    let bnd = T::boundary();
    let a = fill_inputs(seed, 64, &bnd, T::from_rng);
    let rev: Vec<T> = bnd.iter().copied().rev().collect();
    let b = fill_inputs(seed ^ 0x9E37_79B9_7F4A_7C15, 64, &rev, T::from_rng);
    let c = fill_inputs(seed ^ 0xBF58_476D_1CE4_E5B9, 64, &bnd, T::from_rng);

    check_against_emu!(d, T, D, a, b, c, |_d, va, vb, _vc| va + vb, "add");
    check_against_emu!(d, T, D, a, b, c, |_d, va, vb, _vc| va - vb, "sub");
    check_against_emu!(d, T, D, a, b, c, |_d, va, vb, _vc| va * vb, "mul");
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, _vc| {
            let mut x = va;
            x += vb;
            x -= va;
            x *= vb;
            x
        },
        "assign"
    );
    check_against_emu!(d, T, D, a, b, c, |_d, va, vb, _vc| va.min(vb), "min");
    check_against_emu!(d, T, D, a, b, c, |_d, va, vb, _vc| va.max(vb), "max");
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, vc| va.eq(vb).if_then_else(va, vc),
        "eq"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, vc| va.gt(vb).if_then_else(va, vc),
        "gt"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, _vc| va.eq_zero().if_then_zero_else(vb),
        "eqz_zero_else"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, _vb, vc| va.eq(va).if_then_zero_else(vc),
        "zero_else"
    );

    let dlen = SimdVec::<T, D>::LEN;
    let va = SimdVec::load(d, &a);
    assert_eq!(SimdVec::<T, _>::zero(d).extract_lane0(), T::default());
    assert_eq!(SimdVec::splat(d, a[3]).extract_lane0(), a[3]);
    assert_eq!(va.extract_lane0(), a[0]);

    let mut arr = SimdVec::<T, D>::make_array_slice(&a[..dlen])[0];
    SimdVec::<T, D>::load_array(d, &arr)
        .bitcast::<u8>()
        .bitcast::<T>()
        .store_array(&mut arr);
    assert_eq!(arr.as_ref(), &a[..dlen]);
    let mut dst_mut = vec![T::default(); dlen];
    SimdVec::<T, D>::make_array_slice_mut(&mut dst_mut)[0] = arr;
    assert_eq!(dst_mut, a[..dlen]);
}

#[inline]
fn test_interleave<T: TestScalar + PartialEq, D: SimdDescriptor>(d: D, seed: u64) {
    let bnd = T::boundary();
    let len = SimdVec::<T, D>::LEN;
    let total = 4 * len;
    let src = fill_inputs(seed, total, &bnd, T::from_rng);
    let va = SimdVec::load(d, &src);
    let vb = SimdVec::load(d, &src[len..]);
    let vc = SimdVec::load(d, &src[2 * len..]);
    let vd = SimdVec::load(d, &src[3 * len..]);

    let mut buf2 = vec![T::default(); 2 * len];
    SimdVec::store_interleaved_2(va, vb, &mut buf2);
    for i in 0..len {
        assert_eq!(buf2[2 * i], src[i]);
        assert_eq!(buf2[2 * i + 1], src[len + i]);
    }

    let mut buf3 = vec![T::default(); 3 * len];
    SimdVec::store_interleaved_3(va, vb, vc, &mut buf3);
    for i in 0..len {
        assert_eq!(buf3[3 * i], src[i]);
        assert_eq!(buf3[3 * i + 1], src[len + i]);
        assert_eq!(buf3[3 * i + 2], src[2 * len + i]);
    }

    let mut buf4 = vec![T::default(); 4 * len];
    SimdVec::store_interleaved_4(va, vb, vc, vd, &mut buf4);
    for i in 0..len {
        assert_eq!(buf4[4 * i], src[i]);
        assert_eq!(buf4[4 * i + 1], src[len + i]);
        assert_eq!(buf4[4 * i + 2], src[2 * len + i]);
        assert_eq!(buf4[4 * i + 3], src[3 * len + i]);
    }

    let mut out = vec![T::default(); len];
    let (b128_0, b128_1) = va.interleave_blocks_128(vb);
    let mut out2 = vec![T::default(); len];
    b128_0.store(&mut out);
    b128_1.store(&mut out2);
    if D::BYTES == 16 {
        assert_eq!(out, src[..len]);
        assert_eq!(out2, src[len..2 * len]);
    } else {
        let blk = 16 / std::mem::size_of::<T>();
        for pair in (0..D::BYTES / 16).step_by(2) {
            assert_eq!(
                &out[pair * blk..(pair + 1) * blk],
                &src[pair * blk..(pair + 1) * blk]
            );
            assert_eq!(
                &out[(pair + 1) * blk..(pair + 2) * blk],
                &src[len + pair * blk..len + (pair + 1) * blk]
            );
            assert_eq!(
                &out2[pair * blk..(pair + 1) * blk],
                &src[(pair + 1) * blk..(pair + 2) * blk]
            );
            assert_eq!(
                &out2[(pair + 1) * blk..(pair + 2) * blk],
                &src[len + (pair + 1) * blk..len + (pair + 2) * blk]
            );
        }
    }

    let (b256_0, b256_1) = va.interleave_blocks_256(vb);
    b256_0.store(&mut out);
    b256_1.store(&mut out2);
    if D::BYTES <= 32 {
        assert_eq!(out, src[..len]);
        assert_eq!(out2, src[len..2 * len]);
    } else {
        let half = len / 2;
        assert_eq!(&out[..half], &src[..half]);
        assert_eq!(&out[half..], &src[len..len + half]);
        assert_eq!(&out2[..half], &src[half..len]);
        assert_eq!(&out2[half..], &src[len + half..2 * len]);
    }
}

#[inline]
fn test_signed_ops<T: TestScalar + SignedElement + MulElement, D: SimdDescriptor>(d: D, seed: u64) {
    let bnd = T::boundary();
    let a = fill_inputs(seed, 64, &bnd, T::from_rng);
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| -va, "neg_op");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.abs(), "abs");
    check_against_emu!(
        d,
        T,
        D,
        a,
        a,
        a,
        |_d, va, vb, vc| va.lt_zero().if_then_else(vb, vc),
        "lt_zero"
    );
}

#[inline]
fn test_float_ops<T, D: SimdDescriptor>(d: D, seed: u64)
where
    T: TestScalar + FloatElement + PartialOrd + std::ops::Neg<Output = T>,
    f64: From<T>,
{
    let bnd = T::boundary();
    let a = fill_inputs(seed, 64, &bnd, T::from_rng);
    let b = fill_inputs(seed ^ 17, 64, &a[..8], T::from_rng);
    let c = fill_inputs(seed ^ 33, 64, &bnd, T::from_rng);
    let nonzero: Vec<T> = b
        .iter()
        .map(|&v| if v == T::default() { bnd[2] } else { v })
        .collect();
    let pos: Vec<T> = a
        .iter()
        .map(|&v| if v < T::default() { -v } else { v })
        .collect();

    check_against_emu!(d, T, D, a, nonzero, c, |_d, va, vb, _vc| va / vb, "div_op");
    check_against_emu!(d, T, D, pos, a, a, |_d, va, _vb, _vc| va.sqrt(), "sqrt");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.floor(), "floor");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.ceil(), "ceil");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.round(), "round");
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        a,
        |_d, va, vb, _vc| va.copysign(vb),
        "copysign"
    );

    let len = SimdVec::<T, D>::LEN;
    let va = SimdVec::load(d, &a);
    let vb = SimdVec::load(d, &b);
    let vc = SimdVec::load(d, &c);
    let mut fma_out = vec![T::default(); len];
    let mut fnma_out = vec![T::default(); len];
    va.mul_add(vb, vc).store(&mut fma_out);
    va.neg_mul_add(vb, vc).store(&mut fnma_out);
    for i in 0..len {
        let (x, y, z) = (f64::from(a[i]), f64::from(b[i]), f64::from(c[i]));
        let scale = (x * y).abs() + z.abs();
        let e1 = x.mul_add(y, z);
        let e2 = (-x).mul_add(y, z);
        assert!((f64::from(fma_out[i]) - e1).abs() <= 1e-5 * (1.0 + scale));
        assert!((f64::from(fnma_out[i]) - e2).abs() <= 1e-5 * (1.0 + scale));
    }
}

#[inline]
fn test_int_ops<T: TestScalar + IntElement, D: SimdDescriptor, const MID: u32, const HI: u32>(
    d: D,
    seed: u64,
) {
    let bnd = T::boundary();
    let a = fill_inputs(seed, 64, &bnd, T::from_rng);
    let b = fill_inputs(seed ^ 55, 64, &bnd, T::from_rng);

    check_against_emu!(d, T, D, a, b, b, |_d, va, vb, _vc| va & vb, "band");
    check_against_emu!(d, T, D, a, b, b, |_d, va, vb, _vc| va | vb, "bor");
    check_against_emu!(d, T, D, a, b, b, |_d, va, vb, _vc| va ^ vb, "bxor");
    check_against_emu!(d, T, D, a, b, b, |_d, va, _vb, _vc| !va, "bnot");
    check_against_emu!(d, T, D, a, b, b, |_d, va, vb, _vc| va.andnot(vb), "andnot");
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| crate::shl!(va, 0),
        "shl0"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| crate::shr!(va, 0),
        "shr0"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| crate::shl!(va, 1),
        "shl1"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| crate::shr!(va, 1),
        "shr1"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| va.shl::<MID>(),
        "shl_mid"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| va.shr::<MID>(),
        "shr_mid"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| va.shl::<HI>(),
        "shl_hi"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        b,
        |_d, va, _vb, _vc| va.shr::<HI>(),
        "shr_hi"
    );
}

#[inline]
fn test_masks<W: MaskWidth, T: TestScalar + ComparableElement<Mask = W>, D: SimdDescriptor>(d: D) {
    let bnd = T::boundary();
    let len = SimdVec::<T, D>::LEN;
    let (zeros, ones) = (vec![T::default(); len], vec![bnd[1]; len]);
    let mut mixed = zeros.clone();
    mixed[0] = bnd[1];
    let v0 = SimdVec::load(d, &zeros);
    let v1 = SimdVec::load(d, &ones);
    let vm = SimdVec::load(d, &mixed);
    let m_false = v0.gt(v1);
    let m_true = v1.gt(v0);
    let m_one = vm.gt(v0);
    assert!(!m_false.all() && !m_false.any());
    assert!(m_true.all() && m_true.any());
    assert!(!m_one.all() && m_one.any());
    let m_true2 = m_true;
    assert!(
        (m_one & m_true).any() && (m_one | m_false).any() && (m_true ^ m_true2).eq_zero_check()
    );
    let mut mc = m_one;
    mc &= m_true;
    mc |= m_false;
    mc ^= !m_true;
    assert_eq!(mc.any(), m_one.any());
    assert!(m_one.andnot(m_true).any());
    assert!(!m_one.andnot(m_one).any());
}

trait MaskEqZeroCheck {
    fn eq_zero_check(self) -> bool;
}
impl<W: MaskWidth, D: SimdDescriptor> MaskEqZeroCheck for SimdMask<W, D> {
    fn eq_zero_check(self) -> bool {
        !self.any()
    }
}

#[simd_test]
fn test_all_types_and_ops<D: SimdDescriptor>(d: D) {
    test_lane_arith_cmp::<f32, D>(d, 1);
    test_lane_arith_cmp::<u8, D>(d, 4);
    test_lane_arith_cmp::<i16, D>(d, 5);
    test_lane_arith_cmp::<u16, D>(d, 6);
    test_lane_arith_cmp::<i32, D>(d, 7);
    test_lane_arith_cmp::<u32, D>(d, 8);
    test_lane_arith_cmp::<u64, D>(d, 10);

    test_interleave::<f32, D>(d, 11);
    test_interleave::<u8, D>(d, 14);
    test_interleave::<i16, D>(d, 15);
    test_interleave::<u16, D>(d, 16);
    test_interleave::<i32, D>(d, 17);
    test_interleave::<u32, D>(d, 18);
    test_interleave::<u64, D>(d, 20);

    test_signed_ops::<f32, D>(d, 21);
    test_signed_ops::<i16, D>(d, 24);
    test_signed_ops::<i32, D>(d, 25);

    test_float_ops::<f32, D>(d, 27);

    test_int_ops::<u8, D, 3, 7>(d, 30);
    test_int_ops::<i16, D, 7, 15>(d, 31);
    test_int_ops::<u16, D, 7, 15>(d, 32);
    test_int_ops::<i32, D, 15, 31>(d, 33);
    test_int_ops::<u32, D, 15, 31>(d, 34);
    test_int_ops::<u64, D, 31, 63>(d, 36);

    test_masks::<W8, u8, D>(d);
    test_masks::<W16, i16, D>(d);
    test_masks::<W32, i32, D>(d);
    test_masks::<W64, u64, D>(d);
}

#[simd_test]
fn test_extra_conversions_and_lut<D: SimdDescriptor>(d: D) {
    let flen = F32Vec::<D>::LEN;
    let fvals: Vec<f32> = (0..flen).map(|i| (i as f32) * 3.5 + 0.5).collect();
    let (mut u8o, mut u16o, mut i32o) = (vec![0u8; flen], vec![0u16; flen], vec![0i32; flen]);
    let fv = F32Vec::load(d, &fvals);
    fv.round_store_u8(&mut u8o);
    fv.round_store_u16(&mut u16o);
    fv.as_i32().store(&mut i32o);
    for i in 0..flen {
        assert_eq!(u8o[i], fvals[i].round_ties_even() as u8);
        assert_eq!(u16o[i], fvals[i].round_ties_even() as u16);
        assert_eq!(i32o[i], fvals[i] as i32);
    }
    let mut back_f = vec![0.0f32; flen];
    I32Vec::load(d, &i32o).as_f32().store(&mut back_f);
    assert_eq!(back_f, i32o.iter().map(|&x| x as f32).collect::<Vec<_>>());

    let src16 = [
        -32768i16, -1, 0, 1, 42, 32767, 100, -100, 1, 2, 3, 4, 5, 6, 7, 8,
    ];
    let srcu16 = [0u16, 1, 65535, 32768, 12, 99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    I32Vec::load_from_i16(d, &src16).store(&mut i32o);
    assert_eq!(
        i32o,
        src16[..flen].iter().map(|&x| x as i32).collect::<Vec<_>>()
    );
    I32Vec::load_from_u16(d, &srcu16).store(&mut i32o);
    assert_eq!(
        i32o,
        srcu16[..flen].iter().map(|&x| x as i32).collect::<Vec<_>>()
    );
    let iv_src = [
        0x1000_0001i32,
        -0x2000_0000,
        0x7FFF_FFFF,
        i32::MIN,
        4,
        -5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    ];
    let iv = I32Vec::load(d, &iv_src);
    iv.store_u16(&mut u16o);
    iv.store_u8(&mut u8o);
    iv.mul_wide_take_high(iv).store(&mut i32o);
    for i in 0..flen {
        assert_eq!(u16o[i], iv_src[i] as u16);
        assert_eq!(u8o[i], iv_src[i] as u8);
        assert_eq!(
            i32o[i],
            ((iv_src[i] as i64 * iv_src[i] as i64) >> 32) as i32
        );
    }

    let i16_src = fill_inputs(53, I16Vec::<D>::LEN, &src16, i16::from_rng);
    let i16v = I16Vec::load(d, &i16_src);
    let mut i16u8 = vec![0u8; I16Vec::<D>::LEN];
    let mut i16hi = vec![0i16; I16Vec::<D>::LEN];
    i16v.store_u8(&mut i16u8);
    i16v.mul_wide_take_high(i16v).store(&mut i16hi);
    for i in 0..I16Vec::<D>::LEN {
        assert_eq!(i16u8[i], i16_src[i] as u8);
        assert_eq!(
            i16hi[i],
            ((i16_src[i] as i32 * i16_src[i] as i32) >> 16) as i16
        );
    }

    let mut s8 = vec![0.0f32; 8 * flen];
    F32Vec::store_interleaved_8(fv, fv, fv, fv, fv, fv, fv, fv, &mut s8);
    for i in 0..flen {
        for c in 0..8 {
            assert_eq!(s8[8 * i + c], fvals[i]);
        }
    }

    let blen = U8Vec::<D>::LEN;
    let tbl: Vec<u8> = (0..blen).map(|i| (i * 7 + 13) as u8).collect();
    let idx_cases = [0u8, 15, 16, 17, 31, 127, 128, 255];
    let mut idx = vec![0u8; blen];
    for i in 0..blen {
        idx[i] = idx_cases[i % idx_cases.len()];
    }
    let mut out_u8 = vec![0u8; blen];
    U8Vec::load(d, &tbl)
        .table_lookup_bytes_or_zero(U8Vec::load(d, &idx))
        .store(&mut out_u8);
    for i in 0..blen {
        let base = (i / 16) * 16;
        let expected = if idx[i] < 16 {
            tbl[base + idx[i] as usize]
        } else {
            0
        };
        assert_eq!(out_u8[i], expected, "u8 lut idx={}", idx[i]);
    }

    // Also test constant shifts across wider shift amounts (`0`, mid, `BITS-1`).
    assert_eq!(crate::shr!(U8Vec::splat(d, 0x80), 7).extract_lane0(), 1);
    assert_eq!(
        crate::shl!(I16Vec::splat(d, 1), 15).extract_lane0(),
        i16::MIN
    );
    assert_eq!(
        crate::shr!(U64Vec::splat(d, u64::MAX), 63).extract_lane0(),
        1
    );
}

#[simd_test]
fn test_load_store_partial<D: SimdDescriptor>(d: D) {
    let len = F32Vec::<D>::LEN;
    let src: Vec<f32> = (1..=(len + 3)).map(|i| i as f32).collect();
    for count in [0, 1, len / 2, len - 1, len, len + 2] {
        let mut full = vec![-1.0f32; len];
        F32Vec::load_partial(d, &src[..count]).store(&mut full);
        for i in 0..len {
            assert_eq!(full[i], if i < count { src[i] } else { 0.0 });
        }
        let mut dst = vec![-99.0f32; len + 4];
        let store_count = count.min(len);
        F32Vec::load(d, &src).store_partial(&mut dst[..store_count]);
        for (i, &val) in dst.iter().enumerate() {
            assert_eq!(val, if i < store_count { src[i] } else { -99.0 });
        }
    }
}

#[simd_test]
fn test_prepare_table_bf16_8<D: SimdDescriptor>(d: D) {
    let lut: [f32; 8] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    let len = F32Vec::<D>::LEN;
    let prepared = F32Vec::prepare_table_bf16_8(d, &lut);
    let indices: Vec<i32> = (0..len).map(|i| (i % 8) as i32).collect();
    let mut output = vec![0.0f32; len];
    F32Vec::table_lookup_bf16_8(d, prepared, I32Vec::load(d, &indices)).store(&mut output);
    for i in 0..len {
        let exp = lut[indices[i] as usize];
        assert!((output[i] - exp).abs() < 0.01 * exp.max(1.0));
    }
}

#[simd_test]
fn test_f16_all_u16_values<D: SimdDescriptor>(d: D) {
    let len = F32Vec::<D>::LEN;
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let (mut f32_buf, mut rt_buf) = (vec![0.0f32; len], vec![0u16; len]);
    for chunk in all.chunks_exact(len) {
        let v = F32Vec::load_f16_bits(d, chunk);
        v.store(&mut f32_buf);
        v.store_f16_bits(&mut rt_buf);
        for i in 0..len {
            let s = crate::f16::from_bits(chunk[i]).to_f32();
            if s.is_nan() {
                assert!(f32_buf[i].is_nan());
                assert_eq!((rt_buf[i] >> 10) & 0x1F, 0x1F);
                assert_ne!(rt_buf[i] & 0x03FF, 0);
            } else {
                assert_eq!(f32_buf[i].to_bits(), s.to_bits());
                assert_eq!(rt_buf[i], chunk[i]);
            }
        }
    }
}

#[simd_test]
fn test_f16_store_scalar_equivalent<D: SimdDescriptor>(d: D) {
    let len = F32Vec::<D>::LEN;
    let src = fill_inputs(999, len * 4, &f32::boundary(), f32::from_rng);
    let mut out = vec![0u16; len];
    for chunk in src.chunks_exact(len) {
        F32Vec::load(d, chunk).store_f16_bits(&mut out);
        for i in 0..len {
            assert_eq!(out[i], crate::f16::from_f32(chunk[i]).to_bits());
        }
    }
}

#[simd_test]
fn test_call_and_inlined<D: SimdDescriptor>(d: D) {
    let a = d.call(|d| F32Vec::splat(d, 3.0) + F32Vec::splat(d, 4.0));
    let b = d.call_inlined(|d| F32Vec::splat(d, 2.0) * F32Vec::splat(d, 5.0));
    assert_eq!(a.extract_lane0(), 7.0);
    assert_eq!(b.extract_lane0(), 10.0);
}
