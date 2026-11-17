// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    ArithElement, ComparableElement, DemoteTo, Emu128Descriptor, F32Vec, F64Vec, FloatElement,
    I8Vec, I16Vec, I32Vec, I64Vec, IntElement, MaskWidth, MulElement, PromoteTo, SatElement,
    SignedElement, SimdDescriptor, SimdElement, SimdMask, SimdVec, U8Vec, U16Vec, U32Vec, U64Vec,
    W8, W16, W32, W64, simd_specialize, simd_test,
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

impl TestScalar for f64 {
    fn boundary() -> Vec<Self> {
        vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -1.5, -2.5, 123.75, -123.75, 1e-8, -1e8,
        ]
    }
    fn from_rng(rng: &mut Rng) -> Self {
        let m = (rng.next_u64() & 0xFFFF_FFFF) as f64 / 65536.0;
        let s = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        s * (0.25 + m)
    }
    fn check_eq(e: Self, a: Self, ctx: &str) {
        if e == 0.0 && a == 0.0 {
            return;
        }
        assert!(
            (e - a).abs() <= 1e-12 * (1.0 + e.abs().max(a.abs())),
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
impl_test_scalar_int!(i8, u8, i16, u16, i32, u32, i64, u64);

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
        |_d, va, vb, vc| va.lt(vb).if_then_else(va, vc),
        "lt"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, vc| va.ge(vb).if_then_else(va, vc),
        "ge"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, vc| va.le(vb).if_then_else(va, vc),
        "le"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        c,
        |_d, va, vb, _vc| va.eq_zero().if_then_else_zero(vb),
        "eqz_zero"
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
    let (da, db) = SimdVec::load_deinterleaved_2(d, &buf2);
    let mut out = vec![T::default(); len];
    da.store(&mut out);
    assert_eq!(out, src[..len]);
    db.store(&mut out);
    assert_eq!(out, src[len..2 * len]);

    let mut buf3 = vec![T::default(); 3 * len];
    SimdVec::store_interleaved_3(va, vb, vc, &mut buf3);
    for i in 0..len {
        assert_eq!(buf3[3 * i], src[i]);
        assert_eq!(buf3[3 * i + 1], src[len + i]);
        assert_eq!(buf3[3 * i + 2], src[2 * len + i]);
    }
    let (la, lb, lc) = SimdVec::load_deinterleaved_3(d, &buf3);
    la.store(&mut out);
    assert_eq!(out, src[..len]);
    lb.store(&mut out);
    assert_eq!(out, src[len..2 * len]);
    lc.store(&mut out);
    assert_eq!(out, src[2 * len..3 * len]);

    let mut buf4 = vec![T::default(); 4 * len];
    SimdVec::store_interleaved_4(va, vb, vc, vd, &mut buf4);
    for i in 0..len {
        assert_eq!(buf4[4 * i], src[i]);
        assert_eq!(buf4[4 * i + 1], src[len + i]);
        assert_eq!(buf4[4 * i + 2], src[2 * len + i]);
        assert_eq!(buf4[4 * i + 3], src[3 * len + i]);
    }
    let (fa, fb, fc, fd) = SimdVec::load_deinterleaved_4(d, &buf4);
    fa.store(&mut out);
    assert_eq!(out, src[..len]);
    fb.store(&mut out);
    assert_eq!(out, src[len..2 * len]);
    fc.store(&mut out);
    assert_eq!(out, src[2 * len..3 * len]);
    fd.store(&mut out);
    assert_eq!(out, src[3 * len..4 * len]);

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
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.neg(), "neg_fn");
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
    check_against_emu!(
        d,
        T,
        D,
        a,
        nonzero,
        c,
        |_d, va, vb, _vc| va.div(vb),
        "div_fn"
    );
    check_against_emu!(d, T, D, pos, a, a, |_d, va, _vb, _vc| va.sqrt(), "sqrt");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.floor(), "floor");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.ceil(), "ceil");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.round(), "round");
    check_against_emu!(d, T, D, a, a, a, |_d, va, _vb, _vc| va.trunc(), "trunc");
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
    let cnts: Vec<T> = (0..a.len())
        .map(|i| T::from_u8((i as u32 % <T as crate::sealed::IntElement>::BITS) as u8))
        .collect();

    check_against_emu!(d, T, D, a, b, cnts, |_d, va, vb, _vc| va & vb, "band");
    check_against_emu!(d, T, D, a, b, cnts, |_d, va, vb, _vc| va | vb, "bor");
    check_against_emu!(d, T, D, a, b, cnts, |_d, va, vb, _vc| va ^ vb, "bxor");
    check_against_emu!(d, T, D, a, b, cnts, |_d, va, _vb, _vc| !va, "bnot");
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, vb, _vc| va.andnot(vb),
        "andnot"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| crate::shl!(va, 0),
        "shl0"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| crate::shr!(va, 0),
        "shr0"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| crate::shl!(va, 1),
        "shl1"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| crate::shr!(va, 1),
        "shr1"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shl::<MID>(),
        "shl_mid"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shr::<MID>(),
        "shr_mid"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shl::<HI>(),
        "shl_hi"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shr::<HI>(),
        "shr_hi"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shl_vec(_vc),
        "shl_vec"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.shr_vec(_vc),
        "shr_vec"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.count_ones(),
        "count_ones"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.leading_zeros(),
        "clz"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        cnts,
        |_d, va, _vb, _vc| va.trailing_zeros(),
        "ctz"
    );
}

#[inline]
fn test_sat_ops<T: TestScalar + SatElement, D: SimdDescriptor>(d: D, seed: u64) {
    let bnd = T::boundary();
    let a = fill_inputs(seed, 64, &bnd, T::from_rng);
    let b = fill_inputs(seed ^ 77, 64, &bnd, T::from_rng);
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        a,
        |_d, va, vb, _vc| va.sat_add(vb),
        "sat_add"
    );
    check_against_emu!(
        d,
        T,
        D,
        a,
        b,
        a,
        |_d, va, vb, _vc| va.sat_sub(vb),
        "sat_sub"
    );
}

#[inline]
fn test_promote_demote<N, W, D: SimdDescriptor>(
    d: D,
    seed: u64,
    expected_lo_hi: fn(&[N]) -> Vec<W>,
    expected_demote: fn(&[W]) -> Vec<N>,
) where
    N: TestScalar + PromoteTo<W>,
    W: TestScalar + DemoteTo<N>,
{
    let nlen = SimdVec::<N, D>::LEN;
    let wlen = SimdVec::<W, D>::LEN;
    let src = fill_inputs(seed, nlen, &N::boundary(), N::from_rng);
    let vn = SimdVec::load(d, &src);
    let (mut lo, mut hi) = (vec![W::default(); wlen], vec![W::default(); wlen]);
    vn.promote_lo().store(&mut lo);
    vn.promote_hi().store(&mut hi);
    let exp = expected_lo_hi(&src[..nlen]);
    assert_eq!(lo, exp[..wlen]);
    assert_eq!(hi, exp[wlen..]);

    let wide = fill_inputs(seed ^ 99, 2 * wlen, &W::boundary(), W::from_rng);
    let (vlo, vhi) = (
        SimdVec::<W, D>::load(d, &wide),
        SimdVec::<W, D>::load(d, &wide[wlen..]),
    );
    let mut dem = vec![N::default(); nlen];
    vlo.demote_to(vhi).store(&mut dem);
    assert_eq!(dem, expected_demote(&wide[..2 * wlen]));
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
    test_lane_arith_cmp::<f64, D>(d, 2);
    test_lane_arith_cmp::<i8, D>(d, 3);
    test_lane_arith_cmp::<u8, D>(d, 4);
    test_lane_arith_cmp::<i16, D>(d, 5);
    test_lane_arith_cmp::<u16, D>(d, 6);
    test_lane_arith_cmp::<i32, D>(d, 7);
    test_lane_arith_cmp::<u32, D>(d, 8);
    test_lane_arith_cmp::<i64, D>(d, 9);
    test_lane_arith_cmp::<u64, D>(d, 10);

    test_interleave::<f32, D>(d, 11);
    test_interleave::<f64, D>(d, 12);
    test_interleave::<i8, D>(d, 13);
    test_interleave::<u8, D>(d, 14);
    test_interleave::<i16, D>(d, 15);
    test_interleave::<u16, D>(d, 16);
    test_interleave::<i32, D>(d, 17);
    test_interleave::<u32, D>(d, 18);
    test_interleave::<i64, D>(d, 19);
    test_interleave::<u64, D>(d, 20);

    test_signed_ops::<f32, D>(d, 21);
    test_signed_ops::<f64, D>(d, 22);
    test_signed_ops::<i8, D>(d, 23);
    test_signed_ops::<i16, D>(d, 24);
    test_signed_ops::<i32, D>(d, 25);
    test_signed_ops::<i64, D>(d, 26);

    test_float_ops::<f32, D>(d, 27);
    test_float_ops::<f64, D>(d, 28);

    test_int_ops::<i8, D, 3, 7>(d, 29);
    test_int_ops::<u8, D, 3, 7>(d, 30);
    test_int_ops::<i16, D, 7, 15>(d, 31);
    test_int_ops::<u16, D, 7, 15>(d, 32);
    test_int_ops::<i32, D, 15, 31>(d, 33);
    test_int_ops::<u32, D, 15, 31>(d, 34);
    test_int_ops::<i64, D, 31, 63>(d, 35);
    test_int_ops::<u64, D, 31, 63>(d, 36);

    test_sat_ops::<i8, D>(d, 37);
    test_sat_ops::<u8, D>(d, 38);
    test_sat_ops::<i16, D>(d, 39);
    test_sat_ops::<u16, D>(d, 40);

    test_masks::<W8, i8, D>(d);
    test_masks::<W16, i16, D>(d);
    test_masks::<W32, i32, D>(d);
    test_masks::<W64, i64, D>(d);

    test_promote_demote::<f32, f64, D>(
        d,
        41,
        |s| s.iter().map(|&x| x as f64).collect(),
        |w| w.iter().map(|&x| x as f32).collect(),
    );
    test_promote_demote::<i8, i16, D>(
        d,
        42,
        |s| s.iter().map(|&x| x as i16).collect(),
        |w| {
            w.iter()
                .map(|&x| x.clamp(i8::MIN as i16, i8::MAX as i16) as i8)
                .collect()
        },
    );
    test_promote_demote::<u8, u16, D>(
        d,
        43,
        |s| s.iter().map(|&x| x as u16).collect(),
        |w| w.iter().map(|&x| x.min(u8::MAX as u16) as u8).collect(),
    );
    test_promote_demote::<i16, i32, D>(
        d,
        44,
        |s| s.iter().map(|&x| x as i32).collect(),
        |w| {
            w.iter()
                .map(|&x| x.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
                .collect()
        },
    );
    test_promote_demote::<u16, u32, D>(
        d,
        45,
        |s| s.iter().map(|&x| x as u32).collect(),
        |w| w.iter().map(|&x| x.min(u16::MAX as u32) as u16).collect(),
    );
    test_promote_demote::<i32, i64, D>(
        d,
        46,
        |s| s.iter().map(|&x| x as i64).collect(),
        |w| {
            w.iter()
                .map(|&x| x.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                .collect()
        },
    );
    test_promote_demote::<u32, u64, D>(
        d,
        47,
        |s| s.iter().map(|&x| x as u64).collect(),
        |w| w.iter().map(|&x| x.min(u32::MAX as u64) as u32).collect(),
    );

    // Asymmetric u8->i16, u16->i32, u32->i64 promote + signed->unsigned i16->u8, i32->u16 demote
    let u8s = fill_inputs(48, U8Vec::<D>::LEN, &u8::boundary(), u8::from_rng);
    let (mut plo, mut phi) = (vec![0i16; I16Vec::<D>::LEN], vec![0i16; I16Vec::<D>::LEN]);
    U8Vec::load(d, &u8s).promote_lo::<i16>().store(&mut plo);
    U8Vec::load(d, &u8s).promote_hi::<i16>().store(&mut phi);
    assert_eq!(
        plo.iter().chain(phi.iter()).copied().collect::<Vec<_>>(),
        u8s[..U8Vec::<D>::LEN]
            .iter()
            .map(|&x| x as i16)
            .collect::<Vec<_>>()
    );
    let i16s = fill_inputs(49, 2 * I16Vec::<D>::LEN, &i16::boundary(), i16::from_rng);
    let mut u8d = vec![0u8; U8Vec::<D>::LEN];
    I16Vec::load(d, &i16s)
        .demote_to(I16Vec::load(d, &i16s[I16Vec::<D>::LEN..]))
        .store(&mut u8d);
    assert_eq!(
        u8d,
        i16s[..U8Vec::<D>::LEN]
            .iter()
            .map(|&x| x.clamp(0, 255) as u8)
            .collect::<Vec<_>>()
    );
    let i32s = fill_inputs(50, 2 * I32Vec::<D>::LEN, &i32::boundary(), i32::from_rng);
    let mut u16d = vec![0u16; U16Vec::<D>::LEN];
    I32Vec::load(d, &i32s)
        .demote_to(I32Vec::load(d, &i32s[I32Vec::<D>::LEN..]))
        .store(&mut u16d);
    assert_eq!(
        u16d,
        i32s[..U16Vec::<D>::LEN]
            .iter()
            .map(|&x| x.clamp(0, 65535) as u16)
            .collect::<Vec<_>>()
    );
    let u16s = fill_inputs(51, U16Vec::<D>::LEN, &u16::boundary(), u16::from_rng);
    let (mut p32l, mut p32h) = (vec![0i32; I32Vec::<D>::LEN], vec![0i32; I32Vec::<D>::LEN]);
    U16Vec::load(d, &u16s).promote_lo::<i32>().store(&mut p32l);
    U16Vec::load(d, &u16s).promote_hi::<i32>().store(&mut p32h);
    assert_eq!(
        p32l.iter().chain(p32h.iter()).copied().collect::<Vec<_>>(),
        u16s[..U16Vec::<D>::LEN]
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>()
    );
    let u32s = fill_inputs(52, U32Vec::<D>::LEN, &u32::boundary(), u32::from_rng);
    let (mut p64l, mut p64h) = (vec![0i64; I64Vec::<D>::LEN], vec![0i64; I64Vec::<D>::LEN]);
    U32Vec::load(d, &u32s).promote_lo::<i64>().store(&mut p64l);
    U32Vec::load(d, &u32s).promote_hi::<i64>().store(&mut p64h);
    assert_eq!(
        p64l.iter().chain(p64h.iter()).copied().collect::<Vec<_>>(),
        u32s[..U32Vec::<D>::LEN]
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
    );
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

    let dlen = F64Vec::<D>::LEN;
    let f64s: Vec<f64> = (0..dlen).map(|i| (i as f64) * -12.75 + 1.25).collect();
    let mut i64o = vec![0i64; dlen];
    F64Vec::load(d, &f64s).as_i64().store(&mut i64o);
    assert_eq!(i64o, f64s.iter().map(|&x| x as i64).collect::<Vec<_>>());
    let mut f64_back = vec![0.0f64; dlen];
    I64Vec::load(d, &i64o).as_f64().store(&mut f64_back);
    assert_eq!(f64_back, i64o.iter().map(|&x| x as f64).collect::<Vec<_>>());

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
    assert_eq!(crate::shl!(I8Vec::splat(d, 1), 7).extract_lane0(), i8::MIN);
    assert_eq!(crate::shr!(I8Vec::splat(d, -128), 7).extract_lane0(), -1);
    assert_eq!(crate::shr!(U8Vec::splat(d, 0x80), 7).extract_lane0(), 1);
    assert_eq!(
        crate::shl!(I16Vec::splat(d, 1), 15).extract_lane0(),
        i16::MIN
    );
    assert_eq!(
        crate::shr!(I64Vec::splat(d, i64::MIN), 63).extract_lane0(),
        -1
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

#[test]
fn test_simd_specialize_dispatch_and_lifetimes() {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn tag_avx512(d: crate::Avx512Descriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        let len = F32Vec::<crate::Avx512Descriptor>::LEN;
        let _ = (d, len);
        dst[0] = src[0] + 512.0;
        "avx512"
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    fn tag_avx(d: crate::AvxDescriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        let _ = d;
        dst[0] = src[0] + 256.0;
        "avx"
    }
    #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
    fn tag_sse42(d: crate::Sse42Descriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        let _ = d;
        dst[0] = src[0] + 128.0;
        "sse42"
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn tag_neon(d: crate::NeonDescriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        let _ = d;
        dst[0] = src[0] + 64.0;
        "neon"
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd128"))]
    fn tag_simd128(d: crate::Simd128Descriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        let _ = d;
        dst[0] = src[0] + 32.0;
        "simd128"
    }
    fn tag_emu128(_d: Emu128Descriptor, src: &[f32], dst: &mut [f32]) -> &'static str {
        dst[0] = src[0] + 1.0;
        "emu128"
    }

    #[simd_specialize(
        Avx512Descriptor => tag_avx512,
        AvxDescriptor => tag_avx,
        Sse42Descriptor => tag_sse42,
        NeonDescriptor => tag_neon,
        Simd128Descriptor => tag_simd128,
        Emu128Descriptor => tag_emu128,
    )]
    fn dispatch_tag<D: SimdDescriptor>(_d: D, src: &[f32], dst: &mut [f32]) -> &'static str {
        dst[0] = src[0];
        "fallback"
    }

    #[simd_specialize(AvxDescriptor => tag_avx)]
    fn partial_avx_only<D: SimdDescriptor>(_d: D, src: &[f32], dst: &mut [f32]) -> &'static str {
        dst[0] = src[0] + 9.0;
        "fallback"
    }

    let mut dst = [0.0f32; 1];
    assert_eq!(
        dispatch_tag(Emu128Descriptor::new().unwrap(), &[2.0], &mut dst),
        "emu128"
    );
    assert_eq!(dst[0], 3.0);
    assert_eq!(
        partial_avx_only(Emu128Descriptor::new().unwrap(), &[2.0], &mut dst),
        "fallback"
    );
    assert_eq!(dst[0], 11.0);

    crate::__if_sse42! {
        if let Some(d) = crate::Sse42Descriptor::new() {
            assert_eq!(dispatch_tag(d, &[2.0], &mut dst), "sse42");
            assert_eq!(dst[0], 130.0);
            assert_eq!(partial_avx_only(d, &[2.0], &mut dst), "fallback");
            assert_eq!(dst[0], 11.0);
        }
    }
    crate::__if_avx! {
        if let Some(d) = crate::AvxDescriptor::new() {
            assert_eq!(dispatch_tag(d, &[2.0], &mut dst), "avx");
            assert_eq!(dst[0], 258.0);
            assert_eq!(partial_avx_only(d, &[2.0], &mut dst), "avx");
            assert_eq!(dst[0], 258.0);
        }
    }
    crate::__if_avx512! {
        if let Some(d) = crate::Avx512Descriptor::new() {
            assert_eq!(dispatch_tag(d, &[2.0], &mut dst), "avx512");
            assert_eq!(dst[0], 514.0);
            assert_eq!(partial_avx_only(d, &[2.0], &mut dst), "fallback");
            assert_eq!(dst[0], 11.0);
        }
    }
    crate::__if_neon! {
        if let Some(d) = crate::NeonDescriptor::new() {
            assert_eq!(dispatch_tag(d, &[2.0], &mut dst), "neon");
            assert_eq!(dst[0], 66.0);
            assert_eq!(partial_avx_only(d, &[2.0], &mut dst), "fallback");
            assert_eq!(dst[0], 11.0);
        }
    }
    crate::__if_simd128! {
        if let Some(d) = crate::Simd128Descriptor::new() {
            assert_eq!(dispatch_tag(d, &[2.0], &mut dst), "simd128");
            assert_eq!(dst[0], 34.0);
            assert_eq!(partial_avx_only(d, &[2.0], &mut dst), "fallback");
            assert_eq!(dst[0], 11.0);
        }
    }
}

#[simd_test]
fn test_simd_specialize_elided_ret_mut_early_return_and_result<D: SimdDescriptor>(d: D) {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    fn slice_prefix_avx(_d: crate::AvxDescriptor, a: &[f32], mut n: usize, _: u32) -> &[f32] {
        if n == 0 {
            return &[];
        }
        n = n.min(a.len());
        &a[..n]
    }
    #[simd_specialize(AvxDescriptor => slice_prefix_avx)]
    fn slice_prefix<D: SimdDescriptor>(_d: D, a: &[f32], mut n: usize, _: u32) -> &[f32] {
        if n == 0 {
            return &[];
        }
        n = n.min(a.len());
        &a[..n]
    }

    let data = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(slice_prefix(d, &data, 2, 0), &[1.0, 2.0]);
    assert!(slice_prefix(d, &data, 0, 1).is_empty());

    #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
    fn checked_first_sse42(_d: crate::Sse42Descriptor, src: &[i32]) -> Result<i32, &'static str> {
        src.first().copied().ok_or("empty")
    }
    #[simd_specialize(Sse42Descriptor => checked_first_sse42)]
    fn checked_first<D: SimdDescriptor>(d: D, src: &[i32]) -> Result<i32, &'static str> {
        let _ = d;
        let first = *src.first().ok_or("empty")?;
        Ok(first + 1 - 1)
    }

    assert_eq!(checked_first(d, &[42, 7]), Ok(42));
    assert_eq!(checked_first(d, &[]), Err("empty"));
}

#[simd_test]
fn test_simd_specialize_vec_value_args_and_return<D: SimdDescriptor>(d: D) {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn scale_bias_avx512<T: FloatElement, const N: usize>(
        d: crate::Avx512Descriptor,
        v: SimdVec<T, crate::Avx512Descriptor>,
        gain: T,
    ) -> SimdVec<T, crate::Avx512Descriptor> {
        v * SimdVec::splat(d, gain)
            + SimdVec::load(
                d,
                &[gain; 16][..SimdVec::<T, crate::Avx512Descriptor>::LEN.max(N)],
            )
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn scale_bias_neon<T: FloatElement, const N: usize>(
        d: crate::NeonDescriptor,
        v: SimdVec<T, crate::NeonDescriptor>,
        gain: T,
    ) -> SimdVec<T, crate::NeonDescriptor> {
        let _ = N;
        v * SimdVec::splat(d, gain) + SimdVec::splat(d, gain)
    }
    fn scale_bias_emu128<T: FloatElement, const N: usize>(
        d: Emu128Descriptor,
        v: SimdVec<T, Emu128Descriptor>,
        gain: T,
    ) -> SimdVec<T, Emu128Descriptor> {
        let _ = N;
        v * SimdVec::splat(d, gain) + SimdVec::splat(d, gain)
    }

    #[simd_specialize(
        Avx512Descriptor => scale_bias_avx512,
        NeonDescriptor => scale_bias_neon,
        Emu128Descriptor => scale_bias_emu128,
    )]
    fn scale_bias<T, D: SimdDescriptor, const N: usize>(
        d: D,
        v: SimdVec<T, D>,
        gain: T,
    ) -> SimdVec<T, D>
    where
        T: FloatElement,
    {
        let _ = N;
        v * SimdVec::splat(d, gain) + SimdVec::splat(d, gain)
    }

    let x = F32Vec::splat(d, 3.0);
    let y = scale_bias::<f32, D, 4>(d, x, 2.0);
    assert_eq!(y.extract_lane0(), 8.0);
    let xd = F64Vec::splat(d, 5.0);
    let yd = scale_bias::<f64, D, 2>(d, xd, 3.0);
    assert_eq!(yd.extract_lane0(), 18.0);
}

#[simd_test]
fn test_simd_specialize_vec_refs_and_mixed_params<D: SimdDescriptor>(d: D) {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    fn pick_vec_avx<'a, T: AsRef<[u8]>, const K: usize>(
        d: crate::AvxDescriptor,
        extra: &T,
        vecs: &'a mut [F32Vec<crate::AvxDescriptor>],
        mut count: usize,
        arr: &[F32Vec<crate::AvxDescriptor>; K],
        _: u32,
    ) -> &'a F32Vec<crate::AvxDescriptor> {
        count += extra.as_ref().len() + arr.len();
        vecs[0] += F32Vec::splat(d, count as f32);
        &vecs[0]
    }
    fn pick_vec_emu128<'a, T: AsRef<[u8]>, const K: usize>(
        d: Emu128Descriptor,
        extra: &T,
        vecs: &'a mut [F32Vec<Emu128Descriptor>],
        mut count: usize,
        arr: &[F32Vec<Emu128Descriptor>; K],
        _: u32,
    ) -> &'a F32Vec<Emu128Descriptor> {
        count += extra.as_ref().len() + arr.len();
        vecs[0] += F32Vec::splat(d, count as f32);
        &vecs[0]
    }

    #[simd_specialize(AvxDescriptor => pick_vec_avx, Emu128Descriptor => pick_vec_emu128)]
    fn pick_vec<'a, D: SimdDescriptor, T: AsRef<[u8]>, const K: usize>(
        d: D,
        extra: &T,
        vecs: &'a mut [F32Vec<D>],
        mut count: usize,
        arr: &[F32Vec<D>; K],
        _: u32,
    ) -> &'a F32Vec<D> {
        count += extra.as_ref().len() + arr.len();
        vecs[0] += F32Vec::splat(d, count as f32);
        &vecs[0]
    }

    let local_ref: &[u8] = &[1u8, 2, 3];
    let mut vecs = [F32Vec::splat(d, 10.0)];
    let arr = [F32Vec::splat(d, 0.0); 4];
    let mut out = vec![0.0f32; F32Vec::<D>::LEN];
    pick_vec::<D, &[u8], 4>(d, &local_ref, &mut vecs, 5, &arr, 99).store(&mut out);
    assert_eq!(out[0], 22.0);
}

fn check_bitmask_for_width<W, T, D>(d: D, zero: T, one: T)
where
    W: MaskWidth,
    T: SimdElement + ComparableElement<Mask = W>,
    D: SimdDescriptor,
{
    let len = SimdVec::<T, D>::LEN;
    assert_eq!(SimdMask::<W, D>::LEN, len);
    let full_mask = if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    };

    let v_zero = SimdVec::splat(d, zero);
    let v_one = SimdVec::splat(d, one);
    assert_eq!(v_zero.eq(v_one).to_bitmask(), 0);
    assert_eq!(v_one.eq(v_one).to_bitmask(), full_mask);

    for lane in 0..len {
        let mut buf = vec![zero; len];
        buf[lane] = one;
        let m = SimdVec::load(d, &buf).eq(v_one);
        let single = 1u64 << lane;
        assert_eq!(m.to_bitmask(), single, "single lane={lane} len={len}");
        assert_eq!(
            (!m).to_bitmask(),
            full_mask ^ single,
            "inverted lane={lane} len={len}"
        );
    }

    for pat in [
        0x5555_5555_5555_5555u64,
        0xAAAA_AAAA_AAAA_AAAAu64,
        0x8000_0001_8000_0001u64,
        0xF00F_0FF0_C33C_3CC3u64,
        0x8000_0000_0000_0000u64,
        0x0000_0000_8000_0000u64,
    ] {
        let expected = pat & full_mask;
        let buf: Vec<T> = (0..len)
            .map(|i| {
                if ((expected >> i) & 1) != 0 {
                    one
                } else {
                    zero
                }
            })
            .collect();
        let m = SimdVec::load(d, &buf).eq(v_one);
        assert_eq!(m.to_bitmask(), expected, "pattern={pat:#x} len={len}");
    }

    #[cfg(not(miri))]
    arbtest::arbtest(|u| {
        let raw: u64 = u.arbitrary()?;
        let expected = raw & full_mask;
        let buf: Vec<T> = (0..len)
            .map(|i| {
                if ((expected >> i) & 1) != 0 {
                    one
                } else {
                    zero
                }
            })
            .collect();
        let m = SimdVec::load(d, &buf).eq(v_one);
        assert_eq!(m.to_bitmask(), expected);
        Ok(())
    });
}

#[simd_test]
fn test_mask_to_bitmask<D: SimdDescriptor>(d: D) {
    check_bitmask_for_width::<W8, u8, D>(d, 0, 1);
    check_bitmask_for_width::<W16, u16, D>(d, 0, 1);
    check_bitmask_for_width::<W32, u32, D>(d, 0, 1);
    check_bitmask_for_width::<W64, u64, D>(d, 0, 1);

    // Also test float comparison masks (`__m128`/`__m256`/`__mmask*` on x86).
    check_bitmask_for_width::<W32, f32, D>(d, 0.0, 1.0);
    check_bitmask_for_width::<W64, f64, D>(d, 0.0, 1.0);
}

fn check_slide_n<
    T: SimdElement + PartialEq + std::fmt::Debug,
    D: SimdDescriptor,
    const N: usize,
>(
    d: D,
    prev: &[T],
    curr: &[T],
    next: &[T],
) {
    let len = SimdVec::<T, D>::LEN;
    let vp = SimdVec::load(d, prev);
    let vc = SimdVec::load(d, curr);
    let vn = SimdVec::load(d, next);

    let mut out_r = vec![T::default(); len];
    vc.slide_right::<N>(vp).store(&mut out_r);
    for i in 0..len {
        let exp = if i < N {
            prev[len - N + i]
        } else {
            curr[i - N]
        };
        assert_eq!(out_r[i], exp, "slide_right N={N} lane={i} len={len}");
    }

    let mut out_l = vec![T::default(); len];
    vc.slide_left::<N>(vn).store(&mut out_l);
    for i in 0..len {
        let exp = if i + N < len {
            curr[i + N]
        } else {
            next[i + N - len]
        };
        assert_eq!(out_l[i], exp, "slide_left N={N} lane={i} len={len}");
    }
}

macro_rules! check_slides_list {
    ($d:expr, $t:ty, $d_ty:ty, $prev:expr, $curr:expr, $next:expr, [$($n:literal),* $(,)?]) => {{
        $(check_slide_n::<$t, $d_ty, $n>($d, $prev, $curr, $next);)*
    }};
}

fn check_slides_up_to_128b<D: SimdDescriptor>(d: D) {
    let p8: Vec<u8> = (0..64).map(|i| (i + 1) as u8).collect();
    let c8: Vec<u8> = (0..64).map(|i| (i + 101) as u8).collect();
    let n8: Vec<u8> = (0..64).map(|i| (i + 201) as u8).collect();
    check_slides_list!(
        d,
        u8,
        D,
        &p8,
        &c8,
        &n8,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );

    let p16: Vec<u16> = (0..32).map(|i| (0x100 + i) as u16).collect();
    let c16: Vec<u16> = (0..32).map(|i| (0x200 + i) as u16).collect();
    let n16: Vec<u16> = (0..32).map(|i| (0x300 + i) as u16).collect();
    check_slides_list!(d, u16, D, &p16, &c16, &n16, [0, 1, 2, 3, 4, 5, 6, 7, 8]);

    let p32: Vec<u32> = (0..16).map(|i| 0x1000_0000 + i as u32).collect();
    let c32: Vec<u32> = (0..16).map(|i| 0x2000_0000 + i as u32).collect();
    let n32: Vec<u32> = (0..16).map(|i| 0x3000_0000 + i as u32).collect();
    check_slides_list!(d, u32, D, &p32, &c32, &n32, [0, 1, 2, 3, 4]);

    let pi32: Vec<i32> = (0..16).map(|i| -1000 + i).collect();
    let ci32: Vec<i32> = (0..16).map(|i| 2000 + i).collect();
    let ni32: Vec<i32> = (0..16).map(|i| -3000 + i).collect();
    check_slides_list!(d, i32, D, &pi32, &ci32, &ni32, [0, 1, 2, 3, 4]);

    let p64: Vec<u64> = (0..8).map(|i| 0x0102_0304_0506_0700 + i).collect();
    let c64: Vec<u64> = (0..8).map(|i| 0x1112_1314_1516_1700 + i).collect();
    let n64: Vec<u64> = (0..8).map(|i| 0x2122_2324_2526_2700 + i).collect();
    check_slides_list!(d, u64, D, &p64, &c64, &n64, [0, 1, 2]);
}

#[simd_test]
fn test_slide_right_and_left<D: SimdDescriptor>(d: D) {
    check_slides_up_to_128b(d);

    #[cfg(not(miri))]
    arbtest::arbtest(|u| {
        let len = U32Vec::<D>::LEN;
        let mut prev = vec![0u32; len];
        let mut curr = vec![0u32; len];
        let mut next = vec![0u32; len];
        for i in 0..len {
            prev[i] = u.arbitrary()?;
            curr[i] = u.arbitrary()?;
            next[i] = u.arbitrary()?;
        }
        check_slides_list!(d, u32, D, &prev, &curr, &next, [0, 1, 2, 3, 4]);
        Ok(())
    });
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[test]
fn test_slide_wide_avx_all_offsets() {
    let Some(d) = crate::AvxDescriptor::new() else {
        return;
    };
    d.call(|d| {
        let p8: Vec<u8> = (0..32).map(|i| (i + 1) as u8).collect();
        let c8: Vec<u8> = (0..32).map(|i| (i + 65) as u8).collect();
        let n8: Vec<u8> = (0..32).map(|i| (i + 129) as u8).collect();
        check_slides_list!(
            d,
            u8,
            crate::AvxDescriptor,
            &p8,
            &c8,
            &n8,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        );
        let p16: Vec<u16> = (0..16).map(|i| (0x100 + i) as u16).collect();
        let c16: Vec<u16> = (0..16).map(|i| (0x200 + i) as u16).collect();
        let n16: Vec<u16> = (0..16).map(|i| (0x300 + i) as u16).collect();
        check_slides_list!(
            d,
            u16,
            crate::AvxDescriptor,
            &p16,
            &c16,
            &n16,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        let p32: Vec<u32> = (0..8).map(|i| 100 + i as u32).collect();
        let c32: Vec<u32> = (0..8).map(|i| 200 + i as u32).collect();
        let n32: Vec<u32> = (0..8).map(|i| 300 + i as u32).collect();
        check_slides_list!(
            d,
            u32,
            crate::AvxDescriptor,
            &p32,
            &c32,
            &n32,
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        );
        let p64: Vec<u64> = (0..4).map(|i| 1000 + i).collect();
        let c64: Vec<u64> = (0..4).map(|i| 2000 + i).collect();
        let n64: Vec<u64> = (0..4).map(|i| 3000 + i).collect();
        check_slides_list!(
            d,
            u64,
            crate::AvxDescriptor,
            &p64,
            &c64,
            &n64,
            [0, 1, 2, 3, 4]
        );
    });
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn test_slide_wide_avx512_all_offsets() {
    let Some(d) = crate::Avx512Descriptor::new() else {
        return;
    };
    d.call(|d| {
        let p8: Vec<u8> = (0..64).map(|i| (i + 1) as u8).collect();
        let c8: Vec<u8> = (0..64).map(|i| (i + 65) as u8).collect();
        let n8: Vec<u8> = (0..64).map(|i| (i + 129) as u8).collect();
        check_slides_list!(
            d,
            u8,
            crate::Avx512Descriptor,
            &p8,
            &c8,
            &n8,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
            ]
        );
        let p16: Vec<u16> = (0..32).map(|i| (0x100 + i) as u16).collect();
        let c16: Vec<u16> = (0..32).map(|i| (0x200 + i) as u16).collect();
        let n16: Vec<u16> = (0..32).map(|i| (0x300 + i) as u16).collect();
        check_slides_list!(
            d,
            u16,
            crate::Avx512Descriptor,
            &p16,
            &c16,
            &n16,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        );
        let p32: Vec<u32> = (0..16).map(|i| 100 + i as u32).collect();
        let c32: Vec<u32> = (0..16).map(|i| 200 + i as u32).collect();
        let n32: Vec<u32> = (0..16).map(|i| 300 + i as u32).collect();
        check_slides_list!(
            d,
            u32,
            crate::Avx512Descriptor,
            &p32,
            &c32,
            &n32,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        let p64: Vec<u64> = (0..8).map(|i| 1000 + i).collect();
        let c64: Vec<u64> = (0..8).map(|i| 2000 + i).collect();
        let n64: Vec<u64> = (0..8).map(|i| 3000 + i).collect();
        check_slides_list!(
            d,
            u64,
            crate::Avx512Descriptor,
            &p64,
            &c64,
            &n64,
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    });
}

#[simd_test]
fn test_madd_to_i32<D: SimdDescriptor>(d: D) {
    let n16 = I16Vec::<D>::LEN;
    let n32 = I32Vec::<D>::LEN;

    // Explicit check of `(-32768, -32768)` wrapping to `i32::MIN` and boundary pairs.
    let mut corner_a = vec![0i16; n16];
    let mut corner_b = vec![0i16; n16];
    let pairs: [(i16, i16, i16, i16); 4] = [
        (i16::MIN, i16::MIN, i16::MIN, i16::MIN),
        (i16::MAX, i16::MAX, i16::MAX, i16::MAX),
        (i16::MIN, i16::MAX, i16::MIN, 1),
        (-1, 1, 12345, -23456),
    ];
    for k in 0..n32 {
        let (a0, b0, a1, b1) = pairs[k % pairs.len()];
        corner_a[2 * k] = a0;
        corner_b[2 * k] = b0;
        corner_a[2 * k + 1] = a1;
        corner_b[2 * k + 1] = b1;
    }
    let mut out = vec![0i32; n32];
    I16Vec::load(d, &corner_a)
        .madd_to_i32(I16Vec::load(d, &corner_b))
        .store(&mut out);
    assert_eq!(out[0], i32::MIN);
    for k in 0..n32 {
        let exp = (corner_a[2 * k] as i32 * corner_b[2 * k] as i32)
            .wrapping_add(corner_a[2 * k + 1] as i32 * corner_b[2 * k + 1] as i32);
        assert_eq!(out[k], exp, "corner pair k={k}");
    }

    let bnd = i16::boundary();
    let a = fill_inputs(701, 4 * n16, &bnd, i16::from_rng);
    let b = fill_inputs(702, 4 * n16, &bnd, i16::from_rng);
    for chunk in (0..a.len()).step_by(n16) {
        I16Vec::load(d, &a[chunk..])
            .madd_to_i32(I16Vec::load(d, &b[chunk..]))
            .store(&mut out);
        for k in 0..n32 {
            let exp = (a[chunk + 2 * k] as i32 * b[chunk + 2 * k] as i32)
                .wrapping_add(a[chunk + 2 * k + 1] as i32 * b[chunk + 2 * k + 1] as i32);
            assert_eq!(out[k], exp);
        }
    }

    #[cfg(not(miri))]
    arbtest::arbtest(|u| {
        let mut va = vec![0i16; n16];
        let mut vb = vec![0i16; n16];
        for i in 0..n16 {
            va[i] = u.arbitrary()?;
            vb[i] = u.arbitrary()?;
        }
        I16Vec::load(d, &va)
            .madd_to_i32(I16Vec::load(d, &vb))
            .store(&mut out);
        for k in 0..n32 {
            let exp = (va[2 * k] as i32 * vb[2 * k] as i32)
                .wrapping_add(va[2 * k + 1] as i32 * vb[2 * k + 1] as i32);
            assert_eq!(out[k], exp);
        }
        Ok(())
    });
}

macro_rules! check_int_reductions {
    ($d:expr, $t:ty, $d_ty:ty, $seed:expr) => {{
        let len = SimdVec::<$t, $d_ty>::LEN;
        assert_eq!(SimdVec::<$t, $d_ty>::zero($d).reduce_add(), 0);
        assert_eq!(SimdVec::<$t, $d_ty>::zero($d).reduce_or(), 0);
        assert_eq!(SimdVec::<$t, $d_ty>::zero($d).reduce_and(), 0);
        assert_eq!(SimdVec::<$t, $d_ty>::splat($d, !0).reduce_or(), !0);
        assert_eq!(SimdVec::<$t, $d_ty>::splat($d, !0).reduce_and(), !0);

        for lane in 0..len {
            for bit in 0..<$t>::BITS {
                let val = (1 as $t).wrapping_shl(bit);
                let mut only_one = vec![0 as $t; len];
                only_one[lane] = val;
                let v = SimdVec::<$t, $d_ty>::load($d, &only_one);
                assert_eq!(v.reduce_add(), val, "add lane={lane} bit={bit}");
                assert_eq!(v.reduce_or(), val, "or lane={lane} bit={bit}");
                assert_eq!(
                    v.reduce_and(),
                    if len == 1 { val } else { 0 },
                    "and lane={lane} bit={bit}"
                );

                let mut all_but_one = vec![!0 as $t; len];
                all_but_one[lane] = !val;
                let vn = SimdVec::<$t, $d_ty>::load($d, &all_but_one);
                assert_eq!(vn.reduce_and(), !val, "and-clear lane={lane} bit={bit}");
            }
        }

        let bnd = <$t>::boundary();
        let inputs = fill_inputs($seed, 4 * len, &bnd, <$t>::from_rng);
        for chunk in inputs.chunks_exact(len) {
            let v = SimdVec::<$t, $d_ty>::load($d, chunk);
            let exp_add = chunk.iter().copied().fold(0 as $t, <$t>::wrapping_add);
            let exp_or = chunk.iter().copied().fold(0 as $t, |a, b| a | b);
            let exp_and = chunk.iter().copied().fold(!0 as $t, |a, b| a & b);
            assert_eq!(v.reduce_add(), exp_add);
            assert_eq!(v.reduce_or(), exp_or);
            assert_eq!(v.reduce_and(), exp_and);
        }

        #[cfg(not(miri))]
        arbtest::arbtest(|u| {
            let mut buf = vec![0 as $t; len];
            for slot in &mut buf {
                *slot = u.arbitrary()?;
            }
            let v = SimdVec::<$t, $d_ty>::load($d, &buf);
            assert_eq!(
                v.reduce_add(),
                buf.iter().copied().fold(0 as $t, <$t>::wrapping_add)
            );
            assert_eq!(
                v.reduce_or(),
                buf.iter().copied().fold(0 as $t, |a, b| a | b)
            );
            assert_eq!(
                v.reduce_and(),
                buf.iter().copied().fold(!0 as $t, |a, b| a & b)
            );
            Ok(())
        });
    }};
}

fn generic_sum<T: ArithElement, D: SimdDescriptor>(v: SimdVec<T, D>) -> T {
    v.reduce_add()
}

#[simd_test]
fn test_reductions_and_prefetch<D: SimdDescriptor>(d: D) {
    check_int_reductions!(d, i8, D, 801);
    check_int_reductions!(d, u8, D, 802);
    check_int_reductions!(d, i16, D, 803);
    check_int_reductions!(d, u16, D, 804);
    check_int_reductions!(d, i32, D, 805);
    check_int_reductions!(d, u32, D, 806);
    check_int_reductions!(d, i64, D, 807);
    check_int_reductions!(d, u64, D, 808);

    // Float `reduce_add` (including through generic `T: ArithElement`).
    let f32s: Vec<f32> = (0..F32Vec::<D>::LEN)
        .map(|i| (i as f32) * 1.5 - 3.0)
        .collect();
    let exp32: f32 = f32s.iter().sum();
    let got32 = generic_sum(F32Vec::load(d, &f32s));
    assert!((got32 - exp32).abs() <= 1e-5, "got32={got32} exp32={exp32}");

    let f64s: Vec<f64> = (0..F64Vec::<D>::LEN)
        .map(|i| (i as f64) * 2.25 - 1.5)
        .collect();
    let exp64: f64 = f64s.iter().sum();
    let got64 = generic_sum(F64Vec::load(d, &f64s));
    assert!(
        (got64 - exp64).abs() <= 1e-12,
        "got64={got64} exp64={exp64}"
    );

    // Prefetch helpers: ZST, single-cacheline, multi-cacheline, unsized slices, and inside `d.call`.
    let zst = ();
    let empty_arr = [0u8; 0];
    let scalar = 42u64;
    let line = [7u8; 64];
    #[repr(C)]
    struct Big([u8; 300]);
    let big = Big([0xAB; 300]);

    d.prefetch_t0(&zst);
    d.prefetch_et0(&zst);
    d.prefetch_t0(&empty_arr);
    d.prefetch_et0(&empty_arr);
    d.prefetch_t0(&scalar);
    d.prefetch_et0(&scalar);
    d.prefetch_t0(&line);
    d.prefetch_et0(&line);
    d.prefetch_t0(&big);
    d.prefetch_et0(&big);
    d.prefetch_t0(&big.0[..]);
    d.prefetch_et0(&big.0[..0]);

    let scalar_ops = d.call(|d| {
        d.prefetch_t0(&big);
        d.prefetch_et0(&line);
        let x = std::hint::black_box(0x0010_0800u64);
        (
            x.trailing_zeros(),
            x.leading_zeros(),
            x >> std::hint::black_box(5),
        )
    });
    assert_eq!(scalar_ops, (11, 43, 0x0010_0800u64 >> 5));
}
