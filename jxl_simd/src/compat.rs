// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Temporary compatibility shim exposing legacy `D::*Vec`, `*SimdVec` traits,
//! and 1-lane `ScalarDescriptor` while downstream crates migrate incrementally to `SimdVec<T, D>`.

#![allow(dead_code, private_interfaces)]

use std::fmt::Debug;
use std::num::Wrapping;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::sealed::{
    ArithBackend, BytesBackend, CompareBackend, DescriptorSealed, F32ExtraBackend, FloatBackend,
    I16ExtraBackend, I32ExtraBackend, IntBitBackend, LaneBackend, MaskBackend, MulBackend,
    SignedBackend, U8ExtraBackend,
};
use crate::{
    Bf16Table8, Emu128Descriptor, F32Vec, I16Vec, I32Vec, Mask, Mask16, SimdDescriptor,
    SimdDescriptor128, SimdElement, SimdVec, SimdVecArray, U8Vec, U16Vec, U32Vec, U64Vec, W8, W16,
    W32, W64, f16,
};

pub trait SimdDescriptorCompat: Sized {
    type F32Vec: F32SimdVec<Descriptor = Self>;
    type I32Vec: I32SimdVec<Descriptor = Self>;
    type I16Vec: I16SimdVec<Descriptor = Self>;
    type U64Vec: U64SimdVec<Descriptor = Self>;
    type U32Vec: U32SimdVec<Descriptor = Self>;
    type U16Vec: U16SimdVec<Descriptor = Self>;
    type U8Vec: U8SimdVec<Descriptor = Self>;
    type Mask: SimdMask<Descriptor = Self>;
    type Mask16: SimdMask16<Descriptor = Self>;
    type Bf16Table8: Copy + Debug;
}

macro_rules! impl_simd_vec_compat_descr {
    ($d:ty) => {
        impl SimdDescriptorCompat for $d {
            type F32Vec = F32Vec<Self>;
            type I32Vec = I32Vec<Self>;
            type I16Vec = I16Vec<Self>;
            type U64Vec = U64Vec<Self>;
            type U32Vec = U32Vec<Self>;
            type U16Vec = U16Vec<Self>;
            type U8Vec = U8Vec<Self>;
            type Mask = Mask<Self>;
            type Mask16 = Mask16<Self>;
            type Bf16Table8 = Bf16Table8<Self>;
        }
    };
}

impl_simd_vec_compat_descr!(Emu128Descriptor);
#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
impl_simd_vec_compat_descr!(crate::Sse42Descriptor);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl_simd_vec_compat_descr!(crate::AvxDescriptor);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl_simd_vec_compat_descr!(crate::Avx512Descriptor);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl_simd_vec_compat_descr!(crate::NeonDescriptor);
#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
impl_simd_vec_compat_descr!(crate::Simd128Descriptor);

pub trait F32SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
    + DivAssign<Self>
{
    type Descriptor: SimdDescriptor;
    const LEN: usize;
    type UnderlyingArray: Copy + Debug;

    fn splat(d: Self::Descriptor, v: f32) -> Self;
    fn zero(d: Self::Descriptor) -> Self;
    fn mul_add(self, mul: Self, add: Self) -> Self;
    fn neg_mul_add(self, mul: Self, add: Self) -> Self;
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self;
    fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self;
    fn store(&self, mem: &mut [f32]);
    fn store_array(&self, mem: &mut Self::UnderlyingArray);
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]);
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]);
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]);
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    );
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self);
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self);
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self);
    fn round_store_u8(self, dest: &mut [u8]);
    fn round_store_u16(self, dest: &mut [u16]);
    fn abs(self) -> Self;
    fn floor(self) -> Self;
    fn sqrt(self) -> Self;
    fn neg(self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn gt(self, other: Self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::Mask;
    fn as_i32(self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::I32Vec;
    fn bitcast_to_i32(self) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::I32Vec;
    fn prepare_table_bf16_8(
        d: Self::Descriptor,
        table: &[f32; 8],
    ) -> <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::Bf16Table8;
    fn table_lookup_bf16_8(
        d: Self::Descriptor,
        table: <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::Bf16Table8,
        indices: <<Self as F32SimdVec>::Descriptor as SimdDescriptorCompat>::I32Vec,
    ) -> Self;
    fn make_array_slice(slice: &[f32]) -> &[Self::UnderlyingArray];
    fn make_array_slice_mut(slice: &mut [f32]) -> &mut [Self::UnderlyingArray];
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize);
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self;
    fn store_f16_bits(self, dest: &mut [u16]);
}

#[inline(always)]
fn transpose_regs<T: SimdElement, D: SimdDescriptor, const LEN: usize, const K: usize>(
    mut regs: [SimdVec<T, D>; LEN],
) -> [SimdVec<T, D>; LEN] {
    for _ in 0..K.trailing_zeros() {
        let mut next = regs;
        for g in (0..LEN).step_by(K) {
            for i in 0..(K / 2) {
                let a = regs[g + i];
                let b = regs[g + K / 2 + i];
                next[g + 2 * i] = SimdVec::interleave_lo(a, b);
                next[g + 2 * i + 1] = SimdVec::interleave_hi(a, b);
            }
        }
        regs = next;
    }
    if D::BYTES == 32 {
        for i in 0..K {
            let (a, b) = regs[i].interleave_blocks_128(regs[K + i]);
            regs[i] = a;
            regs[K + i] = b;
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

#[inline(always)]
fn compat_transpose_square<T: SimdElement, D: SimdDescriptor>(
    d: D,
    data: &mut [SimdVecArray<T, D>],
    stride: usize,
) {
    let len = SimdVec::<T, D>::LEN;
    assert!(data.len() > (len - 1) * stride);
    macro_rules! run {
        ($len:literal, $k:literal) => {{
            let mut regs = [SimdVec::zero(d); $len];
            for i in 0..$len {
                regs[i] = SimdVec::load_array(d, &data[i * stride]);
            }
            let out = transpose_regs::<T, D, $len, $k>(regs);
            for i in 0..$len {
                out[i].store_array(&mut data[i * stride]);
            }
        }};
    }
    match (len, 16 / std::mem::size_of::<T>()) {
        (4, 4) => run!(4, 4),
        (8, 4) => run!(8, 4),
        (16, 4) => run!(16, 4),
        (8, 8) => run!(8, 8),
        (16, 8) => run!(16, 8),
        (32, 8) => run!(32, 8),
        _ => unreachable!(),
    }
}

impl<
    D: SimdDescriptor
        + SimdDescriptorCompat<I32Vec = I32Vec<D>, Mask = Mask<D>, Bf16Table8 = Bf16Table8<D>>,
> F32SimdVec for F32Vec<D>
{
    type Descriptor = D;
    const LEN: usize = Self::LEN;
    type UnderlyingArray = SimdVecArray<f32, D>;

    #[inline(always)]
    fn splat(d: D, v: f32) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn zero(d: D) -> Self {
        Self::zero(d)
    }
    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        self.mul_add(mul, add)
    }
    #[inline(always)]
    fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        self.neg_mul_add(mul, add)
    }
    #[inline(always)]
    fn load(d: D, mem: &[f32]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn load_array(d: D, mem: &Self::UnderlyingArray) -> Self {
        Self::load_array(d, mem)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn store_array(&self, mem: &mut Self::UnderlyingArray) {
        (*self).store_array(mem)
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        Self::store_interleaved_2(a, b, dest)
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        Self::store_interleaved_3(a, b, c, dest)
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        Self::store_interleaved_4(a, b, c, d, dest)
    }
    #[inline(always)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        Self::store_interleaved_8(a, b, c, d, e, f, g, h, dest)
    }
    #[inline(always)]
    fn load_deinterleaved_2(d: D, src: &[f32]) -> (Self, Self) {
        let len = Self::LEN;
        assert!(src.len() >= 2 * len);
        let (mut a, mut b) = (
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
        );
        for i in 0..len {
            a.as_mut()[i] = src[2 * i];
            b.as_mut()[i] = src[2 * i + 1];
        }
        (Self::load_array(d, &a), Self::load_array(d, &b))
    }
    #[inline(always)]
    fn load_deinterleaved_3(d: D, src: &[f32]) -> (Self, Self, Self) {
        let len = Self::LEN;
        assert!(src.len() >= 3 * len);
        let (mut a, mut b, mut c) = (
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
        );
        for i in 0..len {
            a.as_mut()[i] = src[3 * i];
            b.as_mut()[i] = src[3 * i + 1];
            c.as_mut()[i] = src[3 * i + 2];
        }
        (
            Self::load_array(d, &a),
            Self::load_array(d, &b),
            Self::load_array(d, &c),
        )
    }
    #[inline(always)]
    fn load_deinterleaved_4(d: D, src: &[f32]) -> (Self, Self, Self, Self) {
        let len = Self::LEN;
        assert!(src.len() >= 4 * len);
        let (mut a, mut b, mut c, mut e) = (
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
            <f32 as crate::sealed::Element>::zero_array::<D>(),
        );
        for i in 0..len {
            a.as_mut()[i] = src[4 * i];
            b.as_mut()[i] = src[4 * i + 1];
            c.as_mut()[i] = src[4 * i + 2];
            e.as_mut()[i] = src[4 * i + 3];
        }
        (
            Self::load_array(d, &a),
            Self::load_array(d, &b),
            Self::load_array(d, &c),
            Self::load_array(d, &e),
        )
    }
    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        self.round_store_u8(dest)
    }
    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        self.round_store_u16(dest)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> Mask<D> {
        self.gt(other)
    }
    #[inline(always)]
    fn as_i32(self) -> I32Vec<D> {
        self.as_i32()
    }
    #[inline(always)]
    fn bitcast_to_i32(self) -> I32Vec<D> {
        self.bitcast::<i32>()
    }
    #[inline(always)]
    fn prepare_table_bf16_8(d: D, table: &[f32; 8]) -> Bf16Table8<D> {
        Self::prepare_table_bf16_8(d, table)
    }
    #[inline(always)]
    fn table_lookup_bf16_8(d: D, table: Bf16Table8<D>, indices: I32Vec<D>) -> Self {
        Self::table_lookup_bf16_8(d, table, indices)
    }
    #[inline(always)]
    fn make_array_slice(slice: &[f32]) -> &[Self::UnderlyingArray] {
        Self::make_array_slice(slice)
    }
    #[inline(always)]
    fn make_array_slice_mut(slice: &mut [f32]) -> &mut [Self::UnderlyingArray] {
        Self::make_array_slice_mut(slice)
    }
    #[inline(always)]
    fn transpose_square(d: D, data: &mut [Self::UnderlyingArray], stride: usize) {
        compat_transpose_square::<f32, D>(d, data, stride);
    }
    #[inline(always)]
    fn load_f16_bits(d: D, mem: &[u16]) -> Self {
        Self::load_f16_bits(d, mem)
    }
    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        self.store_f16_bits(dest)
    }
}

pub trait I32SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;
    const LEN: usize;

    fn splat(d: Self::Descriptor, v: i32) -> Self;
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self;
    fn load_from_i16(d: Self::Descriptor, mem: &[i16]) -> Self;
    fn load_from_u16(d: Self::Descriptor, mem: &[u16]) -> Self;
    fn store(&self, mem: &mut [i32]);
    fn abs(self) -> Self;
    fn as_f32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::F32Vec;
    fn bitcast_to_f32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::F32Vec;
    fn bitcast_to_u32(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::U32Vec;
    fn gt(self, other: Self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::Mask;
    fn lt_zero(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::Mask;
    fn eq(self, other: Self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::Mask;
    fn eq_zero(self) -> <<Self as I32SimdVec>::Descriptor as SimdDescriptorCompat>::Mask;
    fn shl<const AMOUNT: u32>(self) -> Self;
    fn shr<const AMOUNT: u32>(self) -> Self;
    fn mul_wide_take_high(self, rhs: Self) -> Self;
    fn store_u16(self, dest: &mut [u16]);
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i32]);
    fn store_u8(self, dest: &mut [u8]);
}

impl<
    D: SimdDescriptor + SimdDescriptorCompat<F32Vec = F32Vec<D>, U32Vec = U32Vec<D>, Mask = Mask<D>>,
> I32SimdVec for I32Vec<D>
{
    type Descriptor = D;
    const LEN: usize = Self::LEN;

    #[inline(always)]
    fn splat(d: D, v: i32) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn load(d: D, mem: &[i32]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn load_from_i16(d: D, mem: &[i16]) -> Self {
        Self::load_from_i16(d, mem)
    }
    #[inline(always)]
    fn load_from_u16(d: D, mem: &[u16]) -> Self {
        Self::load_from_u16(d, mem)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn as_f32(self) -> F32Vec<D> {
        self.as_f32()
    }
    #[inline(always)]
    fn bitcast_to_f32(self) -> F32Vec<D> {
        self.bitcast::<f32>()
    }
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32Vec<D> {
        self.bitcast::<u32>()
    }
    #[inline(always)]
    fn gt(self, other: Self) -> Mask<D> {
        self.gt(other)
    }
    #[inline(always)]
    fn lt_zero(self) -> Mask<D> {
        self.lt_zero()
    }
    #[inline(always)]
    fn eq(self, other: Self) -> Mask<D> {
        self.eq(other)
    }
    #[inline(always)]
    fn eq_zero(self) -> Mask<D> {
        self.eq_zero()
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        self.shl::<AMOUNT>()
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        self.shr::<AMOUNT>()
    }
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        self.mul_wide_take_high(rhs)
    }
    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        self.store_u16(dest)
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i32]) {
        Self::store_interleaved_2(a, b, dest)
    }
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        self.store_u8(dest)
    }
}

pub trait I16SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Neg<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;
    const LEN: usize;
    type UnderlyingArray: Copy + Debug;

    fn splat(d: Self::Descriptor, v: i16) -> Self;
    fn zero(d: Self::Descriptor) -> Self;
    fn load(d: Self::Descriptor, mem: &[i16]) -> Self;
    fn load_array(d: Self::Descriptor, mem: &Self::UnderlyingArray) -> Self;
    fn store(&self, mem: &mut [i16]);
    fn store_array(&self, mem: &mut Self::UnderlyingArray);
    fn make_array_slice(slice: &[i16]) -> &[Self::UnderlyingArray];
    fn make_array_slice_mut(slice: &mut [i16]) -> &mut [Self::UnderlyingArray];
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize);
    fn abs(self) -> Self;
    fn shl<const AMOUNT: u32>(self) -> Self;
    fn shr<const AMOUNT: u32>(self) -> Self;
    fn mul_wide_take_high(self, rhs: Self) -> Self;
    fn gt(self, other: Self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptorCompat>::Mask16;
    fn lt_zero(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptorCompat>::Mask16;
    fn eq(self, other: Self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptorCompat>::Mask16;
    fn eq_zero(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptorCompat>::Mask16;
    fn bitcast_u16(self) -> <<Self as I16SimdVec>::Descriptor as SimdDescriptorCompat>::U16Vec;
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i16]);
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [i16]);
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [i16]);
    fn store_u8(self, dest: &mut [u8]);
}

impl<D: SimdDescriptor + SimdDescriptorCompat<U16Vec = U16Vec<D>, Mask16 = Mask16<D>>> I16SimdVec
    for I16Vec<D>
{
    type Descriptor = D;
    const LEN: usize = Self::LEN;
    type UnderlyingArray = SimdVecArray<i16, D>;

    #[inline(always)]
    fn splat(d: D, v: i16) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn zero(d: D) -> Self {
        Self::zero(d)
    }
    #[inline(always)]
    fn load(d: D, mem: &[i16]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn load_array(d: D, mem: &Self::UnderlyingArray) -> Self {
        Self::load_array(d, mem)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [i16]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn store_array(&self, mem: &mut Self::UnderlyingArray) {
        (*self).store_array(mem)
    }
    #[inline(always)]
    fn make_array_slice(slice: &[i16]) -> &[Self::UnderlyingArray] {
        Self::make_array_slice(slice)
    }
    #[inline(always)]
    fn make_array_slice_mut(slice: &mut [i16]) -> &mut [Self::UnderlyingArray] {
        Self::make_array_slice_mut(slice)
    }
    #[inline(always)]
    fn transpose_square(d: D, data: &mut [Self::UnderlyingArray], stride: usize) {
        compat_transpose_square::<i16, D>(d, data, stride);
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        self.shl::<AMOUNT>()
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        self.shr::<AMOUNT>()
    }
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        self.mul_wide_take_high(rhs)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> Mask16<D> {
        self.gt(other)
    }
    #[inline(always)]
    fn lt_zero(self) -> Mask16<D> {
        self.lt_zero()
    }
    #[inline(always)]
    fn eq(self, other: Self) -> Mask16<D> {
        self.eq(other)
    }
    #[inline(always)]
    fn eq_zero(self) -> Mask16<D> {
        self.eq_zero()
    }
    #[inline(always)]
    fn bitcast_u16(self) -> U16Vec<D> {
        self.bitcast::<u16>()
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i16]) {
        Self::store_interleaved_2(a, b, dest)
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [i16]) {
        Self::store_interleaved_3(a, b, c, dest)
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [i16]) {
        Self::store_interleaved_4(a, b, c, d, dest)
    }
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        self.store_u8(dest)
    }
}

pub trait U64SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + AddAssign<Self>
    + BitAndAssign<Self>
    + BitOrAssign<Self>
    + BitXorAssign<Self>
{
    type Descriptor: SimdDescriptor;
    const LEN: usize;

    fn splat(d: Self::Descriptor, v: u64) -> Self;
    fn load(d: Self::Descriptor, mem: &[u64]) -> Self;
    fn store(&self, mem: &mut [u64]);
    fn shl<const AMOUNT: u32>(self) -> Self;
    fn shr<const AMOUNT: u32>(self) -> Self;
    fn bitcast_to_u32(self) -> <<Self as U64SimdVec>::Descriptor as SimdDescriptorCompat>::U32Vec;
}

impl<D: SimdDescriptor + SimdDescriptorCompat<U32Vec = U32Vec<D>>> U64SimdVec for U64Vec<D> {
    type Descriptor = D;
    const LEN: usize = Self::LEN;

    #[inline(always)]
    fn splat(d: D, v: u64) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn load(d: D, mem: &[u64]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u64]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        self.shl::<AMOUNT>()
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        self.shr::<AMOUNT>()
    }
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32Vec<D> {
        self.bitcast::<u32>()
    }
}

pub trait U32SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;
    const LEN: usize;

    fn bitcast_to_i32(self) -> <<Self as U32SimdVec>::Descriptor as SimdDescriptorCompat>::I32Vec;
    fn shr<const AMOUNT: u32>(self) -> Self;
}

impl<D: SimdDescriptor + SimdDescriptorCompat<I32Vec = I32Vec<D>>> U32SimdVec for U32Vec<D> {
    type Descriptor = D;
    const LEN: usize = Self::LEN;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32Vec<D> {
        self.bitcast::<i32>()
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        self.shr::<AMOUNT>()
    }
}

pub trait U8SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;
    const LEN: usize;

    fn load(d: Self::Descriptor, mem: &[u8]) -> Self;
    fn splat(d: Self::Descriptor, v: u8) -> Self;
    fn store(&self, mem: &mut [u8]);
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]);
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]);
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]);
}

impl<D: SimdDescriptor> U8SimdVec for U8Vec<D> {
    type Descriptor = D;
    const LEN: usize = Self::LEN;

    #[inline(always)]
    fn load(d: D, mem: &[u8]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn splat(d: D, v: u8) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        Self::store_interleaved_2(a, b, dest)
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        Self::store_interleaved_3(a, b, c, dest)
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        Self::store_interleaved_4(a, b, c, d, dest)
    }
}

pub trait U16SimdVec: Sized + Copy + Debug + Send + Sync {
    type Descriptor: SimdDescriptor;
    const LEN: usize;

    fn load(d: Self::Descriptor, mem: &[u16]) -> Self;
    fn splat(d: Self::Descriptor, v: u16) -> Self;
    fn store(&self, mem: &mut [u16]);
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]);
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]);
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]);
    fn bitcast_i16(self) -> <<Self as U16SimdVec>::Descriptor as SimdDescriptorCompat>::I16Vec;
}

impl<D: SimdDescriptor + SimdDescriptorCompat<I16Vec = I16Vec<D>>> U16SimdVec for U16Vec<D> {
    type Descriptor = D;
    const LEN: usize = Self::LEN;

    #[inline(always)]
    fn load(d: D, mem: &[u16]) -> Self {
        Self::load(d, mem)
    }
    #[inline(always)]
    fn splat(d: D, v: u16) -> Self {
        Self::splat(d, v)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        (*self).store(mem)
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        Self::store_interleaved_2(a, b, dest)
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        Self::store_interleaved_3(a, b, c, dest)
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        Self::store_interleaved_4(a, b, c, d, dest)
    }
    #[inline(always)]
    fn bitcast_i16(self) -> I16Vec<D> {
        self.bitcast::<i16>()
    }
}

pub trait SimdMask:
    Sized + Copy + Debug + Send + Sync + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self>
{
    type Descriptor: SimdDescriptor;

    fn if_then_else_f32(
        self,
        if_true: <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::F32Vec,
        if_false: <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::F32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::F32Vec;
    fn if_then_else_i32(
        self,
        if_true: <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::I32Vec,
        if_false: <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::I32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::I32Vec;
    fn maskz_i32(
        self,
        v: <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::I32Vec,
    ) -> <<Self as SimdMask>::Descriptor as SimdDescriptorCompat>::I32Vec;
    fn all(self) -> bool;
    fn andnot(self, rhs: Self) -> Self;
}

impl<D: SimdDescriptor + SimdDescriptorCompat<F32Vec = F32Vec<D>, I32Vec = I32Vec<D>>> SimdMask
    for Mask<D>
{
    type Descriptor = D;

    #[inline(always)]
    fn if_then_else_f32(self, if_true: F32Vec<D>, if_false: F32Vec<D>) -> F32Vec<D> {
        self.if_then_else(if_true, if_false)
    }
    #[inline(always)]
    fn if_then_else_i32(self, if_true: I32Vec<D>, if_false: I32Vec<D>) -> I32Vec<D> {
        self.if_then_else(if_true, if_false)
    }
    #[inline(always)]
    fn maskz_i32(self, v: I32Vec<D>) -> I32Vec<D> {
        self.if_then_zero_else(v)
    }
    #[inline(always)]
    fn all(self) -> bool {
        self.all()
    }
    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        self.andnot(rhs)
    }
}

pub trait SimdMask16:
    Sized + Copy + Debug + Send + Sync + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self>
{
    type Descriptor: SimdDescriptor;

    fn if_then_else_i16(
        self,
        if_true: <<Self as SimdMask16>::Descriptor as SimdDescriptorCompat>::I16Vec,
        if_false: <<Self as SimdMask16>::Descriptor as SimdDescriptorCompat>::I16Vec,
    ) -> <<Self as SimdMask16>::Descriptor as SimdDescriptorCompat>::I16Vec;
    fn maskz_i16(
        self,
        v: <<Self as SimdMask16>::Descriptor as SimdDescriptorCompat>::I16Vec,
    ) -> <<Self as SimdMask16>::Descriptor as SimdDescriptorCompat>::I16Vec;
    fn all(self) -> bool;
    fn andnot(self, rhs: Self) -> Self;
}

impl<D: SimdDescriptor + SimdDescriptorCompat<I16Vec = I16Vec<D>>> SimdMask16 for Mask16<D> {
    type Descriptor = D;

    #[inline(always)]
    fn if_then_else_i16(self, if_true: I16Vec<D>, if_false: I16Vec<D>) -> I16Vec<D> {
        self.if_then_else(if_true, if_false)
    }
    #[inline(always)]
    fn maskz_i16(self, v: I16Vec<D>) -> I16Vec<D> {
        self.if_then_zero_else(v)
    }
    #[inline(always)]
    fn all(self) -> bool {
        self.all()
    }
    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        self.andnot(rhs)
    }
}

// Legacy 1-lane ScalarDescriptor (compatible with unmigrated tests/callers in Commits 2..5).

#[derive(Clone, Copy, Debug)]
pub struct ScalarDescriptor {}

impl SimdDescriptor128 for ScalarDescriptor {}

impl SimdDescriptor for ScalarDescriptor {
    const BYTES: usize = 4;
    const IS_EMULATED: bool = true;
    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline(always)]
    fn new() -> Option<Self> {
        Some(Self {})
    }
    #[inline(always)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }
    #[inline(always)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }
    #[inline(always)]
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

impl SimdDescriptorCompat for ScalarDescriptor {
    type F32Vec = f32;
    type I32Vec = Wrapping<i32>;
    type I16Vec = Wrapping<i16>;
    type U64Vec = Wrapping<u64>;
    type U32Vec = Wrapping<u32>;
    type U8Vec = u8;
    type U16Vec = u16;
    type Mask = bool;
    type Mask16 = bool;
    type Bf16Table8 = [f32; 8];
}

impl BytesBackend for ScalarDescriptor {
    type RawBytes = [u8; 0];
}

impl DescriptorSealed for ScalarDescriptor {
    type RawBf16Table8 = ();
    fn prepare_table_bf16_8(self, _table: &[f32; 8]) {}
    fn table_lookup_bf16_8(self, _table: (), _indices: ()) {}
}

macro_rules! impl_scalar_dummy_mask {
    ($($w:ty),*) => {
        $(impl MaskBackend<$w> for ScalarDescriptor {
            type MaskRaw = ();
            fn mask_and(self, _a: (), _b: ()) {}
            fn mask_or(self, _a: (), _b: ()) {}
            fn mask_xor(self, _a: (), _b: ()) {}
            fn mask_not(self, _a: ()) {}
            fn mask_andnot(self, _a: (), _b: ()) {}
            fn mask_all(self, _a: ()) -> bool { unreachable!() }
            fn mask_any(self, _a: ()) -> bool { unreachable!() }
        })*
    };
}
impl_scalar_dummy_mask!(W8, W16, W32, W64);

macro_rules! impl_scalar_dummy_lane {
    ($($t:ty),*) => {
        $(
            impl LaneBackend<$t> for ScalarDescriptor {
                type Raw = ();
                type Array = [$t; 1];
                fn zero(self) {}
                fn splat(self, _v: $t) {}
                fn extract_lane0(self, _a: ()) -> $t { unreachable!() }
                fn load(self, _mem: &[$t]) {}
                fn store(self, _a: (), _dest: &mut [$t]) {}
                fn load_array(self, _mem: &[$t; 1]) {}
                fn store_array(self, _a: (), _dest: &mut [$t; 1]) {}
                fn zero_array() -> [$t; 1] { [<$t as crate::sealed::Element>::ZERO; 1] }
                fn slice_as_array(_slice: &[$t]) -> &[[$t; 1]] { unreachable!() }
                fn slice_as_array_mut(_slice: &mut [$t]) -> &mut [[$t; 1]] { unreachable!() }
                fn to_raw_bytes(self, _a: ()) -> [u8; 0] { [] }
                fn from_raw_bytes(self, _a: [u8; 0]) {}
                fn interleave_lo(self, _a: (), _b: ()) {}
                fn interleave_hi(self, _a: (), _b: ()) {}
            }
            impl CompareBackend<$t> for ScalarDescriptor {
                fn eq(self, _a: (), _b: ()) {}
                fn gt(self, _a: (), _b: ()) {}
                fn eq_zero(self, _a: ()) {}
                fn min(self, _a: (), _b: ()) {}
                fn max(self, _a: (), _b: ()) {}
                fn mask_select(self, _mask: (), _if_true: (), _if_false: ()) {}
                fn mask_zero_select(self, _mask: (), _if_false: ()) {}
            }
            impl ArithBackend<$t> for ScalarDescriptor {
                fn add(self, _a: (), _b: ()) {}
                fn sub(self, _a: (), _b: ()) {}
            }
            impl MulBackend<$t> for ScalarDescriptor {
                fn mul(self, _a: (), _b: ()) {}
            }
        )*
    };
}
impl_scalar_dummy_lane!(f32, u8, i16, u16, i32, u32, u64);

macro_rules! impl_scalar_dummy_signed {
    ($($t:ty),*) => {
        $(impl SignedBackend<$t> for ScalarDescriptor {
            fn neg(self, _a: ()) {}
            fn abs(self, _a: ()) {}
            fn lt_zero(self, _a: ()) {}
        })*
    };
}
impl_scalar_dummy_signed!(f32, i16, i32);

impl FloatBackend<f32> for ScalarDescriptor {
    fn div(self, _a: (), _b: ()) {}
    fn mul_add(self, _a: (), _mul: (), _add: ()) {}
    fn neg_mul_add(self, _a: (), _mul: (), _add: ()) {}
    fn sqrt(self, _a: ()) {}
    fn floor(self, _a: ()) {}
    fn ceil(self, _a: ()) {}
    fn round(self, _a: ()) {}
    fn copysign(self, _a: (), _sign: ()) {}
}

macro_rules! impl_scalar_dummy_intbit {
    ($($t:ty),*) => {
        $(impl IntBitBackend<$t> for ScalarDescriptor {
            fn bit_and(self, _a: (), _b: ()) {}
            fn bit_or(self, _a: (), _b: ()) {}
            fn bit_xor(self, _a: (), _b: ()) {}
            fn bit_not(self, _a: ()) {}
            fn andnot(self, _not_a: (), _b: ()) {}
            fn shl<const AMOUNT: u32>(self, _a: ()) {}
            fn shr<const AMOUNT: u32>(self, _a: ()) {}
        })*
    };
}
impl_scalar_dummy_intbit!(u8, i16, u16, i32, u32, u64);

impl F32ExtraBackend for ScalarDescriptor {
    fn f32_as_i32(self, _a: ()) {}
    fn f32_round_store_u8(self, _a: (), _dest: &mut [u8]) {}
    fn f32_round_store_u16(self, _a: (), _dest: &mut [u16]) {}
    fn f32_load_f16_bits(self, _mem: &[u16]) {}
    fn f32_store_f16_bits(self, _a: (), _dest: &mut [u16]) {}
}

impl I32ExtraBackend for ScalarDescriptor {
    fn i32_as_f32(self, _a: ()) {}
    fn i32_load_from_i16(self, _mem: &[i16]) {}
    fn i32_load_from_u16(self, _mem: &[u16]) {}
    fn i32_store_u16(self, _a: (), _dest: &mut [u16]) {}
    fn i32_store_u8(self, _a: (), _dest: &mut [u8]) {}
    fn i32_mul_wide_take_high(self, _a: (), _b: ()) {}
}

impl I16ExtraBackend for ScalarDescriptor {
    fn i16_store_u8(self, _a: (), _dest: &mut [u8]) {}
    fn i16_mul_wide_take_high(self, _a: (), _b: ()) {}
}

impl U8ExtraBackend for ScalarDescriptor {
    fn u8_swizzle_128_or_zero(self, _a: (), _indices: ()) {}
}

impl F32SimdVec for f32 {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;
    type UnderlyingArray = [f32; 1];

    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[f32]) -> Self {
        mem[0]
    }
    #[inline(always)]
    fn load_array(_d: Self::Descriptor, mem: &[f32; 1]) -> Self {
        mem[0]
    }
    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        mem[0] = *self;
    }
    #[inline(always)]
    fn store_array(&self, mem: &mut [f32; 1]) {
        mem[0] = *self;
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        dest[0] = a;
        dest[1] = b;
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
        dest[3] = d;
    }
    #[inline(always)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
        dest[3] = d;
        dest[4] = e;
        dest[5] = f;
        dest[6] = g;
        dest[7] = h;
    }
    #[inline(always)]
    fn load_deinterleaved_2(_d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        (src[0], src[1])
    }
    #[inline(always)]
    fn load_deinterleaved_3(_d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        (src[0], src[1], src[2])
    }
    #[inline(always)]
    fn load_deinterleaved_4(_d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        (src[0], src[1], src[2], src[3])
    }
    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        (self * mul) + add
    }
    #[inline(always)]
    fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        -(self * mul) + add
    }
    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn zero(_d: Self::Descriptor) -> Self {
        0.0
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }
    #[inline(always)]
    fn as_i32(self) -> Wrapping<i32> {
        Wrapping(self as i32)
    }
    #[inline(always)]
    fn bitcast_to_i32(self) -> Wrapping<i32> {
        Wrapping(self.to_bits() as i32)
    }
    #[inline(always)]
    fn prepare_table_bf16_8(_d: Self::Descriptor, table: &[f32; 8]) -> [f32; 8] {
        *table
    }
    #[inline(always)]
    fn table_lookup_bf16_8(_d: Self::Descriptor, table: [f32; 8], indices: Wrapping<i32>) -> Self {
        table[indices.0 as usize]
    }
    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        dest[0] = self.round() as u8;
    }
    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        dest[0] = self.round() as u16;
    }
    #[inline(always)]
    fn load_f16_bits(_d: Self::Descriptor, mem: &[u16]) -> Self {
        f16::from_bits(mem[0]).to_f32()
    }
    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        dest[0] = f16::from_f32(self).to_bits();
    }
    #[inline(always)]
    fn make_array_slice(slice: &[f32]) -> &[[f32; 1]] {
        let (ret, rem) = slice.as_chunks();
        assert!(rem.is_empty());
        ret
    }
    #[inline(always)]
    fn make_array_slice_mut(slice: &mut [f32]) -> &mut [[f32; 1]] {
        let (ret, rem) = slice.as_chunks_mut();
        assert!(rem.is_empty());
        ret
    }
    #[inline(always)]
    fn transpose_square(_d: Self::Descriptor, _data: &mut [[f32; 1]], _stride: usize) {}
}

impl I32SimdVec for Wrapping<i32> {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;

    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: i32) -> Self {
        Wrapping(v)
    }
    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[i32]) -> Self {
        Wrapping(mem[0])
    }
    #[inline(always)]
    fn load_from_i16(_d: Self::Descriptor, mem: &[i16]) -> Self {
        Wrapping(mem[0] as i32)
    }
    #[inline(always)]
    fn load_from_u16(_d: Self::Descriptor, mem: &[u16]) -> Self {
        Wrapping(mem[0] as i32)
    }
    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        mem[0] = self.0;
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Wrapping(self.0.abs())
    }
    #[inline(always)]
    fn as_f32(self) -> f32 {
        self.0 as f32
    }
    #[inline(always)]
    fn bitcast_to_f32(self) -> f32 {
        f32::from_bits(self.0 as u32)
    }
    #[inline(always)]
    fn bitcast_to_u32(self) -> Wrapping<u32> {
        Wrapping(self.0 as u32)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self.0 > other.0
    }
    #[inline(always)]
    fn lt_zero(self) -> bool {
        self.0 < 0
    }
    #[inline(always)]
    fn eq(self, other: Self) -> bool {
        self.0 == other.0
    }
    #[inline(always)]
    fn eq_zero(self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 << AMOUNT)
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 >> AMOUNT)
    }
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        Wrapping(((self.0 as i64 * rhs.0 as i64) >> 32) as i32)
    }
    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        dest[0] = self.0 as u16;
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i32]) {
        dest[0] = a.0;
        dest[1] = b.0;
    }
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        dest[0] = self.0 as u8;
    }
}

impl I16SimdVec for Wrapping<i16> {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;
    type UnderlyingArray = [i16; 1];

    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: i16) -> Self {
        Wrapping(v)
    }
    #[inline(always)]
    fn zero(_d: Self::Descriptor) -> Self {
        Wrapping(0)
    }
    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[i16]) -> Self {
        assert!(!mem.is_empty());
        Wrapping(mem[0])
    }
    #[inline(always)]
    fn load_array(_d: Self::Descriptor, mem: &[i16; 1]) -> Self {
        Wrapping(mem[0])
    }
    #[inline(always)]
    fn store(&self, mem: &mut [i16]) {
        assert!(!mem.is_empty());
        mem[0] = self.0;
    }
    #[inline(always)]
    fn store_array(&self, mem: &mut [i16; 1]) {
        mem[0] = self.0;
    }
    #[inline(always)]
    fn make_array_slice(slice: &[i16]) -> &[[i16; 1]] {
        let (ret, rem) = slice.as_chunks();
        assert!(rem.is_empty());
        ret
    }
    #[inline(always)]
    fn make_array_slice_mut(slice: &mut [i16]) -> &mut [[i16; 1]] {
        let (ret, rem) = slice.as_chunks_mut();
        assert!(rem.is_empty());
        ret
    }
    #[inline(always)]
    fn transpose_square(_d: Self::Descriptor, _data: &mut [[i16; 1]], _stride: usize) {}
    #[inline(always)]
    fn abs(self) -> Self {
        Wrapping(self.0.wrapping_abs())
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 << AMOUNT)
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 >> AMOUNT)
    }
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        Wrapping(((self.0 as i32 * rhs.0 as i32) >> 16) as i16)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self.0 > other.0
    }
    #[inline(always)]
    fn lt_zero(self) -> bool {
        self.0 < 0
    }
    #[inline(always)]
    fn eq(self, other: Self) -> bool {
        self.0 == other.0
    }
    #[inline(always)]
    fn eq_zero(self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    fn bitcast_u16(self) -> u16 {
        self.0 as u16
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [i16]) {
        dest[0] = a.0;
        dest[1] = b.0;
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [i16]) {
        dest[0] = a.0;
        dest[1] = b.0;
        dest[2] = c.0;
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [i16]) {
        dest[0] = a.0;
        dest[1] = b.0;
        dest[2] = c.0;
        dest[3] = d.0;
    }
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        assert!(!dest.is_empty());
        dest[0] = self.0 as u8;
    }
}

impl U32SimdVec for Wrapping<u32> {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;

    #[inline(always)]
    fn bitcast_to_i32(self) -> Wrapping<i32> {
        Wrapping(self.0 as i32)
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 >> AMOUNT)
    }
}

impl U64SimdVec for Wrapping<u64> {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;

    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: u64) -> Self {
        Wrapping(v)
    }
    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[u64]) -> Self {
        Wrapping(mem[0])
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u64]) {
        mem[0] = self.0;
    }
    #[inline(always)]
    fn shl<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 << AMOUNT)
    }
    #[inline(always)]
    fn shr<const AMOUNT: u32>(self) -> Self {
        Wrapping(self.0 >> AMOUNT)
    }
    #[inline(always)]
    fn bitcast_to_u32(self) -> Wrapping<u32> {
        Wrapping(self.0 as u32)
    }
}

impl U8SimdVec for u8 {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;

    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[u8]) -> Self {
        mem[0]
    }
    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: u8) -> Self {
        v
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        mem[0] = *self;
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        dest[0] = a;
        dest[1] = b;
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
        dest[3] = d;
    }
}

impl U16SimdVec for u16 {
    type Descriptor = ScalarDescriptor;
    const LEN: usize = 1;

    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[u16]) -> Self {
        mem[0]
    }
    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: u16) -> Self {
        v
    }
    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        mem[0] = *self;
    }
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        dest[0] = a;
        dest[1] = b;
    }
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
    }
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        dest[0] = a;
        dest[1] = b;
        dest[2] = c;
        dest[3] = d;
    }
    #[inline(always)]
    fn bitcast_i16(self) -> Wrapping<i16> {
        Wrapping(self as i16)
    }
}

impl SimdMask for bool {
    type Descriptor = ScalarDescriptor;

    #[inline(always)]
    fn if_then_else_f32(self, if_true: f32, if_false: f32) -> f32 {
        if self { if_true } else { if_false }
    }
    #[inline(always)]
    fn if_then_else_i32(self, if_true: Wrapping<i32>, if_false: Wrapping<i32>) -> Wrapping<i32> {
        if self { if_true } else { if_false }
    }
    #[inline(always)]
    fn maskz_i32(self, v: Wrapping<i32>) -> Wrapping<i32> {
        if self { Wrapping(0) } else { v }
    }
    #[inline(always)]
    fn all(self) -> bool {
        self
    }
    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        (!self) & rhs
    }
}

impl SimdMask16 for bool {
    type Descriptor = ScalarDescriptor;

    #[inline(always)]
    fn if_then_else_i16(self, if_true: Wrapping<i16>, if_false: Wrapping<i16>) -> Wrapping<i16> {
        if self { if_true } else { if_false }
    }
    #[inline(always)]
    fn maskz_i16(self, v: Wrapping<i16>) -> Wrapping<i16> {
        if self { Wrapping(0) } else { v }
    }
    #[inline(always)]
    fn all(self) -> bool {
        self
    }
    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        (!self) & rhs
    }
}
