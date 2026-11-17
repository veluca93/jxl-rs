// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::wrong_self_convention,
    dead_code
)]

use std::fmt::Debug;

use crate::{W8, W16, W32, W64, generic_ops};

pub trait MaskWidth: Sized + Copy + Debug + Send + Sync + 'static {
    type Raw<D: crate::SimdDescriptor>: Copy + Debug + Send + Sync + 'static;
    fn mask_and<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn mask_or<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn mask_xor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn mask_not<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn mask_andnot<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn mask_all<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> bool;
    fn mask_any<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> bool;
}

pub trait Element: Copy + Default + Debug + Send + Sync + 'static {
    const ZERO: Self;
    type Raw<D: crate::SimdDescriptor>: Copy + Debug + Send + Sync + 'static;
    type Array<D: crate::SimdDescriptor>: AsRef<[Self]>
        + AsMut<[Self]>
        + Copy
        + Debug
        + Send
        + Sync
        + 'static;
    fn zero<D: crate::SimdDescriptor>(d: D) -> Self::Raw<D>;
    fn splat<D: crate::SimdDescriptor>(d: D, v: Self) -> Self::Raw<D>;
    fn extract_lane0<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self;
    fn load<D: crate::SimdDescriptor>(d: D, mem: &[Self]) -> Self::Raw<D>;
    fn store<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, dest: &mut [Self]);
    fn load_array<D: crate::SimdDescriptor>(d: D, mem: &Self::Array<D>) -> Self::Raw<D>;
    fn store_array<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, dest: &mut Self::Array<D>);
    fn zero_array<D: crate::SimdDescriptor>() -> Self::Array<D>;
    fn slice_as_array<D: crate::SimdDescriptor>(slice: &[Self]) -> &[Self::Array<D>];
    fn slice_as_array_mut<D: crate::SimdDescriptor>(slice: &mut [Self]) -> &mut [Self::Array<D>];
    fn to_raw_bytes<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> D::RawBytes;
    fn from_raw_bytes<D: crate::SimdDescriptor>(d: D, a: D::RawBytes) -> Self::Raw<D>;
    fn interleave_lo<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn interleave_hi<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn store_interleaved_2<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
        dest: &mut [Self],
    );
    fn store_interleaved_3<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
        c: Self::Raw<D>,
        dest: &mut [Self],
    );
    fn store_interleaved_4<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
        c: Self::Raw<D>,
        e: Self::Raw<D>,
        dest: &mut [Self],
    );
}

pub trait ComparableElement: Element {
    type Mask: crate::MaskWidth;
    fn eq<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
    ) -> <Self::Mask as MaskWidth>::Raw<D>;
    fn gt<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        b: Self::Raw<D>,
    ) -> <Self::Mask as MaskWidth>::Raw<D>;
    fn eq_zero<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
    ) -> <Self::Mask as MaskWidth>::Raw<D>;
    fn min<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn max<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn mask_select<D: crate::SimdDescriptor>(
        d: D,
        mask: <Self::Mask as MaskWidth>::Raw<D>,
        if_true: Self::Raw<D>,
        if_false: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn mask_zero_select<D: crate::SimdDescriptor>(
        d: D,
        mask: <Self::Mask as MaskWidth>::Raw<D>,
        if_false: Self::Raw<D>,
    ) -> Self::Raw<D>;
}

pub trait ArithElement: Element {
    fn add<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn sub<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
}

pub trait MulElement: ArithElement {
    fn mul<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
}

pub trait SignedElement: ComparableElement + ArithElement {
    fn neg<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn abs<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn lt_zero<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
    ) -> <Self::Mask as MaskWidth>::Raw<D>;
}

pub trait FloatElement: SignedElement + MulElement {
    fn div<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn mul_add<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        mul: Self::Raw<D>,
        add: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn neg_mul_add<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        mul: Self::Raw<D>,
        add: Self::Raw<D>,
    ) -> Self::Raw<D>;
    fn sqrt<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn floor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn ceil<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn round<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn copysign<D: crate::SimdDescriptor>(
        d: D,
        a: Self::Raw<D>,
        sign: Self::Raw<D>,
    ) -> Self::Raw<D>;
}

pub trait IntElement: ComparableElement + ArithElement + MulElement {
    const ONE: Self;
    fn bit_and<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn bit_or<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn bit_xor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D>;
    fn bit_not<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn andnot<D: crate::SimdDescriptor>(d: D, not_a: Self::Raw<D>, b: Self::Raw<D>)
    -> Self::Raw<D>;
    fn shl<D: crate::SimdDescriptor, const AMOUNT: u32>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
    fn shr<D: crate::SimdDescriptor, const AMOUNT: u32>(d: D, a: Self::Raw<D>) -> Self::Raw<D>;
}

pub trait MaskBackend<W: MaskWidth>: Copy {
    type MaskRaw: Copy + Debug + Send + Sync + 'static;
    fn mask_and(self, a: Self::MaskRaw, b: Self::MaskRaw) -> Self::MaskRaw;
    fn mask_or(self, a: Self::MaskRaw, b: Self::MaskRaw) -> Self::MaskRaw;
    fn mask_xor(self, a: Self::MaskRaw, b: Self::MaskRaw) -> Self::MaskRaw;
    fn mask_not(self, a: Self::MaskRaw) -> Self::MaskRaw;
    fn mask_andnot(self, a: Self::MaskRaw, b: Self::MaskRaw) -> Self::MaskRaw;
    fn mask_all(self, a: Self::MaskRaw) -> bool;
    fn mask_any(self, a: Self::MaskRaw) -> bool;
}

pub trait BytesBackend: Copy {
    type RawBytes: Copy + Debug + Send + Sync + 'static;
    #[inline(always)]
    fn interleave_128_halves(
        self,
        _a: Self::RawBytes,
        _b: Self::RawBytes,
    ) -> (Self::RawBytes, Self::RawBytes) {
        unreachable!("128-bit descriptor")
    }
    #[inline(always)]
    fn interleave_256_halves(
        self,
        _a: Self::RawBytes,
        _b: Self::RawBytes,
    ) -> (Self::RawBytes, Self::RawBytes) {
        unreachable!("<512-bit descriptor")
    }
}

pub trait LaneBackend<T: Copy + Default + Debug + 'static>: BytesBackend {
    type Raw: Copy + Debug + Send + Sync + 'static;
    type Array: AsRef<[T]> + AsMut<[T]> + Copy + Debug + Send + Sync + 'static;
    fn zero(self) -> Self::Raw;
    fn splat(self, v: T) -> Self::Raw;
    fn extract_lane0(self, a: Self::Raw) -> T;
    fn load(self, mem: &[T]) -> Self::Raw;
    fn store(self, a: Self::Raw, dest: &mut [T]);
    fn load_array(self, mem: &Self::Array) -> Self::Raw;
    fn store_array(self, a: Self::Raw, dest: &mut Self::Array);
    fn zero_array() -> Self::Array;
    fn slice_as_array(slice: &[T]) -> &[Self::Array];
    fn slice_as_array_mut(slice: &mut [T]) -> &mut [Self::Array];
    fn to_raw_bytes(self, a: Self::Raw) -> Self::RawBytes;
    fn from_raw_bytes(self, a: Self::RawBytes) -> Self::Raw;
    fn interleave_lo(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn interleave_hi(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    #[inline(always)]
    fn store_interleaved_2(
        self,
        a: <Self as LaneBackend<T>>::Raw,
        b: <Self as LaneBackend<T>>::Raw,
        dest: &mut [T],
    ) where
        Self: crate::SimdDescriptor,
    {
        generic_ops::store_interleaved_2(self, a, b, dest);
    }
    #[inline(always)]
    fn store_interleaved_3(
        self,
        a: <Self as LaneBackend<T>>::Raw,
        b: <Self as LaneBackend<T>>::Raw,
        c: <Self as LaneBackend<T>>::Raw,
        dest: &mut [T],
    ) where
        Self: crate::SimdDescriptor,
    {
        generic_ops::store_interleaved_3(self, a, b, c, dest);
    }
    #[inline(always)]
    fn store_interleaved_4(
        self,
        a: <Self as LaneBackend<T>>::Raw,
        b: <Self as LaneBackend<T>>::Raw,
        c: <Self as LaneBackend<T>>::Raw,
        e: <Self as LaneBackend<T>>::Raw,
        dest: &mut [T],
    ) where
        Self: crate::SimdDescriptor,
    {
        generic_ops::store_interleaved_4(self, a, b, c, e, dest);
    }
}

pub trait CompareBackend<T: ComparableElement>: LaneBackend<T> + MaskBackend<T::Mask> {
    fn eq(self, a: Self::Raw, b: Self::Raw) -> <Self as MaskBackend<T::Mask>>::MaskRaw;
    fn gt(self, a: Self::Raw, b: Self::Raw) -> <Self as MaskBackend<T::Mask>>::MaskRaw;
    fn eq_zero(self, a: Self::Raw) -> <Self as MaskBackend<T::Mask>>::MaskRaw;
    fn min(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn max(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn mask_select(
        self,
        mask: <Self as MaskBackend<T::Mask>>::MaskRaw,
        if_true: Self::Raw,
        if_false: Self::Raw,
    ) -> Self::Raw;
    fn mask_zero_select(
        self,
        mask: <Self as MaskBackend<T::Mask>>::MaskRaw,
        if_false: Self::Raw,
    ) -> Self::Raw;
}

pub trait ArithBackend<T: ArithElement>: LaneBackend<T> {
    fn add(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn sub(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
}

pub trait MulBackend<T: MulElement>: ArithBackend<T> {
    fn mul(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
}

pub trait SignedBackend<T: SignedElement>: CompareBackend<T> + ArithBackend<T> {
    fn neg(self, a: Self::Raw) -> Self::Raw;
    fn abs(self, a: Self::Raw) -> Self::Raw;
    fn lt_zero(self, a: Self::Raw) -> <Self as MaskBackend<T::Mask>>::MaskRaw;
}

pub trait FloatBackend<T: FloatElement>: SignedBackend<T> + MulBackend<T> {
    fn div(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn mul_add(self, a: Self::Raw, mul: Self::Raw, add: Self::Raw) -> Self::Raw;
    fn neg_mul_add(self, a: Self::Raw, mul: Self::Raw, add: Self::Raw) -> Self::Raw;
    fn sqrt(self, a: Self::Raw) -> Self::Raw;
    fn floor(self, a: Self::Raw) -> Self::Raw;
    fn ceil(self, a: Self::Raw) -> Self::Raw;
    fn round(self, a: Self::Raw) -> Self::Raw;
    fn copysign(self, a: Self::Raw, sign: Self::Raw) -> Self::Raw;
}

pub trait IntBitBackend<T: IntElement>: CompareBackend<T> + MulBackend<T> {
    fn bit_and(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn bit_or(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn bit_xor(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn bit_not(self, a: Self::Raw) -> Self::Raw;
    fn andnot(self, not_a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn shl<const AMOUNT: u32>(self, a: Self::Raw) -> Self::Raw;
    fn shr<const AMOUNT: u32>(self, a: Self::Raw) -> Self::Raw;
}

pub trait F32ExtraBackend: LaneBackend<f32> + LaneBackend<i32> {
    fn f32_as_i32(self, a: <Self as LaneBackend<f32>>::Raw) -> <Self as LaneBackend<i32>>::Raw;
    fn f32_round_store_u8(self, a: <Self as LaneBackend<f32>>::Raw, dest: &mut [u8]);
    fn f32_round_store_u16(self, a: <Self as LaneBackend<f32>>::Raw, dest: &mut [u16]);
    fn f32_load_f16_bits(self, mem: &[u16]) -> <Self as LaneBackend<f32>>::Raw;
    fn f32_store_f16_bits(self, a: <Self as LaneBackend<f32>>::Raw, dest: &mut [u16]);
}

pub trait I32ExtraBackend: LaneBackend<i32> + LaneBackend<f32> {
    fn i32_as_f32(self, a: <Self as LaneBackend<i32>>::Raw) -> <Self as LaneBackend<f32>>::Raw;
    fn i32_load_from_i16(self, mem: &[i16]) -> <Self as LaneBackend<i32>>::Raw;
    fn i32_load_from_u16(self, mem: &[u16]) -> <Self as LaneBackend<i32>>::Raw;
    fn i32_store_u16(self, a: <Self as LaneBackend<i32>>::Raw, dest: &mut [u16]);
    fn i32_store_u8(self, a: <Self as LaneBackend<i32>>::Raw, dest: &mut [u8]);
    fn i32_mul_wide_take_high(
        self,
        a: <Self as LaneBackend<i32>>::Raw,
        b: <Self as LaneBackend<i32>>::Raw,
    ) -> <Self as LaneBackend<i32>>::Raw;
}

pub trait I16ExtraBackend: LaneBackend<i16> {
    fn i16_store_u8(self, a: <Self as LaneBackend<i16>>::Raw, dest: &mut [u8]);
    fn i16_mul_wide_take_high(
        self,
        a: <Self as LaneBackend<i16>>::Raw,
        b: <Self as LaneBackend<i16>>::Raw,
    ) -> <Self as LaneBackend<i16>>::Raw;
}

pub trait U8ExtraBackend: LaneBackend<u8> {
    fn u8_swizzle_128_or_zero(
        self,
        a: <Self as LaneBackend<u8>>::Raw,
        indices: <Self as LaneBackend<u8>>::Raw,
    ) -> <Self as LaneBackend<u8>>::Raw;
}

pub trait DescriptorSealed:
    BytesBackend
    + MaskBackend<W8>
    + MaskBackend<W16>
    + MaskBackend<W32>
    + MaskBackend<W64>
    + LaneBackend<f32>
    + LaneBackend<u8>
    + LaneBackend<i16>
    + LaneBackend<u16>
    + LaneBackend<i32>
    + LaneBackend<u32>
    + LaneBackend<u64>
    + CompareBackend<f32>
    + CompareBackend<u8>
    + CompareBackend<i16>
    + CompareBackend<u16>
    + CompareBackend<i32>
    + CompareBackend<u32>
    + CompareBackend<u64>
    + ArithBackend<f32>
    + ArithBackend<u8>
    + ArithBackend<i16>
    + ArithBackend<u16>
    + ArithBackend<i32>
    + ArithBackend<u32>
    + ArithBackend<u64>
    + MulBackend<f32>
    + MulBackend<u8>
    + MulBackend<i16>
    + MulBackend<u16>
    + MulBackend<i32>
    + MulBackend<u32>
    + MulBackend<u64>
    + SignedBackend<f32>
    + SignedBackend<i16>
    + SignedBackend<i32>
    + FloatBackend<f32>
    + IntBitBackend<u8>
    + IntBitBackend<i16>
    + IntBitBackend<u16>
    + IntBitBackend<i32>
    + IntBitBackend<u32>
    + IntBitBackend<u64>
    + F32ExtraBackend
    + I32ExtraBackend
    + I16ExtraBackend
    + U8ExtraBackend
    + Sized
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
{
    type RawBf16Table8: Copy + Debug + Send + Sync + 'static;

    fn prepare_table_bf16_8(self, table: &[f32; 8]) -> Self::RawBf16Table8;
    fn table_lookup_bf16_8(
        self,
        table: Self::RawBf16Table8,
        indices: <Self as LaneBackend<i32>>::Raw,
    ) -> <Self as LaneBackend<f32>>::Raw;
}

macro_rules! impl_mask_width {
    ($($w:ty),*) => {
        $(impl MaskWidth for $w {
            type Raw<D: crate::SimdDescriptor> = <D as MaskBackend<$w>>::MaskRaw;
            #[inline(always)] fn mask_and<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as MaskBackend<$w>>::mask_and(d, a, b) }
            #[inline(always)] fn mask_or<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as MaskBackend<$w>>::mask_or(d, a, b) }
            #[inline(always)] fn mask_xor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as MaskBackend<$w>>::mask_xor(d, a, b) }
            #[inline(always)] fn mask_not<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as MaskBackend<$w>>::mask_not(d, a) }
            #[inline(always)] fn mask_andnot<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as MaskBackend<$w>>::mask_andnot(d, a, b) }
            #[inline(always)] fn mask_all<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> bool { <D as MaskBackend<$w>>::mask_all(d, a) }
            #[inline(always)] fn mask_any<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> bool { <D as MaskBackend<$w>>::mask_any(d, a) }
        })*
    };
}
impl_mask_width!(W8, W16, W32, W64);

macro_rules! impl_elements {
    ($(($t:ty, $z:expr)),*) => {
        $(impl Element for $t {
            const ZERO: Self = $z;
            type Raw<D: crate::SimdDescriptor> = <D as LaneBackend<$t>>::Raw;
            type Array<D: crate::SimdDescriptor> = <D as LaneBackend<$t>>::Array;
            #[inline(always)] fn zero<D: crate::SimdDescriptor>(d: D) -> Self::Raw<D> { <D as LaneBackend<$t>>::zero(d) }
            #[inline(always)] fn splat<D: crate::SimdDescriptor>(d: D, v: Self) -> Self::Raw<D> { <D as LaneBackend<$t>>::splat(d, v) }
            #[inline(always)] fn extract_lane0<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self { <D as LaneBackend<$t>>::extract_lane0(d, a) }
            #[inline(always)] fn load<D: crate::SimdDescriptor>(d: D, mem: &[Self]) -> Self::Raw<D> { <D as LaneBackend<$t>>::load(d, mem) }
            #[inline(always)] fn store<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, dest: &mut [Self]) { <D as LaneBackend<$t>>::store(d, a, dest); }
            #[inline(always)] fn load_array<D: crate::SimdDescriptor>(d: D, mem: &Self::Array<D>) -> Self::Raw<D> { <D as LaneBackend<$t>>::load_array(d, mem) }
            #[inline(always)] fn store_array<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, dest: &mut Self::Array<D>) { <D as LaneBackend<$t>>::store_array(d, a, dest); }
            #[inline(always)] fn zero_array<D: crate::SimdDescriptor>() -> Self::Array<D> { <D as LaneBackend<$t>>::zero_array() }
            #[inline(always)] fn slice_as_array<D: crate::SimdDescriptor>(s: &[Self]) -> &[Self::Array<D>] { <D as LaneBackend<$t>>::slice_as_array(s) }
            #[inline(always)] fn slice_as_array_mut<D: crate::SimdDescriptor>(s: &mut [Self]) -> &mut [Self::Array<D>] { <D as LaneBackend<$t>>::slice_as_array_mut(s) }
            #[inline(always)] fn to_raw_bytes<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> D::RawBytes { <D as LaneBackend<$t>>::to_raw_bytes(d, a) }
            #[inline(always)] fn from_raw_bytes<D: crate::SimdDescriptor>(d: D, a: D::RawBytes) -> Self::Raw<D> { <D as LaneBackend<$t>>::from_raw_bytes(d, a) }
            #[inline(always)] fn interleave_lo<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as LaneBackend<$t>>::interleave_lo(d, a, b) }
            #[inline(always)] fn interleave_hi<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as LaneBackend<$t>>::interleave_hi(d, a, b) }
            #[inline(always)] fn store_interleaved_2<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>, dest: &mut [Self]) { <D as LaneBackend<$t>>::store_interleaved_2(d, a, b, dest); }
            #[inline(always)] fn store_interleaved_3<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>, c: Self::Raw<D>, dest: &mut [Self]) { <D as LaneBackend<$t>>::store_interleaved_3(d, a, b, c, dest); }
            #[inline(always)] fn store_interleaved_4<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>, c: Self::Raw<D>, e: Self::Raw<D>, dest: &mut [Self]) { <D as LaneBackend<$t>>::store_interleaved_4(d, a, b, c, e, dest); }
        })*
    };
}
impl_elements!(
    (f32, 0.0),
    (u8, 0),
    (i16, 0),
    (u16, 0),
    (i32, 0),
    (u32, 0),
    (u64, 0)
);

macro_rules! impl_comparable {
    ($(($t:ty, $w:ty)),*) => {
        $(impl ComparableElement for $t {
            type Mask = $w;
            #[inline(always)] fn eq<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> <Self::Mask as MaskWidth>::Raw<D> { <D as CompareBackend<Self>>::eq(d, a, b) }
            #[inline(always)] fn gt<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> <Self::Mask as MaskWidth>::Raw<D> { <D as CompareBackend<Self>>::gt(d, a, b) }
            #[inline(always)] fn eq_zero<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> <Self::Mask as MaskWidth>::Raw<D> { <D as CompareBackend<Self>>::eq_zero(d, a) }
            #[inline(always)] fn min<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as CompareBackend<Self>>::min(d, a, b) }
            #[inline(always)] fn max<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as CompareBackend<Self>>::max(d, a, b) }
            #[inline(always)] fn mask_select<D: crate::SimdDescriptor>(d: D, m: <Self::Mask as MaskWidth>::Raw<D>, t: Self::Raw<D>, f: Self::Raw<D>) -> Self::Raw<D> { <D as CompareBackend<Self>>::mask_select(d, m, t, f) }
            #[inline(always)] fn mask_zero_select<D: crate::SimdDescriptor>(d: D, m: <Self::Mask as MaskWidth>::Raw<D>, f: Self::Raw<D>) -> Self::Raw<D> { <D as CompareBackend<Self>>::mask_zero_select(d, m, f) }
        })*
    };
}
impl_comparable!(
    (f32, W32),
    (u8, W8),
    (i16, W16),
    (u16, W16),
    (i32, W32),
    (u32, W32),
    (u64, W64)
);

macro_rules! impl_arith {
    ($($t:ty),*) => {
        $(impl ArithElement for $t {
            #[inline(always)] fn add<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as ArithBackend<Self>>::add(d, a, b) }
            #[inline(always)] fn sub<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as ArithBackend<Self>>::sub(d, a, b) }
        })*
    };
}
impl_arith!(f32, u8, i16, u16, i32, u32, u64);

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(impl MulElement for $t {
            #[inline(always)] fn mul<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as MulBackend<Self>>::mul(d, a, b) }
        })*
    };
}
impl_mul!(f32, u8, i16, u16, i32, u32, u64);

macro_rules! impl_signed {
    ($($t:ty),*) => {
        $(impl SignedElement for $t {
            #[inline(always)] fn neg<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as SignedBackend<Self>>::neg(d, a) }
            #[inline(always)] fn abs<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as SignedBackend<Self>>::abs(d, a) }
            #[inline(always)] fn lt_zero<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> <Self::Mask as MaskWidth>::Raw<D> { <D as SignedBackend<Self>>::lt_zero(d, a) }
        })*
    };
}
impl_signed!(f32, i16, i32);

macro_rules! impl_float {
    ($($t:ty),*) => {
        $(impl FloatElement for $t {
            #[inline(always)] fn div<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::div(d, a, b) }
            #[inline(always)] fn mul_add<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, m: Self::Raw<D>, ad: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::mul_add(d, a, m, ad) }
            #[inline(always)] fn neg_mul_add<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, m: Self::Raw<D>, ad: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::neg_mul_add(d, a, m, ad) }
            #[inline(always)] fn sqrt<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::sqrt(d, a) }
            #[inline(always)] fn floor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::floor(d, a) }
            #[inline(always)] fn ceil<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::ceil(d, a) }
            #[inline(always)] fn round<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::round(d, a) }
            #[inline(always)] fn copysign<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, s: Self::Raw<D>) -> Self::Raw<D> { <D as FloatBackend<Self>>::copysign(d, a, s) }
        })*
    };
}
impl_float!(f32);

macro_rules! impl_int {
    ($($t:ty),*) => {
        $(impl IntElement for $t {
            const ONE: Self = 1;
            #[inline(always)] fn bit_and<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::bit_and(d, a, b) }
            #[inline(always)] fn bit_or<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::bit_or(d, a, b) }
            #[inline(always)] fn bit_xor<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::bit_xor(d, a, b) }
            #[inline(always)] fn bit_not<D: crate::SimdDescriptor>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::bit_not(d, a) }
            #[inline(always)] fn andnot<D: crate::SimdDescriptor>(d: D, na: Self::Raw<D>, b: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::andnot(d, na, b) }
            #[inline(always)] fn shl<D: crate::SimdDescriptor, const N: u32>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::shl::<N>(d, a) }
            #[inline(always)] fn shr<D: crate::SimdDescriptor, const N: u32>(d: D, a: Self::Raw<D>) -> Self::Raw<D> { <D as IntBitBackend<Self>>::shr::<N>(d, a) }
        })*
    };
}
impl_int!(u8, i16, u16, i32, u32, u64);
