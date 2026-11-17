// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]
#![allow(clippy::too_many_arguments, private_bounds)]

extern crate self as jxl_simd;

use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
mod wasm32;

mod emu128;
pub mod float16;
mod generic_ops;
mod sealed;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub use aarch64::NeonDescriptor;
pub use emu128::Emu128Descriptor;
pub use float16::f16;
pub use jxl_simd_macros::{simd_dispatch, simd_test};
#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
pub use wasm32::Simd128Descriptor;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use x86_64::Avx512Descriptor;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub use x86_64::AvxDescriptor;
#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
pub use x86_64::Sse42Descriptor;

#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_sse42 {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "sse42")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_sse42 {
    ($($tt:tt)*) => {};
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_avx {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_avx {
    ($($tt:tt)*) => {};
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_avx512 {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_avx512 {
    ($($tt:tt)*) => {};
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_neon {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_neon {
    ($($tt:tt)*) => {};
}

#[cfg(all(target_arch = "wasm32", feature = "simd128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_simd128 {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

#[cfg(not(all(target_arch = "wasm32", feature = "simd128")))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_simd128 {
    ($($tt:tt)*) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_dispatch {
    (|$d:ident| $call:expr) => {{
        $crate::__if_avx512! {
            if let Some($d) = <$crate::Avx512Descriptor as $crate::SimdDescriptor>::new() {
                return $crate::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $call,
                );
            }
        }
        $crate::__if_avx! {
            if let Some($d) = <$crate::AvxDescriptor as $crate::SimdDescriptor>::new() {
                return $crate::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $call,
                );
            }
        }
        $crate::__if_sse42! {
            if let Some($d) = <$crate::Sse42Descriptor as $crate::SimdDescriptor>::new() {
                return $crate::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $call,
                );
            }
        }
        $crate::__if_neon! {
            if let Some($d) = <$crate::NeonDescriptor as $crate::SimdDescriptor>::new() {
                return $crate::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $call,
                );
            }
        }
        $crate::__if_simd128! {
            if let Some($d) = <$crate::Simd128Descriptor as $crate::SimdDescriptor>::new() {
                return $crate::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $call,
                );
            }
        }
        let $d = <$crate::Emu128Descriptor as $crate::SimdDescriptor>::new().unwrap();
        #[cold]
        #[inline(never)]
        fn emu<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        emu(|| $call)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __simd_test_fns {
    (
        $fn_name:ident,
        [$(#[$test_attr:meta])*],
        $avx512:ident,
        $avx:ident,
        $sse42:ident,
        $neon:ident,
        $simd128:ident,
        $emu128:ident
    ) => {
        $crate::__if_avx512! {
            $(#[$test_attr])*
            #[test]
            fn $avx512() {
                if let Some(d) = <$crate::Avx512Descriptor as $crate::SimdDescriptor>::new() {
                    $crate::SimdDescriptor::call_inlined(
                        d,
                        #[inline(always)]
                        |d| $fn_name(d),
                    );
                }
            }
        }
        $crate::__if_avx! {
            $(#[$test_attr])*
            #[test]
            fn $avx() {
                if let Some(d) = <$crate::AvxDescriptor as $crate::SimdDescriptor>::new() {
                    $crate::SimdDescriptor::call_inlined(
                        d,
                        #[inline(always)]
                        |d| $fn_name(d),
                    );
                }
            }
        }
        $crate::__if_sse42! {
            $(#[$test_attr])*
            #[test]
            fn $sse42() {
                if let Some(d) = <$crate::Sse42Descriptor as $crate::SimdDescriptor>::new() {
                    $crate::SimdDescriptor::call_inlined(
                        d,
                        #[inline(always)]
                        |d| $fn_name(d),
                    );
                }
            }
        }
        $crate::__if_neon! {
            $(#[$test_attr])*
            #[test]
            fn $neon() {
                if let Some(d) = <$crate::NeonDescriptor as $crate::SimdDescriptor>::new() {
                    $crate::SimdDescriptor::call_inlined(
                        d,
                        #[inline(always)]
                        |d| $fn_name(d),
                    );
                }
            }
        }
        $crate::__if_simd128! {
            $(#[$test_attr])*
            #[test]
            fn $simd128() {
                if let Some(d) = <$crate::Simd128Descriptor as $crate::SimdDescriptor>::new() {
                    $crate::SimdDescriptor::call_inlined(
                        d,
                        #[inline(always)]
                        |d| $fn_name(d),
                    );
                }
            }
        }
        $(#[$test_attr])*
        #[test]
        fn $emu128() {
            $fn_name(<$crate::Emu128Descriptor as $crate::SimdDescriptor>::new().unwrap());
        }
    };
}

/// Runs a Criterion benchmark callback across every SIMD instruction set supported by the host CPU.
#[macro_export]
macro_rules! bench_all_instruction_sets {
    ($name:ident, $criterion:ident) => {
        $crate::__if_avx512! {
            if let Some(d) = <$crate::Avx512Descriptor as $crate::SimdDescriptor>::new() {
                $crate::SimdDescriptor::call_inlined(
                    d,
                    #[inline(always)]
                    |d| $name(d, $criterion, "avx512"),
                );
            }
        }
        $crate::__if_avx! {
            if let Some(d) = <$crate::AvxDescriptor as $crate::SimdDescriptor>::new() {
                $crate::SimdDescriptor::call_inlined(
                    d,
                    #[inline(always)]
                    |d| $name(d, $criterion, "avx"),
                );
            }
        }
        $crate::__if_sse42! {
            if let Some(d) = <$crate::Sse42Descriptor as $crate::SimdDescriptor>::new() {
                $crate::SimdDescriptor::call_inlined(
                    d,
                    #[inline(always)]
                    |d| $name(d, $criterion, "sse42"),
                );
            }
        }
        $crate::__if_neon! {
            if let Some(d) = <$crate::NeonDescriptor as $crate::SimdDescriptor>::new() {
                $crate::SimdDescriptor::call_inlined(
                    d,
                    #[inline(always)]
                    |d| $name(d, $criterion, "neon"),
                );
            }
        }
        $crate::__if_simd128! {
            if let Some(d) = <$crate::Simd128Descriptor as $crate::SimdDescriptor>::new() {
                $crate::SimdDescriptor::call_inlined(
                    d,
                    #[inline(always)]
                    |d| $name(d, $criterion, "simd128"),
                );
            }
        }
        $name(
            <$crate::Emu128Descriptor as $crate::SimdDescriptor>::new().unwrap(),
            $criterion,
            "emu128",
        );
    };
}

/// Marker trait for 128-bit SIMD descriptors (16-byte vectors).
pub trait SimdDescriptor128: SimdDescriptor {}

/// Zero-sized capability token describing a SIMD instruction set and vector width.
pub trait SimdDescriptor:
    sealed::DescriptorSealed + Sized + Copy + Debug + Send + Sync + 'static
{
    /// Width of a SIMD register in bytes.
    const BYTES: usize;
    /// True when operations run via portable scalar emulation rather than hardware SIMD.
    const IS_EMULATED: bool;

    /// Descriptor of at most 256 bits compatible with `Self`.
    type Descriptor256: SimdDescriptor<Descriptor256 = Self::Descriptor256>;
    /// Descriptor of 128 bits compatible with `Self`.
    type Descriptor128: SimdDescriptor128<Descriptor128 = Self::Descriptor128>;

    /// Constructs a descriptor if the current CPU supports it.
    fn new() -> Option<Self>;

    /// Returns a vector descriptor suitable for operations on vectors of length up to 256 bits.
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256;

    /// Returns a 128-bit vector descriptor compatible with `Self`.
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128;

    /// Calls `f` inside an outlined (`#[inline(never)]`) `#[target_feature]` boundary matching `Self`.
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R;

    /// Calls `f` inside an inlinable (`#[inline]`) `#[target_feature]` boundary matching `Self`.
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R;
}

/// Marker trait for SIMD mask lane widths (`W8`, `W16`, `W32`, `W64`).
pub trait MaskWidth: sealed::MaskWidth {}

/// 8-bit lane mask width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W8;

/// 16-bit lane mask width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W16;

/// 32-bit lane mask width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W32;

/// 64-bit lane mask width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W64;

impl MaskWidth for W8 {}
impl MaskWidth for W16 {}
impl MaskWidth for W32 {}
impl MaskWidth for W64 {}

/// Sealed marker trait for elements supported in [`SimdVec`].
pub trait SimdElement: sealed::Element {}
/// Marker trait for elements supporting comparisons, min/max, and mask selection.
pub trait ComparableElement: SimdElement + sealed::ComparableElement {}
/// Marker trait for elements supporting `+` and `-`.
pub trait ArithElement: SimdElement + sealed::ArithElement {}
/// Marker trait for elements supporting `*`.
pub trait MulElement: ArithElement + sealed::MulElement {}
/// Marker trait for signed elements supporting unary negation, `abs`, and `lt_zero`.
pub trait SignedElement: ComparableElement + ArithElement + sealed::SignedElement {}
/// Marker trait for floating-point elements supporting division, FMA, `sqrt`, rounding, and `copysign`.
pub trait FloatElement: SignedElement + MulElement + sealed::FloatElement {}
/// Marker trait for integer elements supporting bitwise logic, shifts, and bit counting.
pub trait IntElement: ComparableElement + ArithElement + MulElement + sealed::IntElement {}
/// Marker trait for integer elements supporting saturating addition and subtraction.
pub trait SatElement: IntElement + sealed::SatElement {}
/// Marker trait for element pairs supporting widening promotion (`promote_lo` / `promote_hi`).
pub trait PromoteTo<W: SimdElement>: SimdElement + sealed::PromoteElement<W> {}
/// Marker trait for element pairs supporting narrowing demotion (`demote_to`).
pub trait DemoteTo<N: SimdElement>: SimdElement + sealed::DemoteElement<N> {}

macro_rules! impl_public_marker {
    ($tr:ident: $($t:ty),* $(,)?) => {
        $(impl $tr for $t {})*
    };
}

impl_public_marker!(SimdElement: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64);
impl_public_marker!(ComparableElement: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64);
impl_public_marker!(ArithElement: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64);
impl_public_marker!(MulElement: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64);
impl_public_marker!(SignedElement: f32, f64, i8, i16, i32, i64);
impl_public_marker!(FloatElement: f32, f64);
impl_public_marker!(IntElement: i8, u8, i16, u16, i32, u32, i64, u64);
impl_public_marker!(SatElement: i8, u8, i16, u16);

macro_rules! impl_pair_marker {
    ($tr:ident: $(($a:ty, $b:ty)),* $(,)?) => {
        $(impl $tr<$b> for $a {})*
    };
}
impl_pair_marker!(
    PromoteTo: (f32, f64),
    (i8, i16),
    (u8, u16),
    (u8, i16),
    (i16, i32),
    (u16, u32),
    (u16, i32),
    (i32, i64),
    (u32, u64),
    (u32, i64)
);
impl_pair_marker!(
    DemoteTo: (f64, f32),
    (i16, i8),
    (i16, u8),
    (u16, u8),
    (i32, i16),
    (i32, u16),
    (u32, u16),
    (i64, i32),
    (u64, u32)
);

/// Generic SIMD vector of element type `T` for descriptor `D`.
#[derive(Clone, Copy, Debug)]
pub struct SimdVec<T: SimdElement, D: SimdDescriptor>(pub(crate) T::Raw<D>, pub(crate) D);

/// Generic SIMD mask of lane width `W` for descriptor `D`.
#[derive(Clone, Copy, Debug)]
pub struct SimdMask<W: MaskWidth, D: SimdDescriptor>(pub(crate) W::Raw<D>, pub(crate) D);

/// Prepared 8-entry BF16 lookup table for fast approximate lookups.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Table8<D: SimdDescriptor>(pub(crate) D::RawBf16Table8, pub(crate) D);

pub type SimdVecArray<T, D> = <T as sealed::Element>::Array<D>;
pub type F32Vec<D> = SimdVec<f32, D>;
pub type F64Vec<D> = SimdVec<f64, D>;
pub type I8Vec<D> = SimdVec<i8, D>;
pub type U8Vec<D> = SimdVec<u8, D>;
pub type I16Vec<D> = SimdVec<i16, D>;
pub type U16Vec<D> = SimdVec<u16, D>;
pub type I32Vec<D> = SimdVec<i32, D>;
pub type U32Vec<D> = SimdVec<u32, D>;
pub type I64Vec<D> = SimdVec<i64, D>;
pub type U64Vec<D> = SimdVec<u64, D>;
pub type F32VecArray<D> = SimdVecArray<f32, D>;
pub type F64VecArray<D> = SimdVecArray<f64, D>;
pub type I8VecArray<D> = SimdVecArray<i8, D>;
pub type U8VecArray<D> = SimdVecArray<u8, D>;
pub type I16VecArray<D> = SimdVecArray<i16, D>;
pub type U16VecArray<D> = SimdVecArray<u16, D>;
pub type I32VecArray<D> = SimdVecArray<i32, D>;
pub type U32VecArray<D> = SimdVecArray<u32, D>;
pub type I64VecArray<D> = SimdVecArray<i64, D>;
pub type U64VecArray<D> = SimdVecArray<u64, D>;
pub type Mask8<D> = SimdMask<W8, D>;
pub type Mask16<D> = SimdMask<W16, D>;
pub type Mask32<D> = SimdMask<W32, D>;
pub type Mask64<D> = SimdMask<W64, D>;
pub type Mask<D> = SimdMask<W32, D>;

/// Shifts each integer lane left by a compile-time constant amount.
#[macro_export]
macro_rules! shl {
    ($val: expr, $amount: expr) => {
        $val.shl::<{ $amount as u32 }>()
    };
}

/// Shifts each integer lane right (arithmetic for signed, logical for unsigned) by a constant amount.
#[macro_export]
macro_rules! shr {
    ($val: expr, $amount: expr) => {
        $val.shr::<{ $amount as u32 }>()
    };
}

// Generic operator implementations over `SimdVec` and `SimdMask`.

impl<T: ArithElement, D: SimdDescriptor> Add for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        SimdVec(T::add(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: ArithElement, D: SimdDescriptor> AddAssign for SimdVec<T, D> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = T::add(self.1, self.0, rhs.0);
    }
}

impl<T: ArithElement, D: SimdDescriptor> Sub for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        SimdVec(T::sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: ArithElement, D: SimdDescriptor> SubAssign for SimdVec<T, D> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = T::sub(self.1, self.0, rhs.0);
    }
}

impl<T: MulElement, D: SimdDescriptor> Mul for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        SimdVec(T::mul(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: MulElement, D: SimdDescriptor> MulAssign for SimdVec<T, D> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = T::mul(self.1, self.0, rhs.0);
    }
}

impl<T: FloatElement, D: SimdDescriptor> Div for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        SimdVec(T::div(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: FloatElement, D: SimdDescriptor> DivAssign for SimdVec<T, D> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = T::div(self.1, self.0, rhs.0);
    }
}

impl<T: SignedElement, D: SimdDescriptor> Neg for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        SimdVec(T::neg(self.1, self.0), self.1)
    }
}

impl<T: IntElement, D: SimdDescriptor> BitAnd for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        SimdVec(T::bit_and(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: IntElement, D: SimdDescriptor> BitAndAssign for SimdVec<T, D> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = T::bit_and(self.1, self.0, rhs.0);
    }
}

impl<T: IntElement, D: SimdDescriptor> BitOr for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        SimdVec(T::bit_or(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: IntElement, D: SimdDescriptor> BitOrAssign for SimdVec<T, D> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = T::bit_or(self.1, self.0, rhs.0);
    }
}

impl<T: IntElement, D: SimdDescriptor> BitXor for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        SimdVec(T::bit_xor(self.1, self.0, rhs.0), self.1)
    }
}

impl<T: IntElement, D: SimdDescriptor> BitXorAssign for SimdVec<T, D> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = T::bit_xor(self.1, self.0, rhs.0);
    }
}

impl<T: IntElement, D: SimdDescriptor> Not for SimdVec<T, D> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        SimdVec(T::bit_not(self.1, self.0), self.1)
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitAnd for SimdMask<W, D> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        SimdMask(W::mask_and(self.1, self.0, rhs.0), self.1)
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitAndAssign for SimdMask<W, D> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = W::mask_and(self.1, self.0, rhs.0);
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitOr for SimdMask<W, D> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        SimdMask(W::mask_or(self.1, self.0, rhs.0), self.1)
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitOrAssign for SimdMask<W, D> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = W::mask_or(self.1, self.0, rhs.0);
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitXor for SimdMask<W, D> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        SimdMask(W::mask_xor(self.1, self.0, rhs.0), self.1)
    }
}

impl<W: MaskWidth, D: SimdDescriptor> BitXorAssign for SimdMask<W, D> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = W::mask_xor(self.1, self.0, rhs.0);
    }
}

impl<W: MaskWidth, D: SimdDescriptor> Not for SimdMask<W, D> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        SimdMask(W::mask_not(self.1, self.0), self.1)
    }
}

// Generic methods on `SimdMask`.

impl<W: MaskWidth, D: SimdDescriptor> SimdMask<W, D> {
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn from_raw(d: D, raw: W::Raw<D>) -> Self {
        Self(raw, d)
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn raw(self) -> W::Raw<D> {
        self.0
    }

    #[inline(always)]
    pub fn descriptor(self) -> D {
        self.1
    }

    #[inline(always)]
    pub fn if_then_else<T: ComparableElement<Mask = W>>(
        self,
        if_true: SimdVec<T, D>,
        if_false: SimdVec<T, D>,
    ) -> SimdVec<T, D> {
        SimdVec(
            T::mask_select(self.1, self.0, if_true.0, if_false.0),
            self.1,
        )
    }

    #[inline(always)]
    pub fn if_then_else_zero<T: ComparableElement<Mask = W>>(
        self,
        if_true: SimdVec<T, D>,
    ) -> SimdVec<T, D> {
        SimdVec(T::mask_select_zero(self.1, self.0, if_true.0), self.1)
    }

    /// Returns `if_false` where mask is false and zero where mask is true.
    #[inline(always)]
    pub fn if_then_zero_else<T: ComparableElement<Mask = W>>(
        self,
        if_false: SimdVec<T, D>,
    ) -> SimdVec<T, D> {
        SimdVec(T::mask_zero_select(self.1, self.0, if_false.0), self.1)
    }

    #[inline(always)]
    pub fn all(self) -> bool {
        W::mask_all(self.1, self.0)
    }

    #[inline(always)]
    pub fn any(self) -> bool {
        W::mask_any(self.1, self.0)
    }

    /// Computes `!self & rhs`.
    #[inline(always)]
    pub fn andnot(self, rhs: Self) -> Self {
        Self(W::mask_andnot(self.1, self.0, rhs.0), self.1)
    }
}

// Generic methods on `SimdVec`.

impl<T: SimdElement, D: SimdDescriptor> SimdVec<T, D> {
    /// Number of lanes of type `T` in a vector for descriptor `D`.
    pub const LEN: usize = D::BYTES / std::mem::size_of::<T>();

    #[inline(always)]
    pub fn descriptor(self) -> D {
        self.1
    }

    #[inline(always)]
    pub fn zero(d: D) -> Self {
        Self(T::zero(d), d)
    }

    #[inline(always)]
    pub fn splat(d: D, v: T) -> Self {
        Self(T::splat(d, v), d)
    }

    #[inline(always)]
    pub fn extract_lane0(self) -> T {
        T::extract_lane0(self.1, self.0)
    }

    /// Loads `Self::LEN` elements from `mem`. Panics if `mem.len() < Self::LEN`.
    #[inline(always)]
    pub fn load(d: D, mem: &[T]) -> Self {
        Self(T::load(d, mem), d)
    }

    /// Stores `Self::LEN` elements into `mem`. Panics if `mem.len() < Self::LEN`.
    #[inline(always)]
    pub fn store(self, mem: &mut [T]) {
        T::store(self.1, self.0, mem);
    }

    /// Loads `mem.len().min(Self::LEN)` elements from `mem`, zero-padding any remaining lanes.
    #[inline(always)]
    pub fn load_partial(d: D, mem: &[T]) -> Self {
        if mem.len() >= Self::LEN {
            Self::load(d, mem)
        } else {
            let mut buf = T::zero_array::<D>();
            for (dst, &src) in buf.as_mut().iter_mut().zip(mem) {
                *dst = src;
            }
            Self::load_array(d, &buf)
        }
    }

    /// Stores `dest.len().min(Self::LEN)` elements into `dest`.
    #[inline(always)]
    pub fn store_partial(self, dest: &mut [T]) {
        if dest.len() >= Self::LEN {
            self.store(dest);
        } else {
            let mut buf = T::zero_array::<D>();
            self.store_array(&mut buf);
            for (dst, &src) in dest.iter_mut().zip(buf.as_ref()) {
                *dst = src;
            }
        }
    }

    #[inline(always)]
    pub fn load_array(d: D, mem: &SimdVecArray<T, D>) -> Self {
        Self(T::load_array(d, mem), d)
    }

    #[inline(always)]
    pub fn store_array(self, mem: &mut SimdVecArray<T, D>) {
        T::store_array(self.1, self.0, mem);
    }

    #[inline(always)]
    pub fn make_array_slice(slice: &[T]) -> &[SimdVecArray<T, D>] {
        T::slice_as_array::<D>(slice)
    }

    #[inline(always)]
    pub fn make_array_slice_mut(slice: &mut [T]) -> &mut [SimdVecArray<T, D>] {
        T::slice_as_array_mut::<D>(slice)
    }

    #[inline(always)]
    pub fn bitcast<U: SimdElement>(self) -> SimdVec<U, D> {
        SimdVec(
            U::from_raw_bytes(self.1, T::to_raw_bytes(self.1, self.0)),
            self.1,
        )
    }

    #[inline(always)]
    pub fn interleave_lo(self, b: Self) -> Self {
        Self(T::interleave_lo(self.1, self.0, b.0), self.1)
    }

    #[inline(always)]
    pub fn interleave_hi(self, b: Self) -> Self {
        Self(T::interleave_hi(self.1, self.0, b.0), self.1)
    }

    /// Interleaves 128-bit blocks within each 256-bit pair (`[a0, b0, ...]` and `[a1, b1, ...]`).
    /// On 128-bit descriptors (`D::BYTES == 16`), returns `(self, b)` unchanged.
    #[inline(always)]
    pub fn interleave_blocks_128(self, b: Self) -> (Self, Self) {
        if D::BYTES == 16 {
            (self, b)
        } else {
            let d = self.1;
            let (lo, hi) = sealed::BytesBackend::interleave_128_halves(
                d,
                T::to_raw_bytes(d, self.0),
                T::to_raw_bytes(d, b.0),
            );
            (
                Self(T::from_raw_bytes(d, lo), d),
                Self(T::from_raw_bytes(d, hi), d),
            )
        }
    }

    /// Interleaves 256-bit halves within 512-bit vectors (`[a_lo, b_lo]` and `[a_hi, b_hi]`).
    /// On `<= 256`-bit descriptors (`D::BYTES <= 32`), returns `(self, b)` unchanged.
    #[inline(always)]
    pub fn interleave_blocks_256(self, b: Self) -> (Self, Self) {
        if D::BYTES <= 32 {
            (self, b)
        } else {
            let d = self.1;
            let (lo, hi) = sealed::BytesBackend::interleave_256_halves(
                d,
                T::to_raw_bytes(d, self.0),
                T::to_raw_bytes(d, b.0),
            );
            (
                Self(T::from_raw_bytes(d, lo), d),
                Self(T::from_raw_bytes(d, hi), d),
            )
        }
    }

    /// Stores two vectors interleaved: `[a0, b0, a1, b1, ...]`.
    #[inline(always)]
    pub fn store_interleaved_2(a: Self, b: Self, dest: &mut [T]) {
        T::store_interleaved_2(a.1, a.0, b.0, dest);
    }

    /// Stores three vectors interleaved: `[a0, b0, c0, a1, b1, c1, ...]`.
    #[inline(always)]
    pub fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [T]) {
        T::store_interleaved_3(a.1, a.0, b.0, c.0, dest);
    }

    /// Stores four vectors interleaved: `[a0, b0, c0, d0, a1, b1, c1, d1, ...]`.
    #[inline(always)]
    pub fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [T]) {
        T::store_interleaved_4(a.1, a.0, b.0, c.0, d.0, dest);
    }

    /// Loads two deinterleaved vectors from `[a0, b0, a1, b1, ...]`.
    #[inline(always)]
    pub fn load_deinterleaved_2(d: D, src: &[T]) -> (Self, Self) {
        let (a, b) = T::load_deinterleaved_2(d, src);
        (Self(a, d), Self(b, d))
    }

    /// Loads three deinterleaved vectors from `[a0, b0, c0, ...]`.
    #[inline(always)]
    pub fn load_deinterleaved_3(d: D, src: &[T]) -> (Self, Self, Self) {
        let (a, b, c) = T::load_deinterleaved_3(d, src);
        (Self(a, d), Self(b, d), Self(c, d))
    }

    /// Loads four deinterleaved vectors from `[a0, b0, c0, d0, ...]`.
    #[inline(always)]
    pub fn load_deinterleaved_4(d: D, src: &[T]) -> (Self, Self, Self, Self) {
        let (a, b, c, e) = T::load_deinterleaved_4(d, src);
        (Self(a, d), Self(b, d), Self(c, d), Self(e, d))
    }

    /// Widens the lower half of lanes into `W`.
    #[inline(always)]
    pub fn promote_lo<W: SimdElement>(self) -> SimdVec<W, D>
    where
        T: PromoteTo<W>,
    {
        SimdVec(T::promote_lo(self.1, self.0), self.1)
    }

    /// Widens the upper half of lanes into `W`.
    #[inline(always)]
    pub fn promote_hi<W: SimdElement>(self) -> SimdVec<W, D>
    where
        T: PromoteTo<W>,
    {
        SimdVec(T::promote_hi(self.1, self.0), self.1)
    }

    /// Narrows two vectors `(self, hi)` into half-width element type `N`.
    #[inline(always)]
    pub fn demote_to<N: SimdElement>(self, hi: Self) -> SimdVec<N, D>
    where
        T: DemoteTo<N>,
    {
        SimdVec(T::demote_to(self.1, self.0, hi.0), self.1)
    }
}

impl<T: ComparableElement, D: SimdDescriptor> SimdVec<T, D> {
    #[inline(always)]
    pub fn eq(self, other: Self) -> SimdMask<T::Mask, D> {
        SimdMask(
            <T as sealed::ComparableElement>::eq(self.1, self.0, other.0),
            self.1,
        )
    }

    #[inline(always)]
    pub fn gt(self, other: Self) -> SimdMask<T::Mask, D> {
        SimdMask(T::gt(self.1, self.0, other.0), self.1)
    }

    #[inline(always)]
    pub fn lt(self, other: Self) -> SimdMask<T::Mask, D> {
        other.gt(self)
    }

    #[inline(always)]
    pub fn ge(self, other: Self) -> SimdMask<T::Mask, D> {
        self.gt(other) | self.eq(other)
    }

    #[inline(always)]
    pub fn le(self, other: Self) -> SimdMask<T::Mask, D> {
        self.lt(other) | self.eq(other)
    }

    #[inline(always)]
    pub fn eq_zero(self) -> SimdMask<T::Mask, D> {
        SimdMask(T::eq_zero(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(T::min(self.1, self.0, other.0), self.1)
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(T::max(self.1, self.0, other.0), self.1)
    }
}

impl<T: SignedElement, D: SimdDescriptor> SimdVec<T, D> {
    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn neg(self) -> Self {
        Self(T::neg(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(T::abs(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn lt_zero(self) -> SimdMask<T::Mask, D> {
        SimdMask(T::lt_zero(self.1, self.0), self.1)
    }
}

impl<T: FloatElement, D: SimdDescriptor> SimdVec<T, D> {
    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn div(self, other: Self) -> Self {
        Self(T::div(self.1, self.0, other.0), self.1)
    }

    #[inline(always)]
    pub fn mul_add(self, mul: Self, add: Self) -> Self {
        Self(T::mul_add(self.1, self.0, mul.0, add.0), self.1)
    }

    /// Computes `add - self * mul`.
    #[inline(always)]
    pub fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        Self(T::neg_mul_add(self.1, self.0, mul.0, add.0), self.1)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(T::sqrt(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(T::floor(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(T::ceil(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(T::round(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(T::trunc(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn copysign(self, sign: Self) -> Self {
        Self(T::copysign(self.1, self.0, sign.0), self.1)
    }
}

impl<T: IntElement, D: SimdDescriptor> SimdVec<T, D> {
    /// Computes `!self & other`.
    #[inline(always)]
    pub fn andnot(self, other: Self) -> Self {
        Self(T::andnot(self.1, self.0, other.0), self.1)
    }

    #[inline(always)]
    pub fn shl<const AMOUNT: u32>(self) -> Self {
        Self(T::shl::<D, AMOUNT>(self.1, self.0), self.1)
    }

    #[inline(always)]
    pub fn shr<const AMOUNT: u32>(self) -> Self {
        Self(T::shr::<D, AMOUNT>(self.1, self.0), self.1)
    }

    /// Variable per-lane left shift (`0 <= count < T::BITS`).
    #[inline(always)]
    pub fn shl_vec(self, counts: Self) -> Self {
        Self(T::shl_vec(self.1, self.0, counts.0), self.1)
    }

    /// Variable per-lane right shift (`0 <= count < T::BITS`).
    #[inline(always)]
    pub fn shr_vec(self, counts: Self) -> Self {
        Self(T::shr_vec(self.1, self.0, counts.0), self.1)
    }

    /// Population count (`1` bits) per lane.
    #[inline(always)]
    pub fn count_ones(self) -> Self {
        generic_ops::int_count_ones(self)
    }

    /// Leading zero bits per lane.
    #[inline(always)]
    pub fn leading_zeros(self) -> Self {
        generic_ops::int_leading_zeros(self)
    }

    /// Trailing zero bits per lane.
    #[inline(always)]
    pub fn trailing_zeros(self) -> Self {
        generic_ops::int_trailing_zeros(self)
    }
}

impl<T: SatElement, D: SimdDescriptor> SimdVec<T, D> {
    #[inline(always)]
    pub fn sat_add(self, rhs: Self) -> Self {
        Self(T::sat_add(self.1, self.0, rhs.0), self.1)
    }

    #[inline(always)]
    pub fn sat_sub(self, rhs: Self) -> Self {
        Self(T::sat_sub(self.1, self.0, rhs.0), self.1)
    }
}

impl<D: SimdDescriptor> SimdVec<f32, D> {
    #[inline(always)]
    pub fn as_i32(self) -> I32Vec<D> {
        SimdVec(self.1.f32_as_i32(self.0), self.1)
    }

    /// Rounds to nearest integer and stores as `u8`. Unspecified if values overflow `u8`.
    #[inline(always)]
    pub fn round_store_u8(self, dest: &mut [u8]) {
        self.1.f32_round_store_u8(self.0, dest);
    }

    /// Rounds to nearest integer and stores as `u16`. Unspecified if values overflow `u16`.
    #[inline(always)]
    pub fn round_store_u16(self, dest: &mut [u16]) {
        self.1.f32_round_store_u16(self.0, dest);
    }

    #[inline(always)]
    pub fn load_f16_bits(d: D, mem: &[u16]) -> Self {
        Self(d.f32_load_f16_bits(mem), d)
    }

    #[inline(always)]
    pub fn store_f16_bits(self, dest: &mut [u16]) {
        self.1.f32_store_f16_bits(self.0, dest);
    }

    #[inline(always)]
    pub fn store_interleaved_8(
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
        generic_ops::f32_store_interleaved_8(a.1, a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0, dest);
    }

    #[inline(always)]
    pub fn prepare_table_bf16_8(d: D, table: &[f32; 8]) -> Bf16Table8<D> {
        Bf16Table8(d.prepare_table_bf16_8(table), d)
    }

    #[inline(always)]
    pub fn table_lookup_bf16_8(d: D, table: Bf16Table8<D>, indices: I32Vec<D>) -> Self {
        Self(d.table_lookup_bf16_8(table.0, indices.0), d)
    }
}

impl<D: SimdDescriptor> SimdVec<f64, D> {
    #[inline(always)]
    pub fn as_i64(self) -> I64Vec<D> {
        SimdVec(self.1.f64_as_i64(self.0), self.1)
    }
}

impl<D: SimdDescriptor> SimdVec<i64, D> {
    #[inline(always)]
    pub fn as_f64(self) -> F64Vec<D> {
        SimdVec(self.1.i64_as_f64(self.0), self.1)
    }
}

impl<D: SimdDescriptor> SimdVec<i32, D> {
    #[inline(always)]
    pub fn as_f32(self) -> F32Vec<D> {
        SimdVec(self.1.i32_as_f32(self.0), self.1)
    }

    #[inline(always)]
    pub fn load_from_i16(d: D, mem: &[i16]) -> Self {
        Self(d.i32_load_from_i16(mem), d)
    }

    #[inline(always)]
    pub fn load_from_u16(d: D, mem: &[u16]) -> Self {
        Self(d.i32_load_from_u16(mem), d)
    }

    #[inline(always)]
    pub fn store_u16(self, dest: &mut [u16]) {
        self.1.i32_store_u16(self.0, dest);
    }

    #[inline(always)]
    pub fn store_u8(self, dest: &mut [u8]) {
        self.1.i32_store_u8(self.0, dest);
    }

    #[inline(always)]
    pub fn mul_wide_take_high(self, rhs: Self) -> Self {
        Self(self.1.i32_mul_wide_take_high(self.0, rhs.0), self.1)
    }
}

impl<D: SimdDescriptor> SimdVec<i16, D> {
    #[inline(always)]
    pub fn store_u8(self, dest: &mut [u8]) {
        self.1.i16_store_u8(self.0, dest);
    }

    #[inline(always)]
    pub fn mul_wide_take_high(self, rhs: Self) -> Self {
        Self(self.1.i16_mul_wide_take_high(self.0, rhs.0), self.1)
    }
}

impl<D: SimdDescriptor> SimdVec<u8, D> {
    /// Per-128-bit-lane 16-byte lookup; indices `>= 16` produce `0`.
    #[inline(always)]
    pub fn table_lookup_bytes_or_zero(self, indices: Self) -> Self {
        Self(self.1.u8_swizzle_128_or_zero(self.0, indices.0), self.1)
    }
}

#[cfg(test)]
mod tests;
