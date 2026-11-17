// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

pub(super) mod arch_ops;
#[cfg(feature = "simd128")]
pub(super) mod simd128;

#[cfg(feature = "simd128")]
use crate::{SimdDescriptor, SimdDescriptor128, TypeFamily};

#[cfg(feature = "simd128")]
#[derive(Clone, Copy, Debug)]
pub struct Simd128Descriptor(());

#[cfg(feature = "simd128")]
impl Simd128Descriptor {
    /// # Safety
    /// Caller must guarantee `simd128` features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

#[cfg(feature = "simd128")]
impl SimdDescriptor128 for Simd128Descriptor {}

#[cfg(feature = "simd128")]
impl SimdDescriptor for Simd128Descriptor {
    const BYTES: usize = 16;
    const IS_EMULATED: bool = false;
    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline]
    fn new() -> Option<Self> {
        if cfg!(target_feature = "simd128") {
            // SAFETY: `simd128` is enabled at compile time.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    #[inline(always)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[inline(never)]
        fn inner<R>(d: Simd128Descriptor, f: impl FnOnce(Simd128Descriptor) -> R) -> R {
            f(d)
        }
        inner(self, f)
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }

    #[inline(always)]
    fn try_simd128<F: TypeFamily<Self> + TypeFamily<Simd128Descriptor>, P, IndRet>(
        self,
        p: P,
        d_args: <F as TypeFamily<Self>>::Args,
        f: impl FnOnce(
            Simd128Descriptor,
            P,
            <F as TypeFamily<Simd128Descriptor>>::Args,
        ) -> <F as TypeFamily<Simd128Descriptor>>::Ret<IndRet>,
    ) -> Result<<F as TypeFamily<Self>>::Ret<IndRet>, (P, <F as TypeFamily<Self>>::Args)> {
        self.call_inlined(
            #[inline(always)]
            |d| Ok(f(d, p, d_args)),
        )
    }
}

#[cfg(feature = "simd128")]
#[allow(unused_macros)]
macro_rules! simd128_call {
    ($d:expr, || -> $ret:ty $body:block) => {{
        #[target_feature(enable = "simd128")]
        #[inline]
        fn inner() -> $ret $body
        let _d: $crate::wasm32::Simd128Descriptor = $d;
        // SAFETY: descriptor invariant guarantees `simd128`.
        #[allow(unsafe_code)]
        unsafe { inner() }
    }};
    ($d:expr, |$($arg:ident : $aty:ty),+ $(,)?| $(-> $ret:ty)? $body:block) => {{
        #[target_feature(enable = "simd128")]
        #[inline]
        fn inner($($arg: $aty),+) $(-> $ret)? $body
        let _d: $crate::wasm32::Simd128Descriptor = $d;
        // SAFETY: descriptor invariant guarantees `simd128`.
        #[allow(unsafe_code)]
        unsafe { inner($($arg),+) }
    }};
}

#[cfg(feature = "simd128")]
#[allow(unused_macros)]
macro_rules! simd128_shift {
    ($d:expr, $amt:ident, |$a:ident : $aty:ty| -> $ret:ty $body:block) => {{
        #[target_feature(enable = "simd128")]
        #[inline]
        fn inner<const $amt: u32>($a: $aty) -> $ret $body
        let _d: $crate::wasm32::Simd128Descriptor = $d;
        // SAFETY: descriptor invariant guarantees `simd128`.
        #[allow(unsafe_code)]
        unsafe { inner::<$amt>($a) }
    }};
}

#[cfg(feature = "simd128")]
#[allow(unused_imports)]
pub(super) use {simd128_call, simd128_shift};
