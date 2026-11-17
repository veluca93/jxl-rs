// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code, unused_unsafe)]
#![allow(clippy::identity_op)]

pub(super) mod arch_ops;
#[cfg(feature = "neon")]
pub(super) mod neon;

#[cfg(feature = "neon")]
use crate::{SimdDescriptor, SimdDescriptor128, TypeFamily};

#[cfg(feature = "neon")]
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(());

#[cfg(feature = "neon")]
impl NeonDescriptor {
    /// # Safety
    /// Caller must guarantee `neon` features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

#[cfg(feature = "neon")]
impl SimdDescriptor128 for NeonDescriptor {}

#[cfg(feature = "neon")]
impl SimdDescriptor for NeonDescriptor {
    const BYTES: usize = 16;
    const IS_EMULATED: bool = false;
    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline]
    fn new() -> Option<Self> {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: target feature was verified via `cfg!` or runtime detection.
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
        #[target_feature(enable = "neon")]
        #[inline(never)]
        fn inner<R>(d: NeonDescriptor, f: impl FnOnce(NeonDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `neon`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "neon")]
        #[inline]
        fn inner<R>(d: NeonDescriptor, f: impl FnOnce(NeonDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `neon`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn prefetch_t0<T: ?Sized>(self, r: &T) {
        // SAFETY: descriptor invariant guarantees `neon`.
        unsafe { arch_ops::prefetch_t0(r) };
    }

    #[inline(always)]
    fn prefetch_et0<T: ?Sized>(self, r: &T) {
        // SAFETY: descriptor invariant guarantees `neon`.
        unsafe { arch_ops::prefetch_et0(r) };
    }

    #[inline(always)]
    fn try_neon<F: TypeFamily<Self> + TypeFamily<NeonDescriptor>, P, IndRet>(
        self,
        p: P,
        d_args: <F as TypeFamily<Self>>::Args,
        f: impl FnOnce(
            NeonDescriptor,
            P,
            <F as TypeFamily<NeonDescriptor>>::Args,
        ) -> <F as TypeFamily<NeonDescriptor>>::Ret<IndRet>,
    ) -> Result<<F as TypeFamily<Self>>::Ret<IndRet>, (P, <F as TypeFamily<Self>>::Args)> {
        self.call_inlined(
            #[inline(always)]
            |d| Ok(f(d, p, d_args)),
        )
    }
}

#[cfg(feature = "neon")]
#[allow(unused_macros)]
macro_rules! neon_call {
    ($d:expr, || -> $ret:ty $body:block) => {{
        #[target_feature(enable = "neon")]
        #[inline]
        fn inner() -> $ret $body
        let _d: $crate::aarch64::NeonDescriptor = $d;
        // SAFETY: descriptor invariant guarantees `neon`.
        #[allow(unsafe_code, unused_unsafe)]
        unsafe { inner() }
    }};
    ($d:expr, |$($arg:ident : $aty:ty),+ $(,)?| $(-> $ret:ty)? $body:block) => {{
        #[target_feature(enable = "neon")]
        #[inline]
        fn inner($($arg: $aty),+) $(-> $ret)? $body
        let _d: $crate::aarch64::NeonDescriptor = $d;
        // SAFETY: descriptor invariant guarantees `neon`.
        #[allow(unsafe_code, unused_unsafe)]
        unsafe { inner($($arg),+) }
    }};
}

#[cfg(feature = "neon")]
#[allow(unused_macros)]
macro_rules! neon_shift {
    ($d:expr, $amt:ident, |$a:ident : $aty:ty| -> $ret:ty $body:block) => {{
        #[target_feature(enable = "neon")]
        #[inline]
        fn inner<const $amt: u32>($a: $aty) -> $ret $body
        let _d: $crate::aarch64::NeonDescriptor = $d;
        // SAFETY: descriptor invariant guarantees `neon`.
        #[allow(unsafe_code, unused_unsafe)]
        unsafe { inner::<$amt>($a) }
    }};
}

#[cfg(feature = "neon")]
#[allow(unused_imports)]
pub(super) use {neon_call, neon_shift};
