// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

pub(super) mod arch_ops;
#[cfg(feature = "avx")]
pub(super) mod avx;
#[cfg(feature = "avx512")]
pub(super) mod avx512;
#[cfg(feature = "sse42")]
pub(super) mod sse42;

#[cfg(any(feature = "sse42", feature = "avx", feature = "avx512"))]
use crate::{SimdDescriptor, SimdDescriptor128, TypeFamily};

#[cfg(feature = "sse42")]
#[derive(Clone, Copy, Debug)]
pub struct Sse42Descriptor(());

#[cfg(feature = "sse42")]
impl Sse42Descriptor {
    /// # Safety
    /// Caller must guarantee `sse4.2` features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

#[cfg(feature = "sse42")]
impl SimdDescriptor128 for Sse42Descriptor {}

#[cfg(feature = "sse42")]
impl SimdDescriptor for Sse42Descriptor {
    const BYTES: usize = 16;
    const IS_EMULATED: bool = false;
    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline]
    fn new() -> Option<Self> {
        if cfg!(target_feature = "sse4.2") || std::arch::is_x86_feature_detected!("sse4.2") {
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
        #[target_feature(enable = "sse4.2")]
        #[inline(never)]
        fn inner<R>(d: Sse42Descriptor, f: impl FnOnce(Sse42Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `sse4.2`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "sse4.2")]
        #[inline]
        fn inner<R>(d: Sse42Descriptor, f: impl FnOnce(Sse42Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `sse4.2`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn try_sse42<F: TypeFamily<Self> + TypeFamily<Sse42Descriptor>, P, IndRet>(
        self,
        p: P,
        d_args: <F as TypeFamily<Self>>::Args,
        f: impl FnOnce(
            Sse42Descriptor,
            P,
            <F as TypeFamily<Sse42Descriptor>>::Args,
        ) -> <F as TypeFamily<Sse42Descriptor>>::Ret<IndRet>,
    ) -> Result<<F as TypeFamily<Self>>::Ret<IndRet>, (P, <F as TypeFamily<Self>>::Args)> {
        self.call_inlined(
            #[inline(always)]
            |d| Ok(f(d, p, d_args)),
        )
    }
}

#[cfg(feature = "avx")]
#[derive(Clone, Copy, Debug)]
pub struct AvxDescriptor(());

#[cfg(feature = "avx")]
impl AvxDescriptor {
    /// # Safety
    /// Caller must guarantee `avx2,fma,f16c` features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }

    #[inline(always)]
    pub fn as_sse42(&self) -> Sse42Descriptor {
        // SAFETY: AVX2 implies SSE4.2.
        unsafe { Sse42Descriptor::new_unchecked() }
    }
}

#[cfg(feature = "avx")]
impl SimdDescriptor for AvxDescriptor {
    const BYTES: usize = 32;
    const IS_EMULATED: bool = false;
    type Descriptor256 = Self;
    type Descriptor128 = Sse42Descriptor;

    #[inline]
    fn new() -> Option<Self> {
        if cfg!(all(
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c"
        )) || (std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            // SAFETY: target features were verified via `cfg!` or runtime detection.
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
        self.as_sse42()
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx2,fma,f16c")]
        #[inline(never)]
        fn inner<R>(d: AvxDescriptor, f: impl FnOnce(AvxDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `avx2,fma,f16c`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx2,fma,f16c")]
        #[inline]
        fn inner<R>(d: AvxDescriptor, f: impl FnOnce(AvxDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `avx2,fma,f16c`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn try_avx<F: TypeFamily<Self> + TypeFamily<AvxDescriptor>, P, IndRet>(
        self,
        p: P,
        d_args: <F as TypeFamily<Self>>::Args,
        f: impl FnOnce(
            AvxDescriptor,
            P,
            <F as TypeFamily<AvxDescriptor>>::Args,
        ) -> <F as TypeFamily<AvxDescriptor>>::Ret<IndRet>,
    ) -> Result<<F as TypeFamily<Self>>::Ret<IndRet>, (P, <F as TypeFamily<Self>>::Args)> {
        self.call_inlined(
            #[inline(always)]
            |d| Ok(f(d, p, d_args)),
        )
    }
}

#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512Descriptor(());

#[cfg(feature = "avx512")]
impl Avx512Descriptor {
    /// # Safety
    /// Caller must guarantee `avx512f,avx512bw,avx2,fma,f16c` features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }

    #[inline(always)]
    pub fn as_avx(&self) -> AvxDescriptor {
        // SAFETY: `Avx512Descriptor` invariant includes `avx2,fma,f16c`.
        unsafe { AvxDescriptor::new_unchecked() }
    }
}

#[cfg(feature = "avx512")]
impl SimdDescriptor for Avx512Descriptor {
    const BYTES: usize = 64;
    const IS_EMULATED: bool = false;
    type Descriptor256 = AvxDescriptor;
    type Descriptor128 = Sse42Descriptor;

    #[inline]
    fn new() -> Option<Self> {
        if cfg!(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c"
        )) || (std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            // SAFETY: target features were verified via `cfg!` or runtime detection.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self.as_avx()
    }

    #[inline(always)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self.as_avx().as_sse42()
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx512f,avx512bw,avx2,fma,f16c")]
        #[inline(never)]
        fn inner<R>(d: Avx512Descriptor, f: impl FnOnce(Avx512Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `avx512f,avx512bw,avx2,fma,f16c`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx512f,avx512bw,avx2,fma,f16c")]
        #[inline]
        fn inner<R>(d: Avx512Descriptor, f: impl FnOnce(Avx512Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: descriptor invariant guarantees `avx512f,avx512bw,avx2,fma,f16c`.
        unsafe { inner(self, f) }
    }

    #[inline(always)]
    fn try_avx512<F: TypeFamily<Self> + TypeFamily<Avx512Descriptor>, P, IndRet>(
        self,
        p: P,
        d_args: <F as TypeFamily<Self>>::Args,
        f: impl FnOnce(
            Avx512Descriptor,
            P,
            <F as TypeFamily<Avx512Descriptor>>::Args,
        ) -> <F as TypeFamily<Avx512Descriptor>>::Ret<IndRet>,
    ) -> Result<<F as TypeFamily<Self>>::Ret<IndRet>, (P, <F as TypeFamily<Self>>::Args)> {
        self.call_inlined(
            #[inline(always)]
            |d| Ok(f(d, p, d_args)),
        )
    }
}

#[cfg(any(feature = "sse42", feature = "avx", feature = "avx512"))]
#[allow(unused_macros)]
macro_rules! impl_x86_call {
    ($descr:ident, $feat:literal, $d:expr, || -> $ret:ty $body:block) => {{
        #[target_feature(enable = $feat)]
        #[inline]
        fn inner() -> $ret $body
        let _d: $crate::x86_64::$descr = $d;
        // SAFETY: descriptor invariant guarantees required target features.
        #[allow(unsafe_code)]
        unsafe { inner() }
    }};
    ($descr:ident, $feat:literal, $d:expr, |$($arg:ident : $aty:ty),+ $(,)?| $(-> $ret:ty)? $body:block) => {{
        #[target_feature(enable = $feat)]
        #[inline]
        fn inner($($arg: $aty),+) $(-> $ret)? $body
        let _d: $crate::x86_64::$descr = $d;
        // SAFETY: descriptor invariant guarantees required target features.
        #[allow(unsafe_code)]
        unsafe { inner($($arg),+) }
    }};
}

#[cfg(feature = "sse42")]
#[allow(unused_macros)]
macro_rules! sse42_call {
    ($d:expr, $($rest:tt)*) => {
        $crate::x86_64::impl_x86_call!(Sse42Descriptor, "sse4.2", $d, $($rest)*)
    };
}

#[cfg(feature = "sse42")]
#[allow(unused_macros)]
macro_rules! sse42_shift {
    ($d:expr, $amt:ident, |$a:ident : $aty:ty| -> $ret:ty $body:block) => {{
        #[target_feature(enable = "sse4.2")]
        #[inline]
        fn inner<const $amt: u32>($a: $aty) -> $ret $body
        let _d: $crate::x86_64::Sse42Descriptor = $d;
        // SAFETY: descriptor invariant guarantees `sse4.2`.
        #[allow(unsafe_code)]
        unsafe { inner::<$amt>($a) }
    }};
}

#[cfg(feature = "sse42")]
#[allow(unused_imports)]
pub(super) use {sse42_call, sse42_shift};

#[cfg(feature = "avx")]
#[allow(unused_macros)]
macro_rules! avx_call {
    ($d:expr, $($rest:tt)*) => {
        $crate::x86_64::impl_x86_call!(AvxDescriptor, "avx2,fma,f16c", $d, $($rest)*)
    };
}

#[cfg(feature = "avx")]
#[allow(unused_macros)]
macro_rules! avx_shift {
    ($d:expr, $amt:ident, |$a:ident : $aty:ty| -> $ret:ty $body:block) => {{
        #[target_feature(enable = "avx2,fma,f16c")]
        #[inline]
        fn inner<const $amt: u32>($a: $aty) -> $ret $body
        let _d: $crate::x86_64::AvxDescriptor = $d;
        // SAFETY: descriptor invariant guarantees `avx2,fma,f16c`.
        #[allow(unsafe_code)]
        unsafe { inner::<$amt>($a) }
    }};
}

#[cfg(feature = "avx")]
#[allow(unused_imports)]
pub(super) use {avx_call, avx_shift};

#[cfg(feature = "avx512")]
#[allow(unused_macros)]
macro_rules! avx512_call {
    ($d:expr, $($rest:tt)*) => {
        $crate::x86_64::impl_x86_call!(Avx512Descriptor, "avx512f,avx512bw,avx2,fma,f16c", $d, $($rest)*)
    };
}

#[cfg(feature = "avx512")]
#[allow(unused_macros)]
macro_rules! avx512_shift {
    ($d:expr, $amt:ident, |$a:ident : $aty:ty| -> $ret:ty $body:block) => {{
        #[target_feature(enable = "avx512f,avx512bw,avx2,fma,f16c")]
        #[inline]
        fn inner<const $amt: u32>($a: $aty) -> $ret $body
        let _d: $crate::x86_64::Avx512Descriptor = $d;
        // SAFETY: descriptor invariant guarantees `avx512f,avx512bw,avx2,fma,f16c`.
        #[allow(unsafe_code)]
        unsafe { inner::<$amt>($a) }
    }};
}

#[cfg(any(feature = "sse42", feature = "avx", feature = "avx512"))]
#[allow(unused_imports)]
pub(crate) use impl_x86_call;
#[cfg(feature = "avx512")]
#[allow(unused_imports)]
pub(super) use {avx512_call, avx512_shift};
