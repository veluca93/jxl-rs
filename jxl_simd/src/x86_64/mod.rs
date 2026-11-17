// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

#[cfg(feature = "avx")]
pub(super) mod avx;
#[cfg(feature = "avx512")]
pub(super) mod avx512;
#[cfg(feature = "sse42")]
pub(super) mod sse42;
