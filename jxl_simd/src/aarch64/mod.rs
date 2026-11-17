// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

#[cfg(feature = "neon")]
pub(super) mod neon;
