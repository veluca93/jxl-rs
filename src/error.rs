// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Read out of bounds")]
    OutOfBounds,
    #[error("Non-zero padding bits")]
    NonZeroPadding,
    #[error("Invalid signature {0:02x}{1:02x}, expected ff0a")]
    InvalidSignature(u8, u8),
}