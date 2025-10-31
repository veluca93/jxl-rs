// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(veluca): consider moving reinterpreting_dct and lowest_frequencies_from_lf here too. That
// requires moving Image types to a new crate first.

// TODO(veluca): we can probably afford an indirect call or two here, so to improve build times we
// could try to figure out how to convince rustc to monomorphize all the relevant generics here.

pub mod dct;
pub mod scales;
#[cfg(test)]
mod tests;
pub mod transform;
pub mod transform_map;
