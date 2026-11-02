// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;
use std::marker::PhantomData;

use super::internal::RawImageBuffer;
use super::{RawImageRectMut, Rect};
use crate::image::ImageDataType;

/// Callback that transforms a horizontal span of `num_pixels` decoded pixels from `src` into
/// `dst`: `(x, y, src, dst)`, starting at output coordinates `(x, y)`.
///
/// `src` has length `num_pixels * src_bytes_per_pixel` (and is aligned to the sample type of the
/// channel's [`crate::api::JxlDataFormat`]), and `dst` has length `num_pixels * bytes_per_pixel`.
pub type JxlOutputTransform<'a> = dyn Fn(usize, usize, &[u8], &mut [u8]) + Sync + 'a;

#[derive(Clone, Copy)]
pub(crate) struct OutputTransform<'a> {
    pub(crate) bytes_per_pixel: usize,
    pub(crate) func: &'a JxlOutputTransform<'a>,
}

impl Debug for OutputTransform<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputTransform")
            .field("bytes_per_pixel", &self.bytes_per_pixel)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct JxlOutputBuffer<'a> {
    // Safety invariant: `self` has exclusive (write) access to the accessible bytes of `inner`.
    inner: RawImageBuffer,
    transform: Option<OutputTransform<'a>>,
    _ph: PhantomData<&'a mut u8>,
}

impl<'a> JxlOutputBuffer<'a> {
    /// Creates a new JxlOutputBuffer from raw pointers.
    ///
    /// # Safety
    /// - `buf` must be valid for writes for all bytes in the range
    ///   `buf[i*bytes_between_rows..i*bytes_between_rows+bytes_per_row]` for all values of `i`
    ///   from `0` to `num_rows-1`.
    /// - `buf` must be initialized for all the bytes in these ranges.
    /// - The bytes in these ranges must not be accessed as long as the returned `Self` is in scope.
    /// - All the bytes in those ranges (and in between) must be part of the same allocated object.
    pub unsafe fn new_from_ptr(
        buf: *mut u8,
        num_rows: usize,
        bytes_per_row: usize,
        bytes_between_rows: usize,
    ) -> Self {
        JxlOutputBuffer {
            // SAFETY: the safety conditions on RawImageBuffer::new_from_ptr are strictly weaker.
            // We are promised write access to the underlying data, so our own safety invariant is
            // respected.
            inner: unsafe {
                RawImageBuffer::new_from_ptr(
                    buf.cast(),
                    num_rows,
                    bytes_per_row,
                    bytes_between_rows,
                )
            },
            transform: None,
            _ph: PhantomData,
        }
    }

    pub fn from_image_rect_mut(raw: RawImageRectMut<'a>) -> Self {
        Self {
            // Safety note: since `raw` has exclusive access to the data, we are just transferring
            // this access.
            inner: raw.data,
            transform: None,
            _ph: PhantomData,
        }
    }

    pub fn new_with_stride(
        buf: &'a mut [u8],
        num_rows: usize,
        bytes_per_row: usize,
        byte_stride: usize,
    ) -> Self {
        assert_ne!(num_rows, 0);
        assert!(
            buf.len()
                >= byte_stride
                    .checked_mul(num_rows - 1)
                    .unwrap()
                    .checked_add(bytes_per_row)
                    .unwrap()
        );
        // SAFETY: The assert above guarantees that `buf` has enough space to satisfy the first
        // safety requirement, and the rest follow from borrowing from a &mut [].
        unsafe { Self::new_from_ptr(buf.as_mut_ptr(), num_rows, bytes_per_row, byte_stride) }
    }

    pub fn new(buf: &'a mut [u8], num_rows: usize, bytes_per_row: usize) -> Self {
        Self::new_with_stride(buf, num_rows, bytes_per_row, bytes_per_row)
    }

    /// Attaches a pixel transform to this [`JxlOutputBuffer`], specifying the destination
    /// `bytes_per_pixel` in this buffer (which must be non-zero).
    pub fn with_transform(
        mut self,
        bytes_per_pixel: usize,
        transform: &'a JxlOutputTransform<'a>,
    ) -> Self {
        assert_ne!(bytes_per_pixel, 0, "bytes_per_pixel must be non-zero");
        self.transform = Some(OutputTransform {
            bytes_per_pixel,
            func: transform,
        });
        self
    }

    pub(crate) fn bytes_per_pixel(&self) -> Option<usize> {
        self.transform.map(|t| t.bytes_per_pixel)
    }

    pub(crate) fn transform(&self) -> Option<OutputTransform<'a>> {
        self.transform
    }

    pub(crate) fn reborrow(lender: &'a mut JxlOutputBuffer<'_>) -> JxlOutputBuffer<'a> {
        // Safety note: this is effectively equivalent to a reborrow.
        Self {
            _ph: PhantomData,
            ..*lender
        }
    }

    pub(crate) fn typed_row_mut<T: ImageDataType>(&mut self, row: usize) -> &mut [T] {
        T::cast_slice_mut(self.row_mut(row))
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [u8] {
        // SAFETY: we have write access to the data due to safety invariant.
        unsafe { self.inner.row_mut(row) }
    }

    pub fn byte_size(&self) -> (usize, usize) {
        self.inner.byte_size()
    }

    /// # Safety
    /// The caller must guarantee that there are no outstanding lends
    /// of the provided `rect`.
    pub(crate) unsafe fn rect(&self, rect: Rect) -> JxlOutputBuffer<'_> {
        // Safety note: the return value borrows from `self`, so we are lending our memory to the
        // returned JxlOutputBuffer. The caller guarantees that the region identified by
        // `rect` does not overlap with other regions with outstanding borrows, so we have
        // exclusive access to that region.
        Self {
            inner: self.inner.rect(rect),
            transform: self.transform,
            _ph: PhantomData,
        }
    }
}
