// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    io::IoSliceMut,
    ops::{Deref, Range},
};

use crate::error::Result;

use super::{JxlBitstreamInput, JxlDecoderInner, JxlOutputBuffer, ProcessingResult};

// General implementation strategy:
// - Anything that is not a section is read into a small buffer.
// - As soon as we know section sizes, data is read directly into sections.
// When the start of the populated range in `buf` goes past half of its length,
// the data in the buffer is moved back to the beginning.

pub(super) struct SmallBuffer<const SIZE: usize> {
    buf: [u8; SIZE],
    range: Range<usize>,
}

impl<const SIZE: usize> SmallBuffer<SIZE> {
    pub(super) fn refill<In: JxlBitstreamInput>(&mut self, input: &mut In) -> Result<()> {
        loop {
            if self.range.start >= SIZE / 2 {
                let start = self.range.start;
                let len = self.range.len();
                let (pre, post) = self.buf.split_at_mut(start);
                pre[0..len].copy_from_slice(&post[0..len]);
                self.range.start -= start;
                self.range.end -= start;
            }
            if self.range.len() >= SIZE / 2 {
                break;
            }
            let num = input.read(&mut [IoSliceMut::new(&mut self.buf[self.range.end..])])?;
            self.range.end += num;
            if num == 0 {
                break;
            }
        }
        Ok(())
    }

    pub(super) fn take(&mut self, mut buffers: &mut [IoSliceMut]) -> usize {
        let mut num = 0;
        while self.range.len() > 0 {
            let Some((buf, rest)) = buffers.split_first_mut() else {
                break;
            };
            buffers = rest;
            let len = self.range.len().min(buf.len());
            buf[..len].copy_from_slice(&self.buf[self.range.clone()]);
            self.range.start += len;
            num += len;
        }
        num
    }

    pub(super) fn consume(&mut self, amount: usize) {
        assert!(amount <= self.range.len());
        self.range.start += amount;
    }

    pub(super) fn new() -> Self {
        Self {
            buf: [0; SIZE],
            range: 0..0,
        }
    }
}

impl<const SIZE: usize> Deref for SmallBuffer<SIZE> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.buf[self.range.clone()]
    }
}

impl JxlDecoderInner {
    /// Process more of the input file.
    /// This function will return when reaching the next decoding stage (i.e. finished decoding
    /// file/frame header, or finished decoding a frame).
    pub fn process<'a, In: JxlBitstreamInput, Out: JxlOutputBuffer<'a> + ?Sized>(
        &mut self,
        input: &mut In,
        buffers: Option<&'a mut [&'a mut Out]>,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Skip the next `count` frames.
    pub fn skip_frames(
        &mut self,
        input: &mut impl JxlBitstreamInput,
        count: usize,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Skip the current frame.
    pub fn skip_frame(
        &mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Draws all the pixels we have data for.
    pub fn flush_pixels<'a, Out: JxlOutputBuffer<'a> + ?Sized>(
        &mut self,
        buffers: &'a mut [&'a mut Out],
    ) -> Result<()> {
        todo!()
    }
}
