// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::marker::PhantomData;

use states::*;

use super::{
    JxlAuxBox, JxlAuxBoxType, JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner,
    JxlDecoderOptions, JxlDecoderStatus, JxlFrameHeader, JxlOutputBuffer, JxlParallelRunner,
    JxlPixelFormat, ProcessingResult, VisibleFrameInfo, VisibleFrameSeekTarget,
};
use crate::error::Result;

pub mod states {
    pub trait JxlState {}
    pub struct Initialized;
    pub struct WithImageInfo;
    pub struct WithFrameInfo;
    pub struct InTrailingBox;
    impl JxlState for Initialized {}
    impl JxlState for WithImageInfo {}
    impl JxlState for WithFrameInfo {}
    impl JxlState for InTrailingBox {}
}

/// High level API using the typestate pattern to forbid invalid usage.
pub struct JxlDecoder<State: JxlState> {
    inner: Box<JxlDecoderInner>,
    _state: PhantomData<State>,
}

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: Box<JxlDecoderInner>) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Returns visible frame info entries collected so far.
    ///
    /// When `JxlDecoderOptions::scan_frames_only` is enabled this is the
    /// primary output of decoding.
    pub fn scanned_frames(&self) -> &[VisibleFrameInfo] {
        self.inner.scanned_frames()
    }

    pub fn aux_boxes(&self, box_type: JxlAuxBoxType) -> &[JxlAuxBox] {
        self.inner.aux_boxes(box_type)
    }
}

impl JxlDecoder<Initialized> {
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self::wrap_inner(Box::new(JxlDecoderInner::new(options)))
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        match self.inner.process(input, None, parallel_runner)? {
            JxlDecoderStatus::BasicInfo => Ok(ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                })
            }
            status => panic!("unexpected status in Initialized: {status:?}"),
        }
    }
}

impl JxlDecoder<WithImageInfo> {
    /// Obtains the image's basic information.
    pub fn basic_info(&self) -> &JxlBasicInfo {
        self.inner.basic_info().unwrap()
    }

    /// Retrieves the file's color profile.
    pub fn embedded_color_profile(&self) -> &JxlColorProfile {
        self.inner.embedded_color_profile().unwrap()
    }

    /// Retrieves the current output color profile.
    pub fn output_color_profile(&self) -> &JxlColorProfile {
        self.inner.output_color_profile().unwrap()
    }

    /// Retrieves the current pixel format for output buffers.
    pub fn current_pixel_format(&self) -> &JxlPixelFormat {
        self.inner.current_pixel_format().unwrap()
    }

    /// Specifies pixel format for output buffers.
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) -> Result<()> {
        self.inner.set_pixel_format(pixel_format)
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        match self.inner.process(input, None, parallel_runner)? {
            JxlDecoderStatus::FrameHeader => Ok(ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                })
            }
            status => panic!("unexpected status in WithImageInfo: {status:?}"),
        }
    }

    /// Feeds additional trailing data, potentially parsing more trailing auxiliary boxes.
    pub fn process_trailing_data(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<InTrailingBox>, Self>> {
        match self.inner.process(input, None, None)? {
            JxlDecoderStatus::Complete => Ok(ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                })
            }
            status => panic!("unexpected status in process_trailing_data: {status:?}"),
        }
    }

    /// Draws all the pixels we have data for. This is useful for i.e. previewing LF frames.
    pub fn flush_pixels(
        &mut self,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<bool> {
        self.inner.flush_pixels(buffers, parallel_runner)
    }

    pub fn has_more_frames(&self) -> bool {
        self.inner.has_more_frames()
    }

    pub fn file_length(&self) -> Option<u64> {
        self.inner.file_length()
    }

    pub fn start_new_frame(&mut self, seek_target: VisibleFrameSeekTarget) {
        self.inner.start_new_frame(seek_target).unwrap();
    }
}

impl JxlDecoder<WithFrameInfo> {
    pub fn skip_frame(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        match self.inner.process(input, None, None)? {
            JxlDecoderStatus::FrameComplete => Ok(ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                })
            }
            status => panic!("unexpected status in WithFrameInfo skip_frame: {status:?}"),
        }
    }

    pub fn frame_header(&self) -> JxlFrameHeader {
        self.inner.frame_header().unwrap().clone()
    }

    pub fn flush_pixels(
        &mut self,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<bool> {
        self.inner.flush_pixels(buffers, parallel_runner)
    }

    pub fn process<In: JxlBitstreamInput>(
        mut self,
        input: &mut In,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        match self.inner.process(input, Some(buffers), parallel_runner)? {
            JxlDecoderStatus::FrameComplete => Ok(ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                })
            }
            status => panic!("unexpected status in WithFrameInfo: {status:?}"),
        }
    }
}

impl JxlDecoder<InTrailingBox> {
    pub fn trailing_box(&self) -> Option<&JxlAuxBox> {
        None
    }

    pub fn process_trailing_data(
        &mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<(), ()>> {
        match self.inner.process(input, None, None)? {
            JxlDecoderStatus::Complete => Ok(ProcessingResult::Complete { result: () }),
            JxlDecoderStatus::NeedsMoreInput { size_hint } => {
                Ok(ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: (),
                })
            }
            status => panic!("unexpected status in InTrailingBox: {status:?}"),
        }
    }

    pub fn start_new_frame(
        mut self,
        seek_target: VisibleFrameSeekTarget,
    ) -> JxlDecoder<WithImageInfo> {
        self.inner.start_new_frame(seek_target).unwrap();
        JxlDecoder::wrap_inner(self.inner)
    }
}
