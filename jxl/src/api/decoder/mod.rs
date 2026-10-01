// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use box_parser::BoxParser;
use codestream_parser::CodestreamParser;

use super::{JxlBasicInfo, JxlColorProfile, JxlDecoderOptions, JxlPixelFormat};
use crate::api::JxlFrameHeader;
use crate::error::{Error, Result};

mod box_parser;
mod codestream_parser;
mod process;

pub use box_parser::{BoxParserCheckpoint, JxlAuxBox, JxlAuxBoxType};

/// Event-driven status yielded by `process()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JxlDecoderStatus {
    /// Basic info (dimensions, bit depth, animation parameters, extra channels)
    /// and color profiles are decoded. Pixel format can now be configured.
    BasicInfo,

    /// A visible frame header has been decoded. Metadata (dimensions, duration,
    /// name) is accessible. Output buffers can be provided on the next
    /// call to decode pixels, or omitted (`None`) to skip pixel rendering.
    FrameHeader,

    /// A frame has been completely decoded into the provided buffers (or skipped).
    FrameComplete,

    /// More input bytes are needed to make progress.
    NeedsMoreInput { size_hint: usize },

    /// The entire file (including all frames and trailing metadata) has finished.
    Complete,
}

/// Information about a single visible frame discovered while decoding.
#[derive(Debug, Clone)]
pub struct VisibleFrameInfo {
    /// Zero-based index among visible frames.
    pub index: usize,
    /// Duration in milliseconds (0 for still images or the last frame).
    pub duration_ms: f64,
    /// Duration in raw ticks from the animation header.
    pub duration_ticks: u32,
    /// Byte offset of this frame's header in the input file.
    pub file_offset: u64,
    /// Whether this is the last frame in the codestream.
    pub is_last: bool,
    /// Whether this frame is a seek-keyframe for visible-frame playback.
    ///
    /// This is equivalent to `seek_target.visible_frames_to_skip == 0`.
    pub is_keyframe: bool,
    /// Precomputed seek inputs for this visible frame.
    pub seek_target: VisibleFrameSeekTarget,
    /// Frame name, if any.
    pub name: String,
}

/// Computed seek inputs for a target visible frame.
#[derive(Debug, Clone, Copy)]
pub struct VisibleFrameSeekTarget {
    /// File byte offset to start feeding input from.
    pub decode_start_file_offset: u64,
    /// State of the box parser at the file offset we want to seek to.
    /// Pass this to [`JxlDecoder::start_new_frame`].
    pub box_parser_checkpoint: BoxParserCheckpoint,
    /// Number of visible frames to skip after seek-start before decoding the
    /// requested target frame.
    pub visible_frames_to_skip: usize,
}

/// Event-driven JPEG XL decoder.
pub struct JxlDecoder {
    options: JxlDecoderOptions,
    box_parser: BoxParser,
    codestream_parser: CodestreamParser,
}

impl JxlDecoder {
    /// Creates a new decoder with the given options.
    pub fn new(options: JxlDecoderOptions) -> Self {
        let box_parser = BoxParser::with_aux_boxes(options.request_aux_boxes.iter().copied());
        JxlDecoder {
            options,
            box_parser,
            codestream_parser: CodestreamParser::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn file_header(&self) -> Option<&crate::headers::FileHeader> {
        if self.codestream_parser.image_info.is_complete() {
            Some(self.codestream_parser.image_info.file_header())
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_frame_header(&self) -> Option<&crate::headers::frame_header::FrameHeader> {
        self.codestream_parser.frame_info.current_frame_header()
    }

    #[cfg(test)]
    pub(crate) fn toc(&self) -> Option<&crate::headers::toc::Toc> {
        self.codestream_parser.frame_info.current_toc()
    }

    /// Obtains the image's basic information, if available.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        if self.codestream_parser.image_info.is_complete() {
            Some(self.codestream_parser.image_info.basic_info())
        } else {
            None
        }
    }

    /// Retrieves the file's color profile, if available.
    pub fn embedded_color_profile(&self) -> Option<&JxlColorProfile> {
        if self.codestream_parser.image_info.is_complete() {
            Some(self.codestream_parser.image_info.embedded_color_profile())
        } else {
            None
        }
    }

    /// Retrieves the current output color profile, if available.
    pub fn output_color_profile(&self) -> Option<&JxlColorProfile> {
        self.codestream_parser.output_color_profile.as_ref()
    }

    /// Retrieves the current pixel format for output buffers, if available.
    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        self.codestream_parser.pixel_format.as_ref()
    }

    /// Specifies pixel format for output buffers.
    ///
    /// Setting this may also change output color profile in some cases, if the profile was not set
    /// manually before.
    ///
    /// The pixel format can only be changed before the first frame header is
    /// decoded, i.e. right after basic info becomes available; afterwards
    /// this returns an error.
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) -> Result<()> {
        // TODO(veluca): return an error if we are asking for both planar and
        // interleaved-in-color alpha.
        // Frame render pipelines are built for the pixel format that is
        // current when the frame's TOC is parsed, so the format can only be
        // changed before the first frame header is decoded, i.e. right after
        // basic info becomes available.
        if !self.codestream_parser.can_change_pixel_format() {
            return Err(Error::APIUsageError(
                "cannot change pixel format before BasicInfo or after FrameHeader",
            ));
        }
        if pixel_format.extra_channel_format.len()
            != self
                .codestream_parser
                .image_info
                .basic_info()
                .extra_channels
                .len()
        {
            return Err(Error::APIUsageError(
                "extra_channel_format length does not match extra_channels count",
            ));
        }
        self.codestream_parser.pixel_format = Some(pixel_format);
        self.codestream_parser.update_default_output_options();
        Ok(())
    }

    pub fn frame_header(&self) -> Option<JxlFrameHeader> {
        self.codestream_parser.frame_header()
    }

    /// Returns visible frame info entries collected so far.
    ///
    /// When `JxlDecoderOptions::scan_frames_only` is enabled this is the
    /// primary output of decoding.
    pub fn scanned_frames(&self) -> &[VisibleFrameInfo] {
        self.codestream_parser.scanned_frames()
    }

    /// Resets frame-level state to prepare for decoding a new frame.
    ///
    /// After seeking the first time, scanned frame information will no longer
    /// be updated. If you seek before having completed decoding once, the scanned
    /// frames might be incomplete.
    ///
    /// After calling this, provide raw file input starting from
    /// `seek_target.decode_start_file_offset`.
    pub fn start_new_frame(&mut self, seek_target: VisibleFrameSeekTarget) -> Result<()> {
        if !self.codestream_parser.image_info.is_complete() {
            return Err(Error::APIUsageError(
                "cannot seek before basic info is available",
            ));
        }
        self.box_parser
            .reset_to_checkpoint(seek_target.box_parser_checkpoint);
        self.codestream_parser.start_new_frame(
            seek_target.visible_frames_to_skip,
            seek_target.box_parser_checkpoint.consumed_codestream,
        );
        Ok(())
    }

    /// Returns the total length of the JPEG XL file, once decoding is finished.
    /// This is needed because the decoder might over-consume bytes from the
    /// provided input stream in some cases.
    pub fn file_length(&self) -> Option<u64> {
        self.codestream_parser.file_length
    }

    pub fn aux_boxes(&self, box_type: JxlAuxBoxType) -> &[JxlAuxBox] {
        self.box_parser.aux_boxes(box_type)
    }
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new(JxlDecoderOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::JxlDecoder;
    use crate::api::{JxlAuxBoxType, JxlDecoderOptions, JxlDecoderStatus};

    #[test]
    fn basic_info_not_visible_before_embedded_profile() {
        let data = std::fs::read("resources/test/conformance_test_images/cmyk_layers.jxl").unwrap();
        let mut decoder = JxlDecoder::default();

        for chunk in data.chunks(64) {
            let mut input = chunk;
            let _ = decoder.process(&mut input, None, None);

            if decoder.embedded_color_profile().is_none() {
                assert!(decoder.basic_info().is_none());
            }

            if decoder.basic_info().is_some() {
                assert!(decoder.embedded_color_profile().is_some());
                return;
            }
        }

        panic!("failed to reach image-info state while parsing cmyk_layers.jxl");
    }

    #[test]
    fn incomplete_ooo_jxlp() {
        let data = include_bytes!("../../../tests/testdata/incomplete_ooo_jxlp.jxl");

        let mut decoder = JxlDecoder::default();
        let mut input = data.as_slice();
        let result = decoder.process(&mut input, None, None);
        assert!(
            matches!(result, Err(crate::error::Error::UnexpectedCodestreamBoxEnd)),
            "{result:?}"
        );
    }

    #[test]
    fn aux_boxes() {
        let data = [
            (&include_bytes!("../../../tests/testdata/exif.jxl")[..], 170),
            (
                &include_bytes!("../../../tests/testdata/exif_brob.jxl")[..],
                120,
            ),
            (
                &include_bytes!("../../../tests/testdata/exif_trailing_finite.jxl")[..],
                170,
            ),
            (
                &include_bytes!("../../../tests/testdata/exif_brob_trailing_finite.jxl")[..],
                120,
            ),
            (
                &include_bytes!("../../../tests/testdata/exif_trailing_infinite.jxl")[..],
                170,
            ),
            (
                &include_bytes!("../../../tests/testdata/exif_brob_trailing_infinite.jxl")[..],
                120,
            ),
        ];

        for (mut buf, expected_size) in data {
            let total_len = buf.len() as u64;
            let options = JxlDecoderOptions {
                request_aux_boxes: vec![JxlAuxBoxType::EXIF],
                scan_frames_only: true,
                ..Default::default()
            };
            let mut decoder = JxlDecoder::new(options);

            while decoder.process(&mut buf, None, None).unwrap() != JxlDecoderStatus::Complete {
                assert!(decoder.file_length().is_none());
            }

            let exif = &decoder.aux_boxes(JxlAuxBoxType::EXIF)[0];
            assert_eq!(exif.raw_data().len(), expected_size);
            #[cfg(feature = "brotli")]
            assert_eq!(exif.data().unwrap().len(), 170);
            assert_eq!(decoder.file_length(), Some(total_len));
        }
    }

    #[test]
    fn aux_box_seek() {
        let data = include_bytes!("../../../tests/testdata/multiple_aux.jxl");

        let ty_foo = JxlAuxBoxType(*b"foo ");
        let ty_bar = JxlAuxBoxType(*b"bar ");

        let options = JxlDecoderOptions {
            request_aux_boxes: vec![ty_bar],
            scan_frames_only: true,
            ..Default::default()
        };
        let mut decoder = JxlDecoder::new(options);

        let mut buf = &data[..];
        while decoder.process(&mut buf, None, None).unwrap() != JxlDecoderStatus::Complete {}
        assert!(decoder.aux_boxes(ty_foo).is_empty());
        assert_eq!(decoder.aux_boxes(ty_bar).len(), 2);
        assert_eq!(decoder.file_length(), Some(data.len() as u64));

        let seek_target = decoder.scanned_frames()[0].seek_target;
        decoder.start_new_frame(seek_target).unwrap();
        assert!(decoder.file_length().is_none());

        let mut buf = &data[(seek_target.decode_start_file_offset as usize)..];
        while decoder.process(&mut buf, None, None).unwrap() != JxlDecoderStatus::Complete {}
        assert!(decoder.aux_boxes(ty_foo).is_empty());
        assert_eq!(decoder.aux_boxes(ty_bar).len(), 2);
        assert_eq!(decoder.file_length(), Some(data.len() as u64));
    }
}
