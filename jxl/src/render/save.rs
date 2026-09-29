// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::{JxlColorType, JxlDataFormat, JxlOutputBuffer};
use crate::error::{Error, Result};
use crate::headers::Orientation;
use crate::headers::bit_depth::BitDepth;
use crate::image::DataTypeTag;
use crate::util::{SmallVec, StackOnly};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineChannelType {
    F32,
    I16(BitDepth),
    I32(BitDepth),
}

impl PipelineChannelType {
    pub fn data_type(&self) -> DataTypeTag {
        match self {
            Self::F32 => DataTypeTag::F32,
            Self::I16(_) => DataTypeTag::I16,
            Self::I32(_) => DataTypeTag::I32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelConversion {
    None,
    F32ToU8 {
        bit_depth: u8,
        dither_channel: usize,
    },
    I16ToU8 {
        multiplier: i32,
        max: i32,
    },
    I32ToU8 {
        multiplier: i32,
        max: i32,
    },
    F32ToU16 {
        bit_depth: u8,
    },
    F32ToF16,
}

impl ChannelConversion {
    pub fn can_save_integer(output_format: JxlDataFormat, src_bit_depth: BitDepth) -> bool {
        match output_format {
            JxlDataFormat::U8 { bit_depth: b } => {
                b != 0
                    && !src_bit_depth.floating_point_sample()
                    && src_bit_depth.bits_per_sample() != 0
                    && (b as u32).is_multiple_of(src_bit_depth.bits_per_sample())
            }
            _ => false,
        }
    }

    pub fn new(
        input_type: PipelineChannelType,
        output_format: JxlDataFormat,
        dither_channel: usize,
    ) -> Self {
        match input_type {
            PipelineChannelType::I16(src_bit_depth) | PipelineChannelType::I32(src_bit_depth)
                if let JxlDataFormat::U8 { bit_depth: b } = output_format
                    && Self::can_save_integer(output_format, src_bit_depth) =>
            {
                let bit_depth = src_bit_depth.bits_per_sample() as u8;
                let mult = (((1u32 << b) - 1) / ((1u32 << bit_depth) - 1)) as i32;
                let max = ((1u32 << b) - 1) as i32;
                if matches!(input_type, PipelineChannelType::I16(_)) {
                    ChannelConversion::I16ToU8 {
                        multiplier: mult,
                        max,
                    }
                } else {
                    ChannelConversion::I32ToU8 {
                        multiplier: mult,
                        max,
                    }
                }
            }
            _ => match output_format {
                JxlDataFormat::U8 { bit_depth } => ChannelConversion::F32ToU8 {
                    bit_depth,
                    dither_channel,
                },
                JxlDataFormat::U16 { bit_depth, .. } => ChannelConversion::F32ToU16 { bit_depth },
                JxlDataFormat::F16 { .. } => ChannelConversion::F32ToF16,
                JxlDataFormat::F32 { .. } => ChannelConversion::None,
            },
        }
    }

    pub fn input_type(&self, default_type: DataTypeTag) -> DataTypeTag {
        match self {
            Self::None => default_type,
            Self::F32ToU8 { .. } | Self::F32ToU16 { .. } | Self::F32ToF16 => DataTypeTag::F32,
            Self::I16ToU8 { .. } => DataTypeTag::I16,
            Self::I32ToU8 { .. } => DataTypeTag::I32,
        }
    }
}

#[derive(Debug)]
pub struct SaveStage {
    pub(super) channels: Vec<usize>,
    pub(super) orientation: Orientation,
    pub(super) output_buffer_index: usize,
    pub(super) color_type: JxlColorType,
    pub(super) data_format: JxlDataFormat,
    /// When true, fill alpha channel with opaque (1.0) values.
    /// Used when RGBA output is requested but image has no alpha channel.
    pub(super) fill_opaque_alpha: bool,
    pub(super) conversions: SmallVec<ChannelConversion, 4, StackOnly>,
}

impl SaveStage {
    pub fn new(
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        mut color_type: JxlColorType,
        data_format: JxlDataFormat,
        channel_types: &[PipelineChannelType],
    ) -> SaveStage {
        let fill_opaque_alpha =
            color_type.has_alpha() && channels.len() + 1 == color_type.samples_per_pixel();
        let expected_channels = match color_type {
            JxlColorType::Grayscale => 1,
            JxlColorType::GrayscaleAlpha => {
                if fill_opaque_alpha {
                    1
                } else {
                    2
                }
            }
            JxlColorType::Rgb | JxlColorType::Bgr => 3,
            JxlColorType::Rgba | JxlColorType::Bgra => {
                if fill_opaque_alpha {
                    3
                } else {
                    4
                }
            }
            JxlColorType::Cmyk => 4,
        };
        assert_eq!(
            channels.len(),
            expected_channels,
            "SaveStage for {color_type:?} expected {expected_channels} channels, got {}",
            channels.len()
        );
        assert_eq!(
            channels.len(),
            channel_types.len(),
            "SaveStage expected {} channel types, got {}",
            channels.len(),
            channel_types.len()
        );
        let mut conversions = SmallVec::new();
        for (i, &c) in channels.iter().enumerate() {
            conversions.push(ChannelConversion::new(channel_types[i], data_format, c));
        }
        let mut channels = channels.to_vec();
        if color_type == JxlColorType::Bgr {
            color_type = JxlColorType::Rgb;
            channels.swap(0, 2);
            conversions.swap(0, 2);
        }
        if color_type == JxlColorType::Bgra {
            color_type = JxlColorType::Rgba;
            channels.swap(0, 2);
            conversions.swap(0, 2);
        }
        Self {
            channels,
            orientation,
            output_buffer_index,
            color_type,
            data_format,
            fill_opaque_alpha,
            conversions,
        }
    }

    /// Returns the number of output channels (including filled alpha if applicable)
    pub fn output_channels(&self) -> usize {
        self.color_type.samples_per_pixel()
    }

    pub fn uses_channel(&self, c: usize) -> bool {
        self.channels.contains(&c)
    }

    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    pub fn channel_input_type(&self, c: usize) -> DataTypeTag {
        let idx = self.channels.iter().position(|&chan| chan == c);
        match idx {
            Some(i) => self
                .conversions
                .get(i)
                .map(|conv| conv.input_type(self.data_format.data_type()))
                .unwrap_or(self.data_format.data_type()),
            None => self.data_format.data_type(),
        }
    }

    pub fn check_buffer_size(
        &self,
        size: (usize, usize),
        buffer: Option<&JxlOutputBuffer>,
    ) -> Result<()> {
        let Some(buf) = buffer else {
            return Ok(());
        };
        let osize = self.orientation.map_size(size);

        let expected_w = self.output_channels() * self.data_format.bytes_per_sample() * osize.0;

        if buf.byte_size() != (expected_w, osize.1) {
            return Err(Error::InvalidOutputBufferSize(
                buf.byte_size().0,
                buf.byte_size().1,
                osize.0,
                osize.1,
                self.color_type,
                self.data_format,
            ));
        }
        Ok(())
    }
}

impl std::fmt::Display for SaveStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "save channels {:?} (type {:?} {:?})",
            self.channels, self.color_type, self.data_format
        )
    }
}
