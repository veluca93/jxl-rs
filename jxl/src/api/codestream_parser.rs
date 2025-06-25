// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;

use crate::{
    error::Result,
    frame::{DecoderState, Frame},
    headers::FileHeader,
};

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, box_parser::BoxParser, process::SmallBuffer,
};

struct SectionBuffer {
    len: usize,
    data: Vec<u8>,
    section_id: usize,
}

// This number should be big enough to guarantee that we can always make progress by reading
// fragments of size at most *half* of it, if not reading a section.
const NON_SECTION_CHUNK_SIZE: usize = 4096;

pub(super) struct CodestreamParser {
    // These fields are populated once image information is available.
    decoder_state: Option<DecoderState>,
    pub(super) basic_info: Option<JxlBasicInfo>,
    pub(super) embedded_color_profile: Option<JxlColorProfile>,
    pub(super) output_color_profile: Option<JxlColorProfile>,
    pub(super) pixel_format: Option<JxlPixelFormat>,
    // These fields are populated when starting to decode a frame, and cleared once
    // the frame is done.
    pub(super) frame: Option<Frame>,
    // Buffers.
    non_section_buf: SmallBuffer<NON_SECTION_CHUNK_SIZE>,
    sections: VecDeque<SectionBuffer>,
}

impl CodestreamParser {
    pub(super) fn new() -> Self {
        Self {
            decoder_state: None,
            basic_info: None,
            embedded_color_profile: None,
            output_color_profile: None,
            pixel_format: None,
            frame: None,
            non_section_buf: SmallBuffer::new(),
            sections: VecDeque::new(),
        }
    }

    pub(super) fn process<'a, In: JxlBitstreamInput, Out: JxlOutputBuffer<'a> + ?Sized>(
        &mut self,
        box_parser: &mut BoxParser,
        input: &mut In,
        decode_options: &JxlDecoderOptions,
        output_buffers: Option<&'a mut [&'a mut Out]>,
    ) -> Result<()> {
        todo!()
    }
}

/*
 *
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    let bit_depth = file_header.image_metadata.bit_depth;
    let input_xsize = file_header.size.xsize();
    let input_ysize = file_header.size.ysize();
    let (output_xsize, output_ysize) = if file_header.image_metadata.orientation.is_transposing() {
        (input_ysize, input_xsize)
    } else {
        (input_xsize, input_ysize)
    };
    info!("Image size: {} x {}", output_xsize, output_ysize);
    let original_icc_bytes = if file_header.image_metadata.color_encoding.want_icc {
        let mut r = IncrementalIccReader::new(&mut br)?;
        r.read_all(&mut br)?;
        let icc = r.finalize()?;
        println!("found {}-byte ICC", icc.len());
        icc
    } else {
        // TODO: handle potential error here?
        file_header
            .image_metadata
            .color_encoding
            .maybe_create_profile()?
            .unwrap()
    };
    let data_icc_bytes = if file_header.image_metadata.xyb_encoded {
        if options.xyb_output_linear {
            let grayscale =
                file_header.image_metadata.color_encoding.color_space == ColorSpace::Gray;
            let mut color_encoding = ColorEncoding::srgb(grayscale);
            color_encoding.tf.transfer_function = TransferFunction::Linear;
            Some(color_encoding.maybe_create_profile()?.unwrap())
        } else {
            // Regular (non-linear) sRGB.
            None
        }
    } else {
        Some(original_icc_bytes.clone())
    };

    br.jump_to_byte_boundary()?;
    let mut image_data: ImageData<f32> = ImageData {
        size: (output_xsize as usize, output_ysize as usize),
        frames: vec![],
    };
    let mut decoder_state = DecoderState::new(file_header);
    decoder_state.xyb_output_linear = options.xyb_output_linear;
    decoder_state.enable_output = options.enable_output;
    decoder_state.render_spotcolors = options.render_spotcolors;
    loop {
        let mut frame = Frame::new(&mut br, decoder_state)?;
        let mut section_readers = frame.sections(&mut br)?;

        info!("read frame with {} sections", section_readers.len());

        frame.decode_lf_global(&mut section_readers[frame.get_section_idx(Section::LfGlobal)])?;

        for group in 0..frame.header().num_lf_groups() {
            frame.decode_lf_group(
                group,
                &mut section_readers[frame.get_section_idx(Section::Lf { group })],
            )?;
        }

        frame.decode_hf_global(&mut section_readers[frame.get_section_idx(Section::HfGlobal)])?;

        frame.prepare_render_pipeline()?;

        for pass in 0..frame.header().passes.num_passes as usize {
            for group in 0..frame.header().num_groups() {
                frame.decode_hf_group(
                    group,
                    pass,
                    &mut section_readers[frame.get_section_idx(Section::Hf { group, pass })],
                )?;
            }
        }

        if let Some(ref mut callback) = options.frame_callback {
            callback(&frame)?;
        }
        let result = frame.finalize()?;
        if let Some(channels) = result.channels {
            image_data.frames.push(ImageFrame {
                size: channels[0].size(),
                channels,
            });
        }
        if let Some(state) = result.decoder_state {
            decoder_state = state;
        } else {
            break;
        }
    }

    Ok(DecodeResult {
        image_data,
        bit_depth,
        original_icc: original_icc_bytes,
        data_icc: data_icc_bytes,
    })
*/

/*
fn maybe_take_section(&mut self) -> Option<Vec<u8>> {
    let SectionParseState::Read(pop) = &mut self.section_state else {
        return None;
    };
    let Some(first) = self.section_buf.pop_front() else {
        return None;
    };
    let first_section_size = first.len as u64;
    if first_section_size <= *pop {
        *pop -= first_section_size;
        assert!(!first.data.is_empty());
        Some(first.data)
    } else {
        self.section_buf.push_front(first);
        None
    }
}
*/
