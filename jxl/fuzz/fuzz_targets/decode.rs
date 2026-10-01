// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{JxlDecoder, JxlDecoderOptions, JxlDecoderStatus};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use libfuzzer_sys::fuzz_target;

// Note: This is adapted from jxl_cli/src/dec/mod.rs
fn fuzz_decode(mut data: &[u8]) -> Result<(), ()> {
    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = Some(1 << 22);
    let mut decoder = JxlDecoder::new(decoder_options);

    match decoder.process(&mut data, None, None) {
        Ok(JxlDecoderStatus::BasicInfo) => {}
        _ => return Err(()),
    }
    match decoder.process(&mut data, None, None) {
        Ok(JxlDecoderStatus::FrameHeader) => {}
        _ => return Err(()),
    }

    let info = decoder.basic_info().unwrap();
    let frame_size = info.size;
    let extra_channels = info.extra_channels.len();
    let samples_per_pixel = decoder
        .current_pixel_format()
        .unwrap()
        .color_type
        .samples_per_pixel();

    let mut outputs =
        vec![Image::<f32>::new((frame_size.0 * samples_per_pixel, frame_size.1)).map_err(|_| ())?];
    for _ in 0..extra_channels {
        outputs.push(Image::<f32>::new(frame_size).map_err(|_| ())?);
    }
    let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
        .iter_mut()
        .map(|x| {
            let rect = Rect {
                size: x.size(),
                origin: (0, 0),
            };
            JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
        })
        .collect();

    while matches!(
        decoder.process(&mut data, Some(&mut output_bufs), None),
        Ok(JxlDecoderStatus::FrameHeader | JxlDecoderStatus::FrameComplete)
    ) {}

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode(data);
});
