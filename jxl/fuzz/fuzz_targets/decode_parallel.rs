// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{
    JxlDecoderInner as JxlDecoder, JxlDecoderOptions, JxlDecoderStatus, JxlParallelRunner,
};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use libfuzzer_sys::fuzz_target;

struct SimpleParallelRunner {
    max_threads: usize,
}

impl JxlParallelRunner for SimpleParallelRunner {
    fn run(
        &mut self,
        num: usize,
        fun: &jxl::api::JxlParallelRunnerFun<'_>,
    ) -> Result<(), jxl::error::Error> {
        if num <= 1 || self.max_threads <= 1 {
            for i in 0..num {
                fun(i)?;
            }
            return Ok(());
        }
        let num_threads = self.max_threads.min(num);
        let next_task = std::sync::atomic::AtomicUsize::new(0);
        let error = std::sync::Mutex::new(None);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for _ in 0..num_threads {
                handles.push(s.spawn(|| {
                    loop {
                        if error.lock().unwrap().is_some() {
                            break;
                        }
                        let task = next_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if task >= num {
                            break;
                        }
                        if let Err(e) = fun(task) {
                            let mut err = error.lock().unwrap();
                            if err.is_none() {
                                *err = Some(e);
                            }
                            break;
                        }
                    }
                }));
            }
            for handle in handles {
                if let Err(e) = handle.join() {
                    std::panic::resume_unwind(e);
                }
            }
        });

        if let Some(err) = error.into_inner().unwrap() {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn num_threads(&self) -> usize {
        self.max_threads
    }
}

fn fuzz_decode_parallel(mut data: &[u8]) -> Result<(), ()> {
    let mut runner = SimpleParallelRunner { max_threads: 2 };

    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = Some(1 << 22);
    let mut decoder = JxlDecoder::new(decoder_options);
    match decoder.process(&mut data, None, Some(&mut runner)) {
        Ok(JxlDecoderStatus::BasicInfo) => {}
        _ => return Err(()),
    }
    match decoder.process(&mut data, None, Some(&mut runner)) {
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
        decoder.process(&mut data, Some(&mut output_bufs), Some(&mut runner)),
        Ok(JxlDecoderStatus::FrameHeader | JxlDecoderStatus::FrameComplete)
    ) {}

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode_parallel(data);
});
