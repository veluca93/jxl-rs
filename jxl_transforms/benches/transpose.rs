// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::hint::black_box;

use criterion::measurement::Measurement;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use jxl_simd::{bench_all_instruction_sets, SimdDescriptor};
use jxl_transforms::transpose::{transpose, transpose_in_place};

const MAX_AREA: usize = 256 * 256;

fn bench_transpose<D: SimdDescriptor>(
    d: D,
    c: &mut BenchmarkGroup<'_, impl Measurement>,
    name: &str,
) {
    let src: Vec<f32> = (0..MAX_AREA).map(|i| i as f32).collect();
    let mut dst = vec![0.0f32; MAX_AREA];

    macro_rules! run {
        ($rows:expr, $cols:expr, $label:literal) => {
            c.throughput(Throughput::Bytes(($rows * $cols * 4) as u64));
            let id = BenchmarkId::new(name, format_args!("{}", $label));
            c.bench_function(id, |b| {
                b.iter(|| {
                    d.call(
                        #[inline(always)]
                        |d| {
                            transpose(
                                d,
                                black_box(&src[..$rows * $cols]),
                                black_box(&mut dst[..$rows * $cols]),
                                $rows,
                                $cols,
                            );
                        },
                    );
                })
            });
        };
    }

    run!(4, 4, "4x4");
    run!(4, 8, "4x8");
    run!(8, 4, "8x4");
    run!(8, 8, "8x8");
    run!(16, 8, "16x8");
    run!(8, 16, "8x16");
    run!(16, 16, "16x16");
    run!(32, 8, "32x8");
    run!(8, 32, "8x32");
    run!(32, 16, "32x16");
    run!(16, 32, "16x32");
    run!(32, 32, "32x32");
    run!(64, 32, "64x32");
    run!(32, 64, "32x64");
    run!(64, 64, "64x64");
    run!(128, 64, "128x64");
    run!(64, 128, "64x128");
    run!(128, 128, "128x128");
    run!(256, 128, "256x128");
    run!(128, 256, "128x256");
    run!(256, 256, "256x256");
}

fn bench_transpose_in_place<D: SimdDescriptor>(
    d: D,
    c: &mut BenchmarkGroup<'_, impl Measurement>,
    name: &str,
) {
    let mut data: Vec<f32> = (0..MAX_AREA).map(|i| i as f32).collect();

    macro_rules! run {
        ($n:expr, $label:literal) => {
            c.throughput(Throughput::Bytes(($n * $n * 4) as u64));
            let id = BenchmarkId::new(name, format_args!("{}", $label));
            c.bench_function(id, |b| {
                b.iter(|| {
                    d.call(
                        #[inline(always)]
                        |d| transpose_in_place(d, black_box(&mut data[..$n * $n]), $n),
                    );
                })
            });
        };
    }

    run!(4, "4x4");
    run!(8, "8x8");
    run!(16, "16x16");
    run!(32, "32x32");
    run!(64, "64x64");
    run!(128, "128x128");
    run!(256, "256x256");
}

fn transpose_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    let g = &mut group;

    bench_all_instruction_sets!(bench_transpose, g);

    group.finish();
}

fn transpose_in_place_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_in_place");
    let g = &mut group;

    bench_all_instruction_sets!(bench_transpose_in_place, g);

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = transpose_benches, transpose_in_place_benches
);
criterion_main!(benches);
