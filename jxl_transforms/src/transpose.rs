// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{simd_specialize, F32SimdVec, F32VecArray, SimdDescriptor};

#[inline(always)]
fn transpose_f32_pow2<D: SimdDescriptor, const N: usize, const N4: usize>(
    d: D,
    data: &mut [F32VecArray<D>],
    stride: usize,
) {
    assert_eq!(N, D::F32Vec::LEN);
    assert_eq!(N4, 4 * N);
    assert!(data.len() > (N - 1) * stride);

    let mut cur = [D::F32Vec::zero(d); N];
    let mut nxt = [D::F32Vec::zero(d); N];
    let mut tmp = [0.0f32; N4];
    for i in 0..N {
        cur[i] = D::F32Vec::load_array(d, &data[i * stride]);
    }

    let mut rem = N;
    let q = N / 4;
    while rem >= 4 {
        for i in 0..q {
            D::F32Vec::store_interleaved_4(
                cur[i],
                cur[i + q],
                cur[i + 2 * q],
                cur[i + 3 * q],
                &mut tmp,
            );
            let tmp_arr = D::F32Vec::make_array_slice(&tmp);
            for k in 0..4 {
                nxt[4 * i + k] = D::F32Vec::load_array(d, &tmp_arr[k]);
            }
        }
        cur.copy_from_slice(&nxt);
        rem /= 4;
    }
    if rem == 2 {
        let h = N / 2;
        let tmp_half = &mut tmp[..2 * N];
        for i in 0..h {
            D::F32Vec::store_interleaved_2(cur[i], cur[i + h], tmp_half);
            let tmp_arr = D::F32Vec::make_array_slice(tmp_half);
            nxt[2 * i] = D::F32Vec::load_array(d, &tmp_arr[0]);
            nxt[2 * i + 1] = D::F32Vec::load_array(d, &tmp_arr[1]);
        }
        cur.copy_from_slice(&nxt);
    }

    for i in 0..N {
        cur[i].store_array(&mut data[i * stride]);
    }
}

#[inline(always)]
fn transpose_square_f32_fallback<D: SimdDescriptor>(
    d: D,
    data: &mut [F32VecArray<D>],
    stride: usize,
) {
    match D::F32Vec::LEN {
        0 | 1 => {}
        2 => transpose_f32_pow2::<D, 2, 8>(d, data, stride),
        4 => transpose_f32_pow2::<D, 4, 16>(d, data, stride),
        8 => transpose_f32_pow2::<D, 8, 32>(d, data, stride),
        16 => transpose_f32_pow2::<D, 16, 64>(d, data, stride),
        32 => transpose_f32_pow2::<D, 32, 128>(d, data, stride),
        n => panic!("unsupported F32Vec::LEN = {n}"),
    }
}

/// Transposes the `D::F32Vec::LEN` x `D::F32Vec::LEN` matrix formed by array elements
/// `data[stride * i]` for `i = 0..D::F32Vec::LEN`.
#[inline(always)]
pub fn transpose_square_f32<D: SimdDescriptor>(d: D, data: &mut [F32VecArray<D>], stride: usize) {
    simd_specialize!(
        fn(d: D, data: &mut [F32VecArray<D>], stride: usize) {
            avx512 => {
                use std::arch::x86_64::*;

                use jxl_simd::F32VecAvx512;

                assert!(data.len() > stride * 15);

                let r0 = F32VecAvx512::load_array(d, &data[0]).raw();
                let r1 = F32VecAvx512::load_array(d, &data[stride]).raw();
                let r2 = F32VecAvx512::load_array(d, &data[2 * stride]).raw();
                let r3 = F32VecAvx512::load_array(d, &data[3 * stride]).raw();
                let r4 = F32VecAvx512::load_array(d, &data[4 * stride]).raw();
                let r5 = F32VecAvx512::load_array(d, &data[5 * stride]).raw();
                let r6 = F32VecAvx512::load_array(d, &data[6 * stride]).raw();
                let r7 = F32VecAvx512::load_array(d, &data[7 * stride]).raw();
                let r8 = F32VecAvx512::load_array(d, &data[8 * stride]).raw();
                let r9 = F32VecAvx512::load_array(d, &data[9 * stride]).raw();
                let r10 = F32VecAvx512::load_array(d, &data[10 * stride]).raw();
                let r11 = F32VecAvx512::load_array(d, &data[11 * stride]).raw();
                let r12 = F32VecAvx512::load_array(d, &data[12 * stride]).raw();
                let r13 = F32VecAvx512::load_array(d, &data[13 * stride]).raw();
                let r14 = F32VecAvx512::load_array(d, &data[14 * stride]).raw();
                let r15 = F32VecAvx512::load_array(d, &data[15 * stride]).raw();

                // Stage 1: Unpack low/high pairs
                let t0 = _mm512_unpacklo_ps(r0, r1);
                let t1 = _mm512_unpackhi_ps(r0, r1);
                let t2 = _mm512_unpacklo_ps(r2, r3);
                let t3 = _mm512_unpackhi_ps(r2, r3);
                let t4 = _mm512_unpacklo_ps(r4, r5);
                let t5 = _mm512_unpackhi_ps(r4, r5);
                let t6 = _mm512_unpacklo_ps(r6, r7);
                let t7 = _mm512_unpackhi_ps(r6, r7);
                let t8 = _mm512_unpacklo_ps(r8, r9);
                let t9 = _mm512_unpackhi_ps(r8, r9);
                let t10 = _mm512_unpacklo_ps(r10, r11);
                let t11 = _mm512_unpackhi_ps(r10, r11);
                let t12 = _mm512_unpacklo_ps(r12, r13);
                let t13 = _mm512_unpackhi_ps(r12, r13);
                let t14 = _mm512_unpacklo_ps(r14, r15);
                let t15 = _mm512_unpackhi_ps(r14, r15);

                // Cast to 64 bits.
                let t0 = _mm512_castps_pd(t0);
                let t1 = _mm512_castps_pd(t1);
                let t2 = _mm512_castps_pd(t2);
                let t3 = _mm512_castps_pd(t3);
                let t4 = _mm512_castps_pd(t4);
                let t5 = _mm512_castps_pd(t5);
                let t6 = _mm512_castps_pd(t6);
                let t7 = _mm512_castps_pd(t7);
                let t8 = _mm512_castps_pd(t8);
                let t9 = _mm512_castps_pd(t9);
                let t10 = _mm512_castps_pd(t10);
                let t11 = _mm512_castps_pd(t11);
                let t12 = _mm512_castps_pd(t12);
                let t13 = _mm512_castps_pd(t13);
                let t14 = _mm512_castps_pd(t14);
                let t15 = _mm512_castps_pd(t15);

                // Stage 2: Shuffle to group 32-bit elements
                let s0 = _mm512_unpacklo_pd(t0, t2);
                let s1 = _mm512_unpackhi_pd(t0, t2);
                let s2 = _mm512_unpacklo_pd(t1, t3);
                let s3 = _mm512_unpackhi_pd(t1, t3);
                let s4 = _mm512_unpacklo_pd(t4, t6);
                let s5 = _mm512_unpackhi_pd(t4, t6);
                let s6 = _mm512_unpacklo_pd(t5, t7);
                let s7 = _mm512_unpackhi_pd(t5, t7);
                let s8 = _mm512_unpacklo_pd(t8, t10);
                let s9 = _mm512_unpackhi_pd(t8, t10);
                let s10 = _mm512_unpacklo_pd(t9, t11);
                let s11 = _mm512_unpackhi_pd(t9, t11);
                let s12 = _mm512_unpacklo_pd(t12, t14);
                let s13 = _mm512_unpackhi_pd(t12, t14);
                let s14 = _mm512_unpacklo_pd(t13, t15);
                let s15 = _mm512_unpackhi_pd(t13, t15);

                // Stage 3: 128-bit permute
                let idx_hi = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
                let idx_lo = _mm512_add_epi64(idx_hi, _mm512_set1_epi64(2));

                let c0 = _mm512_permutex2var_pd(s0, idx_hi, s4);
                let c1 = _mm512_permutex2var_pd(s1, idx_hi, s5);
                let c2 = _mm512_permutex2var_pd(s2, idx_hi, s6);
                let c3 = _mm512_permutex2var_pd(s3, idx_hi, s7);
                let c4 = _mm512_permutex2var_pd(s0, idx_lo, s4);
                let c5 = _mm512_permutex2var_pd(s1, idx_lo, s5);
                let c6 = _mm512_permutex2var_pd(s2, idx_lo, s6);
                let c7 = _mm512_permutex2var_pd(s3, idx_lo, s7);
                let c8 = _mm512_permutex2var_pd(s8, idx_hi, s12);
                let c9 = _mm512_permutex2var_pd(s9, idx_hi, s13);
                let c10 = _mm512_permutex2var_pd(s10, idx_hi, s14);
                let c11 = _mm512_permutex2var_pd(s11, idx_hi, s15);
                let c12 = _mm512_permutex2var_pd(s8, idx_lo, s12);
                let c13 = _mm512_permutex2var_pd(s9, idx_lo, s13);
                let c14 = _mm512_permutex2var_pd(s10, idx_lo, s14);
                let c15 = _mm512_permutex2var_pd(s11, idx_lo, s15);

                // Stage 4: 256-bit permute
                let idx_hi = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
                let idx_lo = _mm512_add_epi64(idx_hi, _mm512_set1_epi64(4));

                let o0 = _mm512_permutex2var_pd(c0, idx_hi, c8);
                let o1 = _mm512_permutex2var_pd(c1, idx_hi, c9);
                let o2 = _mm512_permutex2var_pd(c2, idx_hi, c10);
                let o3 = _mm512_permutex2var_pd(c3, idx_hi, c11);
                let o4 = _mm512_permutex2var_pd(c4, idx_hi, c12);
                let o5 = _mm512_permutex2var_pd(c5, idx_hi, c13);
                let o6 = _mm512_permutex2var_pd(c6, idx_hi, c14);
                let o7 = _mm512_permutex2var_pd(c7, idx_hi, c15);
                let o8 = _mm512_permutex2var_pd(c0, idx_lo, c8);
                let o9 = _mm512_permutex2var_pd(c1, idx_lo, c9);
                let o10 = _mm512_permutex2var_pd(c2, idx_lo, c10);
                let o11 = _mm512_permutex2var_pd(c3, idx_lo, c11);
                let o12 = _mm512_permutex2var_pd(c4, idx_lo, c12);
                let o13 = _mm512_permutex2var_pd(c5, idx_lo, c13);
                let o14 = _mm512_permutex2var_pd(c6, idx_lo, c14);
                let o15 = _mm512_permutex2var_pd(c7, idx_lo, c15);

                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o0)).store_array(&mut data[0]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o1)).store_array(&mut data[stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o2)).store_array(&mut data[2 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o3)).store_array(&mut data[3 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o4)).store_array(&mut data[4 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o5)).store_array(&mut data[5 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o6)).store_array(&mut data[6 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o7)).store_array(&mut data[7 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o8)).store_array(&mut data[8 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o9)).store_array(&mut data[9 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o10))
                    .store_array(&mut data[10 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o11))
                    .store_array(&mut data[11 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o12))
                    .store_array(&mut data[12 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o13))
                    .store_array(&mut data[13 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o14))
                    .store_array(&mut data[14 * stride]);
                F32VecAvx512::from_raw(d, _mm512_castpd_ps(o15))
                    .store_array(&mut data[15 * stride]);
            }
            neon => {
                use std::arch::aarch64::*;

                use jxl_simd::{F32VecNeon, unaligned::aarch64 as unaligned};

                if stride == 1 {
                    assert!(data.len() > 3);
                    let float32x4x4_t(p0, p1, p2, p3) =
                        unaligned::vld4q_f32(data.as_flattened().first_chunk().unwrap());

                    F32VecNeon::from_raw(d, p0).store_array(&mut data[0]);
                    F32VecNeon::from_raw(d, p1).store_array(&mut data[1]);
                    F32VecNeon::from_raw(d, p2).store_array(&mut data[2]);
                    F32VecNeon::from_raw(d, p3).store_array(&mut data[3]);
                } else {
                    assert!(data.len() > 3 * stride);

                    let p0 = F32VecNeon::load_array(d, &data[0]).raw();
                    let p1 = F32VecNeon::load_array(d, &data[stride]).raw();
                    let p2 = F32VecNeon::load_array(d, &data[2 * stride]).raw();
                    let p3 = F32VecNeon::load_array(d, &data[3 * stride]).raw();

                    // Stage 1: Transpose within each of 2x2 blocks
                    let tr0 = vreinterpretq_f64_f32(vtrn1q_f32(p0, p1));
                    let tr1 = vreinterpretq_f64_f32(vtrn2q_f32(p0, p1));
                    let tr2 = vreinterpretq_f64_f32(vtrn1q_f32(p2, p3));
                    let tr3 = vreinterpretq_f64_f32(vtrn2q_f32(p2, p3));

                    // Stage 2: Transpose 2x2 grid of 2x2 blocks
                    let p0 = vreinterpretq_f32_f64(vzip1q_f64(tr0, tr2));
                    let p1 = vreinterpretq_f32_f64(vzip1q_f64(tr1, tr3));
                    let p2 = vreinterpretq_f32_f64(vzip2q_f64(tr0, tr2));
                    let p3 = vreinterpretq_f32_f64(vzip2q_f64(tr1, tr3));

                    F32VecNeon::from_raw(d, p0).store_array(&mut data[0]);
                    F32VecNeon::from_raw(d, p1).store_array(&mut data[stride]);
                    F32VecNeon::from_raw(d, p2).store_array(&mut data[2 * stride]);
                    F32VecNeon::from_raw(d, p3).store_array(&mut data[3 * stride]);
                }
            }
            v256 => {
                assert!(data.len() > 7 * stride);

                let r0 = D::F32Vec::load_array(d, &data[0]);
                let r1 = D::F32Vec::load_array(d, &data[stride]);
                let r2 = D::F32Vec::load_array(d, &data[2 * stride]);
                let r3 = D::F32Vec::load_array(d, &data[3 * stride]);
                let r4 = D::F32Vec::load_array(d, &data[4 * stride]);
                let r5 = D::F32Vec::load_array(d, &data[5 * stride]);
                let r6 = D::F32Vec::load_array(d, &data[6 * stride]);
                let r7 = D::F32Vec::load_array(d, &data[7 * stride]);

                let mut tmp = [[0.0f32; 8]; 8];
                D::F32Vec::store_interleaved_8(
                    r0,
                    r1,
                    r2,
                    r3,
                    r4,
                    r5,
                    r6,
                    r7,
                    tmp.as_flattened_mut(),
                );
                for (i, row) in tmp.iter().enumerate() {
                    D::F32Vec::load_array(d, row).store_array(&mut data[i * stride]);
                }
            }
            v128 => {
                assert!(data.len() > 3 * stride);

                let r0 = D::F32Vec::load_array(d, &data[0]);
                let r1 = D::F32Vec::load_array(d, &data[stride]);
                let r2 = D::F32Vec::load_array(d, &data[2 * stride]);
                let r3 = D::F32Vec::load_array(d, &data[3 * stride]);

                let mut tmp = [[0.0f32; 4]; 4];
                D::F32Vec::store_interleaved_4(r0, r1, r2, r3, tmp.as_flattened_mut());
                for (i, row) in tmp.iter().enumerate() {
                    D::F32Vec::load_array(d, row).store_array(&mut data[i * stride]);
                }
            }
            _ => transpose_square_f32_fallback(d, data, stride),
        }
    );
}

#[cfg(test)]
mod tests {
    use jxl_simd::{test_all_instruction_sets, F32SimdVec, SimdDescriptor};

    use super::{transpose_square_f32, transpose_square_f32_fallback};

    fn check_transpose_with<D: SimdDescriptor>(
        d: D,
        f: fn(D, &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray], usize),
    ) {
        let len = D::F32Vec::LEN;
        let input: Vec<f32> = (0..len * len).map(|i| i as f32).collect();

        let mut output = input.clone();
        f(d, D::F32Vec::make_array_slice_mut(&mut output), 1);
        for i in 0..len {
            for j in 0..len {
                assert_eq!(output[i * len + j], input[j * len + i]);
            }
        }

        let stride = 2;
        let mut input_strided = vec![0.0f32; len * len * stride];
        for i in 0..len {
            for j in 0..len {
                input_strided[(i * stride) * len + j] = (i * len + j) as f32;
            }
        }
        let mut output_strided = input_strided.clone();
        f(
            d,
            D::F32Vec::make_array_slice_mut(&mut output_strided),
            stride,
        );
        for i in 0..len {
            for j in 0..len {
                assert_eq!(
                    output_strided[(i * stride) * len + j],
                    input_strided[(j * stride) * len + i]
                );
            }
        }
    }

    fn test_transpose_square_f32<D: SimdDescriptor>(d: D) {
        check_transpose_with(d, transpose_square_f32);
    }
    test_all_instruction_sets!(test_transpose_square_f32);

    fn test_transpose_square_f32_fallback<D: SimdDescriptor>(d: D) {
        check_transpose_with(d, transpose_square_f32_fallback);
    }
    test_all_instruction_sets!(test_transpose_square_f32_fallback);
}
