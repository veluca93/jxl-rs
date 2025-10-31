// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
use crate::dct::{
    compute_scaled_dct, dct2d, idct2d, DCT1DImpl, IDCT1DImpl, DCT1D, IDCT1D, MAX_SCRATCH_SPACE,
};
use jxl_simd::ScalarDescriptor;
use test_log::test;

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;
use std::f64::consts::SQRT_2;

#[inline(always)]
fn alpha(u: usize) -> f64 {
    if u == 0 {
        FRAC_1_SQRT_2
    } else {
        1.0
    }
}

pub fn dct1d(input_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_rows = input_matrix.len();

    if num_rows == 0 {
        return Vec::new();
    }

    let num_cols = input_matrix[0].len();

    let mut output_matrix = vec![vec![0.0f64; num_cols]; num_rows];

    let scale: f64 = SQRT_2;

    // Precompute the DCT matrix (size: n_rows x n_rows)
    let mut dct_coeff_matrix = vec![vec![0.0f64; num_rows]; num_rows];
    for (u_freq, row) in dct_coeff_matrix.iter_mut().enumerate() {
        let alpha_u_val = alpha(u_freq);
        for (y_spatial, coeff) in row.iter_mut().enumerate() {
            *coeff = alpha_u_val
                * ((y_spatial as f64 + 0.5) * u_freq as f64 * PI / num_rows as f64).cos()
                * scale;
        }
    }

    // Perform the DCT calculation column by column
    for x_col_idx in 0..num_cols {
        for u_freq_idx in 0..num_rows {
            let mut sum = 0.0;
            for (y_spatial_idx, col) in input_matrix.iter().enumerate() {
                // This access `input_matrix[y_spatial_idx][x_col_idx]` assumes the input_matrix
                // is rectangular. If not, it might panic here.
                sum += dct_coeff_matrix[u_freq_idx][y_spatial_idx] * col[x_col_idx];
            }
            output_matrix[u_freq_idx][x_col_idx] = sum;
        }
    }

    output_matrix
}

pub fn idct1d(input_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_rows = input_matrix.len();

    if num_rows == 0 {
        return Vec::new();
    }

    let num_cols = input_matrix[0].len();

    let mut output_matrix = vec![vec![0.0f64; num_cols]; num_rows];

    let scale: f64 = SQRT_2;

    // Precompute the DCT matrix (size: num_rows x num_rows)
    let mut dct_coeff_matrix = vec![vec![0.0f64; num_rows]; num_rows];
    for (u_freq, row) in dct_coeff_matrix.iter_mut().enumerate() {
        let alpha_u_val = alpha(u_freq);
        for (y_def_idx, coeff) in row.iter_mut().enumerate() {
            *coeff = alpha_u_val
                * ((y_def_idx as f64 + 0.5) * u_freq as f64 * PI / num_rows as f64).cos()
                * scale;
        }
    }

    // Perform the IDCT calculation column by column
    for x_col_idx in 0..num_cols {
        for (y_row_idx, row) in output_matrix.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (u_freq_idx, col) in input_matrix.iter().enumerate() {
                // This access input_coeffs_matrix[u_freq_idx][x_col_idx] assumes input_coeffs_matrix
                // is rectangular. If not, it might panic here.
                sum += dct_coeff_matrix[u_freq_idx][y_row_idx] * col[x_col_idx];
            }
            row[x_col_idx] = sum;
        }
    }

    output_matrix
}

#[track_caller]
fn check_close(a: f64, b: f64, max_abs: f64) {
    let abs = (a - b).abs();
    assert!(abs < max_abs, "a: {a} b: {b} abs diff: {abs:?}");
}

#[track_caller]
fn check_all_close(a: &[f64], b: &[f64], max_abs: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        check_close(*a, *b, max_abs);
    }
}

#[test]
fn test_slow_dct1d() {
    const N_ROWS: usize = 8;
    const M_COLS: usize = 1;

    let flat_input_data: [f64; N_ROWS] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Prepare input_matrix for dct1d
    // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx]
    // For N_ROWS=8, M_COLS=1, this means 8 rows, each containing a Vec with 1 element.
    let input_matrix: Vec<Vec<f64>> = flat_input_data.iter().map(|&value| vec![value]).collect();

    let output_matrix: Vec<Vec<f64>> = dct1d(&input_matrix);

    let mut result_column: Vec<f64> = Vec::with_capacity(N_ROWS);
    if M_COLS > 0 {
        for row in output_matrix.iter() {
            result_column.push(row[0]);
        }
    }

    let expected = [
        2.80000000e+01,
        -1.82216412e+01,
        -1.38622135e-15,
        -1.90481783e+00,
        0.00000000e+00,
        -5.68239222e-01,
        -1.29520973e-15,
        -1.43407825e-01,
    ];

    check_all_close(&result_column, &expected, 1e-7);
}

#[test]
fn test_slow_dct1d_same_on_columns() {
    const N_ROWS: usize = 8;
    const M_COLS: usize = 5;

    // Prepare input_matrix for dct1d
    // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
    // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS).map(|r| vec![r as f64; M_COLS]).collect();

    // Call the refactored dct1d function which returns a new matrix
    let output_matrix: Vec<Vec<f64>> = dct1d(&input_matrix);

    // Expected output for a single column [0.0 .. N_ROWS-1.0]
    let single_column_dct_expected = [
        2.80000000e+01,
        -1.82216412e+01,
        -1.38622135e-15,
        -1.90481783e+00,
        0.00000000e+00,
        -5.68239222e-01,
        -1.29520973e-15,
        -1.43407825e-01,
    ];

    for r_freq_idx in 0..N_ROWS {
        let expected_row_values: Vec<f64> = vec![single_column_dct_expected[r_freq_idx]; M_COLS];
        check_all_close(&output_matrix[r_freq_idx], &expected_row_values, 1e-7);
    }
}

#[test]
fn test_slow_idct1d() {
    const N: usize = 8;
    const M: usize = 1;

    let flat_input_data: [f64; N] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let input_coeffs_matrix_p1: Vec<Vec<f64>> =
        flat_input_data.iter().map(|&value| vec![value]).collect();
    // Prepare input_matrix for dct1d
    // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
    // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].k
    let output_matrix: Vec<Vec<f64>> = idct1d(&input_coeffs_matrix_p1);

    let mut result_column: Vec<f64> = Vec::with_capacity(N);
    if M > 0 {
        for row_vec in output_matrix.iter() {
            result_column.push(row_vec[0]);
        }
    }

    let expected = [
        20.63473963,
        -22.84387206,
        8.99218712,
        -7.77138893,
        4.05078387,
        -3.47821595,
        1.32990088,
        -0.91413457,
    ];
    check_all_close(&result_column, &expected, 1e-7);
}

#[test]
fn test_slow_idct1d_same_on_columns() {
    const N_ROWS: usize = 8;
    const M_COLS: usize = 5;

    // Prepare input_matrix for idct1d
    // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
    // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS).map(|r| vec![r as f64; M_COLS]).collect();

    // Call the refactored idct1d function which returns a new matrix
    let output_matrix: Vec<Vec<f64>> = idct1d(&input_matrix);

    // Expected spatial output for a single input coefficient column [0.0 .. N_FREQUENCIES-1.0]
    // This is taken from the single-column test `test_slow_idct1d`
    let single_column_idct_expected = [
        20.63473963,
        -22.84387206,
        8.99218712,
        -7.77138893,
        4.05078387,
        -3.47821595,
        1.32990088,
        -0.91413457,
    ];

    // Verify each row of output_spatial_matrix.
    // The row output_spatial_matrix[r_spatial_idx] should consist of M_COLS elements,
    // all equal to single_column_idct_expected[r_spatial_idx].
    for r_spatial_idx in 0..N_ROWS {
        let expected_row_values: Vec<f64> =
            vec![single_column_idct_expected[r_spatial_idx]; M_COLS];
        check_all_close(&output_matrix[r_spatial_idx], &expected_row_values, 1e-7);
    }
}

#[test]
fn test_dct_idct_scaling() {
    const N_ROWS: usize = 7;
    const M_COLS: usize = 13;
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS)
        .map(|r_idx| {
            (0..M_COLS)
                // some arbitrary pattern
                .map(|c_idx| (r_idx + c_idx) as f64 * 7.7)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let dct_output = dct1d(&input_matrix);
    let idct_output = idct1d(&dct_output);

    // Verify that idct1d(dct1d(input)) == N_ROWS * input
    for r_idx in 0..N_ROWS {
        let expected_current_row_scaled: Vec<f64> = input_matrix[r_idx]
            .iter()
            .map(|&val| val * (N_ROWS as f64))
            .collect();

        check_all_close(&idct_output[r_idx], &expected_current_row_scaled, 1e-7);
    }
}

#[test]
fn test_idct_dct_scaling() {
    const N_ROWS: usize = 17;
    const M_COLS: usize = 11;
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS)
        .map(|r_idx| {
            (0..M_COLS)
                // some arbitrary pattern
                .map(|c_idx| (r_idx + c_idx) as f64 * 12.34)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let idct_output = idct1d(&input_matrix);
    let dct_output = dct1d(&idct_output);

    // Verify that dct1d(idct1d(input)) == N_ROWS * input
    for r_idx in 0..N_ROWS {
        let expected_current_row_scaled: Vec<f64> = input_matrix[r_idx]
            .iter()
            .map(|&val| val * (N_ROWS as f64))
            .collect();

        check_all_close(&dct_output[r_idx], &expected_current_row_scaled, 1e-7);
    }
}

macro_rules! test_dct1d_eq_slow_n {
    ($test_name:ident, $n_val:expr, $tolerance:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = 1;
            const NM: usize = N * M;

            // Generate input data for the reference dct1d.
            // Results in vec![vec![1.0], vec![2.0], ..., vec![N.0]]
            let input_matrix_for_ref: Vec<Vec<f64>> =
                std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                    .chunks(M)
                    .map(|row_slice| row_slice.to_vec())
                    .collect();

            let output_matrix_slow: Vec<Vec<f64>> = dct1d(&input_matrix_for_ref);

            // DCT1DImpl expects data in [[f32; M]; N] format.
            let mut input_arr_2d = [[0.0f32; M]; N];
            for r_idx in 0..N {
                for c_idx in 0..M {
                    input_arr_2d[r_idx][c_idx] = input_matrix_for_ref[r_idx][c_idx] as f32;
                }
            }

            let mut output = input_arr_2d;
            let d = ScalarDescriptor {};
            DCT1DImpl::<N>::do_dct::<_, M>(d, &mut output);

            for i in 0..N {
                check_close(output[i][0] as f64, output_matrix_slow[i][0], $tolerance);
            }
        }
    };
}

macro_rules! test_idct1d_eq_slow_n {
    ($test_name:ident, $n_val:expr, $tolerance:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = 1;
            const NM: usize = N * M;

            // Generate input data for the reference idct1d.
            // Results in vec![vec![1.0], vec![2.0], ..., vec![N.0]]
            let input_matrix_for_ref: Vec<Vec<f64>> =
                std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                    .chunks(M)
                    .map(|row_slice| row_slice.to_vec())
                    .collect();

            let output_matrix_slow: Vec<Vec<f64>> = idct1d(&input_matrix_for_ref);

            let mut input_arr_2d = [0.0f32; NM];
            for r_idx in 0..N {
                for c_idx in 0..M {
                    input_arr_2d[r_idx * M + c_idx] = input_matrix_for_ref[r_idx][c_idx] as f32;
                }
            }

            let mut output = input_arr_2d;
            let d = ScalarDescriptor {};

            let mut scratch = vec![0.0; MAX_SCRATCH_SPACE];
            IDCT1DImpl::<N>::idct_wrapper(d, &mut output, &mut scratch, M);

            for i in 0..N * M {
                check_close(output[i * M] as f64, output_matrix_slow[i][0], $tolerance);
            }
        }
    };
}

test_dct1d_eq_slow_n!(test_dct1d_1x1_eq_slow, 1, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_1x1_eq_slow, 1, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_2x1_eq_slow, 2, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_2x1_eq_slow, 2, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_4x1_eq_slow, 4, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_4x1_eq_slow, 4, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_8x1_eq_slow, 8, 1e-5);
test_idct1d_eq_slow_n!(test_idct1d_8x1_eq_slow, 8, 1e-5);
test_dct1d_eq_slow_n!(test_dct1d_16x1_eq_slow, 16, 1e-4);
test_idct1d_eq_slow_n!(test_idct1d_16x1_eq_slow, 16, 1e-4);
test_dct1d_eq_slow_n!(test_dct1d_32x1_eq_slow, 32, 1e-3);
test_idct1d_eq_slow_n!(test_idct1d_32x1_eq_slow, 32, 1e-3);
test_dct1d_eq_slow_n!(test_dct1d_64x1_eq_slow, 64, 1e-2);
test_idct1d_eq_slow_n!(test_idct1d_64x1_eq_slow, 64, 1e-2);
test_dct1d_eq_slow_n!(test_dct1d_128x1_eq_slow, 128, 1e-2);
test_idct1d_eq_slow_n!(test_idct1d_128x1_eq_slow, 128, 1e-2);
test_dct1d_eq_slow_n!(test_dct1d_256x1_eq_slow, 256, 1e-1);
test_idct1d_eq_slow_n!(test_idct1d_256x1_eq_slow, 256, 1e-1);

#[test]
fn test_idct1d_8x3_eq_slow() {
    const N: usize = 8;
    const M: usize = 3;
    const NM: usize = N * M; // 24

    // Initialize an N x M matrix with data from 1.0 to 24.0
    let input_coeffs_matrix_for_ref: Vec<Vec<f64>> =
        std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
            .chunks(M)
            .map(|row_slice| row_slice.to_vec())
            .collect();

    let output_matrix_slow: Vec<Vec<f64>> = idct1d(&input_coeffs_matrix_for_ref);

    // Prepare input for the implementation under test (IDCT1DImpl)
    let mut input_coeffs_for_fast_impl = [0.0f32; NM];
    for r in 0..N {
        for c in 0..M {
            // Use the same source coefficient values as the reference IDCT
            input_coeffs_for_fast_impl[r * M + c] = input_coeffs_matrix_for_ref[r][c] as f32;
        }
    }

    // This will be modified in-place by IDCT1DImpl
    let mut output_fast_impl = input_coeffs_for_fast_impl;

    // Call the implementation under test (operates on 2D data)
    let d = ScalarDescriptor {};
    let mut scratch = vec![0.0; MAX_SCRATCH_SPACE];
    IDCT1DImpl::<N>::idct_wrapper(d, &mut output_fast_impl, &mut scratch, M);

    // Compare results element-wise
    for r_idx in 0..N {
        for c_idx in 0..M {
            check_close(
                output_fast_impl[r_idx * M + c_idx] as f64,
                output_matrix_slow[r_idx][c_idx],
                2e-5,
            );
        }
    }
}

#[test]
fn test_dct1d_8x3_eq_slow() {
    const N: usize = 8;
    const M: usize = 3;
    const NM: usize = N * M; // 24

    // Initialize a 3 x 8 marix with data from 1.0 to 24.0
    let input_matrix_for_ref: Vec<Vec<f64>> = std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
        .chunks(M)
        .map(|row_slice| row_slice.to_vec())
        .collect();

    let output_matrix_slow: Vec<Vec<f64>> = dct1d(&input_matrix_for_ref);

    // Prepare input for the implementation under test (DCT1DImpl)
    // DCT1DImpl expects data in [[f32; M]; N] format.
    let mut input_for_fast_impl = [[0.0f32; M]; N];
    for r in 0..N {
        for c in 0..M {
            // Use the same source values as the reference DCT
            input_for_fast_impl[r][c] = input_matrix_for_ref[r][c] as f32;
        }
    }

    // This will be modified in-place by DCT1DImpl
    let mut output_fast_impl = input_for_fast_impl;

    // Call the implementation under test (operates on 2D data)
    let d = ScalarDescriptor {};
    DCT1DImpl::<N>::do_dct::<_, M>(d, &mut output_fast_impl);

    // Compare results element-wise
    for r_freq_idx in 0..N {
        for c_col_idx in 0..M {
            check_close(
                output_fast_impl[r_freq_idx][c_col_idx] as f64,
                output_matrix_slow[r_freq_idx][c_col_idx],
                1e-5,
            );
        }
    }
}

// TODO(firsching): possibly change these tests to test against slow
// (i)dct method (after adding 2d-variant there)
macro_rules! test_idct2d_exists_n_m {
    ($test_name:ident, $n_val:expr, $m_val:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = $m_val;
            let mut data = [0.0f32; M * N];
            let mut scratch = vec![0.0; MAX_SCRATCH_SPACE];
            let d = ScalarDescriptor {};
            idct2d::<_, N, M>(d, &mut data, &mut scratch);
        }
    };
}
macro_rules! test_dct2d_exists_n_m {
    ($test_name:ident, $n_val:expr, $m_val:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = $m_val;
            let mut data = [0.0f32; M * N];
            let mut scratch = vec![0.0; MAX_SCRATCH_SPACE];
            let d = ScalarDescriptor {};
            dct2d::<_, N, M>(d, &mut data, &mut scratch);
        }
    };
}
test_dct2d_exists_n_m!(test_dct2d_exists_1_1, 1, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_1_1, 1, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_1_2, 1, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_1_2, 1, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_1_4, 1, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_1_4, 1, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_1_8, 1, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_1_8, 1, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_1_16, 1, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_1_16, 1, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_1_32, 1, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_1_32, 1, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_1_64, 1, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_1_64, 1, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_1_128, 1, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_1_128, 1, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_1_256, 1, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_1_256, 1, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_2_1, 2, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_2_1, 2, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_2_2, 2, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_2_2, 2, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_2_4, 2, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_2_4, 2, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_2_8, 2, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_2_8, 2, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_2_16, 2, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_2_16, 2, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_2_32, 2, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_2_32, 2, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_2_64, 2, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_2_64, 2, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_2_128, 2, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_2_128, 2, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_2_256, 2, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_2_256, 2, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_4_1, 4, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_4_1, 4, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_4_2, 4, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_4_2, 4, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_4_4, 4, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_4_4, 4, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_4_8, 4, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_4_8, 4, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_4_16, 4, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_4_16, 4, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_4_32, 4, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_4_32, 4, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_4_64, 4, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_4_64, 4, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_4_128, 4, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_4_128, 4, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_4_256, 4, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_4_256, 4, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_8_1, 8, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_8_1, 8, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_8_2, 8, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_8_2, 8, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_8_4, 8, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_8_4, 8, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_8_8, 8, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_8_8, 8, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_8_16, 8, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_8_16, 8, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_8_32, 8, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_8_32, 8, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_8_64, 8, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_8_64, 8, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_8_128, 8, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_8_128, 8, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_8_256, 8, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_8_256, 8, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_16_1, 16, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_16_1, 16, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_16_2, 16, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_16_2, 16, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_16_4, 16, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_16_4, 16, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_16_8, 16, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_16_8, 16, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_16_16, 16, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_16_16, 16, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_16_32, 16, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_16_32, 16, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_16_64, 16, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_16_64, 16, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_16_128, 16, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_16_128, 16, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_16_256, 16, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_16_256, 16, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_32_1, 32, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_32_1, 32, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_32_2, 32, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_32_2, 32, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_32_4, 32, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_32_4, 32, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_32_8, 32, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_32_8, 32, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_32_16, 32, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_32_16, 32, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_32_32, 32, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_32_32, 32, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_32_64, 32, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_32_64, 32, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_32_128, 32, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_32_128, 32, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_32_256, 32, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_32_256, 32, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_64_1, 64, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_64_1, 64, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_64_2, 64, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_64_2, 64, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_64_4, 64, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_64_4, 64, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_64_8, 64, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_64_8, 64, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_64_16, 64, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_64_16, 64, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_64_32, 64, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_64_32, 64, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_64_64, 64, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_64_64, 64, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_64_128, 64, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_64_128, 64, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_64_256, 64, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_64_256, 64, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_128_1, 128, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_128_1, 128, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_128_2, 128, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_128_2, 128, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_128_4, 128, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_128_4, 128, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_128_8, 128, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_128_8, 128, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_128_16, 128, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_128_16, 128, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_128_32, 128, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_128_32, 128, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_128_64, 128, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_128_64, 128, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_128_128, 128, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_128_128, 128, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_128_256, 128, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_128_256, 128, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_256_1, 256, 1);
test_idct2d_exists_n_m!(test_idct2d_exists_256_1, 256, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_256_2, 256, 2);
test_idct2d_exists_n_m!(test_idct2d_exists_256_2, 256, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_256_4, 256, 4);
test_idct2d_exists_n_m!(test_idct2d_exists_256_4, 256, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_256_8, 256, 8);
test_idct2d_exists_n_m!(test_idct2d_exists_256_8, 256, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_256_16, 256, 16);
test_idct2d_exists_n_m!(test_idct2d_exists_256_16, 256, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_256_32, 256, 32);
test_idct2d_exists_n_m!(test_idct2d_exists_256_32, 256, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_256_64, 256, 64);
test_idct2d_exists_n_m!(test_idct2d_exists_256_64, 256, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_256_128, 256, 128);
test_idct2d_exists_n_m!(test_idct2d_exists_256_128, 256, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_256_256, 256, 256);
test_idct2d_exists_n_m!(test_idct2d_exists_256_256, 256, 256);

#[test]
fn test_compute_scaled_dct_wide() {
    let input = [
        [86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0, 87.0],
        [128.0, 122.0, 131.0, 72.0, 156.0, 112.0, 248.0, 55.0],
        [120.0, 31.0, 246.0, 177.0, 119.0, 154.0, 176.0, 248.0],
        [21.0, 151.0, 107.0, 101.0, 202.0, 71.0, 246.0, 48.0],
    ];

    let mut output = [0.0; 4 * 8];

    let d = ScalarDescriptor {};
    compute_scaled_dct::<_, 4, 8>(d, input, &mut output);

    for (a, b) in output.iter().zip([
        135.219, -13.1026, 0.573698, -6.19682, -29.5938, 11.5028, -13.3955, 21.9205, 1.4572,
        11.3448, 16.3991, 2.50104, -20.549, 0.363681, 3.94596, -4.05406, -8.21875, 6.57931,
        0.601308, 1.51804, -20.5312, -9.29264, -19.6983, -0.850355, 12.4189, -5.0881, 5.82096,
        -20.1997, 3.87769, 2.80762, 24.6634, -8.93341,
    ]) {
        check_close(*a as f64, b, 1e-3);
    }
}

#[test]
fn test_compute_scaled_dct_tall() {
    let input = [
        [86.0, 239.0, 213.0, 36.0],
        [34.0, 142.0, 248.0, 87.0],
        [128.0, 122.0, 131.0, 72.0],
        [156.0, 112.0, 248.0, 55.0],
        [120.0, 31.0, 246.0, 177.0],
        [119.0, 154.0, 176.0, 248.0],
        [21.0, 151.0, 107.0, 101.0],
        [202.0, 71.0, 246.0, 48.0],
    ];

    let mut output = [0.0; 8 * 4];

    let d = ScalarDescriptor {};
    compute_scaled_dct::<_, 8, 4>(d, input, &mut output);

    for (a, b) in output.iter().zip([
        135.219, -0.899633, -4.54363, 9.7776, 7.65625, -7.7203, 10.5073, -11.9921, -8.31418,
        5.39457, 11.3896, -17.5006, 11.6535, 12.6257, 9.27026, -0.767252, -29.5938, -19.9538,
        -17.5214, -0.467021, -3.28125, -7.67861, 11.3504, 5.01615, 24.9226, -4.19572, -7.10474,
        -16.7029, 24.2961, -16.8923, -3.32708, -4.09777,
    ]) {
        check_close(*a as f64, b, 1e-3);
    }
}
