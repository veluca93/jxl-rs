// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use jxl_simd::{F32SimdVec, ScalarDescriptor, SimdDescriptor};
use crate::scales::WcMultipliers;
use std::f64::consts::SQRT_2;

/// Temporary storage needed, in floats.
pub const MAX_SCRATCH_SPACE: usize = 256 * 256 * 3;

// TODO(veluca): remove the second generic.
struct CoeffBundle<const N: usize, const SZ: usize>;

pub struct DCT1DImpl<const SIZE: usize>;
pub struct IDCT1DImpl<const SIZE: usize>;

pub trait DCT1D {
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(d: D, data: &mut [[f32; COLUMNS]]);
}
pub trait IDCT1D {
    const SIZE: usize;

    // D = instruction level available.
    // VD = descriptor defining vector length.
    // Keeping those distinct allows to potentially generate better code, at the cost of an
    // increase in code size and compilation time.
    fn do_idct<D: SimdDescriptor, VD: SimdDescriptor>(
        d: D,
        vd: VD,
        data: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
        scratch: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
    );

    fn idct_wrapper<D: SimdDescriptor>(d: D, data: &mut [f32], scratch: &mut [f32], columns: usize);
}

// This is a macro because Rust otherwise complains about creating arrays of size Self::SIZE.
macro_rules! impl_idct_wrapper {
    () => {
    #[inline(always)]
    fn idct_wrapper<D: SimdDescriptor>(
        d: D,
        data: &mut [f32],
        scratch: &mut [f32],
        columns: usize,
    ) {
        macro_rules! run_with_descriptor {
            ($descr: ty, $get_descr: expr) => {
                let data = <$descr as SimdDescriptor>::F32Vec::make_array_slice_mut(data);
                let scratch = <$descr as SimdDescriptor>::F32Vec::make_array_slice_mut(scratch);
                let vecs_per_column = columns / <$descr as SimdDescriptor>::F32Vec::LEN;
                for i in 0..vecs_per_column {
                    let vd = $get_descr;
                    // Ensure the compiler does not believe that writes to `scratch` are
                    // important, if the total stack size of scratch space we need is not too high.
                    let mut data_buf =
                        [<<$descr as SimdDescriptor>::F32Vec as F32SimdVec>::UnderlyingArray::default();
                            Self::SIZE];
                    let (data_buf, scratch) = if Self::SIZE <= 16 {
                        (&mut data_buf[..], &mut scratch[..])
                    } else {
                        scratch.split_at_mut(Self::SIZE)
                    };
                    for (db, d) in data_buf
                        .iter_mut()
                        .zip(data.iter().skip(i).step_by(vecs_per_column))
                    {
                        *db = *d;
                    }
                    Self::do_idct(d, vd, data_buf, scratch);
                    for (db, d) in data_buf
                        .iter()
                        .zip(data.iter_mut().skip(i).step_by(vecs_per_column))
                    {
                        *d = *db;
                    }
                }
            };
        }
        if columns < 4 {
            run_with_descriptor!(ScalarDescriptor, ScalarDescriptor::new().unwrap());
        } else if columns < 8 {
            run_with_descriptor!(D::Descriptor128, d.maybe_downgrade_128bit());
        } else if columns < 16 {
            run_with_descriptor!(D::Descriptor256, d.maybe_downgrade_256bit());
        } else {
            run_with_descriptor!(D, d);
        }
    }
    };
}

/// Threshold for choosing scalar vs SIMD paths in DCT operations.
/// For column counts <= this value, scalar code is faster than masked SIMD operations
/// due to the overhead of mask setup and partial vector operations.
/// This threshold is tuned for typical SIMD vector lengths (AVX: 8 floats, AVX-512: 16 floats).
const SIMD_THRESHOLD: usize = 4;

impl DCT1D for DCT1DImpl<1> {
    #[inline(always)]
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(_d: D, _data: &mut [[f32; COLUMNS]]) {
        // Do nothing
    }
}
impl IDCT1D for IDCT1DImpl<1> {
    const SIZE: usize = 1;
    #[inline(always)]
    fn do_idct<D: SimdDescriptor, VD: SimdDescriptor>(
        _d: D,
        _vd: VD,
        _data: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
        _scratch: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
    ) {
        // Nothing to do.
    }

    fn idct_wrapper<D: SimdDescriptor>(
        _d: D,
        _data: &mut [f32],
        _scratch: &mut [f32],
        _columns: usize,
    ) {
        // Nothing to do.
    }
}

impl DCT1D for DCT1DImpl<2> {
    #[inline(always)]
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(d: D, data: &mut [[f32; COLUMNS]]) {
        if COLUMNS <= SIMD_THRESHOLD {
            // Scalar path: faster than masked SIMD for small sizes
            for j in 0..COLUMNS {
                let temp0 = data[0][j];
                let temp1 = data[1][j];
                data[0][j] = temp0 + temp1;
                data[1][j] = temp0 - temp1;
            }
        } else {
            // SIMD path - loop over multiple vectors
            let mut j = 0;

            // Process full vectors
            while j + D::F32Vec::LEN <= COLUMNS {
                let temp0 = D::F32Vec::load(d, &data[0][j..]);
                let temp1 = D::F32Vec::load(d, &data[1][j..]);
                (temp0 + temp1).store(&mut data[0][j..]);
                (temp0 - temp1).store(&mut data[1][j..]);
                j += D::F32Vec::LEN;
            }

            // Handle remainder
            while j < COLUMNS {
                let temp0 = data[0][j];
                let temp1 = data[1][j];
                data[0][j] = temp0 + temp1;
                data[1][j] = temp0 - temp1;
                j += 1;
            }
        }
    }
}

impl IDCT1D for IDCT1DImpl<2> {
    const SIZE: usize = 2;
    #[inline(always)]
    fn do_idct<D: SimdDescriptor, VD: SimdDescriptor>(
        _d: D,
        vd: VD,
        data: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
        _scratch: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
    ) {
        let temp0 = VD::F32Vec::load_array(vd, &data[0]);
        let temp1 = VD::F32Vec::load_array(vd, &data[1]);
        (temp0 + temp1).store_array(&mut data[0]);
        (temp0 - temp1).store_array(&mut data[1]);
    }

    impl_idct_wrapper!();
}

// Inline the recursive calls of DCTs of size up to 16, so that the compiler can optimize them
// better.
macro_rules! maybe_call {
    ($d:expr, 2, $($call:tt)*) => {
        $($call)*
    };
    ($d:expr, 4, $($call:tt)*) => {
        $($call)*
    };
    ($d:expr, 8, $($call:tt)*) => {
        $($call)*
    };
    ($d:expr, 16, $($call:tt)*) => {
        $d.call(#[inline(always)] |_d| $($call)*)
    };
    ($d:expr, 32, $($call:tt)*) => {
        $d.call(#[inline(always)] |_d| $($call)*)
    };
    ($d:expr, 64, $($call:tt)*) => {
        $d.call(#[inline(always)] |_d| $($call)*)
    };
    ($d:expr, 128, $($call:tt)*) => {
        $d.call(#[inline(always)] |_d| $($call)*)
    };
}

macro_rules! define_dct_1d {
    ($n:literal, $nhalf: tt) => {
        // Helper functions for CoeffBundle operating on $nhalf rows
        impl<const SZ: usize> CoeffBundle<$nhalf, SZ> {
            /// Adds a_in1[i] and a_in2[$nhalf - 1 - i], storing in a_out[i].
            #[inline(always)]
            fn add_reverse<D: SimdDescriptor, const COLUMNS: usize>(
                d: D,
                a_in1: &[[f32; SZ]],
                a_in2: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
            ) {
                const N_HALF_CONST: usize = $nhalf;

                if COLUMNS <= SIMD_THRESHOLD {
                    // Scalar for small sizes (faster than masked SIMD)
                    for i in 0..N_HALF_CONST {
                        for j in 0..COLUMNS {
                            a_out[i][j] = a_in1[i][j] + a_in2[N_HALF_CONST - 1 - i][j];
                        }
                    }
                } else {
                    // SIMD path - loop over multiple vectors
                    for i in 0..N_HALF_CONST {
                        let mut j = 0;

                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            let in1 = D::F32Vec::load(d, &a_in1[i][j..]);
                            let in2 = D::F32Vec::load(d, &a_in2[N_HALF_CONST - 1 - i][j..]);
                            (in1 + in2).store(&mut a_out[i][j..]);
                            j += D::F32Vec::LEN;
                        }

                        // Handle remainder
                        while j < COLUMNS {
                            a_out[i][j] = a_in1[i][j] + a_in2[N_HALF_CONST - 1 - i][j];
                            j += 1;
                        }
                    }
                }
            }

            /// Subtracts a_in2[$nhalf - 1 - i] from a_in1[i], storing in a_out[i].
            #[inline(always)]
            fn sub_reverse<D: SimdDescriptor, const COLUMNS: usize>(
                d: D,
                a_in1: &[[f32; SZ]],
                a_in2: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
            ) {
                const N_HALF_CONST: usize = $nhalf;

                if COLUMNS <= SIMD_THRESHOLD {
                    // Scalar for small sizes
                    for i in 0..N_HALF_CONST {
                        for j in 0..COLUMNS {
                            a_out[i][j] = a_in1[i][j] - a_in2[N_HALF_CONST - 1 - i][j];
                        }
                    }
                } else {
                    // SIMD path - loop over multiple vectors
                    for i in 0..N_HALF_CONST {
                        let mut j = 0;

                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            let in1 = D::F32Vec::load(d, &a_in1[i][j..]);
                            let in2 = D::F32Vec::load(d, &a_in2[N_HALF_CONST - 1 - i][j..]);
                            (in1 - in2).store(&mut a_out[i][j..]);
                            j += D::F32Vec::LEN;
                        }

                        // Handle remainder
                        while j < COLUMNS {
                            a_out[i][j] = a_in1[i][j] - a_in2[N_HALF_CONST - 1 - i][j];
                            j += 1;
                        }
                    }
                }
            }

            /// Applies the B transform (forward DCT step).
            #[inline(always)]
            fn b<D: SimdDescriptor, const COLUMNS: usize>(d: D, coeff: &mut [[f32; SZ]]) {
                const N_HALF_CONST: usize = $nhalf;

                if COLUMNS <= SIMD_THRESHOLD {
                    // Scalar for small sizes
                    for j in 0..COLUMNS {
                        coeff[0][j] = coeff[0][j] * (SQRT_2 as f32) + coeff[1][j];
                    }
                    #[allow(clippy::reversed_empty_ranges)]
                    for i in 1..(N_HALF_CONST - 1) {
                        for j in 0..COLUMNS {
                            coeff[i][j] += coeff[i + 1][j];
                        }
                    }
                } else {
                    // SIMD path - loop over multiple vectors
                    let sqrt2 = D::F32Vec::splat(d, SQRT_2 as f32);
                    let mut j = 0;

                    // Process full vectors for row 0
                    while j + D::F32Vec::LEN <= COLUMNS {
                        let coeff0 = D::F32Vec::load(d, &coeff[0][j..]);
                        let coeff1 = D::F32Vec::load(d, &coeff[1][j..]);
                        coeff0.mul_add(sqrt2, coeff1).store(&mut coeff[0][j..]);
                        j += D::F32Vec::LEN;
                    }
                    // Handle remainder for row 0
                    while j < COLUMNS {
                        coeff[0][j] = coeff[0][j] * (SQRT_2 as f32) + coeff[1][j];
                        j += 1;
                    }

                    #[allow(clippy::reversed_empty_ranges)]
                    for i in 1..(N_HALF_CONST - 1) {
                        let mut j = 0;
                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            let coeffs_curr = D::F32Vec::load(d, &coeff[i][j..]);
                            let coeffs_next = D::F32Vec::load(d, &coeff[i + 1][j..]);
                            (coeffs_curr + coeffs_next).store(&mut coeff[i][j..]);
                            j += D::F32Vec::LEN;
                        }
                        // Handle remainder
                        while j < COLUMNS {
                            coeff[i][j] += coeff[i + 1][j];
                            j += 1;
                        }
                    }
                }
            }
        }

        // Helper functions for CoeffBundle operating on $n rows
        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            /// Multiplies the second half of `coeff` by WcMultipliers.
            #[inline(always)]
            fn multiply<D: SimdDescriptor, const COLUMNS: usize>(d: D, coeff: &mut [[f32; SZ]]) {
                const N_CONST: usize = $n;
                const N_HALF_CONST: usize = $nhalf;

                if COLUMNS <= SIMD_THRESHOLD {
                    // Scalar path
                    for i in 0..N_HALF_CONST {
                        let mul_val = WcMultipliers::<N_CONST>::K_MULTIPLIERS[i];
                        for j in 0..COLUMNS {
                            coeff[N_HALF_CONST + i][j] *= mul_val;
                        }
                    }
                } else {
                    // SIMD path - loop over multiple vectors
                    for i in 0..N_HALF_CONST {
                        let mul_val_scalar = WcMultipliers::<N_CONST>::K_MULTIPLIERS[i];
                        let mul_val = D::F32Vec::splat(d, mul_val_scalar);
                        let mut j = 0;

                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            let coeffs = D::F32Vec::load(d, &coeff[N_HALF_CONST + i][j..]);
                            (coeffs * mul_val).store(&mut coeff[N_HALF_CONST + i][j..]);
                            j += D::F32Vec::LEN;
                        }

                        // Handle remainder
                        while j < COLUMNS {
                            coeff[N_HALF_CONST + i][j] *= mul_val_scalar;
                            j += 1;
                        }
                    }
                }
            }

            /// De-interleaves `a_in` into `a_out`.
            #[inline(always)]
            fn inverse_even_odd<D: SimdDescriptor, const COLUMNS: usize>(
                d: D,
                a_in: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
            ) {
                const N_HALF_CONST: usize = $nhalf;

                if COLUMNS <= SIMD_THRESHOLD {
                    // Scalar path
                    for i in 0..N_HALF_CONST {
                        for j in 0..COLUMNS {
                            a_out[2 * i][j] = a_in[i][j];
                        }
                    }
                    for i in 0..N_HALF_CONST {
                        for j in 0..COLUMNS {
                            a_out[2 * i + 1][j] = a_in[N_HALF_CONST + i][j];
                        }
                    }
                } else {
                    // SIMD path - loop over multiple vectors
                    for i in 0..N_HALF_CONST {
                        let mut j = 0;

                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            D::F32Vec::load(d, &a_in[i][j..]).store(&mut a_out[2 * i][j..]);
                            j += D::F32Vec::LEN;
                        }

                        // Handle remainder
                        while j < COLUMNS {
                            a_out[2 * i][j] = a_in[i][j];
                            j += 1;
                        }
                    }
                    for i in 0..N_HALF_CONST {
                        let mut j = 0;

                        // Process full vectors
                        while j + D::F32Vec::LEN <= COLUMNS {
                            D::F32Vec::load(d, &a_in[N_HALF_CONST + i][j..])
                                .store(&mut a_out[2 * i + 1][j..]);
                            j += D::F32Vec::LEN;
                        }

                        // Handle remainder
                        while j < COLUMNS {
                            a_out[2 * i + 1][j] = a_in[N_HALF_CONST + i][j];
                            j += 1;
                        }
                    }
                }
            }
        }

        impl DCT1D for DCT1DImpl<$n> {
            #[inline(always)]
            fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(d: D, data: &mut [[f32; COLUMNS]]) {
                const { assert!($nhalf * 2 == $n, "N/2 * 2 must be N") }
                assert!(
                    data.len() == $n,
                    "Input data must have $n rows for DCT1DImpl<$n>"
                );

                let mut tmp_buffer = [[0.0f32; COLUMNS]; $n];

                // 1. AddReverse
                CoeffBundle::<$nhalf, COLUMNS>::add_reverse::<D, COLUMNS>(
                    d,
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[0..$nhalf],
                );

                // 2. First Recursive Call (do_dct) - first half
                maybe_call!(
                    d,
                    $nhalf,
                    DCT1DImpl::<$nhalf>::do_dct::<D, COLUMNS>(d, &mut tmp_buffer[0..$nhalf],)
                );

                // 3. SubReverse
                CoeffBundle::<$nhalf, COLUMNS>::sub_reverse::<D, COLUMNS>(
                    d,
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[$nhalf..$n],
                );

                // 4. Multiply
                CoeffBundle::<$n, COLUMNS>::multiply::<D, COLUMNS>(d, &mut tmp_buffer);

                // 5. Second Recursive Call (do_dct) - second half
                maybe_call!(
                    d,
                    $nhalf,
                    DCT1DImpl::<$nhalf>::do_dct::<D, COLUMNS>(d, &mut tmp_buffer[$nhalf..$n],)
                );

                // 6. B
                CoeffBundle::<$nhalf, COLUMNS>::b::<D, COLUMNS>(d, &mut tmp_buffer[$nhalf..$n]);

                // 7. InverseEvenOdd
                CoeffBundle::<$n, COLUMNS>::inverse_even_odd::<D, COLUMNS>(d, &tmp_buffer, data);
            }
        }
    };
}
define_dct_1d!(4, 2);
define_dct_1d!(8, 4);
define_dct_1d!(16, 8);
define_dct_1d!(32, 16);
define_dct_1d!(64, 32);
define_dct_1d!(128, 64);
define_dct_1d!(256, 128);

macro_rules! define_idct_1d {
    ($n:literal, $nhalf: tt) => {
        impl CoeffBundle<$nhalf, 1> {
            #[inline(always)]
            fn b_transpose<D: SimdDescriptor>(
                d: D,
                coeff: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray; $nhalf],
            ) {
                for i in (1..$nhalf).rev() {
                    let coeffs_curr = D::F32Vec::load_array(d, &coeff[i]);
                    let coeffs_prev = D::F32Vec::load_array(d, &coeff[i - 1]);
                    (coeffs_curr + coeffs_prev).store_array(&mut coeff[i]);
                }
                let sqrt2 = D::F32Vec::splat(d, SQRT_2 as f32);
                let coeffs = D::F32Vec::load_array(d, &coeff[0]);
                (coeffs * sqrt2).store_array(&mut coeff[0]);
            }
        }

        impl CoeffBundle<$n, 1> {
            #[inline(always)]
            fn forward_even_odd<D: SimdDescriptor>(
                d: D,
                a_in: &[<D::F32Vec as F32SimdVec>::UnderlyingArray; $n],
                a_out: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray; $n],
            ) {
                // Should be optimized away by the compiler.
                for i in 0..($nhalf) {
                    D::F32Vec::load_array(d, &a_in[2 * i]).store_array(&mut a_out[i]);
                }
                for i in ($nhalf)..$n {
                    D::F32Vec::load_array(d, &a_in[2 * (i - $nhalf) + 1])
                        .store_array(&mut a_out[i]);
                }
            }

            #[inline(always)]
            fn multiply_and_add<D: SimdDescriptor>(
                d: D,
                coeff: &[<D::F32Vec as F32SimdVec>::UnderlyingArray; $n],
                out: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray; $n],
            ) {
                for i in 0..($nhalf) {
                    let mul = D::F32Vec::splat(d, WcMultipliers::<$n>::K_MULTIPLIERS[i]);
                    let in1 = D::F32Vec::load_array(d, &coeff[i]);
                    let in2 = D::F32Vec::load_array(d, &coeff[$nhalf + i]);
                    in2.mul_add(mul, in1).store_array(&mut out[i]);
                    in2.neg_mul_add(mul, in1).store_array(&mut out[$n - i - 1]);
                }
            }
        }

        impl IDCT1D for IDCT1DImpl<$n> {
            const SIZE: usize = $n;
            #[inline(always)]
            fn do_idct<D: SimdDescriptor, VD: SimdDescriptor>(
                d: D,
                vd: VD,
                data: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
                scratch: &mut [<VD::F32Vec as F32SimdVec>::UnderlyingArray],
            ) {
                const { assert!($nhalf * 2 == $n, "N/2 * 2 must be N") }

                let data: &mut [_; $n] = data.try_into().unwrap();
                let scratch: &mut [_; $n] = (&mut scratch[..$n]).try_into().unwrap();

                CoeffBundle::<$n, 1>::forward_even_odd(vd, data, scratch);

                maybe_call!(
                    d,
                    $nhalf,
                    IDCT1DImpl::<$nhalf>::do_idct(d, vd, &mut scratch[..$nhalf], data)
                );
                let second_half: &mut [_; $nhalf] = (&mut scratch[$nhalf..]).try_into().unwrap();
                CoeffBundle::<$nhalf, 1>::b_transpose(vd, &mut *second_half);
                maybe_call!(
                    d,
                    $nhalf,
                    IDCT1DImpl::<$nhalf>::do_idct::<D, VD>(d, vd, second_half, data)
                );
                CoeffBundle::<$n, 1>::multiply_and_add(vd, scratch, data);
            }
            impl_idct_wrapper!();
        }
    };
}
define_idct_1d!(4, 2);
define_idct_1d!(8, 4);
define_idct_1d!(16, 8);
define_idct_1d!(32, 16);
define_idct_1d!(64, 32);
define_idct_1d!(128, 64);
define_idct_1d!(256, 128);

#[inline(always)]
pub fn dct2d<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    data: &mut [f32],
    scratch: &mut [f32],
) where
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // 1. Row transforms.
    d.call(|d| {
        let temp_rows = data.as_chunks_mut::<COLS>().0;
        DCT1DImpl::<ROWS>::do_dct::<D, COLS>(d, temp_rows);
    });

    // 2. Transpose.
    let temp_cols_slice = &mut scratch[..ROWS * COLS];
    d.transpose(data, temp_cols_slice, ROWS, COLS);

    // 3. Column transforms.
    d.call(|d| {
        let temp_cols = temp_cols_slice.as_chunks_mut::<ROWS>().0;
        DCT1DImpl::<COLS>::do_dct::<D, ROWS>(d, temp_cols);
    });

    // 4. Transpose back.
    d.transpose(temp_cols_slice, data, COLS, ROWS);
}

#[inline(always)]
pub fn idct2d<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    data: &mut [f32],
    scratch: &mut [f32],
) where
    IDCT1DImpl<ROWS>: IDCT1D,
    IDCT1DImpl<COLS>: IDCT1D,
{
    // TODO(veluca): can we avoid the copy_from_slice below? This would need some kind of
    // transpose_in_place.
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // Column IDCTs
    let (transposed_input, scratch) = scratch.split_at_mut(ROWS * COLS);
    if ROWS < COLS {
        d.transpose(data, transposed_input, ROWS, COLS);
    } else {
        transposed_input.copy_from_slice(data);
    }

    IDCT1DImpl::<COLS>::idct_wrapper(d, transposed_input, scratch, ROWS);

    // 2. Transpose back
    d.transpose(transposed_input, data, COLS, ROWS);

    // 3. Row IDCT
    IDCT1DImpl::<ROWS>::idct_wrapper(d, data, scratch, COLS);
}

#[inline(always)]
pub fn compute_scaled_dct<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    mut from: [[f32; COLS]; ROWS],
    to: &mut [f32],
) where
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    // Row transforms
    d.call(|d| {
        DCT1DImpl::<ROWS>::do_dct::<D, COLS>(d, &mut from);
    });

    // Transpose
    let mut transposed_dct_buffer = [[0.0; ROWS]; COLS];
    d.transpose(
        from.as_flattened(),
        transposed_dct_buffer.as_flattened_mut(),
        ROWS,
        COLS,
    );

    // Column transforms
    d.call(|d| {
        DCT1DImpl::<COLS>::do_dct::<D, ROWS>(d, &mut transposed_dct_buffer);
    });

    // Normalization and output
    let norm_factor_scalar = 1.0 / (ROWS * COLS) as f32;

    if ROWS >= COLS {
        // For small sizes (≤64 elements), always use scalar normalization
        // Even on SIMD descriptors, scalar is faster due to better compiler optimization
        if ROWS * COLS <= 64 {
            for (dst, &src) in to[..ROWS * COLS]
                .iter_mut()
                .zip(transposed_dct_buffer.as_flattened())
            {
                *dst = src * norm_factor_scalar;
            }
        } else {
            // For large sizes, use SIMD normalization with runtime loop
            let normalization_factor = D::F32Vec::splat(d, norm_factor_scalar);
            if ROWS * COLS < D::F32Vec::LEN {
                let coeffs =
                    D::F32Vec::load_partial(d, ROWS * COLS, transposed_dct_buffer.as_flattened());
                (coeffs * normalization_factor).store_partial(ROWS * COLS, to);
            } else {
                assert_eq!(ROWS * COLS % D::F32Vec::LEN, 0);
                for i in (0..ROWS * COLS).step_by(D::F32Vec::LEN) {
                    let coeffs =
                        D::F32Vec::load(d, transposed_dct_buffer.as_flattened()[i..].as_ref());
                    (coeffs * normalization_factor).store(to[i..].as_mut());
                }
            }
        }
    } else {
        d.transpose(
            transposed_dct_buffer.as_flattened(),
            to[..ROWS * COLS].as_mut(),
            COLS,
            ROWS,
        );

        // For small sizes (≤64 elements), always use scalar normalization
        // Even on SIMD descriptors, scalar is faster due to better compiler optimization
        if ROWS * COLS <= 64 {
            for item in to.iter_mut().take(ROWS * COLS) {
                *item *= norm_factor_scalar;
            }
        } else {
            // For large sizes, use SIMD normalization with runtime loop
            let normalization_factor = D::F32Vec::splat(d, norm_factor_scalar);
            if ROWS * COLS < D::F32Vec::LEN {
                let coeffs = D::F32Vec::load_partial(d, ROWS * COLS, to);
                (coeffs * normalization_factor).store_partial(ROWS * COLS, to);
            } else {
                assert_eq!(ROWS * COLS % D::F32Vec::LEN, 0);
                for i in (0..ROWS * COLS).step_by(D::F32Vec::LEN) {
                    let coeffs = D::F32Vec::load(d, to[i..].as_ref());
                    (coeffs * normalization_factor).store(to[i..].as_mut());
                }
            }
        }
    }
}
