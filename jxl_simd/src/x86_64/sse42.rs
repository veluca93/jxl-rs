// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::super::{F32SimdVec, I32SimdVec, ScalarDescriptor, SimdDescriptor, SimdMask};
use std::{
    arch::x86_64::{
        __m128, __m128i, _mm_abs_epi32, _mm_add_epi32, _mm_add_ps, _mm_andnot_si128, _mm_blendv_ps,
        _mm_castps_si128, _mm_castsi128_ps, _mm_cmpgt_epi32, _mm_cvtepi32_ps, _mm_div_ps,
        _mm_loadu_ps, _mm_loadu_si128, _mm_maskload_ps, _mm_maskstore_ps, _mm_max_ps,
        _mm_mul_epi32, _mm_mul_ps, _mm_set1_epi32, _mm_set1_ps, _mm_storeu_ps, _mm_sub_epi32,
        _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps, _mm_xor_si128,
    },
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

// Safety invariant: this type is only ever constructed if sse4.2 is available.
#[derive(Clone, Copy, Debug)]
pub struct Sse42Descriptor(());

impl Sse42Descriptor {
    /// # Safety
    /// The caller must guarantee that the sse4.2 target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Sse42Descriptor {
    #[target_feature(enable = "sse4.2")]
    #[inline]
    pub(super) fn transpose4x4f32(
        self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
    ) {
        assert!(input_stride >= 4);
        assert!(input.len() >= input_stride.checked_mul(3).unwrap().checked_add(4).unwrap());
        assert!(
            output.len()
                >= output_stride
                    .checked_mul(3)
                    .unwrap()
                    .checked_add(4)
                    .unwrap()
        );

        // SAFETY: input is verified to be large enough for this pointer arithmetic.
        let (p0, p1, p2, p3) = unsafe {
            (
                _mm_loadu_ps(input.as_ptr()),
                _mm_loadu_ps(input.as_ptr().add(input_stride)),
                _mm_loadu_ps(input.as_ptr().add(2 * input_stride)),
                _mm_loadu_ps(input.as_ptr().add(3 * input_stride)),
            )
        };

        let q0 = _mm_unpacklo_ps(p0, p2);
        let q1 = _mm_unpacklo_ps(p1, p3);
        let q2 = _mm_unpackhi_ps(p0, p2);
        let q3 = _mm_unpackhi_ps(p1, p3);

        let r0 = _mm_unpacklo_ps(q0, q1);
        let r1 = _mm_unpackhi_ps(q0, q1);
        let r2 = _mm_unpacklo_ps(q2, q3);
        let r3 = _mm_unpackhi_ps(q2, q3);

        // SAFETY: output is verified to be large enough for this pointer arithmetic.
        unsafe {
            _mm_storeu_ps(output.as_mut_ptr(), r0);
            _mm_storeu_ps(output.as_mut_ptr().add(output_stride), r1);
            _mm_storeu_ps(output.as_mut_ptr().add(2 * output_stride), r2);
            _mm_storeu_ps(output.as_mut_ptr().add(3 * output_stride), r3);
        }
    }
}

impl SimdDescriptor for Sse42Descriptor {
    type F32Vec = F32VecSse42;
    type I32Vec = I32VecSse42;
    type Mask = MaskAvx;

    fn maybe_downgrade_256bit(self) -> Option<impl SimdDescriptor> {
        None::<Self>
    }

    fn maybe_downgrade_128bit(self) -> Option<impl SimdDescriptor> {
        None::<Self>
    }

    fn new() -> Option<Self> {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: we just checked sse4.2.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), ROWS * COLS);
        assert_eq!(output.len(), ROWS * COLS);

        if ROWS.is_multiple_of(4) && COLS.is_multiple_of(4) {
            for r in (0..ROWS).step_by(4) {
                let input_row = &input[r * COLS..];
                for c in (0..COLS).step_by(4) {
                    let output_row = &mut output[c * ROWS..];
                    // SAFETY: We know sse4.2 is available from the safety invariant on `self`.
                    unsafe {
                        self.transpose4x4f32(&input_row[c..], COLS, &mut output_row[r..], ROWS);
                    }
                }
            }
        } else {
            let scalar = ScalarDescriptor {};
            scalar.transpose::<ROWS, COLS>(input, output);
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "sse4.2")]
        unsafe fn inner<R>(d: Sse42Descriptor, f: impl FnOnce(Sse42Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees sse4.2.
        unsafe { inner(self, f) }
    }
}

// TODO(veluca): retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_sse42 {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "sse4.2")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff sse42 are available.
            unsafe { inner(self, $($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecSse42(__m128, Sse42Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskAvx(__m128, Sse42Descriptor);

#[target_feature(enable = "sse4.2")]
#[inline]
fn get_partial_mask(size: usize) -> __m128i {
    const MASKS: [i32; 7] = [-1, -1, -1, -1, 0, 0, 0];
    // SAFETY: the pointer arithmetic is safe because:
    // `(size - 1) & 3` is between 0 and 3 (inclusive);
    // `3 - ((size - 1) & 3)` is therefore also between 0 and 3 (inclusive);
    // all starting indices between 0 and 3 into a 7-element array leave at least 4 elements to load.
    unsafe { _mm_loadu_si128(MASKS.as_ptr().add(3 - ((size - 1) & 3)) as *const _) }
}

impl F32SimdVec for F32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know sse4.2 is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm_loadu_ps(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self {
        debug_assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        // Fast path: avoid mask setup overhead when loading full vectors
        // This optimization skips the expensive mask creation and masked load when size == LEN
        if size == Self::LEN {
            return Self::load(d, mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know sse4.2 is available
        // from the safety invariant on `d`.
        Self(
            unsafe { _mm_maskload_ps(mem.as_ptr(), get_partial_mask(size)) },
            d,
        )
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know sse4.2 is available
        // from the safety invariant on `self.1`.
        unsafe { _mm_storeu_ps(mem.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    fn store_partial(&self, size: usize, mem: &mut [f32]) {
        assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        if size == Self::LEN {
            return self.store(mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know sse4.2 is available
        // from the safety invariant on `d`.
        unsafe { _mm_maskstore_ps(mem.as_mut_ptr(), get_partial_mask(size), self.0) }
    }

    fn_sse42!(this: F32VecSse42, fn mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        this * mul + add
    });

    fn_sse42!(this: F32VecSse42, fn neg_mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        add - this * mul
    });

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know sse4.2 is available from the safety invariant on `d`.
        unsafe { Self(_mm_set1_ps(v), d) }
    }

    fn_sse42!(this: F32VecSse42, fn abs() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_andnot_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn neg() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_xor_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn max(other: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_max_ps(this.0, other.0), this.1)
    });
}

impl Add<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn add(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn sub(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn mul(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn div(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn add_assign(rhs: F32VecSse42) {
        this.0 = _mm_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn sub_assign(rhs: F32VecSse42) {
        this.0 = _mm_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn mul_assign(rhs: F32VecSse42) {
        this.0 = _mm_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn div_assign(rhs: F32VecSse42) {
        this.0 = _mm_div_ps(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecSse42(__m128i, Sse42Descriptor);

impl I32SimdVec for I32VecSse42 {
    type Descriptor = Sse42Descriptor;
    type F32Vec = F32VecSse42;
    type Mask = MaskAvx;

    const LEN: usize = 4;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know sse4.2 is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm_loadu_si128(mem.as_ptr() as *const _) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        // SAFETY: We know sse4.2 is available from the safety invariant on `d`.
        unsafe { Self(_mm_set1_epi32(v), d) }
    }

    fn_sse42!(this: I32VecSse42, fn as_f32() -> F32VecSse42 {
        F32VecSse42(_mm_cvtepi32_ps(this.0), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn abs() -> I32VecSse42 {
        I32VecSse42(
            _mm_abs_epi32(
                this.0,
            ),
            this.1)
    });

    fn_sse42!(this: I32VecSse42, fn gt(rhs: I32VecSse42) -> MaskAvx {
        MaskAvx(
            _mm_castsi128_ps(_mm_cmpgt_epi32(this.0, rhs.0)),
            this.1,
        )
    });
}

impl Add<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn add(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_add_epi32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn sub(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_sub_epi32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn mul(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_mul_epi32(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn add_assign(rhs: I32VecSse42) {
        this.0 = _mm_add_epi32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn sub_assign(rhs: I32VecSse42) {
        this.0 = _mm_sub_epi32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn mul_assign(rhs: I32VecSse42) {
        this.0 = _mm_mul_epi32(this.0, rhs.0)
    });
}

impl SimdMask for MaskAvx {
    type Descriptor = Sse42Descriptor;
    type F32Vec = F32VecSse42;
    type I32Vec = I32VecSse42;

    fn_sse42!(this: MaskAvx, fn if_then_else_f32(if_true: F32VecSse42, if_false: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_blendv_ps(if_false.0, if_true.0, this.0), this.1)
    });
}
