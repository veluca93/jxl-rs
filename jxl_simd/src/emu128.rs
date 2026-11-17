// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]

use std::array::from_fn;

use crate::sealed::{
    ArithBackend, BytesBackend, CompareBackend, DescriptorSealed, F32ExtraBackend, F64ExtraBackend,
    FloatBackend, I16ExtraBackend, I32ExtraBackend, I64ExtraBackend, IntBitBackend, LaneBackend,
    MaskBackend, MulBackend, SatBackend, SignedBackend, U8ExtraBackend,
};
use crate::{SimdDescriptor, SimdDescriptor128, W8, W16, W32, W64, f16, generic_ops};

/// Portable 128-bit SIMD descriptor backed by plain Rust arrays.
#[derive(Clone, Copy, Debug)]
pub struct Emu128Descriptor(());

impl Emu128Descriptor {
    /// Constructs the emulated 128-bit descriptor without runtime CPU checks.
    ///
    /// # Safety
    /// Always safe to call on any CPU; kept `unsafe` for symmetry with hardware descriptors.
    #[allow(unsafe_code)]
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl SimdDescriptor128 for Emu128Descriptor {}

impl SimdDescriptor for Emu128Descriptor {
    const BYTES: usize = 16;
    const IS_EMULATED: bool = true;
    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline(always)]
    fn new() -> Option<Self> {
        Some(Self(()))
    }

    #[inline(always)]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    #[inline(always)]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    #[inline(always)]
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }

    #[inline(always)]
    fn call_inlined<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

impl BytesBackend for Emu128Descriptor {
    type RawBytes = [u8; 16];
}

impl DescriptorSealed for Emu128Descriptor {
    type RawBf16Table8 = [u8; 16];

    #[inline(always)]
    fn prepare_table_bf16_8(self, table: &[f32; 8]) -> [u8; 16] {
        generic_ops::prepare_table_bf16_8_128(self, table).0
    }

    #[inline(always)]
    fn table_lookup_bf16_8(self, table: [u8; 16], indices: [i32; 4]) -> [f32; 4] {
        let idx = indices.map(|v| v as usize);
        from_fn(|i| {
            let b = [table[2 * idx[i]], table[2 * idx[i] + 1]];
            f32::from_bits((u16::from_ne_bytes(b) as u32) << 16)
        })
    }
}

macro_rules! impl_emu_mask {
    ($w:ty, $n:literal) => {
        impl MaskBackend<$w> for Emu128Descriptor {
            type MaskRaw = [bool; $n];
            #[inline(always)]
            fn mask_and(self, a: [bool; $n], b: [bool; $n]) -> [bool; $n] {
                from_fn(|i| a[i] & b[i])
            }
            #[inline(always)]
            fn mask_or(self, a: [bool; $n], b: [bool; $n]) -> [bool; $n] {
                from_fn(|i| a[i] | b[i])
            }
            #[inline(always)]
            fn mask_xor(self, a: [bool; $n], b: [bool; $n]) -> [bool; $n] {
                from_fn(|i| a[i] ^ b[i])
            }
            #[inline(always)]
            fn mask_not(self, a: [bool; $n]) -> [bool; $n] {
                a.map(|x| !x)
            }
            #[inline(always)]
            fn mask_andnot(self, a: [bool; $n], b: [bool; $n]) -> [bool; $n] {
                from_fn(|i| !a[i] & b[i])
            }
            #[inline(always)]
            fn mask_all(self, a: [bool; $n]) -> bool {
                a.into_iter().all(|x| x)
            }
            #[inline(always)]
            fn mask_any(self, a: [bool; $n]) -> bool {
                a.into_iter().any(|x| x)
            }
        }
    };
}
impl_emu_mask!(W8, 16);
impl_emu_mask!(W16, 8);
impl_emu_mask!(W32, 4);
impl_emu_mask!(W64, 2);

macro_rules! impl_emu_lane {
    ($t:ty, $n:literal, $z:expr, $to_b:expr, $from_b:expr) => {
        impl LaneBackend<$t> for Emu128Descriptor {
            type Raw = [$t; $n];
            type Array = [$t; $n];
            #[inline(always)]
            fn zero(self) -> [$t; $n] {
                [$z; $n]
            }
            #[inline(always)]
            fn splat(self, v: $t) -> [$t; $n] {
                [v; $n]
            }
            #[inline(always)]
            fn extract_lane0(self, a: [$t; $n]) -> $t {
                a[0]
            }
            #[inline(always)]
            fn load(self, mem: &[$t]) -> [$t; $n] {
                *mem.first_chunk::<$n>().unwrap()
            }
            #[inline(always)]
            fn store(self, a: [$t; $n], dest: &mut [$t]) {
                *dest.first_chunk_mut::<$n>().unwrap() = a;
            }
            #[inline(always)]
            fn load_array(self, mem: &[$t; $n]) -> [$t; $n] {
                *mem
            }
            #[inline(always)]
            fn store_array(self, a: [$t; $n], dest: &mut [$t; $n]) {
                *dest = a;
            }
            #[inline(always)]
            fn zero_array() -> [$t; $n] {
                [$z; $n]
            }
            #[inline(always)]
            fn slice_as_array(slice: &[$t]) -> &[[$t; $n]] {
                generic_ops::chunks(slice)
            }
            #[inline(always)]
            fn slice_as_array_mut(slice: &mut [$t]) -> &mut [[$t; $n]] {
                generic_ops::chunks_mut(slice)
            }
            #[inline(always)]
            fn to_raw_bytes(self, a: [$t; $n]) -> [u8; 16] {
                ($to_b)(a)
            }
            #[inline(always)]
            fn from_raw_bytes(self, b: [u8; 16]) -> [$t; $n] {
                ($from_b)(b)
            }
            #[inline(always)]
            fn interleave_lo(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| if i % 2 == 0 { a[i / 2] } else { b[i / 2] })
            }
            #[inline(always)]
            fn interleave_hi(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| {
                    let j = i / 2 + $n / 2;
                    if i % 2 == 0 { a[j] } else { b[j] }
                })
            }
            #[inline(always)]
            fn store_interleaved_2(self, a: [$t; $n], b: [$t; $n], dest: &mut [$t]) {
                assert!(dest.len() >= 2 * $n);
                for i in 0..$n {
                    dest[2 * i] = a[i];
                    dest[2 * i + 1] = b[i];
                }
            }
            #[inline(always)]
            fn store_interleaved_3(self, a: [$t; $n], b: [$t; $n], c: [$t; $n], dest: &mut [$t]) {
                assert!(dest.len() >= 3 * $n);
                for i in 0..$n {
                    dest[3 * i] = a[i];
                    dest[3 * i + 1] = b[i];
                    dest[3 * i + 2] = c[i];
                }
            }
            #[inline(always)]
            fn store_interleaved_4(
                self,
                a: [$t; $n],
                b: [$t; $n],
                c: [$t; $n],
                e: [$t; $n],
                dest: &mut [$t],
            ) {
                assert!(dest.len() >= 4 * $n);
                for i in 0..$n {
                    dest[4 * i] = a[i];
                    dest[4 * i + 1] = b[i];
                    dest[4 * i + 2] = c[i];
                    dest[4 * i + 3] = e[i];
                }
            }
        }
    };
}

macro_rules! bytes_conv {
    ($t:ty, $n:literal, $sz:literal) => {
        (
            |a: [$t; $n]| {
                let mut out = [0u8; 16];
                for i in 0..$n {
                    out[i * $sz..(i + 1) * $sz].copy_from_slice(&a[i].to_ne_bytes());
                }
                out
            },
            |b: [u8; 16]| {
                from_fn::<$t, $n, _>(|i| {
                    <$t>::from_ne_bytes(b[i * $sz..(i + 1) * $sz].try_into().unwrap())
                })
            },
        )
    };
}

impl_emu_lane!(
    f32,
    4,
    0.0,
    |a: [f32; 4]| {
        let mut out = [0u8; 16];
        for i in 0..4 {
            out[i * 4..i * 4 + 4].copy_from_slice(&a[i].to_ne_bytes());
        }
        out
    },
    |b: [u8; 16]| from_fn(|i| f32::from_ne_bytes([
        b[4 * i],
        b[4 * i + 1],
        b[4 * i + 2],
        b[4 * i + 3]
    ]))
);
impl_emu_lane!(
    f64,
    2,
    0.0,
    |a: [f64; 2]| {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&a[0].to_ne_bytes());
        out[8..].copy_from_slice(&a[1].to_ne_bytes());
        out
    },
    |b: [u8; 16]| [
        f64::from_ne_bytes(b[..8].try_into().unwrap()),
        f64::from_ne_bytes(b[8..].try_into().unwrap()),
    ]
);
impl_emu_lane!(u8, 16, 0, |a| a, |b| b);
impl_emu_lane!(
    i8,
    16,
    0,
    |a: [i8; 16]| a.map(|x| x as u8),
    |b: [u8; 16]| b.map(|x| x as i8)
);
impl_emu_lane!(
    i16,
    8,
    0,
    bytes_conv!(i16, 8, 2).0,
    bytes_conv!(i16, 8, 2).1
);
impl_emu_lane!(
    u16,
    8,
    0,
    bytes_conv!(u16, 8, 2).0,
    bytes_conv!(u16, 8, 2).1
);
impl_emu_lane!(
    i32,
    4,
    0,
    bytes_conv!(i32, 4, 4).0,
    bytes_conv!(i32, 4, 4).1
);
impl_emu_lane!(
    u32,
    4,
    0,
    bytes_conv!(u32, 4, 4).0,
    bytes_conv!(u32, 4, 4).1
);
impl_emu_lane!(
    i64,
    2,
    0,
    bytes_conv!(i64, 2, 8).0,
    bytes_conv!(i64, 2, 8).1
);
impl_emu_lane!(
    u64,
    2,
    0,
    bytes_conv!(u64, 2, 8).0,
    bytes_conv!(u64, 2, 8).1
);

macro_rules! impl_emu_compare {
    ($t:ty, $n:literal, $z:expr, $min:expr, $max:expr) => {
        impl CompareBackend<$t> for Emu128Descriptor {
            #[inline(always)]
            fn eq(self, a: [$t; $n], b: [$t; $n]) -> [bool; $n] {
                from_fn(|i| a[i] == b[i])
            }
            #[inline(always)]
            fn gt(self, a: [$t; $n], b: [$t; $n]) -> [bool; $n] {
                from_fn(|i| a[i] > b[i])
            }
            #[inline(always)]
            fn eq_zero(self, a: [$t; $n]) -> [bool; $n] {
                a.map(|x| x == $z)
            }
            #[inline(always)]
            fn min(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| ($min)(a[i], b[i]))
            }
            #[inline(always)]
            fn max(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| ($max)(a[i], b[i]))
            }
            #[inline(always)]
            fn mask_select(self, m: [bool; $n], t: [$t; $n], f: [$t; $n]) -> [$t; $n] {
                from_fn(|i| if m[i] { t[i] } else { f[i] })
            }
            #[inline(always)]
            fn mask_select_zero(self, m: [bool; $n], t: [$t; $n]) -> [$t; $n] {
                from_fn(|i| if m[i] { t[i] } else { $z })
            }
            #[inline(always)]
            fn mask_zero_select(self, m: [bool; $n], f: [$t; $n]) -> [$t; $n] {
                from_fn(|i| if m[i] { $z } else { f[i] })
            }
        }
    };
}
impl_emu_compare!(f32, 4, 0.0, f32::min, f32::max);
impl_emu_compare!(f64, 2, 0.0, f64::min, f64::max);
impl_emu_compare!(i8, 16, 0, Ord::min, Ord::max);
impl_emu_compare!(u8, 16, 0, Ord::min, Ord::max);
impl_emu_compare!(i16, 8, 0, Ord::min, Ord::max);
impl_emu_compare!(u16, 8, 0, Ord::min, Ord::max);
impl_emu_compare!(i32, 4, 0, Ord::min, Ord::max);
impl_emu_compare!(u32, 4, 0, Ord::min, Ord::max);
impl_emu_compare!(i64, 2, 0, Ord::min, Ord::max);
impl_emu_compare!(u64, 2, 0, Ord::min, Ord::max);

macro_rules! impl_emu_float_arith_signed {
    ($t:ty, $n:literal) => {
        impl ArithBackend<$t> for Emu128Descriptor {
            #[inline(always)]
            fn add(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i] + b[i])
            }
            #[inline(always)]
            fn sub(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i] - b[i])
            }
        }
        impl MulBackend<$t> for Emu128Descriptor {
            #[inline(always)]
            fn mul(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i] * b[i])
            }
        }
        impl SignedBackend<$t> for Emu128Descriptor {
            #[inline(always)]
            fn neg(self, a: [$t; $n]) -> [$t; $n] {
                a.map(|x| -x)
            }
            #[inline(always)]
            fn abs(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::abs)
            }
            #[inline(always)]
            fn lt_zero(self, a: [$t; $n]) -> [bool; $n] {
                a.map(|x| x < 0.0)
            }
        }
        impl FloatBackend<$t> for Emu128Descriptor {
            #[inline(always)]
            fn div(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i] / b[i])
            }
            #[inline(always)]
            fn mul_add(self, a: [$t; $n], m: [$t; $n], add: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i] * m[i] + add[i])
            }
            #[inline(always)]
            fn neg_mul_add(self, a: [$t; $n], m: [$t; $n], add: [$t; $n]) -> [$t; $n] {
                from_fn(|i| -(a[i] * m[i]) + add[i])
            }
            #[inline(always)]
            fn sqrt(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::sqrt)
            }
            #[inline(always)]
            fn floor(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::floor)
            }
            #[inline(always)]
            fn ceil(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::ceil)
            }
            #[inline(always)]
            fn round(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::round_ties_even)
            }
            #[inline(always)]
            fn trunc(self, a: [$t; $n]) -> [$t; $n] {
                a.map(<$t>::trunc)
            }
            #[inline(always)]
            fn copysign(self, a: [$t; $n], sign: [$t; $n]) -> [$t; $n] {
                from_fn(|i| a[i].copysign(sign[i]))
            }
        }
    };
}
impl_emu_float_arith_signed!(f32, 4);
impl_emu_float_arith_signed!(f64, 2);

macro_rules! impl_emu_int_ops {
    ($(($t:ty, $n:literal)),*) => {
        $(
            impl ArithBackend<$t> for Emu128Descriptor {
                #[inline(always)] fn add(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i].wrapping_add(b[i])) }
                #[inline(always)] fn sub(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i].wrapping_sub(b[i])) }
            }
            impl MulBackend<$t> for Emu128Descriptor {
                #[inline(always)] fn mul(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i].wrapping_mul(b[i])) }
            }
            impl IntBitBackend<$t> for Emu128Descriptor {
                #[inline(always)] fn bit_and(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i] & b[i]) }
                #[inline(always)] fn bit_or(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i] | b[i]) }
                #[inline(always)] fn bit_xor(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i] ^ b[i]) }
                #[inline(always)] fn bit_not(self, a: [$t; $n]) -> [$t; $n] { a.map(|x| !x) }
                #[inline(always)] fn andnot(self, na: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| !na[i] & b[i]) }
                #[inline(always)] fn shl<const N: u32>(self, a: [$t; $n]) -> [$t; $n] { a.map(|x| x.wrapping_shl(N)) }
                #[inline(always)] fn shr<const N: u32>(self, a: [$t; $n]) -> [$t; $n] { a.map(|x| x.wrapping_shr(N)) }
            }
        )*
    };
}
impl_emu_int_ops!(
    (i8, 16),
    (u8, 16),
    (i16, 8),
    (u16, 8),
    (i32, 4),
    (u32, 4),
    (i64, 2),
    (u64, 2)
);

macro_rules! impl_emu_signed_int {
    ($(($t:ty, $n:literal)),*) => {
        $(impl SignedBackend<$t> for Emu128Descriptor {
            #[inline(always)] fn neg(self, a: [$t; $n]) -> [$t; $n] { a.map(<$t>::wrapping_neg) }
            #[inline(always)] fn abs(self, a: [$t; $n]) -> [$t; $n] { a.map(<$t>::wrapping_abs) }
            #[inline(always)] fn lt_zero(self, a: [$t; $n]) -> [bool; $n] { a.map(|x| x < 0) }
        })*
    };
}
impl_emu_signed_int!((i8, 16), (i16, 8), (i32, 4), (i64, 2));

macro_rules! impl_emu_sat {
    ($(($t:ty, $n:literal)),*) => {
        $(impl SatBackend<$t> for Emu128Descriptor {
            #[inline(always)] fn sat_add(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i].saturating_add(b[i])) }
            #[inline(always)] fn sat_sub(self, a: [$t; $n], b: [$t; $n]) -> [$t; $n] { from_fn(|i| a[i].saturating_sub(b[i])) }
        })*
    };
}
impl_emu_sat!((i8, 16), (u8, 16), (i16, 8), (u16, 8));

impl F32ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn f32_as_i32(self, a: [f32; 4]) -> [i32; 4] {
        a.map(|x| x as i32)
    }
    #[inline(always)]
    fn f32_round_store_u8(self, a: [f32; 4], dest: &mut [u8]) {
        *dest.first_chunk_mut::<4>().unwrap() = a.map(|x| x.round_ties_even() as u8);
    }
    #[inline(always)]
    fn f32_round_store_u16(self, a: [f32; 4], dest: &mut [u16]) {
        *dest.first_chunk_mut::<4>().unwrap() = a.map(|x| x.round_ties_even() as u16);
    }
    #[inline(always)]
    fn f32_load_f16_bits(self, mem: &[u16]) -> [f32; 4] {
        mem.first_chunk::<4>()
            .unwrap()
            .map(|bits| f16::from_bits(bits).to_f32())
    }
    #[inline(always)]
    fn f32_store_f16_bits(self, a: [f32; 4], dest: &mut [u16]) {
        *dest.first_chunk_mut::<4>().unwrap() = a.map(|x| f16::from_f32(x).to_bits());
    }
}

impl F64ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn f64_as_i64(self, a: [f64; 2]) -> [i64; 2] {
        a.map(|x| x as i64)
    }
    #[inline(always)]
    fn f64_from_f32_lo(self, a: [f32; 4]) -> [f64; 2] {
        [a[0] as f64, a[1] as f64]
    }
    #[inline(always)]
    fn f64_from_f32_hi(self, a: [f32; 4]) -> [f64; 2] {
        [a[2] as f64, a[3] as f64]
    }
    #[inline(always)]
    fn f64_to_f32(self, lo: [f64; 2], hi: [f64; 2]) -> [f32; 4] {
        [lo[0] as f32, lo[1] as f32, hi[0] as f32, hi[1] as f32]
    }
}

impl I32ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn i32_as_f32(self, a: [i32; 4]) -> [f32; 4] {
        a.map(|x| x as f32)
    }
    #[inline(always)]
    fn i32_load_from_i16(self, mem: &[i16]) -> [i32; 4] {
        mem.first_chunk::<4>().unwrap().map(|x| x as i32)
    }
    #[inline(always)]
    fn i32_load_from_u16(self, mem: &[u16]) -> [i32; 4] {
        mem.first_chunk::<4>().unwrap().map(|x| x as i32)
    }
    #[inline(always)]
    fn i32_store_u16(self, a: [i32; 4], dest: &mut [u16]) {
        *dest.first_chunk_mut::<4>().unwrap() = a.map(|x| x as u16);
    }
    #[inline(always)]
    fn i32_store_u8(self, a: [i32; 4], dest: &mut [u8]) {
        *dest.first_chunk_mut::<4>().unwrap() = a.map(|x| x as u8);
    }
    #[inline(always)]
    fn i32_mul_wide_take_high(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        from_fn(|i| ((a[i] as i64 * b[i] as i64) >> 32) as i32)
    }
}

impl I64ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn i64_as_f64(self, a: [i64; 2]) -> [f64; 2] {
        a.map(|x| x as f64)
    }
}

impl I16ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn i16_store_u8(self, a: [i16; 8], dest: &mut [u8]) {
        *dest.first_chunk_mut::<8>().unwrap() = a.map(|x| x as u8);
    }
    #[inline(always)]
    fn i16_mul_wide_take_high(self, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        from_fn(|i| ((a[i] as i32 * b[i] as i32) >> 16) as i16)
    }
}

impl U8ExtraBackend for Emu128Descriptor {
    #[inline(always)]
    fn u8_swizzle_128_or_zero(self, a: [u8; 16], indices: [u8; 16]) -> [u8; 16] {
        indices.map(|idx| if idx < 16 { a[idx as usize] } else { 0 })
    }
}
