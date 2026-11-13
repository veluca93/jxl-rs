// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::aarch64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use crate::unaligned::aarch64 as unaligned;
use crate::{U32SimdVec, U64SimdVec, impl_i16_array_interface};

// Safety invariant: this type is only ever constructed if neon is available.
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(());

impl NeonDescriptor {
    /// # Safety
    /// The caller must guarantee that the "neon" target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

/// Prepared 8-entry BF16 lookup table for NEON.
/// Contains 8 BF16 values packed into 16 bytes (uint8x16_t).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Neon(uint8x16_t);

impl crate::SimdDescriptor128 for NeonDescriptor {}

impl SimdDescriptor for NeonDescriptor {
    #[inline(always)]
    fn as_neon(self) -> Option<Self> {
        Some(self)
    }

    type F32Vec = F32VecNeon;

    type I32Vec = I32VecNeon;

    type I16Vec = I16VecNeon;

    type U64Vec = U64VecNeon;

    type U32Vec = U32VecNeon;

    type U16Vec = U16VecNeon;

    type U8Vec = U8VecNeon;

    type Mask = MaskNeon;
    type Mask16 = Mask16Neon;
    type Bf16Table8 = Bf16Table8Neon;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn new() -> Option<Self> {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: we just checked neon.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    fn maybe_downgrade_256bit(self) -> Self {
        self
    }

    fn maybe_downgrade_128bit(self) -> Self {
        self
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "neon")]
        #[inline(never)]
        unsafe fn inner<R>(d: NeonDescriptor, f: impl FnOnce(NeonDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees neon.
        unsafe { inner(self, f) }
    }
}

// TODO: retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_neon {
    {} => {};
    {$(
        fn $name:ident($this:ident: $self_ty:ty $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty )?
        $body: block
    )*} => {$(
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "neon")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff neon is available.
            unsafe { inner(self, $($arg),*) }
        }
    )*};
    (
        $this:ident: $self_ty:ty,
        fn $name:ident$(<$(const $cg:ident: $cg_ty:ty),* $(,)?>)?($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name$(<$(const $cg: $cg_ty),*>)?(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "neon")]
            #[inline]
            fn inner$(<$(const $cg: $cg_ty),*>)?($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff neon is available.
            unsafe { inner$(::<$($cg),*>)?(self, $($arg),*) }
        }
    };
    (
        assoc fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name($($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "neon")]
            #[inline]
            fn inner($($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: the descriptor or vector argument witnesses that neon is available.
            unsafe { inner($($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(float32x4_t, NeonDescriptor);

impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, F32VecNeon);
    crate::impl_isa_array_conversions!(neon, f32, 4);

    const LEN: usize = 4;

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: f32) -> F32VecNeon {
        F32VecNeon(vdupq_n_f32(v), d)
    });

    fn_neon!(assoc fn zero(d: NeonDescriptor) -> F32VecNeon {
        F32VecNeon(vdupq_n_f32(0.0), d)
    });

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[f32]) -> F32VecNeon {
        F32VecNeon(unaligned::vld1q_f32(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(this: &F32VecNeon, fn store(mem: &mut [f32]) {
        unaligned::vst1q_f32(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon!(assoc fn store_interleaved_2(a: F32VecNeon, b: F32VecNeon, dest: &mut [f32]) {
        unaligned::vst2q_f32(dest.first_chunk_mut().unwrap(), float32x4x2_t(a.0, b.0));
    });

    fn_neon!(assoc fn store_interleaved_3(a: F32VecNeon, b: F32VecNeon, c: F32VecNeon, dest: &mut [f32]) {
        unaligned::vst3q_f32(
            dest.first_chunk_mut().unwrap(),
            float32x4x3_t(a.0, b.0, c.0),
        );
    });

    fn_neon!(assoc fn store_interleaved_4(a: F32VecNeon, b: F32VecNeon, c: F32VecNeon, d: F32VecNeon, dest: &mut [f32]) {
        unaligned::vst4q_f32(
            dest.first_chunk_mut().unwrap(),
            float32x4x4_t(a.0, b.0, c.0, d.0),
        );
    });

    fn_neon!(assoc fn store_interleaved_8(
        a: F32VecNeon,
        b: F32VecNeon,
        c: F32VecNeon,
        d: F32VecNeon,
        e: F32VecNeon,
        f: F32VecNeon,
        g: F32VecNeon,
        h: F32VecNeon,
        dest: &mut [f32],
    ) {
        let [d0, d1, d2, d3, d4, d5, d6, d7, ..] = dest.as_chunks_mut::<4>().0 else {
            panic!("dest too short");
        };
        // NEON doesn't have vst8, so we use manual interleaving
        // For 4-wide vectors, output is 32 elements: [a0,b0,c0,d0,e0,f0,g0,h0, a1,...]

        // Use zip to interleave pairs
        let ae_lo = vzip1q_f32(a.0, e.0); // [a0, e0, a1, e1]
        let ae_hi = vzip2q_f32(a.0, e.0); // [a2, e2, a3, e3]
        let bf_lo = vzip1q_f32(b.0, f.0);
        let bf_hi = vzip2q_f32(b.0, f.0);
        let cg_lo = vzip1q_f32(c.0, g.0);
        let cg_hi = vzip2q_f32(c.0, g.0);
        let dh_lo = vzip1q_f32(d.0, h.0);
        let dh_hi = vzip2q_f32(d.0, h.0);

        // Now interleave ae with bf, and cg with dh
        let aebf_0 = vzip1q_f32(ae_lo, bf_lo); // [a0, b0, e0, f0]
        let aebf_1 = vzip2q_f32(ae_lo, bf_lo); // [a1, b1, e1, f1]
        let aebf_2 = vzip1q_f32(ae_hi, bf_hi);
        let aebf_3 = vzip2q_f32(ae_hi, bf_hi);
        let cgdh_0 = vzip1q_f32(cg_lo, dh_lo); // [c0, d0, g0, h0]
        let cgdh_1 = vzip2q_f32(cg_lo, dh_lo);
        let cgdh_2 = vzip1q_f32(cg_hi, dh_hi);
        let cgdh_3 = vzip2q_f32(cg_hi, dh_hi);

        // Final interleave to get [a0,b0,c0,d0,e0,f0,g0,h0]
        let out0 = vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(aebf_0),
            vreinterpretq_f64_f32(cgdh_0),
        ));
        let out1 = vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(aebf_0),
            vreinterpretq_f64_f32(cgdh_0),
        ));
        let out2 = vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(aebf_1),
            vreinterpretq_f64_f32(cgdh_1),
        ));
        let out3 = vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(aebf_1),
            vreinterpretq_f64_f32(cgdh_1),
        ));
        let out4 = vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(aebf_2),
            vreinterpretq_f64_f32(cgdh_2),
        ));
        let out5 = vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(aebf_2),
            vreinterpretq_f64_f32(cgdh_2),
        ));
        let out6 = vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(aebf_3),
            vreinterpretq_f64_f32(cgdh_3),
        ));
        let out7 = vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(aebf_3),
            vreinterpretq_f64_f32(cgdh_3),
        ));

        unaligned::vst1q_f32(d0, out0);
        unaligned::vst1q_f32(d1, out1);
        unaligned::vst1q_f32(d2, out2);
        unaligned::vst1q_f32(d3, out3);
        unaligned::vst1q_f32(d4, out4);
        unaligned::vst1q_f32(d5, out5);
        unaligned::vst1q_f32(d6, out6);
        unaligned::vst1q_f32(d7, out7);
    });

    fn_neon!(assoc fn load_deinterleaved_2(d: NeonDescriptor, src: &[f32]) -> (F32VecNeon, F32VecNeon) {
        let float32x4x2_t(a, b) = unaligned::vld2q_f32(src.first_chunk().unwrap());
        (F32VecNeon(a, d), F32VecNeon(b, d))
    });

    fn_neon!(assoc fn load_deinterleaved_3(d: NeonDescriptor, src: &[f32]) -> (F32VecNeon, F32VecNeon, F32VecNeon) {
        let float32x4x3_t(a, b, c) = unaligned::vld3q_f32(src.first_chunk().unwrap());
        (F32VecNeon(a, d), F32VecNeon(b, d), F32VecNeon(c, d))
    });

    fn_neon!(assoc fn load_deinterleaved_4(d: NeonDescriptor, src: &[f32]) -> (F32VecNeon, F32VecNeon, F32VecNeon, F32VecNeon) {
        let float32x4x4_t(a, b, c, e) = unaligned::vld4q_f32(src.first_chunk().unwrap());
        (F32VecNeon(a, d), F32VecNeon(b, d), F32VecNeon(c, d), F32VecNeon(e, d))
    });

    crate::impl_f32_array_interface!();

    fn_neon! {
        fn mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmaq_f32(add.0, this.0, mul.0), this.1)
        }

        fn neg_mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmsq_f32(add.0, this.0, mul.0), this.1)
        }

        fn abs(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vabsq_f32(this.0), this.1)
        }

        fn floor(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vrndmq_f32(this.0), this.1)
        }

        fn sqrt(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsqrtq_f32(this.0), this.1)
        }

        fn neg(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vnegq_f32(this.0), this.1)
        }

        fn copysign(this: F32VecNeon, sign: F32VecNeon) -> F32VecNeon {
            F32VecNeon(
                vbslq_f32(vdupq_n_u32(0x8000_0000), sign.0, this.0),
                this.1,
            )
        }

        fn max(this: F32VecNeon, other: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmaxq_f32(this.0, other.0), this.1)
        }

        fn min(this: F32VecNeon, other: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vminq_f32(this.0, other.0), this.1)
        }

        fn gt(this: F32VecNeon, other: F32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_f32(this.0, other.0), this.1)
        }

        fn as_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vcvtq_s32_f32(this.0), this.1)
        }

        fn bitcast_to_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vreinterpretq_s32_f32(this.0), this.1)
        }

        fn round_store_u8(this: F32VecNeon, dest: &mut [u8]) {
            // Round to nearest integer
            let rounded = vrndnq_f32(this.0);
            // Convert to i32, then to u16, then to u8
            let i32s = vcvtq_s32_f32(rounded);
            let u16s = vqmovun_s32(i32s);
            let u8s = vqmovn_u16(vcombine_u16(u16s, u16s));
            // Store lower 4 bytes
            *dest.first_chunk_mut::<4>().unwrap() =
                vget_lane_u32::<0>(vreinterpret_u32_u8(u8s)).to_ne_bytes();
        }

        fn round_store_u16(this: F32VecNeon, dest: &mut [u16]) {
            // Round to nearest integer
            let rounded = vrndnq_f32(this.0);
            // Convert to i32, then to u16
            let i32s = vcvtq_s32_f32(rounded);
            let u16s = vqmovun_s32(i32s);
            // Store 4 u16s (8 bytes)
            unaligned::vst1_u16(dest.first_chunk_mut().unwrap(), u16s);
        }

        fn store_f16_bits(this: F32VecNeon, dest: &mut [u16]) {
            let out = dest.first_chunk_mut().unwrap();
            // Use inline asm because Rust stdarch incorrectly requires fp16 target feature
            // for vcvt_f16_f32 (fixed in https://github.com/rust-lang/stdarch/pull/1978)
            let f16_bits: uint16x4_t;
            // SAFETY: NEON is available (guaranteed by descriptor).
            unsafe {
                std::arch::asm!(
                    "fcvtn {out:v}.4h, {inp:v}.4s",
                    inp = in(vreg) this.0,
                    out = out(vreg) f16_bits,
                    options(pure, nomem, nostack),
                );
            }
            unaligned::vst1_u16(out, f16_bits);
        }
    }

    fn_neon!(assoc fn load_f16_bits(d: NeonDescriptor, mem: &[u16]) -> F32VecNeon {
        let f16_bits = unaligned::vld1_u16(mem.first_chunk().unwrap());
        // Use inline asm because Rust stdarch incorrectly requires fp16 target feature
        // for vcvt_f32_f16 (fixed in https://github.com/rust-lang/stdarch/pull/1978)
        let result: float32x4_t;
        // SAFETY: NEON is available (guaranteed by descriptor).
        unsafe {
            std::arch::asm!(
                "fcvtl {out:v}.4s, {inp:v}.4h",
                inp = in(vreg) f16_bits,
                out = out(vreg) result,
                options(pure, nomem, nostack),
            );
        }
        F32VecNeon(result, d)
    });

    fn_neon!(assoc fn prepare_table_bf16_8(_d: NeonDescriptor, table: &[f32; 8]) -> Bf16Table8Neon {
        let ([lo, hi], []) = table.as_chunks::<4>() else {
            unreachable!()
        };
        let table_lo = unaligned::vld1q_f32(lo);
        let table_hi = unaligned::vld1q_f32(hi);

        // Reinterpret as u32 to extract high 16 bits
        let table_lo_u32 = vreinterpretq_u32_f32(table_lo);
        let table_hi_u32 = vreinterpretq_u32_f32(table_hi);

        // Shift right by 16 AND narrow to 16-bit in one instruction
        let bf16_lo_u16 = vshrn_n_u32::<16>(table_lo_u32);
        let bf16_hi_u16 = vshrn_n_u32::<16>(table_hi_u32);

        // Combine into 8 x u16 = 16 bytes
        let bf16_table_u16 = vcombine_u16(bf16_lo_u16, bf16_hi_u16);
        Bf16Table8Neon(vreinterpretq_u8_u16(bf16_table_u16))
    });

    #[inline(always)]
    fn table_lookup_bf16_8(d: NeonDescriptor, table: Bf16Table8Neon, indices: I32VecNeon) -> Self {
        #[target_feature(enable = "neon")]
        #[inline]
        fn lookup_impl(bf16_table: uint8x16_t, indices: int32x4_t) -> float32x4_t {
            // Build shuffle mask efficiently using arithmetic on 32-bit indices.
            // For each index i (0-7), we need to select bytes [2*i, 2*i+1] from bf16_table
            // and place them in the high 16 bits of each 32-bit f32 lane (bytes 2,3),
            // with bytes 0,1 set to zero (using 0x80 which gives 0 in vqtbl1q).
            //
            // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
            // As a 32-bit value: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
            //                  = 0x8080 | (i << 17) | (i << 25) | (1 << 24)
            //                  = (i << 17) | (i << 25) | 0x01008080
            let indices_u32 = vreinterpretq_u32_s32(indices);
            let shl17 = vshlq_n_u32::<17>(indices_u32);
            let shl25 = vshlq_n_u32::<25>(indices_u32);
            let base = vdupq_n_u32(0x01008080);
            let shuffle_mask = vorrq_u32(vorrq_u32(shl17, shl25), base);

            // Perform the table lookup (out of range indices give 0)
            let result = vqtbl1q_u8(bf16_table, vreinterpretq_u8_u32(shuffle_mask));

            // Result has bf16 in high 16 bits of each 32-bit lane = valid f32
            vreinterpretq_f32_u8(result)
        }
        // SAFETY: neon is available from the safety invariant on the descriptor
        F32VecNeon(unsafe { lookup_impl(table.0, indices.0) }, d)
    }
}

impl Add<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn add(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vaddq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn sub(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsubq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn mul(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmulq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Div<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn div(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vdivq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vaddq_f32(this.0, rhs.0);
        }
    }
}

impl SubAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vsubq_f32(this.0, rhs.0);
        }
    }
}

impl MulAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vmulq_f32(this.0, rhs.0);
        }
    }
}

impl DivAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn div_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vdivq_f32(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(int32x4_t, NeonDescriptor);

impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, I32VecNeon);

    const LEN: usize = 4;

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: i32) -> I32VecNeon {
        I32VecNeon(vdupq_n_s32(v), d)
    });

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[i32]) -> I32VecNeon {
        I32VecNeon(unaligned::vld1q_s32(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(assoc fn load_from_i16(d: NeonDescriptor, mem: &[i16]) -> I32VecNeon {
        I32VecNeon(vmovl_s16(unaligned::vld1_s16(mem.first_chunk().unwrap())), d)
    });

    fn_neon!(assoc fn load_from_u16(d: NeonDescriptor, mem: &[u16]) -> I32VecNeon {
        I32VecNeon(
            vreinterpretq_s32_u32(vmovl_u16(unaligned::vld1_u16(mem.first_chunk().unwrap()))),
            d,
        )
    });

    fn_neon!(this: &I32VecNeon, fn store(mem: &mut [i32]) {
        unaligned::vst1q_s32(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon! {
        fn abs(this: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vabsq_s32(this.0), this.1)
        }

        fn as_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vcvtq_f32_s32(this.0), this.1)
        }

        fn bitcast_to_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vreinterpretq_f32_s32(this.0), this.1)
        }

        fn bitcast_to_u32(this: I32VecNeon) -> U32VecNeon {
            U32VecNeon(vreinterpretq_u32_s32(this.0), this.1)
        }

        fn gt(this: I32VecNeon, other: I32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_s32(this.0, other.0), this.1)
        }

        fn lt_zero(this: I32VecNeon) -> MaskNeon {
            MaskNeon(vcltzq_s32(this.0), this.1)
        }

        fn eq(this: I32VecNeon, other: I32VecNeon) -> MaskNeon {
            MaskNeon(vceqq_s32(this.0, other.0), this.1)
        }

        fn eq_zero(this: I32VecNeon) -> MaskNeon {
            MaskNeon(vceqzq_s32(this.0), this.1)
        }

        fn mul_wide_take_high(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            let l = vmull_s32(vget_low_s32(this.0), vget_low_s32(rhs.0));
            let l = vreinterpretq_s32_s64(l);
            let h = vmull_high_s32(this.0, rhs.0);
            let h = vreinterpretq_s32_s64(h);
            I32VecNeon(vuzp2q_s32(l, h), this.1)
        }
    }

    fn_neon!(this: I32VecNeon, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecNeon {
        I32VecNeon(vshlq_n_s32::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I32VecNeon {
        I32VecNeon(vshrq_n_s32::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn store_u16(dest: &mut [u16]) {
        // vmovn narrows i32 to i16 by taking the lower 16 bits
        let narrowed = vmovn_s32(this.0);
        unaligned::vst1_u16(
            dest.first_chunk_mut().unwrap(),
            vreinterpret_u16_s16(narrowed),
        );
    });

    fn_neon!(this: I32VecNeon, fn store_u8(dest: &mut [u8]) {
        // vmovn narrows i32 -> i16 -> i8
        let narrowed_i16 = vmovn_s32(this.0);
        let combined_i16 = vcombine_s16(narrowed_i16, narrowed_i16);
        let narrowed_i8 = vmovn_s16(combined_i16);
        *dest.first_chunk_mut::<4>().unwrap() =
            vget_lane_u32::<0>(vreinterpret_u32_s8(narrowed_i8)).to_ne_bytes();
    });
}

impl Add<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn add(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vaddq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn sub(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vsubq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn mul(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vmulq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Neg for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn neg(this: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vnegq_s32(this.0), this.1)
        }
    }
}

impl BitAnd<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitand(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vandq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitor(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vorrq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl BitXor<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitxor(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(veorq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vaddq_s32(this.0, rhs.0)
        }
    }
}

impl SubAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vsubq_s32(this.0, rhs.0)
        }
    }
}

impl MulAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vmulq_s32(this.0, rhs.0)
        }
    }
}

impl BitAndAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitand_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vandq_s32(this.0, rhs.0);
        }
    }
}

impl BitOrAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitor_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vorrq_s32(this.0, rhs.0);
        }
    }
}

impl BitXorAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitxor_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = veorq_s32(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecNeon(int16x8_t, NeonDescriptor);

impl I16SimdVec for I16VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, I16VecNeon);
    crate::impl_isa_array_conversions!(neon, i16, 8);

    const LEN: usize = 8;

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[i16]) -> I16VecNeon {
        I16VecNeon(unaligned::vld1q_s16(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(this: &I16VecNeon, fn store(mem: &mut [i16]) {
        unaligned::vst1q_s16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: i16) -> I16VecNeon {
        I16VecNeon(vdupq_n_s16(v), d)
    });

    fn_neon!(assoc fn zero(d: NeonDescriptor) -> I16VecNeon {
        I16VecNeon(vdupq_n_s16(0), d)
    });

    fn_neon! {
        fn abs(this: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vabsq_s16(this.0), this.1)
        }

        fn gt(this: I16VecNeon, other: I16VecNeon) -> Mask16Neon {
            Mask16Neon(vcgtq_s16(this.0, other.0), this.1)
        }

        fn lt_zero(this: I16VecNeon) -> Mask16Neon {
            Mask16Neon(vcltq_s16(this.0, vdupq_n_s16(0)), this.1)
        }

        fn eq(this: I16VecNeon, other: I16VecNeon) -> Mask16Neon {
            Mask16Neon(vceqq_s16(this.0, other.0), this.1)
        }

        fn eq_zero(this: I16VecNeon) -> Mask16Neon {
            Mask16Neon(vceqzq_s16(this.0), this.1)
        }

        fn mul_wide_take_high(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            let l = vmull_s16(vget_low_s16(this.0), vget_low_s16(rhs.0));
            let l = vreinterpretq_s16_s32(l);
            let h = vmull_high_s16(this.0, rhs.0);
            let h = vreinterpretq_s16_s32(h);
            I16VecNeon(vuzp2q_s16(l, h), this.1)
        }
    }

    fn_neon!(this: I16VecNeon, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecNeon {
        I16VecNeon(vshlq_n_s16::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon!(this: I16VecNeon, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> I16VecNeon {
        I16VecNeon(vshrq_n_s16::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon!(this: I16VecNeon, fn bitcast_u16() -> U16VecNeon {
        U16VecNeon(vreinterpretq_u16_s16(this.0), this.1)
    });

    impl_i16_array_interface!();

    fn_neon!(this: I16VecNeon, fn store_u8(dest: &mut [u8]) {
        let narrowed_i8 = vmovn_s16(this.0);
        unaligned::vst1_u8(
            dest.first_chunk_mut().unwrap(),
            vreinterpret_u8_s8(narrowed_i8),
        );
    });
}

impl Add<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn add(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vaddq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn sub(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vsubq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl Neg for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn neg(this: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vnegq_s16(this.0), this.1)
        }
    }
}

impl BitAnd<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn bitand(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vandq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn bitor(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vorrq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl BitXor<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn bitxor(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(veorq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn add_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = vaddq_s16(this.0, rhs.0);
        }
    }
}

impl SubAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = vsubq_s16(this.0, rhs.0);
        }
    }
}

impl BitAndAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn bitand_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = vandq_s16(this.0, rhs.0);
        }
    }
}

impl BitOrAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn bitor_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = vorrq_s16(this.0, rhs.0);
        }
    }
}

impl BitXorAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn bitxor_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = veorq_s16(this.0, rhs.0);
        }
    }
}

impl Mul<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;
    fn_neon! {
        fn mul(this: I16VecNeon, rhs: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vmulq_s16(this.0, rhs.0), this.1)
        }
    }
}

impl MulAssign<I16VecNeon> for I16VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut I16VecNeon, rhs: I16VecNeon) {
            this.0 = vmulq_s16(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecNeon(uint32x4_t, NeonDescriptor);

impl U32SimdVec for U32VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, U32VecNeon);

    const LEN: usize = 4;

    fn_neon! {
        fn bitcast_to_i32(this: U32VecNeon) -> I32VecNeon {
            I32VecNeon(vreinterpretq_s32_u32(this.0), this.1)
        }
    }

    fn_neon!(this: U32VecNeon, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U32VecNeon {
        U32VecNeon(vshrq_n_u32::<AMOUNT_I>(this.0), this.1)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecNeon(uint64x2_t, NeonDescriptor);

impl U64SimdVec for U64VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, U64VecNeon);

    const LEN: usize = 2;

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: u64) -> U64VecNeon {
        U64VecNeon(vdupq_n_u64(v), d)
    });

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[u64]) -> U64VecNeon {
        U64VecNeon(unaligned::vld1q_u64(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(this: &U64VecNeon, fn store(mem: &mut [u64]) {
        unaligned::vst1q_u64(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon!(this: U64VecNeon, fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecNeon {
        U64VecNeon(vshlq_n_u64::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon!(this: U64VecNeon, fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>() -> U64VecNeon {
        U64VecNeon(vshrq_n_u64::<AMOUNT_I>(this.0), this.1)
    });

    fn_neon! {
        fn bitcast_to_u32(this: U64VecNeon) -> U32VecNeon {
            U32VecNeon(vreinterpretq_u32_u64(this.0), this.1)
        }
    }
}

impl Add<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;
    fn_neon! {
        fn add(this: U64VecNeon, rhs: U64VecNeon) -> U64VecNeon {
            U64VecNeon(vaddq_u64(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<U64VecNeon> for U64VecNeon {
    fn_neon! {
        fn add_assign(this: &mut U64VecNeon, rhs: U64VecNeon) {
            this.0 = vaddq_u64(this.0, rhs.0);
        }
    }
}

impl BitAnd<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;
    fn_neon! {
        fn bitand(this: U64VecNeon, rhs: U64VecNeon) -> U64VecNeon {
            U64VecNeon(vandq_u64(this.0, rhs.0), this.1)
        }
    }
}

impl BitAndAssign<U64VecNeon> for U64VecNeon {
    fn_neon! {
        fn bitand_assign(this: &mut U64VecNeon, rhs: U64VecNeon) {
            this.0 = vandq_u64(this.0, rhs.0);
        }
    }
}

impl BitOr<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;
    fn_neon! {
        fn bitor(this: U64VecNeon, rhs: U64VecNeon) -> U64VecNeon {
            U64VecNeon(vorrq_u64(this.0, rhs.0), this.1)
        }
    }
}

impl BitOrAssign<U64VecNeon> for U64VecNeon {
    fn_neon! {
        fn bitor_assign(this: &mut U64VecNeon, rhs: U64VecNeon) {
            this.0 = vorrq_u64(this.0, rhs.0);
        }
    }
}

impl BitXor<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;
    fn_neon! {
        fn bitxor(this: U64VecNeon, rhs: U64VecNeon) -> U64VecNeon {
            U64VecNeon(veorq_u64(this.0, rhs.0), this.1)
        }
    }
}

impl BitXorAssign<U64VecNeon> for U64VecNeon {
    fn_neon! {
        fn bitxor_assign(this: &mut U64VecNeon, rhs: U64VecNeon) {
            this.0 = veorq_u64(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecNeon(uint8x16_t, NeonDescriptor);

impl U8SimdVec for U8VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, U8VecNeon);
    const LEN: usize = 16;

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[u8]) -> U8VecNeon {
        U8VecNeon(unaligned::vld1q_u8(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: u8) -> U8VecNeon {
        U8VecNeon(vdupq_n_u8(v), d)
    });

    fn_neon!(this: &U8VecNeon, fn store(mem: &mut [u8]) {
        unaligned::vst1q_u8(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon!(assoc fn store_interleaved_2(a: U8VecNeon, b: U8VecNeon, dest: &mut [u8]) {
        unaligned::vst2q_u8(dest.first_chunk_mut().unwrap(), uint8x16x2_t(a.0, b.0));
    });

    fn_neon!(assoc fn store_interleaved_3(a: U8VecNeon, b: U8VecNeon, c: U8VecNeon, dest: &mut [u8]) {
        unaligned::vst3q_u8(
            dest.first_chunk_mut().unwrap(),
            uint8x16x3_t(a.0, b.0, c.0),
        );
    });

    fn_neon!(assoc fn store_interleaved_4(a: U8VecNeon, b: U8VecNeon, c: U8VecNeon, d: U8VecNeon, dest: &mut [u8]) {
        unaligned::vst4q_u8(
            dest.first_chunk_mut().unwrap(),
            uint8x16x4_t(a.0, b.0, c.0, d.0),
        );
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecNeon(uint16x8_t, NeonDescriptor);

impl U16SimdVec for U16VecNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, U16VecNeon);
    const LEN: usize = 8;

    fn_neon!(assoc fn load(d: NeonDescriptor, mem: &[u16]) -> U16VecNeon {
        U16VecNeon(unaligned::vld1q_u16(mem.first_chunk().unwrap()), d)
    });

    fn_neon!(assoc fn splat(d: NeonDescriptor, v: u16) -> U16VecNeon {
        U16VecNeon(vdupq_n_u16(v), d)
    });

    fn_neon!(this: &U16VecNeon, fn store(mem: &mut [u16]) {
        unaligned::vst1q_u16(mem.first_chunk_mut().unwrap(), this.0);
    });

    fn_neon!(assoc fn store_interleaved_2(a: U16VecNeon, b: U16VecNeon, dest: &mut [u16]) {
        unaligned::vst2q_u16(dest.first_chunk_mut().unwrap(), uint16x8x2_t(a.0, b.0));
    });

    fn_neon!(assoc fn store_interleaved_3(a: U16VecNeon, b: U16VecNeon, c: U16VecNeon, dest: &mut [u16]) {
        unaligned::vst3q_u16(
            dest.first_chunk_mut().unwrap(),
            uint16x8x3_t(a.0, b.0, c.0),
        );
    });

    fn_neon!(assoc fn store_interleaved_4(a: U16VecNeon, b: U16VecNeon, c: U16VecNeon, d: U16VecNeon, dest: &mut [u16]) {
        unaligned::vst4q_u16(
            dest.first_chunk_mut().unwrap(),
            uint16x8x4_t(a.0, b.0, c.0, d.0),
        );
    });

    fn_neon!(this: U16VecNeon, fn bitcast_i16() -> I16VecNeon {
        I16VecNeon(vreinterpretq_s16_u16(this.0), this.1)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(uint32x4_t, NeonDescriptor);

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, MaskNeon);

    fn_neon! {
        fn if_then_else_f32(
            this: MaskNeon,
            if_true: F32VecNeon,
            if_false: F32VecNeon,
        ) -> F32VecNeon {
            F32VecNeon(vbslq_f32(this.0, if_true.0, if_false.0), this.1)
        }

        fn if_then_else_i32(
            this: MaskNeon,
            if_true: I32VecNeon,
            if_false: I32VecNeon,
        ) -> I32VecNeon {
            I32VecNeon(vbslq_s32(this.0, if_true.0, if_false.0), this.1)
        }

        fn maskz_i32(this: MaskNeon, v: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vbicq_s32(v.0, vreinterpretq_s32_u32(this.0)), this.1)
        }

        fn andnot(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vbicq_u32(rhs.0, this.0), this.1)
        }

        fn all(this: MaskNeon) -> bool {
            vminvq_u32(this.0) == u32::MAX
        }
    }
}

impl BitAnd<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon! {
        fn bitand(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vandq_u32(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon! {
        fn bitor(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vorrq_u32(this.0, rhs.0), this.1)
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Neon(uint16x8_t, NeonDescriptor);

impl SimdMask16 for Mask16Neon {
    type Descriptor = NeonDescriptor;
    crate::impl_isa_conversions!(neon, Mask16Neon);

    fn_neon! {
        fn if_then_else_i16(
            this: Mask16Neon,
            if_true: I16VecNeon,
            if_false: I16VecNeon,
        ) -> I16VecNeon {
            I16VecNeon(vbslq_s16(this.0, if_true.0, if_false.0), this.1)
        }

        fn maskz_i16(this: Mask16Neon, v: I16VecNeon) -> I16VecNeon {
            I16VecNeon(vbicq_s16(v.0, vreinterpretq_s16_u16(this.0)), this.1)
        }

        fn andnot(this: Mask16Neon, rhs: Mask16Neon) -> Mask16Neon {
            Mask16Neon(vbicq_u16(rhs.0, this.0), this.1)
        }

        fn all(this: Mask16Neon) -> bool {
            vminvq_u16(this.0) == u16::MAX
        }
    }
}

impl BitAnd<Mask16Neon> for Mask16Neon {
    type Output = Mask16Neon;
    fn_neon! {
        fn bitand(this: Mask16Neon, rhs: Mask16Neon) -> Mask16Neon {
            Mask16Neon(vandq_u16(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<Mask16Neon> for Mask16Neon {
    type Output = Mask16Neon;
    fn_neon! {
        fn bitor(this: Mask16Neon, rhs: Mask16Neon) -> Mask16Neon {
            Mask16Neon(vorrq_u16(this.0, rhs.0), this.1)
        }
    }
}

crate::impl_raw_accessors!(F32VecNeon, float32x4_t, NeonDescriptor);
crate::impl_raw_accessors!(I32VecNeon, int32x4_t, NeonDescriptor);
crate::impl_raw_accessors!(I16VecNeon, int16x8_t, NeonDescriptor);
crate::impl_raw_accessors!(U32VecNeon, uint32x4_t, NeonDescriptor);
crate::impl_raw_accessors!(U64VecNeon, uint64x2_t, NeonDescriptor);
crate::impl_raw_accessors!(U8VecNeon, uint8x16_t, NeonDescriptor);
crate::impl_raw_accessors!(U16VecNeon, uint16x8_t, NeonDescriptor);
crate::impl_raw_accessors!(MaskNeon, uint32x4_t, NeonDescriptor);
crate::impl_raw_accessors!(Mask16Neon, uint16x8_t, NeonDescriptor);
