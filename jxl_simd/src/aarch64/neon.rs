// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::aarch64::{
    float32x4_t, float32x4x2_t, float32x4x3_t, float32x4x4_t, int16x8_t, int32x4_t, uint8x16_t,
    uint8x16x2_t, uint8x16x3_t, uint8x16x4_t, uint16x8_t, uint16x8x2_t, uint16x8x3_t, uint16x8x4_t,
    uint32x4_t, uint64x2_t, vabsq_f32, vabsq_s16, vabsq_s32, vaddq_f32, vaddq_s16, vaddq_s32,
    vaddq_u64, vandq_s16, vandq_s32, vandq_u16, vandq_u32, vandq_u64, vbicq_s16, vbicq_s32,
    vbicq_u16, vbicq_u32, vbslq_f32, vbslq_s16, vbslq_s32, vceqq_s16, vceqq_s32, vceqzq_s16,
    vceqzq_s32, vcgtq_f32, vcgtq_s16, vcgtq_s32, vcltq_s16, vcltzq_s32, vcombine_s16,
    vcombine_u16, vcvt_f16_f32, vcvt_f32_f16, vcvtq_f32_s32, vcvtq_s32_f32, vdivq_f32,
    vdupq_n_f32, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32, vdupq_n_u64,
    veorq_s16, veorq_s32, veorq_u64, vfmaq_f32, vfmsq_f32, vget_low_s16, vget_low_s32, vmaxq_f32,
    vminq_f32, vminvq_u16, vminvq_u32, vmovl_s16, vmovl_u16, vmovn_s16, vmovn_s32, vmulq_f32,
    vmulq_s16, vmulq_s32, vmull_high_s16, vmull_high_s32, vmull_s16, vmull_s32, vnegq_f32,
    vnegq_s16, vnegq_s32, vorrq_s16, vorrq_s32, vorrq_u16, vorrq_u32, vorrq_u64, vqmovn_u16,
    vqmovun_s32, vqtbl1q_u8, vreinterpret_f16_u16, vreinterpret_u8_s8, vreinterpret_u16_f16,
    vreinterpret_u16_s16, vreinterpret_u32_s8, vreinterpret_u32_u8, vreinterpretq_f32_f64,
    vreinterpretq_f32_s32, vreinterpretq_f32_u8, vreinterpretq_f64_f32, vreinterpretq_s16_s32,
    vreinterpretq_s16_s64, vreinterpretq_s16_u16, vreinterpretq_s32_f32, vreinterpretq_s32_s16,
    vreinterpretq_s32_s64, vreinterpretq_s32_u32, vreinterpretq_s64_s16, vreinterpretq_u8_u16,
    vreinterpretq_u8_u32, vreinterpretq_u16_s16, vreinterpretq_u32_f32, vreinterpretq_u32_s32,
    vreinterpretq_u32_u64, vrndmq_f32, vrndnq_f32, vshlq_n_s16, vshlq_n_s32, vshlq_n_u32,
    vshlq_n_u64, vshrn_n_u32, vshrq_n_s16, vshrq_n_s32, vshrq_n_u32, vshrq_n_u64, vsqrtq_f32,
    vsubq_f32, vsubq_s16, vsubq_s32, vtrn1q_f32, vtrn2q_f32, vuzp2q_s16, vuzp2q_s32, vzip1q_f32,
    vzip1q_f64, vzip1q_s16, vzip1q_s32, vzip1q_s64, vzip2q_f32, vzip2q_f64, vzip2q_s16,
    vzip2q_s32, vzip2q_s64,
};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::super::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec, U16SimdVec,
};
use super::unaligned::{
    vld1_s16, vld1_u16, vld1q_f32, vld1q_s16, vld1q_s32, vld1q_u8, vld1q_u16, vld1q_u64,
    vld2q_f32, vld3q_f32, vld4q_f32, vld4q_f32_rows, vst1_lane_u32, vst1_u8, vst1_u16, vst1q_f32,
    vst1q_s16, vst1q_s32, vst1q_u8, vst1q_u16, vst1q_u64, vst2q_f32, vst2q_u8, vst2q_u16,
    vst3q_f32, vst3q_u8, vst3q_u16, vst4q_f32, vst4q_u8, vst4q_u16,
};
use crate::{U32SimdVec, U64SimdVec, impl_i16_array_interface};

// Safety invariant: this type is only ever constructed if neon is available.
#[target_feature(enable = "neon")]
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(());

/// Prepared 8-entry BF16 lookup table for NEON.
/// Contains 8 BF16 values packed into 16 bytes (uint8x16_t).
#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Neon(uint8x16_t, NeonDescriptor);

impl SimdDescriptor for NeonDescriptor {
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

    #[allow(unsafe_code)]
    fn new() -> Option<Self> {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: we just checked neon.
            Some(unsafe { Self(()) })
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
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(float32x4_t, NeonDescriptor);

impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(vdupq_n_f32(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(vdupq_n_f32(0.0), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        Self(vld1q_f32(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        vst1q_f32(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        vst2q_f32(dest.first_chunk_mut().unwrap(), float32x4x2_t(a.0, b.0));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        vst3q_f32(dest.first_chunk_mut().unwrap(), float32x4x3_t(a.0, b.0, c.0));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        vst4q_f32(
            dest.first_chunk_mut().unwrap(),
            float32x4x4_t(a.0, b.0, c.0, d.0),
        );
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        assert!(dest.len() >= 8 * F32VecNeon::LEN);
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

        let dest = &mut dest.as_chunks_mut::<4>().0[..8];
        vst1q_f32(&mut dest[0], out0);
        vst1q_f32(&mut dest[1], out1);
        vst1q_f32(&mut dest[2], out2);
        vst1q_f32(&mut dest[3], out3);
        vst1q_f32(&mut dest[4], out4);
        vst1q_f32(&mut dest[5], out5);
        vst1q_f32(&mut dest[6], out6);
        vst1q_f32(&mut dest[7], out7);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        let float32x4x2_t(a, b) = vld2q_f32(src.first_chunk().unwrap());
        (Self(a, d), Self(b, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        let float32x4x3_t(a, b, c) = vld3q_f32(src.first_chunk().unwrap());
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        let float32x4x4_t(a, b, c, e) = vld4q_f32(src.first_chunk().unwrap());
        (Self(a, d), Self(b, d), Self(c, d), Self(e, d))
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn transpose_square(d: NeonDescriptor, data: &mut [[f32; 4]], stride: usize) {
        if stride == 1 {
            let data = data.first_chunk_mut::<4>().unwrap();
            let float32x4x4_t(p0, p1, p2, p3) = vld4q_f32_rows(data);

            F32VecNeon(p0, d).store_array(&mut data[0]);
            F32VecNeon(p1, d).store_array(&mut data[1]);
            F32VecNeon(p2, d).store_array(&mut data[2]);
            F32VecNeon(p3, d).store_array(&mut data[3]);
        } else {
            assert!(data.len() > 3 * stride);

            let p0 = F32VecNeon::load_array(d, &data[0]).0;
            let p1 = F32VecNeon::load_array(d, &data[1 * stride]).0;
            let p2 = F32VecNeon::load_array(d, &data[2 * stride]).0;
            let p3 = F32VecNeon::load_array(d, &data[3 * stride]).0;

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

            F32VecNeon(p0, d).store_array(&mut data[0]);
            F32VecNeon(p1, d).store_array(&mut data[1 * stride]);
            F32VecNeon(p2, d).store_array(&mut data[2 * stride]);
            F32VecNeon(p3, d).store_array(&mut data[3 * stride]);
        }
    }

    crate::impl_f32_array_interface!();

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_add(self, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vfmaq_f32(add.0, self.0, mul.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg_mul_add(self, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vfmsq_f32(add.0, self.0, mul.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> F32VecNeon {
        F32VecNeon(vabsq_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn floor(self) -> F32VecNeon {
        F32VecNeon(vrndmq_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn sqrt(self) -> F32VecNeon {
        F32VecNeon(vsqrtq_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> F32VecNeon {
        F32VecNeon(vnegq_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn copysign(self, sign: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vbslq_f32(vdupq_n_u32(0x8000_0000), sign.0, self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn max(self, other: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vmaxq_f32(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn min(self, other: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vminq_f32(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, other: F32VecNeon) -> MaskNeon {
        MaskNeon(vcgtq_f32(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn as_i32(self) -> I32VecNeon {
        I32VecNeon(vcvtq_s32_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecNeon {
        I32VecNeon(vreinterpretq_s32_f32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        let dest = dest.first_chunk_mut().unwrap();
        // Round to nearest integer
        let rounded = vrndnq_f32(self.0);
        // Convert to i32, then to u16, then to u8
        let i32s = vcvtq_s32_f32(rounded);
        let u16s = vqmovun_s32(i32s);
        let u8s = vqmovn_u16(vcombine_u16(u16s, u16s));
        // Store lower 4 bytes
        vst1_lane_u32::<0>(dest, vreinterpret_u32_u8(u8s));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        let dest = dest.first_chunk_mut().unwrap();
        // Round to nearest integer
        let rounded = vrndnq_f32(self.0);
        // Convert to i32, then to u16
        let i32s = vcvtq_s32_f32(rounded);
        let u16s = vqmovun_s32(i32s);
        // Store 4 u16s (8 bytes)
        vst1_u16(dest, u16s);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        let dest = dest.first_chunk_mut().unwrap();
        let f16_bits = vreinterpret_u16_f16(vcvt_f16_f32(self.0));
        vst1_u16(dest, f16_bits);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        let mem = mem.first_chunk().unwrap();
        let f16_bits = vreinterpret_f16_u16(vld1_u16(mem));
        F32VecNeon(vcvt_f32_f16(f16_bits), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn prepare_table_bf16_8(d: NeonDescriptor, table: &[f32; 8]) -> Bf16Table8Neon {
        // Convert f32 table to BF16 packed in 128 bits (16 bytes for 8 entries)
        // BF16 is the high 16 bits of f32
        let ([t0, t1], _) = table.as_chunks::<4>() else {
            unreachable!()
        };
        let (table_lo, table_hi) = (vld1q_f32(t0), vld1q_f32(t1));

        // Reinterpret as u32 to extract high 16 bits
        let table_lo_u32 = vreinterpretq_u32_f32(table_lo);
        let table_hi_u32 = vreinterpretq_u32_f32(table_hi);

        // Shift right by 16 AND narrow to 16-bit in one instruction
        let bf16_lo_u16 = vshrn_n_u32::<16>(table_lo_u32);
        let bf16_hi_u16 = vshrn_n_u32::<16>(table_hi_u32);

        // Combine into 8 x u16 = 16 bytes
        let bf16_table_u16 = vcombine_u16(bf16_lo_u16, bf16_hi_u16);
        Bf16Table8Neon(vreinterpretq_u8_u16(bf16_table_u16), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn table_lookup_bf16_8(d: NeonDescriptor, table: Bf16Table8Neon, indices: I32VecNeon) -> Self {
        // Build shuffle mask efficiently using arithmetic on 32-bit indices.
        // For each index i (0-7), we need to select bytes [2*i, 2*i+1] from bf16_table
        // and place them in the high 16 bits of each 32-bit f32 lane (bytes 2,3),
        // with bytes 0,1 set to zero (using 0x80 which gives 0 in vqtbl1q).
        //
        // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
        // As a 32-bit value: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
        //                  = 0x8080 | (i << 17) | (i << 25) | (1 << 24)
        //                  = (i << 17) | (i << 25) | 0x01008080
        let indices_u32 = vreinterpretq_u32_s32(indices.0);
        let shl17 = vshlq_n_u32::<17>(indices_u32);
        let shl25 = vshlq_n_u32::<25>(indices_u32);
        let base = vdupq_n_u32(0x01008080);
        let shuffle_mask = vorrq_u32(vorrq_u32(shl17, shl25), base);

        // Perform the table lookup (out of range indices give 0)
        let result = vqtbl1q_u8(table.0, vreinterpretq_u8_u32(shuffle_mask));

        // Result has bf16 in high 16 bits of each 32-bit lane = valid f32
        F32VecNeon(vreinterpretq_f32_u8(result), d)
    }
}

impl Add<F32VecNeon> for F32VecNeon {
    type Output = Self;

    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vaddq_f32(self.0, rhs.0), self.1)
    }
}

impl Sub<F32VecNeon> for F32VecNeon {
    type Output = Self;

    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vsubq_f32(self.0, rhs.0), self.1)
    }
}

impl Mul<F32VecNeon> for F32VecNeon {
    type Output = Self;

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vmulq_f32(self.0, rhs.0), self.1)
    }
}

impl Div<F32VecNeon> for F32VecNeon {
    type Output = Self;

    #[target_feature(from_args)]
    #[inline(always)]
    fn div(self, rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vdivq_f32(self.0, rhs.0), self.1)
    }
}

impl AddAssign<F32VecNeon> for F32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: F32VecNeon) {
        self.0 = vaddq_f32(self.0, rhs.0);
    }
}

impl SubAssign<F32VecNeon> for F32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: F32VecNeon) {
        self.0 = vsubq_f32(self.0, rhs.0);
    }
}

impl MulAssign<F32VecNeon> for F32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: F32VecNeon) {
        self.0 = vmulq_f32(self.0, rhs.0);
    }
}

impl DivAssign<F32VecNeon> for F32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn div_assign(&mut self, rhs: F32VecNeon) {
        self.0 = vdivq_f32(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(int32x4_t, NeonDescriptor);

impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(vdupq_n_s32(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        Self(vld1q_s32(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_from_i16(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(vmovl_s16(vld1_s16(mem.first_chunk().unwrap())), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load_from_u16(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(mem.first_chunk().unwrap()))),
            d,
        )
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        vst1q_s32(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> I32VecNeon {
        I32VecNeon(vabsq_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn as_f32(self) -> F32VecNeon {
        F32VecNeon(vcvtq_f32_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_f32(self) -> F32VecNeon {
        F32VecNeon(vreinterpretq_f32_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecNeon {
        U32VecNeon(vreinterpretq_u32_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, other: I32VecNeon) -> MaskNeon {
        MaskNeon(vcgtq_s32(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn lt_zero(self) -> MaskNeon {
        MaskNeon(vcltzq_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq(self, other: I32VecNeon) -> MaskNeon {
        MaskNeon(vceqq_s32(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq_zero(self) -> MaskNeon {
        MaskNeon(vceqzq_s32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: I32VecNeon) -> I32VecNeon {
        let l = vmull_s32(vget_low_s32(self.0), vget_low_s32(rhs.0));
        let l = vreinterpretq_s32_s64(l);
        let h = vmull_high_s32(self.0, rhs.0);
        let h = vreinterpretq_s32_s64(h);
        I32VecNeon(vuzp2q_s32(l, h), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshlq_n_s32::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshrq_n_s32::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        let dest = dest.first_chunk_mut().unwrap();
        // vmovn narrows i32 to i16 by taking the lower 16 bits
        let narrowed = vmovn_s32(self.0);
        vst1_u16(dest, vreinterpret_u16_s16(narrowed));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        let dest = dest.first_chunk_mut().unwrap();
        // vmovn narrows i32 -> i16 -> i8
        let narrowed_i16 = vmovn_s32(self.0);
        let combined_i16 = vcombine_s16(narrowed_i16, narrowed_i16);
        let narrowed_i8 = vmovn_s16(combined_i16);
        vst1_lane_u32::<0>(dest, vreinterpret_u32_s8(narrowed_i8));
    }
}

impl Add<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vaddq_s32(self.0, rhs.0), self.1)
    }
}

impl Sub<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vsubq_s32(self.0, rhs.0), self.1)
    }
}

impl Mul<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vmulq_s32(self.0, rhs.0), self.1)
    }
}

impl Neg for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> I32VecNeon {
        I32VecNeon(vnegq_s32(self.0), self.1)
    }
}

impl BitAnd<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vandq_s32(self.0, rhs.0), self.1)
    }
}

impl BitOr<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vorrq_s32(self.0, rhs.0), self.1)
    }
}

impl BitXor<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(veorq_s32(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: I32VecNeon) {
        self.0 = vaddq_s32(self.0, rhs.0)
    }
}

impl SubAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I32VecNeon) {
        self.0 = vsubq_s32(self.0, rhs.0)
    }
}

impl MulAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I32VecNeon) {
        self.0 = vmulq_s32(self.0, rhs.0)
    }
}

impl BitAndAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I32VecNeon) {
        self.0 = vandq_s32(self.0, rhs.0);
    }
}

impl BitOrAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I32VecNeon) {
        self.0 = vorrq_s32(self.0, rhs.0);
    }
}

impl BitXorAssign<I32VecNeon> for I32VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I32VecNeon) {
        self.0 = veorq_s32(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I16VecNeon(int16x8_t, NeonDescriptor);

impl I16SimdVec for I16VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 8;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i16]) -> Self {
        Self(vld1q_s16(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [i16]) {
        vst1q_s16(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i16) -> Self {
        Self(vdupq_n_s16(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(vdupq_n_s16(0), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn abs(self) -> I16VecNeon {
        I16VecNeon(vabsq_s16(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn gt(self, other: I16VecNeon) -> Mask16Neon {
        Mask16Neon(vcgtq_s16(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn lt_zero(self) -> Mask16Neon {
        Mask16Neon(vcltq_s16(self.0, vdupq_n_s16(0)), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq(self, other: I16VecNeon) -> Mask16Neon {
        Mask16Neon(vceqq_s16(self.0, other.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn eq_zero(self) -> Mask16Neon {
        Mask16Neon(vceqzq_s16(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_wide_take_high(self, rhs: I16VecNeon) -> I16VecNeon {
        let l = vmull_s16(vget_low_s16(self.0), vget_low_s16(rhs.0));
        let l = vreinterpretq_s16_s32(l);
        let h = vmull_high_s16(self.0, rhs.0);
        let h = vreinterpretq_s16_s32(h);
        I16VecNeon(vuzp2q_s16(l, h), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshlq_n_s16::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshrq_n_s16::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_u16(self) -> U16VecNeon {
        U16VecNeon(vreinterpretq_u16_s16(self.0), self.1)
    }

    impl_i16_array_interface!();

    #[target_feature(from_args)]
    #[inline(always)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 7);

        let r0 = I16VecNeon::load_array(d, &data[0]).0;
        let r1 = I16VecNeon::load_array(d, &data[1 * stride]).0;
        let r2 = I16VecNeon::load_array(d, &data[2 * stride]).0;
        let r3 = I16VecNeon::load_array(d, &data[3 * stride]).0;
        let r4 = I16VecNeon::load_array(d, &data[4 * stride]).0;
        let r5 = I16VecNeon::load_array(d, &data[5 * stride]).0;
        let r6 = I16VecNeon::load_array(d, &data[6 * stride]).0;
        let r7 = I16VecNeon::load_array(d, &data[7 * stride]).0;

        // Stage 1: 16-bit zip
        let t0 = vzip1q_s16(r0, r1);
        let t1 = vzip2q_s16(r0, r1);
        let t2 = vzip1q_s16(r2, r3);
        let t3 = vzip2q_s16(r2, r3);
        let t4 = vzip1q_s16(r4, r5);
        let t5 = vzip2q_s16(r4, r5);
        let t6 = vzip1q_s16(r6, r7);
        let t7 = vzip2q_s16(r6, r7);

        // Stage 2: 32-bit zip
        let u0 = vreinterpretq_s16_s32(vzip1q_s32(
            vreinterpretq_s32_s16(t0),
            vreinterpretq_s32_s16(t2),
        ));
        let u1 = vreinterpretq_s16_s32(vzip2q_s32(
            vreinterpretq_s32_s16(t0),
            vreinterpretq_s32_s16(t2),
        ));
        let u2 = vreinterpretq_s16_s32(vzip1q_s32(
            vreinterpretq_s32_s16(t1),
            vreinterpretq_s32_s16(t3),
        ));
        let u3 = vreinterpretq_s16_s32(vzip2q_s32(
            vreinterpretq_s32_s16(t1),
            vreinterpretq_s32_s16(t3),
        ));
        let u4 = vreinterpretq_s16_s32(vzip1q_s32(
            vreinterpretq_s32_s16(t4),
            vreinterpretq_s32_s16(t6),
        ));
        let u5 = vreinterpretq_s16_s32(vzip2q_s32(
            vreinterpretq_s32_s16(t4),
            vreinterpretq_s32_s16(t6),
        ));
        let u6 = vreinterpretq_s16_s32(vzip1q_s32(
            vreinterpretq_s32_s16(t5),
            vreinterpretq_s32_s16(t7),
        ));
        let u7 = vreinterpretq_s16_s32(vzip2q_s32(
            vreinterpretq_s32_s16(t5),
            vreinterpretq_s32_s16(t7),
        ));

        // Stage 3: 64-bit zip
        let out0 = vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(u0),
            vreinterpretq_s64_s16(u4),
        ));
        let out1 = vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(u0),
            vreinterpretq_s64_s16(u4),
        ));
        let out2 = vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(u1),
            vreinterpretq_s64_s16(u5),
        ));
        let out3 = vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(u1),
            vreinterpretq_s64_s16(u5),
        ));
        let out4 = vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(u2),
            vreinterpretq_s64_s16(u6),
        ));
        let out5 = vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(u2),
            vreinterpretq_s64_s16(u6),
        ));
        let out6 = vreinterpretq_s16_s64(vzip1q_s64(
            vreinterpretq_s64_s16(u3),
            vreinterpretq_s64_s16(u7),
        ));
        let out7 = vreinterpretq_s16_s64(vzip2q_s64(
            vreinterpretq_s64_s16(u3),
            vreinterpretq_s64_s16(u7),
        ));

        I16VecNeon(out0, d).store_array(&mut data[0]);
        I16VecNeon(out1, d).store_array(&mut data[1 * stride]);
        I16VecNeon(out2, d).store_array(&mut data[2 * stride]);
        I16VecNeon(out3, d).store_array(&mut data[3 * stride]);
        I16VecNeon(out4, d).store_array(&mut data[4 * stride]);
        I16VecNeon(out5, d).store_array(&mut data[5 * stride]);
        I16VecNeon(out6, d).store_array(&mut data[6 * stride]);
        I16VecNeon(out7, d).store_array(&mut data[7 * stride]);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        let dest = dest.first_chunk_mut().unwrap();
        let narrowed_i8 = vmovn_s16(self.0);
        vst1_u8(dest, vreinterpret_u8_s8(narrowed_i8));
    }
}

impl Add<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vaddq_s16(self.0, rhs.0), self.1)
    }
}

impl Sub<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn sub(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vsubq_s16(self.0, rhs.0), self.1)
    }
}

impl Neg for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn neg(self) -> I16VecNeon {
        I16VecNeon(vnegq_s16(self.0), self.1)
    }
}

impl BitAnd<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vandq_s16(self.0, rhs.0), self.1)
    }
}

impl BitOr<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vorrq_s16(self.0, rhs.0), self.1)
    }
}

impl BitXor<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(veorq_s16(self.0, rhs.0), self.1)
    }
}

impl AddAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: I16VecNeon) {
        self.0 = vaddq_s16(self.0, rhs.0);
    }
}

impl SubAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn sub_assign(&mut self, rhs: I16VecNeon) {
        self.0 = vsubq_s16(self.0, rhs.0);
    }
}

impl BitAndAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: I16VecNeon) {
        self.0 = vandq_s16(self.0, rhs.0);
    }
}

impl BitOrAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: I16VecNeon) {
        self.0 = vorrq_s16(self.0, rhs.0);
    }
}

impl BitXorAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: I16VecNeon) {
        self.0 = veorq_s16(self.0, rhs.0);
    }
}

impl Mul<I16VecNeon> for I16VecNeon {
    type Output = I16VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn mul(self, rhs: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vmulq_s16(self.0, rhs.0), self.1)
    }
}

impl MulAssign<I16VecNeon> for I16VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn mul_assign(&mut self, rhs: I16VecNeon) {
        self.0 = vmulq_s16(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecNeon(uint32x4_t, NeonDescriptor);

impl U32SimdVec for U32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecNeon {
        I32VecNeon(vreinterpretq_s32_u32(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshrq_n_u32::<AMOUNT_I>(self.0), self.1)
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U64VecNeon(uint64x2_t, NeonDescriptor);

impl U64SimdVec for U64VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 2;

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u64) -> Self {
        Self(vdupq_n_u64(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u64]) -> Self {
        Self(vld1q_u64(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u64]) {
        vst1q_u64(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshlq_n_u64::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(vshrq_n_u64::<AMOUNT_I>(self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecNeon {
        U32VecNeon(vreinterpretq_u32_u64(self.0), self.1)
    }
}

impl Add<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn add(self, rhs: U64VecNeon) -> U64VecNeon {
        U64VecNeon(vaddq_u64(self.0, rhs.0), self.1)
    }
}

impl AddAssign<U64VecNeon> for U64VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn add_assign(&mut self, rhs: U64VecNeon) {
        self.0 = vaddq_u64(self.0, rhs.0);
    }
}

impl BitAnd<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: U64VecNeon) -> U64VecNeon {
        U64VecNeon(vandq_u64(self.0, rhs.0), self.1)
    }
}

impl BitAndAssign<U64VecNeon> for U64VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: U64VecNeon) {
        self.0 = vandq_u64(self.0, rhs.0);
    }
}

impl BitOr<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: U64VecNeon) -> U64VecNeon {
        U64VecNeon(vorrq_u64(self.0, rhs.0), self.1)
    }
}

impl BitOrAssign<U64VecNeon> for U64VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: U64VecNeon) {
        self.0 = vorrq_u64(self.0, rhs.0);
    }
}

impl BitXor<U64VecNeon> for U64VecNeon {
    type Output = U64VecNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor(self, rhs: U64VecNeon) -> U64VecNeon {
        U64VecNeon(veorq_u64(self.0, rhs.0), self.1)
    }
}

impl BitXorAssign<U64VecNeon> for U64VecNeon {
    #[target_feature(from_args)]
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: U64VecNeon) {
        self.0 = veorq_u64(self.0, rhs.0);
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecNeon(uint8x16_t, NeonDescriptor);

impl U8SimdVec for U8VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 16;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        Self(vld1q_u8(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(vdupq_n_u8(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        vst1q_u8(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        vst2q_u8(dest.first_chunk_mut().unwrap(), uint8x16x2_t(a.0, b.0));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        vst3q_u8(dest.first_chunk_mut().unwrap(), uint8x16x3_t(a.0, b.0, c.0));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        vst4q_u8(
            dest.first_chunk_mut().unwrap(),
            uint8x16x4_t(a.0, b.0, c.0, d.0),
        );
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecNeon(uint16x8_t, NeonDescriptor);

impl U16SimdVec for U16VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 8;

    #[target_feature(from_args)]
    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        Self(vld1q_u16(mem.first_chunk().unwrap()), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(vdupq_n_u16(v), d)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        vst1q_u16(mem.first_chunk_mut().unwrap(), self.0);
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        vst2q_u16(dest.first_chunk_mut().unwrap(), uint16x8x2_t(a.0, b.0));
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        vst3q_u16(
            dest.first_chunk_mut().unwrap(),
            uint16x8x3_t(a.0, b.0, c.0),
        );
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        vst4q_u16(
            dest.first_chunk_mut().unwrap(),
            uint16x8x4_t(a.0, b.0, c.0, d.0),
        );
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitcast_i16(self) -> I16VecNeon {
        I16VecNeon(vreinterpretq_s16_u16(self.0), self.1)
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(uint32x4_t, NeonDescriptor);

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_f32(self, if_true: F32VecNeon, if_false: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vbslq_f32(self.0, if_true.0, if_false.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_i32(self, if_true: I32VecNeon, if_false: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vbslq_s32(self.0, if_true.0, if_false.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn maskz_i32(self, v: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vbicq_s32(v.0, vreinterpretq_s32_u32(self.0)), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn andnot(self, rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vbicq_u32(rhs.0, self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn all(self) -> bool {
        vminvq_u32(self.0) == u32::MAX
    }
}

impl BitAnd<MaskNeon> for MaskNeon {
    type Output = MaskNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vandq_u32(self.0, rhs.0), self.1)
    }
}

impl BitOr<MaskNeon> for MaskNeon {
    type Output = MaskNeon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vorrq_u32(self.0, rhs.0), self.1)
    }
}

#[target_feature(inherit)]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Mask16Neon(uint16x8_t, NeonDescriptor);

impl SimdMask16 for Mask16Neon {
    type Descriptor = NeonDescriptor;

    #[target_feature(from_args)]
    #[inline(always)]
    fn if_then_else_i16(self, if_true: I16VecNeon, if_false: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vbslq_s16(self.0, if_true.0, if_false.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn maskz_i16(self, v: I16VecNeon) -> I16VecNeon {
        I16VecNeon(vbicq_s16(v.0, vreinterpretq_s16_u16(self.0)), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn andnot(self, rhs: Mask16Neon) -> Mask16Neon {
        Mask16Neon(vbicq_u16(rhs.0, self.0), self.1)
    }

    #[target_feature(from_args)]
    #[inline(always)]
    fn all(self) -> bool {
        vminvq_u16(self.0) == u16::MAX
    }
}

impl BitAnd<Mask16Neon> for Mask16Neon {
    type Output = Mask16Neon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitand(self, rhs: Mask16Neon) -> Mask16Neon {
        Mask16Neon(vandq_u16(self.0, rhs.0), self.1)
    }
}

impl BitOr<Mask16Neon> for Mask16Neon {
    type Output = Mask16Neon;

    #[target_feature(from_args)]
    #[inline(always)]
    fn bitor(self, rhs: Mask16Neon) -> Mask16Neon {
        Mask16Neon(vorrq_u16(self.0, rhs.0), self.1)
    }
}
