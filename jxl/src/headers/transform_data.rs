// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
#[derive(Default)]
pub struct CustomTransformDataNonserialized {
    pub xyb_encoded: bool,
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct OpsinInverseMatrix {
    #[all_default]
    // TODO(firsching): remove once we use this!
    #[allow(dead_code)]
    all_default: bool,
    #[default([11.031566901960783, -9.866943921568629, -0.16462299647058826,
               -3.254147380392157,  4.418770392156863,  -0.16462299647058826,
               -3.6588512862745097, 2.7129230470588235, 1.9459282392156863])]
    pub inverse_matrix: [f32; 9],
    #[default([-0.0037930732552754493; 3])]
    pub opsin_biases: [f32; 3],
    #[default([1.0 - 0.05465007330715401, 1.0 - 0.07005449891748593, 1.0 - 0.049935103337343655, 0.145])]
    pub quant_biases: [f32; 4],
}

const DEFAULT_KERN_2: [f32; 15] = [
    -0.01716200,
    -0.03452303,
    -0.04022174,
    -0.02921014,
    -0.00624645,
    0.14111091,
    0.28896755,
    0.00278718,
    -0.01610267,
    0.56661550,
    0.03777607,
    -0.01986694,
    -0.03144731,
    -0.01185068,
    -0.00213539,
];

const DEFAULT_KERN_4: [f32; 55] = [
    -0.02419067,
    -0.03491987,
    -0.03693351,
    -0.03094285,
    -0.00529785,
    -0.01663432,
    -0.03556863,
    -0.03888905,
    -0.03516850,
    -0.00989469,
    0.23651958,
    0.33392945,
    -0.01073543,
    -0.01313181,
    -0.03556694,
    0.13048175,
    0.40103025,
    0.03951150,
    -0.02077584,
    0.46914198,
    -0.00209270,
    -0.01484589,
    -0.04064806,
    0.18942530,
    0.56279892,
    0.06674400,
    -0.02335494,
    -0.03551682,
    -0.00754830,
    -0.02267919,
    -0.02363578,
    0.00315804,
    -0.03399098,
    -0.01359519,
    -0.00091653,
    -0.00335467,
    -0.01163294,
    -0.01610294,
    -0.00974088,
    -0.00191622,
    -0.01095446,
    -0.03198464,
    -0.04455121,
    -0.02799790,
    -0.00645912,
    0.06390599,
    0.22963888,
    0.00630981,
    -0.01897349,
    0.67537268,
    0.08483369,
    -0.02534994,
    -0.02205197,
    -0.01667999,
    -0.00384443,
];

const DEFAULT_KERN_8: [f32; 210] = [
    -0.02928613,
    -0.03706353,
    -0.03783812,
    -0.03324558,
    -0.00447632,
    -0.02519406,
    -0.03752601,
    -0.03901508,
    -0.03663285,
    -0.00646649,
    -0.02066407,
    -0.03838633,
    -0.04002101,
    -0.03900035,
    -0.00901973,
    -0.01626393,
    -0.03954148,
    -0.04046620,
    -0.03979621,
    -0.01224485,
    0.29895328,
    0.35757708,
    -0.02447552,
    -0.01081748,
    -0.04314594,
    0.23903219,
    0.41119301,
    -0.00573046,
    -0.01450239,
    -0.04246845,
    0.17567618,
    0.45220643,
    0.02287757,
    -0.01936783,
    -0.03583255,
    0.11572472,
    0.47416733,
    0.06284440,
    -0.02685066,
    0.42720050,
    -0.02248939,
    -0.01155273,
    -0.04562755,
    0.28689496,
    0.49093869,
    -0.00007891,
    -0.01545926,
    -0.04562659,
    0.21238920,
    0.53980934,
    0.03369474,
    -0.02070211,
    -0.03866988,
    0.14229550,
    0.56593398,
    0.08045181,
    -0.02888298,
    -0.03680918,
    -0.00542229,
    -0.02920477,
    -0.02788574,
    -0.02118180,
    -0.03942402,
    -0.00775547,
    -0.02433614,
    -0.03193943,
    -0.02030828,
    -0.04044014,
    -0.01074016,
    -0.01930822,
    -0.03620399,
    -0.01974125,
    -0.03919545,
    -0.01456093,
    -0.00045072,
    -0.00360110,
    -0.01020207,
    -0.01231907,
    -0.00638988,
    -0.00071592,
    -0.00279122,
    -0.00957115,
    -0.01288327,
    -0.00730937,
    -0.00107783,
    -0.00210156,
    -0.00890705,
    -0.01317668,
    -0.00813895,
    -0.00153491,
    -0.02128481,
    -0.04173044,
    -0.04831487,
    -0.03293190,
    -0.00525260,
    -0.01720322,
    -0.04052736,
    -0.05045706,
    -0.03607317,
    -0.00738030,
    -0.01341764,
    -0.03965629,
    -0.05151616,
    -0.03814886,
    -0.01005819,
    0.18968273,
    0.33063684,
    -0.01300105,
    -0.01372950,
    -0.04017465,
    0.13727832,
    0.36402234,
    0.01027890,
    -0.01832107,
    -0.03365072,
    0.08734506,
    0.38194295,
    0.04338228,
    -0.02525993,
    0.56408126,
    0.00458352,
    -0.01648227,
    -0.04887868,
    0.24585519,
    0.62026135,
    0.04314807,
    -0.02213737,
    -0.04158014,
    0.16637289,
    0.65027023,
    0.09621636,
    -0.03101388,
    -0.04082742,
    -0.00904519,
    -0.02790922,
    -0.02117818,
    0.00798662,
    -0.03995711,
    -0.01243427,
    -0.02231705,
    -0.02946266,
    0.00992055,
    -0.03600283,
    -0.01684920,
    -0.00111684,
    -0.00411204,
    -0.01297130,
    -0.01723725,
    -0.01022545,
    -0.00165306,
    -0.00313110,
    -0.01218016,
    -0.01763266,
    -0.01125620,
    -0.00231663,
    -0.01374149,
    -0.03797620,
    -0.05142937,
    -0.03117307,
    -0.00581914,
    -0.01064003,
    -0.03608089,
    -0.05272168,
    -0.03375670,
    -0.00795586,
    0.09628104,
    0.27129991,
    -0.00353779,
    -0.01734151,
    -0.03153981,
    0.05686230,
    0.28500998,
    0.02230594,
    -0.02374955,
    0.68214326,
    0.05018048,
    -0.02320852,
    -0.04383616,
    0.18459474,
    0.71517975,
    0.10805613,
    -0.03263677,
    -0.03637639,
    -0.01394373,
    -0.02511203,
    -0.01728636,
    0.05407331,
    -0.02867568,
    -0.01893131,
    -0.00240854,
    -0.00446511,
    -0.01636187,
    -0.02377053,
    -0.01522848,
    -0.00333334,
    -0.00819975,
    -0.02964169,
    -0.04499287,
    -0.02745350,
    -0.00612408,
    0.02727416,
    0.19446600,
    0.00159832,
    -0.02232473,
    0.74982506,
    0.11452620,
    -0.03348048,
    -0.01605681,
    -0.02070339,
    -0.00458223,
];

// TODO(firsching): remove once we use this!
#[allow(dead_code)]
#[derive(UnconditionalCoder, Debug, Clone)]
#[nonserialized(CustomTransformDataNonserialized)]
pub struct CustomTransformData {
    #[all_default]
    all_default: bool,
    #[condition(nonserialized.xyb_encoded)]
    #[default(OpsinInverseMatrix::default(&field_nonserialized))]
    pub opsin_inverse_matrix: OpsinInverseMatrix,
    #[default(0)]
    #[coder(Bits(3))]
    custom_weight_mask: u32,
    #[condition((custom_weight_mask & 1) != 0)]
    #[default(DEFAULT_KERN_2)]
    pub weights2: [f32; 15],
    #[condition((custom_weight_mask & 2) != 0)]
    #[default(DEFAULT_KERN_4)]
    pub weights4: [f32; 55],
    #[condition((custom_weight_mask & 4) != 0)]
    #[default(DEFAULT_KERN_8)]
    pub weights8: [f32; 210],
}
