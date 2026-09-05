// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::HashSet;

use crate::entropy_coding::decode::{Codes, Histograms};
use crate::frame::modular::Predictor;
use crate::frame::modular::tree::TreeNode;
use crate::headers::modular::GroupHeader;

pub struct CodegenInput<'a> {
    pub tree: &'a [TreeNode],
    pub xsize: usize,
    pub header: &'a GroupHeader,
    pub histograms: &'a Histograms,
    pub single_symbol: Option<i32>,
    pub num_ref_channels: usize,
}

pub fn generate_c_code(input: &CodegenInput) -> String {
    let mut out = String::with_capacity(8192);

    // 1. Collect used properties, predictors, and clusters
    let mut used_properties = HashSet::new();
    let mut uses_wp = false;

    for node in input.tree {
        match node {
            TreeNode::Split { property, .. } => {
                used_properties.insert(*property);
                if *property == 15 {
                    uses_wp = true;
                }
            }
            TreeNode::Leaf { predictor, .. } => {
                if *predictor == Predictor::Weighted {
                    uses_wp = true;
                }
            }
        }
    }

    let has_varying_multiplier = input.tree.iter().any(|node| match node {
        TreeNode::Leaf { multiplier, .. } => *multiplier != 1,
        _ => false,
    });

    // 2. Identify unique HybridUint configs used in the tree leaves
    let mut used_clusters = HashSet::new();
    for node in input.tree {
        if let TreeNode::Leaf { id, .. } = node {
            used_clusters.insert(*id as usize);
        }
    }

    let mut unique_uint_configs = Vec::new();
    let mut cluster_to_cfg_idx = std::collections::HashMap::new();

    for &cluster in &used_clusters {
        let cfg = input.histograms.uint(cluster);
        let key = (
            cfg.split_token(),
            cfg.split_exponent(),
            cfg.msb_in_token(),
            cfg.lsb_in_token(),
        );
        let idx = if let Some(pos) = unique_uint_configs.iter().position(|k| *k == key) {
            pos
        } else {
            unique_uint_configs.push(key);
            unique_uint_configs.len() - 1
        };
        cluster_to_cfg_idx.insert(cluster, idx);
    }

    // 3. Emit headers and defines
    out.push_str(
        r#"#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

"#,
    );

    out.push_str(&format!("#define XSIZE {}\n", input.xsize));
    out.push_str(&format!(
        "#define NUM_REF_CHANNELS {}\n",
        input.num_ref_channels
    ));
    out.push_str(&format!(
        "#define USES_WP {}\n",
        if uses_wp { 1 } else { 0 }
    ));

    let is_ans = matches!(input.histograms.codes(), Codes::Ans(_));
    out.push_str(&format!("#define IS_ANS {}\n", if is_ans { 1 } else { 0 }));

    if let Some(sym) = input.single_symbol {
        out.push_str("#define HAS_SINGLE_SYM 1\n");
        out.push_str(&format!("#define SINGLE_SYM ({sym})\n"));
    } else {
        out.push_str("#define HAS_SINGLE_SYM 0\n");
    }

    out.push_str(&format!(
        "#define HAS_VARYING_MULTIPLIER {}\n\n",
        if has_varying_multiplier { 1 } else { 0 }
    ));

    // 4. BitReader definitions
    out.push_str(
        r#"typedef struct {
    const uint8_t* data;
    size_t data_len;
    uint64_t bit_buf;
    size_t bits_in_buf;
    size_t total_bits_read;
} CBitReader;

static inline void br_refill(CBitReader* br) {
    if (br->data_len >= 8) {
        uint64_t bits;
        memcpy(&bits, br->data, 8);
        br->bit_buf |= (bits << br->bits_in_buf);
        size_t read_bytes = (63 - br->bits_in_buf) >> 3;
        br->bits_in_buf |= 56;
        br->data += read_bytes;
        br->data_len -= read_bytes;
    } else {
        while (br->bits_in_buf < 56 && br->data_len > 0) {
            br->bit_buf |= ((uint64_t)*br->data) << br->bits_in_buf;
            br->bits_in_buf += 8;
            br->data++;
            br->data_len--;
        }
    }
}

static inline uint64_t br_peek(CBitReader* br, size_t num) {
    if (br->bits_in_buf < num) {
        br_refill(br);
    }
    return br->bit_buf & ((1ULL << num) - 1ULL);
}

static inline void br_consume(CBitReader* br, size_t num) {
    br->bit_buf >>= num;
    br->bits_in_buf = br->bits_in_buf >= num ? br->bits_in_buf - num : 0;
    br->total_bits_read += num;
}

"#,
    );

    // 5. Predictor helpers
    out.push_str(
        r#"static inline int64_t clamped_gradient(int64_t left, int64_t top, int64_t topleft) {
    int64_t min = left < top ? left : top;
    int64_t max = left > top ? left : top;
    int64_t grad = left + top - topleft;
    int64_t grad_clamp_max = topleft < min ? max : grad;
    return topleft > max ? min : grad_clamp_max;
}

static inline int64_t select_pred(int64_t left, int64_t top, int64_t topleft) {
    int64_t p = left + top - topleft;
    return llabs(p - left) < llabs(p - top) ? left : top;
}

static inline int32_t unpack_signed(uint32_t val) {
    return (val & 1) ? -(int32_t)(val >> 1) - 1 : (int32_t)(val >> 1);
}

"#,
    );

    // 6. Entropy tables definitions
    if is_ans {
        out.push_str(
            r#"typedef struct {
    uint8_t alias_symbol;
    uint8_t alias_cutoff;
    uint16_t dist;
    uint16_t alias_offset;
    uint16_t alias_dist_xor;
} CAnsBucket;

typedef struct {
    const CAnsBucket* buckets;
    uint32_t log_bucket_size;
    uint32_t bucket_mask;
} CAnsHistogram;

static inline uint32_t read_ans(CBitReader* br, const CAnsHistogram* h, uint32_t* state) {
    uint32_t idx = *state & 0xfff;
    uint32_t i = idx >> h->log_bucket_size;
    uint32_t pos = idx & h->bucket_mask;
    CAnsBucket bucket = h->buckets[i];
    uint32_t map_to_alias = pos >= bucket.alias_cutoff;
    uint32_t symbol = map_to_alias ? bucket.alias_symbol : i;
    uint32_t offset = (map_to_alias ? bucket.alias_offset : 0) + pos;
    uint32_t dist = bucket.dist ^ (map_to_alias ? bucket.alias_dist_xor : 0);
    uint32_t next_state = (*state >> 12) * dist + offset;
    if (next_state < (1 << 16)) {
        *state = (next_state << 16) | (uint32_t)br_peek(br, 16);
        br_consume(br, 16);
    } else {
        *state = next_state;
    }
    return symbol;
}

"#,
        );
    } else {
        out.push_str(
            r#"typedef struct {
    uint8_t bits;
    uint16_t value;
} TableEntry;

static inline uint32_t read_huffman(CBitReader* br, const TableEntry* table) {
    size_t pos = br_peek(br, 8);
    size_t n_bits = table[pos].bits;
    if (n_bits > 8) {
        br_consume(br, 8);
        n_bits -= 8;
        pos += table[pos].value;
        pos += br_peek(br, n_bits);
    }
    TableEntry entry = table[pos];
    br_consume(br, entry.bits);
    return entry.value;
}

"#,
        );
    }

    // 7. Specialized HybridUint functions
    for (idx, &(split_token, split_exponent, msb, lsb)) in unique_uint_configs.iter().enumerate() {
        out.push_str(&format!(
            r#"static inline uint32_t decode_hybrid_uint_cfg_{idx}(CBitReader* br, uint32_t symbol) {{
    if (symbol < {split_token}U) {{
        return symbol;
    }}
"#,
        ));

        let bits_in_token = lsb + msb;
        let base_shift = split_exponent.saturating_sub(bits_in_token);
        out.push_str(&format!(
            r#"    uint32_t nbits = ({base_shift}U + ((symbol - {split_token}U) >> {bits_in_token}U)) & 31U;
    uint32_t bits = (uint32_t)br_peek(br, nbits);
    br_consume(br, nbits);
"#
        ));

        if lsb == 0 {
            let mask = (1u32 << msb) - 1;
            let hi_bit = 1u32 << msb;
            out.push_str(&format!(
                r#"    uint32_t hi = (symbol & {mask}U) | {hi_bit}U;
    return (hi << nbits) | bits;
}}

"#
            ));
        } else {
            let lsb_mask = (1u32 << lsb) - 1;
            let msb_mask = (1u32 << msb) - 1;
            let hi_bit = 1u32 << msb;
            out.push_str(&format!(
                r#"    uint32_t low = symbol & {lsb_mask}U;
    uint32_t symbol_nolow = symbol >> {lsb}U;
    uint32_t hi = (symbol_nolow & {msb_mask}U) | {hi_bit}U;
    return (((hi << nbits) | bits) << {lsb}U) | low;
}}

"#
            ));
        }
    }

    // Unified HybridUint dispatcher
    if unique_uint_configs.is_empty() {
        out.push_str(
            r#"static inline uint32_t decode_hybrid_uint(CBitReader* br, uint32_t token, uint32_t cluster) {
    (void)br;
    (void)cluster;
    return token;
}

"#,
        );
    } else if unique_uint_configs.len() == 1 {
        out.push_str(
            r#"static inline uint32_t decode_hybrid_uint(CBitReader* br, uint32_t token, uint32_t cluster) {
    (void)cluster;
    return decode_hybrid_uint_cfg_0(br, token);
}

"#,
        );
    } else {
        let num_clusters = input.histograms.num_histograms();
        out.push_str(&format!(
            "static const uint8_t CLUSTER_TO_CFG[{num_clusters}] = {{"
        ));
        for c in 0..num_clusters {
            let cfg_idx = cluster_to_cfg_idx.get(&c).copied().unwrap_or(0);
            out.push_str(&format!("{cfg_idx}, "));
        }
        out.push_str("};\n\n");

        out.push_str(
            r#"static inline uint32_t decode_hybrid_uint(CBitReader* br, uint32_t token, uint32_t cluster) {
    switch (CLUSTER_TO_CFG[cluster]) {
"#,
        );
        for idx in 0..unique_uint_configs.len() {
            out.push_str(&format!(
                "        case {idx}: return decode_hybrid_uint_cfg_{idx}(br, token);\n"
            ));
        }
        out.push_str(
            r#"        default: return decode_hybrid_uint_cfg_0(br, token);
    }
}

"#,
        );
    }

    // 8. Reference channel struct
    out.push_str(
        r#"typedef struct {
    const int32_t* data;
    size_t stride;
} CReferenceChannel;

"#,
    );

    // 9. Weighted Predictor state and functions
    if uses_wp {
        let wp = &input.header.wp_header;
        out.push_str(&format!(
            r#"#define NUM_ERRORS ((XSIZE + 1) * 2)
static const uint32_t DIVLOOKUP[64] = {{
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144
}};

#define WP_W0 {w0}
#define WP_W1 {w1}
#define WP_W2 {w2}
#define WP_W3 {w3}
#define WP_P1C {p1c}
#define WP_P2C {p2c}
#define WP_P3C0 {p3ca}
#define WP_P3C1 {p3cb}
#define WP_P3C2 {p3cc}
#define WP_P3C3 {p3cd}
#define WP_P3C4 {p3ce}

static inline uint32_t floor_log2_u64(uint64_t v) {{
    return 63 - __builtin_clzll(v);
}}

"#,
            w0 = wp.w0,
            w1 = wp.w1,
            w2 = wp.w2,
            w3 = wp.w3,
            p1c = wp.p1c,
            p2c = wp.p2c,
            p3ca = wp.p3ca,
            p3cb = wp.p3cb,
            p3cc = wp.p3cc,
            p3cd = wp.p3cd,
            p3ce = wp.p3ce,
        ));
    }

    // 10. The main decoding function: jit_decode_channel
    out.push_str(
        r#"void jit_decode_channel(
    int32_t* channel_data,
    size_t stride,
    size_t ysize,
    CBitReader* br,
    const void* entropy_tables,
    uint32_t* ans_state,
    const CReferenceChannel* ref_channels
) {
#if !HAS_SINGLE_SYM
#if IS_ANS
    const CAnsHistogram* ans_histograms = (const CAnsHistogram*)entropy_tables;
#else
    const TableEntry* const* huffman_tables = (const TableEntry* const*)entropy_tables;
#endif
#endif

#if USES_WP
    int32_t error[NUM_ERRORS] = {0};
    uint32_t pred_errors[NUM_ERRORS][4] = {{0}};
#endif

"#,
    );

    // Check which reference channels are actually accessed by used_properties
    let mut used_ref_channels: Vec<usize> = used_properties
        .iter()
        .filter_map(|&p| {
            if p >= 16 {
                let ref_idx = ((p - 16) / 4) as usize;
                if ref_idx < input.num_ref_channels {
                    Some(ref_idx)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    used_ref_channels.sort_unstable();
    used_ref_channels.dedup();

    out.push_str(
        r#"    for (size_t y = 0; y < ysize; ++y) {
        int32_t* row = channel_data + y * stride;
        const int32_t* row_top = (y > 0) ? channel_data + (y - 1) * stride : NULL;
        const int32_t* row_toptop = (y > 1) ? channel_data + (y - 2) * stride : NULL;
        int32_t prev_grad = 0;

        int32_t last = 0;
        int32_t cur_left = 0;
        int32_t cur_top = 0;
        int32_t cur_toptop = 0;
        int32_t cur_topleft = 0;
        int32_t cur_topright = 0;
        int32_t cur_leftleft = 0;
        int32_t cur_toprightright = 0;

"#,
    );

    for &ref_idx in &used_ref_channels {
        out.push_str(&format!(
            "        const int32_t* r{ref_idx}_row = ref_channels[{ref_idx}].data + y * ref_channels[{ref_idx}].stride;\n"
        ));
        out.push_str(&format!(
            "        const int32_t* r{ref_idx}_prev = (y > 0) ? ref_channels[{ref_idx}].data + (y - 1) * ref_channels[{ref_idx}].stride : r{ref_idx}_row;\n"
        ));
    }

    out.push_str(
        r#"
        for (size_t x = 0; x < XSIZE; ++x) {
            // Update neighborhood
            if (y >= 2 && x >= 2 && x + 2 < XSIZE) {
                cur_leftleft = cur_left;
                cur_left = last;
                cur_topleft = cur_top;
                cur_top = cur_topright;
                cur_topright = cur_toprightright;
                cur_toprightright = row_top[x + 2];
                cur_toptop = row_toptop[x];
            } else {
                cur_left = (x > 0) ? row[x - 1] : ((y > 0) ? row_top[0] : 0);
                cur_top = (y > 0) ? row_top[x] : cur_left;
                cur_topleft = (x > 0 && y > 0) ? row_top[x - 1] : cur_left;
                cur_topright = (x + 1 < XSIZE && y > 0) ? row_top[x + 1] : cur_top;
                cur_leftleft = (x > 1) ? row[x - 2] : cur_left;
                cur_toptop = (y > 1) ? row_toptop[x] : cur_top;
                cur_toprightright = (x + 2 < XSIZE && y > 0) ? row_top[x + 2] : cur_topright;
            }

            int64_t wp_pred = 0;
            int32_t wp_prop = 0;
            int64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            int64_t raw_pred = 0;

#if USES_WP
            size_t cur_row = (y & 1) ? 0 : (XSIZE + 1);
            size_t prev_row = (y & 1) ? (XSIZE + 1) : 0;
            size_t pos_ne = (x + 1 < XSIZE) ? (x + 1) : x;
            size_t pos_nw = (x > 0) ? (x - 1) : 0;

            const uint32_t* err_n = pred_errors[prev_row + x];
            const uint32_t* err_ne = pred_errors[prev_row + pos_ne];
            const uint32_t* err_nw = pred_errors[prev_row + pos_nw];

            uint32_t err0 = err_n[0] + err_ne[0] + err_nw[0];
            uint32_t err1 = err_n[1] + err_ne[1] + err_nw[1];
            uint32_t err2 = err_n[2] + err_ne[2] + err_nw[2];
            uint32_t err3 = err_n[3] + err_ne[3] + err_nw[3];

            uint32_t shift0 = (err0 + 1) > 0 ? (floor_log2_u64(err0 + 1) >= 5 ? floor_log2_u64(err0 + 1) - 5 : 0) : 0;
            uint32_t shift1 = (err1 + 1) > 0 ? (floor_log2_u64(err1 + 1) >= 5 ? floor_log2_u64(err1 + 1) - 5 : 0) : 0;
            uint32_t shift2 = (err2 + 1) > 0 ? (floor_log2_u64(err2 + 1) >= 5 ? floor_log2_u64(err2 + 1) - 5 : 0) : 0;
            uint32_t shift3 = (err3 + 1) > 0 ? (floor_log2_u64(err3 + 1) >= 5 ? floor_log2_u64(err3 + 1) - 5 : 0) : 0;

            uint32_t div0 = DIVLOOKUP[(err0 >> shift0) & 63];
            uint32_t div1 = DIVLOOKUP[(err1 >> shift1) & 63];
            uint32_t div2 = DIVLOOKUP[(err2 >> shift2) & 63];
            uint32_t div3 = DIVLOOKUP[(err3 >> shift3) & 63];

            uint32_t w0 = 4 + ((WP_W0 * div0) >> shift0);
            uint32_t w1 = 4 + ((WP_W1 * div1) >> shift1);
            uint32_t w2 = 4 + ((WP_W2 * div2) >> shift2);
            uint32_t w3 = 4 + ((WP_W3 * div3) >> shift3);

            int64_t te_w = (int64_t)error[cur_row + x];
            int64_t te_n = (int64_t)error[prev_row + 1 + x];
            int64_t te_nw = (int64_t)error[prev_row + 1 + pos_nw];
            int64_t te_ne = (int64_t)error[prev_row + 1 + pos_ne];
            int64_t sum_wn = te_n + te_w;

            int64_t p = te_w;
            if (llabs(te_n) > llabs(p)) p = te_n;
            if (llabs(te_nw) > llabs(p)) p = te_nw;
            if (llabs(te_ne) > llabs(p)) p = te_ne;
            wp_prop = (int32_t)p;

            int64_t n = (int64_t)cur_top << 3;
            int64_t w = (int64_t)cur_left << 3;
            int64_t ne = (int64_t)cur_topright << 3;
            int64_t nw = (int64_t)cur_topleft << 3;
            int64_t nn = (int64_t)cur_toptop << 3;

            p0 = w + ne - n;
            p1 = n - (((sum_wn + te_ne) * WP_P1C) >> 5);
            p2 = w - (((sum_wn + te_nw) * WP_P2C) >> 5);
            p3 = n - ((te_nw * WP_P3C0 + te_n * WP_P3C1 + te_ne * WP_P3C2 + (nn - n) * WP_P3C3 + (nw - w) * WP_P3C4) >> 5);

            uint32_t log_weight = floor_log2_u64((uint64_t)w0 + w1 + w2 + w3);
            uint32_t shift_w = log_weight >= 4 ? log_weight - 4 : 0;
            int64_t w0s = w0 >> shift_w;
            int64_t w1s = w1 >> shift_w;
            int64_t w2s = w2 >> shift_w;
            int64_t w3s = w3 >> shift_w;

            int64_t weight_sum = w0s + w1s + w2s + w3s;
            int64_t sum = (weight_sum >> 1) - 1 + w0s * p0 + w1s * p1 + w2s * p2 + w3s * p3;
            raw_pred = (sum * (int64_t)DIVLOOKUP[(weight_sum - 1) & 63]) >> 24;

            if (((te_n ^ te_w) | (te_n ^ te_nw)) <= 0) {
                int64_t mx = w > ne ? (w > n ? w : n) : (ne > n ? ne : n);
                int64_t mn = w < ne ? (w < n ? w : n) : (ne < n ? ne : n);
                raw_pred = mn > raw_pred ? mn : (mx < raw_pred ? mx : raw_pred);
            }
            wp_pred = (raw_pred + 3) >> 3;
#endif

            // Properties calculation: recurrent gradient properties
            int32_t prop8 = (int32_t)((uint32_t)cur_left - (uint32_t)prev_grad);
            int32_t prop9 = (int32_t)((uint32_t)cur_left + (uint32_t)cur_top - (uint32_t)cur_topleft);
            prev_grad = prop9;
"#,
    );

    // Standard properties
    for &p in &[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15] {
        if used_properties.contains(&p) {
            match p {
                0 => out.push_str("            int32_t prop0 = 0;\n"),
                1 => out.push_str("            int32_t prop1 = 0;\n"),
                2 => out.push_str("            int32_t prop2 = (int32_t)y;\n"),
                3 => out.push_str("            int32_t prop3 = (int32_t)x;\n"),
                4 => out.push_str("            int32_t prop4 = abs(cur_top);\n"),
                5 => out.push_str("            int32_t prop5 = abs(cur_left);\n"),
                6 => out.push_str("            int32_t prop6 = cur_top;\n"),
                7 => out.push_str("            int32_t prop7 = cur_left;\n"),
                10 => out.push_str(
                    "            int32_t prop10 = (int32_t)((uint32_t)cur_left - (uint32_t)cur_topleft);\n",
                ),
                11 => out.push_str(
                    "            int32_t prop11 = (int32_t)((uint32_t)cur_topleft - (uint32_t)cur_top);\n",
                ),
                12 => out.push_str(
                    "            int32_t prop12 = (int32_t)((uint32_t)cur_top - (uint32_t)cur_topright);\n",
                ),
                13 => out.push_str(
                    "            int32_t prop13 = (int32_t)((uint32_t)cur_top - (uint32_t)cur_toptop);\n",
                ),
                14 => out.push_str(
                    "            int32_t prop14 = (int32_t)((uint32_t)cur_left - (uint32_t)cur_leftleft);\n",
                ),
                15 => out.push_str("            int32_t prop15 = wp_prop;\n"),
                _ => {}
            }
        }
    }

    // Reference channel calculations
    for &ref_idx in &used_ref_channels {
        out.push_str(&format!(
            r#"            int32_t r{ref_idx}_v = r{ref_idx}_row[x];
            int32_t r{ref_idx}_vl = (x > 0) ? r{ref_idx}_row[x - 1] : 0;
            int32_t r{ref_idx}_vt = (y > 0) ? r{ref_idx}_prev[x] : r{ref_idx}_vl;
            int32_t r{ref_idx}_vtl = (x > 0 && y > 0) ? r{ref_idx}_prev[x - 1] : r{ref_idx}_vl;
            int64_t r{ref_idx}_vp = clamped_gradient(r{ref_idx}_vl, r{ref_idx}_vt, r{ref_idx}_vtl);
            int64_t r{ref_idx}_vdiff = (int64_t)r{ref_idx}_v - r{ref_idx}_vp;
"#
        ));
    }

    let mut sorted_ref_props: Vec<u8> = used_properties
        .iter()
        .copied()
        .filter(|&p| p >= 16)
        .collect();
    sorted_ref_props.sort_unstable();
    for p in sorted_ref_props {
        let ref_idx = ((p - 16) / 4) as usize;
        let sub_idx = (p - 16) % 4;
        if ref_idx < input.num_ref_channels {
            match sub_idx {
                0 => out.push_str(&format!(
                    "            int32_t prop{p} = abs(r{ref_idx}_v);\n"
                )),
                1 => out.push_str(&format!("            int32_t prop{p} = r{ref_idx}_v;\n")),
                2 => out.push_str(&format!(
                    "            int32_t prop{p} = (int32_t)llabs(r{ref_idx}_vdiff);\n"
                )),
                3 => out.push_str(&format!(
                    "            int32_t prop{p} = (int32_t)r{ref_idx}_vdiff;\n"
                )),
                _ => unreachable!(),
            }
        } else {
            out.push_str(&format!("            int32_t prop{p} = 0;\n"));
        }
    }

    // 11. Tree decision variables
    out.push_str(
        r#"
            int64_t guess = 0;
            uint32_t cluster = 0;
#if HAS_VARYING_MULTIPLIER
            uint32_t multiplier = 1;
#endif

"#,
    );

    // 12. Generate decision tree code
    emit_tree_code(&mut out, input.tree, 0, has_varying_multiplier);

    // 13. Centralized entropy decode, pixel reconstruct, and WP error update
    out.push_str(
        r#"
#if HAS_SINGLE_SYM
            int32_t residual = SINGLE_SYM;
#elif IS_ANS
            uint32_t token = read_ans(br, &ans_histograms[cluster], ans_state);
            int32_t residual = unpack_signed(decode_hybrid_uint(br, token, cluster));
#else
            uint32_t token = read_huffman(br, huffman_tables[cluster]);
            int32_t residual = unpack_signed(decode_hybrid_uint(br, token, cluster));
#endif

#if HAS_VARYING_MULTIPLIER
            int32_t val = (int32_t)((uint32_t)residual * multiplier) + (int32_t)guess;
#else
            int32_t val = residual + (int32_t)guess;
#endif
            row[x] = val;
            last = val;

#if USES_WP
            int64_t val_bits = (int64_t)val << 3;
            error[cur_row + x + 1] = (int32_t)(raw_pred - val_bits);
            uint32_t e0 = (uint32_t)((llabs(p0 - val_bits) + 3) >> 3);
            uint32_t e1 = (uint32_t)((llabs(p1 - val_bits) + 3) >> 3);
            uint32_t e2 = (uint32_t)((llabs(p2 - val_bits) + 3) >> 3);
            uint32_t e3 = (uint32_t)((llabs(p3 - val_bits) + 3) >> 3);
            pred_errors[cur_row + x][0] = e0;
            pred_errors[cur_row + x][1] = e1;
            pred_errors[cur_row + x][2] = e2;
            pred_errors[cur_row + x][3] = e3;
            pred_errors[prev_row + x + 1][0] += e0;
            pred_errors[prev_row + x + 1][1] += e1;
            pred_errors[prev_row + x + 1][2] += e2;
            pred_errors[prev_row + x + 1][3] += e3;
#endif
        }
    }
}
"#,
    );

    out
}

fn emit_tree_code(
    out: &mut String,
    tree: &[TreeNode],
    node_idx: usize,
    has_varying_multiplier: bool,
) {
    if node_idx >= tree.len() {
        return;
    }

    match tree[node_idx] {
        TreeNode::Split {
            property,
            val,
            left,
            right,
        } => {
            out.push_str(&format!("if (prop{property} > {val}) {{\n"));
            emit_tree_code(out, tree, left as usize, has_varying_multiplier);
            out.push_str("} else {\n");
            emit_tree_code(out, tree, right as usize, has_varying_multiplier);
            out.push_str("}\n");
        }
        TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id,
        } => {
            let pred_expr = get_predictor_expr(predictor);
            let cluster = id as usize;

            if offset == 0 {
                out.push_str(&format!("    guess = ({pred_expr});\n"));
            } else {
                out.push_str(&format!("    guess = ({pred_expr}) + ({offset}LL);\n"));
            }
            out.push_str(&format!("    cluster = {cluster};\n"));
            if has_varying_multiplier {
                out.push_str(&format!("    multiplier = {multiplier}U;\n"));
            }
        }
    }
}

fn get_predictor_expr(predictor: Predictor) -> &'static str {
    match predictor {
        Predictor::Zero => "0LL",
        Predictor::West => "(int64_t)cur_left",
        Predictor::North => "(int64_t)cur_top",
        Predictor::AverageWestAndNorth => "((int64_t)cur_top + (int64_t)cur_left) / 2",
        Predictor::Select => "select_pred(cur_left, cur_top, cur_topleft)",
        Predictor::Gradient => "clamped_gradient(cur_left, cur_top, cur_topleft)",
        Predictor::Weighted => "wp_pred",
        Predictor::NorthEast => "(int64_t)cur_topright",
        Predictor::NorthWest => "(int64_t)cur_topleft",
        Predictor::WestWest => "(int64_t)cur_leftleft",
        Predictor::AverageWestAndNorthWest => "((int64_t)cur_left + (int64_t)cur_topleft) / 2",
        Predictor::AverageNorthAndNorthWest => "((int64_t)cur_top + (int64_t)cur_topleft) / 2",
        Predictor::AverageNorthAndNorthEast => "((int64_t)cur_top + (int64_t)cur_topright) / 2",
        Predictor::AverageAll => {
            "(6LL * cur_top - 2LL * cur_toptop + 7LL * cur_left + cur_leftleft + cur_toprightright + 3LL * cur_topright + 8LL) / 16"
        }
    }
}
