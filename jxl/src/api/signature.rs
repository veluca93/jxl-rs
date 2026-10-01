// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// The magic bytes for a bare JPEG XL codestream.
const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];
/// The magic bytes for a file using the JPEG XL container format.
pub(crate) const CONTAINER_SIGNATURE: [u8; 12] =
    [0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xd, 0xa, 0x87, 0xa];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JxlSignature {
    Codestream,
    Container,
    Invalid,
    NeedsMoreInput { size_hint: usize },
}

/// Checks if the given buffer starts with a valid JPEG XL signature.
pub fn check_signature(file_prefix: &[u8]) -> JxlSignature {
    for (sig, ret) in [
        (&CODESTREAM_SIGNATURE[..], JxlSignature::Codestream),
        (&CONTAINER_SIGNATURE[..], JxlSignature::Container),
    ] {
        if file_prefix.starts_with(sig) {
            return ret;
        }
        if sig.starts_with(file_prefix) {
            return JxlSignature::NeedsMoreInput {
                size_hint: sig.len() - file_prefix.len(),
            };
        }
    }
    JxlSignature::Invalid
}

#[cfg(test)]
mod tests {
    use super::{CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE, JxlSignature, check_signature};

    #[test]
    fn signature_detection() {
        let mut container_extra = CONTAINER_SIGNATURE.to_vec();
        container_extra.extend_from_slice(&[0x11, 0x22, 0x33]);
        let mut codestream_extra = CODESTREAM_SIGNATURE.to_vec();
        codestream_extra.extend_from_slice(&[0x44, 0x55, 0x66]);

        let cases: &[(&[u8], JxlSignature)] = &[
            (
                &[],
                JxlSignature::NeedsMoreInput {
                    size_hint: CODESTREAM_SIGNATURE.len(),
                },
            ),
            (
                &CODESTREAM_SIGNATURE[..1],
                JxlSignature::NeedsMoreInput {
                    size_hint: CODESTREAM_SIGNATURE.len() - 1,
                },
            ),
            (&CODESTREAM_SIGNATURE, JxlSignature::Codestream),
            (&codestream_extra, JxlSignature::Codestream),
            (
                &CONTAINER_SIGNATURE[..5],
                JxlSignature::NeedsMoreInput {
                    size_hint: CONTAINER_SIGNATURE.len() - 5,
                },
            ),
            (&CONTAINER_SIGNATURE, JxlSignature::Container),
            (&container_extra, JxlSignature::Container),
            (&[0x12, 0x34, 0x56, 0x77], JxlSignature::Invalid),
        ];

        for (prefix, expected) in cases {
            assert_eq!(check_signature(prefix), *expected, "prefix: {prefix:?}");
        }
    }
}
