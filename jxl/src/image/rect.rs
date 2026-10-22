// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::DataTypeTag;
use crate::util::ShiftRightCeil;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rect {
    pub origin: (usize, usize),
    // width, height
    pub size: (usize, usize),
}

impl Rect {
    pub fn check_within(&self, size: (usize, usize)) {
        if self.origin.0.checked_add(self.size.0).unwrap() > size.0
            || self.origin.1.checked_add(self.size.1).unwrap() > size.1
        {
            panic!(
                "Rect out of bounds: {}x{}+{}+{} rect in {}x{} view",
                self.size.0, self.size.1, self.origin.0, self.origin.1, size.0, size.1
            );
        }
    }

    pub fn to_byte_rect(&self, data_type: DataTypeTag) -> Rect {
        self.to_byte_rect_sz(data_type.size())
    }

    pub fn to_byte_rect_sz(&self, sz: usize) -> Rect {
        Rect {
            origin: (self.origin.0 * sz, self.origin.1),
            size: (self.size.0 * sz, self.size.1),
        }
    }

    pub fn downsample(&self, downsample: (u8, u8)) -> Rect {
        Rect {
            origin: (self.origin.0 >> downsample.0, self.origin.1 >> downsample.1),
            size: (
                self.size.0.shrc(downsample.0),
                self.size.1.shrc(downsample.1),
            ),
        }
    }

    pub fn end(&self) -> (usize, usize) {
        (self.origin.0 + self.size.0, self.origin.1 + self.size.1)
    }

    pub fn clip(&self, size: (usize, usize)) -> Rect {
        let end = self.end();
        Rect {
            origin: self.origin,
            size: (
                end.0.min(size.0).saturating_sub(self.origin.0),
                end.1.min(size.1).saturating_sub(self.origin.1),
            ),
        }
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.end().0 > other.origin.0
            && other.end().0 > self.origin.0
            && self.end().1 > other.origin.1
            && other.end().1 > self.origin.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_downsample_partitions_unaligned_size() {
        for ds in 0..=3u8 {
            let align = 1usize << ds;
            for width in 1..=65usize {
                let mut prev_end = 0;
                for x in (0..width).step_by(align * 3) {
                    let w = (align * 3).min(width - x);
                    let r = Rect {
                        origin: (x, 0),
                        size: (w, width),
                    }
                    .downsample((ds, ds));
                    assert_eq!(r.origin.0, prev_end);
                    prev_end = r.end().0;
                    assert_eq!(r.size.1, width.shrc(ds));
                }
                assert_eq!(prev_end, width.shrc(ds));
            }
        }
    }
}
