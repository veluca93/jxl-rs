// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::OwnedRawImage;
use std::sync::Mutex;

pub const NUM_SIZE_CLASSES: usize = 14;
pub const COMMON_SIZES: [usize; NUM_SIZE_CLASSES] = [
    1024,    // 1 KB (LF group size i32)
    2048,    // 2 KB
    4096,    // 4 KB
    8192,    // 8 KB
    16384,   // 16 KB (group size u8)
    32768,   // 32 KB
    65536,   // 64 KB (group size f32)
    131072,  // 128 KB
    262144,  // 256 KB (256x256 f32)
    524288,  // 512 KB (512x256 f32)
    1048576, // 1 MB (1024x256 f32)
    2097152, // 2 MB
    4194304, // 4 MB
    8388608, // 8 MB
];

struct BufferPoolInner {
    // bins[i] stores buffers of size COMMON_SIZES[i]
    bins: [Vec<OwnedRawImage>; NUM_SIZE_CLASSES],
    max_bytes: usize,
    current_bytes: usize,
    hits: usize,
    misses: usize,
    frees: usize,
    drops: usize,
}

pub struct BufferPool(Mutex<BufferPoolInner>);

impl BufferPool {
    pub fn new(max_bytes: usize) -> Self {
        Self(Mutex::new(BufferPoolInner {
            bins: std::array::from_fn(|_| Vec::new()),
            max_bytes,
            current_bytes: 0,
            hits: 0,
            misses: 0,
            frees: 0,
            drops: 0,
        }))
    }

    /// Returns a buffer of size at least `size` (rounded up to a size class).
    /// If no buffer is available in the exact size class or the next larger one,
    /// returns None.
    pub fn alloc(&self, size: usize) -> Option<OwnedRawImage> {
        let mut inner = self.0.lock().unwrap();

        // Find the smallest size class >= size
        let idx = COMMON_SIZES.iter().position(|&s| s >= size);

        let res = (|| {
            let idx = idx?;
            // 1. Try exact size class
            if let Some(img) = inner.bins[idx].pop() {
                inner.current_bytes -= COMMON_SIZES[idx];
                return Some(img);
            }

            // 2. Try immediately next size class (idx + 1) to allow up to 2x oversizing
            if idx + 1 < COMMON_SIZES.len() {
                if let Some(img) = inner.bins[idx + 1].pop() {
                    inner.current_bytes -= COMMON_SIZES[idx + 1];
                    return Some(img);
                }
            }
            None
        })();

        if res.is_some() {
            inner.hits += 1;
        } else {
            inner.misses += 1;
        }
        res
    }

    /// Returns a buffer to the pool.
    /// The buffer's allocation_size must match one of the COMMON_SIZES.
    pub fn free(&self, img: OwnedRawImage) {
        let size = img.allocation_size;
        let mut inner = self.0.lock().unwrap();

        if let Some(idx) = COMMON_SIZES.iter().position(|&s| s == size) {
            if inner.current_bytes + size <= inner.max_bytes {
                inner.bins[idx].push(img);
                inner.current_bytes += size;
                inner.frees += 1;
                return;
            }
        }
        inner.drops += 1;
        // If it doesn't fit or doesn't match a size class, it is implicitly dropped
        // and its memory will be deallocated (since OwnedRawImage's pool is None now,
        // it will go to normal drop).
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        let inner = self.0.get_mut().unwrap();
        eprintln!(
            "BufferPool stats on drop: hits={}, misses={}, frees={}, drops={}, final_bytes={}",
            inner.hits, inner.misses, inner.frees, inner.drops, inner.current_bytes
        );
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.lock().unwrap();
        f.debug_struct("BufferPool")
            .field("max_bytes", &inner.max_bytes)
            .field("current_bytes", &inner.current_bytes)
            .finish()
    }
}
