// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

pub type JitDecodeChannelFn = unsafe extern "C" fn(
    channel_data: *mut i32,
    stride: usize,
    ysize: usize,
    br: *mut c_void,
    entropy_tables: *const c_void,
    ans_state: *mut u32,
    ref_channels: *const c_void,
);

pub struct JitModule {
    handle: *mut c_void,
    pub decode_channel: JitDecodeChannelFn,
}

// SAFETY: The compiled .so handle and decode function are thread-safe to transfer across threads.
unsafe impl Send for JitModule {}
// SAFETY: The compiled .so handle and decode function are thread-safe to reference across threads.
unsafe impl Sync for JitModule {}

impl Drop for JitModule {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: self.handle is a valid pointer returned by dlopen and not yet closed.
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }
}

static IN_MEMORY_CACHE: OnceLock<Mutex<HashMap<String, Arc<JitModule>>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<String, Arc<JitModule>>> {
    IN_MEMORY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

const CACHE_DIR: &str = "/tmp/jxl_jit_cache";

pub fn get_or_compile_module(c_code: &str) -> Result<Arc<JitModule>, String> {
    let mut hasher = DefaultHasher::new();
    c_code.hash(&mut hasher);
    let hash_key = format!("{:016x}", hasher.finish());

    // 1. Check in-memory cache
    {
        let cache = get_cache().lock().unwrap();
        if let Some(module) = cache.get(&hash_key) {
            return Ok(module.clone());
        }
    }

    // 2. Ensure cache directory exists
    fs::create_dir_all(CACHE_DIR).map_err(|e| format!("Failed to create JIT cache dir: {e}"))?;

    let c_path = PathBuf::from(CACHE_DIR).join(format!("tree_{hash_key}.c"));
    let so_path = PathBuf::from(CACHE_DIR).join(format!("tree_{hash_key}.so"));

    // 3. Compile if .so does not exist permanently in /tmp
    if !so_path.exists() {
        fs::write(&c_path, c_code).map_err(|e| format!("Failed to write C source file: {e}"))?;

        let tmp_so_path =
            PathBuf::from(CACHE_DIR).join(format!("tree_{hash_key}.tmp.{}.so", std::process::id()));

        let output = Command::new("clang")
            .arg("-O3")
            .arg("-shared")
            .arg("-fPIC")
            .arg("-fomit-frame-pointer")
            .arg("-march=native")
            .arg("-o")
            .arg(&tmp_so_path)
            .arg(&c_path)
            .output()
            .map_err(|e| format!("Failed to invoke clang: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Clang compilation failed:\n{stderr}"));
        }

        // Atomically rename to final .so to avoid race conditions
        let _ = fs::rename(&tmp_so_path, &so_path);
    }

    // 4. dlopen the .so
    let c_so_path =
        CString::new(so_path.to_str().unwrap()).map_err(|e| format!("Invalid path string: {e}"))?;

    // SAFETY: c_so_path is a valid null-terminated C string.
    let handle = unsafe { libc::dlopen(c_so_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        // SAFETY: dlerror returns a valid null-terminated string or null.
        let err_msg = unsafe {
            let err_ptr = libc::dlerror();
            if err_ptr.is_null() {
                "Unknown dlopen error".to_string()
            } else {
                std::ffi::CStr::from_ptr(err_ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        return Err(format!(
            "dlopen failed for {}: {}",
            so_path.display(),
            err_msg
        ));
    }

    // 5. dlsym jit_decode_channel
    let sym_name = CString::new("jit_decode_channel").unwrap();
    // SAFETY: handle is a valid dlopen handle and sym_name is a valid null-terminated C string.
    let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
    if sym.is_null() {
        // SAFETY: handle is valid and we close it before returning error.
        unsafe { libc::dlclose(handle) };
        return Err("Symbol jit_decode_channel not found in compiled .so".to_string());
    }

    // SAFETY: The function signature in C matches JitDecodeChannelFn.
    let decode_channel: JitDecodeChannelFn = unsafe { std::mem::transmute(sym) };
    let module = Arc::new(JitModule {
        handle,
        decode_channel,
    });

    // 6. Insert into in-memory cache
    {
        let mut cache = get_cache().lock().unwrap();
        cache.insert(hash_key, module.clone());
    }

    Ok(module)
}
