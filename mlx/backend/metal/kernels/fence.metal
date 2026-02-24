// Copyright © 2024 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to update timestamp
[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to spin wait for timestamp value
// Two-tier strategy: fast path (volatile reads) for the common coherent case,
// then fallback to system-scope atomic load to break through stale GPU caches.
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  constexpr uint fast_path_limit = 1000000;
  uint iters = 0;
  while (1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    if (timestamp[0] >= value) {
      break;
    }
    iters++;
    if (iters >= fast_path_limit) {
      // Reinterpret the buffer as atomic and perform an atomic load.
      // This forces the GPU to re-fetch from the coherence point rather
      // than reading a potentially stale cached copy.
      device metal::atomic_uint* atomic_ts =
          (device metal::atomic_uint*)timestamp;
      uint val = atomic_load_explicit(
          atomic_ts, metal::memory_order_relaxed);
      if (val >= value) {
        break;
      }
      // Reset counter to avoid overflow and retry the fast path briefly
      iters = 0;
    }
  }
}
