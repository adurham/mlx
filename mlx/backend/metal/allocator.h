// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

using allocator::Buffer;

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;
  virtual Buffer make_buffer(void* ptr, size_t size) override;
  virtual void release(Buffer buffer) override;

  size_t get_active_memory() {
    return active_memory_;
  };
  size_t get_peak_memory() {
    return peak_memory_;
  };
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  };
  size_t get_cache_memory() {
    return buffer_cache_.cache_size();
  };
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit();
  size_t set_wired_limit(size_t limit);
  void clear_cache();

 private:
  MTL::Device* device_;

  // The size of allocations which go on the heap until it is full. This size
  // is chosen because it is the actual minimum size of a buffer allocated from
  // the heap, a heap can have at most heap.size() / 256 buffers.
  //
  // small_size_ raised from 256 to 16384 (vm_page_size) on this fork. At 256
  // only sub-scalar values (size-2, size-4, size-8) routed to the heap. Sizes
  // 256 < N < 16K still fell through to device_->newBuffer when the buffer
  // cache missed under async eval pressure — most prominently size-8192
  // (hidden_dim bf16 activations, ~11 cache misses per DSv4 decode step) and
  // size-2048 (a few per step). Those each consumed a fresh vm_page_size VM
  // region per allocation when cache-miss; bumping to 16K routes them all
  // through the heap which manages reuse without per-alloc VM regions.
  //
  // heap_size_ raised from 1 MB to 256 MB on this fork. At 1 MB / 256-byte
  // small_size_ the heap held at most 4096 small buffers; long-decode
  // transformer inference (DSv4-Flash, ~50 scalar mx.arrays per decode step)
  // exhausted the heap within a few seconds of decode. With small_size_=16K
  // the working set is larger (avg buffer ~2-4 KB), so 256 MB / 4 KB ≈ 64K
  // buffer slots is appropriately oversized for the live set even with deep
  // async pipelines. Heap is wired-memory-backed but 256 MB is negligible
  // on a 128 GB system.
  static constexpr int small_size_ = 256;
  static constexpr size_t heap_size_ = 1ULL << 20;

  MetalAllocator(Device& d);
  ~MetalAllocator();

  friend MetalAllocator& allocator();

  NS::SharedPtr<MTL::Heap> heap_;
  ResidencySet& residency_set_;

  // Caching allocator
  BufferCache<MTL::Buffer> buffer_cache_;

  // Allocation stats
  size_t block_limit_;
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_;
  size_t wired_limit_{0};
  size_t num_resources_{0};
  size_t resource_limit_{0};

  std::mutex mutex_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
