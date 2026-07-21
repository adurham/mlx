// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

MLX_API std::string type_to_name(const Dtype& t);
MLX_API std::string type_to_name(const array& a);

// Compute the grid and block dimensions, check backend/common/utils.h for docs.
MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);
MTL::Size get_2d_grid_dims(const Shape& shape, const Strides& strides);
MTL::Size
get_2d_grid_dims(const Shape& shape, const Strides& strides, size_t divisor);

inline NS::String* make_string(std::ostringstream& os) {
  std::string string = os.str();
  return NS::String::string(string.c_str(), NS::UTF8StringEncoding);
}

inline void debug_set_stream_queue_label(MTL::CommandQueue* queue, int index) {
#ifdef MLX_METAL_DEBUG
  std::ostringstream label;
  label << "Stream " << index;
  queue->setLabel(make_string(label));
#endif
}

inline void debug_set_primitive_buffer_label(
    MTL::CommandBuffer* command_buffer,
    Primitive& primitive) {
#ifdef MLX_METAL_DEBUG
  std::ostringstream label;
  if (auto cbuf_label = command_buffer->label(); cbuf_label) {
    label << cbuf_label->utf8String();
  }
  label << primitive.name();
  command_buffer->setLabel(make_string(label));
#endif
}

// exo-stall-diag (2026-07-21): cheap, ALWAYS-ON variant of the above
// (not gated by MLX_METAL_DEBUG, which isn't defined in the production
// build). Sets the buffer's label to ONLY the most recent primitive
// name encoded into it (overwriting, not accumulating -- so this is a
// single NS::String allocation per eval() call, not a growing one).
// For a live-stuck command buffer, this tells you the LAST op that was
// encoded before commit, i.e. a strong hint at what's executing right
// now if the buffer is still Scheduled/running when read back via
// EXO_CMDBUF_RING_DIAG. No-ops (near-zero cost) unless
// EXO_CMDBUF_RING_DIAG=1 is set, checked once via a cached atomic bool
// to avoid a getenv() on every single eval() call.
inline void debug_set_primitive_buffer_label_cheap(
    MTL::CommandBuffer* command_buffer,
    Primitive& primitive) {
  static const bool enabled = [] {
    auto* env = std::getenv("EXO_CMDBUF_RING_DIAG");
    return env != nullptr && std::string(env) == "1";
  }();
  if (!enabled) {
    return;
  }
  command_buffer->setLabel(
      NS::String::string(primitive.name(), NS::UTF8StringEncoding));
}

template <typename T>
constexpr bool is_numeric_except_char = std::is_arithmetic_v<T> &&
    !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
    !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t>;

template <typename T>
void concatenate(std::string& acc, T first) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
}

template <typename T, typename... Args>
void concatenate(std::string& acc, T first, Args... args) {
  if constexpr (is_numeric_except_char<T>) {
    acc += std::to_string(first);
  } else {
    acc += first;
  }
  concatenate(acc, args...);
}

inline int get_work_per_thread(Dtype dtype) {
  return std::max(1, 8 / dtype.size());
}
inline int get_work_per_thread(Dtype dtype, size_t size) {
  constexpr size_t wpt_threshold = 1 << 16;
  return size < wpt_threshold ? 1 : std::max(1, 8 / dtype.size());
}

inline size_t ceildiv(size_t n, size_t m) {
  return (n + m - 1) / m;
}

inline void check_kernel_threadgroup_size(
    const MTL::ComputePipelineState* kernel,
    MTL::Size group_dims,
    const std::string& name) {
  auto max_size = kernel->maxTotalThreadsPerThreadgroup();
  auto requested_size = group_dims.width * group_dims.height * group_dims.depth;

  if (max_size < requested_size) {
    std::ostringstream msg;
    msg << "Maximum threads per threadgroup is " << max_size
        << " but requested " << requested_size << " for kernel " << name << ".";
    throw std::runtime_error(msg.str());
  }
}

} // namespace mlx::core
