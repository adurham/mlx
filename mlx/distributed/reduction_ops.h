// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mlx::core::distributed::detail {

// NEON-vectorized float sum: process 4 floats per iteration
inline void neon_sum_f32(const float* input, float* output, size_t N) {
  size_t i = 0;
  for (; i + 4 <= N; i += 4) {
    float32x4_t a = vld1q_f32(output + i);
    float32x4_t b = vld1q_f32(input + i);
    vst1q_f32(output + i, vaddq_f32(a, b));
  }
  for (; i < N; i++) {
    output[i] += input[i];
  }
}

// NEON-vectorized float16 sum: process 8 halfs per iteration
inline void neon_sum_f16(const uint16_t* input, uint16_t* output, size_t N) {
  size_t i = 0;
  for (; i + 8 <= N; i += 8) {
    float16x8_t a = vld1q_f16(reinterpret_cast<const __fp16*>(output + i));
    float16x8_t b = vld1q_f16(reinterpret_cast<const __fp16*>(input + i));
    vst1q_f16(reinterpret_cast<__fp16*>(output + i), vaddq_f16(a, b));
  }
  for (; i < N; i++) {
    // Scalar fallback for remainder
    __fp16 a = *reinterpret_cast<const __fp16*>(output + i);
    __fp16 b = *reinterpret_cast<const __fp16*>(input + i);
    *reinterpret_cast<__fp16*>(output + i) = a + b;
  }
}

template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, size_t N) const {
    if constexpr (std::is_same_v<T, float>) {
      neon_sum_f32(input, output, N);
    } else if constexpr (sizeof(T) == 2) {
      // bfloat16 / float16 — treat as uint16_t for NEON
      neon_sum_f16(
          reinterpret_cast<const uint16_t*>(input),
          reinterpret_cast<uint16_t*>(output),
          N);
    } else {
      while (N-- > 0) {
        *output += *input;
        input++;
        output++;
      }
    }
  }
};

template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = std::max(*output, *input);
      input++;
      output++;
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = std::min(*output, *input);
      input++;
      output++;
    }
  }
};

} // namespace mlx::core::distributed::detail
