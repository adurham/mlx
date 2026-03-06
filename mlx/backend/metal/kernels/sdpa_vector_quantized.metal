#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector_quantized.h"

using namespace metal;

// Fused quantized SDPA kernel instantiations.
// Pass 2 (aggregation) reuses sdpa_vector_2pass_2 from sdpa_vector.h since
// it only reduces float intermediates and never touches K/V.

#define instantiate_sdpa_vector_quantized(type, qk_dim, value_dim)                  \
  instantiate_kernel(                                                                \
      "sdpa_vector_quantized_" #type "_" #qk_dim "_" #value_dim,                    \
      sdpa_vector_quantized,                                                         \
      type,                                                                          \
      qk_dim,                                                                        \
      value_dim)                                                                     \
  instantiate_kernel(                                                                \
      "sdpa_vector_2pass_1_quantized_" #type "_" #qk_dim "_" #value_dim,            \
      sdpa_vector_2pass_1_quantized,                                                 \
      type,                                                                          \
      qk_dim,                                                                        \
      value_dim)

#define instantiate_sdpa_vector_quantized_heads(type)      \
  instantiate_sdpa_vector_quantized(type, 64, 64)          \
  instantiate_sdpa_vector_quantized(type, 96, 96)          \
  instantiate_sdpa_vector_quantized(type, 128, 128)        \
  instantiate_sdpa_vector_quantized(type, 256, 256)        \
  instantiate_sdpa_vector_quantized(type, 512, 512)

instantiate_sdpa_vector_quantized_heads(float16_t)
instantiate_sdpa_vector_quantized_heads(bfloat16_t)
// clang-format on
