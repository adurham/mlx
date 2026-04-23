#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"
// Quantized-KV variant. MUST be included after sdpa_vector.h so the
// shared function_constants (do_causal, blocks, ...) are in scope.
// quantized.h transitively needs steel/gemm.h + quantized_utils.h for
// its heavyweight QMM kernels (we only use the small inline helpers
// `load_vector`, `qdot`, `dequantize<>`), so include them here to keep
// the TU fully resolvable — mirrors the include chain in quantized.metal.
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector_quant.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 192, 192)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 192) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)

// Quantized-KV SDPA instantiations.
// head_dims: 64, 128 (MiniMax + Qwen + most current models)
// bits     : 4, 5, 8 (5 for MiniMax-M2.7, 4/8 for completeness)
// group    : 64 (the mlx-lm default)
// V and Q dtypes: float / bfloat16_t / float16_t.
#define instantiate_sdpa_vector_quant(type, qk_dim, value_dim, bits, grp) \
  instantiate_kernel(                                                     \
      "sdpa_vector_2pass_1_quant_" #type "_" #qk_dim "_" #value_dim "_"   \
      #bits "_" #grp,                                                     \
      sdpa_vector_2pass_1_quant,                                          \
      type,                                                               \
      qk_dim,                                                             \
      value_dim,                                                          \
      bits,                                                               \
      grp)

#define instantiate_sdpa_vector_quant_heads(type)          \
  instantiate_sdpa_vector_quant(type, 64, 64, 4, 64)       \
  instantiate_sdpa_vector_quant(type, 64, 64, 5, 64)       \
  instantiate_sdpa_vector_quant(type, 64, 64, 8, 64)       \
  instantiate_sdpa_vector_quant(type, 128, 128, 4, 64)     \
  instantiate_sdpa_vector_quant(type, 128, 128, 5, 64)     \
  instantiate_sdpa_vector_quant(type, 128, 128, 8, 64)

instantiate_sdpa_vector_quant_heads(float)
instantiate_sdpa_vector_quant_heads(bfloat16_t)
instantiate_sdpa_vector_quant_heads(float16_t)
    // clang-format on
