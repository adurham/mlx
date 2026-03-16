#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

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

#define instantiate_sdpa_split_scores(type, qk_dim)               \
  instantiate_kernel(                                              \
      "sdpa_vector_split_scores_" #type "_" #qk_dim,               \
      sdpa_vector_split_scores,                                    \
      type,                                                        \
      qk_dim)

#define instantiate_sdpa_split_values(type, qk_dim, value_dim)     \
  instantiate_kernel(                                              \
      "sdpa_vector_split_values_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_split_values,                                    \
      type,                                                        \
      qk_dim,                                                      \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector(type, 512, 512)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256) \
  instantiate_sdpa_vector_aggregation(type, 512) \
  instantiate_sdpa_split_scores(type, 64)        \
  instantiate_sdpa_split_scores(type, 96)        \
  instantiate_sdpa_split_scores(type, 128)       \
  instantiate_sdpa_split_scores(type, 256)       \
  instantiate_sdpa_split_scores(type, 512)       \
  instantiate_sdpa_split_values(type, 64, 64)    \
  instantiate_sdpa_split_values(type, 96, 96)    \
  instantiate_sdpa_split_values(type, 128, 128)  \
  instantiate_sdpa_split_values(type, 256, 256)  \
  instantiate_sdpa_split_values(type, 512, 512)  \
  instantiate_kernel(                            \
      "sdpa_vector_2loop_" #type "_64_64",        \
      sdpa_vector_2loop, type, 64, 64)            \
  instantiate_kernel(                            \
      "sdpa_vector_2loop_" #type "_96_96",        \
      sdpa_vector_2loop, type, 96, 96)            \
  instantiate_kernel(                            \
      "sdpa_vector_2loop_" #type "_128_128",      \
      sdpa_vector_2loop, type, 128, 128)          \
  instantiate_kernel(                            \
      "sdpa_vector_2loop_" #type "_256_256",      \
      sdpa_vector_2loop, type, 256, 256)          \
  instantiate_kernel(                            \
      "sdpa_vector_2loop_" #type "_512_512",      \
      sdpa_vector_2loop, type, 512, 512)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
