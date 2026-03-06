// Copyright © 2024-2026 Apple Inc.
// Fused quantized SDPA: reads int-quantized K/V with on-the-fly dequantization.
// Based on sdpa_vector.h — same online softmax / simdgroup structure.

#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant bool q_has_mask [[function_constant(20)]];
constant bool q_query_transposed [[function_constant(21)]];
constant bool q_do_causal [[function_constant(22)]];
constant bool q_bool_mask [[function_constant(23)]];
constant bool q_float_mask [[function_constant(24)]];
constant int q_blocks [[function_constant(26)]];

// Inline dequantize: extract uint8 from packed uint32 and convert to float.
// For 8-bit: each uint32 holds 4 packed uint8 values.
template <typename U>
inline U dequant8(uint32_t packed, int idx, U scale, U bias) {
  uint8_t val = (packed >> (idx * 8)) & 0xFF;
  return fma(scale, static_cast<U>(val), bias);
}

// Dequantize all 4 bytes from a packed uint32 into a float4-like struct.
template <typename U>
struct dequant4_result { U v0, v1, v2, v3; };

template <typename U>
inline dequant4_result<U> dequant4(uint32_t packed, U scale, U bias) {
  uchar4 bytes = as_type<uchar4>(packed);
  return {
    fma(scale, static_cast<U>(bytes.x), bias),
    fma(scale, static_cast<U>(bytes.y), bias),
    fma(scale, static_cast<U>(bytes.z), bias),
    fma(scale, static_cast<U>(bytes.w), bias),
  };
}

// Single-pass fused quantized SDPA for decode (small query length).
// K and V are stored as quantized (data: uint32, scales: T, biases: T).
template <typename T, int D, int V_DIM = D>
[[kernel]] void sdpa_vector_quantized(
    const device T* queries [[buffer(0)]],
    const device uint32_t* k_data [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* v_data [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& group_size [[buffer(10)]],
    // K strides (in elements of their respective types)
    const constant size_t& k_head_stride [[buffer(11)]],
    const constant size_t& k_seq_stride [[buffer(12)]],
    const constant size_t& ks_head_stride [[buffer(13)]],
    const constant size_t& ks_seq_stride [[buffer(14)]],
    // V strides
    const constant size_t& v_head_stride [[buffer(15)]],
    const constant size_t& v_seq_stride [[buffer(16)]],
    const constant size_t& vs_head_stride [[buffer(17)]],
    const constant size_t& vs_seq_stride [[buffer(18)]],
    const constant float& scale [[buffer(19)]],
    const device bool* bmask [[buffer(20), function_constant(q_bool_mask)]],
    const device T* fmask [[buffer(21), function_constant(q_float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(22), function_constant(q_has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(23), function_constant(q_has_mask)]],
    const constant int& mask_head_stride
    [[buffer(24), function_constant(q_has_mask)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V_DIM / BD;
  // For 8-bit quant: 4 values packed per uint32
  constexpr int pack_factor = 4;
  constexpr int qk_packed_per_thread = (qk_per_thread + pack_factor - 1) / pack_factor;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      q_query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;

  // K data pointers: advance to kv_head, then simd_gid selects the starting seq position.
  // k_data shape: (B, n_kv_heads, N, D/pack_factor) — uint32
  // k_scales shape: (B, n_kv_heads, N, D/group_size) — T
  const device uint32_t* k_d = k_data + kv_head_idx * k_head_stride +
      simd_gid * k_seq_stride + simd_lid * qk_packed_per_thread;
  const device T* k_s = k_scales + kv_head_idx * ks_head_stride +
      simd_gid * ks_seq_stride;
  const device T* k_b = k_biases + kv_head_idx * ks_head_stride +
      simd_gid * ks_seq_stride;

  // V data pointers
  const device uint32_t* v_d = v_data + kv_head_idx * v_head_stride +
      simd_gid * v_seq_stride + simd_lid * (v_per_thread / pack_factor);
  const device T* v_s = v_scales + kv_head_idx * vs_head_stride +
      simd_gid * vs_seq_stride;
  const device T* v_b = v_biases + kv_head_idx * vs_head_stride +
      simd_gid * vs_seq_stride;

  int inner_k_d_stride = BN * int(k_seq_stride);
  int inner_k_s_stride = BN * int(ks_seq_stride);
  int inner_v_d_stride = BN * int(v_seq_stride);
  int inner_v_s_stride = BN * int(vs_seq_stride);

  if (q_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (q_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V_DIM + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e38;
  U sum_exp_score = 0;

  // Determine which scale/bias group this thread's elements belong to
  // Thread simd_lid handles elements [simd_lid * qk_per_thread .. (simd_lid+1) * qk_per_thread - 1]
  const int elem_start = simd_lid * qk_per_thread;
  const int v_elem_start = simd_lid * v_per_thread;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (q_do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (q_bool_mask) {
      use_key = bmask[0];
    } else if (q_float_mask) {
      use_key = (fmask[0] >= -1e38);
    }
    if (use_key) {
      // Prefetch both K and V packed data — overlaps memory latency
      auto k_packed = k_d[0];
      auto v_packed = v_d[0];

      // Dequantize key
      int k_group = elem_start / group_size;
      U ks = static_cast<U>(k_s[k_group]);
      U kb = static_cast<U>(k_b[k_group]);
      auto kv = dequant4(k_packed, ks, kb);

      U score = q[0] * kv.v0 + q[1] * kv.v1 + q[2] * kv.v2 + q[3] * kv.v3;
      score = simd_sum(score);
      if (q_float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators (online softmax)
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Dequantize value (using prefetched data)
      int v_group = v_elem_start / group_size;
      U vs_val = static_cast<U>(v_s[v_group]);
      U vb_val = static_cast<U>(v_b[v_group]);
      auto vv = dequant4(v_packed, vs_val, vb_val);

      o[0] = o[0] * factor + exp_score * vv.v0;
      o[1] = o[1] * factor + exp_score * vv.v1;
      o[2] = o[2] * factor + exp_score * vv.v2;
      o[3] = o[3] * factor + exp_score * vv.v3;
    }

    // Move the pointers to the next kv
    k_d += inner_k_d_stride;
    k_s += inner_k_s_stride;
    k_b += inner_k_s_stride;
    v_d += inner_v_d_stride;
    v_s += inner_v_s_stride;
    v_b += inner_v_s_stride;
    if (q_bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (q_float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.
  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// Two-pass fused quantized SDPA — pass 1 (block-level partial results).
template <typename T, int D, int V_DIM = D>
[[kernel]] void sdpa_vector_2pass_1_quantized(
    const device T* queries [[buffer(0)]],
    const device uint32_t* k_data [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* v_data [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& group_size [[buffer(11)]],
    // K strides
    const constant size_t& k_head_stride [[buffer(12)]],
    const constant size_t& k_seq_stride [[buffer(13)]],
    const constant size_t& ks_head_stride [[buffer(14)]],
    const constant size_t& ks_seq_stride [[buffer(15)]],
    // V strides
    const constant size_t& v_head_stride [[buffer(16)]],
    const constant size_t& v_seq_stride [[buffer(17)]],
    const constant size_t& vs_head_stride [[buffer(18)]],
    const constant size_t& vs_seq_stride [[buffer(19)]],
    const constant float& scale [[buffer(20)]],
    const device bool* bmask [[buffer(21), function_constant(q_bool_mask)]],
    const device T* fmask [[buffer(22), function_constant(q_float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(23), function_constant(q_has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(24), function_constant(q_has_mask)]],
    const constant int& mask_head_stride
    [[buffer(25), function_constant(q_has_mask)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V_DIM / BD;
  constexpr int pack_factor = 4;
  constexpr int qk_packed_per_thread = (qk_per_thread + pack_factor - 1) / pack_factor;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      q_query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // K data pointers
  const device uint32_t* k_d = k_data + kv_batch_head_idx * k_head_stride +
      block_idx * k_seq_stride + simd_lid * qk_packed_per_thread;
  const device T* k_s = k_scales + kv_batch_head_idx * ks_head_stride +
      block_idx * ks_seq_stride;
  const device T* k_b = k_biases + kv_batch_head_idx * ks_head_stride +
      block_idx * ks_seq_stride;

  // V data pointers
  const device uint32_t* v_d = v_data + kv_batch_head_idx * v_head_stride +
      block_idx * v_seq_stride + simd_lid * (v_per_thread / pack_factor);
  const device T* v_s = v_scales + kv_batch_head_idx * vs_head_stride +
      block_idx * vs_seq_stride;
  const device T* v_b = v_biases + kv_batch_head_idx * vs_head_stride +
      block_idx * vs_seq_stride;

  out += o_offset * q_blocks * V_DIM + block_idx * V_DIM + simd_lid * v_per_thread;

  if (q_bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (q_float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * q_blocks + block_idx;
  maxs += o_offset * q_blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = -1e38;
  U sum_exp_score = 0;

  const int elem_start = simd_lid * qk_per_thread;
  const int v_elem_start = simd_lid * v_per_thread;

  // For each key in this block
  for (int i = block_idx; i < N; i += q_blocks) {
    bool use_key = true;
    if (q_do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (q_bool_mask) {
      use_key = bmask[0];
    } else if (q_float_mask) {
      use_key = (fmask[0] >= -1e38);
    }
    if (use_key) {
      // Prefetch both K and V packed data — overlaps memory latency
      auto k_packed = k_d[0];
      auto v_packed = v_d[0];

      // Dequantize key
      int k_group = elem_start / group_size;
      U ks = static_cast<U>(k_s[k_group]);
      U kb = static_cast<U>(k_b[k_group]);
      auto kv = dequant4(k_packed, ks, kb);

      U score = q[0] * kv.v0 + q[1] * kv.v1 + q[2] * kv.v2 + q[3] * kv.v3;
      score = simd_sum(score);

      if (q_float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Dequantize value (using prefetched data)
      int v_group = v_elem_start / group_size;
      U vs_val = static_cast<U>(v_s[v_group]);
      U vb_val = static_cast<U>(v_b[v_group]);
      auto vv = dequant4(v_packed, vs_val, vb_val);

      o[0] = o[0] * factor + exp_score * vv.v0;
      o[1] = o[1] * factor + exp_score * vv.v1;
      o[2] = o[2] * factor + exp_score * vv.v2;
      o[3] = o[3] * factor + exp_score * vv.v3;
    }

    // Move to next block position
    k_d += q_blocks * int(k_seq_stride);
    k_s += q_blocks * int(ks_seq_stride);
    k_b += q_blocks * int(ks_seq_stride);
    v_d += q_blocks * int(v_seq_stride);
    v_s += q_blocks * int(vs_seq_stride);
    v_b += q_blocks * int(vs_seq_stride);
    if (q_bool_mask) {
      bmask += q_blocks * mask_kv_seq_stride;
    }
    if (q_float_mask) {
      fmask += q_blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Note: The 2-pass aggregation kernel (pass 2) is identical to the non-quantized
// version since it reduces float intermediates. Reuse sdpa_vector_2pass_2 from
// sdpa_vector.h.
