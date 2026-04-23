// Copyright © 2026 Apple Inc.
// Quantized-KV variant of sdpa_vector_2pass_1. Reuses MLX's quantized
// helpers (`load_vector`, `qdot`, `dequantize`) from quantized.h so
// bit-widths 2/3/4/5/6/8 all work through the same template. See
// docs/minimax-quantized-sdpa-design.md for the design rationale.
//
// WIP — not yet instantiated in scaled_dot_product_attention.metal.
// The template is inert until the macro is added there alongside the
// existing `instantiate_sdpa_vector_heads`. Kept in a separate header
// so a syntax error here can't break the bf16 path before we wire in
// the C++ dispatch + Python binding.

#pragma once

#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

// Bit width and group_size are *template* parameters — specialize at
// kernel compile time so the pack layout is baked into instructions.
// Function constants (slot 20–26) for mask / causal / sinks are shared
// with sdpa_vector.h when both headers land in the same .metal TU.

template <typename T, int D, int V, int BITS, int GROUP_SIZE>
[[kernel]] void sdpa_vector_2pass_1_quant(
    // ── unchanged from sdpa_vector_2pass_1 ──
    const device T* queries [[buffer(0)]],
    // ── NEW: quantized K (packed bytes + scales + biases) ──
    const device uint8_t* k_packed [[buffer(1)]],
    const device T* k_scales       [[buffer(2)]],
    const device T* k_biases       [[buffer(3)]],
    // ── NEW: quantized V (same shape) ──
    const device uint8_t* v_packed [[buffer(4)]],
    const device T* v_scales       [[buffer(5)]],
    const device T* v_biases       [[buffer(6)]],
    // ── unchanged ──
    device T* out                  [[buffer(7)]],
    device float* sums             [[buffer(8)]],
    device float* maxs             [[buffer(9)]],
    const constant int& N          [[buffer(11)]],
    // Packed-K strides are in BYTES; scale/bias strides are in ELEMENTS.
    const constant size_t& k_head_stride_bytes  [[buffer(12)]],
    const constant size_t& k_seq_stride_bytes   [[buffer(13)]],
    const constant size_t& k_head_stride_scale  [[buffer(14)]],
    const constant size_t& k_seq_stride_scale   [[buffer(15)]],
    const constant size_t& v_head_stride_bytes  [[buffer(16)]],
    const constant size_t& v_seq_stride_bytes   [[buffer(17)]],
    const constant size_t& v_head_stride_scale  [[buffer(18)]],
    const constant size_t& v_seq_stride_scale   [[buffer(19)]],
    const constant float& scale    [[buffer(20)]],
    // Optional mask / sinks reuse sdpa_vector.h's function constants
    // (has_mask=20, bool_mask=23, float_mask=24, has_sinks=25).
    // Buffer slots are shifted up by the extra quant arguments.
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  // Values-per-thread along head_dim. For D=128, BD=32 → 4.
  // Must be a multiple of the bit-width's natural group (8 for 5-bit,
  // 4 for 4-bit, etc) — enforced by the static_assert below.
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();

  // Each thread processes qk_per_thread values. For BITS=5, pack_factor
  // is 8 (8 values per 5-byte group). qk_per_thread=4 would not align
  // to an 8-value pack. For now require per-thread coverage to be a
  // multiple of pack_factor — at D=128, BD=32 this requires BITS ∈ {2,4,8}
  // to use qk_per_thread=4; BITS=5 would need D=256 or BD=16 to align.
  // Workaround for BITS=5, D=128: widen qk_per_thread=8 (fewer threads
  // active, each doing more work). Addressed in Session 3.
  static_assert(
      qk_per_thread % pack_factor == 0 || BITS == 2 || BITS == 4 || BITS == 8,
      "qk_per_thread must align to quantization pack_factor; "
      "for BITS=5, D=128 needs per-thread widening (see TODO).");
  static_assert(
      GROUP_SIZE % pack_factor == 0,
      "group_size must be a multiple of pack_factor for clean scale reads");

  typedef float U;

  thread U q_thread[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions — matches sdpa_vector_2pass_1's layout.
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
  const int q_offset = o_offset;  // query_transposed handled in full variant

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  // Packed-byte K pointer advances by k_head_stride_bytes per head etc.
  const int thread_byte_offset = (simd_lid * qk_per_thread) * BITS / 8;
  k_packed += kv_batch_head_idx * k_head_stride_bytes
            + block_idx * k_seq_stride_bytes
            + thread_byte_offset;
  v_packed += kv_batch_head_idx * v_head_stride_bytes
            + block_idx * v_seq_stride_bytes
            + (simd_lid * v_per_thread) * BITS / 8;

  // Scale/bias pointers indexed by group: one scale/bias per group_size
  // elements. Each thread needs the scale for its group.
  const int k_group_idx_for_thread = (simd_lid * qk_per_thread) / GROUP_SIZE;
  const int v_group_idx_for_thread = (simd_lid * v_per_thread) / GROUP_SIZE;
  const int k_groups_per_row = D / GROUP_SIZE;
  const int v_groups_per_row = V / GROUP_SIZE;

  k_scales += kv_batch_head_idx * k_head_stride_scale
            + block_idx * k_seq_stride_scale
            + k_group_idx_for_thread;
  k_biases += kv_batch_head_idx * k_head_stride_scale
            + block_idx * k_seq_stride_scale
            + k_group_idx_for_thread;
  v_scales += kv_batch_head_idx * v_head_stride_scale
            + block_idx * v_seq_stride_scale
            + v_group_idx_for_thread;
  v_biases += kv_batch_head_idx * v_head_stride_scale
            + block_idx * v_seq_stride_scale
            + v_group_idx_for_thread;

  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // ── Pre-scale Q using load_vector (from quantized.h) ──
  // This lets us use `qdot` on packed K bytes directly.
  U q_sum = load_vector<T, U, qk_per_thread, BITS>(queries, q_thread);
  // Apply attention scale to Q (absorbs into the dot product).
  #pragma unroll
  for (int i = 0; i < qk_per_thread; i++) {
    q_thread[i] *= static_cast<U>(scale);
  }
  q_sum *= static_cast<U>(scale);

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // ── For each K position ──
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    }
    // TODO(session-3): mask support

    if (use_key) {
      // Load this K-row's scale/bias for this thread's group.
      U k_scale = static_cast<U>(*k_scales);
      U k_bias = static_cast<U>(*k_biases);

      // Compute partial score using qdot — packed K bytes,
      // pre-scaled Q, scale, bias, and Q's sum (bias correction).
      U score = qdot<U, qk_per_thread, BITS>(
          k_packed, q_thread, k_scale, k_bias, q_sum);
      score = simd_sum(score);

      // Online softmax update.
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // ── Dequantize this V-row into thread registers and accumulate ──
      U v_scale = static_cast<U>(*v_scales);
      U v_bias = static_cast<U>(*v_biases);

      // Per-thread V dequantize. Inline the dequantize<> body (which
      // writes to threadgroup U*) as a thread-local variant. For
      // v_per_thread values we inline the bit-ops for each supported
      // BITS. TODO(session-3): factor this out of quantized.h's
      // `dequantize` helper into a `dequantize_thread` variant so this
      // block doesn't duplicate logic.
      thread U v_deq[v_per_thread];
      {
        // Placeholder: replace with proper per-bit-width unpack mirroring
        // quantized.h:483's `dequantize<>`. Expressed here as BITS=8
        // fallback so the scaffold compiles; 4-bit / 5-bit cases filled
        // in next session.
        if (BITS == 8) {
          #pragma unroll
          for (int j = 0; j < v_per_thread; j++) {
            v_deq[j] = v_scale * U(v_packed[j]) + v_bias;
          }
        } else {
          // TODO(session-3): 4-bit and 5-bit unpack, mirroring the
          // dequantize<U,N,bits>() helper at
          // mlx/backend/metal/kernels/quantized.h:483-561. See
          // docs/minimax-quantized-sdpa-design.md §"Kernel specification".
          #pragma unroll
          for (int j = 0; j < v_per_thread; j++) {
            v_deq[j] = 0;
          }
        }
      }

      #pragma unroll
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * v_deq[j];
      }
    }

    // Advance pointers by `blocks` rows of K/V. Stride is given in
    // elements for scale/bias (one per group-row) and in bytes for
    // the packed data.
    k_packed += blocks * k_seq_stride_bytes;
    v_packed += blocks * v_seq_stride_bytes;
    k_scales += blocks * k_seq_stride_scale;
    k_biases += blocks * k_seq_stride_scale;
    v_scales += blocks * v_seq_stride_scale;
    v_biases += blocks * v_seq_stride_scale;
  }

  // ── Partial output per block ──
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  #pragma unroll
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Note: the second pass (`sdpa_vector_2pass_2` in sdpa_vector.h) is
// quantization-agnostic — it only reads the per-block partial outputs /
// sums / maxs that the first pass produced in unquantized form — so we
// reuse it as-is for the quant variant. No `_quant` version needed.
