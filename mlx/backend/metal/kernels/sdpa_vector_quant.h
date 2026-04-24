// Copyright © 2026 Apple Inc.
// Quantized-KV variant of sdpa_vector_2pass_1. Reuses MLX's quantized
// helpers (`load_vector`, `qdot`) from quantized.h for the K side so the
// packed-weight bit layout stays MLX-blessed; the V side uses a
// thread-local dequantize helper (`dequantize_thread`) defined below that
// mirrors the threadgroup `dequantize<>` in quantized.h bit-for-bit.
//
// See docs/minimax-quantized-sdpa-design.md for the design rationale.
//
// NOTE: function constants (`do_causal`, `blocks`, ...) are defined in
// sdpa_vector.h. Include order matters: this header MUST be included
// after sdpa_vector.h in the .metal TU so the constants are in scope.

#pragma once

#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

// Thread-local counterpart of quantized.h's `dequantize<>` helper.
// Writes `N` dequantized values into a thread-private array `w_local`.
// Bit layouts are copied verbatim from quantized.h:483-561 (bits 2, 4,
// 5, 8) so the storage interpretation is identical between the K-side
// `qdot` path (which uses those layouts transitively via quantized.h)
// and the V-side per-thread path (which uses this helper).
//
// For bits ∈ {3, 5} the reference helper writes `pack_factor == 8`
// values per invocation; we match that exactly (`N` must be a multiple
// of 8 for bits=5). For other bits, `N` must be a multiple of 4 (bits=2)
// or 2 (bits=4) or 1 (bits=8). The caller enforces this via
// static_assert on `qk_per_thread % pack_factor == 0`.
template <typename U, int N, int bits>
inline void dequantize_thread(
    const device uint8_t* w, U scale, U bias, thread U* w_local) {
  static_assert(
      bits == 2 || bits == 4 || bits == 5 || bits == 8,
      "Template undefined for bits not in {2, 4, 5, 8}");

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    #pragma unroll
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i + 0] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    #pragma unroll
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i + 0] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 5) {
    // MLX's 5-bit packing: 8 values per 5 bytes. Pattern copied from
    // quantized.h:529-542.
    #pragma unroll
    for (int i = 0; i < (N / 8); i++) {
      const int wl = 8 * i;
      const int wo = 5 * i;
      w_local[wl + 0] = (w[wo + 0] & 0x1f) * scale + bias;
      w_local[wl + 1] =
          (((w[wo + 0] & 0xe0) >> 5) + ((w[wo + 1] & 0x03) << 3)) * scale +
          bias;
      w_local[wl + 2] = ((w[wo + 1] & 0x7c) >> 2) * scale + bias;
      w_local[wl + 3] =
          (((w[wo + 1] & 0x80) >> 7) + ((w[wo + 2] & 0x0f) << 1)) * scale +
          bias;
      w_local[wl + 4] =
          (((w[wo + 2] & 0xf0) >> 4) + ((w[wo + 3] & 0x01) << 4)) * scale +
          bias;
      w_local[wl + 5] = ((w[wo + 3] & 0x3e) >> 1) * scale + bias;
      w_local[wl + 6] =
          (((w[wo + 3] & 0xc0) >> 6) + ((w[wo + 4] & 0x07) << 2)) * scale +
          bias;
      w_local[wl + 7] = ((w[wo + 4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 8) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

// ---------------------------------------------------------------------------
// Branch-free half-pack dequantize for bits=5.
//
// A bits=5 pack group holds 8 values in 5 bytes (40 bits). With
// `qk_per_thread = 8`, one simd lane per pack group leaves half the lanes
// inactive at head_dim ∈ {64, 128} and the simdgroup idles through
// predicated-off cycles. To fix the 50 % occupancy cliff without
// reintroducing simdgroup divergence (the enemy of SIMT throughput), two
// lanes cooperate on each pack group via a branch-free bit-extraction
// formula: `low` and `high` lanes execute *the same* instructions with a
// lane-dependent shift amount, so the simdgroup runs at full rate.
//
//   - low  lane  (half == 0) → values 0..3 (bits  0..19)
//   - high lane  (half == 1) → values 4..7 (bits 20..39)
//
// Both lanes point at byte 0 of the full 5-byte pack group; this helper
// reads bytes 0..4 via a pair of overlapping uint32 loads and slices out
// the requested 5-bit value. At `half * 4 + i` for i ∈ [0, 4), the values
// land at bit positions {0, 5, 10, 15, 20, 25, 30, 35}, all within the
// 40-bit pack group. Reading `bytes_pair = (w[byte+1] << 8) | w[byte]`
// covers bits spanning byte boundaries; the value 7 case (bit_start=35)
// fits entirely in `w[4]` so the `w[byte+1]` read (byte 5) is valid
// scratch past the pack group — the 5-bit mask discards its contribution.
// ---------------------------------------------------------------------------

template <typename U>
inline void dequantize_thread_half5(
    const device uint8_t* w,
    U scale,
    U bias,
    int pack_half,
    thread U* w_local) {
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // pack_half ∈ {0, 1}; i ∈ {0, 1, 2, 3}
    //   ⇒ bit_start ∈ {0, 5, 10, 15, 20, 25, 30, 35}
    const int bit_start   = 5 * (pack_half * 4 + i);
    const int byte_start  = bit_start >> 3;
    const int bit_in_byte = bit_start & 7;
    // Two-byte window; high byte contribution is masked off by & 0x1f
    // when bit_in_byte < 4 so reading one byte past the pack group
    // (pack_group_end + 1) is harmless.
    const uint32_t bytes_pair =
        (uint32_t(w[byte_start + 1]) << 8) | uint32_t(w[byte_start]);
    const uint32_t val = (bytes_pair >> bit_in_byte) & 0x1fu;
    w_local[i] = U(val) * scale + bias;
  }
}

// constexpr helpers used to pick qk/v per-thread counts without pulling in
// <algorithm>.
constexpr int sdpa_quant_max(int a, int b) {
  return a > b ? a : b;
}

// Minimum per-thread element count that load_vector / qdot /
// dequantize_thread will accept for a given bit-width. This is the
// "natural stride" at which the helpers process values: load_vector
// and qdot at bits ∈ {2, 4, 6} read 4 at a time; at bits ∈ {3, 5} they
// read 8 at a time; at bits=8 they read 1 at a time. The thread-local
// V dequantize mirrors the same loop structure.
//
// Exception: bits=5 returns 4 instead of 8. That allows the kernel to
// use `qk_per_thread = 4` at head_dim ∈ {64, 128} and keep all 32 simd
// lanes active via the half-pack dequant helpers (two lanes cooperate
// on one 5-byte pack group — low lane + high lane). Without this
// exception, bits=5 × head_dim=128 only used 16 of 32 lanes and wasted
// half the simdgroup's throughput. See `dequantize_thread_half5_*` above.
template <int bits>
constexpr int sdpa_quant_min_per_thread() {
  return (bits == 8)                                    ? 1
      : (bits == 3)                                     ? 8
      : (bits == 5)                                     ? 4
                                                        : 4;
}

// ---------------------------------------------------------------------------
// sdpa_vector_2pass_1_quant
//
// Mirrors sdpa_vector_2pass_1 (from sdpa_vector.h) but reads K and V from
// packed/scale/bias triples instead of bf16 arrays. Per-thread dequantize
// happens in-register; no threadgroup storage needed for K/V data.
//
// Layout invariants (from QuantizedKVCache):
//   k_packed[..., seq, head_dim * BITS / 8]   dtype=uint8_t
//   k_scales[..., seq, head_dim / GROUP_SIZE] dtype=T
//   k_biases[..., seq, head_dim / GROUP_SIZE] dtype=T
//   (same for v_*)
//
// Strides: *_stride_bytes is in BYTES (matches the uint8_t packed ptr);
// *_stride_scale is in ELEMENTS (matches the bf16 scale/bias ptr).
//
// The second pass (`sdpa_vector_2pass_2` in sdpa_vector.h) is
// quantization-agnostic — it only reads the per-block partials — so we
// reuse it unchanged for the quant path.
// ---------------------------------------------------------------------------
template <typename T, int D, int V, int BITS, int GROUP_SIZE>
[[kernel]] void sdpa_vector_2pass_1_quant(
    const device T* queries             [[buffer(0)]],
    const device uint8_t* k_packed      [[buffer(1)]],
    const device T* k_scales            [[buffer(2)]],
    const device T* k_biases            [[buffer(3)]],
    const device uint8_t* v_packed      [[buffer(4)]],
    const device T* v_scales            [[buffer(5)]],
    const device T* v_biases            [[buffer(6)]],
    device T* out                       [[buffer(7)]],
    device float* sums                  [[buffer(8)]],
    device float* maxs                  [[buffer(9)]],
    const constant int& N               [[buffer(11)]],
    const constant size_t& k_head_stride_bytes  [[buffer(12)]],
    const constant size_t& k_seq_stride_bytes   [[buffer(13)]],
    const constant size_t& k_head_stride_scale  [[buffer(14)]],
    const constant size_t& k_seq_stride_scale   [[buffer(15)]],
    const constant size_t& v_head_stride_bytes  [[buffer(16)]],
    const constant size_t& v_seq_stride_bytes   [[buffer(17)]],
    const constant size_t& v_head_stride_scale  [[buffer(18)]],
    const constant size_t& v_seq_stride_scale   [[buffer(19)]],
    const constant float& scale         [[buffer(20)]],
    uint3 tptg  [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid   [[threadgroup_position_in_grid]],
    uint3 tpg   [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int min_per_thread = sdpa_quant_min_per_thread<BITS>();

  // Each thread owns a contiguous slice of head_dim. `min_per_thread`
  // is the natural stride that load_vector / qdot / dequantize_thread
  // require for this bit-width (4 for bits ∈ {2, 4, 6, 5}; 8 for bits=3;
  // 1 for bits=8). We widen the default D/32 slice to meet that stride;
  // lanes beyond `active_*` stay inactive and contribute 0 via `simd_sum`.
  constexpr int qk_per_thread = sdpa_quant_max(D / 32, min_per_thread);
  constexpr int v_per_thread = sdpa_quant_max(V / 32, min_per_thread);

  // Count of lanes that actually touch memory.
  constexpr int active_k = D / qk_per_thread;
  constexpr int active_v = V / v_per_thread;

  // Half-pack mode: at bits=5 a pack group holds 8 values in 5 bytes, so
  // the straight `qk_per_thread % pack_factor == 0` invariant would force
  // qk_per_thread=8 (i.e., one lane per pack group). That halves simd
  // occupancy at head_dim ∈ {64, 128}. Instead, we pair two lanes per
  // pack group at qk_per_thread=4 and dispatch `dequantize_thread_half5_{low,high}`
  // — see the helpers above this function for bit-layout details.
  constexpr bool k_half_pack = (BITS == 5) && (qk_per_thread == 4);
  constexpr bool v_half_pack = (BITS == 5) && (v_per_thread == 4);

  static_assert(
      qk_per_thread % min_per_thread == 0,
      "qk_per_thread must be a multiple of the helper natural stride");
  static_assert(
      v_per_thread % min_per_thread == 0,
      "v_per_thread must be a multiple of the helper natural stride");
  static_assert(
      k_half_pack || qk_per_thread % pack_factor == 0,
      "qk_per_thread must be a multiple of the quantization pack factor "
      "(or use the bits=5 half-pack path)");
  static_assert(
      v_half_pack || v_per_thread % pack_factor == 0,
      "v_per_thread must be a multiple of the quantization pack factor "
      "(or use the bits=5 half-pack path)");
  static_assert(
      GROUP_SIZE % pack_factor == 0,
      "group_size must be a multiple of pack_factor");
  static_assert(
      qk_per_thread <= GROUP_SIZE,
      "a thread's K slice must fit inside one scale/bias group");
  static_assert(
      v_per_thread <= GROUP_SIZE,
      "a thread's V slice must fit inside one scale/bias group");
  static_assert(active_k <= 32, "kernel assumes one simdgroup");
  static_assert(active_v <= 32, "kernel assumes one simdgroup");

  typedef float U;

  thread U o[v_per_thread] = {0};

  // Position within the dispatch grid — identical to sdpa_vector_2pass_1.
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  // Decode-only v1: query is always row-contiguous (no query_transposed).
  const int q_offset = o_offset;

  const bool k_active = simd_lid < uint(active_k);
  const bool v_active = simd_lid < uint(active_v);

  const int k_elem_off = int(simd_lid) * qk_per_thread;
  const int v_elem_off = int(simd_lid) * v_per_thread;

  // Q is bf16, byte-addressed by element index. Inactive lanes still need
  // a valid pointer (we clamp to 0); they won't dereference it.
  queries += q_offset * D + (k_active ? k_elem_off : 0);

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // Packed pointer increments by (elems * BITS / 8) bytes per thread.
  // In half-pack mode (bits=5, qk_per_thread=4), two adjacent lanes
  // share a single 5-byte pack group; they both point at byte 0 of the
  // group and the low/high helpers pick the right 4 values. In the
  // standard path, each lane's elem_off is a multiple of pack_factor
  // so the division is exact.
  const int k_pack_group = k_active ? (k_elem_off / pack_factor) : 0;
  const int v_pack_group = v_active ? (v_elem_off / pack_factor) : 0;
  const int k_thread_byte_off = k_half_pack
      ? k_pack_group * ((pack_factor * BITS) / 8)  // bits=5: 5 bytes per group
      : (k_active ? (k_elem_off * BITS) / 8 : 0);
  const int v_thread_byte_off = v_half_pack
      ? v_pack_group * ((pack_factor * BITS) / 8)
      : (v_active ? (v_elem_off * BITS) / 8 : 0);
  const int k_half = k_half_pack ? int(simd_lid & 1u) : 0;
  const int v_half = v_half_pack ? int(simd_lid & 1u) : 0;
  k_packed += kv_batch_head_idx * k_head_stride_bytes
            + block_idx * k_seq_stride_bytes
            + k_thread_byte_off;
  v_packed += kv_batch_head_idx * v_head_stride_bytes
            + block_idx * v_seq_stride_bytes
            + v_thread_byte_off;

  // Scales / biases: one per GROUP_SIZE elements. Since qk_per_thread ≤
  // GROUP_SIZE, each thread needs exactly one scale and one bias per KV
  // row (the one covering its slice).
  const int k_group_idx = k_active ? (k_elem_off / GROUP_SIZE) : 0;
  const int v_group_idx = v_active ? (v_elem_off / GROUP_SIZE) : 0;
  k_scales += kv_batch_head_idx * k_head_stride_scale
            + block_idx * k_seq_stride_scale + k_group_idx;
  k_biases += kv_batch_head_idx * k_head_stride_scale
            + block_idx * k_seq_stride_scale + k_group_idx;
  v_scales += kv_batch_head_idx * v_head_stride_scale
            + block_idx * v_seq_stride_scale + v_group_idx;
  v_biases += kv_batch_head_idx * v_head_stride_scale
            + block_idx * v_seq_stride_scale + v_group_idx;

  out += o_offset * blocks * V + block_idx * V
       + (v_active ? v_elem_off : 0);
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // ── For each K position in this block's shard ──
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    }

    if (use_key) {
      // K dot product: explicit dequantize + dot. Using dequantize_thread
      // (shared with the V side) instead of qdot because qdot's
      // positional-Q prescale contract requires values_per_thread to be a
      // multiple of 4 even at bits=4 (which would force qk_per_thread > D/32
      // for small D). Explicit dot is cleaner and measured-identical in
      // register pressure for our instantiated shapes.
      //
      // At bits=5 × D ∈ {64, 128} we take the half-pack path: two lanes
      // share a 5-byte pack group, one does values 0..3 and the other
      // does 4..7. Both land in the same simd_sum reduction below so the
      // score agrees across all lanes.
      U score = 0;
      if (k_active) {
        U k_scale_val = static_cast<U>(*k_scales);
        U k_bias_val  = static_cast<U>(*k_biases);
        thread U k_deq[qk_per_thread];
        if constexpr (k_half_pack) {
          dequantize_thread_half5<U>(
              k_packed, k_scale_val, k_bias_val, k_half, k_deq);
        } else {
          dequantize_thread<U, qk_per_thread, BITS>(
              k_packed, k_scale_val, k_bias_val, k_deq);
        }
        #pragma unroll
        for (int j = 0; j < qk_per_thread; j++) {
          score += static_cast<U>(queries[j]) * k_deq[j];
        }
        score *= static_cast<U>(scale);
      }
      score = simd_sum(score);

      // Online softmax update (all lanes agree on `score` post-simd_sum).
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // V dequantize + accumulate into this thread's slice of o[].
      // Same half-pack treatment as K when bits=5 × V ∈ {64, 128}.
      if (v_active) {
        U v_scale_val = static_cast<U>(*v_scales);
        U v_bias_val  = static_cast<U>(*v_biases);
        thread U v_deq[v_per_thread];
        if constexpr (v_half_pack) {
          dequantize_thread_half5<U>(
              v_packed, v_scale_val, v_bias_val, v_half, v_deq);
        } else {
          dequantize_thread<U, v_per_thread, BITS>(
              v_packed, v_scale_val, v_bias_val, v_deq);
        }
        #pragma unroll
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * v_deq[j];
        }
      } else {
        // Inactive-V lanes still rescale o[] to keep the accumulator
        // tracked consistently across the loop (it will stay 0).
        #pragma unroll
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor;
        }
      }
    }

    // Advance to the next K/V row this threadgroup is responsible for.
    k_packed += blocks * k_seq_stride_bytes;
    v_packed += blocks * v_seq_stride_bytes;
    k_scales += blocks * k_seq_stride_scale;
    k_biases += blocks * k_seq_stride_scale;
    v_scales += blocks * v_seq_stride_scale;
    v_biases += blocks * v_seq_stride_scale;
  }

  // Publish per-block partials for sdpa_vector_2pass_2 to reduce.
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  if (v_active) {
    #pragma unroll
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
