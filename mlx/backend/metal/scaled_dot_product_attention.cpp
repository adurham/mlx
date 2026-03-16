// Copyright © 2024 Apple Inc.
#include <cstdlib>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

// Target total threadgroups for M3/M4 SDPA 2-pass kernel.
// Configurable via MLX_SDPA_MAX_TG environment variable.
// Default 320 = ~8 rounds on M4 Max (40 cores).
int sdpa_max_threadgroups() {
  static int val = [] {
    if (auto* env = std::getenv("MLX_SDPA_MAX_TG")) {
      int v = std::atoi(env);
      return v > 0 ? v : 320;
    }
    return 320;
  }();
  return val;
}

// Use split-pass SDPA (separate score+max pass, then streaming V pass).
// Optimized for Apple Silicon's high bandwidth — eliminates online softmax
// serial dependency in the V accumulation loop.
bool use_split_pass() {
  static bool val = [] {
    if (auto* env = std::getenv("MLX_SDPA_SPLIT_PASS")) {
      return std::string(env) == "1";
    }
    return false;
  }();
  return val;
}

// CPU-assisted SDPA: fraction of KV positions to process on CPU.
// 0.0 = disabled, 0.1 = 10% of positions on CPU.
float cpu_assist_fraction() {
  static float val = [] {
    if (auto* env = std::getenv("MLX_SDPA_CPU_FRACTION")) {
      float v = std::atof(env);
      return (v > 0.0f && v < 1.0f) ? v : 0.0f;
    }
    return 0.0f;
  }();
  return val;
}

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

// CPU attention using Accelerate BLAS.
// Processes a contiguous range of KV positions on CPU while GPU handles the rest.
// Results are written into the block slot at `block_idx` in the intermediate arrays.
void cpu_sdpa_block(
    const float* q_data,       // (num_q_heads, D) — queries as float32
    const float* k_data,       // (N_cpu, D) per KV head — contiguous float32 K
    const float* v_data,       // (N_cpu, D) per KV head — contiguous float32 V
    float* out_data,           // output slot: (num_q_heads, D)
    float* sums_data,          // sum_exp slot: (num_q_heads,)
    float* maxs_data,          // max_score slot: (num_q_heads,)
    int num_q_heads,
    int num_kv_heads,
    int N_cpu,
    int D,
    float scale,
    size_t k_head_stride,
    size_t v_head_stride) {

  int gqa = num_q_heads / num_kv_heads;

  // Temp buffers for scores and exp
  float* scores = (float*)malloc(gqa * N_cpu * sizeof(float));
  float* exp_buf = (float*)malloc(gqa * N_cpu * sizeof(float));

  for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
    const float* k_head = k_data + kv_h * k_head_stride;
    const float* v_head = v_data + kv_h * v_head_stride;
    const float* q_group = q_data + kv_h * gqa * D;
    float* out_group = out_data + kv_h * gqa * D;
    float* max_group = maxs_data + kv_h * gqa;
    float* sum_group = sums_data + kv_h * gqa;

    // Scores = Q @ K^T: (gqa, D) × (D, N_cpu) → (gqa, N_cpu)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gqa, N_cpu, D,
                scale, q_group, D,
                k_head, D,
                0.0f, scores, N_cpu);

    // Per-Q-head softmax
    for (int qi = 0; qi < gqa; qi++) {
      float* s = scores + qi * N_cpu;
      float* e = exp_buf + qi * N_cpu;

      float max_s;
      vDSP_maxv(s, 1, &max_s, N_cpu);

      float neg_max = -max_s;
      vDSP_vsadd(s, 1, &neg_max, e, 1, N_cpu);
      int n_int = N_cpu;
      vvexpf(e, e, &n_int);

      float sum_e;
      vDSP_sve(e, 1, &sum_e, N_cpu);

      max_group[qi] = max_s;
      sum_group[qi] = sum_e;
    }

    // Output = exp_scores @ V: (gqa, N_cpu) × (N_cpu, D) → (gqa, D)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gqa, D, N_cpu,
                1.0f, exp_buf, N_cpu,
                v_head, D,
                0.0f, out_group, D);
  }

  free(scores);
  free(exp_buf);
}
#endif // __APPLE__

void sdpa_full_self_attention_nax(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 64;
  int bk = 32;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = d.get_command_encoder(s.index);

  auto kernel = get_steel_attention_nax_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  if (metal::is_nax_available() && q.shape(3) != 80 &&
      (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& q = */ q,
        /* const array& k = */ k,
        /* const array& v = */ v,
        /* const float scale = */ scale,
        /* array& o = */ o,
        /* bool do_causal_ = */ do_causal_,
        /* const std::optional<array>& mask = */ mask,
        /* const std::optional<array>& sinks = */ sinks);
  }

  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 32;
  int bk = bd < 128 ? 32 : 16;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = d.get_command_encoder(s.index);

  auto kernel = get_steel_attention_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);

  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 16);
    compute_encoder.set_bytes(q.shape(1), 17);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int n_simds = gqa_factor * q.shape(2);

  char devc = d.get_architecture().back();
  int N = k.shape(2);
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    // M3/M4 family — scale blocks with N like other architectures
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
    // Cap blocks to limit threadgroup scheduling overhead on M3/M4 GPUs.
    // Each layer dispatches (kv_heads × batch × blocks) threadgroups, but
    // M4 Max has only 40 GPU cores (M4 Pro: 18, M3 Max: 40). When blocks
    // scales to 512+ at high context, scheduling 1000+ threadgroups through
    // 40 cores creates 25+ sequential rounds with the memory bus idle between
    // them. Capping total threadgroups to ~320 limits to ~8 rounds while
    // keeping enough iterations per block for full SIMD utilization.
    int tg_per_block = k.shape(1) * q.shape(0);
    if (tg_per_block > 0) {
      int max_blocks = std::max(32, sdpa_max_threadgroups() / tg_per_block);
      blocks = std::min(blocks, max_blocks);
    }
  }
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  // CPU-assisted SDPA: reduce GPU's N and add a CPU block for the tail positions.
  float cpu_frac = cpu_assist_fraction();
  int N_gpu = N;
  int N_cpu = 0;
  bool do_cpu_assist = false;
#ifdef __APPLE__
  if (cpu_frac > 0.0f && N > 1024 && q.shape(2) == 1 && !mask.has_value()) {
    N_cpu = std::max(64, static_cast<int>(N * cpu_frac));
    N_gpu = N - N_cpu;
    do_cpu_assist = true;
  }
#endif

  // When CPU-assisted, allocate blocks+1 for the extra CPU block
  int total_blocks = do_cpu_assist ? blocks + 1 : blocks;

  MTL::Size group_dims(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);

  // Allocate the intermediates (with extra block slot if CPU-assisted)
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(total_blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  d.add_temporary(intermediate, s.index);
  d.add_temporary(sums, s.index);
  d.add_temporary(maxs, s.index);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();

  // Two-loop SDPA: single kernel with threadgroup memory for scores.
  // Loop 1: K read + score + max. Loop 2: V read + exp(score - known_max) * V.
  // No intermediate device memory, no extra kernel launches.
  if (use_split_pass()) {
    // Check threadgroup memory fits: gqa_factor * positions_per_block floats
    int positions_per_block = (N + blocks - 1) / blocks;
    int smem_floats = gqa_factor * positions_per_block;
    if (smem_floats <= 8192) {  // 32KB limit
      // Use 2loop kernel
      std::string lk = "sdpa_vector_2loop_";
      lk += get_type_string(q.dtype());
      lk += "_";
      lk += std::to_string(q.shape(-1));
      lk += "_";
      lk += std::to_string(v.shape(-1));

      metal::MTLFCList fc = {
          {&has_mask, MTL::DataType::DataTypeBool, 20},
          {&query_transposed, MTL::DataType::DataTypeBool, 21},
          {&do_causal, MTL::DataType::DataTypeBool, 22},
          {&bool_mask, MTL::DataType::DataTypeBool, 23},
          {&float_mask, MTL::DataType::DataTypeBool, 24},
          {&has_sinks, MTL::DataType::DataTypeBool, 25},
          {&blocks, MTL::DataType::DataTypeInt, 26},
      };
      std::string lh = lk;
      lh += has_mask ? (bool_mask ? "_bm" : "_fm") : "_nm";
      lh += query_transposed ? "_qt" : "_qnt";
      lh += do_causal ? "_c" : "_nc";
      lh += has_sinks ? "_s_" : "_ns_";
      lh += std::to_string(blocks);

      auto& enc = d.get_command_encoder(s.index);
      auto kernel = d.get_kernel(lk, lh, fc);
      check_kernel_threadgroup_size(kernel, group_dims, lh);

      enc.set_compute_pipeline_state(kernel);
      enc.set_input_array(q, 0);
      enc.set_input_array(k, 1);
      enc.set_input_array(v, 2);
      enc.set_output_array(intermediate, 3);
      enc.set_output_array(sums, 4);
      enc.set_output_array(maxs, 5);
      enc.set_bytes(N, 7);
      enc.set_bytes(k_head_stride, 8);
      enc.set_bytes(k_seq_stride, 9);
      enc.set_bytes(v_head_stride, 10);
      enc.set_bytes(v_seq_stride, 11);
      enc.set_bytes(scale, 12);
      if (has_mask) {
        auto& m = *mask;
        enc.set_input_array(m, 13 + float_mask);
        int32_t kv_seq_stride_m = m.shape(3) > 1 ? m.strides(3) : 0;
        int32_t q_seq_stride_m = m.shape(2) > 1 ? m.strides(2) : 0;
        int32_t head_stride_m =
            m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
        enc.set_bytes(kv_seq_stride_m, 15);
        enc.set_bytes(q_seq_stride_m, 16);
        enc.set_bytes(head_stride_m, 17);
      }
      if (has_sinks) {
        enc.set_input_array(*sinks, 18);
      }
      enc.dispatch_threadgroups(grid_dims, group_dims);

      // Final merge pass (reuse 2pass_2)
      std::string mk = "sdpa_vector_2pass_2_";
      mk += get_type_string(q.dtype());
      mk += "_";
      mk += std::to_string(v.shape(-1));
      kernel = d.get_kernel(mk);
      enc.set_compute_pipeline_state(kernel);
      enc.set_input_array(intermediate, 0);
      enc.set_input_array(sums, 1);
      enc.set_input_array(maxs, 2);
      enc.set_output_array(out, 3);
      enc.set_bytes(blocks, 4);
      MTL::Size mg(1024, 1, 1);
      MTL::Size mgrid(q.shape(0) * q.shape(1), q.shape(2), 1);
      check_kernel_threadgroup_size(kernel, mg, mk);
      enc.dispatch_threadgroups(mgrid, mg);
      return;
    }
    // Fall through to standard 2pass if threadgroup memory too small
  }

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(N_gpu, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 18);
  }

  // Launch GPU kernel (2pass_1)
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

#ifdef __APPLE__
  // CPU-assisted attention: process tail positions on CPU while GPU runs 2pass_1.
  // The GPU processes [0, N_gpu) and the CPU processes [N_gpu, N).
  // CPU writes its results as an extra block in the intermediate arrays.
  // The 2pass_2 merge kernel then combines GPU blocks + CPU block.
  if (do_cpu_assist) {
    int num_q_heads_total = q.shape(0) * q.shape(1);
    int D = q.shape(-1);
    int V_dim = v.shape(-1);

    // Convert Q to float32 for CPU BLAS (Q is small — just 1 token)
    std::vector<float> q_f32(num_q_heads_total * D);
    if (q.dtype() == float32) {
      auto* qp = q.data<float>();
      for (int i = 0; i < num_q_heads_total; i++) {
        memcpy(q_f32.data() + i * D, qp + i * D, D * sizeof(float));
      }
    } else {
      // bfloat16 → float32
      auto* qp = reinterpret_cast<const uint16_t*>(q.data<bfloat16_t>());
      for (int i = 0; i < num_q_heads_total * D; i++) {
        uint32_t bits = static_cast<uint32_t>(qp[i]) << 16;
        memcpy(&q_f32[i], &bits, 4);
      }
    }

    // Convert CPU portion of K and V to float32
    int kv_heads = k.shape(1);
    std::vector<float> k_f32(kv_heads * N_cpu * D);
    std::vector<float> v_f32(kv_heads * N_cpu * D);

    for (int h = 0; h < kv_heads; h++) {
      if (k.dtype() == float32) {
        auto* kp = k.data<float>() + h * k_head_stride + N_gpu * k_seq_stride;
        auto* vp = v.data<float>() + h * v_head_stride + N_gpu * v_seq_stride;
        for (int pos = 0; pos < N_cpu; pos++) {
          memcpy(k_f32.data() + h * N_cpu * D + pos * D, kp + pos * k_seq_stride, D * sizeof(float));
          memcpy(v_f32.data() + h * N_cpu * D + pos * D, vp + pos * v_seq_stride, D * sizeof(float));
        }
      } else {
        // bfloat16 → float32
        auto* kp = reinterpret_cast<const uint16_t*>(k.data<bfloat16_t>()) + h * k_head_stride + N_gpu * k_seq_stride;
        auto* vp = reinterpret_cast<const uint16_t*>(v.data<bfloat16_t>()) + h * v_head_stride + N_gpu * v_seq_stride;
        for (int pos = 0; pos < N_cpu; pos++) {
          for (int d = 0; d < D; d++) {
            uint32_t kb = static_cast<uint32_t>(kp[pos * k_seq_stride + d]) << 16;
            uint32_t vb = static_cast<uint32_t>(vp[pos * v_seq_stride + d]) << 16;
            memcpy(&k_f32[h * N_cpu * D + pos * D + d], &kb, 4);
            memcpy(&v_f32[h * N_cpu * D + pos * D + d], &vb, 4);
          }
        }
      }
    }

    // Run CPU attention (runs while GPU processes 2pass_1)
    std::vector<float> cpu_out(num_q_heads_total * V_dim, 0.0f);
    std::vector<float> cpu_maxs(num_q_heads_total, -1e38f);
    std::vector<float> cpu_sums(num_q_heads_total, 0.0f);

    cpu_sdpa_block(
        q_f32.data(), k_f32.data(), v_f32.data(),
        cpu_out.data(), cpu_sums.data(), cpu_maxs.data(),
        num_q_heads_total, kv_heads, N_cpu, D, scale,
        N_cpu * D, N_cpu * D);

    // Write CPU results to the extra block slot in intermediate arrays.
    // intermediate layout: [batch*q_heads*q_seq, total_blocks, V_dim]
    // sums/maxs layout: [batch*q_heads*q_seq, total_blocks]
    int cpu_block = blocks;  // extra block index
    for (int qh = 0; qh < num_q_heads_total; qh++) {
      // Write CPU output to intermediate
      float* int_ptr = intermediate.data<float>() +
          qh * total_blocks * V_dim + cpu_block * V_dim;
      // Convert float32 → output dtype if needed
      if (q.dtype() == float32) {
        memcpy(int_ptr, cpu_out.data() + qh * V_dim, V_dim * sizeof(float));
      } else {
        // float32 → bfloat16
        auto* out_bf16 = reinterpret_cast<uint16_t*>(
            intermediate.data<bfloat16_t>()) +
            qh * total_blocks * V_dim + cpu_block * V_dim;
        for (int d = 0; d < V_dim; d++) {
          uint32_t bits;
          memcpy(&bits, &cpu_out[qh * V_dim + d], 4);
          out_bf16[d] = static_cast<uint16_t>(bits >> 16);
        }
      }

      // Write CPU max and sum_exp
      float* sums_ptr = sums.data<float>() + qh * total_blocks + cpu_block;
      float* maxs_ptr = maxs.data<float>() + qh * total_blocks + cpu_block;
      *sums_ptr = cpu_sums[qh];
      *maxs_ptr = cpu_maxs[qh];
    }
  }
#endif

  // Final pass (2pass_2) — merges all blocks including CPU block
  kname.clear();
  kname = "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Get the kernel
  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(total_blocks, 4);

  // Launch
  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (is_training) {
    // It's faster for training on Metal to use the unfused SDPA for both
    // forward and backward.
    return true;
  }
  if (output_logsumexp) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = k.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256);
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);

  const bool sdpa_full_supported_mask = !has_mask || has_arr_mask ||
      (query_sequence_length <= key_sequence_length && do_causal);

  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim &&
      (query_sequence_length * gqa_factor) <= 32;

  return !(supports_sdpa_full || supports_sdpa_vector);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = outputs[0];

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(inputs.size());
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // We are in vector mode ie single query
  if (q_pre.shape(2) <= 8) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    bool q_copied = !q_copy_unless(q_pre);
    array q = (q_copied) ? contiguous_copy_gpu(q_pre, s) : q_pre;
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      if (q_copied) {
        copies.push_back(q);
      }
      o.set_data(allocator::malloc(o.nbytes()));
    }

    auto mask_copy_unless = [&q](const array& arr) {
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
          (strides[0] == strides[1] * shape[1]);
    };

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    }
  }

  // Full attention mode
  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
        : std::nullopt;

    sdpa_full_self_attention_metal(
        s, d, q, k, v, scale_, o, do_causal_, mask, sinks);
  }

  d.add_temporaries(std::move(copies), s.index);
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("NYI");
}

namespace {

void sdpa_vector_quantized_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k_data,
    const array& k_scales,
    const array& k_biases,
    const array& v_data,
    const array& v_scales,
    const array& v_biases,
    array& out,
    float scale,
    int group_size,
    bool do_causal,
    const std::optional<array>& mask) {
  int D = q.shape(-1);
  int V_DIM = v_data.shape(-1) * 4; // 8-bit: 4 values per uint32

  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_quantized_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(D);
  kname += "_";
  kname += std::to_string(V_DIM);

  int gqa_factor = q.shape(1) / k_data.shape(1);
  int N = k_data.shape(2);

  size_t k_head_stride =
      k_data.shape(1) == 1 ? k_data.strides(0) : k_data.strides(1);
  size_t k_seq_stride = k_data.strides(2);
  size_t ks_head_stride =
      k_scales.shape(1) == 1 ? k_scales.strides(0) : k_scales.strides(1);
  size_t ks_seq_stride = k_scales.strides(2);
  size_t v_head_stride =
      v_data.shape(1) == 1 ? v_data.strides(0) : v_data.strides(1);
  size_t v_seq_stride = v_data.strides(2);
  size_t vs_head_stride =
      v_scales.shape(1) == 1 ? v_scales.strides(0) : v_scales.strides(1);
  size_t vs_seq_stride = v_scales.strides(2);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_data, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_input_array(v_data, 4);
  compute_encoder.set_input_array(v_scales, 5);
  compute_encoder.set_input_array(v_biases, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(gqa_factor, 8);
  compute_encoder.set_bytes(N, 9);
  compute_encoder.set_bytes(group_size, 10);
  compute_encoder.set_bytes(k_head_stride, 11);
  compute_encoder.set_bytes(k_seq_stride, 12);
  compute_encoder.set_bytes(ks_head_stride, 13);
  compute_encoder.set_bytes(ks_seq_stride, 14);
  compute_encoder.set_bytes(v_head_stride, 15);
  compute_encoder.set_bytes(v_seq_stride, 16);
  compute_encoder.set_bytes(vs_head_stride, 17);
  compute_encoder.set_bytes(vs_seq_stride, 18);
  compute_encoder.set_bytes(scale, 19);

  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 20 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 22);
    compute_encoder.set_bytes(q_seq_stride, 23);
    compute_encoder.set_bytes(head_stride, 24);
  }

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass_quantized_dispatch(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k_data,
    const array& k_scales,
    const array& k_biases,
    const array& v_data,
    const array& v_scales,
    const array& v_biases,
    array& out,
    float scale,
    int group_size,
    bool do_causal,
    const std::optional<array>& mask) {
  int D = q.shape(-1);
  int V_DIM = v_data.shape(-1) * 4;

  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_quantized_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(D);
  kname += "_";
  kname += std::to_string(V_DIM);

  int gqa_factor = q.shape(1) / k_data.shape(1);
  int n_simds = gqa_factor * q.shape(2);

  char devc = d.get_architecture().back();
  int N = k_data.shape(2);
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    // M3/M4 family — scale blocks with N like other architectures
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
    // Cap blocks to limit threadgroup scheduling overhead on M3/M4 GPUs.
    // See comment in non-quantized 2-pass kernel above for rationale.
    int tg_per_block = k_data.shape(1) * q.shape(0);
    if (tg_per_block > 0) {
      int max_blocks = std::max(32, sdpa_max_threadgroups() / tg_per_block);
      blocks = std::min(blocks, max_blocks);
    }
  }

  size_t k_head_stride =
      k_data.shape(1) == 1 ? k_data.strides(0) : k_data.strides(1);
  size_t k_seq_stride = k_data.strides(2);
  size_t ks_head_stride =
      k_scales.shape(1) == 1 ? k_scales.strides(0) : k_scales.strides(1);
  size_t ks_seq_stride = k_scales.strides(2);
  size_t v_head_stride =
      v_data.shape(1) == 1 ? v_data.strides(0) : v_data.strides(1);
  size_t v_seq_stride = v_data.strides(2);
  size_t vs_head_stride =
      v_scales.shape(1) == 1 ? v_scales.strides(0) : v_scales.strides(1);
  size_t vs_seq_stride = v_scales.strides(2);

  MTL::Size group_dims_pass1(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims_pass1(k_data.shape(1), q.shape(0), blocks);

  // Allocate intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  d.add_temporary(intermediate, s.index);
  d.add_temporary(sums, s.index);
  d.add_temporary(maxs, s.index);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;

  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += "_";
  hash_name += std::to_string(blocks);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims_pass1, hash_name);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Pass 1 arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_data, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_input_array(v_data, 4);
  compute_encoder.set_input_array(v_scales, 5);
  compute_encoder.set_input_array(v_biases, 6);
  compute_encoder.set_output_array(intermediate, 7);
  compute_encoder.set_output_array(sums, 8);
  compute_encoder.set_output_array(maxs, 9);
  compute_encoder.set_bytes(N, 10);
  compute_encoder.set_bytes(group_size, 11);
  compute_encoder.set_bytes(k_head_stride, 12);
  compute_encoder.set_bytes(k_seq_stride, 13);
  compute_encoder.set_bytes(ks_head_stride, 14);
  compute_encoder.set_bytes(ks_seq_stride, 15);
  compute_encoder.set_bytes(v_head_stride, 16);
  compute_encoder.set_bytes(v_seq_stride, 17);
  compute_encoder.set_bytes(vs_head_stride, 18);
  compute_encoder.set_bytes(vs_seq_stride, 19);
  compute_encoder.set_bytes(scale, 20);

  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 21 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 23);
    compute_encoder.set_bytes(q_seq_stride, 24);
    compute_encoder.set_bytes(head_stride, 25);
  }

  compute_encoder.dispatch_threadgroups(grid_dims_pass1, group_dims_pass1);

  // Pass 2: reuse sdpa_vector_2pass_2 (reduces float intermediates only)
  kname.clear();
  kname = "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(V_DIM);

  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  MTL::Size group_dims_pass2(1024, 1, 1);
  MTL::Size grid_dims_pass2(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims_pass2, kname);
  compute_encoder.dispatch_threadgroups(grid_dims_pass2, group_dims_pass2);
}

} // namespace

void QuantizedScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q = inputs[0];
  auto& k_data = inputs[1];
  auto& k_scales = inputs[2];
  auto& k_biases = inputs[3];
  auto& v_data = inputs[4];
  auto& v_scales = inputs[5];
  auto& v_biases = inputs[6];
  auto& o = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());

  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  auto is_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  auto q_copy_unless = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return true;
    }
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (shape[0] == 1 || shape[1] == 1) {
      auto bidx = shape[0] == 1 ? 1 : 0;
      return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
          (strides[bidx] == shape[3]);
    }
    return false;
  };

  auto kv_copy_unless = [](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (strides.back() != 1) {
      return false;
    }
    if (shape[0] == 1 || shape[1] == 1) {
      return true;
    }
    return (strides[0] == strides[1] * shape[1]);
  };

  bool q_copied = !q_copy_unless(q);
  array q_arr = q_copied ? contiguous_copy_gpu(q, s) : q;
  const auto& kd = copy_unless(kv_copy_unless, k_data);
  const auto& ks = copy_unless(kv_copy_unless, k_scales);
  const auto& kb = copy_unless(kv_copy_unless, k_biases);
  const auto& vd = copy_unless(kv_copy_unless, v_data);
  const auto& vs = copy_unless(kv_copy_unless, v_scales);
  const auto& vb = copy_unless(kv_copy_unless, v_biases);

  if (q_arr.is_donatable() && q_arr.flags().row_contiguous &&
      q_arr.size() == o.size()) {
    o.copy_shared_buffer(q_arr);
  } else {
    if (q_copied) {
      copies.push_back(q_arr);
    }
    o.set_data(allocator::malloc(o.nbytes()));
  }

  bool has_arr_mask = inputs.size() > 7;
  auto mask = has_arr_mask
      ? std::optional<array>{copy_unless(is_contiguous, inputs[7])}
      : std::nullopt;

  bool do_causal = do_causal_ && q_arr.shape(2) > 1;
  int N = kd.shape(2);

  char devc = d.get_architecture().back();
  if (((devc == 'd' || devc == 's') && N >= 1024) ||
      (kd.shape(1) < q_arr.shape(1) && N >= 4096)) {
    sdpa_vector_2pass_quantized_dispatch(
        s, d, q_arr, kd, ks, kb, vd, vs, vb, o, scale_, group_size_,
        do_causal, mask);
  } else {
    sdpa_vector_quantized_dispatch(
        s, d, q_arr, kd, ks, kb, vd, vs, vb, o, scale_, group_size_,
        do_causal, mask);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
