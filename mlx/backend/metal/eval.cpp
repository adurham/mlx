// Copyright © 2023-2024 Apple Inc.
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

namespace {
// Thread-safe deferred error from Metal completion handlers.
// Completion handlers run on Apple's com.Metal.CompletionQueueDispatch
// (a libdispatch queue). C++ exceptions cannot unwind through GCD
// dispatch blocks — a throw in a completion handler hits std::terminate
// and aborts the process unconditionally. Store the error here and
// re-throw at the next eval()/synchronize() call where the call stack
// has a real catch frame.
//
// Ported from upstream PR #3318 (closed without merge but production-
// validated by Thump604 with 5 days zero crashes on Qwen3.5-122B). See
// mlx#3224, #3317, #3390 — same root cause for our DSv4-Flash silent
// SIGABRTs after ~10-12 minutes of decode.
std::mutex deferred_error_mutex;
std::string deferred_error_message;

void set_deferred_error(const std::string& msg) {
  std::lock_guard<std::mutex> lock(deferred_error_mutex);
  if (deferred_error_message.empty()) {
    deferred_error_message = msg;
  }
}

void check_deferred_error() {
  std::lock_guard<std::mutex> lock(deferred_error_mutex);
  if (!deferred_error_message.empty()) {
    std::string msg = std::move(deferred_error_message);
    deferred_error_message.clear();
    throw std::runtime_error(msg);
  }
}
} // namespace

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

// Safe version for Metal completion handlers (GCD callbacks).
// Cannot throw — stores the error for deferred propagation at the
// next eval()/synchronize() call.
inline void check_error_deferred(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    const char* desc =
        cbuf->error() && cbuf->error()->localizedDescription()
        ? cbuf->error()->localizedDescription()->utf8String()
        : "(no localizedDescription)";
    // fprintf is async-signal-safe-ish and gives us a deterministic
    // log line even if the deferred re-throw never reaches Python
    // (e.g., process is being torn down).
    std::fprintf(
        stderr, "[METAL] Command buffer execution failed: %s\n", desc);
    std::fflush(stderr);
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: " << desc;
    set_deferred_error(msg.str());
  }
}

// Accumulate GPU-busy time from a completed command buffer when
// MLX_GPU_TIME is enabled. The Metal-recorded GPUStartTime/EndTime are
// CFTimeInterval (seconds) and reflect actual GPU wall execution. This
// is essentially free — Metal records them automatically on completion.
//
// Filter buffers with invalid timestamps. Buffers that never actually
// hit the GPU (CPU-only command buffers, jaccl RDMA collective stubs,
// errored buffers) report GPUStartTime/GPUEndTime=0 or near-zero, which
// makes ``end - start`` either zero or a multi-billion-second nonsense
// reading depending on the dispatcher. Reject anything where either
// timestamp is non-positive or the delta exceeds 1 s (a single Metal
// kernel realistically takes < 100 ms on Apple Silicon at the model
// sizes we run; 1 s is a generous upper bound that catches the
// pathological start=0 case).
inline void accumulate_gpu_time_if_enabled(MTL::CommandBuffer* cbuf) {
  if (!metal::gpu_time_enabled()) {
    return;
  }
  double start_s = cbuf->GPUStartTime();
  double end_s = cbuf->GPUEndTime();
  if (start_s <= 0.0 || end_s <= 0.0) {
    return;
  }
  double delta_s = end_s - start_s;
  if (delta_s <= 0.0 || delta_s > 1.0) {
    return;
  }
  uint64_t ns = static_cast<uint64_t>(delta_s * 1e9);
  metal::accumulate_gpu_time_ns(ns);
}

void eval(array& arr) {
  // Re-throw any deferred error from a prior completion handler.
  check_deferred_error();
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    debug_set_primitive_buffer_label_cheap(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error_deferred(cbuf);
          accumulate_gpu_time_if_enabled(cbuf);
        });
    encoder.commit();
  } else {
    // Don't accumulate gpu_time here: we are still appending to the
    // current command buffer (no commit yet). N successive eval()s on
    // the same buffer would each register a completion handler, and
    // each would observe the SAME GPUStartTime/GPUEndTime when the
    // buffer eventually commits — overcounting busy time by N×. The
    // single commit-time handler (the if-branch above, or finalize())
    // is the canonical accumulation point.
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error_deferred(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) {
    check_error_deferred(cbuf);
    accumulate_gpu_time_if_enabled(cbuf);
  });
  encoder.commit();
}

void synchronize(Stream s) {
  // Re-throw any deferred error from a prior completion handler.
  check_deferred_error();
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
}

} // namespace mlx::core::gpu
