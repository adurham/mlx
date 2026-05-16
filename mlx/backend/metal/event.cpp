// Copyright © 2024 Apple Inc.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

// MLX_SIGNAL_PROBE diagnostic — read once at startup.
// When set to "1" the GPU Event::signal path logs, for every signal encoded:
//   tag=SIGNAL_PROBE stream=<idx> value=<event_value> ops=<buffer_ops at encode>
//   t_enc_us=<wall when signal encoded> t_done_us=<wall when GPU completes>
//   gap_us=<t_done_us - t_enc_us>
// Output goes to stderr (caught by exo.log). Zero overhead when disabled.
bool signal_probe_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("MLX_SIGNAL_PROBE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return enabled;
}

inline uint64_t now_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    auto p = metal::new_scoped_memory_pool();
    static_cast<MTL::SharedEvent*>(ptr)->release();
  };
  auto p = metal::new_scoped_memory_pool();
  event_ = std::shared_ptr<void>(
      metal::device(Device::gpu).mtl_device()->newSharedEvent(), dtor);
  if (event_ == nullptr) {
    throw std::runtime_error(
        "[Event::Event] Failed to create Metal shared event.");
  }
}

void Event::wait() {
  if (!static_cast<MTL::SharedEvent*>(event_.get())
           ->waitUntilSignaledValue(value(), -1)) {
    throw std::runtime_error("[Event::wait] Timed out");
  }
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.end_encoding();
    auto* command_buffer = encoder.get_command_buffer();
    command_buffer->encodeWait(static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable {
      static_cast<MTL::SharedEvent*>(event_.get())->setSignaledValue(value());
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.end_encoding();
    auto* command_buffer = encoder.get_command_buffer();
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(event_.get()), value());

    if (signal_probe_enabled()) {
      // Capture state at signal-encode time. ops is the number of compute
      // primitives accumulated in the *uncommitted* command buffer that the
      // signal lands at the tail of. gap_us is the wall-clock delta between
      // encoding the signal and the GPU completion handler firing — i.e.
      // approximately the latency we suspect the bistable stall is paying
      // every cycle.
      const int ops = encoder.buffer_ops();
      const int stream_idx = stream.index;
      const uint64_t value_at_signal = value();
      const uint64_t t_enc_us = now_us();
      // stderr write at encode time so we still get a record even if the
      // buffer execution faults (no completion handler ever fires).
      std::fprintf(
          stderr,
          "tag=SIGNAL_PROBE_ENC stream=%d value=%llu ops=%d t_enc_us=%llu\n",
          stream_idx,
          static_cast<unsigned long long>(value_at_signal),
          ops,
          static_cast<unsigned long long>(t_enc_us));
      std::fflush(stderr);
      command_buffer->addCompletedHandler(
          [*this, stream_idx, value_at_signal, ops, t_enc_us](
              MTL::CommandBuffer*) {
            const uint64_t t_done_us = now_us();
            std::fprintf(
                stderr,
                "tag=SIGNAL_PROBE_DONE stream=%d value=%llu ops=%d "
                "t_enc_us=%llu t_done_us=%llu gap_us=%llu\n",
                stream_idx,
                static_cast<unsigned long long>(value_at_signal),
                ops,
                static_cast<unsigned long long>(t_enc_us),
                static_cast<unsigned long long>(t_done_us),
                static_cast<unsigned long long>(t_done_us - t_enc_us));
            std::fflush(stderr);
          });
    } else {
      command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
    }
  }
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(event_.get())->signaledValue() >=
      value();
}

} // namespace mlx::core
