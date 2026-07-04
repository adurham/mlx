// Copyright © 2024 Apple Inc.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

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
  // exo-jaccl-fix (2026-07-05): INTERRUPTIBLE wait. Poll the shared event's
  // value in USER space (MTL::SharedEvent::signaledValue() — a non-blocking
  // shared-memory read) instead of Apple's waitUntilSignaledValue, which traps
  // into an UNINTERRUPTIBLE kernel GPU-wait (iokit_user_client_trap) and, when
  // the GPU/collective is wedged, ignores its timeout and blocks forever.
  //
  // That kernel block is the c>=2 PEER-rank trap: a wedged TP collective never
  // signals this event, and the peer's main thread is stuck in the kernel where
  // it can neither honor a timeout, surface a captured stream exception, nor
  // call reconnect() — only SIGKILL frees it (faulting the GPU + leaking RDMA
  // QPs). Polling keeps this thread runnable, so the loop can (a) rethrow a
  // captured stream-worker exception and (b) honor a total timeout — letting a
  // wedged peer SELF-ABORT and have the runner reconnect() the transport
  // in-place instead of dying. Fast path is one signaledValue() read; then a
  // brief spin (cheap for sub-µs completions), then a low-CPU sleep-poll.
  // Tunables: MLX_EVENT_WAIT_POLL_US (sleep granularity, default 50),
  // MLX_EVENT_WAIT_SPIN (spins before sleeping, default 2000),
  // MLX_EVENT_WAIT_TIMEOUT_MS (self-abort deadline, default 40000; 0 disables).
  auto* ev = static_cast<MTL::SharedEvent*>(event_.get());
  const uint64_t target = value();
  if (ev->signaledValue() >= target) {
    return;
  }
  static const uint64_t sleep_us = [] {
    const char* v = std::getenv("MLX_EVENT_WAIT_POLL_US");
    return v ? std::strtoull(v, nullptr, 10) : 50ULL;
  }();
  // Escape hatch: sleep_us==0 restores the legacy uninterruptible blocking
  // wait (A/B only — it reintroduces the peer-hang).
  if (sleep_us == 0) {
    if (!ev->waitUntilSignaledValue(target, -1)) {
      throw std::runtime_error("[Event::wait] Timed out");
    }
    return;
  }
  static const uint64_t timeout_us = [] {
    const char* v = std::getenv("MLX_EVENT_WAIT_TIMEOUT_MS");
    return (v ? std::strtoull(v, nullptr, 10) : 40000ULL) * 1000ULL;
  }();
  static const uint64_t spin_iters = [] {
    const char* v = std::getenv("MLX_EVENT_WAIT_SPIN");
    return v ? std::strtoull(v, nullptr, 10) : 2000ULL;
  }();
  uint64_t spins = 0;
  uint64_t waited_us = 0;
  while (ev->signaledValue() < target) {
    if (spins < spin_iters) {
      ++spins;
#if defined(__aarch64__)
      __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__)
      __asm__ __volatile__("pause" ::: "memory");
#endif
      continue;
    }
    // Unlike the kernel wait, this loop actually runs: surface a captured
    // stream-worker exception, then honor the total self-abort timeout.
    scheduler::throw_if_stream_exception();
    if (timeout_us != 0 && waited_us >= timeout_us) {
      throw std::runtime_error(
          "[Event::wait] Timed out: GPU event not signaled and no stream "
          "exception (peer rank stuck on an abandoned c>=2 collective); "
          "surfacing a clean fault for in-place reconnect / restart.");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    waited_us += sleep_us;
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
