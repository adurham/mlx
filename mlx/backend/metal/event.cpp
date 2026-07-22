// Copyright © 2024 Apple Inc.

#include <cstdio>
#include <cstdlib>
#include <thread>

#include "mlx/backend/metal/event.h"
#include "mlx/backend/metal/metal.h"
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

///////////////////////////////////////////////////////////////////////////////
// EventImpl implementations
///////////////////////////////////////////////////////////////////////////////

namespace metal {

EventImpl::EventImpl(Device& d) {
  auto p = new_scoped_memory_pool();
  mtl_event_ = NS::TransferPtr(d.mtl_device()->newSharedEvent());
  if (!mtl_event_) {
    throw std::runtime_error(
        "[Event::Event] Failed to create Metal shared event.");
  }
}

EventImpl::~EventImpl() {
  auto p = new_scoped_memory_pool();
  mtl_event_.reset();
}

void EventImpl::wait(uint64_t value) {
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
  // captured stream-worker exception, (b) surface an error poisoned onto this
  // event by a failed command buffer (upstream's error-propagation model —
  // check_error() below), and (c) honor a total timeout — letting a wedged
  // peer SELF-ABORT and have the runner reconnect() the transport in-place
  // instead of dying. Fast path is one signaledValue() read; then a brief
  // spin (cheap for sub-µs completions), then a low-CPU sleep-poll.
  // Tunables: MLX_EVENT_WAIT_POLL_US (sleep granularity, default 50),
  // MLX_EVENT_WAIT_SPIN (spins before sleeping, default 2000),
  // MLX_EVENT_WAIT_TIMEOUT_MS (self-abort deadline, default 40000; 0 disables).
  check_error();
  if (mtl_event_->signaledValue() >= value) {
    return;
  }
  static const uint64_t sleep_us = [] {
    const char* v = std::getenv("MLX_EVENT_WAIT_POLL_US");
    return v ? std::strtoull(v, nullptr, 10) : 50ULL;
  }();
  // Escape hatch: sleep_us==0 restores the legacy uninterruptible blocking
  // wait (A/B only — it reintroduces the peer-hang).
  if (sleep_us == 0) {
    if (!mtl_event_->waitUntilSignaledValue(value, -1)) {
      throw std::runtime_error("[Event::wait] Timed out");
    }
    check_error();
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
  // Measure REAL wall-clock elapsed with steady_clock — NOT an accumulated
  // sleep_us counter. macOS sleep granularity + scheduling make each poll
  // iteration take far longer than the requested sleep, so summing sleep_us
  // undercounts elapsed time by multiples and the timeout would not fire until
  // hundreds of real seconds — long after the 45s _check_hang SIGKILL.
  const auto start = std::chrono::steady_clock::now();
  bool logged_slow = false;
  while (mtl_event_->signaledValue() < value) {
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
    // stream-worker exception, then check for an error poisoned onto this
    // event by a failed command buffer (upstream's error-propagation model),
    // then honor the total self-abort timeout. Checking check_error() every
    // iteration (not just once at the end) matters here: if the command
    // buffer that would have signaled this event itself failed, the event
    // may never reach `value` at all, and we want to surface the real
    // root-cause error instead of spinning for the full timeout and then
    // throwing a generic "wedged event" message that hides it.
    scheduler::throw_if_stream_exception();
    check_error();
    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    // Diagnostic (only fires on a genuinely stuck wait — healthy waits finish
    // in << 1s — so no hot-path spam): proves this poll is actually running on
    // a wedged peer and accumulating wall-clock toward the self-abort.
    if (!logged_slow && elapsed_us >= 3'000'000) {
      logged_slow = true;
      fprintf(
          stderr,
          "[Event::wait] slow wait: elapsed=%.1fs signaled=%llu target=%llu "
          "(polling; self-abort at %llums)\n",
          elapsed_us / 1e6,
          static_cast<unsigned long long>(mtl_event_->signaledValue()),
          static_cast<unsigned long long>(value),
          static_cast<unsigned long long>(timeout_us / 1000));
      fflush(stderr);
      // exo-stall-diag (2026-07-21): dump recent command-buffer commit/
      // schedule/completion status right here, at the exact moment a slow
      // wait is first detected -- this is the one call site that answers
      // "was a command buffer even committed for the event we're stuck
      // waiting on, and if so, where is it stuck" without needing to
      // correlate separate tools' timestamps after the fact. No-op unless
      // EXO_CMDBUF_RING_DIAG=1.
      metal::dump_recent_command_buffers(stderr);
    }
    if (timeout_us != 0 && elapsed_us >= 0 &&
        static_cast<uint64_t>(elapsed_us) >= timeout_us) {
      throw std::runtime_error(
          "[Event::wait] Timed out: GPU event not signaled and no stream "
          "exception (peer rank stuck on an abandoned c>=2 collective); "
          "surfacing a clean fault for in-place reconnect / restart.");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
  }
  check_error();
}

void EventImpl::signal(uint64_t value) {
  mtl_event_->setSignaledValue(value);
}

void EventImpl::set_error(std::shared_ptr<std::string> error) {
  std::atomic_store(&error_, std::move(error));
}

void EventImpl::check_error() {
  auto error = std::atomic_exchange(&error_, {});
  if (error) {
    throw std::runtime_error(*error);
  }
}

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::wait() {
  static_cast<metal::EventImpl*>(event_.get())->wait(value());
}

void Event::wait(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [impl = std::move(impl), value = value()]() {
      impl->wait(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.wait_event(std::move(impl), value());
  }
}

void Event::signal(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [impl = std::move(impl), value = value()]() {
      impl->signal(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.signal_event(std::move(impl), value());

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
      encoder.get_command_buffer()->addCompletedHandler(
          [stream_idx, value_at_signal, ops, t_enc_us](MTL::CommandBuffer*) {
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
    }
  }
}

bool Event::is_signaled() const {
  auto* mtl_event = static_cast<metal::EventImpl*>(event_.get())->mtl_event();
  return mtl_event->signaledValue() >= value();
}

} // namespace mlx::core
