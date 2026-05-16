// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

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
    // exo-mlx-tune (2026-05-16): commit the current command buffer
    // BEFORE encoding the signal event, so the signal lands in a
    // FRESH buffer with no other ops in front of it.
    //
    // Why: signal latency = GPU time to drain everything in the
    // command buffer up to the signal point. mlx's encoder
    // accumulates ops (needs_commit() defers the commit until
    // op/byte thresholds trip), so by the time we reach Event::signal
    // the buffer can contain many primitives' worth of work. The
    // signal fires only after ALL of that work drains.
    //
    // On DSv4 MTP γ=2 100K c=1 decode, this creates the asymmetric
    // bistable stall diagnosed 2026-05-16 (JACCL_POLL_INSTRUMENT
    // commit 55dee34c): the downstream CPU stream worker is blocked
    // in waitUntilSignaledValue() inside Event::wait(), can't dispatch
    // the next distributed all_reduce until the signal fires, so
    // RDMA send-post is delayed, peer's CQE arrives ~10ms late, and
    // decode throughput collapses from 32 t/s to 6 t/s.
    //
    // By committing before encoding the signal:
    //   1. The "fat" buffer of accumulated work ships off without
    //      the signal — GPU starts executing it immediately.
    //   2. The signal goes into a fresh, near-empty buffer that
    //      commits below; it fires the moment its (trivial) GPU
    //      work completes.
    //
    // This decouples signal latency from how deep the encoder's
    // op-accumulation got, which is the root cause of the bistable
    // stall behavior. Trade-off: one extra command buffer commit
    // per signal event — cheap compared to a single GPU op.
    encoder.commit();
    auto* command_buffer = encoder.get_command_buffer();
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
  }
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(event_.get())->signaledValue() >=
      value();
}

} // namespace mlx::core
