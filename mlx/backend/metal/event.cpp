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
    auto* command_buffer = encoder.get_command_buffer();
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
    // exo-mlx-tune (2026-05-16): commit the command buffer immediately
    // after attaching a signal event. Without this commit, the buffer
    // (with the signal embedded) sits unsubmitted until CommandEncoder
    // sees enough subsequent ops/bytes to trigger needs_commit(). On
    // M4 Max with EXO_MAX_ACTIVE_TASKS=5, this can leave the signal
    // sitting behind 4+ queued command buffers, deferring its firing
    // by 5-15 ms. The CPU stream worker, blocked in
    // waitUntilSignaledValue() inside Event::wait(), then can't start
    // a downstream distributed all_reduce until the signal fires —
    // RDMA send-post is delayed, peer's CQE arrives ~10ms late, and
    // we see the asymmetric "bistable stall" diagnosed on DSv4 MTP
    // γ=2 100K c=1 decode: clean 32 t/s flips to stalled 6 t/s
    // depending on whether the queue happens to be shallow or deep
    // when the iter starts.
    //
    // Forcing commit here makes signal latency proportional to GPU
    // compute time, not queue depth, which decouples it from the
    // bistability mechanism. Trade-off: more commit() calls means
    // more, smaller command buffers — slight loss in batch efficiency
    // for in-process GPU work. Acceptable trade vs the stall.
    encoder.commit();
  }
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(event_.get())->signaledValue() >=
      value();
}

} // namespace mlx::core
