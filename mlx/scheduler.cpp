// Copyright © 2023-2026 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/utils.h"

namespace mlx::core {

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
    f.wait();
  } else {
    gpu::synchronize(s);
  }
  // exo-jaccl-fix (2026-07-01): after the stream has drained to this
  // synchronization point, rethrow any exception a task captured on the worker
  // thread (e.g. a JACCL RDMA collective fault). This surfaces the failure on
  // the CALLING thread as a normal catchable C++/Python exception instead of
  // having let it std::terminate the process on the worker thread. The runner's
  // try/except then converts it to a clean RunnerTerminationError.
  if (auto e = scheduler::scheduler().take_stream_exception(s)) {
    std::rethrow_exception(e);
  }
}

void synchronize(ThreadLocalStream s) {
  synchronize(stream_from_thread_local_stream(s));
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

void clear_streams() {
  cpu::clear_streams();
  gpu::clear_streams();
}

namespace scheduler {

Scheduler::Scheduler() {
  is_main_thread();
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    std::unique_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it == threads_.end()) {
      it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
    }
    st = it->second.get();
  }
  st->enqueue(std::move(task));
}

// exo-jaccl-fix (2026-07-01): consume the captured worker-thread exception for
// stream ``s``. Returns nullptr when the stream never faulted or was never
// created. Threads map is guarded by threads_mtx_.
std::exception_ptr Scheduler::take_stream_exception(Stream s) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    return nullptr;
  }
  return st->take_exception();
}

// Leak the scheduler singleton on all platforms. During static destruction,
// worker threads may still be executing JIT-compiled code that has been
// unmapped, causing SIGSEGV (macOS/Linux) or join() deadlocks (Windows/MSVC
// CRT).
// The OS reclaims all resources at process exit anyway.
Scheduler& scheduler() {
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
}

} // namespace scheduler
} // namespace mlx::core
