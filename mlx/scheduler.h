// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::scheduler {

class StreamThread;

class MLX_API Scheduler {
 public:
  Scheduler();
  ~Scheduler();

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void enqueue(Stream s, std::function<void()> task);
  void wait_event(Stream s, Event event, std::function<void(Event&)> task);
  void signal_event(Stream s, Event event, std::function<void(Event&)> task);
  void check_error(Stream s);

  // exo-jaccl-fix (2026-07-01): consume and return the captured exception for
  // stream ``s`` (nullptr if none). Called by synchronize() to rethrow a
  // worker-thread fault on the Python-facing thread instead of terminating.
  std::exception_ptr take_stream_exception(Stream s);

  // exo-jaccl-fix (2026-07-01): sweep ALL stream worker threads and return the
  // first captured exception (consuming it), nullptr if all clean. Used by the
  // eval() path, which waits on arrays/events rather than calling
  // synchronize(Stream) and so can't target a single stream index. Defined in
  // scheduler.cpp because StreamThread is only forward-declared here (upstream
  // 2026-09 moved its definition into the .cpp).
  std::exception_ptr take_any_stream_exception();

  void notify_new_task(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_++;
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_--;
    }
    completion_cv.notify_all();
  }

  int n_active_tasks() const {
    return n_active_tasks_;
  }

  void wait_for_one() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = n_active_tasks();
    if (n_tasks_old <= 1) {
      return;
    }
    auto pred = [this, n_tasks_old] {
      return this->n_active_tasks() < n_tasks_old;
    };
    // exo-jaccl-fix (2026-07-05): INTERRUPTIBLE wait. A bare completion_cv.wait
    // blocks the MAIN thread forever if a stream task is wedged (a c>=2 peer
    // whose comm-stream collective the primary abandoned) — and unlike
    // Event::wait, this path has no timeout, so the peer hangs to the 45s
    // _check_hang SIGKILL, is killed mid-op, and the in-place reconnect never
    // gets a partner. Poll with a bounded wait + total timeout so the peer
    // surfaces a clean fault and self-aborts -> reconnect. Same knob as
    // Event::wait; 0 restores the legacy infinite wait.
    static const uint64_t timeout_ms = [] {
      const char* v = std::getenv("MLX_EVENT_WAIT_TIMEOUT_MS");
      return v ? std::strtoull(v, nullptr, 10) : 40000ULL;
    }();
    if (timeout_ms == 0) {
      completion_cv.wait(lk, pred);
      return;
    }
    auto start = std::chrono::steady_clock::now();
    bool logged_slow = false;
    while (!completion_cv.wait_for(lk, std::chrono::milliseconds(200), pred)) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      // Fires only on a genuinely stuck wait (healthy tasks complete in << 1s),
      // so no hot-path spam: proves this is where a wedged peer parks.
      if (!logged_slow && elapsed_ms >= 3000) {
        logged_slow = true;
        fprintf(
            stderr,
            "[wait_for_one] slow: elapsed=%.1fs n_active=%d (polling; "
            "self-abort at %llums)\n",
            elapsed_ms / 1000.0,
            n_active_tasks(),
            static_cast<unsigned long long>(timeout_ms));
        fflush(stderr);
      }
      if (elapsed_ms >= 0 && static_cast<uint64_t>(elapsed_ms) >= timeout_ms) {
        throw std::runtime_error(
            "[wait_for_one] Timed out: stream task not completing (peer stuck "
            "on an abandoned c>=2 collective); surfacing a clean fault for "
            "in-place reconnect / restart.");
      }
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  StreamThread& get_thread(Stream s);

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;

};

MLX_API Scheduler& scheduler();

template <typename F>
inline void enqueue(Stream s, F&& f) {
  scheduler().enqueue(s, std::forward<F>(f));
}

// Like enqueue but the task is used for processing the passed event.
template <typename F>
inline void wait_event(Stream s, Event event, F&& f) {
  scheduler().wait_event(s, std::move(event), std::forward<F>(f));
}

template <typename F>
inline void signal_event(Stream s, Event event, F&& f) {
  scheduler().signal_event(s, std::move(event), std::forward<F>(f));
}

// Throw and clear the error stored in the stream, if any.
inline void check_error(Stream s) {
  scheduler().check_error(s);
}

inline int n_active_tasks() {
  return scheduler().n_active_tasks();
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void wait_for_one() {
  scheduler().wait_for_one();
}

// exo-jaccl-fix (2026-07-01): rethrow (on the calling thread) any exception
// captured by a stream worker thread. No-op when all streams are clean. Called
// from eval()/synchronize() so a JACCL RDMA collective fault surfaces as a
// normal catchable exception instead of std::terminate on the worker thread.
inline void throw_if_stream_exception() {
  if (auto e = scheduler().take_any_stream_exception()) {
    std::rethrow_exception(e);
  }
}

// exo-jaccl-fix (2026-09-10): render an exception_ptr as text without
// rethrowing it to the caller. Used to fold a drained worker-thread fault
// into another exception's message; see combined_eval_failure below.
inline std::string describe_exception(const std::exception_ptr& e) {
  if (!e) {
    return "<none>";
  }
  try {
    std::rethrow_exception(e);
  } catch (const std::exception& ex) {
    return std::string(typeid(ex).name()) + ": " + ex.what();
  } catch (...) {
    return "<non-std exception>";
  }
}

// exo-jaccl-fix (2026-09-10): the exception thrown when a synchronous
// primitive failure and an asynchronous stream-worker fault (e.g. a JACCL
// RDMA collective error) happen in the SAME eval.
//
// WHY THIS TYPE EXISTS
// --------------------
// Two fault-reporting mechanisms read the same one-shot per-stream
// exception slot (StreamThread::stored_exception, drained by
// take_exception()):
//
//   1. This fork's throw_if_stream_exception(), called at the END of eval()
//      (transforms.cpp), which surfaces async worker-thread faults.
//   2. Upstream fe92a0565's cleanup handler inside eval_impl()'s tape loop,
//      which calls synchronize(s) on every open stream when a primitive
//      throws SYNCHRONOUSLY. synchronize() itself drains the slot.
//
// When both fire in one eval, (2) runs first and consumes the slot. Worse,
// eval() has NO try/catch around its eval_impl() call, so once eval_impl
// rethrows, eval()'s throw_if_stream_exception() is never reached at all.
// The async fault is therefore lost twice over: drained by synchronize(),
// then discarded by the bare catch(...) that guarded it. A real JACCL
// transport fault would be replaced by whatever downstream symptom threw
// synchronously -- exactly the wrong diagnosis for an ops team.
//
// The fix keeps BOTH mechanisms whole: upstream's cleanup still runs (it
// protects against real state corruption), and the drained fault is carried
// out on this exception instead of being dropped. Both causes are reported.
//
// Deliberately derives from std::runtime_error so every existing handler
// (including exo's runner bootstrap, which converts any Exception into a
// RunnerTerminationError) keeps working unchanged, and so pybind/nanobind
// maps it to a Python RuntimeError like any other MLX error.
class CombinedEvalFailure : public std::runtime_error {
 public:
  CombinedEvalFailure(
      std::string what,
      std::exception_ptr primitive_failure,
      std::exception_ptr stream_failure)
      : std::runtime_error(std::move(what)),
        primitive_failure_(std::move(primitive_failure)),
        stream_failure_(std::move(stream_failure)) {}

  // The synchronous throw from inside the tape loop.
  const std::exception_ptr& primitive_failure() const noexcept {
    return primitive_failure_;
  }

  // The async worker-thread fault drained during cleanup (the JACCL fault).
  const std::exception_ptr& stream_failure() const noexcept {
    return stream_failure_;
  }

 private:
  std::exception_ptr primitive_failure_;
  std::exception_ptr stream_failure_;
};

// exo-jaccl-fix (2026-09-10): build the combined exception. Kept here rather
// than in transforms.cpp so it is unit-testable without a full MLX build.
inline CombinedEvalFailure combined_eval_failure(
    const std::exception_ptr& primitive_failure,
    const std::exception_ptr& stream_failure) {
  std::string msg =
      "[eval] a primitive threw during eval AND a stream worker thread "
      "reported a fault. Both are reported below; the stream fault is "
      "usually the root cause (e.g. a JACCL RDMA collective error whose "
      "corrupt output made a downstream primitive fail).\n"
      "  (1) stream worker fault: " +
      describe_exception(stream_failure) +
      "\n"
      "  (2) primitive failure:   " +
      describe_exception(primitive_failure);
  return CombinedEvalFailure(
      std::move(msg), primitive_failure, stream_failure);
}

} // namespace mlx::core::scheduler
