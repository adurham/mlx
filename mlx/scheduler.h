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

namespace mlx::core::scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  std::thread thread;
  // exo-jaccl-fix (2026-07-01): holds the first exception thrown by a task on
  // this stream's worker thread. Captured instead of re-thrown (which would
  // std::terminate the process); rethrown on the calling thread at the next
  // take_exception() call (invoked from synchronize()). Guarded by mtx.
  std::exception_ptr stored_exception{nullptr};

  StreamThread() : stop(false), thread(&StreamThread::thread_fn, this) {}

  ~StreamThread() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cond.notify_one();
    thread.join();
  }

  void thread_fn() {
#if defined(__APPLE__)
    // exo-mlx-tune: env-gated QoS pin for stream worker threads.
    // Diagnosed via JACCL_TRACE_PROGRESS=1: rank 0 (the API/master
    // host) sees asymmetric busy-poll stalls (17M+ poll_iters)
    // during MTP verify all_reduces while rank 1 completes in 2
    // poll_iters. The comm-stream worker thread is getting
    // descheduled by the macOS scheduler — purely C++ busy-poll,
    // no Python re-entry. Pinning the thread to a higher QoS
    // keeps it on a P-core under contention.
    //
    // Off by default (safety: USER_INTERACTIVE has misbehaved on
    // some cluster states — opt in deliberately). Set
    // MLX_STREAM_QOS=user_initiated|user_interactive|default|utility
    // to enable. Once-per-process getenv() — cheap.
    static const int qos_class = [] {
      const char* v = std::getenv("MLX_STREAM_QOS");
      if (v == nullptr) return -1;
      if (std::strcmp(v, "user_interactive") == 0) return (int)QOS_CLASS_USER_INTERACTIVE;
      if (std::strcmp(v, "user_initiated") == 0) return (int)QOS_CLASS_USER_INITIATED;
      if (std::strcmp(v, "default") == 0) return (int)QOS_CLASS_DEFAULT;
      if (std::strcmp(v, "utility") == 0) return (int)QOS_CLASS_UTILITY;
      if (std::strcmp(v, "off") == 0) return -1;
      return -1;
    }();
    if (qos_class != -1) {
      pthread_set_qos_class_self_np((qos_class_t)qos_class, 0);
    }
#endif
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cond.wait(lk, [this] { return !this->q.empty() || this->stop; });
        if (q.empty() && stop) {
          return;
        }
        task = std::move(q.front());
        q.pop();
      }

      try {
        task();
      } catch (const std::exception& e) {
        // exo-jaccl-fix (2026-07-01): a task threw on the stream worker
        // thread. The OLD behavior re-threw here, which unwinds out of
        // thread_fn / std::thread -> std::terminate() -> SIGABRT, killing the
        // whole runner process. For a JACCL RDMA collective fault
        // (``[jaccl] all_reduce wc.status=N``) that meant a single transport
        // blip took down the entire cluster unrecoverably, with no catchable
        // error surfacing to Python.
        //
        // NEW: capture the exception into a per-stream exception_ptr and keep
        // the worker thread ALIVE. The next synchronize() on this stream
        // rethrows it on the CALLING (Python-facing) thread, where the exo
        // runner's try/except converts it into a clean RunnerTerminationError
        // and the supervisor restarts just this instance. Draining any queued
        // tasks below (see stored_exception_) prevents a wedged peer rank.
        std::fprintf(
            stderr,
            "[mlx scheduler] captured %s in task (surfacing at next "
            "synchronize): %s\n",
            typeid(e).name(),
            e.what());
        std::fflush(stderr);
        {
          std::lock_guard<std::mutex> lk(mtx);
          if (!stored_exception) {
            stored_exception = std::current_exception();
          }
        }
      } catch (...) {
        std::fprintf(
            stderr,
            "[mlx scheduler] captured unknown exception in task (surfacing "
            "at next synchronize)\n");
        std::fflush(stderr);
        {
          std::lock_guard<std::mutex> lk(mtx);
          if (!stored_exception) {
            stored_exception = std::current_exception();
          }
        }
      }
    }
  }

  void enqueue(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      if (stop) {
        throw std::runtime_error(
            "Cannot enqueue work after stream is stopped.");
      }
      q.emplace(std::move(f));
    }
    cond.notify_one();
  }

  // exo-jaccl-fix (2026-07-01): if a prior task on this stream captured an
  // exception, clear and return it so the caller (synchronize()) can rethrow
  // it on the Python-facing thread. Returns nullptr when the stream is clean.
  // One-shot: the stored exception is consumed so a single fault surfaces
  // exactly once and the stream is usable again after the instance restarts.
  std::exception_ptr take_exception() {
    std::lock_guard<std::mutex> lk(mtx);
    std::exception_ptr e = stored_exception;
    stored_exception = nullptr;
    return e;
  }
};

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

  // exo-jaccl-fix (2026-07-01): consume and return the captured exception for
  // stream ``s`` (nullptr if none). Called by synchronize() to rethrow a
  // worker-thread fault on the Python-facing thread instead of terminating.
  std::exception_ptr take_stream_exception(Stream s);

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

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;

 public:
  // exo-jaccl-fix (2026-07-01): sweep ALL stream worker threads and return the
  // first captured exception (consuming it), nullptr if all clean. Used by the
  // eval() path, which waits on arrays/events rather than calling
  // synchronize(Stream) and so can't target a single stream index.
  std::exception_ptr take_any_stream_exception() {
    std::shared_lock lock(threads_mtx_);
    for (auto& [idx, st] : threads_) {
      if (auto e = st->take_exception()) {
        return e;
      }
    }
    return nullptr;
  }
};

MLX_API Scheduler& scheduler();

template <typename F>
void enqueue(const Stream& stream, F&& f) {
  scheduler().enqueue(stream, std::forward<F>(f));
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

} // namespace mlx::core::scheduler
