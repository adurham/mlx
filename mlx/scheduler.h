// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <typeinfo>
#include <unordered_map>

#if defined(__APPLE__)
#include <pthread/qos.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
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

    // exo-mlx-tune: Mach real-time time-constraint thread policy.
    //
    // Background: USER_INTERACTIVE QoS is a hint, not a hard pin.
    // Under prefill GPU-enqueue contention the macOS scheduler still
    // preempts the comm-stream worker thread mid-busy-poll, producing
    // the asymmetric JACCL stalls diagnosed 2026-05-15 (one rank's
    // poll_iters in 5-28M while peer rank polls 1-10 for the same
    // call_id; hardware CQE was ready, the poll thread just wasn't
    // running).
    //
    // THREAD_TIME_CONSTRAINT_POLICY is the real-time class macOS
    // uses for CoreAudio HAL threads — the OS will NOT preempt a
    // thread in this class during its computation window. It's a
    // hard contract: "give me <computation> µs of CPU within
    // <constraint> µs of the trigger, do not deschedule."
    //
    // We're not periodic, so period=0. The computation budget should
    // exceed typical all_reduce CPU wall (a few hundred µs for the
    // 1 MB main-model TP all_sums) but not be so large that the
    // scheduler refuses to admit us. constraint = computation +
    // tolerance.
    //
    // Defaults: computation=500µs, constraint=2000µs. Tunable via
    // env (microseconds): MLX_STREAM_RT_COMPUTATION_US,
    // MLX_STREAM_RT_CONSTRAINT_US, MLX_STREAM_RT_PERIOD_US.
    //
    // Set MLX_STREAM_RT=1 to enable. Off by default (RT priority
    // is system-wide; if the contract numbers are wrong it can starve
    // other threads).
    static const bool rt_enabled = [] {
      const char* v = std::getenv("MLX_STREAM_RT");
      return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (rt_enabled) {
      // Cache the timebase ratio (ns → mach absolute time units).
      static const mach_timebase_info_data_t tbi = [] {
        mach_timebase_info_data_t t;
        mach_timebase_info(&t);
        return t;
      }();
      auto us_to_abs = [](uint64_t us) -> uint32_t {
        // abs = ns * denom / numer; convert µs first.
        return static_cast<uint32_t>((us * 1000ULL * tbi.denom) / tbi.numer);
      };
      auto getenv_us = [](const char* name, uint64_t dflt) -> uint64_t {
        const char* v = std::getenv(name);
        if (v == nullptr) return dflt;
        char* end = nullptr;
        unsigned long long n = std::strtoull(v, &end, 10);
        return (end == v) ? dflt : static_cast<uint64_t>(n);
      };

      thread_time_constraint_policy_data_t policy;
      policy.period = us_to_abs(getenv_us("MLX_STREAM_RT_PERIOD_US", 0));
      policy.computation = us_to_abs(getenv_us("MLX_STREAM_RT_COMPUTATION_US", 500));
      policy.constraint = us_to_abs(getenv_us("MLX_STREAM_RT_CONSTRAINT_US", 2000));
      policy.preemptible = TRUE; // IGNORED per docs, but set for clarity.

      mach_port_t self_port = mach_thread_self();
      kern_return_t kr = thread_policy_set(
          self_port,
          THREAD_TIME_CONSTRAINT_POLICY,
          reinterpret_cast<thread_policy_t>(&policy),
          THREAD_TIME_CONSTRAINT_POLICY_COUNT);
      mach_port_deallocate(mach_task_self(), self_port);
      if (kr != KERN_SUCCESS) {
        std::fprintf(
            stderr,
            "[mlx scheduler] thread_policy_set(TIME_CONSTRAINT) failed: kr=%d "
            "(period=%u computation=%u constraint=%u abs-units)\n",
            kr,
            policy.period,
            policy.computation,
            policy.constraint);
        std::fflush(stderr);
        // Non-fatal: fall through; if RT setup failed the worker
        // still runs under whatever QoS was set above.
      }
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
        std::fprintf(
            stderr,
            "[mlx scheduler] uncaught %s in task: %s\n",
            typeid(e).name(),
            e.what());
        std::fflush(stderr);
        throw;
      } catch (...) {
        std::fprintf(stderr, "[mlx scheduler] unknown uncaught exception\n");
        std::fflush(stderr);
        throw;
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
    if (n_tasks_old > 1) {
      completion_cv.wait(lk, [this, n_tasks_old] {
        return this->n_active_tasks() < n_tasks_old;
      });
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;
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

} // namespace mlx::core::scheduler
