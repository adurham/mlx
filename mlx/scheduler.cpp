// Copyright © 2023-2026 Apple Inc.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
#include <typeinfo>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/compile_impl.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core {

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
    f.wait();
    scheduler::check_error(s);
  } else {
    gpu::synchronize(s);
  }
  // exo-jaccl-fix (2026-07-01): after the stream has drained to this
  // synchronization point, rethrow any exception a task captured on the worker
  // thread (e.g. a JACCL RDMA collective fault). This surfaces the failure on
  // the CALLING thread as a normal catchable C++/Python exception instead of
  // having let it std::terminate the process on the worker thread. The runner's
  // try/except then converts it to a clean RunnerTerminationError.
  //
  // Kept alongside upstream's Error-based check_error() above rather than
  // replaced by it: check_error() only fires for CPU streams and only carries
  // a message string, whereas this fork's slot preserves the original
  // exception TYPE (nanobind maps by C++ type) and covers GPU streams too.
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
  detail::compile_clear_cache(detail::compile_cache());
  cpu::clear_streams();
  gpu::clear_streams();
}

namespace scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  std::thread thread;
  Error error;
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
    // exo-mlx-tune: env-gated QoS pin for stream worker threads. Diagnosed via
    // JACCL_TRACE_PROGRESS=1: the comm-stream worker thread gets descheduled by
    // the macOS scheduler during MTP verify all_reduces. Pinning it to a higher
    // QoS keeps it on a P-core under contention. Off by default; set
    // MLX_STREAM_QOS=user_initiated|user_interactive|default|utility.
    static const int qos_class = [] {
      const char* v = std::getenv("MLX_STREAM_QOS");
      if (v == nullptr)
        return -1;
      if (std::strcmp(v, "user_interactive") == 0)
        return (int)QOS_CLASS_USER_INTERACTIVE;
      if (std::strcmp(v, "user_initiated") == 0)
        return (int)QOS_CLASS_USER_INITIATED;
      if (std::strcmp(v, "default") == 0)
        return (int)QOS_CLASS_DEFAULT;
      if (std::strcmp(v, "utility") == 0)
        return (int)QOS_CLASS_UTILITY;
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

      // exo-jaccl-fix (2026-07-01): a task throwing here used to unwind out of
      // thread_fn -> std::thread -> std::terminate() -> SIGABRT, so a single
      // JACCL transport blip took down the whole runner with no catchable
      // error reaching Python. Capture into a per-stream exception_ptr and keep
      // the worker thread ALIVE; the next synchronize()/eval() rethrows it on
      // the calling thread. Scheduler::enqueue's own wrapper already converts
      // std::exception into upstream's Error, so only non-std throws and the
      // event paths reach this handler -- it is the last-resort net.
      try {
        task();
      } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "[mlx scheduler] captured %s in task (surfacing at next "
            "synchronize): %s\n",
            typeid(e).name(),
            e.what());
        std::fflush(stderr);
        std::lock_guard<std::mutex> lk(mtx);
        if (!stored_exception) {
          stored_exception = std::current_exception();
        }
      } catch (...) {
        std::fprintf(
            stderr,
            "[mlx scheduler] captured unknown exception in task (surfacing "
            "at next synchronize)\n");
        std::fflush(stderr);
        std::lock_guard<std::mutex> lk(mtx);
        if (!stored_exception) {
          stored_exception = std::current_exception();
        }
      }
    }
  }

  // exo-jaccl-fix (2026-07-01): if a prior task on this stream captured an
  // exception, clear and return it so the caller (synchronize()) can rethrow
  // it on the Python-facing thread. Returns nullptr when the stream is clean.
  std::exception_ptr take_exception() {
    std::lock_guard<std::mutex> lk(mtx);
    std::exception_ptr e = stored_exception;
    stored_exception = nullptr;
    return e;
  }

  void enqueue(std::function<void()> f) {
    if (is_main_thread()) {
      error.check();
    }
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

Scheduler::Scheduler() {
  is_main_thread();
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  auto& st = get_thread(s);
  st.enqueue([&st, task = std::move(task)]() mutable {
    try {
      task();
    } catch (const std::exception& error) {
      // Set error to stream only when no error happended before, to preserve
      // the earliest error.
      if (!st.error.valid()) {
        st.error.set_message(std::make_shared<std::string>(error.what()));
      }
      // exo-jaccl-fix (2026-07-01): ALSO stash the exception_ptr. Upstream's
      // Error only carries what(), but nanobind maps Python exception classes
      // from the C++ TYPE, and exo distinguishes JACCL faults by type -- so the
      // fork's slot has to keep the original object, not just its message.
      std::lock_guard<std::mutex> lk(st.mtx);
      if (!st.stored_exception) {
        st.stored_exception = std::current_exception();
      }
    }
  });
}

void Scheduler::wait_event(
    Stream s,
    Event event,
    std::function<void(Event&)> task) {
  assert(s.device == Device::cpu);
  auto& st = get_thread(s);
  st.enqueue([&st, event = std::move(event), task = std::move(task)]() mutable {
    task(event);
    // Poison current stream if the waited event has error.
    st.error.store_if_valid(event.load_error());
  });
}

void Scheduler::signal_event(
    Stream s,
    Event event,
    std::function<void(Event&)> task) {
  assert(s.device == Device::cpu);
  auto& st = get_thread(s);
  st.enqueue([&st, event = std::move(event), task = std::move(task)]() mutable {
    // Poison the signal event if current stream has error.
    if (st.error.valid()) {
      event.set_error(st.error);
    }
    task(event);
  });
}

void Scheduler::check_error(Stream s) {
  get_thread(s).error.check();
}

StreamThread& Scheduler::get_thread(Stream s) {
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      return *it->second.get();
    }
  }
  std::unique_lock lock(threads_mtx_);
  auto it = threads_.find(s.index);
  if (it == threads_.end()) {
    it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
  }
  return *it->second.get();
}

// exo-jaccl-fix (2026-07-01): consume the captured worker-thread exception for
// stream ``s``. Returns nullptr when the stream never faulted or was never
// created. Deliberately does NOT create the thread if it doesn't exist.
std::exception_ptr Scheduler::take_stream_exception(Stream s) {
  std::shared_lock lock(threads_mtx_);
  auto it = threads_.find(s.index);
  if (it == threads_.end()) {
    return nullptr;
  }
  return it->second->take_exception();
}

std::exception_ptr Scheduler::take_any_stream_exception() {
  std::shared_lock lock(threads_mtx_);
  for (auto& [idx, st] : threads_) {
    if (auto e = st->take_exception()) {
      return e;
    }
  }
  return nullptr;
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
