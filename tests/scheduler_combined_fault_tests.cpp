// Copyright © 2026 Apple Inc.
//
// exo-jaccl-fix (2026-09-10): regression tests for the interaction between
// this fork's async stream-fault reporting and upstream fe92a0565's
// synchronous-throw cleanup handler in eval_impl().
//
// THE BUG THESE COVER
// -------------------
// Both mechanisms read the SAME one-shot per-stream exception slot
// (StreamThread::stored_exception, drained by take_exception()):
//
//   1. throw_if_stream_exception() at the end of eval(), which surfaces
//      async worker-thread faults such as a JACCL RDMA collective error.
//   2. fe92a0565's catch handler inside eval_impl()'s tape loop, which on a
//      SYNCHRONOUS primitive throw calls synchronize(s) on every open
//      stream -- and synchronize() drains that same slot.
//
// JACCL pins its collectives to one communication stream and AllReduce is
// constructed with it, so that stream is in open_streams whenever the tape
// holds a collective. When both faults occur in one eval, (2) runs first,
// drains the slot, and upstream's bare catch(...) discarded the result.
// eval() has no try/catch around eval_impl(), so the rethrow also skips
// eval()'s own throw_if_stream_exception(). The JACCL fault was lost twice
// over and the operator saw only the downstream symptom.
//
// WHAT IS AND IS NOT COVERED HERE
// -------------------------------
// Covered, with real objects and real threads: the one-shot drain semantics
// that make the shadowing possible, first-wins storage, and that
// combined_eval_failure carries BOTH causes and names them in what().
//
// NOT covered here: the end-to-end path through eval_impl() with an actual
// JACCL collective on the tape. That needs a live two-node RDMA cluster and
// deliberately induced transport faults; it cannot be unit-tested. See
// scheduler_combined_fault_tests.cpp's standalone harness note.

#include "doctest/doctest.h"

#include <functional>
#include <future>
#include <stdexcept>
#include <string>

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;

namespace {

struct JacclLikeFault : public std::runtime_error {
  JacclLikeFault() : std::runtime_error("[jaccl] all_reduce wc.status=12") {}
};

struct PrimitiveLikeFault : public std::runtime_error {
  PrimitiveLikeFault()
      : std::runtime_error("[matmul] shapes (2,3) and (4,5) do not match") {}
};

// Upstream 2026-09 moved StreamThread's definition out of scheduler.h into
// scheduler.cpp, so these tests can no longer construct one directly. Drive a
// real dedicated CPU stream through the public scheduler API instead -- same
// slot, same first-wins/one-shot semantics, and it exercises the wrapper in
// Scheduler::enqueue too (which is where a std::exception is now captured).
struct FaultStream {
  Stream s = new_stream(Device::cpu);

  void enqueue(std::function<void()> f) {
    scheduler::enqueue(s, std::move(f));
  }

  // Block until every task queued so far has run.
  void drain() {
    std::promise<void> done;
    auto fut = done.get_future();
    scheduler::enqueue(s, [&done]() { done.set_value(); });
    fut.wait();
  }

  std::exception_ptr take_exception() {
    return scheduler::scheduler().take_stream_exception(s);
  }

  // Upstream's Error slot is populated alongside ours by Scheduler::enqueue;
  // clear it so a later synchronize() in an unrelated test does not throw.
  void clear_upstream_error() {
    try {
      scheduler::check_error(s);
    } catch (...) {
    }
  }
};

std::exception_ptr make_ptr_from(void (*thrower)()) {
  try {
    thrower();
  } catch (...) {
    return std::current_exception();
  }
  return nullptr;
}

void throw_jaccl() {
  throw JacclLikeFault();
}
void throw_primitive() {
  throw PrimitiveLikeFault();
}

} // namespace

// The property that makes shadowing possible in the first place. If this
// ever changes to non-consuming, the fix below becomes unnecessary -- so
// pin it down explicitly rather than leaving it implied.
TEST_CASE("stream exception slot is one-shot") {
  FaultStream st;

  st.enqueue([]() { throw JacclLikeFault(); });
  // Drain the queue deterministically: a second task cannot run until the
  // first has been through the capture handler.
  st.drain();

  auto first = st.take_exception();
  REQUIRE(first != nullptr);
  CHECK_THROWS_AS(std::rethrow_exception(first), JacclLikeFault);

  // Consumed: a second read must come back empty. This is precisely why
  // synchronize() draining the slot hides the fault from a later
  // throw_if_stream_exception().
  CHECK(st.take_exception() == nullptr);
  st.clear_upstream_error();
}

// First-wins storage, matching the `if (!stored_exception)` guard in
// thread_fn. The combined-failure handler mirrors this, so keep them honest
// about each other.
TEST_CASE("stream exception slot keeps the FIRST fault") {
  FaultStream st;

  st.enqueue([]() { throw JacclLikeFault(); });
  st.enqueue([]() { throw PrimitiveLikeFault(); });
  st.drain();

  auto e = st.take_exception();
  REQUIRE(e != nullptr);
  CHECK_THROWS_AS(std::rethrow_exception(e), JacclLikeFault);
  CHECK(st.take_exception() == nullptr);
  st.clear_upstream_error();
}

// The worker thread must SURVIVE a fault -- the whole point of capturing
// instead of rethrowing (a rethrow unwinds out of std::thread and
// terminates the process).
TEST_CASE("stream worker survives a captured fault") {
  FaultStream st;

  st.enqueue([]() { throw JacclLikeFault(); });
  std::promise<void> ran;
  auto fut = ran.get_future();
  st.enqueue([&ran]() { ran.set_value(); });
  // Would hang (then fail the suite) if the worker had died.
  CHECK(fut.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  CHECK(st.take_exception() != nullptr);
  st.clear_upstream_error();
}

TEST_CASE("describe_exception renders without rethrowing to the caller") {
  CHECK(scheduler::describe_exception(nullptr) == "<none>");

  auto e = make_ptr_from(&throw_jaccl);
  auto text = scheduler::describe_exception(e);
  CHECK(text.find("all_reduce wc.status=12") != std::string::npos);

  // Must not have consumed or invalidated the pointer.
  CHECK_THROWS_AS(std::rethrow_exception(e), JacclLikeFault);
}

// The actual fix: both causes survive, and both are visible in what().
TEST_CASE("combined_eval_failure carries BOTH causes") {
  auto prim = make_ptr_from(&throw_primitive);
  auto stream = make_ptr_from(&throw_jaccl);
  REQUIRE(prim != nullptr);
  REQUIRE(stream != nullptr);

  auto combined = scheduler::combined_eval_failure(prim, stream);

  const std::string what = combined.what();
  // The JACCL fault -- the one upstream's bare catch(...) discarded -- is
  // the regression being guarded. Its absence is the bug.
  CHECK(what.find("all_reduce wc.status=12") != std::string::npos);
  CHECK(what.find("shapes (2,3) and (4,5)") != std::string::npos);
  CHECK(what.find("stream worker fault") != std::string::npos);
  CHECK(what.find("primitive failure") != std::string::npos);

  // Both are also retrievable structurally, not just as text, so a handler
  // can branch on the real types.
  CHECK_THROWS_AS(
      std::rethrow_exception(combined.stream_failure()), JacclLikeFault);
  CHECK_THROWS_AS(
      std::rethrow_exception(combined.primitive_failure()),
      PrimitiveLikeFault);
}

// Must stay catchable by every handler that already exists -- notably exo's
// runner bootstrap, which converts any Exception into a
// RunnerTerminationError, and the nanobind translator that maps
// std::runtime_error to a Python RuntimeError.
TEST_CASE("CombinedEvalFailure is catchable as std::runtime_error") {
  auto prim = make_ptr_from(&throw_primitive);
  auto stream = make_ptr_from(&throw_jaccl);

  CHECK_THROWS_AS(
      throw scheduler::combined_eval_failure(prim, stream),
      std::runtime_error);
  CHECK_THROWS_AS(
      throw scheduler::combined_eval_failure(prim, stream), std::exception);
}

// Regression guard on the SHAPE of the fix. Reproduces eval_impl()'s catch
// handler in miniature: a synchronous throw, cleanup that drains the slot,
// and the requirement that the drained fault is not lost. Written against
// the same helper the real handler uses.
TEST_CASE("cleanup that drains the slot must not lose the fault") {
  FaultStream st;

  // An async JACCL-like fault lands in the slot.
  st.enqueue([]() { throw JacclLikeFault(); });
  st.drain();

  // A primitive throws synchronously; the handler runs cleanup.
  std::exception_ptr primitive_failure = make_ptr_from(&throw_primitive);
  std::exception_ptr stream_failure;

  // Cleanup drains the slot, exactly as synchronize(s) does.
  if (auto drained = st.take_exception()) {
    try {
      std::rethrow_exception(drained);
    } catch (...) {
      if (!stream_failure) {
        stream_failure = std::current_exception();
      }
    }
  }

  // The pre-fix behaviour discarded stream_failure here and rethrew only
  // the primitive failure, and eval()'s throw_if_stream_exception() could
  // not recover it because the slot was already empty:
  CHECK(st.take_exception() == nullptr);

  // Post-fix: the fault is still reported.
  REQUIRE(stream_failure != nullptr);
  bool reported_jaccl = false;
  try {
    throw scheduler::combined_eval_failure(primitive_failure, stream_failure);
  } catch (const scheduler::CombinedEvalFailure& e) {
    reported_jaccl =
        std::string(e.what()).find("all_reduce wc.status=12") !=
        std::string::npos;
  }
  CHECK(reported_jaccl);
  st.clear_upstream_error();
}
