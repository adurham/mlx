// Copyright © 2026 Apple Inc.
//
// exo-jaccl-fix (2026-09-11): END-TO-END regression guard for the combined
// fault path, driving the REAL eval() -> eval_impl() control flow.
//
// WHY THIS FILE EXISTS SEPARATELY FROM scheduler_combined_fault_tests.cpp
// ----------------------------------------------------------------------
// That file tests the scheduler primitives in isolation (one-shot drain,
// first-wins storage, combined_eval_failure construction) plus a hand-rolled
// replica of eval_impl()'s catch block. None of its cases call eval(), which
// means **it passes unchanged with the fix reverted** -- a regression guard
// that cannot fail is not a guard. Measured 2026-09-11: reverting the
// transforms.cpp hunk leaves that file at 7/7 green.
//
// The cases below close that gap. They build a real tape, dispatch a real
// throwing task onto a real stream worker thread, and call the real eval().
// With the fix reverted, "combined fault reports BOTH causes" FAILS: the
// async fault is drained by upstream's cleanup synchronize() and discarded by
// its bare catch(...), and eval()'s own throw_if_stream_exception() is never
// reached because eval() has no try/catch around eval_impl().
//
// WHAT IS STILL NOT COVERED
// -------------------------
// The RDMA -> slot leg: that a real ``wc.status != IBV_WC_SUCCESS`` lands in
// StreamThread::stored_exception during a live two-node run. That needs the
// production cluster and JACCL_INJECT_WC_ERROR. Everything downstream of the
// slot -- which is all this fix touches -- is exercised here.
//
// PORTABILITY: every primitive below is pinned to an explicit CPU stream, so
// these cases behave identically in a Metal, CUDA, or CPU-only build and need
// no GPU, no network, and no JACCL group.

#include "doctest/doctest.h"

#include <chrono>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

using namespace mlx::core;

namespace {

// Shaped like a real JACCL transport fault. jaccl.cpp's all_sum body is
// ``encoder.dispatch([]{ group_->all_sum(...); })`` and the fault is a
// std::runtime_error thrown from that lambda on the stream worker thread --
// so a dispatched throwing lambda reproduces the fault's path into
// StreamThread::stored_exception exactly, with no RDMA involved.
constexpr const char* kJacclFault =
    "[jaccl] all_reduce wc.status=12 wr_id=0x1 byte_len=0";
constexpr const char* kSecondStreamFault =
    "[otherstream] unrelated worker-thread fault";
// Shaped like the downstream symptom: a collective whose output is garbage or
// unwritten makes the next op fail its own validation, synchronously.
constexpr const char* kPrimitiveFault =
    "[matmul] shapes (2,3) and (4,5) do not match";

// Faults ASYNCHRONOUSLY, on the stream worker thread, like a JACCL collective.
class AsyncStreamFault : public UnaryPrimitive {
 public:
  AsyncStreamFault(Stream s, const char* msg)
      : UnaryPrimitive(s), msg_(msg) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override {
    out.set_data(allocator::malloc(out.nbytes()));
    auto& encoder = cpu::get_command_encoder(stream());
    encoder.set_input_array(inputs[0]);
    encoder.set_output_array(out);
    const char* msg = msg_;
    encoder.dispatch([msg]() { throw std::runtime_error(msg); });
  }

  void eval_gpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("AsyncStreamFault is CPU-only");
  }

  const char* name() const override {
    return "AsyncStreamFault";
  }

 private:
  const char* msg_;
};

// Faults SYNCHRONOUSLY, from inside eval_cpu, like an argument-validation
// failure. Deliberately std::invalid_argument: that is what MLX throws for
// most validation errors, and it is what nanobind maps to a Python
// ValueError, so it also pins the exception-type behaviour of the no-stream-
// fault path.
class SyncPrimitiveFault : public UnaryPrimitive {
 public:
  explicit SyncPrimitiveFault(Stream s) : UnaryPrimitive(s) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::invalid_argument(kPrimitiveFault);
  }

  void eval_gpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("SyncPrimitiveFault is CPU-only");
  }

  const char* name() const override {
    return "SyncPrimitiveFault";
  }
};

bool mentions(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

// The exception slot is process-global and swept across ALL streams by
// throw_if_stream_exception(). Leaving one behind would fail an unrelated
// later test case, so every case brackets itself with this.
void drain_all_stream_faults() {
  for (int i = 0; i < 8; ++i) {
    try {
      scheduler::throw_if_stream_exception();
      return;
    } catch (...) {
    }
  }
}

struct DrainGuard {
  DrainGuard() {
    drain_all_stream_faults();
  }
  ~DrainGuard() {
    drain_all_stream_faults();
  }
};

} // namespace

// ===========================================================================
// THE regression guard. This is the case that FAILS with the fix reverted.
// ===========================================================================
//
// Tape: ones -> AsyncStreamFault (pinned CPU "comm" stream, throws on the
// worker thread) -> SyncPrimitiveFault (throws synchronously). That is the
// exact production shape: an RDMA fault aborts the collective, and the op
// consuming its output fails in the SAME eval.
//
// Ordering is deterministic, not racy: the cleanup loop calls synchronize()
// on the comm stream, which enqueues a task BEHIND the throwing one and waits
// for it, so stored_exception is always populated before take_exception()
// reads it.
TEST_CASE("eval reports BOTH a stream fault and a primitive fault") {
  DrainGuard guard;

  auto def = new_stream(Device::cpu);
  auto comm = new_stream(Device::cpu); // JACCL's pinned communication stream

  bool threw_combined = false;
  bool stream_cause_by_type = false;
  bool primitive_cause_by_type = false;
  std::string what;

  try {
    auto a = ones({4}, float32, def);
    auto b = array(
        a.shape(),
        a.dtype(),
        std::make_shared<AsyncStreamFault>(comm, kJacclFault),
        {a});
    auto c = array(
        b.shape(), b.dtype(), std::make_shared<SyncPrimitiveFault>(def), {b});
    eval(c);
  } catch (const scheduler::CombinedEvalFailure& e) {
    threw_combined = true;
    what = e.what();
    try {
      std::rethrow_exception(e.stream_failure());
    } catch (const std::exception& inner) {
      stream_cause_by_type = mentions(inner.what(), kJacclFault);
    } catch (...) {
    }
    try {
      std::rethrow_exception(e.primitive_failure());
    } catch (const std::invalid_argument& inner) {
      primitive_cause_by_type = mentions(inner.what(), kPrimitiveFault);
    } catch (...) {
    }
  } catch (const std::exception& e) {
    what = e.what();
  }

  INFO("eval() threw: " << what);

  REQUIRE(threw_combined);

  // The pre-fix bug in one assertion: the async fault is ABSENT from what the
  // operator sees, leaving only the downstream symptom.
  CHECK(mentions(what, "all_reduce wc.status=12"));
  CHECK(mentions(what, "shapes (2,3) and (4,5)"));

  // Both causes must also survive structurally, with their original types, so
  // a handler can branch on them rather than string-matching.
  CHECK(stream_cause_by_type);
  CHECK(primitive_cause_by_type);
}

// Upstream fe92a0565's own case must be untouched: with no stream fault, the
// original exception propagates UNWRAPPED and with its original type. This is
// what keeps nanobind's std::invalid_argument -> Python ValueError mapping
// intact for every ordinary validation error, and guards against the fix
// over-wrapping.
TEST_CASE("eval leaves an ordinary primitive failure unwrapped") {
  DrainGuard guard;

  auto def = new_stream(Device::cpu);

  bool wrapped = false;
  bool unwrapped_invalid_argument = false;
  std::string what;

  try {
    auto a = ones({4}, float32, def);
    auto c = array(
        a.shape(), a.dtype(), std::make_shared<SyncPrimitiveFault>(def), {a});
    eval(c);
  } catch (const scheduler::CombinedEvalFailure& e) {
    wrapped = true;
    what = e.what();
  } catch (const std::invalid_argument& e) {
    unwrapped_invalid_argument = true;
    what = e.what();
  } catch (const std::exception& e) {
    what = e.what();
  }

  INFO("eval() threw: " << what);
  CHECK_FALSE(wrapped);
  CHECK(unwrapped_invalid_argument);
  CHECK(mentions(what, kPrimitiveFault));
}

// The 2026-07-01 mechanism must still work on its own: an async fault with no
// synchronous throw surfaces via eval()'s throw_if_stream_exception(), NOT as
// a CombinedEvalFailure (there is no primitive failure to combine it with).
TEST_CASE("eval surfaces a lone stream fault unwrapped") {
  DrainGuard guard;

  auto def = new_stream(Device::cpu);
  auto comm = new_stream(Device::cpu);

  bool wrapped = false;
  std::string what;

  try {
    auto a = ones({4}, float32, def);
    auto b = array(
        a.shape(),
        a.dtype(),
        std::make_shared<AsyncStreamFault>(comm, kJacclFault),
        {a});
    eval(b);
  } catch (const scheduler::CombinedEvalFailure& e) {
    wrapped = true;
    what = e.what();
  } catch (const std::exception& e) {
    what = e.what();
  }

  INFO("eval() threw: " << what);
  CHECK_FALSE(wrapped);
  CHECK(mentions(what, "all_reduce wc.status=12"));
}

// The fix deliberately does NOT put the drained fault back into the slot.
// Re-storing it would make an unrelated later eval() report it wildly out of
// context -- a misattributed fault is worse than a lost one. Pin that: after
// a combined failure, the next eval on the SAME stream must be clean.
TEST_CASE("a combined failure does not leak into a later eval") {
  DrainGuard guard;

  auto def = new_stream(Device::cpu);
  auto comm = new_stream(Device::cpu);

  try {
    auto a = ones({4}, float32, def);
    auto b = array(
        a.shape(),
        a.dtype(),
        std::make_shared<AsyncStreamFault>(comm, kJacclFault),
        {a});
    auto c = array(
        b.shape(), b.dtype(), std::make_shared<SyncPrimitiveFault>(def), {b});
    eval(c);
  } catch (...) {
  }

  // Same comm stream, ordinary work. Must not resurrect the JACCL fault, and
  // the worker thread must have survived capturing it.
  bool clean = false;
  std::string leaked;
  try {
    auto x = ones({4}, float32, comm);
    auto y = add(x, x, comm);
    eval(y);
    clean = (y.data<float>()[0] == 2.0f && y.data<float>()[3] == 2.0f);
  } catch (const std::exception& e) {
    leaked = e.what();
  }

  INFO("later eval threw: " << leaked);
  CHECK(leaked.empty());
  CHECK(clean);
}

// Documents a real limitation rather than asserting a stronger guarantee than
// the code makes. The handler keeps only the FIRST stream fault, and
// open_streams is a std::set<Stream> ordered by index -- so with two faulting
// streams in one eval, the lower-indexed one wins regardless of relevance,
// and the other is dropped. In production JACCL's comm stream is created at
// group-init and normally has a low index, so it normally wins; but "the
// JACCL fault is never lost" is NOT unconditionally true. If this is ever
// changed to report all faults, this case is the one to update.
TEST_CASE("with two faulting streams only one fault is surfaced") {
  DrainGuard guard;

  auto def = new_stream(Device::cpu);
  auto comm = new_stream(Device::cpu);
  auto other = new_stream(Device::cpu);

  bool threw_combined = false;
  std::string what;

  try {
    auto a = ones({4}, float32, def);
    auto b = array(
        a.shape(),
        a.dtype(),
        std::make_shared<AsyncStreamFault>(comm, kJacclFault),
        {a});
    auto c = array(
        a.shape(),
        a.dtype(),
        std::make_shared<AsyncStreamFault>(other, kSecondStreamFault),
        {a});
    auto d = add(b, c, def);
    auto e = array(
        d.shape(), d.dtype(), std::make_shared<SyncPrimitiveFault>(def), {d});
    eval(e);
  } catch (const scheduler::CombinedEvalFailure& e) {
    threw_combined = true;
    what = e.what();
  } catch (const std::exception& e) {
    what = e.what();
  }

  INFO("eval() threw: " << what);
  CHECK(threw_combined);
  // At least one stream fault is reported, and the primitive failure always
  // is. Which stream wins is index-ordered, so it is not asserted here.
  // (Computed into a local: doctest's expression decomposition rejects a
  // top-level || with "Expression Too Complex".)
  const bool some_stream_fault_reported =
      mentions(what, "all_reduce wc.status=12") ||
      mentions(what, "[otherstream]");
  CHECK(some_stream_fault_reported);
  CHECK(mentions(what, "shapes (2,3) and (4,5)"));
}

// Everything above throws on purpose. Make sure the process is still healthy
// afterwards: stream worker threads alive, ordinary eval still correct.
TEST_CASE("streams remain usable after combined failures") {
  DrainGuard guard;

  auto s = new_stream(Device::cpu);
  auto x = ones({4}, float32, s);
  auto y = add(x, x, s);
  eval(y);
  CHECK(y.data<float>()[0] == 2.0f);
  CHECK(y.data<float>()[3] == 2.0f);
}
