# `fe92a0565` cleanup can swallow a JACCL fault (fixed)

**Branch:** `upstream-sync-2026-09` (not yet merged to main).
**Found:** 2026-09-10, fork-aware review of the upstream cherry-pick batch.
**Status:** fixed on this branch. Not yet exercised on real hardware — see §6.

---

## 1. What happens

Two independent fault-reporting mechanisms read the **same one-shot
per-stream exception slot** (`StreamThread::stored_exception`, drained by
`take_exception()`, which clears the slot unconditionally on read):

1. **The fork's**, added 2026-07-01: `throw_if_stream_exception()` at the end
   of `eval()` (`mlx/transforms.cpp:392,398`). It exists because a JACCL RDMA
   collective fault is thrown on a *stream worker thread*; rethrowing there
   unwinds out of `std::thread` and `std::terminate`s the whole runner. The
   worker captures it into the slot instead, and this call surfaces it on the
   Python-facing thread, where exo's runner converts it to a clean
   `RunnerTerminationError` and the supervisor restarts just that instance.

2. **Upstream `fe92a0565`'s**, newly cherry-picked as `94f5832af`: a
   `catch(...)` around `eval_impl()`'s tape loop. When a primitive throws
   *synchronously*, it signals pending events and calls `synchronize(s)` on
   every open stream so that half-committed command buffers cannot leave an
   array readable-but-unwritten. `synchronize()` **also drains the slot**
   (`mlx/scheduler.cpp:26-28`) — and upstream discarded whatever came back
   via a bare `catch(...) {}` commented `// Preserve the original exception.`

JACCL pins every collective to one communication stream
(`JACCLGroup::communication_stream_`, `distributed/jaccl/jaccl.cpp:88`), and
`AllReduce` is constructed with that stream (`distributed/ops.cpp:34`).
`eval_impl` inserts `arr.primitive().stream()` into `open_streams`
(`transforms.cpp:248-249`). **So whenever the tape contains a JACCL
collective, that stream is in the set this loop synchronizes.**

Net effect before the fix: a tape containing both a JACCL collective (whose
async fault has landed in the slot) and a later synchronously-throwing
primitive loses the JACCL fault entirely.

## 2. It is worse than "the fork's mechanism runs second"

The original review framed this as a race for the slot. Reading `eval()`
closes it harder than that:

```cpp
eval_impl(std::move(outputs), false).wait();
scheduler::throw_if_stream_exception();   // transforms.cpp:398
```

There is **no `try`/`catch` around the `eval_impl()` call** — verified:
`grep -n 'try\|catch' mlx/transforms.cpp` reports handlers only at lines
243/325/335/339/343/345, all inside `eval_impl`. So once `eval_impl` rethrows,
line 398 never executes at all. The fork's mechanism does not lose a race; it
is skipped outright. Even leaving the fault in the slot would not help: it
would be picked up by some unrelated *later* `eval()` and reported against
the wrong operation, which is worse than losing it.

## 3. Real-world likelihood — why this was worth fixing

Both faults must land in one tape. That sounds unlikely until you notice the
two are **causally linked, not independent**:

- An RDMA fault does not fail cleanly on one side. It corrupts or aborts the
  collective on the peer too, so the collective's output array is garbage or
  never written.
- A tensor-parallel decode tape does not stop at the collective. Downstream
  ops consume its output in the *same* `eval()` — that is the point of
  overlapping compute and communication. A downstream op fed a corrupt or
  unwritten buffer is exactly the kind of thing that throws synchronously
  (shape/validation/allocation).

So the JACCL fault is a plausible *cause* of the synchronous throw that hides
it. Multiply by a server doing many `eval()`s per second for days on real
RDMA hardware, and "rare per call" stops being reassuring.

The cost is also asymmetric. The failure is silent and actively misleading:
the operator sees a downstream symptom (`[matmul] shapes ... do not match`)
and no hint of the transport fault that caused it — the single worst outcome
for diagnosing a distributed system, and precisely the class of fault the
2026-07-01 mechanism was hard-won to catch.

Verdict: **real, worth fixing.** Not a theoretical edge case, and the fix is
small and self-contained.

## 4. The fix

Keep both mechanisms whole. Upstream's cleanup is a genuine state-corruption
fix and is **not** weakened — every open stream is still flushed, in the same
order, and the original exception still propagates when nothing else is
wrong. The only change is that the drained fault is no longer thrown away:

- `mlx/scheduler.h` gains `CombinedEvalFailure` (derives from
  `std::runtime_error`, so every existing handler and the nanobind →
  Python `RuntimeError` mapping keep working unchanged), plus
  `combined_eval_failure()` and `describe_exception()`.
- `mlx/transforms.cpp`'s catch block captures what `synchronize(s)` throws
  (first wins, matching the slot's own first-wins semantics) instead of
  discarding it, and throws the combined exception naming **both** causes,
  with the stream fault listed first as the likely root cause.

Rejected alternative: putting the fault back into the slot. It fights the
data structure (single-producer, first-wins, cleared-on-read), races with a
genuinely new fault on that stream, and — per §2 — would surface against an
unrelated later `eval()`.

## 5. Test evidence

`tests/scheduler_combined_fault_tests.cpp` (registered in
`tests/CMakeLists.txt`), run against the **real** `mlx/scheduler.h` with
g++ 11.4 / C++20 — **7 test cases, 24 assertions, all passing.** It covers
the one-shot drain, first-wins storage, worker survival after a fault, and
that the combined exception carries both causes both textually and
structurally.

Two negative controls, because a green test proves nothing on its own:

- **Pre-fix behaviour fails the regression test.** Reproducing upstream's
  bare `catch(...) {}` and asking what the operator sees yields
  `[matmul] shapes (2,3) and (4,5) do not match` with the JACCL fault
  nowhere in it — the check fails, `[doctest] Status: FAILURE!`. Post-fix
  the same check passes.
- **Control-flow harness over the patched block's exact structure**, 8/8:
  the original exception propagates *unwrapped* when there is no stream
  fault; both causes are recoverable by type when there is; every stream is
  still flushed even when an early one faults; and the block does not run at
  all when the primitive succeeds.

`mlx/transforms.cpp` and the new test both pass `g++ -std=c++20
-fsyntax-only`.

## 6. What is NOT tested, and why

**No full MLX build, and no end-to-end JACCL test.** Both are honest gaps:

- This branch has never been built. The Linux host available here has no
  `cmake` and no LAPACK/BLAS headers, and `CMakeLists.txt` forces
  `MLX_BUILD_METAL` off on non-Darwin and gates the JACCL subdirectory on
  `Darwin AND MACOS_SDK_VERSION >= 26.2` — so JACCL cannot compile here at
  all, let alone run.
- The only machines that could build it are the two Mac Studios, which are
  serving live production inference. A full build was deliberately not run
  against them.
- Triggering the real combined scenario needs a two-node RDMA cluster with
  deliberately induced transport faults. That is not something to
  manufacture on the production cluster.

So: the *logic* of the fix is genuinely tested (it is pure exception
composition and control flow, and it is exercised by running code against
the real header). The *integration* — that a live JACCL fault plus a live
synchronous primitive throw produces this exception on real hardware —
remains unverified. **Before merging this branch, build it on a Studio while
the cluster is drained and run `tests`.** The new test needs no GPU, no
network, and no JACCL group.
