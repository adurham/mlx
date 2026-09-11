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

## 6. What was NOT tested at authoring time, and why

> **SUPERSEDED 2026-09-11 — see section 7.** The claims in this section were
> the author's honest belief at the time, and two of them turned out to be
> wrong: the fix *was* buildable and testable off-cluster all along. Kept
> verbatim for the record; read section 7 for what is actually verified.

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

## 7. Verification and merge (2026-09-11)

Independent review and re-verification corrected two claims in section 6 and
found two blockers, both now closed. Nothing about the fix's logic changed;
what changed is that it is now actually exercised.

### 7.1 Section 6 was wrong about testability

- **"JACCL is gated on SDK >= 26.2 so only the Studios can build it"** is
  irrelevant *and* wrong. Irrelevant because the code under test is not JACCL
  code: it is generic CPU-stream machinery in `transforms.cpp`/`scheduler.h`.
  JACCL pins collectives to a plain **CPU** stream
  (`jaccl.cpp:88 communication_stream_ = new_stream(Device::cpu)`) and
  `all_sum` is `encoder.dispatch([]{...})` (`jaccl.cpp:113`) whose lambda
  throws on the stream worker thread — no RDMA, no ibverbs, no Metal, no
  second node required to reproduce the fault's *routing*. Wrong because
  `MACOS_SDK_VERSION` is only set inside `if(MLX_BUILD_METAL)`, so with
  `MLX_BUILD_METAL=OFF` the gate is never even evaluated; and a developer
  laptop's SDK (26.5) satisfies it anyway.
- **"The only machines that could build it are the production Studios"** is
  false. All functional verification below ran on a non-production MacBook.

Only the **RDMA → slot** leg still needs the cluster: that a real
`wc.status != IBV_WC_SUCCESS` lands in `stored_exception` during a live
two-node run. The hook for it already exists (`JACCL_INJECT_WC_ERROR=K`,
`lib/jaccl/mesh_impl.h:1753`). Everything downstream of the slot — which is
all this fix touches — is now verified by real execution.

### 7.2 Blocker: the committed tests could not fail

`tests/scheduler_combined_fault_tests.cpp` passes **7/7 with the fix
reverted** (measured, not inferred). No case in it calls `eval()` or
`eval_impl()`; it tests scheduler helpers plus a hand-rolled replica of the
catch block. A regression guard that cannot fail is not a guard.

Closed by `tests/eval_combined_fault_tests.cpp`, which drives the real
`eval()` → `eval_impl()` path with a tape of
`ones → AsyncStreamFault (pinned CPU comm stream) → SyncPrimitiveFault`.
Measured on a CPU-only build (macOS 26.6.2 / Xcode 26.6):

| Tree | New guard | Old committed file |
|---|---|---|
| With the fix | **6/6 pass**, 17/17 assertions | 7/7 pass |
| `transforms.cpp` hunk reverted | **FAILS, 4/6, Status: FAILURE** | 7/7 pass |

On the reverted tree `eval()` throws a bare `std::invalid_argument` reading
only `[matmul] shapes (2,3) and (4,5) do not match`, JACCL fault absent —
the original bug, reproduced by the guard. Full suite with the fix:
**257/257 cases, 3278/3278 assertions, 0 failures, 0 `ERROR:` lines**.
Guard re-run 10× consecutively: 10/10, no flakiness.

### 7.3 Blocker: the branch lacked main's Metal 4.1/Xcode 27 fixes

`origin/main` was **not** an ancestor of `upstream-sync-2026-09`
(merge-base `e40a416b2`), so merging as-is would have shipped a tree without
`837339cd1` / `e65e4f893` / `e19a1bbdf` and broken the Metal build on the
Studios (Xcode 27.0 / Metal 4.1). Closed by merging `origin/main` into the
branch first. The two commit ranges touch **disjoint file sets** (37 files vs
18, zero overlap), so the merge is structural with no content resolution:
every main-side file is byte-identical to `origin/main` in the result, and
1,085 parent-unique lines across both sides survive with zero loss.

### 7.4 Known limitation, deliberately not "fixed"

The handler keeps only the **first** stream fault, and `open_streams` is a
`std::set<Stream>` ordered by index — so with two faulting streams in one
eval, the lower-indexed one wins regardless of relevance and the other is
dropped. In production JACCL's comm stream is created at group-init and
normally has a low index, so it normally wins, but *"the JACCL fault is never
lost"* is not unconditionally true. Documented by the
`with two faulting streams only one fault is surfaced` case rather than
papered over. Collecting all stream faults into a vector would remove the
caveat; not done here to keep the merge diff minimal.

### 7.5 Python-visible exception type change (release-note item)

nanobind maps by C++ type: `std::invalid_argument` → **ValueError**,
`std::runtime_error` → **RuntimeError**, and there are no custom exception
translators in `python/src/`. `CombinedEvalFailure` derives from
`std::runtime_error`, so **in the combined case only**, Python sees
`RuntimeError` where it previously saw `ValueError`. Proven at runtime: the
combined exception is catchable as `std::runtime_error` and not as
`std::invalid_argument`.

Audited independently with an AST walk (not grep) over `src/exo` +
`mlx-lm`: **388 handlers scanned, 164 narrow (type-specific) ones, 15 of
which also wrap an eval-forcing call**, and **0 `except RuntimeError`
handlers wrapping an eval-forcing call** that would newly swallow the wrapped
type. Of the 15, only `utils_mlx.py:230` (`except ValueError` around
`mx.eval(layer)`) is both narrow-non-RuntimeError and eval-forcing on a
collective-capable path — and it sits inside `if group is None:`, the
single-device branch, where no JACCL group and therefore no combined case
exists. `utils_mlx.py:1918`'s `except RuntimeError` re-raises unless the
message contains `"does not support a TCP-only coord group"`, which
`CombinedEvalFailure`'s message does not. **No regression found**, but the
type change is real and worth a release note.

