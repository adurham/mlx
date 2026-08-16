// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <span>
#include <thread>
#include <vector>
#include <mach/mach_time.h>

#include "jaccl/rdma.h"

constexpr int MESH_MAX_PEERS = 8;
constexpr int MESH_PIPELINE = 2;
constexpr int64_t MAX_BUFFER_SIZE = FRAME_SIZE * (1 << (BUFFER_SIZES - 1));

namespace jaccl {

// Pre-lambda ack barrier gate. Default OFF.
//
// Adds ack_sync_pre() calls at the top of every collective lambda to
// close the inter-lambda window where peer SEND lands at our empty
// data-QP recv FIFO and UC silently drops → permanent wedge.
// Gated behind a runtime env for A/B testing once bootstrap is stable.
inline bool jaccl_ack_sync_pre_enabled() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char* e = std::getenv("MLX_JACCL_ACK_SYNC_PRE");
    enabled = (e != nullptr && e[0] == '1' && e[1] == '\0');
    checked = true;
  }
  return enabled;
}

// Per-stage RDMA progress logging gated on JACCL_TRACE_PROGRESS=1.
// Output goes to stderr. Inline so the gate-check inlines and the
// branch is dead-code eliminated when the env var is unset (runtime
// check is once per call site, not once per WC).
inline bool jaccl_progress_enabled() {
  static bool checked = false;
  static bool enabled = false;
  if (!checked) {
    const char* e = std::getenv("JACCL_TRACE_PROGRESS");
    enabled = (e != nullptr && e[0] == '1' && e[1] == '\0');
    checked = true;
  }
  return enabled;
}

// Poll-loop instrumentation gated on JACCL_POLL_INSTRUMENT=1.
//
// Captures per-call statistics that distinguish thread-was-not-scheduled
// (Mach RT class would fix) from time-spent-inside-driver (Mach RT class
// CANNOT fix; the stall is in librdma/Apple kernel). For each poll call
// we record:
//   total_wall_us       — total elapsed time of the while-loop
//   total_iters         — how many times the loop body ran
//   iters_with_cqes     — iterations where ibv_poll_cq returned ≥ 1 CQE
//   wall_us_in_poll     — cumulative time spent INSIDE ibv_poll_cq
//   max_single_poll_us  — slowest single ibv_poll_cq call
//
// If wall_us_in_poll ≈ total_wall_us, the thread WAS running the poll
// continuously — time vanished inside the driver. Mach RT (already
// tested 2026-05-16 and FAILED to fix the stall) cannot help here.
//
// If wall_us_in_poll ≪ total_wall_us, the thread was descheduled
// between iterations. Mach RT should have fixed this.
//
// If max_single_poll_us is in the seconds range, librdma is blocking
// inside ibv_poll_cq for some reason (driver lock, NIC interrupt
// coalescing, etc).
//
// Emits one stderr line per call whose total_wall_us exceeds
// JACCL_POLL_INSTRUMENT_THRESHOLD_US (default 100000 = 100 ms).
inline bool jaccl_poll_instrument_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("JACCL_POLL_INSTRUMENT");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return v;
}

inline uint64_t jaccl_poll_instrument_threshold_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("JACCL_POLL_INSTRUMENT_THRESHOLD_US");
    if (e == nullptr) return (uint64_t)100000;
    char* end = nullptr;
    unsigned long long n = std::strtoull(e, &end, 10);
    return (end == e) ? (uint64_t)100000 : (uint64_t)n;
  }();
  return v;
}

inline uint64_t mach_ticks_to_us(uint64_t ticks) {
  static const mach_timebase_info_data_t tbi = [] {
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  // ticks * numer / denom = nanoseconds; / 1000 = microseconds.
  return (ticks * tbi.numer) / (tbi.denom * 1000ULL);
}

// UC-drop stall recovery timeout (microseconds).
//
// Thunderbolt RDMA connections are UC (unreliable, no retransmit). Under
// c>=2 the two TP ranks can lead each other by hundreds-to-thousands of
// collectives; when that lead overruns the ACK_RECV pool (or the NIC's
// completion ring wedges — ring_indicies_err), a posted ACK/data SEND or
// RECV work completion is silently lost. The owning collective's poll loop
// then spins on a counter that will never reach zero, hanging the runner
// until exo's supervisor SIGKILLs it 45-180s later (with IOConnectUnmapMemory
// GPU-teardown noise) and force-replaces the whole instance.
//
// Instead: if a collective poll loop makes ZERO forward progress for this
// long, throw. mlx's jaccl scheduler exception path (2026-07-01) catches it
// and turns it into a clean instance re-place in seconds — no SIGKILL, no GPU
// fault, far below the supervisor watchdog. Normal collectives complete in
// well under a second, so an 8s default has a >8x safety margin over the
// slowest healthy collective observed. 0 disables (legacy hang-until-SIGKILL).
inline uint64_t jaccl_stall_timeout_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("MLX_JACCL_STALL_TIMEOUT_US");
    return e ? std::strtoull(e, nullptr, 10) : 8000000ULL;
  }();
  return v;
}

// Soft-RC (software reliability over UC): how long the ACK barrier waits with
// zero progress before RETRANSMITTING its outstanding ACK work-requests, and
// how many times before giving up. UC silently drops a SEND/RECV (or its CQE
// is lost when the completion ring wedges), so rather than spin forever we
// re-post the outstanding ACK_SEND/RECV. ACKs are idempotent — a duplicate
// ACK_RECV is absorbed by cached_ack_recvs_, and need_send-- tolerates the
// extra completion — so retransmit is safe with no dedup. This turns the
// silent-drop wedge into a self-healing collective (no throw, no re-place)
// for the common transient-loss case. After _MAX attempts fail (e.g. a truly
// wedged completion ring) we fall through to the StallWatch throw so the
// runner still self-heals. Retransmit interval defaults to 500ms — far above a
// healthy sub-ms ACK, so it only fires on genuine loss. 0 disables (revert to
// pure StallWatch-throw behavior).
inline uint64_t jaccl_ack_retransmit_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("MLX_JACCL_ACK_RETRANSMIT_US");
    return e ? std::strtoull(e, nullptr, 10) : 500000ULL;
  }();
  return v;
}

// Quiet period for the p2p send()/recv() DRAIN loops specifically,
// separate from the collective ack retransmit timer above.
//
// WHY THIS IS SPLIT OUT (design doc Section 71). Both loops previously
// reused jaccl_ack_retransmit_us() (500ms). Measured consequence on the
// PP batched-decode path: when the first send of a token is genuinely
// lost on the wire (a real, silent UC drop -- UC has no flow control and
// no NAK), the receiver sits ~525ms in its drain loop waiting for a
// chunk that will never arrive, and the sender then waits its own 500ms
// before retransmitting. Recovery from ONE dropped frame therefore costs
// ~1.0s, and at roughly one drop per decode token that is the entire
// ~60x decode-throughput shortfall (0.47 tok/s against a 30 tok/s bar).
//
// Healthy calls on this link complete in 69-150us end to end, and p50
// barrier latency is 39us -- so 500ms is ~5000x the real round trip.
// The timer is not protecting against anything at that scale; it is
// purely the cost of NOTICING a drop.
//
// Section 51 already proved that lowering the GLOBAL retransmit timer to
// 10ms breaks generation outright: at that value it fires below the real
// round trip for large collective transfers and retransmits frames that
// are merely in flight, producing zero output. That result is why this
// is a SEPARATE knob rather than a change to the value above -- the p2p
// drain loops are the only place where the measured round trip is
// microseconds, so they can afford a far tighter quiet period without
// touching the collective path's timing at all.
//
// Default 25ms: ~170x the observed healthy round trip (still far above
// any plausible in-flight window, so it cannot mistake slow for lost),
// while cutting the per-drop penalty from ~1.0s to ~50ms -- a 20x
// reduction in the amplifier. Set MLX_JACCL_P2P_DRAIN_QUIET_US to
// override, or to 500000 to restore the old behaviour exactly.
inline uint64_t jaccl_p2p_drain_quiet_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("MLX_JACCL_P2P_DRAIN_QUIET_US");
    return e ? std::strtoull(e, nullptr, 10) : 25000ULL;
  }();
  return v;
}
inline int jaccl_ack_retransmit_max() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_ACK_RETRANSMIT_MAX");
    return e ? std::atoi(e) : 40;
  }();
  return v;
}
// Liveness backstop for send()/recv()'s new RDMA-native p2p_retry_exchange
// (design doc Section 37 Phase 1, 2026-08-10), tracking zero-progress time
// on the PEER's reported bitmask specifically -- NOT the generic
// jaccl_stall_timeout_us() (8s default) used by drain_acks/StallWatch
// elsewhere in this file. This distinction is load-bearing, not cosmetic:
// the OLD TCP p2p_channel_ this replaces was deliberately given its OWN
// much longer deadline (MLX_JACCL_P2P_RECV_RETRY_DEADLINE_SECS, 300s
// default -- see mesh.cpp's rebuild_p2p_channel(), design doc Section 38)
// specifically because real-hardware evidence showed the peer can
// legitimately take far longer than 8s to report new progress under
// genuine production load (confirmed via a live faulthandler dump: the
// peer was actively mid-decode-step, not wedged). Reusing the generic 8s
// default here would reintroduce exactly the false-positive-timeout bug
// Section 39 fixed on the DATA path, just relocated to this barrier.
// Same cost-asymmetry justification as Section 38: a false positive here
// drops healthy in-flight work for no reason, while a genuinely dead peer
// still surfaces (RDMA completion errors, not just silence). Default
// matches Section 38's 300s; distinct env var since the mechanism changed
// from a TCP-recv deadline to a progress-based StallWatch.
inline uint64_t jaccl_p2p_retry_stall_timeout_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("MLX_JACCL_P2P_RETRY_STALL_TIMEOUT_SECS");
    const double secs = e ? std::atof(e) : 300.0;
    return static_cast<uint64_t>(secs * 1e6);
  }();
  return v;
}
// Confirmed (ack-of-ack) barrier. Default OFF (adds a round-trip per collective).
// When ON, the ACK barrier is a reliable two-round handshake: a rank does not
// proceed until it has confirmation the peer RECEIVED its ack — deterministically
// closing the recv-side UC-drop race where one rank's ack is lost, it proceeds,
// and the peer wedges. Enable for c>=2 correctness (perf penalty).
inline bool jaccl_env_true(const char* name) {
  const char* e = std::getenv(name);
  return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}
inline bool jaccl_confirmed_barrier_enabled() {
  static const bool v = jaccl_env_true("MLX_JACCL_CONFIRMED_BARRIER");
  return v;
}
// Split gates so pre vs post can be isolated (the pre barrier is entangled with
// RDMA data-recv ordering; post runs after the data has drained). Either the
// combined flag or the specific one enables each side.
// Standing pre-posted recv pool on the DATA QP for the sz=0 size class --
// design doc Section 52 (2026-08-15). Closes the empty-recv-FIFO UC drop
// that cost a 500ms retransmit timer on ~4.5% of PP decode barriers; see
// post_data_recv_pool()'s own comment for the full measurement and for why
// a pool is used instead of a cross-rank barrier. Default ON: the failure it
// prevents is a silent 500ms-per-occurrence throughput collapse. Set
// MLX_JACCL_DATA_RECV_POOL=0 to A/B against the old behaviour.
inline bool jaccl_data_recv_pool_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("MLX_JACCL_DATA_RECV_POOL");
    return e == nullptr || !(e[0] == '0' && e[1] == '\0');
  }();
  return v;
}
inline bool jaccl_confirmed_barrier_pre() {
  static const bool v = jaccl_confirmed_barrier_enabled() ||
      jaccl_env_true("MLX_JACCL_CONFIRMED_BARRIER_PRE");
  return v;
}
inline bool jaccl_confirmed_barrier_post() {
  static const bool v = jaccl_confirmed_barrier_enabled() ||
      jaccl_env_true("MLX_JACCL_CONFIRMED_BARRIER_POST");
  return v;
}
// Reliable data-phase all_reduce (ARQ over UC): sequence-tagged chunks +
// full-message assembly + coordinator-barrier bitmask retransmit, so a dropped
// reduction chunk is detected and re-sent instead of wedging (all_reduce
// STALLED). Requires a coordinator (top-level group). Default OFF (perf cost +
// core-path change). Implies the confirmed barrier machinery.
inline bool jaccl_reliable_data_enabled() {
  static const bool v = jaccl_env_true("MLX_JACCL_RELIABLE_DATA");
  return v;
}
// Cap the reliable-path chunk buffer size class. Large UC sends (>= ~64KB /
// sz>=4) do not reliably COMPLETE on Apple's librdma (they stick, which is
// likely the same failure the UC all_reduce wedged on); cap the chunk to a
// size class that reliably completes. Default 0 (FRAME_SIZE=4096). Tunable.
inline int jaccl_reliable_max_sz() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_RELIABLE_MAX_SZ");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}
// Microseconds to sleep per idle drain poll (0 completions). Prevents the
// reliable-path drain from 100%-spinning and starving Metal/GPU threads.
inline int jaccl_reliable_idle_us() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_RELIABLE_IDLE_US");
    return e ? std::atoi(e) : 15;
  }();
  return v;
}
// Pipeline depth: how many chunks to keep in flight (sends AND recvs) at sz=0.
// Small (4KB) UC sends are clean and can overlap, so pipelining hides the RTT
// that made the stop-and-wait path too slow for c>=2 prefill. Capped at
// NUM_BUFFERS at the call site. Default 2 (the current buffer allotment).
inline int jaccl_reliable_inflight() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_RELIABLE_INFLIGHT");
    return e ? std::atoi(e) : 2;
  }();
  return v;
}

// Optimistic reliable path (v2): eliminates the per-collective TCP coordinator
// barrier for SMALL collectives (num_chunks <= jaccl_reliable_small_chunks).
// Decode all_sums (8-24KB) become a single UC message each way with NO
// rendezvous: a rank exits as soon as it has the peer's data and its own send
// CQEs. Reliability is preserved by (a) a standing pre-posted recv pool that
// can never be "not ready", (b) send buffers partitioned by call parity and
// retained one collective so a stuck peer's quiet-timeout STATUS message
// (carrying its got-bitmask) can be answered with retransmits from the NEXT
// collective's poll loop, and (c) the unchanged 15s deadline as backstop.
// Cross-collective skew is provably <= 1 (a rank cannot enter call K+1
// without the peer's K data, which the peer only sends from inside K).
// LARGE collectives (prefill) keep the TCP-barrier rendezvous exit (the
// barrier is amortized there) but share the same standing recv pool and the
// 12-byte {call_id, seq, len} wire header. Default OFF.
inline bool jaccl_reliable_optimistic_enabled() {
  static const bool v = jaccl_env_true("MLX_JACCL_RELIABLE_OPTIMISTIC");
  return v;
}
// Max chunks for the optimistic (no-rendezvous) exit. Must stay <=
// NUM_BUFFERS/2 - 1 (parity partition reserves one slot for STATUS).
inline int jaccl_reliable_small_chunks() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_RELIABLE_SMALL_CHUNKS");
    int n = e ? std::atoi(e) : 3;
    return std::min(n, NUM_BUFFERS / 2 - 1);
  }();
  return v;
}

// Watches a collective poll loop for a permanently-stuck in-flight counter
// (a lost UC completion). tick() is called once per poll iteration with the
// loop's current progress metric (in_flight, or need_send+need_recv): any
// change resets the deadline; no change for jaccl_stall_timeout_us() throws.
// Overhead is one mach_absolute_time() per iteration, negligible beside the
// ibv_poll_cq the loop already performs.
struct StallWatch {
  uint64_t timeout_us;
  uint64_t last_progress_ticks;
  long last_metric;
  explicit StallWatch(long metric)
      : timeout_us(jaccl_stall_timeout_us()),
        last_progress_ticks(mach_absolute_time()),
        last_metric(metric) {}
  void tick(long metric, const char* what, int rank, uint32_t call_id) {
    if (timeout_us == 0) {
      return;
    }
    if (metric != last_metric) {
      last_metric = metric;
      last_progress_ticks = mach_absolute_time();
      return;
    }
    if (mach_ticks_to_us(mach_absolute_time() - last_progress_ticks) >
        timeout_us) {
      std::ostringstream msg;
      msg << "[jaccl] " << what << " STALLED rank=" << rank
          << " call_id=" << call_id << " metric=" << metric
          << " (no forward progress for >" << (timeout_us / 1000)
          << "ms; UC completion lost — throwing for clean re-place)";
      throw std::runtime_error(msg.str());
    }
  }
};

class MeshImpl {
 public:
  MeshImpl(
      int rank,
      int size,
      std::vector<Connection>& conns,
      std::vector<Connection>& ack_conns,
      std::vector<Connection>& pool_conns,
      std::vector<Connection>& p2p_retry_conns,
      std::vector<SharedBuffer>& buffers,
      std::vector<SharedBuffer>& ack_send_buffers,
      std::vector<SharedBuffer>& ack_recv_buffers,
      std::vector<SharedBuffer>& p2p_retry_send_buffers,
      std::vector<SharedBuffer>& p2p_retry_recv_buffers)
      : rank_(rank),
        size_(size),
        connections_(conns),
        ack_connections_(ack_conns),
        pool_connections_(pool_conns),
        p2p_retry_connections_(p2p_retry_conns),
        buffers_(buffers),
        ack_send_buffers_(ack_send_buffers),
        ack_recv_buffers_(ack_recv_buffers),
        p2p_retry_send_buffers_(p2p_retry_send_buffers),
        p2p_retry_recv_buffers_(p2p_retry_recv_buffers) {}

  MeshImpl() : rank_(0), size_(1) {}

  // Wire up the reliable TCP coordinator (top-level group only) for the
  // confirmed (ack-of-ack) barrier. Non-owning; the SideChannel outlives this.
  void set_coordinator(SideChannel* coordinator) {
    coordinator_ = coordinator;
  }

  // Wire up the dedicated p2p retry channel for send()/recv() (2026-07-17).
  // Isolated from coordinator_ -- see p2p_channel_ member comment in
  // mesh.h for the collision this fixes. Non-owning; outlives this.
  void set_p2p_channel(SideChannel* p2p_channel) {
    p2p_channel_ = p2p_channel;
  }

  // ── reliable_all_reduce v2: optimistic, no per-collective TCP barrier ──
  // See jaccl_reliable_optimistic_enabled() for the protocol summary. Wire
  // format: every message is [V2Hdr{call_id, seq, len}][payload]. seq ==
  // V2_STATUS_SEQ marks a STATUS message whose payload is the sender's
  // got[] byte-mask for that call (sent only on quiet-timeout while stuck).

  struct V2Hdr {
    uint32_t call_id;
    uint32_t seq;
    uint32_t len;
  };
  static constexpr uint32_t V2_STATUS_SEQ = 0xFFFFFFFFu;
  // Fixed tag for send()/recv()'s p2p_retry_barrier calls (2026-07-17).
  // Both send() and recv() always pass this same constant for every round
  // of a transfer -- see p2p_retry_barrier's doc comment in rdma.h for why
  // a fixed tag (not call_id) is used here.
  static constexpr uint32_t kP2PRetryTag = 0xA5A5A5A5u;
  // Sentinel round numbers for send()/recv()'s pre/post-transfer
  // rendezvous calls on p2p_channel_ (2026-07-17, revision 3 -- these
  // used to be raw, unframed barrier() calls; now framed p2p_retry_barrier
  // calls, to keep every message on p2p_channel_ uniformly framed -- see
  // send()'s pre-transfer-rendezvous comment for the full rationale).
  // max()/max()-1 can never collide with a real retry-loop round (bounded
  // by max_rounds, ~8-32 typically) or with each other.
  static constexpr uint32_t kP2PPreBarrierRound = 0xFFFFFFFFu;
  static constexpr uint32_t kP2PPostBarrierRound = 0xFFFFFFFEu;
  static constexpr int V2_HDR = static_cast<int>(sizeof(V2Hdr));

  // ── Section 37 Phase 1 (2026-08-10): send()/recv()'s got-bitmask retry
  // exchange, migrated off the TCP p2p_retry_barrier onto a dedicated
  // RDMA-UC QP (p2p_retry_connections_). Design consult-reviewed across
  // three rounds before implementation -- see design doc Section 42/43 for
  // the full rationale. Three load-bearing decisions this implementation
  // depends on, summarized here so a future reader doesn't have to
  // re-derive them from the review transcripts:
  //
  // 1. NO lockstep rendezvous. The old TCP barrier was a true 2-way
  //    blocking rendezvous (both sides send-then-recv simultaneously), so
  //    round numbers trivially matched on the wire. An RDMA-UC exchange
  //    CANNOT reproduce that without reintroducing a deadlock risk (a
  //    consult review caught this directly: naive round-for-round matching
  //    over lossy UC reintroduces the exact lockstep-desync bug class this
  //    migration exists to eliminate). Instead: `round` is a DIAGNOSTIC-
  //    ONLY field, never gated on. Correctness comes from (2) and (3).
  //
  // 2. seq (send_seq_[dst] / recv_seq_[src], already computed once per
  //    transfer by both send() and recv() -- see their own field comments)
  //    is the epoch key, NOT call_id. A consult review confirmed call_id
  //    is wrong here for the same reason the 2026-08-08 header-seq fix
  //    exists: call_id is a per-process global counter, not guaranteed
  //    equal between the two ranks for "the same" logical transfer. seq is
  //    a per-(ordered-pair) counter that increments symmetrically on both
  //    sides by construction. Combined with data_src_rank (below) it
  //    uniquely identifies one transfer even if both directions are
  //    active concurrently (send_seq_[B] on rank A and recv_seq_[A] on
  //    rank B can coincidentally share a numeric value with an unrelated
  //    transfer in the OPPOSITE direction, since the two directions have
  //    independent counters).
  //
  // 3. NO acking. Bitmask frames are periodically REBROADCAST by whichever
  //    side has one to report, at the caller's own natural round cadence
  //    (drain_quiet_us) -- exactly like the data path's own chunk
  //    retransmit, and exactly like drain_acks' soft-RC retransmit
  //    elsewhere in this file. The receiver of those frames OR-accumulates
  //    them (idempotent, safe under drop/dup/reorder -- a got-bit, once
  //    true, is true for the rest of the transfer). A consult review
  //    confirmed an explicit ack protocol is not just unnecessary but
  //    actively risky here: it reintroduces a two-generals tail case (the
  //    last ack can always be the dropped message) that requires either an
  //    unbounded resend-forever, a probabilistic N-retransmit mitigation
  //    (not a real fix), or a retained-replay responder cache (adds real
  //    complexity). Removing acking sidesteps the whole class: nobody
  //    waits to learn whether the OTHER side received THEIR report; each
  //    side only cares about accumulating a complete picture of what IT
  //    has heard, which is entirely a function of frames it has itself
  //    received. The existing StallWatch backstop (throw after
  //    jaccl_stall_timeout_us of zero forward progress, caught by jaccl's
  //    scheduler exception path for a clean re-place) is the sole failure
  //    mode if a peer is genuinely gone -- same backstop this file already
  //    relies on everywhere else, not a new one.
  //
  // Wire frame -- one per RDMA-UC message, fits in one FRAME_SIZE (4096B)
  // buffer on the dedicated p2p_retry_connections_ QP. Payload is the
  // SAME byte-per-chunk representation send()/recv() already use for
  // `got` (vector<uint8_t>, 1 byte per chunk, not bit-packed) -- no new
  // encoding, no bit-indexing scheme, just chunked transmission of the
  // existing representation. A transfer's bitmask (up to 0xFFFF chunks
  // per send()'s own throw-guard) is at most 65535 bytes, split into
  // ceil(65535 / (FRAME_SIZE - sizeof(P2PFrameHdr))) = 17 frames.
  struct P2PFrameHdr {
    uint32_t magic; // framing sanity; distinct from V2Hdr/reliable_barrier
    uint32_t data_src_rank; // rank_ of send()'s caller for this transfer --
                             // disambiguates two transfers sharing a
                             // numerically equal `seq` in OPPOSITE
                             // directions (send_seq_/recv_seq_ are
                             // independent per-direction counters, so
                             // equality is only guaranteed WITHIN one
                             // direction -- see class doc comment above)
    uint32_t seq; // send_seq_[dst] / recv_seq_[src] for this transfer
    uint32_t round; // DIAGNOSTIC ONLY -- never gated on, see class comment
    uint32_t num_frames; // frame count of the bitmask this message is
                          // part of; receiver sanity-checks against its
                          // own independently-computed value, does not
                          // blindly trust it for buffer sizing
    uint32_t frame_index; // 0..num_frames-1
    uint32_t frame_len; // valid payload bytes in THIS frame
  };
  static constexpr uint32_t kP2PFrameMagic = 0x4a325032u; // "J2P2", reused
                                                           // from the old
                                                           // p2p_retry_
                                                           // barrier's tag
  static constexpr int P2P_HDR = static_cast<int>(sizeof(P2PFrameHdr));
  static constexpr int P2P_PAYLOAD_CAP = FRAME_SIZE - P2P_HDR;
  // Standing recv-pool depth per peer on p2p_retry_connections_. Must
  // exceed the max 17 frames/bitmask (0xFFFF chunks) with headroom for a
  // retransmit burst landing before the previous round's frames are all
  // drained; mirrors ACK_RECV_POOL's pre-post-and-replenish shape.
  static constexpr int P2P_RETRY_NUM_SLOTS = 24;

  // v2 uses ONE uniform size class (the reliable cap) for every message.
  // Apple librdma errors when a send's size class doesn't match the posted
  // recv's (IBV_WC_LOC_LEN_ERR — the same FIFO-mismatch that motivated the
  // subgroup ACK QP), and the standing pool recvs are posted long before the
  // message sizes are known. Uniform framing sidesteps the whole class:
  // decode all_sums (8-24KB) are 1-2 chunks; a 2-byte barrier wastes a frame
  // (1.6us wire time at 80Gbps — irrelevant).
  static int v2_size_class(int64_t msg) {
    (void)msg;
    return std::min(jaccl_reliable_max_sz(), BUFFER_SIZES - 1);
  }

  void v2_ensure_pool(int peer) {
    if (v2_pool_posted_) {
      return;
    }
    v2_pool_sz_ = std::min(jaccl_reliable_max_sz(), BUFFER_SIZES - 1);
    // ROOT-CAUSE FIX (2026-07-17): post on pool_connections_ (dedicated QP),
    // not connections_ (shared with raw send()/recv(), used by exo's
    // Pipeline-Parallel p2p handoff). See pool_connections_ member comment
    // for the full collision this isolation fixes.
    for (int b = 0; b < NUM_BUFFERS; b++) {
      auto& rb = recv_buffer(v2_pool_sz_, b, peer);
      zero_recv_buffer(rb);
      pool_connections_[peer].post_recv(
          rb, make_wr_id(0, POOL_RECV_WR, b, peer));
    }
    v2_pool_posted_ = true;
    std::fprintf(
        stderr,
        "[jaccl-v2] rank=%d standing pool armed (%d recvs, sz=%d)\n",
        rank_, NUM_BUFFERS, v2_pool_sz_);
    std::fflush(stderr);
  }

  template <typename T, typename ReduceOp>
  void reliable_all_reduce_v2(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }
    if (size_ <= 1 || size == 0) {
      return;
    }
    if (size_ != 2) {
      throw std::runtime_error(
          "[jaccl] reliable_all_reduce_v2 only supports 2 ranks");
    }
    // Loud-fail instead of silent UB (2026-07-17): reliable_all_reduce_v2
    // should only be reachable when pool_connections_ was populated at
    // construction (see MeshGroup ctor / pool_connections_ member
    // comment) -- but that's currently enforced transitively via the
    // coordinator_ != nullptr gate at the call site in all_reduce(). If
    // that gating logic ever drifts, indexing an empty pool_connections_
    // span below is silent undefined behavior. Assert the real invariant
    // directly instead of relying on it staying in sync elsewhere.
    if (pool_connections_.empty()) {
      throw std::runtime_error(
          "[jaccl] reliable_all_reduce_v2 called with no pool_connections_ "
          "(dedicated v2 QP) -- this should be unreachable; check the "
          "coordinator_ gating in all_reduce()");
    }
    const int peer = (rank_ == 0) ? 1 : 0;
    v2_ensure_pool(peer);

    const int64_t total_bytes = size * static_cast<int64_t>(sizeof(T));
    const int sz = v2_size_class(total_bytes);
    const int64_t chunk_bytes =
        static_cast<int64_t>(FRAME_SIZE) * (1 << sz) - V2_HDR;
    const int num_chunks =
        static_cast<int>((total_bytes + chunk_bytes - 1) / chunk_bytes);
    const bool small = num_chunks <= jaccl_reliable_small_chunks();
    const int half = NUM_BUFFERS / 2;
    const int base = (call_id & 1) ? half : 0;
    const int status_slot = base + half - 1;
    // Data slots: small -> base+c (c < half-1, status slot reserved).
    // Large -> rotate over all `half` parity slots.
    const int data_slots = small ? (half - 1) : half;

    std::vector<char> asm_buf(total_bytes, 0);
    std::vector<uint8_t> got(num_chunks, 0);
    int all_recv = 0;
    std::vector<uint8_t> peer_want; // peer's missing-chunk mask (this call)
    bool have_peer_status = false;
    bool peer_in_call = false; // any message of THIS call seen from peer
    int chunks_posted = 0; // first-pass sends issued

    const uint64_t _t0 = mach_absolute_time();
    const uint64_t _deadline_us = 15000000;
    const uint64_t quiet_us = jaccl_ack_retransmit_us();

    static std::atomic<int> _v2_calls{0};
    const bool _log = _v2_calls.fetch_add(1) < 8 || jaccl_progress_enabled();
    if (_log) {
      std::fprintf(
          stderr,
          "[jaccl-v2] ENTER rank=%d call_id=%u total_bytes=%lld sz=%d "
          "num_chunks=%d small=%d\n",
          rank_, call_id, (long long)total_bytes, sz, num_chunks, small ? 1 : 0);
      std::fflush(stderr);
    }

    // Apply lookahead stash from the previous call's loop.
    if (v2_stash_.call_id == call_id) {
      for (auto& [seq, bytes] : v2_stash_.chunks) {
        if (seq < static_cast<uint32_t>(num_chunks) && !got[seq]) {
          int64_t off = static_cast<int64_t>(seq) * chunk_bytes;
          int64_t len = std::min(
              static_cast<int64_t>(bytes.size()), total_bytes - off);
          if (len > 0) {
            std::memcpy(asm_buf.data() + off, bytes.data(), len);
            got[seq] = 1;
            all_recv++;
          }
        }
      }
      if (small && v2_stash_.has_status &&
          v2_stash_.peer_got.size() ==
              static_cast<size_t>(num_chunks)) {
        peer_want.assign(num_chunks, 0);
        for (int k = 0; k < num_chunks; k++) {
          peer_want[k] = v2_stash_.peer_got[k] ? 0 : 1;
        }
        have_peer_status = true;
      }
      peer_in_call = true;
      v2_stash_ = V2Stash{};
    } else if (v2_stash_.call_id != 0 && v2_stash_.call_id < call_id) {
      v2_stash_ = V2Stash{}; // stale
    }

    // Write chunk c of out_ptr (with header) into `slot` and post it.
    auto post_chunk = [&](uint32_t c, int slot) {
      auto& sb = send_buffer(sz, slot);
      char* p = sb.data<char>();
      int64_t off = static_cast<int64_t>(c) * chunk_bytes;
      int64_t len = std::min(chunk_bytes, total_bytes - off);
      V2Hdr hdr{call_id, c, static_cast<uint32_t>(len)};
      std::memcpy(p, &hdr, V2_HDR);
      std::memcpy(
          p + V2_HDR, reinterpret_cast<const char*>(out_ptr) + off,
          static_cast<size_t>(len));
      JACCL_DMA_BARRIER();
      pool_connections_[peer].post_send(
          sb, make_wr_id(call_id, SEND_WR, slot, peer));
      v2_send_outstanding_[slot]++;
    };

    auto post_status = [&]() {
      auto& sb = send_buffer(v2_pool_sz_, status_slot);
      char* p = sb.data<char>();
      V2Hdr hdr{call_id, V2_STATUS_SEQ, static_cast<uint32_t>(num_chunks)};
      std::memcpy(p, &hdr, V2_HDR);
      std::memcpy(p + V2_HDR, got.data(), num_chunks);
      JACCL_DMA_BARRIER();
      pool_connections_[peer].post_send(
          sb, make_wr_id(call_id, SEND_WR, status_slot, peer));
      v2_send_outstanding_[status_slot]++;
    };

    // Retransmit-service for the PREVIOUS small call: re-post the retained
    // parity buffers verbatim (they still hold [hdr][payload]) when free.
    auto service_prev = [&]() {
      if (v2_prev_want_.empty() || !v2_prev_small_) {
        return;
      }
      const int prev_base = (v2_prev_call_ & 1) ? half : 0;
      for (int k = 0; k < v2_prev_num_chunks_; k++) {
        int slot = prev_base + k;
        if (v2_prev_want_[k] && v2_send_outstanding_[slot] == 0) {
          auto& sb = send_buffer(v2_prev_sz_, slot);
          pool_connections_[peer].post_send(
              sb, make_wr_id(v2_prev_call_, SEND_WR, slot, peer));
          v2_send_outstanding_[slot]++;
          v2_prev_want_[k] = 0;
        }
      }
      bool any = false;
      for (auto w : v2_prev_want_) {
        any = any || (w != 0);
      }
      if (!any) {
        v2_prev_want_.clear();
      }
    };

    // Consume one standing-pool recv completion (buffer index `buff`).
    // Returns true if it made forward progress for THIS call.
    auto consume_pool = [&](int buff) -> bool {
      JACCL_DMA_BARRIER();
      auto& rb = recv_buffer(v2_pool_sz_, buff, peer);
      const char* p = rb.data<char>();
      V2Hdr hdr;
      std::memcpy(&hdr, p, V2_HDR);
      bool progress = false;
      if (hdr.call_id == 0) {
        // Spurious/empty completion (pre-zeroed buffer). Log and drop.
        std::fprintf(
            stderr, "[jaccl-v2] rank=%d empty pool recv (call=%u)\n",
            rank_, call_id);
      } else if (hdr.call_id == call_id) {
        peer_in_call = true;
        if (hdr.seq == V2_STATUS_SEQ) {
          if (hdr.len == static_cast<uint32_t>(num_chunks)) {
            peer_want.assign(num_chunks, 0);
            for (int k = 0; k < num_chunks; k++) {
              peer_want[k] =
                  static_cast<uint8_t>(p[V2_HDR + k]) ? 0 : 1;
            }
            have_peer_status = true;
            progress = true;
          }
        } else if (hdr.seq < static_cast<uint32_t>(num_chunks)) {
          int64_t off = static_cast<int64_t>(hdr.seq) * chunk_bytes;
          int64_t len = std::min(
              static_cast<int64_t>(hdr.len), total_bytes - off);
          if (!got[hdr.seq] && len > 0 &&
              hdr.len <= static_cast<uint32_t>(chunk_bytes)) {
            std::memcpy(asm_buf.data() + off, p + V2_HDR, len);
            got[hdr.seq] = 1;
            all_recv++;
            progress = true;
          }
        }
      } else if (hdr.call_id == call_id + 1) {
        // Peer ran ahead (optimistic exit). Stash for the next call.
        if (v2_stash_.call_id != call_id + 1) {
          v2_stash_ = V2Stash{};
          v2_stash_.call_id = call_id + 1;
        }
        if (hdr.seq == V2_STATUS_SEQ) {
          v2_stash_.has_status = true;
          v2_stash_.peer_got.assign(
              p + V2_HDR, p + V2_HDR + std::min<uint32_t>(hdr.len, 16384));
        } else if (v2_stash_.chunks.size() < 512) {
          uint32_t len = std::min<uint32_t>(
              hdr.len, static_cast<uint32_t>(rb.size() - V2_HDR));
          v2_stash_.chunks.emplace_back(
              hdr.seq, std::vector<char>(p + V2_HDR, p + V2_HDR + len));
        }
      } else if (hdr.call_id < call_id) {
        if (hdr.seq == V2_STATUS_SEQ && hdr.call_id == v2_prev_call_ &&
            v2_prev_small_ &&
            hdr.len == static_cast<uint32_t>(v2_prev_num_chunks_)) {
          // Peer is stuck in the previous call: queue retransmits.
          v2_prev_want_.assign(v2_prev_num_chunks_, 0);
          for (int k = 0; k < v2_prev_num_chunks_; k++) {
            v2_prev_want_[k] =
                static_cast<uint8_t>(p[V2_HDR + k]) ? 0 : 1;
          }
          std::fprintf(
              stderr,
              "[jaccl-v2] rank=%d call=%u serving retransmit for prev "
              "call=%u\n",
              rank_, call_id, hdr.call_id);
          std::fflush(stderr);
        }
        // else: stale duplicate data — drop silently.
      } else {
        std::fprintf(
            stderr,
            "[jaccl-v2] PROTOCOL rank=%d call=%u got header call=%u seq=%u "
            "(skew > 1)\n",
            rank_, call_id, hdr.call_id, hdr.seq);
        std::fflush(stderr);
        throw std::runtime_error(
            "[jaccl] reliable v2 protocol violation (skew > 1) — clean "
            "re-place");
      }
      // Re-arm the pool slot (zero first so a dead DMA reads as empty).
      zero_recv_buffer(rb);
      pool_connections_[peer].post_recv(
          rb, make_wr_id(0, POOL_RECV_WR, buff, peer));
      return progress;
    };

    auto my_slots_clear = [&]() {
      for (int s = base; s < base + half; s++) {
        if (v2_send_outstanding_[s] != 0) {
          return false;
        }
      }
      return true;
    };

    // Top-up first-pass sends into free parity slots. Runs every loop pass
    // (NOT only on same-call CQEs) so busy slots from a previous call's
    // retransmit sends cannot starve this call's pipeline.
    auto top_up_sends = [&]() {
      while (chunks_posted < num_chunks) {
        int slot = base + (chunks_posted % data_slots);
        if (v2_send_outstanding_[slot] != 0) {
          break;
        }
        post_chunk(chunks_posted, slot);
        chunks_posted++;
      }
    };
    top_up_sends();

    int round = 0; // TCP-barrier rounds (large path only)
    uint64_t last_progress = mach_absolute_time();
    while (true) {
      if (mach_ticks_to_us(mach_absolute_time() - _t0) > _deadline_us) {
        std::fprintf(
            stderr,
            "[jaccl-v2] DEADLINE rank=%d call_id=%u all_recv=%d/%d "
            "chunks_posted=%d small=%d peer_in_call=%d\n",
            rank_, call_id, all_recv, num_chunks, chunks_posted,
            small ? 1 : 0, peer_in_call ? 1 : 0);
        std::fflush(stderr);
        throw std::runtime_error(
            "[jaccl] reliable_all_reduce_v2 deadline — clean re-place");
      }

      top_up_sends();
      // Retransmits owed to THIS call's peer (status-driven, both paths).
      if (have_peer_status) {
        bool all_served = true;
        for (int k = 0; k < num_chunks; k++) {
          if (!peer_want[k]) {
            continue;
          }
          int slot = base + (k % data_slots);
          if (v2_send_outstanding_[slot] == 0) {
            post_chunk(static_cast<uint32_t>(k), slot);
            peer_want[k] = 0;
          } else {
            all_served = false;
          }
        }
        if (all_served) {
          have_peer_status = false;
        }
      }
      service_prev();

      // Exit checks.
      bool data_done = (all_recv >= num_chunks) &&
          (chunks_posted >= num_chunks);
      if (data_done && my_slots_clear()) {
        if (small) {
          break; // optimistic exit — no rendezvous
        }
        // Large: TCP rendezvous (barrier is amortized over many chunks and
        // is the only sound reconciliation without retained buffers).
        auto peer_got = coordinator_->reliable_barrier(
            call_id, static_cast<uint32_t>(round), got);
        round++;
        bool peer_has_all = std::count(
            peer_got.begin(), peer_got.end(), 1) == num_chunks;
        if (peer_has_all) {
          break;
        }
        for (int k = 0; k < num_chunks; k++) {
          if (!peer_got[k]) {
            int slot = base + (k % half);
            // Serve immediately when free; else next loop pass (peer_want).
            if (v2_send_outstanding_[slot] == 0) {
              post_chunk(static_cast<uint32_t>(k), slot);
            } else {
              if (peer_want.empty()) {
                peer_want.assign(num_chunks, 0);
              }
              peer_want[k] = 1;
              have_peer_status = true;
            }
          }
        }
        last_progress = mach_absolute_time();
        continue;
      }

      ibv_wc wc[16];
      int n = poll(pool_connections_, 16, wc);
      bool progressed = false;
      for (int i = 0; i < n; i++) {
        int wt = wr_id_work_type(wc[i].wr_id);
        int wb = wr_id_buff(wc[i].wr_id);
        if (wc[i].status != IBV_WC_SUCCESS) {
          // A dropped/erred completion must not leak its WR slot: log it,
          // and re-arm pool recvs / free send slots so retransmit recovers.
          std::fprintf(
              stderr,
              "[jaccl-v2] WC_ERR rank=%d call=%u status=%d wt=%d buff=%d\n",
              rank_, call_id, static_cast<int>(wc[i].status), wt, wb);
          std::fflush(stderr);
          if (wt == POOL_RECV_WR && wb >= 0 && wb < NUM_BUFFERS) {
            auto& rb = recv_buffer(v2_pool_sz_, wb, peer);
            zero_recv_buffer(rb);
            pool_connections_[peer].post_recv(
                rb, make_wr_id(0, POOL_RECV_WR, wb, peer));
          } else if (wt == SEND_WR && wb >= 0 && wb < NUM_BUFFERS &&
                     v2_send_outstanding_[wb] > 0) {
            v2_send_outstanding_[wb]--;
          }
          continue;
        }
        if (wt == POOL_RECV_WR) {
          progressed |= consume_pool(wb);
        } else if (wt == SEND_WR) {
          if (wb >= 0 && wb < NUM_BUFFERS && v2_send_outstanding_[wb] > 0) {
            v2_send_outstanding_[wb]--;
          }
          progressed = true;
        }
        // Other completion types (legacy RECV_WR/ACK from a mode switch)
        // are ignored; v2 is all-or-nothing per process.
      }
      if (progressed) {
        last_progress = mach_absolute_time();
        continue;
      }
      if (n == 0) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(jaccl_reliable_idle_us()));
      }
      if (quiet_us != 0 &&
          mach_ticks_to_us(mach_absolute_time() - last_progress) >
              quiet_us) {
        last_progress = mach_absolute_time();
        if (small) {
          // Stuck: tell the peer what I have (idempotent; resent each
          // quiet period until the data flows).
          if (v2_send_outstanding_[status_slot] == 0) {
            post_status();
          }
        } else if (all_recv < num_chunks && peer_in_call) {
          // Large path quiet with peer provably in this call: barrier to
          // exchange bitmasks and trigger retransmits (legacy semantics).
          auto peer_got = coordinator_->reliable_barrier(
              call_id, static_cast<uint32_t>(round), got);
          round++;
          for (int k = 0; k < num_chunks; k++) {
            if (!peer_got[k]) {
              if (peer_want.empty()) {
                peer_want.assign(num_chunks, 0);
              }
              peer_want[k] = 1;
              have_peer_status = true;
            }
          }
        }
        if (round > std::max(8, jaccl_ack_retransmit_max())) {
          throw std::runtime_error(
              "[jaccl] reliable v2 exceeded max retransmit rounds — clean "
              "re-place");
        }
      }
    }

    // Retain retransmit-service info for the next call's loop.
    v2_prev_call_ = call_id;
    v2_prev_num_chunks_ = num_chunks;
    v2_prev_sz_ = sz;
    v2_prev_small_ = small;

    reduce_op(reinterpret_cast<T*>(asm_buf.data()), out_ptr, size);
    if (_log) {
      std::fprintf(
          stderr, "[jaccl-v2] EXIT rank=%d call_id=%u rounds=%d\n",
          rank_, call_id, round);
      std::fflush(stderr);
    }
  }

  // Reliable data-phase all_reduce over UC (see jaccl_reliable_data_enabled).
  // Chunks carry a 4-byte sequence header; the receiver assembles each peer's
  // FULL message keyed by sequence (duplicates overwrite -> idempotent) and the
  // reduction is deferred to the end and applied ONCE. A dropped chunk can't
  // wedge: after a bounded drain, ranks exchange received-bitmasks over the
  // reliable coordinator and retransmit exactly the missing chunks, looping
  // until every chunk is in. Written for the 2-rank TP case (num_peers==1);
  // larger meshes fall back to the UC path.
  template <typename T, typename ReduceOp>
  void reliable_all_reduce(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }
    if (size_ <= 1 || size == 0) {
      return;
    }
    // Dispatch guarantees size_ == 2 (2-rank TP). Defensive guard otherwise.
    if (size_ != 2) {
      throw std::runtime_error(
          "[jaccl] reliable_all_reduce only supports 2 ranks");
    }
    const int peer = (rank_ == 0) ? 1 : 0;
    const int64_t total_bytes = size * static_cast<int64_t>(sizeof(T));
    auto [sz, buffer_size] = buffer_size_from_message(total_bytes);
    // Cap chunk size to the size class that reliably completes on librdma.
    if (sz > jaccl_reliable_max_sz()) {
      sz = jaccl_reliable_max_sz();
      buffer_size = static_cast<int64_t>(FRAME_SIZE) * (1 << sz);
    }
    const int HDR = static_cast<int>(sizeof(uint32_t));
    int64_t chunk_bytes = static_cast<int64_t>(buffer_size) - HDR;
    chunk_bytes -= chunk_bytes % static_cast<int64_t>(sizeof(T));
    const int num_chunks =
        static_cast<int>((total_bytes + chunk_bytes - 1) / chunk_bytes);
    // Pipeline depth (sends + recvs kept in flight). Large UC sends can't
    // overlap (a 2nd concurrent >=64KB send returns ENOMEM), but small sends
    // (<=16KB, sz<=2) overlap cleanly — measured 2026-07-05 during the chunk-
    // size bisection. Stop-and-wait at sz=2 capped the reliable path at
    // ~105MB/s on an 80Gbps TB5 link (completion latency per 16KB chunk),
    // which bounded long-context prefill at ~150 tok/s with the GPU 25% idle.
    // Pipeline up to NUM_BUFFERS for every clean size class; depth 1 only for
    // capped-but-still-large classes (sz>=3), out of caution.
    const int SEND_INFLIGHT = (sz <= 2)
        ? std::max(1, std::min(jaccl_reliable_inflight(), NUM_BUFFERS))
        : 1;

    std::vector<char> asm_buf(total_bytes, 0); // peer's full message, by seq
    std::vector<uint8_t> got(num_chunks, 0); // chunks received from peer

    // Total-time deadline: converts any silent hang inside this collective into
    // a clean, LOGGED throw (< the 20s Event::wait / 45s _check_hang) so we see
    // exactly which phase/state is stuck.
    const uint64_t _t0 = mach_absolute_time();
    const uint64_t _deadline_us = 15000000;

    static std::atomic<int> _rd_calls{0};
    int _rd_n = _rd_calls.fetch_add(1);
    if (_rd_n < 8 || jaccl_progress_enabled()) {
      std::fprintf(
          stderr,
          "[jaccl-reliable] ENTER rank=%d call_id=%u size=%lld total_bytes=%lld "
          "sz=%d buffer_size=%lld chunk_bytes=%lld num_chunks=%d\n",
          rank_, call_id, (long long)size, (long long)total_bytes, sz,
          (long long)buffer_size, (long long)chunk_bytes, num_chunks);
      std::fflush(stderr);
    }
    // Fill send_buffer(buff) with chunk c of out_ptr (header + data) and post.
    auto post_chunk = [&](int c, int buff) {
      auto& sb = send_buffer(sz, buff);
      char* p = sb.data<char>();
      uint32_t hdr = static_cast<uint32_t>(c);
      std::memcpy(p, &hdr, HDR);
      int64_t off = static_cast<int64_t>(c) * chunk_bytes;
      int64_t len = std::min(chunk_bytes, total_bytes - off);
      std::memcpy(
          p + HDR, reinterpret_cast<const char*>(out_ptr) + off,
          static_cast<size_t>(len));
      if (len < chunk_bytes) {
        std::memset(
            p + HDR + len, 0, static_cast<size_t>(chunk_bytes - len));
      }
      JACCL_DMA_BARRIER();
      try {
        connections_[peer].post_send(
            sb, make_wr_id(call_id, SEND_WR, buff, peer));
      } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "[jaccl-reliable] post_send FAILED rank=%d call_id=%u c=%d buff=%d "
            "num_chunks=%d buffer_size=%lld: %s\n",
            rank_, call_id, c, buff, num_chunks, (long long)buffer_size,
            e.what());
        std::fflush(stderr);
        throw;
      }
    };
    auto post_recv_buff = [&](int buff) {
      auto& rb = recv_buffer(sz, buff, peer);
      zero_recv_buffer(rb);
      connections_[peer].post_recv(rb, make_wr_id(call_id, RECV_WR, buff, peer));
    };
    static std::atomic<int> _recv_log{0};
    // Consume a RECV completion: read seq header, assemble if new.
    auto consume_recv = [&](int buff) {
      JACCL_DMA_BARRIER();
      auto& rb = recv_buffer(sz, buff, peer);
      const char* p = rb.data<char>();
      uint32_t c;
      std::memcpy(&c, p, HDR);
      if (_recv_log.fetch_add(1) < 40 || jaccl_progress_enabled()) {
        std::fprintf(
            stderr,
            "[jaccl-reliable] RECV rank=%d call_id=%u hdr_seq=%u num_chunks=%d "
            "accept=%d\n",
            rank_, call_id, c, num_chunks,
            (c < static_cast<uint32_t>(num_chunks) && !got[c]) ? 1 : 0);
        std::fflush(stderr);
      }
      if (c < static_cast<uint32_t>(num_chunks) && !got[c]) {
        int64_t off = static_cast<int64_t>(c) * chunk_bytes;
        int64_t len = std::min(chunk_bytes, total_bytes - off);
        std::memcpy(asm_buf.data() + off, p + HDR, static_cast<size_t>(len));
        got[c] = 1;
      }
    };

    int all_recv = static_cast<int>(std::count(got.begin(), got.end(), 1));
    int next_send = 0; // first chunk index to (re)send this round
    std::vector<uint8_t> to_resend(num_chunks, 0); // MY chunks the peer needs

    // Sliding-window recv: keep up to min(RECV_INFLIGHT, num_chunks - all_recv)
    // recvs posted. Capping by remaining chunks keeps the invariant
    // posted_recvs <= num_chunks - all_recv, so ZERO recv WRs remain posted at
    // completion (leftover recvs would grab the NEXT collective's sends -> stale
    // call_id -> data lost -> churn/hang), while still pipelining depth
    // RECV_INFLIGHT to hide RTT. Buffers 0..RECV_INFLIGHT-1.
    const int RECV_INFLIGHT = SEND_INFLIGHT;
    int posted_recvs = 0;
    for (int b = 0; b < std::min(RECV_INFLIGHT, num_chunks); b++) {
      post_recv_buff(b);
      posted_recvs++;
    }

    // Round-based reliable exchange.
    const uint64_t drain_quiet_us = jaccl_ack_retransmit_us(); // reuse knob
    const int max_rounds = std::max(8, jaccl_ack_retransmit_max()); // safety net
    for (int round = 0;; round++) {
      if (round > max_rounds) {
        throw std::runtime_error(
            "[jaccl] reliable_all_reduce exceeded max retransmit rounds "
            "(link persistently dropping) — throwing for clean re-place");
      }
      const int send_from = (round == 0) ? 0 : next_send;
      int outstanding_sends = 0;
      int c = send_from;
      // Prime SEND_INFLIGHT sends. Apple librdma rejects a 2nd concurrent large
      // send on a UC QP with ENOMEM(-12) regardless of max_send_wr, so keep only
      // one send outstanding (2 recvs + 1 send is confirmed OK). Perf later.
      for (int buff = 0; c < num_chunks && buff < SEND_INFLIGHT; c++) {
        if (round == 0 || to_resend[c]) {
          post_chunk(c, buff);
          buff++;
          outstanding_sends++;
        }
      }
      // Drain: process completions until either done, or no FORWARD PROGRESS
      // (all_recv + c, both monotonic this round) for drain_quiet_us. Progress-
      // based (not completion-based): a flood of duplicate/straggler completions
      // that don't advance anything still lets us fall through to the barrier,
      // where retransmit reconciles — instead of spinning forever.
      uint64_t last_progress = mach_absolute_time();
      int prev_progress = all_recv + c;
      while (true) {
        if (mach_ticks_to_us(mach_absolute_time() - _t0) > _deadline_us) {
          std::fprintf(
              stderr,
              "[jaccl-reliable] DEADLINE rank=%d call_id=%u round=%d all_recv=%d "
              "c=%d num_chunks=%d outstanding_sends=%d got_sum=%d phase=drain\n",
              rank_, call_id, round, all_recv, c, num_chunks, outstanding_sends,
              static_cast<int>(std::count(got.begin(), got.end(), 1)));
          std::fflush(stderr);
          throw std::runtime_error(
              "[jaccl] reliable_all_reduce deadline in drain — clean re-place");
        }
        if (all_recv >= num_chunks && outstanding_sends == 0 &&
            c >= num_chunks) {
          break; // nothing left to do this round
        }
        int cur_progress = all_recv + c;
        if (cur_progress != prev_progress) {
          prev_progress = cur_progress;
          last_progress = mach_absolute_time();
        } else if (
            mach_ticks_to_us(mach_absolute_time() - last_progress) >
            drain_quiet_us) {
          break; // stalled this round -> barrier + retransmit
        }
        ibv_wc wc[16];
        int n = poll(connections_, 16, wc);
        for (int i = 0; i < n; i++) {
          if (wr_id_call_id(wc[i].wr_id) != call_id) {
            continue; // stale
          }
          if (wc[i].status != IBV_WC_SUCCESS) {
            continue; // dropped/erred completion — barrier+retransmit recovers
          }
          int wt = wr_id_work_type(wc[i].wr_id);
          int wb = wr_id_buff(wc[i].wr_id);
          if (wt == RECV_WR) {
            consume_recv(wb);
            all_recv = static_cast<int>(std::count(got.begin(), got.end(), 1));
            posted_recvs--;
            // Sliding window: re-post only while posted stays <= remaining
            // chunks -> zero leftover recvs once everything is received.
            if (posted_recvs <
                std::min(RECV_INFLIGHT, num_chunks - all_recv)) {
              post_recv_buff(wb);
              posted_recvs++;
            }
          } else if (wt == SEND_WR) {
            outstanding_sends--;
            // Advance the send pipeline in this buffer.
            while (c < num_chunks && !(round == 0 || to_resend[c])) {
              c++;
            }
            if (c < num_chunks) {
              post_chunk(c, wb);
              c++;
              outstanding_sends++;
            }
          }
        }
        if (n == 0) {
          // Idle poll: yield the core so two concurrent comm-worker drains don't
          // 100%-spin and starve the Metal/GPU submission threads under
          // sustained c>=2 load (which parks the peer's main thread in an
          // uninterruptible GPU wait -> _check_hang). Tunable.
          std::this_thread::sleep_for(
              std::chrono::microseconds(jaccl_reliable_idle_us()));
        }
      }
      // Reliable barrier: exchange "chunks received from peer" bitmasks.
      if (round > 0 || jaccl_progress_enabled()) {
        std::fprintf(
            stderr,
            "[jaccl-reliable] BARRIER rank=%d call_id=%u round=%d all_recv=%d "
            "num_chunks=%d\n",
            rank_, call_id, round, all_recv, num_chunks);
        std::fflush(stderr);
      }
      auto peer_got = coordinator_->reliable_barrier(
          call_id, static_cast<uint32_t>(round), got);
      bool i_have_all = std::count(got.begin(), got.end(), 1) == num_chunks;
      bool peer_has_all =
          std::count(peer_got.begin(), peer_got.end(), 1) == num_chunks;
      if (i_have_all && peer_has_all) {
        break;
      }
      // Prepare retransmit set: MY chunks the peer is missing.
      to_resend.assign(num_chunks, 0);
      next_send = num_chunks;
      for (int k = 0; k < num_chunks; k++) {
        if (!peer_got[k]) {
          to_resend[k] = 1;
          if (k < next_send) {
            next_send = k;
          }
        }
      }
    }

    // All chunks present on both ranks: reduce peer's message into out ONCE.
    reduce_op(reinterpret_cast<T*>(asm_buf.data()), out_ptr, size);
  }

  // ── PP-mode warmup all_reduce over the dedicated ACK QP ──
  //
  // WHY (2026-08-10, follow-up to the max_qp=3 mode-gating fix): in PP mode
  // pool_connections_ is empty by design, so all_reduce()'s dispatch below
  // fell through to reliable_all_reduce (non-v2), which posts its sends and
  // recvs on connections_[peer]. That is the SAME physical QP that PP's raw
  // p2p pipeline traffic uses -- send() posts connections_[dst].post_send()
  // and recv() posts connections_[src].post_recv() (see ~line 1780 / ~line
  // 2040). PP's ONE warmup collective (exo's
  // exchange_prefill_peer_layer_count / handshake_metaframe_protocol) therefore
  // interleaved with the MetaFrame header traffic that starts immediately
  // after warmup, and the collective's float payload landed in a MetaFrame
  // header recv buffer -- observed deterministically (20/20) on the real
  // 2-node cluster as "MetaFrame protocol version mismatch: received 16256"
  // (0x3F80 == the high half of IEEE-754 1.0f). Exactly the two-protocols-
  // on-one-QP bug class this file already paid to learn with ack/pool/
  // p2p_retry.
  //
  // FIX: run this tiny collective on ack_connections_ -- PP's third QP, which
  // is built but otherwise carries no traffic in PP mode -- reusing the
  // ALREADY-PRE-POSTED 64-slot ACK_RECV pool that post_ack_recvs(0) arms in
  // MeshGroup's ctor. Crucially this posts NO new recv WRs of its own: a
  // freshly posted recv would sit BEHIND those 64, so the peer's send would
  // consume a pre-existing slot and this rank would read the wrong thing --
  // the same corruption in a new shape. It sends via ack_connections_ with
  // the standard ACK_SEND_WR tagging and consumes completions with a private
  // copy of drain_acks()'s polling loop (see drain_acks_exchange below).
  //
  // Constraints: 2 ranks, and the whole message must fit one FRAME_SIZE
  // (4096B) ack buffer. Both hold for every collective PP actually issues
  // (a world_size-length int32 vector; a single int64). Anything larger, or
  // any other topology, still falls through to reliable_all_reduce.
  template <typename T, typename ReduceOp>
  bool ack_all_reduce_small(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
    if (size_ != 2 || ack_connections_.empty()) {
      return false;
    }
    const int64_t total_bytes = size * static_cast<int64_t>(sizeof(T));
    const int peer = (rank_ == 0) ? 1 : 0;
    if (total_bytes <= 0 ||
        total_bytes > static_cast<int64_t>(ack_send_buffers_[peer].size()) ||
        total_bytes > static_cast<int64_t>(ack_recv_buffers_[peer].size())) {
      return false;
    }
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, static_cast<size_t>(total_bytes));
    }
    // Stage OUR contribution into the ack send buffer. post_send transmits
    // the whole FRAME_SIZE buffer (SharedBuffer's SGE is the full buffer), so
    // zero the tail: the peer reads only the first total_bytes, but leaving
    // stale bytes behind would be gratuitous.
    auto& sbuf = ack_send_buffers_[peer];
    std::memset(sbuf.data<char>(), 0, sbuf.size());
    std::memcpy(sbuf.data<char>(), in_ptr, static_cast<size_t>(total_bytes));
    JACCL_DMA_BARRIER();
    // NOTE: this leaves our payload resident in ack_send_buffers_[peer] for
    // subsequent ack_sync_pre/ack_sync_post sends to retransmit. That is
    // harmless -- drain_acks never reads ACK payload bytes, the ack exchange
    // is purely a completion-count rendezvous.
    ack_connections_[peer].post_send(
        sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
    std::vector<char> peer_bytes(static_cast<size_t>(total_bytes), 0);
    drain_acks_exchange(
        call_id, /*num_peers=*/1, peer_bytes.data(), total_bytes);
    reduce_op(reinterpret_cast<T*>(peer_bytes.data()), out_ptr, size);
    return true;
  }

  // Private variant of drain_acks() for ack_all_reduce_small: identical
  // polling/replenish/caching structure, with ONE addition -- on each
  // ACK_RECV completion it copies the landed payload out of
  // ack_recv_buffers_[peer] BEFORE the replenish path memsets that same
  // buffer and re-posts it. Forked rather than adding an out-parameter to
  // drain_acks() because drain_acks() is on the hot path of every
  // reliable_all_reduce_v2 collective in TP mode (call sites: ack_sync_pre
  // ~line 2675 and ack_sync_post ~line 2727, themselves called from
  // all_reduce's UC path ~1321/1532 and all_gather's ~1568/1650) and its
  // read-then-replenish ordering constraint does not exist for any other
  // caller. No shared bookkeeping semantics change: cached_ack_recvs_ is
  // pushed/consumed exactly as drain_acks does, and the pre-posted pool is
  // replenished one-for-one per consumed ACK_RECV, so later ack_sync_pre/
  // ack_sync_post callers in this process see an unchanged QP state.
  void drain_acks_exchange(
      uint32_t call_id,
      int num_peers,
      char* out_bytes,
      int64_t out_len) {
    int need_send = num_peers;
    int need_recv = num_peers;
    StallWatch _stall(need_send + need_recv);
    // Deliberately does NOT consume cached_ack_recvs_: a cached entry is an
    // ACK_RECV whose payload was already discarded (drain_acks zeroes the
    // buffer on replenish), so it can never be this exchange's message. This
    // exchange is a strictly paired rendezvous (each rank posts exactly one
    // send), and it runs at warmup before any other ack traffic exists, so
    // the cache is empty in practice; leaving it untouched keeps it correct
    // for the ordinary ack callers if it ever isn't.
    while (need_send > 0 || need_recv > 0) {
      _stall.tick(need_send + need_recv, "drain_acks_exchange", rank_, call_id);
      ibv_wc wc[16];
      int n = poll(ack_connections_, 16, wc);
      for (int i = 0; i < n; i++) {
        int wt = wr_id_work_type(wc[i].wr_id);
        if (wt == ACK_RECV_WR) {
          if (wc[i].status != IBV_WC_SUCCESS) {
            std::ostringstream msg;
            msg << "[jaccl] ack exchange (recv) wc.status=" << wc[i].status
                << " wr_id=0x" << std::hex << wc[i].wr_id;
            throw std::runtime_error(msg.str());
          }
          int peer = wr_id_peer(wc[i].wr_id);
          auto& rbuf = ack_recv_buffers_[peer];
          if (need_recv > 0) {
            // READ BEFORE REPLENISH -- the replenish below memsets this very
            // buffer. This ordering is the whole reason for the fork.
            std::memcpy(
                out_bytes, rbuf.data<char>(), static_cast<size_t>(out_len));
          }
          // Replenish the pre-posted pool exactly as drain_acks does
          // (sentinel call_id=0 -- ACK_RECVs are call_id-agnostic).
          std::memset(rbuf.data<char>(), 0, rbuf.size());
          JACCL_DMA_BARRIER();
          ack_connections_[peer].post_recv(
              rbuf, make_wr_id(0, ACK_RECV_WR, 0, peer));
          if (need_recv > 0) {
            need_recv--;
          } else {
            cached_ack_recvs_.push_back(peer);
          }
        } else if (wt == ACK_SEND_WR) {
          if (wr_id_call_id(wc[i].wr_id) != call_id) {
            continue;
          }
          if (wc[i].status != IBV_WC_SUCCESS) {
            std::ostringstream msg;
            msg << "[jaccl] ack exchange (send) wc.status=" << wc[i].status
                << " wr_id=0x" << std::hex << wc[i].wr_id;
            throw std::runtime_error(msg.str());
          }
          need_send--;
        } else {
          continue;
        }
      }
    }
  }

  template <typename T, typename ReduceOp>
  void all_reduce(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
    // Reliable ARQ data path (gated). Top-level 2-rank group only.
    if (coordinator_ != nullptr && size_ == 2 && jaccl_reliable_data_enabled()) {
      // QP-BUDGET GATE (2026-08-10): v2 (optimistic) additionally requires the
      // dedicated pool QP, which PP mode does NOT allocate -- the Thunderbolt
      // HCA caps out at max_qp=3, so PP spends its third QP on
      // p2p_retry_connections_ instead (see mesh.cpp's
      // jaccl_pipeline_mode_enabled() comment). Selecting v2 purely on the env
      // toggle would enter reliable_all_reduce_v2 with an empty
      // pool_connections_ and hit its "should be unreachable" throw on PP's
      // warmup layer-count all_sum. The non-v2 reliable_all_reduce path is
      // pool-free -- it posts exclusively on connections_ (the data QP) and
      // rendezvouses over the TCP coordinator_ -- so it is the correct
      // fallback here, preserving full ARQ reliability without a 4th QP.
      if (jaccl_reliable_optimistic_enabled() && !pool_connections_.empty()) {
        reliable_all_reduce_v2<T>(call_id, in_ptr, out_ptr, size, reduce_op);
      } else if (
          pool_connections_.empty() && !ack_connections_.empty() &&
          ack_all_reduce_small<T>(call_id, in_ptr, out_ptr, size, reduce_op)) {
        // PP mode (empty pool, ack QP present) and the message fits one
        // FRAME_SIZE ack buffer: run it on the otherwise-idle ACK QP rather
        // than on connections_, which PP's raw send()/recv() pipeline traffic
        // owns. See ack_all_reduce_small's comment for the confirmed
        // corruption this avoids. Returns false (falling through below) for
        // anything it can't service, so reliable_all_reduce stays reachable.
      } else {
        reliable_all_reduce<T>(call_id, in_ptr, out_ptr, size, reduce_op);
      }
      return;
    }
    bool _prog = jaccl_progress_enabled();
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] all_reduce ENTER rank=%d call_id=%u size=%lld T_bytes=%zu\n",
          rank_,
          call_id,
          (long long)size,
          sizeof(T));
      std::fflush(stderr);
    }
    // If not inplace all reduce then copy the input to the output first
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }

    // Fully connected all reduce
    T* data = out_ptr;
    auto [sz, buffer_size] = buffer_size_from_message(size * sizeof(T));
    int64_t N = buffer_size / sizeof(T);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(size);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int completed_recv_begin[MESH_MAX_PEERS] = {0};
    int completed_recv_end[MESH_MAX_PEERS] = {0};

    int buff = 0;
    if (coordinator_ != nullptr && jaccl_confirmed_barrier_pre()) {
      // Reliable + ORDERED start barrier (replaces the UC ack_sync_pre for this
      // call). The UC ack barrier wedges on a lost completion; a plain TCP
      // barrier is reliable but corrupts because a data SEND can arrive before
      // the peer posts its matching data RECV (UC drop -> wrong data). Fix:
      // post ALL prefill recvs and fill the send buffers, THEN rendezvous over
      // the reliable TCP coordinator (so BOTH ranks provably have their recvs
      // posted), THEN post the sends. No send can land early -> no data-phase
      // UC drop, no wedge, correct data.
      int first = buff;
      while (read_offset < total && buff < PIPELINE) {
        post_recv_all(call_id, sz, buff);
        std::copy(
            data + read_offset,
            data + std::min(read_offset + N, total),
            send_buffer(sz, buff).begin<T>());
        buff++;
        in_flight += 2 * num_peers;
        read_offset += N;
      }
      confirmed_coord_barrier(call_id, "pre");
      for (int b = first; b < buff; b++) {
        post_send_all(call_id, sz, b);
      }
    } else {
      // Start-of-lambda cross-rank barrier on the dedicated ACK QP.
      // Confirms peer has entered THIS call before we post our first
      // data send. The pre-posted ACK_RECV pool (post_ack_recvs) and
      // sentinel-call_id replenish path in drain_acks keep the ACK QP
      // recv queue full across lambdas.
      if (jaccl_ack_sync_pre_enabled()) {
        ack_sync_pre(call_id);
      }

      // Prefill the pipeline
      while (read_offset < total && buff < PIPELINE) {
        post_recv_all(call_id, sz, buff);
        std::copy(
            data + read_offset,
            data + std::min(read_offset + N, total),
            send_buffer(sz, buff).begin<T>());
        post_send_all(call_id, sz, buff);

        buff++;
        in_flight += 2 * num_peers;
        read_offset += N;
      }
    }

    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] all_reduce PREFILL_DONE rank=%d call_id=%u in_flight=%d N=%lld total=%lld\n",
          rank_,
          call_id,
          in_flight,
          (long long)N,
          (long long)total);
      std::fflush(stderr);
    }

    // Main loop: keep going until we have no data in flight.
    int _poll_iters = 0;
    // Instrumentation locals — zero-cost when JACCL_POLL_INSTRUMENT off.
    bool _instr = jaccl_poll_instrument_enabled();
    uint64_t _instr_t0 = _instr ? mach_absolute_time() : 0;
    uint64_t _instr_total_in_poll_ticks = 0;
    uint64_t _instr_max_single_poll_ticks = 0;
    uint64_t _instr_iters_with_cqes = 0;
    StallWatch _stall(in_flight);
    while (in_flight > 0) {
      _stall.tick(in_flight, "all_reduce", rank_, call_id);
      ++_poll_iters;
      if (_prog) {
        if (_poll_iters <= 4 || (_poll_iters % 1000000) == 0) {
          std::fprintf(
              stderr,
              "[jaccl-prog] all_reduce POLL rank=%d call_id=%u iter=%d in_flight=%d\n",
              rank_,
              call_id,
              _poll_iters,
              in_flight);
          std::fflush(stderr);
        }
      }
      ibv_wc wc[WC_NUM];
      uint64_t _instr_poll_start = _instr ? mach_absolute_time() : 0;
      int n = poll(connections_, WC_NUM, wc);
      if (_instr) {
        uint64_t _dt = mach_absolute_time() - _instr_poll_start;
        _instr_total_in_poll_ticks += _dt;
        if (_dt > _instr_max_single_poll_ticks) _instr_max_single_poll_ticks = _dt;
        if (n > 0) ++_instr_iters_with_cqes;
      }
      for (int i = 0; i < n; i++) {
        // exo-jaccl-fix (2026-07-01): fault-injection hook for validating the
        // scheduler exception-propagation fix. When JACCL_INJECT_WC_ERROR is
        // set to a positive integer K, the K-th all_reduce completion polled
        // across the process is forced to look like a non-success RDMA work
        // completion (wc.status != IBV_WC_SUCCESS). This reproduces the
        // ``[jaccl] all_reduce wc.status=N`` transport fault ON DEMAND so we can
        // confirm it now surfaces as a catchable exception + clean instance
        // restart instead of std::terminate. Default OFF (env unset) → zero
        // cost beyond a single static getenv() and one counter increment.
        static const long _inject_at = [] {
          const char* v = std::getenv("JACCL_INJECT_WC_ERROR");
          return v ? std::atol(v) : 0L;
        }();
        if (_inject_at > 0) {
          static std::atomic<long> _wc_seen{0};
          if (_wc_seen.fetch_add(1) + 1 == _inject_at) {
            std::fprintf(
                stderr,
                "[jaccl] INJECTED wc error at completion #%ld (test hook)\n",
                _inject_at);
            std::fflush(stderr);
            throw std::runtime_error(
                "[jaccl] all_reduce wc.status=4 wr_id=0xINJECTED byte_len=0 "
                "(injected by JACCL_INJECT_WC_ERROR)");
          }
        }
        // Catch any non-success completion or RECV whose byte_len
        // doesn't match the buffer size we posted. UC silent-drop of a
        // foreign-collective send into our recv WR shows up here.
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::ostringstream msg;
          msg << "[jaccl] all_reduce wc.status=" << wc[i].status
              << " wr_id=0x" << std::hex << wc[i].wr_id
              << " byte_len=" << std::dec << wc[i].byte_len;
          throw std::runtime_error(msg.str());
        }
        if ((wr_id_work_type(wc[i].wr_id) == RECV_WR) &&
            wc[i].byte_len != static_cast<uint32_t>(buffer_size)) {
          std::ostringstream msg;
          msg << "[jaccl] all_reduce recv byte_len=" << wc[i].byte_len
              << " expected=" << buffer_size << " wr_id=0x" << std::hex
              << wc[i].wr_id;
          throw std::runtime_error(msg.str());
        }
        // Stale completion from a prior collective: ignore it. Do not
        // decrement in_flight; that buffer belongs to a call that has
        // already returned.
        if (wr_id_call_id(wc[i].wr_id) != call_id) {
          continue;
        }
        int work_type = wr_id_work_type(wc[i].wr_id);
        int buff = wr_id_buff(wc[i].wr_id);
        int rank = wr_id_peer(wc[i].wr_id);

        in_flight--;

        if (_prog) {
          std::fprintf(
              stderr,
              "[jaccl-prog] all_reduce CQE rank=%d call_id=%u type=%s peer=%d buff=%d in_flight=%d\n",
              rank_,
              call_id,
              work_type == SEND_WR ? "SEND" : "RECV",
              rank,
              buff,
              in_flight);
          std::fflush(stderr);
        }

        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                data + read_offset,
                data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<T>());
            post_send_all(call_id, sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        else if (work_type == RECV_WR) {
          // The NIC has DMA'd into recv_buffer; ensure those writes are
          // visible to the CPU before we read from the buffer below.
          JACCL_DMA_BARRIER();
          completed_recv_end[rank]++;
        }
      }

      // Process completed recvs.
      //
      // For each rank we have a range [begin, end) of completed chunks.
      // When we have an unprocessed recv AND the write location is behind
      // read_offset, reduce in-place and optionally post another recv.
      for (int r = 0; r < size_; r++) {
        int s = completed_recv_begin[r];
        int e = completed_recv_end[r];
        int w = s * N;
        while (w < read_offset && e - s > 0) {
          int buff = s % PIPELINE;
          reduce_op(
              recv_buffer(sz, buff, r).begin<T>(),
              data + w,
              std::min(N, total - w));
          w += N;
          s++;
          if (w + (PIPELINE - 1) * N < total) {
            recv_from(call_id, sz, r, buff);
            in_flight++;
          }
        }
        completed_recv_begin[r] = s;
      }
    }
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] all_reduce DATA_DONE rank=%d call_id=%u poll_iters=%d -> ack_sync_post\n",
          rank_,
          call_id,
          _poll_iters);
      std::fflush(stderr);
    }
    if (_instr) {
      uint64_t total_wall_us = mach_ticks_to_us(mach_absolute_time() - _instr_t0);
      if (total_wall_us > jaccl_poll_instrument_threshold_us()) {
        uint64_t in_poll_us = mach_ticks_to_us(_instr_total_in_poll_ticks);
        uint64_t max_poll_us = mach_ticks_to_us(_instr_max_single_poll_ticks);
        std::fprintf(
            stderr,
            "[jaccl-instr] all_reduce SLOW rank=%d call_id=%u total_wall_us=%llu "
            "iters=%d iters_with_cqes=%llu in_poll_us=%llu (=%llu%% of wall) "
            "max_single_poll_us=%llu\n",
            rank_,
            call_id,
            (unsigned long long)total_wall_us,
            _poll_iters,
            (unsigned long long)_instr_iters_with_cqes,
            (unsigned long long)in_poll_us,
            (unsigned long long)(total_wall_us > 0 ? (in_poll_us * 100ULL / total_wall_us) : 0),
            (unsigned long long)max_poll_us);
        std::fflush(stderr);
      }
    }
    ack_sync_post(call_id);
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] all_reduce DONE rank=%d call_id=%u\n",
          rank_,
          call_id);
      std::fflush(stderr);
    }
  }

  void all_gather(
      uint32_t call_id,
      const char* in_ptr,
      char* out_ptr,
      int64_t n_bytes) {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    // Fully connected all gather
    char* data = out_ptr;
    char* our_data = out_ptr + rank_ * n_bytes;
    auto [sz, N] = buffer_size_from_message(n_bytes);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(n_bytes);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int write_offset[MESH_MAX_PEERS] = {0};

    // Start-of-lambda cross-rank barrier. See ack_sync_pre doc above.
    if (jaccl_ack_sync_pre_enabled()) {
      ack_sync_pre(call_id);
    }

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(call_id, sz, buff);
      std::copy(
          our_data + read_offset,
          our_data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<char>());
      post_send_all(call_id, sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop: keep going until we have no data in flight.
    StallWatch _stall(in_flight);
    while (in_flight > 0) {
      _stall.tick(in_flight, "all_gather", rank_, call_id);
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::ostringstream msg;
          msg << "[jaccl] all_gather wc.status=" << wc[i].status
              << " wr_id=0x" << std::hex << wc[i].wr_id
              << " byte_len=" << std::dec << wc[i].byte_len;
          throw std::runtime_error(msg.str());
        }
        if ((wr_id_work_type(wc[i].wr_id) == RECV_WR) &&
            wc[i].byte_len != static_cast<uint32_t>(N)) {
          std::ostringstream msg;
          msg << "[jaccl] all_gather recv byte_len=" << wc[i].byte_len
              << " expected=" << N << " wr_id=0x" << std::hex
              << wc[i].wr_id;
          throw std::runtime_error(msg.str());
        }
        if (wr_id_call_id(wc[i].wr_id) != call_id) {
          continue;
        }
        int work_type = wr_id_work_type(wc[i].wr_id);
        int buff = wr_id_buff(wc[i].wr_id);
        int rank = wr_id_peer(wc[i].wr_id);

        in_flight--;

        // Send completed: send the next chunk if any.
        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                our_data + read_offset,
                our_data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<char>());
            post_send_all(call_id, sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        // Recv completed: copy to output and post another recv if needed.
        else if (work_type == RECV_WR) {
          // Ensure the NIC's DMA writes to recv_buffer are visible to CPU.
          JACCL_DMA_BARRIER();
          std::copy(
              recv_buffer(sz, buff, rank).begin<char>(),
              recv_buffer(sz, buff, rank).begin<char>() +
                  std::min(N, total - write_offset[rank]),
              data + rank * n_bytes + write_offset[rank]);
          write_offset[rank] += N;
          if (write_offset[rank] + N * (PIPELINE - 1) < total) {
            recv_from(call_id, sz, rank, buff);
            in_flight++;
          }
        }
      }
    }
    ack_sync_post(call_id);
  }

  void send(uint32_t call_id, const char* in_ptr, int64_t n_bytes, int dst) {
    // The retry protocol below (unlike the old pre/post barrier, which had
    // a UC-based fallback for subgroups) STRUCTURALLY requires the
    // dedicated p2p retry channel for its retransmit-negotiation barrier --
    // there is no UC-safe way to exchange a got-bitmask. send()/recv() are
    // only ever called on exo's PP top-level 2-rank group, which always
    // has a p2p_channel_ (side_channel_ is populated in that ctor; only
    // split() subgroups lack one, and PP never splits) -- fail loudly
    // instead of silently misbehaving if that invariant is ever violated.
    if (p2p_channel_ == nullptr) {
      throw std::runtime_error(
          "[jaccl] send() called with no dedicated p2p retry channel (p2p_channel_) -- the "
          "retry-on-drop protocol requires it; only top-level groups have "
          "one, and send()/recv() should only be called on a top-level "
          "group (exo's Pipeline-Parallel usage)");
    }
    // QP-BUDGET GATE (2026-08-10): the retry protocol runs over the dedicated
    // p2p_retry_connections_ QP, which is only allocated in PP mode -- TP
    // spends that third-and-final QP slot (max_qp=3 on the Thunderbolt HCA) on
    // pool_connections_ instead. See mesh.cpp's jaccl_pipeline_mode_enabled().
    // Reaching here with an empty vector means TP-mode code called the PP-only
    // p2p path; indexing the empty span below would be silent UB, so fail fast
    // with a message that names the actual misconfiguration.
    if (p2p_retry_connections_.empty()) {
      throw std::runtime_error(
          "[jaccl] send() called with no p2p_retry_connections_ (dedicated p2p "
          "retry QP) -- this QP is only allocated when "
          "MLX_JACCL_SHARDING_MODE=Pipeline. Either TP-mode code reached the "
          "PP-only raw send()/recv() path (TP should use all_reduce/all_gather/"
          "all_sum), or the runner was launched in PP without "
          "MLX_JACCL_SHARDING_MODE=Pipeline being set/propagated");
    }
    // Pre-transfer rendezvous REMOVED (design doc Section 37 Phase 1,
    // 2026-08-10): the old code's comment already noted this was "no
    // longer load-bearing for correctness -- the retry loop below
    // recovers from ANY drop, round 0 included -- just reduces retry
    // rounds needed." The NEW p2p_retry_exchange has no lockstep
    // rendezvous primitive to spend on a non-load-bearing optimization
    // (see class doc comment point 1) -- round-0 misses are now handled
    // exactly like any other round's misses, at the cost of possibly one
    // extra round on a cold start. Zero-byte transfers below similarly no
    // longer need ANY p2p_retry traffic: the old code used the barrier as
    // a generic paired-call-count rendezvous (every p2p_retry_barrier
    // call on one side had to be matched by exactly one on the other, to
    // keep the TCP stream framed correctly); the new exchange has no such
    // pairing requirement (each call is independently self-contained by
    // (data_src_rank, seq)), so a transfer with no data simply needs no
    // wire traffic at all.
    if (n_bytes == 0) {
      return;
    }

    auto [sz, buffer_size] = buffer_size_from_message(n_bytes);
    if (sz > jaccl_reliable_max_sz()) {
      sz = jaccl_reliable_max_sz();
      buffer_size = static_cast<int64_t>(FRAME_SIZE) * (1 << sz);
    }
    const int HDR = static_cast<int>(sizeof(uint32_t));
    const int64_t chunk_bytes = buffer_size - HDR;
    const int num_chunks =
        static_cast<int>((n_bytes + chunk_bytes - 1) / chunk_bytes);
    if (num_chunks > 0xFFFF) {
      // The seq/chunk-index packing below assumes chunk indices fit in
      // the header's lower 16 bits (see the seq-tagging comment on
      // send_seq_/recv_seq_) -- fail loudly rather than silently
      // truncate/collide if a message ever needs more chunks than that.
      throw std::runtime_error(
          "[jaccl] send(): num_chunks exceeds the 16-bit chunk-index "
          "field this header packing assumes -- message too large for "
          "the current buffer size class, or MLX_JACCL_RELIABLE_MAX_SZ "
          "needs raising");
    }
    // Real production incident fix (2026-08-08, see send_seq_/recv_seq_'s
    // own field comment for the full incident) -- one seq value per
    // logical send() call to this peer, packed into the header's upper
    // 16 bits alongside the existing chunk index in the lower 16.
    const uint32_t seq = static_cast<uint32_t>(send_seq_[dst]++);
    if (jaccl_progress_enabled()) {
      // Section 69: per-call counter trace. Section 68 refuted reconnect
      // as the origin of the constant off-by-one, so the divergence
      // happens during ordinary early traffic. Logging every increment
      // on both ranks lets the FIRST divergent call be identified
      // directly by diffing the two ranks' streams, instead of auditing
      // send/recv pairs by hand (which found every control-message pair
      // balanced).
      std::fprintf(
          stderr,
          "[jaccl-seq69] SEND rank=%d dst=%d seq=%u call_id=%u bytes=%lld\n",
          rank_, dst, seq, call_id, (long long)n_bytes);
      std::fflush(stderr);
    }
    const int SEND_INFLIGHT = (sz <= 2)
        ? std::max(1, std::min(jaccl_reliable_inflight(), NUM_BUFFERS))
        : 1;

    auto post_chunk = [&](int c, int buff) {
      auto& sb = send_buffer(sz, buff);
      char* p = sb.data<char>();
      uint32_t hdr = (seq << 16) | static_cast<uint32_t>(c & 0xFFFF);
      std::memcpy(p, &hdr, HDR);
      int64_t off = static_cast<int64_t>(c) * chunk_bytes;
      int64_t len = std::min(chunk_bytes, n_bytes - off);
      std::memcpy(p + HDR, in_ptr + off, static_cast<size_t>(len));
      JACCL_DMA_BARRIER();
      // DIAGNOSTIC (2026-08-11, Section 43 continued): trace every ACTUAL
      // ibv_post_send() call for this data QP, distinct from the ROUND
      // log (which only proves the outer retry loop is iterating, not
      // that post_chunk was ever invoked or that ibv_post_send didn't
      // throw). Answers the anomaly found via debug_dump_qp_state(): a
      // stalled call_id logged 600 ROUNDs with to_resend_count=1 each,
      // yet the QP's own sq_psn was only 7 at the moment of the stall --
      // far too low if post_send were actually firing every round. If
      // POST_CHUNK lines are missing/sparse relative to ROUND lines,
      // the retry loop's OWN control flow (not the NIC/QP) is silently
      // failing to re-invoke post_chunk on later rounds -- a Python-
      // adjacent-tier bug already found once this session (the `or`
      // short-circuit in pipeline_agree_cancel) may have a C++-side
      // sibling here.
      if (jaccl_progress_enabled()) {
        std::fprintf(
            stderr,
            "[jaccl-post] POST_CHUNK rank=%d call_id=%u dst=%d c=%d buff=%d\n",
            rank_,
            call_id,
            dst,
            c,
            buff);
        std::fflush(stderr);
      }
      connections_[dst].post_send(sb, make_wr_id(call_id, SEND_WR, buff, dst));
    };

    std::vector<uint8_t> to_resend; // empty on round 0 (send everything)
    int next_c = 0;

    const uint64_t _t0 = mach_absolute_time();
    const uint64_t _deadline_us = 15000000;
    // Section 71: p2p-specific quiet period. See
    // jaccl_p2p_drain_quiet_us()'s comment -- this loop's real round trip
    // is 69-150us, so the old shared 500ms timer was purely the cost of
    // noticing a dropped frame, at ~5000x the actual latency.
    const uint64_t drain_quiet_us = jaccl_p2p_drain_quiet_us();
    const int max_rounds = std::max(8, jaccl_ack_retransmit_max());

    bool _prog = jaccl_progress_enabled();
    // ROOT-CAUSE FIX (2026-08-09, design doc Section 39, direct user
    // correction of the earlier "extend the deadline" instinct): this
    // loop used to have an ADDITIONAL, unconditional fatal cap
    // (`max_rounds` below, `_deadline_us` inside the drain `while`)
    // layered ON TOP of its own retransmit protocol -- it threw
    // REGARDLESS of whether real progress was happening. Real-hardware
    // evidence (this session): a transfer legitimately ran 29
    // retransmit rounds over 15s, with `p2p_retry_barrier()`'s TCP
    // round-trip SUCCEEDING every single round (proving both ranks
    // alive and the control channel healthy) -- yet still threw
    // fatally at round 29, purely because of this arbitrary cap, not
    // because anything was actually broken. The barrier's own
    // successful completion IS the real liveness proof this protocol
    // needs; a cap layered on top of an already-succeeding liveness
    // check just manufactures failures out of legitimately slow (not
    // dead) transfers.
    //
    // User's direct correction: "the goal is that it's not blocked, or
    // if that can't be done for whatever reason that it's never a
    // fatal wait" -- stop treating "this is taking a long time" as
    // equivalent to "this is broken". The ONLY genuine liveness check
    // this protocol has is `p2p_retry_barrier()`'s own recv (bounded
    // at 300s by the Section 38 fix, throwing a REAL error only when
    // the peer is actually unreachable/dead, not merely slow) -- that
    // was believed to be the SOLE way this loop can end in failure at
    // the time this comment was written. CORRECTION (2026-08-10,
    // Section 43 continued): that belief was wrong -- p2p_retry_barrier/
    // p2p_retry_exchange's own StallWatch only guards the METADATA
    // barrier's liveness, not this transfer's actual data progress; see
    // the DATA-PROGRESS STALLWATCH comment a few lines below for the
    // real second backstop this loop now has. `max_rounds`/
    // `_deadline_us` are RETAINED as env-tunable knobs (still read
    // below, still logged via [jaccl-prog] MAX_ROUNDS_EXCEEDED/
    // DEADLINE_HIT when set) for diagnostics/opt-in strict-timeout
    // testing, but neither one THROWS anymore -- exceeding either is
    // now purely informational, logged once, and the loop continues
    // retrying exactly as before.
    //
    // DATA-PROGRESS STALLWATCH (2026-08-10, Section 43 continued -- fixes
    // the gap Section 39's fix above unknowingly depended on not existing).
    // Section 39's own reasoning explicitly claimed "the ONLY genuine
    // liveness check this protocol has is p2p_retry_exchange's own recv"
    // -- i.e. p2p_retry_exchange()'s internal StallWatch (see its call
    // site below and mesh_impl.h's StallWatch class). That assumption is
    // FALSE: p2p_retry_exchange()'s StallWatch metric is peer_frame_seen
    // popcount, which tracks liveness of the METADATA BARRIER RPC itself
    // (the got-bitmask exchange), not whether the actual DATA CHUNK this
    // send() is transferring is making progress. Confirmed on real
    // hardware: a call_id can have its barrier round-trip succeed every
    // ~500ms forever (round number incrementing, peer_frame_seen churning)
    // while peer_got_count stays PERMANENTLY 0/num_chunks -- the barrier
    // heartbeat never stops, so p2p_retry_exchange's StallWatch never
    // fires, so this loop retries the same dropped data chunk forever
    // with NO liveness backstop at all (grep -c STALLED == 0 on an 8+
    // minute real hung run). This is a genuinely different failure class
    // from what Section 39 fixed: Section 39 removed a fatal cap that
    // fired on legitimately SLOW-BUT-PROGRESSING transfers (round count/
    // wall-clock deadline, blind to whether new chunks were landing);
    // this new watch fires ONLY on genuine ZERO PROGRESS in the peer's
    // reported got-bitmask for the full timeout window, so it does not
    // reintroduce that false-positive -- a transfer that's merely slow
    // but still gaining acknowledged chunks every so often keeps resetting
    // this watch exactly like Section 39 intended.
    StallWatch _data_stall(-1); // -1 sentinel: primed with the real metric
                                 // on the first BARRIER report below
    bool _data_stall_primed = false;
    for (int round = 0;; round++) {
      if (_prog) {
        std::fprintf(
            stderr,
            "[jaccl-prog] send() ROUND rank=%d call_id=%u dst=%d round=%d "
            "num_chunks=%d to_resend_count=%d elapsed_us=%llu\n",
            rank_,
            call_id,
            dst,
            round,
            num_chunks,
            static_cast<int>(std::count(to_resend.begin(), to_resend.end(), 1)),
            (unsigned long long)mach_ticks_to_us(mach_absolute_time() - _t0));
        std::fflush(stderr);
      }
      if (round > max_rounds && _prog) {
        std::fprintf(
            stderr,
            "[jaccl-prog] send() MAX_ROUNDS_EXCEEDED (non-fatal, still "
            "retrying) rank=%d call_id=%u dst=%d round=%d max_rounds=%d\n",
            rank_,
            call_id,
            dst,
            round,
            max_rounds);
        std::fflush(stderr);
      }
      bool resending = !to_resend.empty();

      int outstanding = 0;
      int c = next_c;
      for (int buff = 0; c < num_chunks && buff < SEND_INFLIGHT; c++) {
        if (!resending || to_resend[c]) {
          post_chunk(c, buff);
          buff++;
          outstanding++;
        }
      }

      uint64_t last_progress = mach_absolute_time();
      int prev_progress = c;
      bool _deadline_logged = false;
      while (outstanding > 0) {
        if (!_deadline_logged &&
            mach_ticks_to_us(mach_absolute_time() - _t0) > _deadline_us &&
            _prog) {
          _deadline_logged = true;
          std::fprintf(
              stderr,
              "[jaccl-prog] send() DEADLINE_HIT (non-fatal, still "
              "retrying) rank=%d call_id=%u dst=%d round=%d "
              "outstanding=%d chunks_sent_before_stall=%d/%d "
              "elapsed_us=%llu\n",
              rank_,
              call_id,
              dst,
              round,
              outstanding,
              c,
              num_chunks,
              (unsigned long long)mach_ticks_to_us(
                  mach_absolute_time() - _t0));
          std::fflush(stderr);
        }
        int cur_progress = c;
        if (cur_progress != prev_progress) {
          prev_progress = cur_progress;
          last_progress = mach_absolute_time();
        } else if (
            mach_ticks_to_us(mach_absolute_time() - last_progress) >
            drain_quiet_us) {
          break; // quiet -> fall through to barrier + retransmit
        }
        ibv_wc wc[16];
        int n = connections_[dst].poll(16, wc);
        for (int i = 0; i < n; i++) {
          // DIAGNOSTIC (2026-08-10, Section 43 continued): ibverbs-level
          // completion trace -- every CQE this poll sees, whether or not
          // it matches this call, and its raw status. Added because a
          // deterministic, 100%-reproducible stall was traced down to
          // THIS exact silent-discard point (both the call_id mismatch
          // 'continue' and the status/work_type mismatch 'continue' just
          // below) with zero prior visibility into which branch was
          // actually firing, or whether ibv_poll_cq ever returned
          // anything for this call_id at all. Gated on the same
          // JACCL_TRACE_PROGRESS flag as the rest of this investigation's
          // instrumentation -- zero cost otherwise.
          if (_prog) {
            std::fprintf(
                stderr,
                "[jaccl-cqe] send() CQE rank=%d this_call_id=%u "
                "wc_call_id=%u wc_status=%d wc_work_type=%d wc_buff=%d "
                "matches_call_id=%d\n",
                rank_,
                call_id,
                wr_id_call_id(wc[i].wr_id),
                static_cast<int>(wc[i].status),
                wr_id_work_type(wc[i].wr_id),
                wr_id_buff(wc[i].wr_id),
                wr_id_call_id(wc[i].wr_id) == call_id ? 1 : 0);
            std::fflush(stderr);
          }
          if (wr_id_call_id(wc[i].wr_id) != call_id) {
            continue; // stale (previous call/round)
          }
          if (wc[i].status != IBV_WC_SUCCESS || wr_id_work_type(wc[i].wr_id) != SEND_WR) {
            // Section 74 PROBE. init_attr.send_cq == init_attr.recv_cq
            // (rdma.cpp:171-172), so send() and recv() share ONE
            // completion queue on this QP. A CQE is consumed exactly
            // once, by whichever loop reaps it. A SUCCESSFUL RECV
            // completion arriving here is therefore not an error -- it
            // is recv()'s data, and discarding it makes the frame look
            // lost to recv() even though the wire delivered it.
            //
            // Distinguish that from a genuine error: on UC a real wire
            // drop produces NO CQE at all, so status==SUCCESS with the
            // wrong work_type can only mean the data actually arrived.
            if (wc[i].status == IBV_WC_SUCCESS &&
                wr_id_work_type(wc[i].wr_id) == RECV_WR) {
              ++_xconsume_recv_in_send;
              if (_prog) {
                std::fprintf(
                    stderr,
                    "[jaccl-x74] send() REAPED A RECV CQE rank=%d "
                    "this_call_id=%u wc_call_id=%u buff=%d total=%d\n",
                    rank_, call_id, wr_id_call_id(wc[i].wr_id),
                    wr_id_buff(wc[i].wr_id),
                    _xconsume_recv_in_send);
                std::fflush(stderr);
              }
            }
            continue; // dropped/erred/unexpected — barrier+retransmit recovers
          }
          int wb = wr_id_buff(wc[i].wr_id);
          outstanding--;
          while (c < num_chunks && !(!resending || to_resend[c])) {
            c++;
          }
          if (c < num_chunks) {
            post_chunk(c, wb);
            c++;
            outstanding++;
          }
          last_progress = mach_absolute_time();
          prev_progress = c;
        }
        if (n == 0) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(jaccl_reliable_idle_us()));
        }
      }

      // Report: ask the receiver what it has (we send a dummy -- we're
      // the source, we always "have everything"). Design doc Section 37
      // Phase 1 (2026-08-10): migrated off the framed TCP
      // p2p_channel_->p2p_retry_barrier onto the RDMA-native
      // p2p_retry_exchange (see class doc comment for the full
      // no-lockstep/seq-keyed/no-acking rationale). rank_ is passed as
      // data_src_rank since send() IS the data source for this transfer.
      auto peer_got = p2p_retry_exchange(
          dst, static_cast<uint32_t>(rank_), seq, static_cast<uint32_t>(round),
          std::vector<uint8_t>{});
      bool peer_has_all =
          static_cast<int>(peer_got.size()) == num_chunks &&
          std::count(peer_got.begin(), peer_got.end(), 1) == num_chunks;
      if (_prog) {
        int peer_got_count = static_cast<int>(
            std::count(peer_got.begin(), peer_got.end(), 1));
        std::fprintf(
            stderr,
            "[jaccl-prog] send() BARRIER rank=%d call_id=%u dst=%d round=%d "
            "peer_got_size=%d peer_got_count=%d/%d peer_has_all=%d "
            "elapsed_us=%llu\n",
            rank_,
            call_id,
            dst,
            round,
            static_cast<int>(peer_got.size()),
            peer_got_count,
            num_chunks,
            peer_has_all ? 1 : 0,
            (unsigned long long)mach_ticks_to_us(mach_absolute_time() - _t0));
        std::fflush(stderr);
      }
      if (peer_has_all) {
        break;
      }
      // DATA-PROGRESS STALLWATCH tick (see the loop-entry comment above
      // for the full rationale). Metric: peer_got_count, the ONLY
      // signal that actually reflects whether the receiver is making
      // real progress on THIS transfer -- as opposed to
      // p2p_retry_exchange()'s own StallWatch, whose metric reflects only
      // the metadata barrier's liveness and is blind to this. Primed on
      // the first report (round 0) rather than before the loop so a
      // send() that starts with peer_got_count=0 doesn't immediately
      // look like "no progress from an initial nonzero baseline" --
      // 0 is the expected starting state, not evidence of a stall.
      {
        int peer_got_count = static_cast<int>(
            std::count(peer_got.begin(), peer_got.end(), 1));
        if (!_data_stall_primed) {
          _data_stall = StallWatch(peer_got_count);
          _data_stall.timeout_us = jaccl_p2p_retry_stall_timeout_us();
          _data_stall_primed = true;
        }
        // DIAGNOSTIC (2026-08-11, Section 43 continued): on the verge of
        // throwing, dump BOTH QPs' live ibverbs state first (data QP and
        // p2p_retry QP for this peer) -- answers whether the underlying
        // QP silently entered IBV_QPS_ERR (a real transport failure the
        // retry protocol has no visibility into) vs. remained healthy
        // RTS the whole time (genuinely lost/dropped UC datagrams, a
        // different root cause requiring a different fix). try/catch
        // instead of modifying StallWatch's signature -- keeps this
        // fully isolated to the one call site that needs it.
        try {
          _data_stall.tick(peer_got_count, "send() data-progress", rank_, call_id);
        } catch (const std::runtime_error&) {
          std::fprintf(
              stderr,
              "[jaccl-qp] STALL QP STATE rank=%d call_id=%u data_qp=[%s] "
              "p2p_retry_qp=[%s]\n",
              rank_,
              call_id,
              connections_[dst].debug_dump_qp_state().c_str(),
              p2p_retry_connections_[dst].debug_dump_qp_state().c_str());
          std::fflush(stderr);
          // DIAGNOSTIC (2026-08-11, Section 43 continued): distinguishes
          // a fully-wedged driver-level WQE pipeline (nothing we can fix)
          // from a narrower failure mode (something jaccl/mlx CAN work
          // around). Safe to run here: we're already on the throw path
          // that's about to trigger reconnect_fresh(), which discards
          // this QP entirely -- pushing it to IBV_QPS_ERR via the probe
          // costs nothing extra.
          std::fprintf(
              stderr,
              "[jaccl-qp] POISON PROBE rank=%d call_id=%u result=[%s]\n",
              rank_,
              call_id,
              connections_[dst].poison_send_probe().c_str());
          std::fflush(stderr);
          throw;
        }
      }
      to_resend.assign(num_chunks, 0);
      for (int k = 0; k < num_chunks; k++) {
        if (k >= static_cast<int>(peer_got.size()) || !peer_got[k]) {
          to_resend[k] = 1;
        }
      }
      next_c = 0;
    }
  }

  void recv(uint32_t call_id, char* out_ptr, int64_t n_bytes, int src) {
    // See send()'s matching comment: the retry protocol structurally
    // requires the dedicated p2p retry channel.
    if (p2p_channel_ == nullptr) {
      throw std::runtime_error(
          "[jaccl] recv() called with no dedicated p2p retry channel (p2p_channel_) -- the "
          "retry-on-drop protocol requires it; only top-level groups have "
          "one, and send()/recv() should only be called on a top-level "
          "group (exo's Pipeline-Parallel usage)");
    }
    // QP-BUDGET GATE (2026-08-10): mirrors the identical guard in send() --
    // p2p_retry_connections_ is PP-only because the Thunderbolt HCA caps at
    // max_qp=3. See mesh.cpp's jaccl_pipeline_mode_enabled().
    if (p2p_retry_connections_.empty()) {
      throw std::runtime_error(
          "[jaccl] recv() called with no p2p_retry_connections_ (dedicated p2p "
          "retry QP) -- this QP is only allocated when "
          "MLX_JACCL_SHARDING_MODE=Pipeline. Either TP-mode code reached the "
          "PP-only raw send()/recv() path (TP should use all_reduce/all_gather/"
          "all_sum), or the runner was launched in PP without "
          "MLX_JACCL_SHARDING_MODE=Pipeline being set/propagated");
    }
    // Pre-transfer rendezvous REMOVED and zero-byte early-return simplified
    // -- see send()'s matching comment for the full rationale (design doc
    // Section 37 Phase 1).
    if (n_bytes == 0) {
      return;
    }

    auto [sz, buffer_size] = buffer_size_from_message(n_bytes);
    if (sz > jaccl_reliable_max_sz()) {
      sz = jaccl_reliable_max_sz();
      buffer_size = static_cast<int64_t>(FRAME_SIZE) * (1 << sz);
    }
    const int HDR = static_cast<int>(sizeof(uint32_t));
    const int64_t chunk_bytes = buffer_size - HDR;
    const int num_chunks =
        static_cast<int>((n_bytes + chunk_bytes - 1) / chunk_bytes);
    const int RECV_INFLIGHT = (sz <= 2)
        ? std::max(1, std::min(jaccl_reliable_inflight(), NUM_BUFFERS))
        : 1;
    // Real production incident fix (2026-08-08) -- see send_seq_/
    // recv_seq_'s own field comment and send()'s matching seq-tagging
    // comment for the full incident/rationale. One seq value per
    // logical recv() call from this peer -- must increment in lockstep
    // with the peer's send_seq_[this_rank] on THEIR side, since both
    // are per-(peer, direction) counters on the same ordered channel.
    const uint32_t expected_seq = static_cast<uint32_t>(recv_seq_[src]++);
    if (jaccl_progress_enabled()) {
      // Section 69: the RECV half of the counter trace. Diff this
      // stream against the peer's SEND stream to find the first call
      // where the two counters stop agreeing.
      std::fprintf(
          stderr,
          "[jaccl-seq69] RECV rank=%d src=%d expected_seq=%u call_id=%u "
          "bytes=%lld\n",
          rank_, src, expected_seq, call_id, (long long)n_bytes);
      std::fflush(stderr);
    }

    std::vector<uint8_t> got(num_chunks, 0);
    int all_recv = 0;

    auto post_recv_buff = [&](int buff) {
      auto& rb = recv_buffer(sz, buff, src);
      zero_recv_buffer(rb);
      connections_[src].post_recv(rb, make_wr_id(call_id, RECV_WR, buff, src));
    };
    auto consume_recv = [&](int buff) {
      JACCL_DMA_BARRIER();
      auto& rb = recv_buffer(sz, buff, src);
      const char* p = rb.data<char>();
      uint32_t hdr;
      std::memcpy(&hdr, p, HDR);
      const uint32_t seq = hdr >> 16;
      const uint32_t c = hdr & 0xFFFFu;
      if (seq != expected_seq) {
        // Stale/duplicate message from a call other than the one this
        // recv() is currently servicing (e.g. an orphaned retransmit
        // from an earlier call whose original recv already completed
        // and moved on -- the exact incident this fix closes). Discard
        // -- do NOT touch got[]/all_recv, and do NOT re-post the buffer
        // here: the caller (this recv()'s own polling loop, immediately
        // below) ALREADY unconditionally decrements `posted` and
        // conditionally re-posts the same buffer slot after every
        // consume_recv() call, on both the normal and this discard
        // path -- calling post_recv_buff() here too would double-post
        // the same buffer (two outstanding recv WRs on one physical
        // slot), a real RDMA-semantics violation this fix must not
        // introduce. Loud once per occurrence (not hot-path spam under
        // normal operation, this should be rare) so a real
        // reproduction is traceable.
        std::fprintf(
            stderr,
            "[jaccl] recv() discarded stale message: src=%d buff=%d "
            "received_seq=%u expected_seq=%u chunk=%u\n",
            src, buff, seq, expected_seq, c);
        std::fflush(stderr);
        return;
      }
      if (c < static_cast<uint32_t>(num_chunks) && !got[c]) {
        int64_t off = static_cast<int64_t>(c) * chunk_bytes;
        int64_t len = std::min(chunk_bytes, n_bytes - off);
        std::memcpy(out_ptr + off, p + HDR, static_cast<size_t>(len));
        got[c] = 1;
        all_recv++;
      }
    };

    int posted = 0;
    for (int b = 0; b < std::min(RECV_INFLIGHT, num_chunks); b++) {
      post_recv_buff(b);
      posted++;
    }

    const uint64_t _t0 = mach_absolute_time();
    const uint64_t _deadline_us = 15000000;
    // Section 71: p2p-specific quiet period. See
    // jaccl_p2p_drain_quiet_us()'s comment -- this loop's real round trip
    // is 69-150us, so the old shared 500ms timer was purely the cost of
    // noticing a dropped frame, at ~5000x the actual latency.
    const uint64_t drain_quiet_us = jaccl_p2p_drain_quiet_us();
    const int max_rounds = std::max(8, jaccl_ack_retransmit_max());

    bool _prog = jaccl_progress_enabled();
    // ROOT-CAUSE FIX (2026-08-09, design doc Section 39): mirrors
    // send()'s own fix above -- see that comment for the full
    // rationale (real-hardware evidence of a legitimate 29-round/15s
    // transfer thrown away by this exact cap despite p2p_retry_barrier
    // succeeding every round, plus direct user correction that a slow
    // transfer must never be treated as a fatal one). max_rounds/
    // _deadline_us below no longer THROW -- only p2p_retry_barrier's
    // own recv (the real liveness check, 300s per Section 38) can end
    // this loop in failure now.
    //
    // DATA-PROGRESS STALLWATCH (2026-08-10, Section 43 continued): mirrors
    // send()'s own fix above -- see that comment for the full rationale
    // (Section 39's stated liveness guarantee, p2p_retry_exchange()'s own
    // StallWatch, only tracks the metadata barrier's heartbeat and is
    // blind to a permanently-dropped data chunk; confirmed on real
    // hardware, 8+ min / 900+ rounds with all_recv permanently stuck at
    // 0/num_chunks and zero STALLED throws).
    StallWatch _data_stall(-1); // -1 sentinel: primed on first BARRIER report
    bool _data_stall_primed = false;
    for (int round = 0;; round++) {
      if (_prog) {
        std::fprintf(
            stderr,
            "[jaccl-prog] recv() ROUND rank=%d call_id=%u src=%d round=%d "
            "all_recv=%d/%d elapsed_us=%llu\n",
            rank_,
            call_id,
            src,
            round,
            all_recv,
            num_chunks,
            (unsigned long long)mach_ticks_to_us(mach_absolute_time() - _t0));
        std::fflush(stderr);
      }
      if (round > max_rounds && _prog) {
        std::fprintf(
            stderr,
            "[jaccl-prog] recv() MAX_ROUNDS_EXCEEDED (non-fatal, still "
            "retrying) rank=%d call_id=%u src=%d round=%d max_rounds=%d "
            "all_recv=%d/%d\n",
            rank_,
            call_id,
            src,
            round,
            max_rounds,
            all_recv,
            num_chunks);
        std::fflush(stderr);
      }

      uint64_t last_progress = mach_absolute_time();
      int prev_progress = all_recv;
      bool _deadline_logged = false;
      while (all_recv < num_chunks) {
        if (!_deadline_logged &&
            mach_ticks_to_us(mach_absolute_time() - _t0) > _deadline_us &&
            _prog) {
          _deadline_logged = true;
          std::fprintf(
              stderr,
              "[jaccl-prog] recv() DEADLINE_HIT (non-fatal, still "
              "retrying) rank=%d call_id=%u src=%d "
              "round=%d all_recv=%d/%d elapsed_us=%llu\n",
              rank_,
              call_id,
              src,
              round,
              all_recv,
              num_chunks,
              (unsigned long long)mach_ticks_to_us(
                  mach_absolute_time() - _t0));
          std::fflush(stderr);
        }
        if (all_recv != prev_progress) {
          prev_progress = all_recv;
          last_progress = mach_absolute_time();
        } else if (
            mach_ticks_to_us(mach_absolute_time() - last_progress) >
            drain_quiet_us) {
          break; // quiet -> fall through to barrier (report got)
        }
        ibv_wc wc[16];
        int n = connections_[src].poll(16, wc);
        for (int i = 0; i < n; i++) {
          // DIAGNOSTIC (2026-08-10, Section 43 continued): mirrors send()'s
          // own CQE trace above -- see that comment for the full rationale.
          if (_prog) {
            std::fprintf(
                stderr,
                "[jaccl-cqe] recv() CQE rank=%d this_call_id=%u "
                "wc_call_id=%u wc_status=%d wc_work_type=%d wc_buff=%d "
                "matches_call_id=%d\n",
                rank_,
                call_id,
                wr_id_call_id(wc[i].wr_id),
                static_cast<int>(wc[i].status),
                wr_id_work_type(wc[i].wr_id),
                wr_id_buff(wc[i].wr_id),
                wr_id_call_id(wc[i].wr_id) == call_id ? 1 : 0);
            std::fflush(stderr);
          }
          if (wr_id_call_id(wc[i].wr_id) != call_id) {
            continue;
          }
          if (wc[i].status != IBV_WC_SUCCESS || wr_id_work_type(wc[i].wr_id) != RECV_WR) {
            // Section 74 PROBE, mirror of send()'s. A successful SEND
            // completion reaped here belongs to send(), which is now
            // waiting on a completion that already arrived and was
            // thrown away. Same shared-CQ mechanism, opposite direction.
            if (wc[i].status == IBV_WC_SUCCESS &&
                wr_id_work_type(wc[i].wr_id) == SEND_WR) {
              ++_xconsume_send_in_recv;
              if (_prog) {
                std::fprintf(
                    stderr,
                    "[jaccl-x74] recv() REAPED A SEND CQE rank=%d "
                    "this_call_id=%u wc_call_id=%u buff=%d total=%d\n",
                    rank_, call_id, wr_id_call_id(wc[i].wr_id),
                    wr_id_buff(wc[i].wr_id),
                    _xconsume_send_in_recv);
                std::fflush(stderr);
              }
            }
            continue;
          }
          int wb = wr_id_buff(wc[i].wr_id);
          consume_recv(wb);
          posted--;
          if (posted < std::min(RECV_INFLIGHT, num_chunks - all_recv)) {
            post_recv_buff(wb);
            posted++;
          }
        }
        if (n == 0) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(jaccl_reliable_idle_us()));
        }
      }

      // Report our got-bitmask. Design doc Section 37 Phase 1
      // (2026-08-10): migrated off the framed TCP
      // p2p_channel_->p2p_retry_barrier onto the RDMA-native
      // p2p_retry_exchange (see class doc comment for the full
      // no-lockstep/seq-keyed/no-acking rationale). `src` is passed as
      // data_src_rank since recv()'s peer (the sender) is the data
      // source for this transfer. The returned value (send()'s own
      // trivially-empty report) is intentionally unused here -- recv()'s
      // retry-loop decision (all_recv >= num_chunks) is entirely local,
      // computed from its own got[]/all_recv above; the old barrier's
      // return value was likewise unused on this side (see the pre-
      // migration code, which never read this call's own return).
      p2p_retry_exchange(
          src, static_cast<uint32_t>(src), expected_seq,
          static_cast<uint32_t>(round), got);
      if (_prog) {
        std::fprintf(
            stderr,
            "[jaccl-prog] recv() BARRIER rank=%d call_id=%u src=%d round=%d "
            "got_count=%d/%d all_recv=%d/%d elapsed_us=%llu\n",
            rank_,
            call_id,
            src,
            round,
            static_cast<int>(std::count(got.begin(), got.end(), 1)),
            num_chunks,
            all_recv,
            num_chunks,
            (unsigned long long)mach_ticks_to_us(mach_absolute_time() - _t0));
        std::fflush(stderr);
      }
      if (all_recv >= num_chunks) {
        break;
      }
      // DATA-PROGRESS STALLWATCH tick (see the loop-entry comment above
      // for the full rationale). Metric: all_recv, this rank's own
      // directly-observed count of successfully landed data chunks --
      // strictly more reliable than send()'s peer_got_count (which is
      // self-reported over the metadata barrier) since recv() doesn't
      // need to trust anything the peer says about ITS OWN progress.
      {
        if (!_data_stall_primed) {
          _data_stall = StallWatch(all_recv);
          _data_stall.timeout_us = jaccl_p2p_retry_stall_timeout_us();
          _data_stall_primed = true;
        }
        // DIAGNOSTIC (2026-08-11, Section 43 continued): mirrors send()'s
        // own QP-state dump on the throw path -- see that comment for
        // the full rationale.
        try {
          _data_stall.tick(all_recv, "recv() data-progress", rank_, call_id);
        } catch (const std::runtime_error&) {
          std::fprintf(
              stderr,
              "[jaccl-qp] STALL QP STATE rank=%d call_id=%u data_qp=[%s] "
              "p2p_retry_qp=[%s]\n",
              rank_,
              call_id,
              connections_[src].debug_dump_qp_state().c_str(),
              p2p_retry_connections_[src].debug_dump_qp_state().c_str());
          std::fflush(stderr);
          // DIAGNOSTIC (2026-08-11, Section 43 continued): mirrors
          // send()'s own poison-WR probe -- see that comment for the
          // full rationale. Same QP object serves both directions, so
          // probing its SQ here is still diagnostic of whether the WQE
          // processing pipeline for THIS QP is wedged, even though
          // recv() itself only issues RQ-side work.
          std::fprintf(
              stderr,
              "[jaccl-qp] POISON PROBE rank=%d call_id=%u result=[%s]\n",
              rank_,
              call_id,
              connections_[src].poison_send_probe().c_str());
          std::fflush(stderr);
          throw;
        }
      }
      // Still missing some chunks: keep enough recvs posted for the
      // sender's retransmits to land (sliding window invariant).
      while (posted < std::min(RECV_INFLIGHT, num_chunks - all_recv)) {
        // Find a buff slot not currently posted — reuse index cycling
        // 0..RECV_INFLIGHT-1; safe because posted < window here.
        post_recv_buff(posted % RECV_INFLIGHT);
        posted++;
      }
    }
    // NOTE: no separate post-transfer barrier call here (unlike the old
    // ack_sync_post-based version) -- the loop's own final
    // p2p_retry_barrier call above (the one that observed all_recv >=
    // num_chunks and broke out) already IS the paired rendezvous with
    // send()'s matching final p2p_retry_barrier call (the one that sees
    // peer_has_all=true and breaks). Both sides call p2p_retry_barrier
    // exactly the same number of times per transfer this way; adding an
    // extra call here would desync that count against send(), which has
    // no equivalent trailing call.
  }

  // Pre-post a pool of ACK_RECVs per peer at QP setup time. Called
  // from MeshGroup ctor so the very first ack_sync_post's incoming
  // ACK_SEND from peer always finds a posted recv WR.
  //
  // Pool depth (ACK_RECV_POOL): if peer's ACK_SEND rate exceeds our
  // drain_acks rate, the pool absorbs the burst. drain_acks replenishes
  // one ACK_RECV per consumption. Without enough depth, peer's ack_send
  // can arrive on a QP with no posted recv WR → UC drops → wedge.
  //
  // Pool of 64 is sized for the observed cross-rank coord-lambda lead
  // at c=2 when one rank's master is busy and the other's is idle.
  static constexpr int ACK_RECV_POOL = 64;

  // Clear stale ACK bookkeeping across an in-place reconnect. cached_ack_recvs_
  // belonged to the pre-wedge connection and must not carry into the fresh one.
  void reset_ack_state() {
    cached_ack_recvs_.clear();
    // v2 optimistic-path state belonged to the pre-reconnect QPs.
    v2_pool_posted_ = false;
    v2_stash_ = V2Stash{};
    v2_prev_call_ = 0;
    v2_prev_num_chunks_ = 0;
    v2_prev_sz_ = 0;
    v2_prev_small_ = false;
    v2_prev_want_.clear();
    std::fill(
        std::begin(v2_send_outstanding_), std::end(v2_send_outstanding_), 0);
    // p2p_retry_connections_' QPs were just reset (MeshGroup::reconnect()/
    // reconnect_fresh() call this right after queue_pair_reset()/rebuild),
    // which flushes/discards every in-flight WR -- any slot this rank
    // still believed had an outstanding send is now stale bookkeeping for
    // a completion that will never arrive on the fresh QP. Without this
    // reset, p2p_retry_send_bitmask's wait loop for that slot would spin
    // until jaccl_stall_timeout_us and throw, even though nothing is
    // actually wrong post-reconnect. Same rationale as v2_send_outstanding_
    // just above.
    for (auto& row : p2p_retry_send_outstanding_) {
      std::fill(std::begin(row), std::end(row), false);
    }
  }

  void post_ack_recvs(uint32_t call_id) {
    // No-op when ack_connections_ is empty (top-level group: uses the
    // original inline ack_sync_post on data QP, no pre-posting needed).
    if (ack_connections_.empty()) {
      return;
    }
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& rbuf = ack_recv_buffers_[peer];
      std::memset(rbuf.data<char>(), 0, rbuf.size());
      JACCL_DMA_BARRIER();
      for (int i = 0; i < ACK_RECV_POOL; i++) {
        ack_connections_[peer].post_recv(
            rbuf, make_wr_id(call_id, ACK_RECV_WR, 0, peer));
      }
    }
  }

  // Pre-post a standing recv pool on the DATA QP (connections_) for the
  // sz=0 (<=4096-byte) size class -- design doc Section 52, 2026-08-15.
  //
  // THE BUG THIS CLOSES. PP decode showed a bimodal barrier latency:
  //   <100ms : 29101   (healthy path is 71-122 MICROseconds)
  //   >1s    :  1403   (4.5% of barriers)
  // Every slow one had the identical shape on both ranks -- rank0 posts its
  // send in ~45us, immediately observes peer_got_count=0/1, waits the FULL
  // 500ms retransmit quiet timer, retransmits, and the retransmit SUCCEEDS.
  // ~1403 x 0.5s = ~700s of pure stall in a single 100K-context run, which
  // dominated decode throughput (0.48 tok/s vs 18.01 tok/s on identical
  // config, purely as a function of how many stalls a run happened to hit).
  //
  // It is NOT wire loss: en3 reports Ierrs 0 / Oerrs 0 / Coll 0 on a healthy
  // 8X / 10Gbps link. Two measurements identify it as the empty-recv-FIFO UC
  // drop that this file's own ack_sync_pre comment already describes ("peer
  // SEND lands at our empty data-QP recv FIFO and UC silently drops"):
  //   - SIZE-SPECIFIC: every single retransmit is num_chunks=1; the bulk
  //     2049-chunk transfers lose NOTHING. Only the FIRST chunk of a transfer
  //     is exposed, because after that recv()'s own repost loop stays ahead
  //     (RECV_INFLIGHT == SEND_INFLIGHT, so later chunks can never outrun the
  //     reposts). A single-chunk message is 100% first-chunk, hence 100%
  //     exposed -- exactly the observed distribution.
  //   - SENDER-ASYMMETRIC: rank0 (the driver, which races ahead) took 670
  //     retransmits vs rank1's 181 (3.7x). Wire loss would be symmetric; a
  //     send-arrives-before-recv-is-posted race hits whichever side runs
  //     ahead of its peer's recv() entry.
  //
  // WHY A POOL AND NOT A BARRIER. The obvious fix -- mirroring all_reduce's
  // "post recvs -> confirmed_coord_barrier -> post sends" into the p2p path
  // -- is WRONG here and would be worse than the bug. That barrier is keyed
  // on call_id, which is only incidentally aligned across ranks for
  // collectives (all ranks run them in lockstep). On the p2p path the ranks
  // make unequal numbers of send()/recv() calls by construction, so the
  // per-process call_id counters provably diverge and the barrier could
  // "succeed" while pairing one rank's send #7 with the peer's recv #9 --
  // silently certifying the WRONG ordering, where today's failure at least
  // self-heals via retransmit. It would also convert a 500ms stall into a
  // permanent hang whenever a matching recv() is never entered (the
  // cancel/abort path). A standing pool needs no cross-rank key at all.
  //
  // WHY call_id=0 IS SAFE HERE. Like post_p2p_retry_recvs, these WRs are
  // tagged with a call_id of 0 because a standing pool is by definition
  // posted before any particular call exists. recv()'s poll loop skips
  // completions whose wr_id call_id doesn't match its own, so it will ignore
  // these -- that is intentional and harmless: their PURPOSE is to keep the
  // hardware FIFO non-empty so an early-arriving frame is captured by the
  // NIC rather than silently dropped by UC, not to deliver data. Payload
  // correctness is unaffected because consume_recv() already validates every
  // frame against the on-wire (seq, chunk) header and discards stale ones
  // (the 2026-08-08 stale-message fix) -- call_id is a redundant second gate,
  // not the integrity mechanism.
  //
  // SCOPE (CORRECTED 2026-08-15, design doc Section 64). This pool was
  // originally scoped to sz=0 on the reasoning that "buffer_size_from_message
  // maps every message under FRAME_SIZE (4096B) to size class 0, and the
  // small PP control messages that make up the entire observed loss
  // population are all in it. Larger classes are demonstrably not losing
  // frames."
  //
  // That reasoning was WRONG, and the counter-evidence is direct. Section 52's
  // sz=0 pool drove true lost-send stalls from 1403 to zero FOR THE TRAFFIC IT
  // COVERED, but Section 60 then measured 32 surviving `peer_got_count=0/1`
  // events (peer received NOTHING) after that fix had landed, and Section 63/64
  // traced them to the PP batched-decode per-token activation send -- which
  // runs at sz=2 (65536B, `[jaccl-reliable] ENTER ... sz=2 total_bytes=65536`),
  // not sz=0. "Larger classes are demonstrably not losing frames" was an
  // inference from the pre-fix loss population, not a measurement of the
  // post-fix one.
  //
  // Cost of each such loss is ~500ms: the retransmit is posted ~1us after the
  // barrier reports the drop, but the round-1 barrier confirming it does not
  // return for a full retransmit quiet period, on a link whose healthy round
  // trip is 56-150us. At roughly one loss per decode token that is the entire
  // ~60x requirement-3 decode shortfall.
  //
  // So the pool now covers the size classes the PP data path actually uses,
  // not just sz=0.
  //
  // WHY NOT ALL EIGHT CLASSES. `MAX_RECV_WR` is 32 per QP (rdma.h) and each
  // class costs NUM_BUFFERS=8 WRs, so the hard ceiling is 4 classes per peer;
  // posting all 8 would silently overrun the queue depth. sz 0..3 covers
  // 4096B through 32768B per frame -- which spans both the small PP control
  // messages Section 52 fixed AND the sz=2 (65536B total, 16380B chunks)
  // decode activation send that Sections 63/64 measured losing frames. The
  // buffers themselves are already allocated for every class (see buffers_
  // and recv_buffer()'s sz index), so this registers no new memory; it only
  // keeps the hardware FIFO non-empty so an early-arriving frame is captured
  // by the NIC instead of being silently dropped by UC.
  static constexpr int DATA_RECV_POOL_SIZE_CLASSES = 4;

  void post_data_recv_pool() {
    if (!jaccl_data_recv_pool_enabled()) {
      return;
    }
    if (connections_.empty()) {
      return;
    }
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      for (int sz = 0; sz < DATA_RECV_POOL_SIZE_CLASSES; sz++) {
        for (int b = 0; b < NUM_BUFFERS; b++) {
          auto& rb = recv_buffer(sz, b, peer);
          zero_recv_buffer(rb);
          connections_[peer].post_recv(rb, make_wr_id(0, RECV_WR, b, peer));
        }
      }
    }
  }

  // Pre-post the standing recv pool on p2p_retry_connections_. Called from
  // MeshGroup ctor / reconnect() / reconnect_fresh() (same lifecycle as
  // post_ack_recvs, mirrored for the new QP) so a peer's very first
  // p2p_retry_exchange frame always finds a posted recv WR. Uses a
  // ROTATING pool of P2P_RETRY_NUM_SLOTS buffers (not one fixed slot like
  // ack_recv_buffers_) since a single round can legitimately have several
  // frames of one bitmask in flight at once -- see class doc comment.
  void post_p2p_retry_recvs() {
    if (p2p_retry_connections_.empty()) {
      return;
    }
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      for (int i = 0; i < P2P_RETRY_NUM_SLOTS; i++) {
        auto& rbuf = p2p_retry_recv_buffers_[peer * P2P_RETRY_NUM_SLOTS + i];
        std::memset(rbuf.data<char>(), 0, rbuf.size());
        JACCL_DMA_BARRIER();
        p2p_retry_connections_[peer].post_recv(
            rbuf, make_wr_id(0, P2P_RETRY_RECV_WR, i, peer));
      }
    }
  }

  // Per-(peer, direction, seq) accumulator for one send()/recv() transfer's
  // p2p_retry_exchange. Lives for the duration of one send()/recv() call
  // (constructed fresh each call, not persisted across calls) EXCEPT for
  // last_sent_ below, which IS persisted per-peer across calls to answer
  // stale-round frames from a peer who is still retransmitting into a
  // transfer this rank has already completed and moved past (the tail
  // case -- see class doc comment point 3).
  struct P2PExchange {
    // What WE report to the peer this transfer (send(): trivially empty,
    // we're the source and always "have everything" already, matching the
    // old p2p_retry_barrier's own send()-side convention; recv(): our
    // real, growing got[] bitmask).
    std::vector<uint8_t> my_bitmask;
    // What the PEER has reported to us, accumulated via OR-merge across
    // every frame received for this (peer, direction, seq) regardless of
    // round (see class doc comment point 1 -- round is diagnostic only).
    std::vector<uint8_t> peer_bitmask;
    // Which frame_indices of peer_bitmask have been written at least once
    // (coverage mask, NOT a counter -- correctness requirement from the
    // design doc's consult review). peer_bitmask is only a COMPLETE,
    // trustworthy reconstruction once every frame_index in
    // [0, peer_num_frames) has arrived at least once; until then it's a
    // valid-but-partial OR-accumulation (safe to read speculatively, e.g.
    // to decide MORE retransmits, but not safe to treat as final).
    std::vector<bool> peer_frame_seen;
    int peer_num_frames = -1; // -1 == not yet known (no frame arrived yet)
    // True length of the peer's bitmask, taken from the LAST frame's
    // frame_len (peer_bitmask is over-allocated to a full
    // peer_num_frames*P2P_PAYLOAD_CAP and must be trimmed to this on
    // return -- the caller's exact-length contract (peer_got.size() ==
    // num_chunks) depends on it, matching the old TCP barrier's exact-
    // length framing).
    int peer_total_len = -1;
    bool peer_complete = false;
  };

  // Send every frame of `bitmask` (chunked into P2P_PAYLOAD_CAP-byte
  // frames) to `peer` for (data_src_rank, seq, round). Caller decides
  // cadence; this just posts the sends. `my_bitmask` may be empty (the
  // send()-side "trivially have everything" case) -- num_frames is then 1,
  // a single frame with frame_len=0, so the peer still gets a liveness
  // signal and a definite peer_num_frames=1 to converge on, not silence.
  //
  // Uses a ROTATING per-frame send-buffer pool (p2p_retry_send_buffers_,
  // P2P_RETRY_NUM_SLOTS deep per peer -- same shape as the recv pool),
  // NOT a single per-peer slot. A consult review caught a real bug in an
  // earlier draft that used one shared send buffer and blocked
  // (poll-drain) after every single frame to wait for its own completion
  // before reusing it: that blocking drain polled the SAME CQ the main
  // exchange loop also polls, and silently discarded any RECV completion
  // (the peer's own bitmask frames, delivered by the NIC on its own
  // schedule, NOT gated by any application-level mutex) it happened to
  // see while waiting -- permanently losing that frame's payload AND
  // never reposting its recv WR slot, which both stalls the exchange
  // (missed peer progress) and risks exhausting the standing recv pool
  // over repeated rounds (a UC recv-pool exhaustion is a SILENT packet
  // drop at the transport level, unrecoverable without help from a layer
  // above it). Per-frame slots let sends be fire-and-forget in the common
  // case (posting frame i+1 doesn't need to wait on frame i at all, only
  // on ANY prior send still outstanding for the SPECIFIC slot i+1 is
  // about to reuse -- which, since num_frames <= P2P_RETRY_NUM_SLOTS,
  // means the first pass through a call to this function never blocks).
  // The only wait is bounded and uses the unified completion processor
  // (p2p_retry_process_completion) instead of a special-purpose drain
  // loop that ignores other completion types -- see that method's own
  // comment.
  void p2p_retry_send_bitmask(
      int peer,
      uint32_t data_src_rank,
      uint32_t seq,
      uint32_t round,
      const std::vector<uint8_t>& bitmask,
      P2PExchange& ex) {
    const int n = static_cast<int>(bitmask.size());
    const int num_frames = std::max(1, (n + P2P_PAYLOAD_CAP - 1) / P2P_PAYLOAD_CAP);
    if (num_frames > P2P_RETRY_NUM_SLOTS) {
      // Structural invariant, not a runtime condition: send()'s own
      // num_chunks > 0xFFFF guard bounds bitmask.size() to 65535, which
      // caps num_frames at 17 -- well under P2P_RETRY_NUM_SLOTS (24).
      // Fail loudly if that invariant is ever violated instead of
      // silently truncating a bitmask (which would look like data loss).
      throw std::runtime_error(
          "[jaccl] p2p_retry_send_bitmask: bitmask needs more frames than "
          "P2P_RETRY_NUM_SLOTS provides -- invariant violation, not a "
          "transient condition");
    }
    for (int i = 0; i < num_frames; i++) {
      const int slot = i; // frame index doubles as the send-slot index
      // Wait (if needed) for this specific slot's PRIOR outstanding send
      // to complete before overwriting its buffer -- bounded by the same
      // generous peer-liveness timeout as the main exchange loop (a
      // local send completion not arriving is a local NIC/QP fault, not
      // peer-liveness, but reuses the same backstop for simplicity; the
      // main exchange loop's own StallWatch is the primary liveness
      // guard for the exchange as a whole).
      //
      // BUG FIX (2026-08-11, Section 43 investigation): this loop's
      // deadline check used jaccl_stall_timeout_us() (the GENERIC
      // collective-stall default, 8s) instead of
      // jaccl_p2p_retry_stall_timeout_us() (this exchange's own,
      // "generous", 300s default) -- directly contradicting the comment
      // above, which explicitly documents the intent to reuse the
      // "generous" 300s backstop "for simplicity". Confirmed on real
      // hardware: this wrong constant fires a premature fatal
      // "NIC/QP fault" throw after only 8s whenever a send-slot's own
      // completion is merely late (not lost) under real 2-node load --
      // observed directly, crashing a live generation request at ~18s
      // wall-clock, well before the outer p2p_retry_exchange loop could
      // ever reach ITS correctly-configured 300s window. This is a
      // separate, more proximate bug than the p2p_retry_exchange
      // STALLED issue documented in Section 43 -- it can mask the
      // original bug entirely by crashing the exchange first.
      const uint64_t wait_t0 = mach_absolute_time();
      while (p2p_retry_send_outstanding_[peer][slot]) {
        ibv_wc wc[16];
        int n_wc = p2p_retry_connections_[peer].poll(16, wc);
        for (int w = 0; w < n_wc; w++) {
          p2p_retry_process_completion(wc[w], peer, seq, data_src_rank, ex);
        }
        if (n_wc == 0) {
          std::this_thread::sleep_for(
              std::chrono::microseconds(jaccl_reliable_idle_us()));
        }
        if (mach_ticks_to_us(mach_absolute_time() - wait_t0) >
            jaccl_p2p_retry_stall_timeout_us()) {
          throw std::runtime_error(
              "[jaccl] p2p_retry_send_bitmask: local send-slot completion "
              "never arrived -- NIC/QP fault, not a peer-liveness issue");
        }
      }
      auto& sb = p2p_retry_send_buffers_[peer * P2P_RETRY_NUM_SLOTS + slot];
      char* p = sb.data<char>();
      P2PFrameHdr hdr{};
      hdr.magic = kP2PFrameMagic;
      hdr.data_src_rank = data_src_rank;
      hdr.seq = seq;
      hdr.round = round;
      hdr.num_frames = static_cast<uint32_t>(num_frames);
      hdr.frame_index = static_cast<uint32_t>(i);
      const int off = i * P2P_PAYLOAD_CAP;
      const int len = std::min(P2P_PAYLOAD_CAP, n - off);
      hdr.frame_len = static_cast<uint32_t>(std::max(0, len));
      std::memcpy(p, &hdr, P2P_HDR);
      if (len > 0) {
        std::memcpy(p + P2P_HDR, bitmask.data() + off, static_cast<size_t>(len));
      }
      JACCL_DMA_BARRIER();
      p2p_retry_connections_[peer].post_send(
          sb, make_wr_id(0, P2P_RETRY_SEND_WR, slot, peer));
      p2p_retry_send_outstanding_[peer][slot] = true;
    }
  }

  // Unified completion dispatcher -- the ONLY place that consumes a CQE
  // from p2p_retry_connections_[peer]'s CQ. Used by BOTH the send-slot
  // wait loop above and the main exchange loop below, so no completion
  // type is ever silently dropped regardless of WHERE in this class's
  // logic the poll happened to occur (see p2p_retry_send_bitmask's own
  // comment for the bug this fixes: this QP's single CQ carries both
  // SEND and RECV completions interleaved, and any code path that polls
  // it but only understands ONE completion type risks losing the other).
  void p2p_retry_process_completion(
      const ibv_wc& wc,
      int peer,
      uint32_t seq,
      uint32_t data_src_rank,
      P2PExchange& ex) {
    const int wt = wr_id_work_type(wc.wr_id);
    const int slot = wr_id_buff(wc.wr_id);
    if (wt == P2P_RETRY_SEND_WR) {
      p2p_retry_send_outstanding_[peer][slot] = false;
      return;
    }
    if (wt != P2P_RETRY_RECV_WR) {
      return; // unrelated/foreign wr_id -- defensive, should be unreachable
    }
    // Copy-then-repost (design doc correctness requirement #1): pull the
    // header+payload OFF the raw NIC-DMA'd buffer into locals BEFORE any
    // validation/merge decision, then immediately repost the recv WR.
    // Never hold a live decision-making reference into the buffer while
    // the NIC could be about to reuse/overwrite it for the next inbound
    // frame on this slot.
    JACCL_DMA_BARRIER();
    auto& rbuf = p2p_retry_recv_buffers_[peer * P2P_RETRY_NUM_SLOTS + slot];
    P2PFrameHdr hdr;
    std::memcpy(&hdr, rbuf.data<char>(), P2P_HDR);
    std::vector<uint8_t> payload;
    if (hdr.frame_len > 0 &&
        hdr.frame_len <= static_cast<uint32_t>(P2P_PAYLOAD_CAP)) {
      payload.assign(
          rbuf.data<char>() + P2P_HDR,
          rbuf.data<char>() + P2P_HDR + hdr.frame_len);
    }
    // Immediately repost -- this slot must always have a live recv WR.
    std::memset(rbuf.data<char>(), 0, rbuf.size());
    JACCL_DMA_BARRIER();
    p2p_retry_connections_[peer].post_recv(
        rbuf, make_wr_id(0, P2P_RETRY_RECV_WR, slot, peer));

    // Validation (design doc requirement #1, applied on the COPY above,
    // not the live buffer): magic + seq + data_src_rank must all match.
    // seq alone is not sufficient -- send_seq_/recv_seq_ are independent
    // per-direction counters, so a frame from the OPPOSITE direction's
    // transfer can coincidentally carry the same numeric seq (see class
    // doc comment point 2). round is deliberately NOT checked (point 1:
    // diagnostic only).
    if (hdr.magic != kP2PFrameMagic || hdr.seq != seq ||
        hdr.data_src_rank != data_src_rank) {
      // DIAGNOSTIC (2026-08-11, Section 43 investigation): log every
      // reject so a seq-desync (frames arriving but rejected on a
      // constant offset from the expected seq) is directly visible,
      // vs genuinely zero RECV completions ever landing (which would
      // show no EXCHANGE_REJECT lines at all during a stall). Gated
      // on JACCL_TRACE_PROGRESS=1, zero cost otherwise.
      if (jaccl_progress_enabled()) {
        std::fprintf(
            stderr,
            "[jaccl-p2p] EXCHANGE_REJECT rank=%d peer=%d slot=%d "
            "magic_ok=%d recv_seq=%u expected_seq=%u "
            "recv_data_src_rank=%u expected_data_src_rank=%u\n",
            rank_,
            peer,
            slot,
            hdr.magic == kP2PFrameMagic ? 1 : 0,
            hdr.seq,
            seq,
            hdr.data_src_rank,
            data_src_rank);
        std::fflush(stderr);
      }
      return; // stale/foreign frame -- discard, do not merge
    }
    if (hdr.num_frames == 0 ||
        hdr.num_frames > static_cast<uint32_t>(P2P_RETRY_NUM_SLOTS) ||
        hdr.frame_index >= hdr.num_frames) {
      return; // malformed -- defensive, should be unreachable
    }
    if (ex.peer_num_frames < 0) {
      ex.peer_num_frames = static_cast<int>(hdr.num_frames);
      ex.peer_bitmask.assign(
          static_cast<size_t>(ex.peer_num_frames) * P2P_PAYLOAD_CAP, 0);
      ex.peer_frame_seen.assign(ex.peer_num_frames, false);
    } else if (ex.peer_num_frames != static_cast<int>(hdr.num_frames)) {
      // A peer's num_frames for one (peer,seq) transfer is fixed at
      // transfer-size time and cannot legitimately change mid-exchange --
      // treat a mismatch as a foreign/stale frame rather than corrupting
      // the reassembly buffer with an inconsistent frame count.
      return;
    }
    // OR-accumulate (monotonic, idempotent -- safe under any drop/dup/
    // reorder, and safe to mix frames from different rounds per the
    // round-agnostic design -- design doc requirement #2, bitmap not
    // counter, applied via peer_frame_seen below).
    const int off = static_cast<int>(hdr.frame_index) * P2P_PAYLOAD_CAP;
    for (size_t k = 0; k < payload.size(); k++) {
      ex.peer_bitmask[off + k] |= payload[k];
    }
    ex.peer_frame_seen[hdr.frame_index] = true;
    // The LAST frame's frame_len gives the exact true bitmask length
    // (every earlier frame is always exactly P2P_PAYLOAD_CAP bytes by
    // construction in p2p_retry_send_bitmask -- only the final frame can
    // be short). Recording it whenever the last frame is seen is safe
    // under reordering: an EARLIER frame arriving after the last one
    // doesn't change this value; the last frame's own arrival is what
    // sets it, regardless of when.
    if (static_cast<int>(hdr.frame_index) == ex.peer_num_frames - 1) {
      ex.peer_total_len = off + static_cast<int>(hdr.frame_len);
    }
    if (!ex.peer_complete &&
        std::all_of(
            ex.peer_frame_seen.begin(),
            ex.peer_frame_seen.end(),
            [](bool b) { return b; })) {
      ex.peer_complete = true;
    }
  }


  // Core exchange, called identically by send() and recv() (they differ
  // only in what `my_bitmask` means -- see P2PExchange's field comment).
  // Blocks until this rank has accumulated the PEER's complete reported
  // bitmask (peer_complete == true), for BOTH callers uniformly:
  //   - recv()'s call (my_bitmask == its real, growing got[]): the peer
  //     it's waiting on is send(), whose reported bitmask is trivially a
  //     single empty frame (num_frames=1, frame_len=0) -- so this
  //     resolves near-instantly and is NOT a real wait on send()'s
  //     progress, just the minimum handshake to know send() is alive and
  //     has heard this round's report at least once.
  //   - send()'s call (my_bitmask == empty): the peer it's waiting on is
  //     recv(), whose reported bitmask is its REAL got[] -- this is the
  //     actual information send()'s retry loop needs (mirrors the old
  //     p2p_retry_barrier's peer_got/peer_has_all check exactly).
  // Both callers use the same exit condition; no separate parameter
  // needed -- see class doc comment point 3 for why acking (waiting on
  // the PEER's confirmation of what THIS rank sent) was deliberately
  // removed instead.
  std::vector<uint8_t> p2p_retry_exchange(
      int peer,
      uint32_t data_src_rank,
      uint32_t seq,
      uint32_t round,
      const std::vector<uint8_t>& my_bitmask) {
    if (p2p_retry_connections_.empty()) {
      throw std::runtime_error(
          "[jaccl] p2p_retry_exchange called with no dedicated p2p retry "
          "QP (p2p_retry_connections_) -- only top-level groups have one");
    }
    P2PExchange ex;
    ex.my_bitmask = my_bitmask;

    // DIAGNOSTIC (2026-08-11, Section 43 investigation): entry trace for
    // every p2p_retry_exchange call, to test the seq-desync hypothesis --
    // if the two ranks ever call this a different number of times for the
    // same logical direction, send_seq_[dst]/recv_seq_[src] permanently
    // diverge and every subsequent frame in that direction is silently
    // dropped by the seq check in p2p_retry_process_completion (matches
    // observed symptom: metric=0 for the full 300s, both ranks stalling
    // at overlapping wall-clock times, self-healing via reconnect_fresh's
    // seq reset). Gated on JACCL_TRACE_PROGRESS=1 (already set by
    // start_cluster.sh), zero cost otherwise.
    if (jaccl_progress_enabled()) {
      std::fprintf(
          stderr,
          "[jaccl-p2p] EXCHANGE_ENTER rank=%d peer=%d data_src_rank=%u "
          "seq=%u round=%u my_bitmask_len=%zu\n",
          rank_,
          peer,
          data_src_rank,
          seq,
          round,
          my_bitmask.size());
      std::fflush(stderr);
    }

    const uint64_t quiet_us = jaccl_ack_retransmit_us();
    const uint64_t stall_us = jaccl_p2p_retry_stall_timeout_us();
    StallWatch stall(-1); // -1 sentinel: metric set on first real tick below
    bool stall_primed = false;
    uint64_t last_send = 0; // 0 == "never sent yet", forces an immediate
                             // first broadcast below regardless of quiet_us
    for (;;) {
      const uint64_t now = mach_absolute_time();
      if (last_send == 0 ||
          mach_ticks_to_us(now - last_send) > quiet_us) {
        p2p_retry_send_bitmask(peer, data_src_rank, seq, round, ex.my_bitmask, ex);
        last_send = mach_absolute_time();
      }
      if (ex.peer_complete) {
        break;
      }

      ibv_wc wc[16];
      int n = p2p_retry_connections_[peer].poll(16, wc);
      for (int i = 0; i < n; i++) {
        if (wr_id_peer(wc[i].wr_id) != peer) {
          continue; // not ours (another peer's traffic on a shared poll)
        }
        // Delegates to the SAME unified processor p2p_retry_send_bitmask's
        // wait loop uses -- see that method's comment for why this MUST be
        // unified (a special-cased "only understands RECV" loop here
        // silently drops any interleaved SEND completion's bookkeeping,
        // which would permanently wedge p2p_retry_send_outstanding_ for
        // that slot and eventually make every future send for it block
        // forever waiting on a completion that already arrived and was
        // discarded).
        p2p_retry_process_completion(wc[i], peer, seq, data_src_rank, ex);
      }
      if (n == 0) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(jaccl_reliable_idle_us()));
      }
      // Liveness backstop (class doc comment point 3): throw only on
      // genuinely zero forward progress for jaccl_p2p_retry_stall_timeout_us
      // (300s default) -- StallWatch's own tick() already treats an
      // unchanged metric as "no progress"; feed it peer_frame_seen's
      // popcount so ANY new frame (even a partial re-accumulation) resets
      // the clock, matching drain_acks' own pattern.
      const long metric = static_cast<long>(std::count(
          ex.peer_frame_seen.begin(), ex.peer_frame_seen.end(), true));
      if (!stall_primed) {
        stall = StallWatch(metric);
        stall.timeout_us = stall_us;
        stall_primed = true;
      }
      // DIAGNOSTIC (2026-08-15, Section 44 continued): the poison-WR probe
      // built in mlx@9ccf9b198 was wired ONLY to send()/recv()'s own
      // data-progress StallWatch (the connections_[dst] data QP) -- it was
      // NEVER wired here, to p2p_retry_exchange's own StallWatch, despite
      // THIS being the one that has actually fired in every real
      // reproduction of the stall so far (metric=peer_frame_seen popcount,
      // stuck at 0 for the full 300s -- this rank never saw even one valid
      // reply frame for its own (data_src_rank, seq) query the whole
      // window). Run the SAME probe here, against the SAME QP this loop
      // itself uses (p2p_retry_connections_[peer], NOT connections_[dst]/
      // [src] -- a different physical QP from send()/recv()'s data path)
      // to answer the identical question this bug's whole investigation
      // has been chasing: is the driver still processing WQEs on THIS QP
      // at all, or is the wedge below the ibverbs API here too. Safe to
      // run here for the same reason as the send()/recv() call site: this
      // is directly on the throw path that's about to trigger
      // reconnect_fresh(), which discards this QP entirely regardless.
      try {
        stall.tick(metric, "p2p_retry_exchange", rank_, seq);
      } catch (const std::runtime_error&) {
        std::fprintf(
            stderr,
            "[jaccl-p2p-qp] STALL QP STATE rank=%d peer=%d seq=%u "
            "p2p_retry_qp=[%s]\n",
            rank_,
            peer,
            seq,
            p2p_retry_connections_[peer].debug_dump_qp_state().c_str());
        std::fflush(stderr);
        std::fprintf(
            stderr,
            "[jaccl-p2p-qp] POISON PROBE rank=%d peer=%d seq=%u result=[%s]\n",
            rank_,
            peer,
            seq,
            p2p_retry_connections_[peer].poison_send_probe().c_str());
        std::fflush(stderr);
        throw;
      }

    }
    // Trim to the exact bitmask length the peer reported via its last
    // frame's frame_len (peer_bitmask is over-allocated to
    // peer_num_frames*P2P_PAYLOAD_CAP padding -- see peer_total_len's own
    // field comment). peer_complete==true (the only way this loop exits)
    // guarantees the last frame has been seen, so peer_total_len is set.
    ex.peer_bitmask.resize(static_cast<size_t>(ex.peer_total_len));
    return ex.peer_bitmask;
  }

 private:
  // Cross-rank ack barrier — used at BOTH ends of every lambda.
  //
  //   ack_sync_pre(): called BEFORE the data prefill posts. Confirms
  //     peer has reached the same lambda boundary AND posted its
  //     ack_recv as the very first WR on its QP recv queue.
  //   ack_sync_post(): called AFTER the data main loop. Confirms peer
  //     also drained its main loop. Without this, in_flight==0 only
  //     proves OUR side drained; peer might still be polling, and
  //     our next-lambda send could arrive at peer's still-posted
  //     prior-lambda recv WR (different sz → IBV_WC_LOC_LEN_ERR).
  //
  // Reliable confirmed barrier over the TCP coordinator, SELF-VERIFYING: every
  // rank contributes its call_id and all must agree. If they don't, the ranks
  // have desynced (one is at a different collective/barrier) — detect + throw
  // immediately WITH a log, instead of silently corrupting the stream or
  // hanging in recv forever. The coordinator sockets carry an SO_RCVTIMEO so a
  // stuck barrier fails cleanly (throws) well before the 45s _check_hang.
  void confirmed_coord_barrier(uint32_t call_id, const char* which) {
    auto vals = coordinator_->all_gather<uint32_t>(call_id);
    for (int i = 0; i < static_cast<int>(vals.size()); i++) {
      if (vals[i] != call_id) {
        std::fprintf(
            stderr,
            "[jaccl] CONFIRMED BARRIER DESYNC rank=%d %s: my call_id=%u but "
            "rank %d reported call_id=%u\n",
            rank_,
            which,
            call_id,
            i,
            vals[i]);
        std::fflush(stderr);
        throw std::runtime_error(
            "[jaccl] confirmed barrier desync detected (ranks at different "
            "collectives) — throwing for clean re-place");
      }
    }
  }

  // CRITICAL: callers MUST post the per-peer ack_recv BEFORE any
  // other recvs in the lambda so the ack is at the head of the QP
  // recv queue and matches peer's ack_send first.
  void ack_sync_pre(uint32_t call_id) {
    // NOTE: the confirmed-pre barrier is NOT done here — a plain TCP rendezvous
    // in place of the UC ack corrupts, because a data SEND can arrive before
    // the peer posts its data RECV. Collectives that want the reliable+ordered
    // pre barrier instead inline "post recvs -> coordinator barrier -> post
    // sends" in their prefill (see all_reduce). This UC path stays for the
    // collectives that haven't adopted that ordering (and when confirmed-pre is
    // off).
    // Defensive guard: skip if no dedicated ACK QP exists. Both
    // top-level and subgroup groups populate ack_connections_ when
    // ackqp-net is in effect; this guards against future regressions.
    if (ack_connections_.empty()) {
      return;
    }
    int num_peers = size_ - 1;
    int in_flight = 2 * num_peers;
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& sbuf = ack_send_buffers_[peer];
      ack_connections_[peer].post_send(
          sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
    }
    drain_acks(call_id, in_flight);
  }

  void ack_sync_post(uint32_t call_id) {
    // Confirmed barrier (see ack_sync_pre): reliable TCP-coordinator rendezvous
    // in place of the UC ack exchange that wedges on a lost completion. This is
    // the barrier where the observed recv-side wedge occurs.
    if (coordinator_ != nullptr && jaccl_confirmed_barrier_post()) {
      confirmed_coord_barrier(call_id, "post");
      return;
    }
    bool _prog = jaccl_progress_enabled();
    int num_peers = size_ - 1;
    int in_flight = 2 * num_peers;
    bool has_ack = !ack_connections_.empty();
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& sbuf = ack_send_buffers_[peer];
      if (has_ack) {
        // Dedicated ACK QP path (subgroups). ACK_RECV WRs are
        // pre-posted at QP setup and replenished by drain_acks.
        ack_connections_[peer].post_send(
            sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
      } else {
        // Original inline ack barrier on data QP (top-level group).
        // Post recv + send for THIS call. drain_acks polls data CQ.
        // This avoids per-collective overhead of polling a separate
        // ACK CQ on the master TP hot path. Safe because top-level
        // group's calls are uniform-size-class (all FRAME_SIZE buffers),
        // so the cross-call FIFO mismatch motivating the ACK QP fix
        // doesn't manifest.
        auto& rbuf = ack_recv_buffers_[peer];
        std::memset(rbuf.data<char>(), 0, rbuf.size());
        JACCL_DMA_BARRIER();
        connections_[peer].post_recv(
            rbuf, make_wr_id(call_id, ACK_RECV_WR, 0, peer));
        connections_[peer].post_send(
            sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
      }
    }
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] ack_sync_post POSTED rank=%d call_id=%u in_flight=%d has_ack_qp=%d\n",
          rank_,
          call_id,
          in_flight,
          has_ack ? 1 : 0);
      std::fflush(stderr);
    }
    drain_acks(call_id, in_flight);
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] ack_sync_post DRAINED rank=%d call_id=%u\n",
          rank_,
          call_id);
      std::fflush(stderr);
    }
  }

  void drain_acks(uint32_t call_id, int in_flight) {
    bool _prog = jaccl_progress_enabled();
    int _iters = 0;
    // Split in_flight into per-side accounting (always 2 * num_peers).
    int need_send = in_flight / 2;
    int need_recv = in_flight / 2;
    StallWatch _stall(need_send + need_recv);
    // soft-RC retransmit state (see jaccl_ack_retransmit_us). StallWatch above
    // stays as the final backstop: if retransmit hasn't restored progress by
    // jaccl_stall_timeout_us, it throws for a clean re-place.
    const bool _rtx_has_ack = !ack_connections_.empty();
    const uint64_t _rtx_us = jaccl_ack_retransmit_us();
    const int _rtx_max = jaccl_ack_retransmit_max();
    uint64_t _rtx_last = mach_absolute_time();
    int _rtx_metric = need_send + need_recv;
    int _rtx_count = 0;
    while (need_send > 0 || need_recv > 0) {
      // soft-RC: on a stall, retransmit the outstanding ACK work-requests.
      // Idempotent — a duplicate ACK_RECV is absorbed by cached_ack_recvs_ and
      // the extra local send completion just decrements need_send (which the
      // while-guard tolerates going <=0). Turns a silent UC drop into a
      // self-healing collective with no throw / no re-place.
      if (_rtx_us != 0) {
        const int _m = need_send + need_recv;
        if (_m != _rtx_metric) {
          _rtx_metric = _m;
          _rtx_last = mach_absolute_time();
        } else if (
            (_rtx_max <= 0 || _rtx_count < _rtx_max) &&
            mach_ticks_to_us(mach_absolute_time() - _rtx_last) > _rtx_us) {
          ++_rtx_count;
          for (int peer = 0; peer < size_; peer++) {
            if (peer == rank_) {
              continue;
            }
            auto& conn =
                _rtx_has_ack ? ack_connections_[peer] : connections_[peer];
            if (need_send > 0) {
              conn.post_send(
                  ack_send_buffers_[peer],
                  make_wr_id(call_id, ACK_SEND_WR, 0, peer));
            }
            if (need_recv > 0) {
              auto& rbuf = ack_recv_buffers_[peer];
              std::memset(rbuf.data<char>(), 0, rbuf.size());
              JACCL_DMA_BARRIER();
              conn.post_recv(
                  rbuf,
                  make_wr_id(_rtx_has_ack ? 0 : call_id, ACK_RECV_WR, 0, peer));
            }
          }
          std::fprintf(
              stderr,
              "[jaccl] soft-RC RETRANSMIT rank=%d call_id=%u need_send=%d need_recv=%d attempt=%d\n",
              rank_,
              call_id,
              need_send,
              need_recv,
              _rtx_count);
          std::fflush(stderr);
          _rtx_last = mach_absolute_time();
        }
      }
      _stall.tick(need_send + need_recv, "drain_acks", rank_, call_id);
      if (_prog) {
        ++_iters;
        if (_iters <= 4 || (_iters % 1000000) == 0) {
          std::fprintf(
              stderr,
              "[jaccl-prog] drain_acks POLL rank=%d call_id=%u iter=%d need_send=%d need_recv=%d cached_recvs=%d\n",
              rank_,
              call_id,
              _iters,
              need_send,
              need_recv,
              static_cast<int>(cached_ack_recvs_.size()));
          std::fflush(stderr);
        }
      }
      // Consume cached ACK_RECV completions before polling fresh CQEs.
      while (need_recv > 0 && !cached_ack_recvs_.empty()) {
        int peer = cached_ack_recvs_.back();
        cached_ack_recvs_.pop_back();
        if (_prog) {
          std::fprintf(
              stderr,
              "[jaccl-prog] drain_acks CACHED rank=%d call_id=%u type=ACK_RECV peer=%d need_recv=%d\n",
              rank_,
              call_id,
              peer,
              need_recv - 1);
          std::fflush(stderr);
        }
        need_recv--;
      }
      if (need_send == 0 && need_recv == 0) {
        break;
      }
      ibv_wc wc[16];
      // With dedicated ACK QPs (subgroups), poll only the ACK CQs.
      // Without (top-level group), the ack barrier rides the data CQ
      // alongside data completions — poll connections_ instead.
      int n = ack_connections_.empty()
          ? poll(connections_, 16, wc)
          : poll(ack_connections_, 16, wc);
      bool has_ack = !ack_connections_.empty();
      for (int i = 0; i < n; i++) {
        int wt = wr_id_work_type(wc[i].wr_id);
        if (wt == ACK_RECV_WR) {
          if (!has_ack) {
            // Top-level group: ACK_RECV WRs are per-call; filter stale.
            if (wr_id_call_id(wc[i].wr_id) != call_id) {
              continue;
            }
          }
          if (wc[i].status != IBV_WC_SUCCESS) {
            std::ostringstream msg;
            msg << "[jaccl] ack drain (recv) wc.status=" << wc[i].status
                << " wr_id=0x" << std::hex << wc[i].wr_id;
            throw std::runtime_error(msg.str());
          }
          int peer = wr_id_peer(wc[i].wr_id);
          if (has_ack) {
            // Replenish: post a fresh ACK_RECV on the dedicated ACK QP.
            // Sentinel call_id=0 — ACK_RECVs are call_id-agnostic.
            auto& rbuf = ack_recv_buffers_[peer];
            std::memset(rbuf.data<char>(), 0, rbuf.size());
            JACCL_DMA_BARRIER();
            ack_connections_[peer].post_recv(
                rbuf, make_wr_id(0, ACK_RECV_WR, 0, peer));
          }
          if (need_recv > 0) {
            if (_prog) {
              std::fprintf(
                  stderr,
                  "[jaccl-prog] drain_acks CQE rank=%d call_id=%u type=ACK_RECV peer=%d need_recv=%d (replenished)\n",
                  rank_,
                  call_id,
                  peer,
                  need_recv - 1);
              std::fflush(stderr);
            }
            need_recv--;
          } else {
            // Excess — peer is ahead. Cache for the next drain.
            cached_ack_recvs_.push_back(peer);
            if (_prog) {
              std::fprintf(
                  stderr,
                  "[jaccl-prog] drain_acks EXCESS rank=%d call_id=%u type=ACK_RECV peer=%d cached=%d\n",
                  rank_,
                  call_id,
                  peer,
                  static_cast<int>(cached_ack_recvs_.size()));
              std::fflush(stderr);
            }
          }
        } else if (wt == ACK_SEND_WR) {
          if (wr_id_call_id(wc[i].wr_id) != call_id) {
            continue;
          }
          if (wc[i].status != IBV_WC_SUCCESS) {
            std::ostringstream msg;
            msg << "[jaccl] ack drain (send) wc.status=" << wc[i].status
                << " wr_id=0x" << std::hex << wc[i].wr_id;
            throw std::runtime_error(msg.str());
          }
          if (_prog) {
            std::fprintf(
                stderr,
                "[jaccl-prog] drain_acks CQE rank=%d call_id=%u type=ACK_SEND need_send=%d\n",
                rank_,
                call_id,
                need_send - 1);
            std::fflush(stderr);
          }
          need_send--;
        } else {
          // Leftover non-ack completion (data send/recv). Don't touch
          // in_flight or buffers.
          continue;
        }
      }
    }
  }

  void send_to(uint32_t call_id, int sz, int rank, int buff) {
    connections_[rank].post_send(
        send_buffer(sz, buff), make_wr_id(call_id, SEND_WR, buff, rank));
  }

  // Zero the recv buffer before posting it. Buffer slots are reused
  // across consecutive collectives; if DMA fails to fully overwrite the
  // slot, the reader gets stale bytes. Pre-zeroing means we read zeros
  // if the DMA never lands, which upper layers can detect/route. The DSB
  // after memset ensures the zero is visible to the NIC before it
  // accepts a matching send.
  void zero_recv_buffer(SharedBuffer& buf) {
    std::memset(buf.data<char>(), 0, buf.size());
    JACCL_DMA_BARRIER();
  }

  void recv_from(uint32_t call_id, int sz, int rank, int buff) {
    auto& recv_buf = recv_buffer(sz, buff, rank);
    zero_recv_buffer(recv_buf);
    connections_[rank].post_recv(
        recv_buf, make_wr_id(call_id, RECV_WR, buff, rank));
  }

  SharedBuffer& send_buffer(int sz, int buff) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank_];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int rank) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank];
  }

  void post_send_all(uint32_t call_id, int sz, int buff) {
    auto& b = send_buffer(sz, buff);
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_send(b, make_wr_id(call_id, SEND_WR, buff, i));
    }
  }

  void post_recv_all(uint32_t call_id, int sz, int buff) {
    int b = sz * NUM_BUFFERS * size_ + buff * size_;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto& recv_buf = buffers_[b + i];
      zero_recv_buffer(recv_buf);
      connections_[i].post_recv(
          recv_buf, make_wr_id(call_id, RECV_WR, buff, i));
    }
  }

  int rank_;
  int size_;
  std::span<Connection> connections_;
  // Dedicated per-peer ACK connections — separate PD/CQ/QP from data
  // connections so the ack barrier's pre-posted ACK_RECV doesn't sit
  // at the head of the data recv FIFO. Empty for top-level groups
  // (those use the original inline ack on data QP).
  std::span<Connection> ack_connections_;
  // ROOT-CAUSE FIX (2026-07-17): dedicated QP for the jaccl-v2 reliable-ARQ
  // optimistic standing pool (POOL_RECV_WR). Same rationale as
  // ack_connections_ above -- previously shared connections_ with raw
  // send()/recv() (used by exo's Pipeline-Parallel p2p handoff), whose
  // differently-sized buffers collided with the pool's uniform size class
  // and threw IBV_WC_LOC_LEN_ERR, corrupting both paths' QP state. Empty
  // when the reliable-optimistic path is disabled (nothing to isolate).
  std::span<Connection> pool_connections_;
  // Dedicated QP for send()/recv()'s p2p_retry_exchange (design doc
  // Section 37 Phase 1). Same isolation rationale as ack_connections_/
  // pool_connections_ above.
  std::span<Connection> p2p_retry_connections_;
  // One send buffer slot per peer (rotated through synchronously --
  // p2p_retry_send_bitmask drains each frame's completion before reusing
  // it) and a P2P_RETRY_NUM_SLOTS-deep standing recv pool per peer,
  // flattened as peer * P2P_RETRY_NUM_SLOTS + slot (see
  // post_p2p_retry_recvs).
  std::span<SharedBuffer> p2p_retry_send_buffers_;
  std::span<SharedBuffer> p2p_retry_recv_buffers_;
  std::span<SharedBuffer> buffers_;
  std::span<SharedBuffer> ack_send_buffers_;
  std::span<SharedBuffer> ack_recv_buffers_;
  // Software queue of ACK_RECV completions that arrived early (peer ran
  // ahead). drain_acks pulls from here first before polling the CQ.
  // Element = peer index. drain_acks already replenished the recv WR.
  std::vector<int> cached_ack_recvs_;
  // Per-(peer, slot) outstanding-send tracking for p2p_retry_send_bitmask's
  // rotating send-buffer pool (see that method's own comment for why a
  // single shared per-peer send slot was unsafe). Sized [MESH_MAX_PEERS]
  // [P2P_RETRY_NUM_SLOTS] statically -- MESH_MAX_PEERS is this codebase's
  // existing hard cap (see mesh_impl.h's own top-of-file constant), so no
  // dynamic sizing/allocation is needed here, mirroring send_seq_/
  // recv_seq_'s own fixed-size array style below.
  bool p2p_retry_send_outstanding_[MESH_MAX_PEERS][P2P_RETRY_NUM_SLOTS] = {};


  // 2026-08-08, real production incident fix: send()/recv()'s on-wire
  // header (see post_chunk/consume_recv below) previously carried ONLY
  // the within-call chunk index `c` -- there is no field anywhere on the
  // wire identifying WHICH logical call this data belongs to. Combined
  // with send/recv buffer slots being a small, reused pool
  // (NUM_BUFFERS=8) recycled across every call for the life of the
  // process, a stale message from jaccl's own retry/retransmit layer
  // (an orphaned retransmit whose original recv already completed and
  // moved on, per this fix's own incident writeup) can land in a buffer
  // slot the CURRENT call has since claimed and be silently accepted as
  // belonging to it -- reproduced live: exo's chunk-drive protocol
  // received an 8-byte payload from a call ~700 sends and ~3m39s in the
  // past, well after both ranks had moved on, with zero detection at
  // this layer (exo's OWN application-level advance_seq tripwire is
  // what actually caught it).
  //
  // Fix: a per-(peer, direction) 16-bit sequence counter, incremented
  // once per send()/recv() call to/from that peer -- NOT call_id
  // (next_call_id() is a per-PROCESS global counter; sender's and
  // receiver's call_id values for the "same" logical transfer are
  // independent counters in different processes and do not agree, so
  // call_id cannot be validated on receive). A per-peer send/recv
  // sequence, by construction, increments symmetrically on both sides
  // of one ordered point-to-point channel -- the sender's Nth send() to
  // peer P and peer P's Nth recv() from this rank are always the same
  // logical transfer. Packed into the existing 4-byte header's upper 16
  // bits (`hdr = (seq & 0xFFFF) << 16 | c`) -- the chunk-index `c`
  // itself never exceeds a small buffer-count per call, so 16 bits is
  // ample headroom; this preserves the exact header size, buffer
  // size-class math, and chunk-count calculations send()/recv() already
  // use, keeping the fix to the header-pack/unpack call sites only.
  uint16_t send_seq_[2] = {0, 0}; // indexed by dst rank
  uint16_t recv_seq_[2] = {0, 0}; // indexed by src rank

  // Section 74 probe counters. send() and recv() share ONE completion
  // queue per QP (rdma.cpp:171-172 set init_attr.send_cq ==
  // init_attr.recv_cq), and a CQE is consumed exactly once by whichever
  // poll loop reaps it first. These count the cross-consumptions: a
  // SUCCESSFUL completion of the wrong work_type reaching a loop that
  // then discards it. On UC a genuine wire drop yields NO CQE at all, so
  // status==SUCCESS with the wrong type means the transfer really
  // happened and the notification was thrown away -- which presents to
  // the other side as a lost frame and triggers the expensive retransmit
  // path.
  //
  // Plain ints, NOT std::atomic: an atomic member deletes MeshImpl's
  // implicit copy-assignment, and reconnect_fresh() does `mesh_ =
  // MeshImpl(...)` (mesh.cpp), so an atomic here fails to compile. These
  // are diagnostic counters on a per-QP path that is not concurrently
  // entered by two threads for the same peer, so a plain int is both
  // sufficient and the only option that preserves assignability.
  mutable int _xconsume_recv_in_send{0};
  mutable int _xconsume_send_in_recv{0};

 public:
  // Section 68 diagnostic. Exposes the two sequence counters so the
  // reconnect path can log them AT THE RENDEZVOUS, on both ranks.
  //
  // Why this and not more symptom logging: a constant off-by-one between
  // send_seq_ and the peer's recv_seq_ makes every first chunk look lost
  // (`peer_got_count=0/1`), which costs a full 500ms retransmit quiet
  // period per decode token. The leading theory is that
  // `reconnect_fresh()` rebuilds MeshImpl -- zeroing both counters --
  // on only the rank that faulted, while the peer keeps counting.
  //
  // But that theory predicts an offset of "whatever the peer's counter
  // happened to read", i.e. an arbitrary N. The measured offset is
  // exactly 1, every time, in both directions. So either the reset is
  // symmetric and the real offset comes from a single call counted on
  // one side only (e.g. the in-flight send that triggered the fault), or
  // the mechanism is not reconnect at all. Logging the actual counter
  // values at the barrier distinguishes those directly, rather than
  // testing a predicted symptom -- the failure mode that let four
  // previous hypotheses in this campaign survive testing while being
  // wrong.
  uint16_t send_seq_for(int peer) const {
    return (peer >= 0 && peer < 2) ? send_seq_[peer] : 0;
  }
  uint16_t recv_seq_for(int peer) const {
    return (peer >= 0 && peer < 2) ? recv_seq_[peer] : 0;
  }

 private:

  // ── reliable_all_reduce v2 (optimistic) state ──
  // One-collective lookahead: messages whose header call_id == current+1
  // (peer exited optimistically and ran ahead). Applied on entry to that call.
  struct V2Stash {
    uint32_t call_id = 0;
    bool has_status = false;
    std::vector<uint8_t> peer_got;
    std::vector<std::pair<uint32_t, std::vector<char>>> chunks; // (seq, bytes)
  };
  V2Stash v2_stash_;
  bool v2_pool_posted_ = false;
  int v2_pool_sz_ = 0; // size class of the standing pool recv buffers
  // Previous call's retransmit-service info. Valid only when the previous
  // call was small (optimistic exit): its parity send buffers still hold the
  // exact wire bytes and can be re-posted verbatim to serve a stuck peer.
  uint32_t v2_prev_call_ = 0;
  int v2_prev_num_chunks_ = 0;
  int v2_prev_sz_ = 0;
  bool v2_prev_small_ = false;
  // Chunks of the previous call the peer still needs (status-driven), served
  // opportunistically from any later call's poll loop as slots free up.
  std::vector<uint8_t> v2_prev_want_;
  // Outstanding send WRs per send-buffer slot. A slot may only be rewritten
  // once its previous WR has completed (the NIC reads the buffer at transmit
  // time). CQEs from any call decrement by wr_id buff index.
  int v2_send_outstanding_[NUM_BUFFERS] = {0};
  // Non-owning pointer to the top-level group's reliable TCP coordinator, used
  // by the confirmed (ack-of-ack) barrier. nullptr on subgroups (no coordinator)
  // and when the confirmed barrier is disabled. Set via set_coordinator().
  SideChannel* coordinator_ = nullptr;
  // Dedicated p2p retry channel for send()/recv() (2026-07-17). Isolated
  // from coordinator_ -- see p2p_channel_ member comment in mesh.h for the
  // collision this fixes. nullptr on subgroups (send()/recv() only run on
  // exo PP's top-level 2-rank group, which always gets one). Set via
  // set_p2p_channel().
  SideChannel* p2p_channel_ = nullptr;
};

} // namespace jaccl
