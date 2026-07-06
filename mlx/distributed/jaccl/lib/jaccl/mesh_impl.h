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
inline int jaccl_ack_retransmit_max() {
  static const int v = [] {
    const char* e = std::getenv("MLX_JACCL_ACK_RETRANSMIT_MAX");
    return e ? std::atoi(e) : 40;
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
      std::vector<SharedBuffer>& buffers,
      std::vector<SharedBuffer>& ack_send_buffers,
      std::vector<SharedBuffer>& ack_recv_buffers)
      : rank_(rank),
        size_(size),
        connections_(conns),
        ack_connections_(ack_conns),
        buffers_(buffers),
        ack_send_buffers_(ack_send_buffers),
        ack_recv_buffers_(ack_recv_buffers) {}

  MeshImpl() : rank_(0), size_(1) {}

  // Wire up the reliable TCP coordinator (top-level group only) for the
  // confirmed (ack-of-ack) barrier. Non-owning; the SideChannel outlives this.
  void set_coordinator(SideChannel* coordinator) {
    coordinator_ = coordinator;
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

  template <typename T, typename ReduceOp>
  void all_reduce(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
    // Reliable ARQ data path (gated). Top-level 2-rank group only.
    if (coordinator_ != nullptr && size_ == 2 && jaccl_reliable_data_enabled()) {
      reliable_all_reduce<T>(call_id, in_ptr, out_ptr, size, reduce_op);
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
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;

    // Start-of-lambda cross-rank barrier. See ack_sync_pre doc above.
    if (jaccl_ack_sync_pre_enabled()) {
      ack_sync_pre(call_id);
    }

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < n_bytes && buff < PIPELINE) {
      std::copy(
          in_ptr + read_offset,
          in_ptr + std::min(read_offset + N, n_bytes),
          send_buffer(sz, buff).begin<char>());
      send_to(call_id, sz, dst, buff);

      buff++;
      read_offset += N;
      in_flight++;
    }

    // Main loop
    StallWatch _stall(in_flight);
    while (in_flight > 0) {
      _stall.tick(in_flight, "send", rank_, call_id);
      ibv_wc wc[WC_NUM];
      int n = connections_[dst].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        if (wr_id_call_id(wc[i].wr_id) != call_id) {
          continue;
        }
        int buff = wr_id_buff(wc[i].wr_id);

        in_flight--;

        if (read_offset < n_bytes) {
          std::copy(
              in_ptr + read_offset,
              in_ptr + std::min(read_offset + N, n_bytes),
              send_buffer(sz, buff).begin<char>());
          send_to(call_id, sz, dst, buff);

          read_offset += N;
          in_flight++;
        }
      }
    }
    ack_sync_post(call_id);
  }

  void recv(uint32_t call_id, char* out_ptr, int64_t n_bytes, int src) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t write_offset = 0;

    // Start-of-lambda cross-rank barrier. See ack_sync_pre doc above.
    if (jaccl_ack_sync_pre_enabled()) {
      ack_sync_pre(call_id);
    }

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(call_id, sz, src, buff);
      in_flight++;
      buff++;
    }

    // Main loop
    StallWatch _stall(in_flight);
    while (in_flight > 0) {
      _stall.tick(in_flight, "recv", rank_, call_id);
      ibv_wc wc[WC_NUM];
      int n = connections_[src].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        if (wr_id_call_id(wc[i].wr_id) != call_id) {
          continue;
        }
        int buff = wr_id_buff(wc[i].wr_id);

        in_flight--;

        // Ensure NIC DMA writes to recv_buffer are visible to the CPU.
        JACCL_DMA_BARRIER();
        std::copy(
            recv_buffer(sz, buff, src).begin<char>(),
            recv_buffer(sz, buff, src).begin<char>() +
                std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
            out_ptr + write_offset);
        write_offset += N;

        if (write_offset + (PIPELINE - 1) * N < n_bytes) {
          recv_from(call_id, sz, src, buff);
          in_flight++;
        }
      }
    }
    ack_sync_post(call_id);
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
  std::span<SharedBuffer> buffers_;
  std::span<SharedBuffer> ack_send_buffers_;
  std::span<SharedBuffer> ack_recv_buffers_;
  // Software queue of ACK_RECV completions that arrived early (peer ran
  // ahead). drain_acks pulls from here first before polling the CQ.
  // Element = peer index. drain_acks already replenished the recv WR.
  std::vector<int> cached_ack_recvs_;
  // Non-owning pointer to the top-level group's reliable TCP coordinator, used
  // by the confirmed (ack-of-ack) barrier. nullptr on subgroups (no coordinator)
  // and when the confirmed barrier is disabled. Set via set_coordinator().
  SideChannel* coordinator_ = nullptr;
};

} // namespace jaccl
