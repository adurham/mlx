// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <mach/mach_time.h>

#include "mlx/distributed/jaccl/utils.h"

constexpr int MESH_MAX_PEERS = 8;

namespace mlx::core::distributed::jaccl {

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

  template <typename T, typename ReduceOp>
  void all_reduce(
      uint32_t call_id,
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      ReduceOp reduce_op) {
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

    // Start-of-lambda cross-rank barrier on the dedicated ACK QP.
    // Confirms peer has entered THIS call before we post our first
    // data send. Without it, the inter-lambda window is an empty
    // recv-queue on UC -> silent drop -> permanent wedge in the
    // in_flight>0 poll. See ack_sync_pre doc above. The pre-posted
    // ACK_RECV pool (post_ack_recvs at MeshGroup ctor) and the
    // sentinel-call_id replenish path in drain_acks keep the ACK QP
    // recv queue full across lambdas, so ack_sync_pre always finds
    // a posted recv on peer's side.
    ack_sync_pre(call_id);

    // Prefill the pipeline
    int buff = 0;
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

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    int _poll_iters = 0;
    // Instrumentation locals — zero-cost when JACCL_POLL_INSTRUMENT off.
    bool _instr = jaccl_poll_instrument_enabled();
    uint64_t _instr_t0 = _instr ? mach_absolute_time() : 0;
    uint64_t _instr_total_in_poll_ticks = 0;
    uint64_t _instr_max_single_poll_ticks = 0;
    uint64_t _instr_iters_with_cqes = 0;
    while (in_flight > 0) {
      ++_poll_iters;
      if (_prog) {
        // Log only the first few + every 1M iterations so we don't
        // drown the wedge in noise but still see progress when stuck.
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
      // Poll the hardware for completions.
      //
      // If a send was completed mark how many completions we have received
      // for that buffer. If we have sent the buffer to all peers we can
      // reuse the buffer so copy the next chunk of data and send it to all.
      //
      // If a receive is completed then advance the pointer of completed
      // receives.
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
        // Diagnostic: catch any non-success completion or any RECV
        // whose byte_len doesn't match the buffer size we posted.
        // If UC is silently truncating a peer's foreign-collective
        // send into our recv WR (the suspected mechanism behind the
        // bf16-bytes-in-int-gather corruption), this fires.
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
        // decrement in_flight (it does not belong to ours) and do not
        // touch the buffer it points at — that buffer is owned by a
        // call that has already returned, and reading it produces
        // exactly the cross-call corruption this scheme exists to
        // prevent.
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

      // Process the completed recv
      //
      // For each rank we have a range of completed recv defined by a begin
      // and end inclusive and exlusive in standard C++ fashion.
      //
      // When there is an unprocessed receive we first check if we have
      // finished sending the write location. If so then we reduce in-place
      // and then check if there is more to be received and post a recv.
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
    ack_sync_pre(call_id);

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

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        // Diagnostic: see all_reduce equivalent. Detect UC truncation
        // / status errors that the prior code silently absorbed.
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

        // Send completed. If all sends completed then send the next chunk.
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

        // Recv completed. If we have more chunks then post another recv.
        else if (work_type == RECV_WR) {
          // Ensure the NIC's DMA writes to recv_buffer are visible to
          // the CPU before we std::copy out of it.
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
    ack_sync_pre(call_id);

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
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed and we have more data to send then go ahead
      // and send them.
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
    ack_sync_pre(call_id);

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(call_id, sz, src, buff);

      in_flight++;
      buff++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
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
  // drain_acks rate, the pool absorbs the burst. drain_acks
  // replenishes one ACK_RECV per consumption, keeping steady-state
  // depth at ACK_RECV_POOL. Without enough depth, peer's ack_send
  // can arrive on a QP with no posted recv WR → UC drops → wedge.
  //
  // c=2 with master + coord groups running on separate CPU encoder
  // threads can cause one rank's coord lambdas to race well ahead of
  // the other rank's (e.g. one rank's master is busy with model
  // forwards while the other's master is idle, freeing the encoder
  // thread to dispatch coord lambdas at full rate). Empirically,
  // pool=16 was insufficient on c=2 trial 3 — wedge appeared after
  // stream1 hit EOS and timing diverged further. 64 is sized for the
  // observed cross-rank coord-lambda lead.
  static constexpr int ACK_RECV_POOL = 64;

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
      // Post `ACK_RECV_POOL` recv WRs on the dedicated ACK QP. Each
      // shares the single per-peer ack_recv_buffer (ACK payload is
      // a single sentinel byte; we only need to know "an ack
      // arrived" — the buffer contents don't matter).
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
  // Posts a small ack send + ack recv to every peer over the data
  // QP and waits for both completions per peer.
  //
  // Two calls per lambda:
  //   ack_sync_pre(): called BEFORE the data prefill posts. Confirms
  //     peer has reached the same lambda boundary AND posted its
  //     ack_recv as the very first WR on its QP recv queue. So our
  //     ack_send arrives at peer's first recv (the ack one), not at
  //     a stale prior-lambda recv. Once this returns, both ranks are
  //     synchronized at lambda start.
  //   ack_sync_post(): called AFTER the data main loop. Confirms peer
  //     also drained its main loop. Without this, in_flight==0 only
  //     proves OUR side drained; peer might still be polling, and
  //     our next-lambda send could arrive at peer's still-posted
  //     prior-lambda recv WR (different sz → IBV_WC_LOC_LEN_ERR).
  //
  // CRITICAL: callers MUST post the per-peer ack_recv BEFORE any
  // other recvs in the lambda (so the ack is at the head of the
  // QP recv queue and matches peer's ack_send first). The pre/post
  // ack pair on each call is the ack_send followed by waiting on
  // both ack-side completions.
  //
  // Why this is needed even with single-CPU-stream pinning: in_flight
  // == 0 locally means our local NIC has reported completion for our
  // posts; UC delivers send-completion when transmitted, not when
  // peer received. So our lambda can return before peer's lambda has
  // reached its main-loop exit. The ack barrier closes that window.
  void ack_sync_pre(uint32_t call_id) {
    // Caller has already posted ack_recvs as the FIRST recv WRs.
    // Now post our ack_sends and wait for both completions per peer.
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
    bool _prog = jaccl_progress_enabled();
    int num_peers = size_ - 1;
    int in_flight = 2 * num_peers; // one send + one recv per peer
    bool has_ack = !ack_connections_.empty();
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& sbuf = ack_send_buffers_[peer];
      if (has_ack) {
        // Dedicated ACK QP path (subgroups). ACK_RECV WRs are
        // pre-posted at QP setup (post_ack_recvs) and replenished by
        // drain_acks. Posting only ACK_SEND here.
        ack_connections_[peer].post_send(
            sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
      } else {
        // Original inline ack barrier on data QP (top-level group).
        // Post recv + send for THIS call. drain_acks polls data CQ.
        // This avoids the per-collective overhead of polling a
        // separate ACK CQ on the master TP hot path. Safe because
        // top-level group's calls are uniform-size-class (all
        // FRAME_SIZE buffers), so the cross-call FIFO mismatch
        // motivating the ACK QP fix doesn't manifest.
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
    // Per-peer accounting for what THIS call's barrier requires.
    // Each call needs 1 own ACK_SEND completion and 1 ACK_RECV
    // (peer's send arriving) per remote peer. Track them separately
    // so we don't accidentally over-consume ACK_RECVs that belong to
    // future calls' barriers when peer is running ahead.
    // in_flight from callers is always 2 * num_peers (one send + one
    // recv per peer); split it.
    int need_send = in_flight / 2;
    int need_recv = in_flight / 2;
    while (need_send > 0 || need_recv > 0) {
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
      // Use up any ACK_RECV completions cached by prior drains
      // before polling fresh CQEs.
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
      // Poll the relevant CQs. With dedicated ACK QPs (subgroups),
      // poll only the ACK CQs. Without (top-level group, original
      // path), the ack barrier rides the data CQ alongside data
      // completions — poll connections_ instead.
      int n = ack_connections_.empty()
          ? poll(connections_, 16, wc)
          : poll(ack_connections_, 16, wc);
      bool has_ack = !ack_connections_.empty();
      for (int i = 0; i < n; i++) {
        int wt = wr_id_work_type(wc[i].wr_id);
        if (wt == ACK_RECV_WR) {
          // With dedicated ACK QP: ACK_RECVs are call_id-agnostic
          // (sentinel call_id=0 from pre-post pool). Replenish a
          // fresh recv on the ACK QP after each consumption.
          // Without (top-level group): ACK_RECV WRs are per-call
          // (posted with this call's call_id in ack_sync_post),
          // matched once, no replenish — and stale ones from prior
          // calls must be filtered out.
          if (!has_ack) {
            // Filter stale: only this call's ack_recv counts.
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
            // Replenish: post a fresh ACK_RECV on the ACK QP for the
            // next barrier. Sentinel call_id=0.
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
          // ACK_SEND completions DO carry the call_id we used to post.
          // Filter out stale ones from prior calls.
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
          // Leftover non-ack completion (data send/recv from this or
          // a stale call). Don't touch in_flight or buffers.
          continue;
        }
      }
    }
  }

  // Legacy single-call wrapper. Kept for callers that haven't been
  // migrated to the post_ack_recvs + ack_sync_pre + ack_sync_post
  // pattern.
  void ack_sync(uint32_t call_id) {
    int num_peers = size_ - 1;
    int in_flight = 2 * num_peers; // one send + one recv per peer
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& sbuf = ack_send_buffers_[peer];
      auto& rbuf = ack_recv_buffers_[peer];
      // Zero before post — defends against any stale recv-side bytes
      // (same rationale as zero_recv_buffer in the data path).
      std::memset(rbuf.data<char>(), 0, rbuf.size());
      JACCL_DMA_BARRIER();
      connections_[peer].post_recv(
          rbuf, make_wr_id(call_id, ACK_RECV_WR, 0, peer));
      connections_[peer].post_send(
          sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
    }
    while (in_flight > 0) {
      ibv_wc wc[16];
      int n = poll(connections_, 16, wc);
      for (int i = 0; i < n; i++) {
        // Skip stale completions from prior calls; never decrement
        // in_flight or touch buffers for those.
        if (wr_id_call_id(wc[i].wr_id) != call_id) {
          continue;
        }
        int wt = wr_id_work_type(wc[i].wr_id);
        if (wt != ACK_SEND_WR && wt != ACK_RECV_WR) {
          // Leftover data completion from this same call — drain
          // without touching in_flight (the data side already
          // accounted for it).
          continue;
        }
        if (wc[i].status != IBV_WC_SUCCESS) {
          std::ostringstream msg;
          msg << "[jaccl] ack_sync wc.status=" << wc[i].status
              << " wr_id=0x" << std::hex << wc[i].wr_id;
          throw std::runtime_error(msg.str());
        }
        in_flight--;
      }
    }
  }

  void send_to(uint32_t call_id, int sz, int rank, int buff) {
    connections_[rank].post_send(
        send_buffer(sz, buff), make_wr_id(call_id, SEND_WR, buff, rank));
  }

  // Zero the recv buffer before posting it. JACCL's buffer pool indexes
  // by (sz, buff, peer) and reuses slots across consecutive collectives.
  // If the next NIC DMA-write fails to fully overwrite the slot — which
  // is the symptom on c=2 + MTP after the DSB barriers landed (the recv
  // buffer ends up holding bf16 bit patterns from a prior small
  // all_reduce, and the corruption guard reads them back as integer
  // garbage in the billions) — the reader gets stale bytes instead of
  // fresh data. Pre-zeroing means we read zeros if the DMA never lands,
  // which the upper layers can detect/route. The DSB after memset
  // ensures the zero is visible to the NIC before it accepts a
  // matching send.
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
  // at the head of the data recv FIFO. See MeshGroup::initialize.
  std::span<Connection> ack_connections_;
  std::span<SharedBuffer> buffers_;
  std::span<SharedBuffer> ack_send_buffers_;
  std::span<SharedBuffer> ack_recv_buffers_;
  // Software queue of ACK_RECV completions that arrived early (peer
  // ran ahead and pre-emptively sent ack for a future call). drain_acks
  // pulls from here first before polling the CQ. Element = peer index
  // of the cached completion. drain_acks already replenished the recv
  // WR for these — the buffer is consumed; we just owe the future call
  // a "barrier passed" signal.
  std::vector<int> cached_ack_recvs_;
};

} // namespace mlx::core::distributed::jaccl
