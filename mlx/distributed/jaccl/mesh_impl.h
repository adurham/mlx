// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

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
    while (in_flight > 0) {
      if (_prog) {
        ++_poll_iters;
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
      int n = poll(connections_, WC_NUM, wc);
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
  // 16 is comfortably above the per-collective ack rate observed at
  // c=2 MTP=1 (~50 collectives/sec from runner-coord on coord
  // subgroup) given typical per-call drain latency (<1ms).
  static constexpr int ACK_RECV_POOL = 16;

  void post_ack_recvs(uint32_t call_id) {
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
    // ACK_RECV WRs are pre-posted (post_ack_recvs at MeshGroup ctor,
    // re-posted by drain_acks after each consumption). Posting only
    // ACK_SEND here. Without pre-posting, the ack_sync_post recv was
    // posted in this same lambda — racing with the peer's ack_send
    // arrival. Under UC (IBV_QPT_UC, no retransmit) any send hitting
    // a QP without a posted recv WR is silently dropped: rank 0 ack
    // for call N could fire microseconds before rank 1 had posted
    // its call-N ack_recv → byte lost → rank 1's drain spins forever.
    // Pre-post + replenish ensures the recv WR is ALWAYS waiting.
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& sbuf = ack_send_buffers_[peer];
      ack_connections_[peer].post_send(
          sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
    }
    if (_prog) {
      std::fprintf(
          stderr,
          "[jaccl-prog] ack_sync_post POSTED rank=%d call_id=%u in_flight=%d\n",
          rank_,
          call_id,
          in_flight);
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
      // Poll only the ACK QPs' CQs. Data CQs are handled by the data
      // path's own poll loop (mesh_impl all_reduce, ring_impl, etc.).
      int n = poll(ack_connections_, 16, wc);
      for (int i = 0; i < n; i++) {
        int wt = wr_id_work_type(wc[i].wr_id);
        if (wt == ACK_RECV_WR) {
          // ACK_RECV completions are call_id-AGNOSTIC. The recv WRs
          // are pre-posted at QP setup time (with sentinel call_id=0)
          // and replenished here on each completion, to avoid the
          // race where peer's ACK_SEND arrives before our call-N
          // ACK_RECV is posted (UC drops the byte). recv WRs are
          // FIFO-matched on the dedicated ACK QP, so the firing one
          // is from peer's most recent unprocessed ACK_SEND.
          //
          // CRITICAL: we replenish a fresh ACK_RECV WR (so the pool
          // stays at depth ACK_RECV_POOL) but we only count the
          // completion against `need_recv` IF it's still > 0. Excess
          // CQEs get cached for the NEXT drain — they belong to the
          // next call's barrier, not this one. Without this caching,
          // an over-eager peer's pre-emptive ack_send for call N+1
          // would be consumed by this drain (call N), and call N+1's
          // drain would block waiting for an ack that's already gone.
          if (wc[i].status != IBV_WC_SUCCESS) {
            std::ostringstream msg;
            msg << "[jaccl] ack drain (recv) wc.status=" << wc[i].status
                << " wr_id=0x" << std::hex << wc[i].wr_id;
            throw std::runtime_error(msg.str());
          }
          int peer = wr_id_peer(wc[i].wr_id);
          // Replenish: post a fresh ACK_RECV on the ACK QP for the
          // next barrier. Sentinel call_id=0 since these are
          // call_id-agnostic.
          auto& rbuf = ack_recv_buffers_[peer];
          std::memset(rbuf.data<char>(), 0, rbuf.size());
          JACCL_DMA_BARRIER();
          ack_connections_[peer].post_recv(
              rbuf, make_wr_id(0, ACK_RECV_WR, 0, peer));
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
