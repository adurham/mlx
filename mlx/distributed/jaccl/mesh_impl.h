// Copyright © 2026 Apple Inc.

#pragma once

#include <cstring>
#include <span>

#include "mlx/distributed/jaccl/utils.h"

constexpr int MESH_MAX_PEERS = 8;

namespace mlx::core::distributed::jaccl {

class MeshImpl {
 public:
  MeshImpl(
      int rank,
      int size,
      std::vector<Connection>& conns,
      std::vector<SharedBuffer>& buffers,
      std::vector<SharedBuffer>& ack_send_buffers,
      std::vector<SharedBuffer>& ack_recv_buffers)
      : rank_(rank),
        size_(size),
        connections_(conns),
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

    // Cross-rank start barrier with data_recv pre-posting:
    //   1. Post ack_recv FIRST (head of QP recv queue).
    //   2. Post ALL prefill data_recvs (in QP recv queue AFTER ack_recv).
    //   3. Post ack_send + wait for ack pair completion. Once this
    //      drains, peer has provably reached the same boundary AND
    //      posted its prefill data_recvs (peer also did 1+2 before
    //      sending its ack_send, so peer's ack_send→our ack_recv
    //      success implies peer's recvs are posted too).
    //   4. NOW post prefill data_sends. They'll arrive at peer's
    //      already-posted data_recvs (size match). No drop, no race.
    post_ack_recvs(call_id);
    int prefill_buffs = 0;
    {
      // Step 2: post all prefill recvs (without sends yet).
      int b = 0;
      int64_t ro = 0;
      while (ro < total && b < PIPELINE) {
        post_recv_all(call_id, sz, b);
        b++;
        ro += N;
      }
      prefill_buffs = b;
    }
    ack_sync_pre(call_id);
    // Step 4: post all prefill sends (data_recvs already posted on
    // both sides; safe to send now).
    {
      int b = 0;
      int64_t ro = 0;
      while (ro < total && b < prefill_buffs) {
        std::copy(
            data + ro,
            data + std::min(ro + N, total),
            send_buffer(sz, b).begin<T>());
        post_send_all(call_id, sz, b);
        b++;
        in_flight += 2 * num_peers;
        ro += N;
      }
      read_offset = ro;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
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
    ack_sync_post(call_id);
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

    // Cross-rank start barrier with data_recv pre-posting —
    // see all_reduce equivalent for the full rationale.
    post_ack_recvs(call_id);
    int prefill_buffs = 0;
    {
      int b = 0;
      int ro = 0;
      while (ro < total && b < PIPELINE) {
        post_recv_all(call_id, sz, b);
        b++;
        ro += N;
      }
      prefill_buffs = b;
    }
    ack_sync_pre(call_id);
    {
      int b = 0;
      int ro = 0;
      while (ro < total && b < prefill_buffs) {
        std::copy(
            our_data + ro,
            our_data + std::min(ro + N, total),
            send_buffer(sz, b).begin<char>());
        post_send_all(call_id, sz, b);
        b++;
        in_flight += 2 * num_peers;
        ro += N;
      }
      read_offset = ro;
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

    // Cross-rank start barrier — see all_reduce.
    post_ack_recvs(call_id);
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

    // Cross-rank start barrier — see all_reduce.
    post_ack_recvs(call_id);
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
      connections_[peer].post_send(
          sbuf, make_wr_id(call_id, ACK_SEND_WR, 0, peer));
    }
    drain_acks(call_id, in_flight);
  }

  void ack_sync_post(uint32_t call_id) {
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
    drain_acks(call_id, in_flight);
  }

  // Helper used by post_ack_recvs() to put ack_recv as the FIRST
  // recv WR on each peer's QP queue, before any data recv WRs. This
  // makes peer's ack_send hit our ack_recv first (FIFO matching).
  void post_ack_recvs(uint32_t call_id) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto& rbuf = ack_recv_buffers_[peer];
      std::memset(rbuf.data<char>(), 0, rbuf.size());
      JACCL_DMA_BARRIER();
      connections_[peer].post_recv(
          rbuf, make_wr_id(call_id, ACK_RECV_WR, 0, peer));
    }
  }

  void drain_acks(uint32_t call_id, int in_flight) {
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
          msg << "[jaccl] ack drain wc.status=" << wc[i].status
              << " wr_id=0x" << std::hex << wc[i].wr_id;
          throw std::runtime_error(msg.str());
        }
        in_flight--;
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
  std::span<SharedBuffer> buffers_;
  std::span<SharedBuffer> ack_send_buffers_;
  std::span<SharedBuffer> ack_recv_buffers_;
};

} // namespace mlx::core::distributed::jaccl
