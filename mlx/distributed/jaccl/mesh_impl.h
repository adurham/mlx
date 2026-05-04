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
      std::vector<SharedBuffer>& buffers)
      : rank_(rank), size_(size), connections_(conns), buffers_(buffers) {}

  MeshImpl() : rank_(0), size_(1) {}

  template <typename T, typename ReduceOp>
  void
  all_reduce(const T* in_ptr, T* out_ptr, int64_t size, ReduceOp reduce_op) {
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
      post_recv_all(sz, buff);
      auto& sbuf = send_buffer(sz, buff);
      zero_pool_buffer(sbuf);
      std::copy(
          data + read_offset,
          data + std::min(read_offset + N, total),
          sbuf.begin<T>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
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
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            auto& sbuf = send_buffer(sz, buff);
            zero_pool_buffer(sbuf);
            std::copy(
                data + read_offset,
                data + std::min(read_offset + N, total),
                sbuf.begin<T>());
            post_send_all(sz, buff);

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
            recv_from(sz, r, buff);
            in_flight++;
          }
        }
        completed_recv_begin[r] = s;
      }
    }
  }

  void all_gather(const char* in_ptr, char* out_ptr, int64_t n_bytes) {
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
      post_recv_all(sz, buff);
      auto& sbuf = send_buffer(sz, buff);
      zero_pool_buffer(sbuf);
      std::copy(
          our_data + read_offset,
          our_data + std::min(read_offset + N, total),
          sbuf.begin<char>());
      post_send_all(sz, buff);

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
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        // Send completed. If all sends completed then send the next chunk.
        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            auto& sbuf = send_buffer(sz, buff);
            zero_pool_buffer(sbuf);
            std::copy(
                our_data + read_offset,
                our_data + std::min(read_offset + N, total),
                sbuf.begin<char>());
            post_send_all(sz, buff);

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
            recv_from(sz, rank, buff);
            in_flight++;
          }
        }
      }
    }
  }

  void send(const char* in_ptr, int64_t n_bytes, int dst) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < n_bytes && buff < PIPELINE) {
      auto& sbuf = send_buffer(sz, buff);
      zero_pool_buffer(sbuf);
      std::copy(
          in_ptr + read_offset,
          in_ptr + std::min(read_offset + N, n_bytes),
          sbuf.begin<char>());
      send_to(sz, dst, buff);

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
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (read_offset < n_bytes) {
          auto& sbuf = send_buffer(sz, buff);
          zero_pool_buffer(sbuf);
          std::copy(
              in_ptr + read_offset,
              in_ptr + std::min(read_offset + N, n_bytes),
              sbuf.begin<char>());
          send_to(sz, dst, buff);

          read_offset += N;
          in_flight++;
        }
      }
    }
  }

  void recv(char* out_ptr, int64_t n_bytes, int src) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t write_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(sz, src, buff);

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
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

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
          recv_from(sz, src, buff);

          in_flight++;
        }
      }
    }
  }

 private:
  void send_to(int sz, int rank, int buff) {
    connections_[rank].post_send(
        send_buffer(sz, buff), SEND_WR << 16 | buff << 8 | rank);
  }

  // Zero a pool buffer before reuse. JACCL's buffer pool indexes by
  // (sz, buff, peer) and reuses slots across consecutive collectives.
  // The NIC always transfers `entry.length = buf.size()` bytes per
  // SEND (the SGE length, not the payload length), so a small message
  // (4 bytes for the count gather) leaves the rest of the buffer
  // populated with stale bytes from whatever last filled this slot —
  // typically bf16 from a prior small all_reduce. Without pre-zeroing,
  // those stale bytes propagate over the wire and we end up reading
  // them back on the receive side, producing the corruption-guard
  // fingerprint (1B+ int32 values, 0x3E…/0x3F… bf16 bit patterns)
  // when c=2 + MTP brings sustained mixed-bucket-0 traffic.
  // Zero-fill on every reuse breaks that bleed-through; the DSB
  // ensures the zeros are visible before the NIC reads/writes the slot.
  void zero_pool_buffer(SharedBuffer& buf) {
    std::memset(buf.data<char>(), 0, buf.size());
    JACCL_DMA_BARRIER();
  }

  void zero_recv_buffer(SharedBuffer& buf) {
    zero_pool_buffer(buf);
  }

  void recv_from(int sz, int rank, int buff) {
    auto& recv_buf = recv_buffer(sz, buff, rank);
    zero_recv_buffer(recv_buf);
    connections_[rank].post_recv(
        recv_buf, RECV_WR << 16 | buff << 8 | rank);
  }

  SharedBuffer& send_buffer(int sz, int buff) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank_];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int rank) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank];
  }

  void post_send_all(int sz, int buff) {
    auto& b = send_buffer(sz, buff);
    int wr_id = SEND_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_send(b, wr_id | i);
    }
  }

  void post_recv_all(int sz, int buff) {
    int b = sz * NUM_BUFFERS * size_ + buff * size_;
    int wr_id = RECV_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto& recv_buf = buffers_[b + i];
      zero_recv_buffer(recv_buf);
      connections_[i].post_recv(recv_buf, wr_id | i);
    }
  }

  int rank_;
  int size_;
  std::span<Connection> connections_;
  std::span<SharedBuffer> buffers_;
};

} // namespace mlx::core::distributed::jaccl
