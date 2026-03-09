// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

#include <chrono>
#include <iostream>

namespace mlx::core::distributed::jaccl {

// Ensure stderr is unbuffered so RDMA diagnostic logs are visible
// even when stderr is redirected to a file.
static bool _stderr_unbuffered = [] {
  setvbuf(stderr, nullptr, _IONBF, 0);
  return true;
}();

// Gate verbose RDMA tracing behind EXO_TRACING_ENABLED=true
static bool _jaccl_trace = [] {
  const char* v = std::getenv("EXO_TRACING_ENABLED");
  return v && (std::string(v) == "true" || std::string(v) == "1");
}();

#define JACCL_TRACE(...) do { if (_jaccl_trace) fprintf(stderr, __VA_ARGS__); } while(0)

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const char* coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      side_channel_(rank_, size_, coordinator_addr),
      connections_(create_connections(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);

  // Create the mesh implementation object
  mesh_ = MeshImpl(rank_, size_, connections_, buffers_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);
}

void MeshGroup::initialize() {
  // Create the queue pairs
  for (auto& conn : connections_) {
    if (conn.ctx == nullptr) {
      continue;
    }
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  // NOTE: allocate_buffers() moved AFTER QP transitions.
  // Registering memory regions to multiple PDs before RTR caused EINVAL
  // on Apple's RDMA stack when multiple devices are open simultaneously.

  // First init all connections
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier so
  // that all peers have initialized their connections before attempting to
  // transition to RTS.
  std::vector<Destination> info;
  for (auto& conn : connections_) {
    info.emplace_back(conn.info());
  }
  auto all_infos = side_channel_.all_gather(info);

  // Transition queue pairs to RTS
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_info = all_infos[peer][rank_];
    std::cerr << "[jaccl-diag] rank=" << rank_ << " transitioning peer=" << peer
              << " (size=" << size_ << ")" << std::endl;
    connections_[peer].queue_pair_rtr(peer_info);
    connections_[peer].queue_pair_rts();
  }

  // Allocate and register buffers after QP transitions are complete.
  allocate_buffers();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();
  sub_send_buffers_.clear();
  sub_recv_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      // Ring buffers (1 for each direction)
      for (int j = 0; j < 2; j++) {
        ring_send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        ring_recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      // Sub-group buffers (for split() sub-groups).
      // Only needed for 3+ node meshes (hybrid TP+PP).
      // Skipped for size_ <= 2 to stay within per-PD MR limits.
      if (size_ > 2) {
        sub_send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        sub_recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        // This is our send buffer so register it with all pds so we can send
        // it to all connected devices.
        if (j == rank_) {
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .register_to_protection_domain(conn.protection_domain);
            }
          }
        }

        // This is the recv buffer from rank j so register it to rank j's
        // protection domain.
        else {
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
        }
      }

      // Ring buffers (see ring group for the logic below)
      // We register send buffers to both the right and the left.
      int left = (rank_ + size_ - 1) % size_;
      int right = (rank_ + 1) % size_;
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[right].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[right].protection_domain);

      // Sub-group buffers: register to ALL peer PDs so any sub-group
      // configuration can use them. Only allocated for 3+ node meshes.
      if (size_ > 2) {
        for (auto& conn : connections_) {
          if (conn.ctx != nullptr) {
            sub_send_buffers_[k * NUM_BUFFERS + i]
                .register_to_protection_domain(conn.protection_domain);
            sub_recv_buffers_[k * NUM_BUFFERS + i]
                .register_to_protection_domain(conn.protection_domain);
          }
        }
      }
    }
  }
}

void MeshGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{});
  });
}

void MeshGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
  });
}

void MeshGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{});
  });
}

void MeshGroup::all_gather(const array& input, array& output, Stream stream) {
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  size_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    auto t0 = std::chrono::steady_clock::now();
    mesh_.all_gather(in_ptr, out_ptr, n_bytes);
    auto t1 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    JACCL_TRACE(
        "[MeshGroup::all_gather R%d] %zu bytes in %.2fms\n",
        rank_,
        n_bytes,
        us / 1000.0);
  });
}

void MeshGroup::send(const array& input, int dst, Stream stream) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([data, n_bytes, dst, this]() {
    auto t0 = std::chrono::steady_clock::now();
    JACCL_TRACE(
        "[MeshGroup::send R%d] sending %lld bytes to dst=%d\n",
        rank_,
        (long long)n_bytes,
        dst);
    mesh_.send(data, n_bytes, dst);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    JACCL_TRACE(
        "[MeshGroup::send R%d] send complete to dst=%d in %.2fms\n",
        rank_,
        dst,
        us / 1000.0);
  });
}

void MeshGroup::recv(array& out, int src, Stream stream) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([data, n_bytes, src, this]() {
    auto t0 = std::chrono::steady_clock::now();
    JACCL_TRACE(
        "[MeshGroup::recv R%d] receiving %lld bytes from src=%d\n",
        rank_,
        (long long)n_bytes,
        src);
    mesh_.recv(data, n_bytes, src);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    JACCL_TRACE(
        "[MeshGroup::recv R%d] recv complete from src=%d in %.2fms\n",
        rank_,
        src,
        us / 1000.0);
  });
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  int64_t size = input.size();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, size, this, reduce_op]() {
    auto t0 = std::chrono::steady_clock::now();
    if (size_ > 2 &&
        ((std::is_same_v<T, bfloat16_t> && size > 65536) ||
         size >= 8 * 1024 * 1024 / sizeof(T))) {
      ring_.all_reduce<2>(in_ptr, out_ptr, size, 1, reduce_op);
    } else {
      mesh_.all_reduce(in_ptr, out_ptr, size, reduce_op);
    }
    auto t1 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    JACCL_TRACE(
        "[MeshGroup::all_reduce R%d] %lld elems (%lld bytes) in %.2fms\n",
        rank_,
        (long long)size,
        (long long)(size * sizeof(T)),
        us / 1000.0);
  });
}

// --- MeshGroup::split ---

std::shared_ptr<GroupImpl> MeshGroup::split(int color, int key) {
  // Gather (color, key, global_rank) from all peers via the side channel.
  // Using the side channel is safe here because split() is called once during
  // initialization, not on the hot path.
  struct SplitInfo {
    int color;
    int key;
    int global_rank;
  };
  SplitInfo my_info{color, key >= 0 ? key : rank_, rank_};

  // Exchange split info with all peers
  auto all_info = side_channel_.all_gather(my_info);

  // Collect members of our color, sorted by key (then by global rank for ties)
  std::vector<std::pair<int, int>> members; // (sort_key, global_rank)
  for (auto& info : all_info) {
    if (info.color == color) {
      members.emplace_back(info.key, info.global_rank);
    }
  }
  std::sort(members.begin(), members.end());

  std::vector<int> global_ranks;
  int sub_rank = -1;
  for (int i = 0; i < static_cast<int>(members.size()); i++) {
    global_ranks.push_back(members[i].second);
    if (members[i].second == rank_) {
      sub_rank = i;
    }
  }

  if (sub_rank < 0) {
    throw std::runtime_error("[jaccl] split: this rank not found in sub-group");
  }

  int sub_size = static_cast<int>(global_ranks.size());
  return std::make_shared<SubMeshGroup>(sub_rank, sub_size, std::move(global_ranks), this);
}

// --- SubMeshGroup ---

SubMeshGroup::SubMeshGroup(
    int sub_rank,
    int sub_size,
    std::vector<int> global_ranks,
    MeshGroup* parent)
    : sub_rank_(sub_rank),
      sub_size_(sub_size),
      global_ranks_(std::move(global_ranks)),
      parent_(parent) {
  // Build peer connection pointers (skip self)
  for (int i = 0; i < sub_size_; i++) {
    if (i == sub_rank_) {
      peer_connections_.push_back(nullptr);
    } else {
      peer_connections_.push_back(&parent_->connections_[global_ranks_[i]]);
    }
  }
  // No separate buffer allocation — we reuse parent's pre-registered buffers.
  // JACCL requires ibv_reg_mr before QP transitions to RTS; parent's buffers
  // satisfy this constraint.
}


void SubMeshGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{});
  });
}

void SubMeshGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
  });
}

void SubMeshGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{});
  });
}

template <typename T, typename ReduceOp>
void SubMeshGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {
  if (sub_size_ == 1) {
    // Single-node sub-group: just copy
    auto in_ptr = input.data<T>();
    auto out_ptr = output.data<T>();
    int64_t count = input.size();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, count]() {
      if (in_ptr != out_ptr) {
        std::memcpy(out_ptr, in_ptr, count * sizeof(T));
      }
    });
    return;
  }

  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  int64_t count = input.size();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, count, reduce_op, this]() {
    // Copy input to output (accumulator)
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, count * sizeof(T));
    }

    auto [sz, buffer_size] = buffer_size_from_message(count * sizeof(T));
    int64_t N = buffer_size / sizeof(T);
    constexpr int PIPELINE = 2;
    int64_t total = count;
    auto t_all_reduce_start = std::chrono::steady_clock::now();

    JACCL_TRACE(
        "[SubMeshGroup::all_reduce R%d] count=%lld sz=%d N=%lld total=%lld sub_size=%d\n",
        sub_rank_,
        (long long)count,
        sz,
        (long long)N,
        (long long)total,
        sub_size_);

    // Exchange with each peer and reduce
    for (int p = 0; p < sub_size_; p++) {
      if (p == sub_rank_) {
        continue;
      }
      Connection* peer = peer_connections_[p];

      JACCL_TRACE(
          "[SubMeshGroup::all_reduce R%d] exchanging with peer %d (global %d)\n",
          sub_rank_,
          p,
          global_ranks_[p]);

      int in_flight = 0;
      int64_t read_offset = 0;
      int completed_recv_begin = 0;
      int completed_recv_end = 0;

      // Prefill pipeline
      int buff = 0;
      while (read_offset < total && buff < PIPELINE) {
        peer->post_recv(
            recv_buffer(sz, buff), RECV_WR << 16 | buff << 8);
        std::copy(
            out_ptr + read_offset,
            out_ptr + std::min(read_offset + N, total),
            send_buffer(sz, buff).begin<T>());
        peer->post_send(
            send_buffer(sz, buff), SEND_WR << 16 | buff << 8);
        buff++;
        in_flight += 2;
        read_offset += N;
      }

      JACCL_TRACE(
          "[SubMeshGroup::all_reduce R%d] prefill done: in_flight=%d read_offset=%lld\n",
          sub_rank_,
          in_flight,
          (long long)read_offset);

      // Main loop
      auto stall_start = std::chrono::steady_clock::now();
      bool stall_logged = false;
      while (in_flight > 0) {
        ibv_wc wc[PIPELINE * 2];
        int n = peer->poll(PIPELINE * 2, wc);
        if (n > 0) {
          stall_start = std::chrono::steady_clock::now();
          stall_logged = false;
          JACCL_TRACE(
              "[SubMeshGroup::all_reduce R%d] poll returned %d completions, in_flight=%d\n",
              sub_rank_,
              n,
              in_flight);
        } else {
          auto now = std::chrono::steady_clock::now();
          auto stall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              now - stall_start).count();
          if (stall_ms >= 5000 && !stall_logged) {
            stall_logged = true;
            fprintf(
                stderr,
                "[SubMeshGroup::all_reduce R%d] STALL: no completions for %lldms, "
                "in_flight=%d, read_offset=%lld/%lld\n",
                sub_rank_,
                (long long)stall_ms,
                in_flight,
                (long long)read_offset,
                (long long)total);
          }
        }
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int b = (wc[i].wr_id >> 8) & 0xff;
          in_flight--;

          if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(
                stderr,
                "[SubMeshGroup::all_reduce R%d] ERROR: wc status=%d for wr_id=%llu\n",
                sub_rank_,
                wc[i].status,
                (unsigned long long)wc[i].wr_id);
          }

          if (work_type == SEND_WR && read_offset < total) {
            std::copy(
                out_ptr + read_offset,
                out_ptr + std::min(read_offset + N, total),
                send_buffer(sz, b).begin<T>());
            peer->post_send(
                send_buffer(sz, b), SEND_WR << 16 | b << 8);
            in_flight++;
            read_offset += N;
          } else if (work_type == RECV_WR) {
            completed_recv_end++;
          }
        }

        // Process completed receives
        int64_t w = static_cast<int64_t>(completed_recv_begin) * N;
        while (w < read_offset && completed_recv_end > completed_recv_begin) {
          int b = completed_recv_begin % PIPELINE;
          reduce_op(
              recv_buffer(sz, b).begin<T>(),
              out_ptr + w,
              std::min(N, total - w));
          w += N;
          completed_recv_begin++;
          if (w + (PIPELINE - 1) * N < total) {
            peer->post_recv(
                recv_buffer(sz, b), RECV_WR << 16 | b << 8);
            in_flight++;
          }
        }
      }
      JACCL_TRACE(
          "[SubMeshGroup::all_reduce R%d] peer %d exchange complete\n",
          sub_rank_,
          p);
    }
    auto t_all_reduce_end = std::chrono::steady_clock::now();
    auto all_reduce_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_all_reduce_end - t_all_reduce_start).count();
    JACCL_TRACE(
        "[SubMeshGroup::all_reduce R%d] complete %lld bytes in %.2fms\n",
        sub_rank_,
        (long long)(count * sizeof(T)),
        all_reduce_us / 1000.0);
  });
}

void SubMeshGroup::all_gather(
    const array& input,
    array& output,
    Stream stream) {
  if (sub_size_ == 1) {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes]() {
      std::memcpy(out_ptr, in_ptr, n_bytes);
    });
    return;
  }

  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  size_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    auto t_gather_start = std::chrono::steady_clock::now();
    // Copy our data to the appropriate position
    std::memcpy(out_ptr + sub_rank_ * n_bytes, in_ptr, n_bytes);
    char* our_data = out_ptr + sub_rank_ * n_bytes;

    auto [sz, N] = buffer_size_from_message(n_bytes);
    constexpr int PIPELINE = 2;
    int64_t total = static_cast<int64_t>(n_bytes);

    JACCL_TRACE(
        "[SubMeshGroup::all_gather R%d] n_bytes=%zu sz=%d N=%lld total=%lld sub_size=%d\n",
        sub_rank_,
        n_bytes,
        sz,
        (long long)N,
        (long long)total,
        sub_size_);

    // Exchange with each peer
    for (int p = 0; p < sub_size_; p++) {
      if (p == sub_rank_) {
        continue;
      }
      Connection* peer = peer_connections_[p];

      JACCL_TRACE(
          "[SubMeshGroup::all_gather R%d] exchanging with peer %d (global %d)\n",
          sub_rank_,
          p,
          global_ranks_[p]);

      int in_flight = 0;
      int64_t read_offset = 0;
      int64_t write_offset = 0;

      // Prefill pipeline
      int buff = 0;
      while (read_offset < total && buff < PIPELINE) {
        peer->post_recv(
            recv_buffer(sz, buff), RECV_WR << 16 | buff << 8);
        std::copy(
            our_data + read_offset,
            our_data + std::min(read_offset + static_cast<int64_t>(N), total),
            send_buffer(sz, buff).begin<char>());
        peer->post_send(
            send_buffer(sz, buff), SEND_WR << 16 | buff << 8);
        buff++;
        in_flight += 2;
        read_offset += N;
      }

      JACCL_TRACE(
          "[SubMeshGroup::all_gather R%d] prefill done: in_flight=%d read_offset=%lld\n",
          sub_rank_,
          in_flight,
          (long long)read_offset);

      // Main loop
      auto stall_start = std::chrono::steady_clock::now();
      bool stall_logged = false;
      while (in_flight > 0) {
        ibv_wc wc[PIPELINE * 2];
        int n = peer->poll(PIPELINE * 2, wc);
        if (n > 0) {
          stall_start = std::chrono::steady_clock::now();
          stall_logged = false;
          JACCL_TRACE(
              "[SubMeshGroup::all_gather R%d] poll returned %d completions, in_flight=%d\n",
              sub_rank_,
              n,
              in_flight);
        } else {
          auto now = std::chrono::steady_clock::now();
          auto stall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              now - stall_start).count();
          if (stall_ms >= 5000 && !stall_logged) {
            stall_logged = true;
            fprintf(
                stderr,
                "[SubMeshGroup::all_gather R%d] STALL: no completions for %lldms, "
                "in_flight=%d, read_offset=%lld/%lld\n",
                sub_rank_,
                (long long)stall_ms,
                in_flight,
                (long long)read_offset,
                (long long)total);
          }
        }
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int b = (wc[i].wr_id >> 8) & 0xff;
          in_flight--;

          if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(
                stderr,
                "[SubMeshGroup::all_gather R%d] ERROR: wc status=%d for wr_id=%llu\n",
                sub_rank_,
                wc[i].status,
                (unsigned long long)wc[i].wr_id);
          }

          if (work_type == SEND_WR && read_offset < total) {
            std::copy(
                our_data + read_offset,
                our_data + std::min(
                    read_offset + static_cast<int64_t>(N), total),
                send_buffer(sz, b).begin<char>());
            peer->post_send(
                send_buffer(sz, b), SEND_WR << 16 | b << 8);
            in_flight++;
            read_offset += N;
          } else if (work_type == RECV_WR) {
            std::copy(
                recv_buffer(sz, b).begin<char>(),
                recv_buffer(sz, b).begin<char>() +
                    std::min(
                        static_cast<int64_t>(N), total - write_offset),
                out_ptr + p * n_bytes + write_offset);
            write_offset += N;
            if (write_offset + (PIPELINE - 1) * N < total) {
              peer->post_recv(
                  recv_buffer(sz, b), RECV_WR << 16 | b << 8);
              in_flight++;
            }
          }
        }
      }
      JACCL_TRACE(
          "[SubMeshGroup::all_gather R%d] peer %d exchange complete\n",
          sub_rank_,
          p);
    }
    auto t_gather_end = std::chrono::steady_clock::now();
    auto gather_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_gather_end - t_gather_start).count();
    JACCL_TRACE(
        "[SubMeshGroup::all_gather R%d] complete %zu bytes in %.2fms\n",
        sub_rank_,
        n_bytes,
        gather_us / 1000.0);
  });
}

void SubMeshGroup::send(const array& input, int dst, Stream stream) {
  int global_dst = global_ranks_[dst];
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([data, n_bytes, global_dst, dst, this]() {
    auto t0 = std::chrono::steady_clock::now();
    JACCL_TRACE(
        "[SubMeshGroup::send R%d] sending %lld bytes to sub_dst=%d global_dst=%d\n",
        sub_rank_,
        (long long)n_bytes,
        dst,
        global_dst);
    parent_->mesh_.send(data, n_bytes, global_dst);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    JACCL_TRACE(
        "[SubMeshGroup::send R%d] send complete to global_dst=%d in %.2fms\n",
        sub_rank_,
        global_dst,
        us / 1000.0);
  });
}

void SubMeshGroup::recv(array& out, int src, Stream stream) {
  int global_src = global_ranks_[src];
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([data, n_bytes, global_src, src, this]() {
    auto t0 = std::chrono::steady_clock::now();
    JACCL_TRACE(
        "[SubMeshGroup::recv R%d] receiving %lld bytes from sub_src=%d global_src=%d\n",
        sub_rank_,
        (long long)n_bytes,
        src,
        global_src);
    parent_->mesh_.recv(data, n_bytes, global_src);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    JACCL_TRACE(
        "[SubMeshGroup::recv R%d] recv complete from global_src=%d in %.2fms\n",
        sub_rank_,
        global_src,
        us / 1000.0);
  });
}

} // namespace mlx::core::distributed::jaccl
