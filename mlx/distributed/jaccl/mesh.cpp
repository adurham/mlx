// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

namespace mlx::core::distributed::jaccl {

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

  allocate_buffers();

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
    connections_[peer].queue_pair_rtr(peer_info);
    connections_[peer].queue_pair_rts();
  }
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();

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
  uint32_t call_id = next_call_id();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([call_id, in_ptr, out_ptr, n_bytes, this]() {
    std::lock_guard<std::mutex> guard(collective_mutex_);
    cross_rank_pre_barrier(call_id);
    mesh_.all_gather(call_id, in_ptr, out_ptr, n_bytes);
    cross_rank_post_barrier(call_id);
  });
}

void MeshGroup::send(const array& input, int dst, Stream stream) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  uint32_t call_id = next_call_id();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([call_id, data, n_bytes, dst, this]() {
    std::lock_guard<std::mutex> guard(collective_mutex_);
    cross_rank_pre_barrier(call_id);
    mesh_.send(call_id, data, n_bytes, dst);
    cross_rank_post_barrier(call_id);
  });
}

void MeshGroup::recv(array& out, int src, Stream stream) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  uint32_t call_id = next_call_id();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([call_id, data, n_bytes, src, this]() {
    std::lock_guard<std::mutex> guard(collective_mutex_);
    cross_rank_pre_barrier(call_id);
    mesh_.recv(call_id, data, n_bytes, src);
    cross_rank_post_barrier(call_id);
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
  uint32_t call_id = next_call_id();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([call_id, in_ptr, out_ptr, size, this, reduce_op]() {
    std::lock_guard<std::mutex> guard(collective_mutex_);
    cross_rank_pre_barrier(call_id);
    if (size_ > 2 &&
        ((std::is_same_v<T, bfloat16_t> && size > 65536) ||
         size >= 8 * 1024 * 1024 / sizeof(T))) {
      ring_.all_reduce<2>(call_id, in_ptr, out_ptr, size, 1, reduce_op);
    } else {
      mesh_.all_reduce(call_id, in_ptr, out_ptr, size, reduce_op);
    }
    cross_rank_post_barrier(call_id);
  });
}

void MeshGroup::cross_rank_pre_barrier(uint32_t call_id) {
  // Cross-rank pacing barrier: drain the TCP side-channel before any
  // post_send / post_recv. Without this, local threads on each rank
  // can race to acquire the per-group mutex in different orders
  // (multiple MLX streams + γ=2 MTP at c>1 reliably reproduces it),
  // producing asymmetric sequences of post_send / post_recv at the
  // QP level. UC's FIFO matching then fills our recv WR's buffer
  // with the wrong collective's bytes — even with a per-call wr_id
  // we see the matching id on the completion but read foreign bytes
  // out of our slot. Synchronizing the call_ids forces both ranks
  // to wait for each other before either advances. We do NOT abort
  // on call_id mismatch: at c>1 the two ranks' call_id sequences
  // are independent counters, so mismatch is the normal case. The
  // value of the barrier is timing — both ranks block here until
  // both have arrived.
  (void)side_channel_.all_gather<uint32_t>(call_id);
}

void MeshGroup::cross_rank_post_barrier(uint32_t call_id) {
  // Drain symmetrically: ensure peer also finished THIS collective
  // before we return. Otherwise we might enter the next collective
  // and post sends while peer is still finishing this one — same
  // asymmetric-FIFO failure mode.
  (void)side_channel_.all_gather<uint32_t>(call_id);
}

} // namespace mlx::core::distributed::jaccl
