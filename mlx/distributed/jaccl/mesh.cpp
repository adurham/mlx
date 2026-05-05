// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <unistd.h>

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
      communication_stream_(new_stream(Device::cpu)),
      side_channel_(std::in_place, rank_, size_, coordinator_addr),
      device_names_(device_names),
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
  side_channel_->all_gather<int>(0);

  // Create the mesh implementation object
  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      buffers_,
      ack_send_buffers_,
      ack_recv_buffers_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);

  open_trace_file_if_enabled();
}

MeshGroup::MeshGroup(
    int rank,
    int size,
    std::vector<Connection>&& conns,
    std::vector<std::string> device_names)
    : rank_(rank),
      size_(size),
      communication_stream_(new_stream(Device::cpu)),
      side_channel_(std::nullopt),
      device_names_(std::move(device_names)),
      connections_(std::move(conns)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Caller (split) has already opened device contexts, allocated PD/CQ/QP,
  // and transitioned each QP to RTS. We just need to allocate this
  // subgroup's own buffer pool against those PDs and wire up the impls.
  allocate_buffers();

  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      buffers_,
      ack_send_buffers_,
      ack_recv_buffers_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);

  open_trace_file_if_enabled();
}

std::shared_ptr<GroupImpl> MeshGroup::split(int color, int key) {
  // split is itself a collective. Hold the mutex so no other
  // collective on this parent group can race the side_channel
  // exchange or the QP setup.
  std::lock_guard<std::mutex> guard(collective_mutex_);

  if (!side_channel_.has_value()) {
    throw std::runtime_error(
        "[jaccl] split is only supported on top-level groups (not on a "
        "subgroup created by an earlier split).");
  }

  // Decide membership. For the per-MLX-stream isolation use case, all
  // ranks call split with the same color and we build one new mesh
  // containing all ranks. Mixed-color partitioning would require:
  //   1) filtering device_names_ to the per-color subset of peers, and
  //   2) renumbering ranks within each subgroup (this is what `key`
  //      would feed into).
  // Neither is needed today, so detect mixed colors and throw clearly.
  auto colors = side_channel_->all_gather<int>(color);
  for (int peer = 0; peer < size_; peer++) {
    if (colors[peer] != color) {
      std::ostringstream msg;
      msg << "[jaccl] split requires every rank to use the same color "
          << "(rank " << peer << " gave color=" << colors[peer]
          << ", this rank gave color=" << color
          << "). Mixed-color partitioning is not yet supported.";
      throw std::runtime_error(msg.str());
    }
  }
  // `key` is reserved for sub-rank reassignment when mixed-color is
  // supported. For all-ranks-join we keep parent's rank ordering.
  (void)key;

  // Build subgroup connections that SHARE the parent's ibv_context per
  // peer (borrowing — owns_ctx=false), but allocate their OWN PD/CQ/QP.
  // macOS librdma's ibv_open_device does not return independent
  // contexts on a second call for the same device — recv WRs posted on
  // a "second-open" context fail with IBV_WC_LOC_PROT_ERR despite the
  // PD/MR/QP all being properly bound. The standard RDMA-multi-comm
  // pattern (used by NCCL etc.) is one context per device, with fresh
  // PD/CQ/QP per logical comm. The fresh QPs are what give the
  // subgroup an isolated FIFO from the parent — collectives on the
  // subgroup cannot race-mix posts/recvs with collectives on the
  // parent or on a sibling subgroup.
  std::vector<Connection> new_conns;
  new_conns.reserve(size_);
  for (auto& parent_conn : connections_) {
    new_conns.emplace_back(parent_conn.ctx, /*owns_ctx=*/false);
  }

  // Allocate PD/CQ/QP for each subgroup connection.
  for (auto& conn : new_conns) {
    if (conn.ctx == nullptr) {
      continue;
    }
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  // Transition each peer's QP to INIT.
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    new_conns[peer].queue_pair_init();
  }

  // Exchange destination info via the parent's side channel. We hold
  // the parent's collective_mutex_ so the all_gather has the side
  // channel to itself — no parent collective is dispatching while
  // we're on the wire.
  std::vector<Destination> info;
  for (auto& conn : new_conns) {
    info.emplace_back(conn.info());
  }
  auto all_infos = side_channel_->all_gather(info);

  // Transition each peer's QP to RTR/RTS.
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_info = all_infos[peer][rank_];
    new_conns[peer].queue_pair_rtr(peer_info);
    new_conns[peer].queue_pair_rts();
  }

  // Subgroup takes ownership of the fresh, ready-to-go connections.
  // It allocates its own buffer pool, gets its own MeshImpl/RingImpl,
  // its own next_call_id_ counter, its own collective_mutex_, and
  // its own communication_stream_.
  return std::make_shared<MeshGroup>(
      rank_, size_, std::move(new_conns), device_names_);
}

void MeshGroup::open_trace_file_if_enabled() {
  const char* env = std::getenv("JACCL_TRACE_CALLS");
  if (env == nullptr || std::string_view(env) != "1") {
    return;
  }
  char path[128];
  std::snprintf(
      path,
      sizeof(path),
      "/tmp/jaccl_trace_rank_%d_pid%d.log",
      rank_,
      static_cast<int>(getpid()));
  trace_file_ = std::fopen(path, "w");
  if (trace_file_ == nullptr) {
    std::cerr << "[jaccl] Failed to open trace file " << path << "\n";
    return;
  }
  std::fprintf(
      trace_file_, "# call_id\top\telem_size\tmsg_bytes\n");
  std::fflush(trace_file_);
}

void MeshGroup::trace_call(
    uint32_t call_id,
    const char* op,
    int elem_size,
    int64_t msg_bytes) {
  if (trace_file_ == nullptr) {
    return;
  }
  std::fprintf(
      trace_file_,
      "%u\t%s\t%d\t%lld\n",
      call_id,
      op,
      elem_size,
      static_cast<long long>(msg_bytes));
  std::fflush(trace_file_);
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
  auto all_infos = side_channel_->all_gather(info);

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
  ack_send_buffers_.clear();
  ack_recv_buffers_.clear();
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
  // Per-peer ack buffers used by MeshImpl::ack_sync (one slot per
  // peer including self for index alignment — self is unused). Size
  // matches FRAME_SIZE so ibv_post_send / post_recv can't trip
  // IBV_WC_LOC_PROT_ERR on a sub-page-size SGE — empirically a 4-byte
  // SGE was rejected by macOS librdma at ack-recv time.
  for (int j = 0; j < size_; j++) {
    ack_send_buffers_.emplace_back(FRAME_SIZE);
    ack_recv_buffers_.emplace_back(FRAME_SIZE);
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
  // Register per-peer ack buffers to that peer's PD. Each ack
  // send/recv goes over the data QP for that peer, so they need
  // the same PD as that QP.
  for (int j = 0; j < size_; j++) {
    if (j == rank_ || connections_[j].ctx == nullptr) {
      continue;
    }
    ack_send_buffers_[j].register_to_protection_domain(
        connections_[j].protection_domain);
    ack_recv_buffers_[j].register_to_protection_domain(
        connections_[j].protection_domain);
  }
}

void MeshGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{}, "all_sum");
  });
}

void MeshGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{}, "all_max");
  });
}

void MeshGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{}, "all_min");
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
    trace_call(call_id, "all_gather", 1, static_cast<int64_t>(n_bytes));
    mesh_.all_gather(call_id, in_ptr, out_ptr, n_bytes);
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
    char op[16];
    std::snprintf(op, sizeof(op), "send_dst%d", dst);
    trace_call(call_id, op, 1, n_bytes);
    mesh_.send(call_id, data, n_bytes, dst);
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
    char op[16];
    std::snprintf(op, sizeof(op), "recv_src%d", src);
    trace_call(call_id, op, 1, n_bytes);
    mesh_.recv(call_id, data, n_bytes, src);
  });
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op,
    const char* op_name) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  int64_t size = input.size();
  uint32_t call_id = next_call_id();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch(
      [call_id, in_ptr, out_ptr, size, this, reduce_op, op_name]() {
        std::lock_guard<std::mutex> guard(collective_mutex_);
        trace_call(
            call_id,
            op_name,
            static_cast<int>(sizeof(T)),
            size * static_cast<int64_t>(sizeof(T)));
        if (size_ > 2 &&
            ((std::is_same_v<T, bfloat16_t> && size > 65536) ||
             size >= 8 * 1024 * 1024 / sizeof(T))) {
          ring_.all_reduce<2>(call_id, in_ptr, out_ptr, size, 1, reduce_op);
        } else {
          mesh_.all_reduce(call_id, in_ptr, out_ptr, size, reduce_op);
        }
      });
}

} // namespace mlx::core::distributed::jaccl
