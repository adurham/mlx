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

  // 2026-05-17 PM: restore dedicated ACK QP for top-level groups too.
  //
  // The previous comment (commit 95b1d8d6) argued top-level groups
  // could safely use the legacy inline-ack-on-data-QP path because
  // their traffic is uniform-FRAME_SIZE so cross-call FIFO mismatch
  // wouldn't manifest. That reasoning missed the in-call race:
  // ack_sync_post()'s legacy branch (mesh_impl.h:584-599) posts the
  // ACK_RECV WR *after* the data drain completes. On UC QPs (no
  // retransmit), if peer's ACK_SEND arrives in the window between
  // drain-done and recv-posted, it is silently dropped, drain_acks
  // spins forever on a strict per-call_id filter (mesh_impl.h:690-693),
  // and the collective_mutex_ wedges the cluster until the iter is
  // abandoned. This is the structural mechanism behind the residual
  // gamma>=2 MTP bistability: the chained-collective depth in the
  // draft path multiplies the per-call race opportunities until one
  // hits.
  //
  // The dedicated ACK QP path (subgroups have it via the split() ctor
  // below) pre-posts a pool of ACK_RECV WRs at QP-RTS time
  // (post_ack_recvs called below), replenishes them as they consume,
  // and stashes early-arrival CQEs into cached_ack_recvs_
  // (mesh_impl.h:679-686). No race window, no permanent off-by-one.
  //
  // The 6ms/forward overhead claim that motivated 95b1d8d6 predated
  // the ACK pool deepening to 64 (commit 44d1c40b) and CQ-256 sizing.
  // Actual cost is one extra ibv_poll_cq on a CQ already sized for
  // batch drain -- tens of microseconds, not milliseconds. Net win:
  // eliminate the bistability for a non-measurable perf cost.
  //
  // Each ack-connection borrows its peer's data-connection ibv_context
  // (owns_ctx=false) -- the data Connection owns the ctx lifecycle,
  // the ack Connection just opens its own PD/CQ/QP on top.
  ack_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    ack_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }

  // Initialize all the connections and allocate buffers. Top-level
  // groups exchange QP destinations directly over their own side
  // channel.
  initialize([this](const std::vector<Destination>& info) {
    return side_channel_->all_gather(info);
  });

  // Make sure every node has reached here before continuing
  side_channel_->all_gather<int>(0);

  // Create the mesh implementation object
  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      ack_connections_,
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

  // Pre-post ACK_RECVs into the dedicated ACK QPs so peer's ACK_SEND
  // never arrives at an empty recv FIFO. Same call as the subgroup
  // ctor below.
  mesh_.post_ack_recvs(0);

  // Bootstrap barrier: guarantee both ranks have completed
  // post_ack_recvs(0) before any rank can return from the ctor and
  // issue its first ack_sync_pre. Without this, RANK_A can return
  // first, fire its first lambda's ack_sync_pre, post ACK_SEND to
  // RANK_B's still-empty ACK recv queue, UC silently drops the
  // ACK_SEND, and both ranks wedge. Cheap: one TCP all_gather of one
  // int, called once per group lifetime. Closes the warmup-decode
  // hang that broke b87bddd6 and reappears with ce5c64fd.
  side_channel_->all_gather<int>(0);

  open_trace_file_if_enabled();
}

MeshGroup::MeshGroup(
    int rank,
    int size,
    std::vector<ibv_context*> ctxs,
    std::vector<std::string> device_names,
    bool owns_ctxs,
    const ExchangeFn& exchange,
    std::optional<Stream> parent_stream,
    int color)
    : rank_(rank),
      size_(size),
      color_(color),
      // When JACCL_SPLIT_PARENT_STREAM=1 and the caller passed in the
      // parent's stream, share it. Otherwise allocate a fresh CPU
      // stream as before. Sharing funnels both groups' lambdas onto
      // one cpu::CommandEncoder worker thread, FIFO-serialized at the
      // dispatch level; this is needed on macOS where two distinct
      // encoder threads dispatching concurrently into separate QP
      // sets appears to deadlock at the librdma layer (observed
      // 2026-05-07: rank 1 stalls in mesh_.all_reduce on the 4th coord
      // call right after master's warmup forward completes).
      communication_stream_(
          (parent_stream.has_value() &&
           std::getenv("JACCL_SPLIT_PARENT_STREAM") != nullptr)
              ? *parent_stream
              : new_stream(Device::cpu)),
      side_channel_(std::nullopt),
      device_names_(std::move(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Build Connections from the per-peer ibv_contexts handed to us.
  // owns_ctxs=true → caller opened a fresh context per peer for this
  // subgroup; we close the device on destruction. owns_ctxs=false →
  // we borrow parent's contexts; closing is parent's responsibility.
  connections_.reserve(static_cast<size_t>(size_));
  for (auto* ctx : ctxs) {
    connections_.emplace_back(ctx, /*owns_ctx=*/owns_ctxs);
  }
  // Build ack_connections_ borrowing the same ctxs (owns_ctx=false —
  // data conns or the parent group own those). Each gets its own
  // PD/CQ/QP in initialize() so ACK traffic has an isolated FIFO
  // recv queue.
  ack_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    ack_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }
  if (std::getenv("JACCL_TRACE_SPLIT")) {
    std::cerr << "[jaccl] subgroup ctor rank=" << rank_
              << " owns_ctxs=" << owns_ctxs;
    for (size_t i = 0; i < connections_.size(); i++) {
      std::cerr << " conn[" << i << "].ctx=" << connections_[i].ctx;
    }
    std::cerr << std::endl;
  }

  // Run the same init sequence as the top-level path. The order
  // (PD/CQ/QP → MRs → INIT → exchange → RTR/RTS) matters; macOS
  // librdma locks the QP's MR table at the INIT transition, so MRs
  // must be registered before that.
  initialize(exchange);

  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      ack_connections_,
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

  // Pre-post ACK_RECVs — same rationale as the top-level ctor; the
  // ack_sync_post race (UC byte drop) applies equally to subgroups.
  // The exchange callback above already barriers via the parent's
  // SideChannel, so QPs are RTS on both ranks before we get here.
  mesh_.post_ack_recvs(0);

  // Bootstrap barrier: same rationale as the top-level ctor — without
  // a barrier AFTER post_ack_recvs(0), RANK_A can return from this
  // ctor first, fire its first ack_sync_pre on the subgroup, and
  // post ACK_SEND to RANK_B's still-empty ACK recv queue (UC silent
  // drop → wedge). The exchange() callback uses the parent's
  // SideChannel under the parent's collective_mutex_, mirroring the
  // QP-destination exchange calls above. Sentinel payload: one
  // default-constructed Destination per rank (the lambda's contract
  // is "all-gather this vector"; size=1 is the smallest non-empty
  // payload that avoids any zero-length socket path).
  (void)exchange(std::vector<Destination>{Destination{}});

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

  // Build context list for the subgroup. JACCL_SPLIT_FRESH_CTX=1
  // opens fresh ibv_contexts per subgroup; default (unset) shares
  // the parent's context. Toggle to test which mode actually isolates
  // QPs on macOS librdma.
  std::vector<ibv_context*> ctxs;
  ctxs.reserve(static_cast<size_t>(size_));
  bool fresh_ctx = std::getenv("JACCL_SPLIT_FRESH_CTX") != nullptr;
  if (fresh_ctx) {
    auto fresh_conns = create_connections(device_names_);
    for (auto& c : fresh_conns) {
      ctxs.push_back(c.ctx);
      c.ctx = nullptr; // transfer ownership to subgroup ctor
    }
  } else {
    for (auto& parent_conn : connections_) {
      ctxs.push_back(parent_conn.ctx);
    }
  }

  // Build the subgroup. Its ctor runs the full init pipeline (PD/CQ/QP
  // alloc → MR registration → INIT → exchange → RTR/RTS), with the
  // exchange done over the parent's side channel under our mutex.
  // Pass our stream so the subgroup ctor can opt into sharing it (per
  // JACCL_SPLIT_PARENT_STREAM) — see ctor for rationale. Pass `color`
  // through so the subgroup's trace filename doesn't collide with the
  // parent's.
  return std::make_shared<MeshGroup>(
      rank_,
      size_,
      std::move(ctxs),
      device_names_,
      fresh_ctx,
      [this](const std::vector<Destination>& info) {
        return side_channel_->all_gather(info);
      },
      communication_stream_,
      color);
}

void MeshGroup::open_trace_file_if_enabled() {
  const char* env = std::getenv("JACCL_TRACE_CALLS");
  if (env == nullptr || std::string_view(env) != "1") {
    return;
  }
  char path[160];
  std::snprintf(
      path,
      sizeof(path),
      "/tmp/jaccl_trace_rank_%d_color%x_pid%d.log",
      rank_,
      static_cast<unsigned int>(color_),
      static_cast<int>(getpid()));
  trace_file_ = std::fopen(path, "w");
  if (trace_file_ == nullptr) {
    std::cerr << "[jaccl] Failed to open trace file " << path << "\n";
    return;
  }
  // Hash diagnostic — orthogonal to call tracing. When set, every
  // collective hashes its output bytes after completion and includes
  // the hash in the trace line; rank-divergent hashes localize
  // transport non-bit-exactness.
  const char* hash_env = std::getenv("JACCL_TRACE_HASH");
  hash_enabled_ = (hash_env != nullptr && std::string_view(hash_env) == "1");
  std::fprintf(
      trace_file_,
      "# call_id\top\telem_size\tmsg_bytes%s\n",
      hash_enabled_ ? "\thash" : "");
  std::fflush(trace_file_);
}

void MeshGroup::trace_hash(
    uint32_t call_id, const void* data, int64_t n_bytes) {
  if (trace_file_ == nullptr || !hash_enabled_) {
    return;
  }
  // FNV-1a 64-bit over min(n_bytes, 4096). Capped to bound overhead;
  // 4096 bytes covers a full FRAME_SIZE which suffices to detect any
  // rank-divergent output (one bit-flip → hash divergence).
  const uint8_t* p = static_cast<const uint8_t*>(data);
  int64_t cap = n_bytes < 4096 ? n_bytes : 4096;
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int64_t i = 0; i < cap; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  std::fprintf(
      trace_file_, "\thash=%016llx\n", static_cast<unsigned long long>(h));
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
  // Suppress trailing newline when hash diagnostic is enabled — the
  // hash, computed after the collective completes, will append the
  // newline. Keeps everything for a given call_id on one line.
  std::fprintf(
      trace_file_,
      "%u\t%s\t%d\t%lld%s",
      call_id,
      op,
      elem_size,
      static_cast<long long>(msg_bytes),
      hash_enabled_ ? "" : "\n");
  std::fflush(trace_file_);
}

void MeshGroup::initialize(const ExchangeFn& exchange) {
  // Create PD/CQ/QP for the data connections.
  for (auto& conn : connections_) {
    if (conn.ctx == nullptr) {
      continue;
    }
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }
  // Create PD/CQ/QP for the ACK connections — only if the caller
  // populated ack_connections_ (subgroup ctor does; top-level ctor
  // skips it to avoid the per-collective overhead of polling a
  // separate ACK CQ on the master TP hot path).
  bool has_ack = !ack_connections_.empty();
  if (has_ack) {
    for (auto& conn : ack_connections_) {
      if (conn.ctx == nullptr) {
        continue;
      }
      conn.allocate_protection_domain();
      conn.create_completion_queue(256);
      conn.create_queue_pair();
    }
  }

  allocate_buffers();

  // INIT data QPs. INIT ack QPs only if present.
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
    if (has_ack) {
      ack_connections_[peer].queue_pair_init();
    }
  }

  // Exchange data QP destinations across ranks.
  std::vector<Destination> data_info;
  for (auto& conn : connections_) {
    data_info.emplace_back(conn.info());
  }
  auto data_all_infos = exchange(data_info);

  // Exchange ACK QP destinations across ranks (only if present).
  std::vector<std::vector<Destination>> ack_all_infos;
  if (has_ack) {
    std::vector<Destination> ack_info;
    for (auto& conn : ack_connections_) {
      ack_info.emplace_back(conn.info());
    }
    ack_all_infos = exchange(ack_info);
  }

  // RTR/RTS data QPs.
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_data_info = data_all_infos[peer][rank_];
    if (std::getenv("JACCL_TRACE_SPLIT")) {
      std::cerr << "[jaccl] init rank=" << rank_ << " peer=" << peer
                << " data_qp_num=" << connections_[peer].src.queue_pair_number
                << " peer_data_qp_num=" << peer_data_info.queue_pair_number
                << " peer_lid=" << peer_data_info.local_id << std::endl;
    }
    connections_[peer].queue_pair_rtr(peer_data_info);
    connections_[peer].queue_pair_rts();
  }

  // RTR/RTS ack QPs (only if present).
  if (has_ack) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto peer_ack_info = ack_all_infos[peer][rank_];
      if (std::getenv("JACCL_TRACE_SPLIT")) {
        std::cerr << "[jaccl] init rank=" << rank_ << " peer=" << peer
                  << " ack_qp_num="
                  << ack_connections_[peer].src.queue_pair_number
                  << " peer_ack_qp_num=" << peer_ack_info.queue_pair_number
                  << std::endl;
      }
      ack_connections_[peer].queue_pair_rtr(peer_ack_info);
      ack_connections_[peer].queue_pair_rts();
    }
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
  // Register per-peer ack buffers. If we have a dedicated ACK
  // connection for this peer (subgroup), register to its PD (separate
  // FIFO recv queue). Otherwise (top-level group), register to the
  // data conn's PD — the original ack barrier on data QP path.
  for (int j = 0; j < size_; j++) {
    if (j == rank_ || connections_[j].ctx == nullptr) {
      continue;
    }
    bool has_ack_for_peer =
        !ack_connections_.empty() && ack_connections_[j].ctx != nullptr;
    auto* pd = has_ack_for_peer
        ? ack_connections_[j].protection_domain
        : connections_[j].protection_domain;
    ack_send_buffers_[j].register_to_protection_domain(pd);
    ack_recv_buffers_[j].register_to_protection_domain(pd);
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
    trace_hash(call_id, out_ptr, n_bytes * static_cast<int64_t>(size_));
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
    trace_hash(call_id, data, n_bytes);
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
    trace_hash(call_id, data, n_bytes);
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
        trace_hash(call_id, out_ptr, size * static_cast<int64_t>(sizeof(T)));
      });
}

} // namespace mlx::core::distributed::jaccl
