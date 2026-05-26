// Copyright © 2026 Apple Inc.

#include "jaccl/mesh.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <unistd.h>

#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

namespace jaccl {

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const std::string& coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      side_channel_(std::in_place, rank_, size_, coordinator_addr.c_str()),
      device_names_(device_names),
      connections_(create_connections(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // 2026-05-17: restore dedicated ACK QP for top-level groups too.
  // The prior inline-ack-on-data-QP path has an in-call race: legacy
  // ack_sync_post posts ACK_RECV AFTER the data drain, so peer's
  // ACK_SEND can arrive in the window between drain-done and
  // recv-posted. UC silently drops → drain_acks spins forever. This
  // is the structural mechanism behind residual γ≥2 MTP bistability.
  // Dedicated ACK QP pre-posts ACK_RECV_POOL=64 recvs at init time
  // (post_ack_recvs below) and replenishes on consumption, eliminating
  // the race window at a cost of one extra ibv_poll_cq per collective.
  //
  // Each ack-connection borrows its peer's data-connection ibv_context
  // (owns_ctx=false) — the data Connection owns the ctx lifecycle.
  ack_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    ack_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }

  initialize([this](const std::vector<Destination>& info) {
    return side_channel_->all_gather(info);
  });

  // Make sure every node has completed QP setup before continuing.
  side_channel_->all_gather<int>(0);

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

  mesh_.post_ack_recvs(0);

  // Bootstrap barrier: guarantee both ranks have completed
  // post_ack_recvs(0) before any rank can return from the ctor and
  // issue its first ack_sync_pre. Without this, RANK_A can return
  // first, fire its first lambda's ack_sync_pre, post ACK_SEND to
  // RANK_B's still-empty ACK recv queue, UC silently drops, and both
  // ranks wedge.
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
    int color)
    : rank_(rank),
      size_(size),
      color_(color),
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

  // Pre-post ACK_RECVs — the exchange above already barriers via the
  // parent's SideChannel, so QPs are RTS on both ranks before this.
  mesh_.post_ack_recvs(0);

  // Bootstrap barrier: same rationale as the top-level ctor — without
  // a barrier AFTER post_ack_recvs(0), RANK_A can return from this
  // ctor first, fire its first ack_sync_pre on the subgroup, and
  // post ACK_SEND to RANK_B's still-empty ACK recv queue (UC silent
  // drop → wedge). The exchange callback uses the parent's SideChannel
  // under the parent's collective_mutex_. Sentinel payload: one
  // default-constructed Destination (smallest non-empty all_gather).
  (void)exchange(std::vector<Destination>{Destination{}});

  open_trace_file_if_enabled();
}

std::shared_ptr<Group> MeshGroup::split(int color, int key) {
  // split is itself a collective. Hold the mutex so no other
  // collective on this parent group can race the side_channel
  // exchange or the QP setup.
  std::lock_guard<std::mutex> guard(collective_mutex_);

  if (!side_channel_.has_value()) {
    throw std::runtime_error(
        "[jaccl] split is only supported on top-level groups (not on a "
        "subgroup created by an earlier split).");
  }

  // Verify all ranks agree on the color. Mixed-color partitioning
  // (sub-rank renumbering) is not yet supported.
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
  (void)key; // reserved for sub-rank reassignment

  // Build context list for the subgroup. JACCL_SPLIT_FRESH_CTX=1
  // opens fresh ibv_contexts per subgroup; default (unset) shares the
  // parent's context.
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

  // Build the subgroup. Its ctor runs the full init pipeline
  // (PD/CQ/QP alloc → MR registration → INIT → exchange → RTR/RTS),
  // with the exchange done over the parent's side channel under our
  // mutex.
  return std::make_shared<MeshGroup>(
      rank_,
      size_,
      std::move(ctxs),
      device_names_,
      fresh_ctx,
      [this](const std::vector<Destination>& info) {
        return side_channel_->all_gather(info);
      },
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
  const char* hash_env = std::getenv("JACCL_TRACE_HASH");
  hash_enabled_ = (hash_env != nullptr && std::string_view(hash_env) == "1");
  std::fprintf(
      trace_file_,
      "# call_id\top\tmsg_bytes%s\n",
      hash_enabled_ ? "\thash" : "");
  std::fflush(trace_file_);
}

void MeshGroup::trace_call(
    uint32_t call_id,
    const char* op,
    int64_t msg_bytes) {
  if (trace_file_ == nullptr) {
    return;
  }
  // Suppress trailing newline when hash diagnostic is enabled — the
  // hash, computed after the collective completes, will append it.
  std::fprintf(
      trace_file_,
      "%u\t%s\t%lld%s",
      call_id,
      op,
      static_cast<long long>(msg_bytes),
      hash_enabled_ ? "" : "\n");
  std::fflush(trace_file_);
}

void MeshGroup::trace_hash(uint32_t call_id, const void* data, int64_t n_bytes) {
  if (trace_file_ == nullptr || !hash_enabled_) {
    return;
  }
  // FNV-1a 64-bit over min(n_bytes, 4096). Capped to bound overhead;
  // 4096 bytes covers a full FRAME_SIZE which suffices to detect any
  // rank-divergent output.
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
  // Create PD/CQ/QP for the ACK connections — only if populated.
  // Subgroup ctor does; top-level ctor skips to avoid per-collective
  // overhead of polling a separate ACK CQ on the master TP hot path.
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

  // INIT data QPs (and ACK QPs if present).
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

  // RTR/RTS ACK QPs (only if present).
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
  buffers_.clear();
  ack_send_buffers_.clear();
  ack_recv_buffers_.clear();
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();

  // Allocate data and ring buffers.
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      for (int j = 0; j < 2; j++) {
        ring_send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        ring_recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }
  // Per-peer ack buffers (one slot per peer, including self for index
  // alignment — self slot is unused). FRAME_SIZE avoids macOS librdma
  // rejecting sub-page-size SGEs at ack-recv time.
  for (int j = 0; j < size_; j++) {
    ack_send_buffers_.emplace_back(FRAME_SIZE);
    ack_recv_buffers_.emplace_back(FRAME_SIZE);
  }

  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        if (j == rank_) {
          // Our send buffer: register to all connected peers' PDs so we
          // can send it to all of them.
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .register_to_protection_domain(conn.protection_domain);
            }
          }
        } else {
          // Recv buffer from rank j: register to rank j's PD.
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
        }
      }

      // Ring buffers.
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
  // Register ack buffers. Subgroups have a dedicated ACK connection per
  // peer — register to its PD (isolated FIFO recv queue). Top-level
  // group has no ACK connections — register to the data conn's PD
  // (original ack barrier on data QP path).
  for (int j = 0; j < size_; j++) {
    if (j == rank_ || connections_[j].ctx == nullptr) {
      continue;
    }
    bool has_ack_for_peer =
        !ack_connections_.empty() && ack_connections_[j].ctx != nullptr;
    auto* pd = has_ack_for_peer ? ack_connections_[j].protection_domain
                                : connections_[j].protection_domain;
    ack_send_buffers_[j].register_to_protection_domain(pd);
    ack_recv_buffers_[j].register_to_protection_domain(pd);
  }
}

void MeshGroup::all_sum(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  trace_call(call_id, "all_sum", static_cast<int64_t>(n_bytes));
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(call_id, input, output, n_bytes, SumOp<T>{});
  });
  trace_hash(call_id, output, static_cast<int64_t>(n_bytes));
}

void MeshGroup::all_max(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  trace_call(call_id, "all_max", static_cast<int64_t>(n_bytes));
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(call_id, input, output, n_bytes, MaxOp<T>{});
  });
  trace_hash(call_id, output, static_cast<int64_t>(n_bytes));
}

void MeshGroup::all_min(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  trace_call(call_id, "all_min", static_cast<int64_t>(n_bytes));
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(call_id, input, output, n_bytes, MinOp<T>{});
  });
  trace_hash(call_id, output, static_cast<int64_t>(n_bytes));
}

void MeshGroup::all_gather(
    const void* input,
    void* output,
    size_t n_bytes) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  trace_call(call_id, "all_gather", static_cast<int64_t>(n_bytes));
  mesh_.all_gather(
      call_id,
      static_cast<const char*>(input),
      static_cast<char*>(output),
      n_bytes);
  trace_hash(call_id, output, static_cast<int64_t>(n_bytes) * size_);
}

void MeshGroup::send(const void* input, size_t n_bytes, int dst) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  char op[16];
  std::snprintf(op, sizeof(op), "send_dst%d", dst);
  trace_call(call_id, op, static_cast<int64_t>(n_bytes));
  mesh_.send(
      call_id, static_cast<const char*>(input), static_cast<int64_t>(n_bytes), dst);
  trace_hash(call_id, input, static_cast<int64_t>(n_bytes));
}

void MeshGroup::recv(void* output, size_t n_bytes, int src) {
  std::lock_guard<std::mutex> guard(collective_mutex_);
  uint32_t call_id = next_call_id();
  char op[16];
  std::snprintf(op, sizeof(op), "recv_src%d", src);
  trace_call(call_id, op, static_cast<int64_t>(n_bytes));
  mesh_.recv(
      call_id, static_cast<char*>(output), static_cast<int64_t>(n_bytes), src);
  trace_hash(call_id, output, static_cast<int64_t>(n_bytes));
}

void MeshGroup::barrier() {
  uint8_t b = 0;
  all_sum(&b, &b, sizeof(b), Dtype::UInt8);
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    uint32_t call_id,
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  auto in_ptr = static_cast<const T*>(input);
  auto out_ptr = static_cast<T*>(output);
  int64_t count = n_bytes / sizeof(T);
  if (size_ > 2 &&
      ((std::is_same_v<T, bfloat16_t> && count > 256 * 1024) ||
       count >= 8 * 1024 * 1024 / static_cast<int64_t>(sizeof(T)))) {
    ring_.all_reduce<2>(in_ptr, out_ptr, count, 1, reduce_op);
  } else {
    mesh_.all_reduce(call_id, in_ptr, out_ptr, count, reduce_op);
  }
}

} // namespace jaccl
