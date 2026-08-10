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
      coordinator_addr_(coordinator_addr),
      device_names_(device_names),
      connections_(create_connections(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // CROSS-RANK ASYMMETRIC-DETECTION FIX (2026-08-09, design doc Section 30):
  // side_channel_ is reused as the RECOVERY handshake path in
  // MeshGroup::reconnect()/reconnect_fresh() -- both ranks re-exchange QP
  // info over it after a data-path jaccl fault, and it survives faults
  // (constructed once, here, never rebuilt). Real-hardware evidence: on a
  // genuine data-path fault, the two ranks do NOT detect it at the same
  // time. In the rank0-drives/rank1-mirrors batched-decode topology,
  // rank0's own data-path recv() (bounded by
  // MLX_JACCL_RECV_RETRY_DEADLINE_SECS, default 60s) fired, tore down its
  // device contexts, and entered THIS side channel's recovery handshake to
  // wait for rank1's fresh QP info -- all before rank1 had independently
  // detected the SAME underlying fault on its own, differently-timed
  // data-path recv(). Confirmed on real hardware: rank0 threw its ORIGINAL
  // fault at t=0s and its RECOVERY handshake attempt itself timed out and
  // threw at t=70s (the same 60s deadline + ~10s overhead); rank1 didn't
  // even log its own original fault until t=71s -- 1s AFTER rank0 already
  // gave up and crashed the whole runner (forcing a full ~90s re-place
  // instead of the in-place recovery this coordinator path exists for).
  // The recovery handshake's own deadline was THE SAME constant as the
  // per-rank detection deadline it has to outlast, which structurally can
  // never work when detection is asymmetric -- a design invariant
  // violation, not a tuning question. Fix: give side_channel_'s own retry
  // deadline enough headroom to outlast the worst-case cross-rank
  // detection skew (bounded by one full data-path deadline window: rank A
  // can detect at t=0, rank B's independent clock can still have up to one
  // full deadline-window left before IT detects, i.e. up to
  // t=data_path_deadline) PLUS the actual reconnect_fresh() device
  // teardown/reopen work. NOT applied to p2p_channel_ (the hot-path
  // send()/recv() retry-barrier channel) -- that one should keep failing
  // fast on the data-path's own deadline; only the coordinator/recovery
  // path needs the longer budget. Overridable via
  // MLX_JACCL_COORD_RECV_RETRY_DEADLINE_SECS for diagnostics.
  {
    const double _data_path_deadline = [] {
      const char* e = std::getenv("MLX_JACCL_RECV_RETRY_DEADLINE_SECS");
      return e ? std::atof(e) : 60.0;
    }();
    const double _coord_retry_deadline = [&] {
      const char* e = std::getenv("MLX_JACCL_COORD_RECV_RETRY_DEADLINE_SECS");
      if (e) {
        return std::atof(e);
      }
      return 2.0 * _data_path_deadline + 30.0;
    }();
    side_channel_->set_recv_retry_deadline_secs(_coord_retry_deadline);
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

  // ROOT-CAUSE FIX (2026-07-17): dedicated QP for the jaccl-v2 reliable-ARQ
  // optimistic path (the standing POOL_RECV_WR pool + its own SEND_WR/
  // recv traffic in reliable_all_reduce_v2). This feature exists because
  // Apple's RDMA stack lacks hardware RC connections, so jaccl provides
  // reliability over UC in software for TP's collectives. It used to
  // share connections_ (the data QP) with exo's Pipeline-Parallel raw
  // send()/recv() p2p handoff. The pool's recv buffers use one uniform
  // size class (v2_pool_sz_); send()/recv() posts buffers sized per
  // message. A size mismatch on the same QP throws IBV_WC_LOC_LEN_ERR,
  // and since the completion-filtering in both paths didn't distinguish
  // work_type, an errored pool slot got silently discarded without
  // being re-armed -- eventually starving the pool and wedging BOTH
  // paths (observed: raw send()/recv() stalling mid-transfer even
  // though its own synchronization was correct). Same fix shape as the
  // ACK-QP split above: dedicated PD/CQ/QP, borrows the peer's
  // ibv_context (owns_ctx=false).
  pool_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    pool_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }

  // DESIGN DOC SECTION 37 PHASE 1 (2026-08-10): dedicated QP for send()/
  // recv()'s got-bitmask retry exchange (p2p_retry_exchange), migrated off
  // the TCP p2p_retry_barrier. Same isolation pattern as ack_connections_
  // and pool_connections_ above -- its own PD/CQ/QP, borrowing the peer's
  // ibv_context (owns_ctx=false) -- so its standing recv pool's uniform
  // frame-sized buffers can never collide with the per-message-sized
  // buffers raw send()/recv() posts on connections_ (the IBV_WC_LOC_LEN_ERR
  // class of bug this file has already paid to learn twice). See
  // p2p_retry_connections_'s member comment in mesh.h.
  p2p_retry_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    p2p_retry_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }

  initialize([this](const std::vector<Destination>& info) {
    return side_channel_->all_gather(info);
  });

  // Make sure every node has completed QP setup before continuing.
  side_channel_->all_gather<int>(0);

  // ROOT-CAUSE FIX (2026-07-17, revision 2): dedicated TCP side-channel for
  // send()/recv()'s p2p retry protocol -- see p2p_channel_ member comment
  // in mesh.h for the full collision this isolation fixes (shared
  // coordinator_ socket getting interleaved bytes from mlx-lm's per-forward
  // model-level all_gather AND our retry barrier). Built HERE (right after
  // side_channel_'s own setup barrier, so both ranks are guaranteed to
  // reach this point together -- eager construction, not lazy-on-first-use,
  // per the construction-order hazard Fable flagged: a lazy p2p_channel_
  // would let one rank block in connect() while the peer is still parked
  // inside an unrelated all_gather on the primary channel, an avoidable
  // deadlock). Port is coordinator_addr's own port + 1 -- deterministic,
  // no dynamic port exchange needed since coordinator_addr is already
  // uniquely assigned per-instance by exo's placement layer.
  rebuild_p2p_channel();

  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      ack_connections_,
      pool_connections_,
      p2p_retry_connections_,
      buffers_,
      ack_send_buffers_,
      ack_recv_buffers_,
      p2p_retry_send_buffers_,
      p2p_retry_recv_buffers_);
  // Give the top-level mesh the reliable TCP coordinator for the confirmed
  // (ack-of-ack) barrier. side_channel_ is in-place and outlives mesh_.
  mesh_.set_coordinator(&*side_channel_);
  // Give the top-level mesh the dedicated p2p retry channel for send()/
  // recv(). p2p_channel_ is in-place and outlives mesh_.
  mesh_.set_p2p_channel(&*p2p_channel_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);

  mesh_.post_ack_recvs(0);
  // Pre-post the standing p2p-retry recv pool on the dedicated QP, same
  // lifecycle point as post_ack_recvs above and fenced by the same
  // bootstrap barrier below (UC silently drops a send into an empty recv
  // queue, so both ranks must be armed before either returns from the ctor).
  mesh_.post_p2p_retry_recvs();

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

  // NOTE: like pool_connections_, p2p_retry_connections_ is NOT built for
  // subgroups -- both stay empty here and are passed through to MeshImpl as
  // empty vectors. initialize()'s has_pool/has_p2p_retry guards skip their
  // bring-up accordingly, and post_p2p_retry_recvs()/p2p_retry_exchange()
  // are no-op/throw on an empty span (send()/recv()'s retry path is a
  // top-level-group-only concern -- see mesh_impl.h).
  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      ack_connections_,
      pool_connections_,
      p2p_retry_connections_,
      buffers_,
      ack_send_buffers_,
      ack_recv_buffers_,
      p2p_retry_send_buffers_,
      p2p_retry_recv_buffers_);
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

  // From this point our contexts may be borrowed by a subgroup —
  // reconnect_fresh() (which closes them) is no longer safe.
  has_split_ = true;

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
  // Create PD/CQ/QP for the jaccl-v2 pool connections — always populated
  // (see ctor comment); cheap to set up unconditionally and only used if
  // the reliable-optimistic path is actually enabled at runtime.
  bool has_pool = !pool_connections_.empty();
  if (has_pool) {
    for (auto& conn : pool_connections_) {
      if (conn.ctx == nullptr) {
        continue;
      }
      conn.allocate_protection_domain();
      conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
      conn.create_queue_pair();
    }
  }
  // Create PD/CQ/QP for the dedicated p2p-retry connections — populated by
  // the top-level ctor only (subgroups leave it empty, same as
  // pool_connections_ above).
  bool has_p2p_retry = !p2p_retry_connections_.empty();
  if (has_p2p_retry) {
    for (auto& conn : p2p_retry_connections_) {
      if (conn.ctx == nullptr) {
        continue;
      }
      conn.allocate_protection_domain();
      conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
      conn.create_queue_pair();
    }
  }

  allocate_buffers();

  // INIT data QPs (and ACK/pool QPs if present).
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
    if (has_ack) {
      ack_connections_[peer].queue_pair_init();
    }
    if (has_pool) {
      pool_connections_[peer].queue_pair_init();
    }
    if (has_p2p_retry) {
      p2p_retry_connections_[peer].queue_pair_init();
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

  // Exchange pool QP destinations across ranks (only if present).
  std::vector<std::vector<Destination>> pool_all_infos;
  if (has_pool) {
    std::vector<Destination> pool_info;
    for (auto& conn : pool_connections_) {
      pool_info.emplace_back(conn.info());
    }
    pool_all_infos = exchange(pool_info);
  }

  // Exchange p2p-retry QP destinations across ranks (only if present).
  std::vector<std::vector<Destination>> p2p_retry_all_infos;
  if (has_p2p_retry) {
    std::vector<Destination> p2p_retry_info;
    for (auto& conn : p2p_retry_connections_) {
      p2p_retry_info.emplace_back(conn.info());
    }
    p2p_retry_all_infos = exchange(p2p_retry_info);
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

  // RTR/RTS pool QPs (only if present).
  if (has_pool) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto peer_pool_info = pool_all_infos[peer][rank_];
      if (std::getenv("JACCL_TRACE_SPLIT")) {
        std::cerr << "[jaccl] init rank=" << rank_ << " peer=" << peer
                  << " pool_qp_num="
                  << pool_connections_[peer].src.queue_pair_number
                  << " peer_pool_qp_num=" << peer_pool_info.queue_pair_number
                  << std::endl;
      }
      pool_connections_[peer].queue_pair_rtr(peer_pool_info);
      pool_connections_[peer].queue_pair_rts();
    }
  }

  // RTR/RTS p2p-retry QPs (only if present).
  if (has_p2p_retry) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto peer_p2p_retry_info = p2p_retry_all_infos[peer][rank_];
      if (std::getenv("JACCL_TRACE_SPLIT")) {
        std::cerr << "[jaccl] init rank=" << rank_ << " peer=" << peer
                  << " p2p_retry_qp_num="
                  << p2p_retry_connections_[peer].src.queue_pair_number
                  << " peer_p2p_retry_qp_num="
                  << peer_p2p_retry_info.queue_pair_number << std::endl;
      }
      p2p_retry_connections_[peer].queue_pair_rtr(peer_p2p_retry_info);
      p2p_retry_connections_[peer].queue_pair_rts();
    }
  }
}

void MeshGroup::rebuild_p2p_channel() {
  auto colon = coordinator_addr_.find(':');
  if (colon == std::string::npos) {
    std::ostringstream msg;
    msg << "[jaccl] Can't derive p2p retry channel port from "
           "coordinator_addr (missing ':'): "
        << coordinator_addr_;
    throw std::runtime_error(msg.str());
  }
  std::string ip = coordinator_addr_.substr(0, colon);
  int port = std::stoi(coordinator_addr_.substr(colon + 1));
  std::ostringstream p2p_addr;
  p2p_addr << ip << ":" << (port + 1);
  // Assigning to an already-engaged optional destroys the old SideChannel
  // (and its TCPSocket, whose dtor shuts down + closes the fd) before the
  // new one is constructed, so any bytes left over from a pre-fault
  // p2p_retry_barrier exchange are discarded with the old connection
  // rather than misread by the next call on a "reconnected" but stale
  // stream. SO_REUSEADDR/SO_REUSEPORT on the listener (rank 0) and 4x
  // exponential-backoff retries on the connecting side (TCPSocket::connect,
  // see SideChannel ctor) absorb the TIME_WAIT / accept-not-yet-listening
  // races from tearing down and rebinding the same port back to back.
  p2p_channel_.emplace(rank_, size_, p2p_addr.str().c_str());

  // ROOT-CAUSE FIX (2026-08-09, design doc Section 38, real-hardware
  // finding): p2p_channel_ previously used the SAME 60s deadline as the
  // data path, deliberately ("that one should keep failing fast" -- see
  // this class's own send()/recv() comments). Confirmed on real hardware
  // via direct instrumentation (design doc Section 35): the RDMA data
  // path itself is never the bottleneck (retry rounds converge in
  // 70-200us) -- what actually stalls is p2p_retry_barrier's plain TCP
  // recv, waiting for the PEER to reach its own matching call site,
  // which can legitimately take longer than 60s under real production
  // load (confirmed: the peer was actively mid-decode-step, not wedged
  // or dead, via a faulthandler dump during the exact stall window).
  //
  // Unlike side_channel_'s deadline extension (mesh.cpp ctor, ~line 60),
  // which is DERIVED from a provable bound (cross-rank fault-detection
  // skew <= one data-path deadline window), there is no equivalent
  // principled bound here -- "how long can the peer legitimately be
  // busy before reaching a barrier call" has no formula, only empirical
  // headroom. A `consult` review recommended NOT cloning
  // side_channel_'s 2x+30s formula (that number's derivation doesn't
  // apply to this question) and NOT keeping 60s as the default (already
  // empirically falsified by real hardware) -- instead, pick a
  // generously large default justified by cost asymmetry: a false
  // positive here fires during HEALTHY operation and drops in-flight
  // work for no reason, while a genuinely dead/wedged peer will usually
  // surface via a TCP-level error (ECONNRESET etc., immediate throw,
  // unaffected by this deadline) well before any reasonable deadline
  // fires anyway -- so this deadline is really only guarding the rare
  // hung-but-still-connected case, and erring large costs nothing on
  // the common healthy path.
  const double _p2p_retry_deadline = [] {
    const char* e = std::getenv("MLX_JACCL_P2P_RECV_RETRY_DEADLINE_SECS");
    return e ? std::atof(e) : 300.0;
  }();
  p2p_channel_->set_recv_retry_deadline_secs(_p2p_retry_deadline);
}

void MeshGroup::reconnect() {
  // In-place recovery: reset the wedged QPs and re-establish the connections
  // WITHOUT destroying PD/CQ/MRs or reloading the model. Both ranks call this
  // after a collective fault; the surviving TCP side_channel_ (coordinator)
  // re-exchanges QP info and acts as the cross-rank barrier so neither rank
  // sends before both are RTS. Recovers a UC transport wedge in ~ms instead of
  // a full runner re-place (~90s model reload).
  if (!side_channel_.has_value()) {
    throw std::runtime_error(
        "[jaccl] reconnect is only supported on top-level groups");
  }

  // Fresh (hard) mode: rebuild the device contexts and everything on top of
  // them. Opt-in via MLX_JACCL_RECONNECT_FRESH=1; silently degrades to the
  // QP-only reset once subgroups exist (they borrow our contexts).
  // IMPORTANT: both ranks must take the SAME branch — has_split_ flips at
  // split(), which is itself a collective executed in matching order on all
  // ranks, and the env var must be set identically cluster-wide.
  const char* fresh_env = std::getenv("MLX_JACCL_RECONNECT_FRESH");
  if (fresh_env != nullptr && std::string_view(fresh_env) == "1") {
    if (!has_split_) {
      reconnect_fresh();
      return;
    }
    fprintf(
        stderr,
        "[jaccl] reconnect rank=%d FRESH requested but subgroups borrow our "
        "contexts — falling back to QP-only reset\n",
        rank_);
    fflush(stderr);
  }

  const bool has_ack = !ack_connections_.empty();
  // CRITICAL DIVERGENCE FROM pool_connections_ (2026-08-10): unlike
  // pool_connections_ -- which this soft path deliberately does NOT touch,
  // since jaccl-v2's reliable-optimistic path is opt-in and idle on the
  // default configuration -- p2p_retry_connections_ MUST be reset and
  // re-established here. It replaces p2p_channel_, the ALWAYS-ACTIVE hot
  // path that every single send()/recv() call depends on; leaving its QPs
  // wedged across a reconnect would leave the p2p retry protocol dead on
  // the very next transfer. So it gets the same full lifecycle treatment
  // as connections_/ack_connections_ below (reset -> init -> exchange ->
  // RTR/RTS -> re-post recv pool).
  const bool has_p2p_retry = !p2p_retry_connections_.empty();
  fprintf(
      stderr,
      "[jaccl] reconnect rank=%d ENTER (size=%d has_ack=%d has_p2p_retry=%d)\n",
      rank_,
      size_,
      has_ack ? 1 : 0,
      has_p2p_retry ? 1 : 0);
  fflush(stderr);

  // 1. Reset every QP (flush wedged/in-flight WRs, drain stale CQEs).
  for (auto& conn : connections_) {
    if (conn.ctx != nullptr) {
      conn.queue_pair_reset();
    }
  }
  if (has_ack) {
    for (auto& conn : ack_connections_) {
      if (conn.ctx != nullptr) {
        conn.queue_pair_reset();
      }
    }
  }
  if (has_p2p_retry) {
    for (auto& conn : p2p_retry_connections_) {
      if (conn.ctx != nullptr) {
        conn.queue_pair_reset();
      }
    }
  }

  auto exchange = [this](const std::vector<Destination>& info) {
    return side_channel_->all_gather(info);
  };

  // 2. Re-establish: INIT -> exchange -> RTR/RTS (mirrors initialize(), minus
  //    PD/CQ/QP creation and buffer allocation which are preserved).
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
    if (has_ack) {
      ack_connections_[peer].queue_pair_init();
    }
    if (has_p2p_retry) {
      p2p_retry_connections_[peer].queue_pair_init();
    }
  }

  fprintf(stderr, "[jaccl] reconnect rank=%d QPs reset+init; exchanging data QP info (blocks for peer)...\n", rank_);
  fflush(stderr);
  std::vector<Destination> data_info;
  for (auto& conn : connections_) {
    data_info.emplace_back(conn.info());
  }
  auto data_all_infos = exchange(data_info);
  fprintf(stderr, "[jaccl] reconnect rank=%d data QP info exchanged; exchanging ack QP info...\n", rank_);
  fflush(stderr);

  std::vector<std::vector<Destination>> ack_all_infos;
  if (has_ack) {
    std::vector<Destination> ack_info;
    for (auto& conn : ack_connections_) {
      ack_info.emplace_back(conn.info());
    }
    ack_all_infos = exchange(ack_info);
  }
  std::vector<std::vector<Destination>> p2p_retry_all_infos;
  if (has_p2p_retry) {
    std::vector<Destination> p2p_retry_info;
    for (auto& conn : p2p_retry_connections_) {
      p2p_retry_info.emplace_back(conn.info());
    }
    p2p_retry_all_infos = exchange(p2p_retry_info);
  }
  fprintf(stderr, "[jaccl] reconnect rank=%d ack QP info exchanged; RTR/RTS...\n", rank_);
  fflush(stderr);

  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_rtr(data_all_infos[peer][rank_]);
    connections_[peer].queue_pair_rts();
    if (has_ack) {
      ack_connections_[peer].queue_pair_rtr(ack_all_infos[peer][rank_]);
      ack_connections_[peer].queue_pair_rts();
    }
    if (has_p2p_retry) {
      p2p_retry_connections_[peer].queue_pair_rtr(
          p2p_retry_all_infos[peer][rank_]);
      p2p_retry_connections_[peer].queue_pair_rts();
    }
  }

  // 3. Rebuild the dedicated p2p retry-barrier channel. NOTE (2026-08-10):
  //    this now happens ALONGSIDE p2p_retry_connections_'s QP reset above,
  //    not instead of it -- both mechanisms coexist during the migration.
  //    send()/recv() no longer route their retry exchange through this TCP
  //    socket (they use the dedicated RDMA QP reset in steps 1-2), so
  //    p2p_channel_ is structurally unused by the hot path; its lifecycle
  //    plumbing is deliberately left in place here and in the ctor/
  //    reconnect_fresh() rather than ripped out mid-migration. The original
  //    rationale still holds for as long as it IS wired up: it survives the
  //    RDMA QP reset/re-init untouched (separate TCP socket, not part of the
  //    ibv device state), so any bytes left mid-frame on it from the fault
  //    that triggered this reconnect are still sitting there -- reusing it
  //    would let a p2p_retry_barrier() call read a stale reply and throw
  //    DESYNC. Tear down + rebuild on the same derived port instead (see
  //    rebuild_p2p_channel()).
  if (p2p_channel_.has_value()) {
    rebuild_p2p_channel();
    mesh_.set_p2p_channel(&*p2p_channel_);
  }

  // 4. Clear stale ACK bookkeeping and re-post the ACK_RECV pool on the fresh
  //    QPs, then barrier so both ranks are ready before the next collective.
  //    This barrier also fences step 3 above -- neither rank proceeds until
  //    both have finished rebuilding p2p_channel_.
  mesh_.reset_ack_state();
  mesh_.post_ack_recvs(0);
  // The p2p-retry QPs were reset above, which discarded their standing recv
  // pool along with every other posted WR -- re-arm it here, fenced by the
  // same final barrier, or the peer's first post-reconnect retry frame lands
  // on an empty recv queue and UC silently drops it.
  mesh_.post_p2p_retry_recvs();
  fprintf(stderr, "[jaccl] reconnect rank=%d RTS done; final barrier...\n", rank_);
  fflush(stderr);
  side_channel_->all_gather<int>(0);
  fprintf(stderr, "[jaccl] reconnect rank=%d COMPLETE\n", rank_);
  fflush(stderr);
}

void MeshGroup::reconnect_fresh() {
  fprintf(
      stderr,
      "[jaccl] reconnect_fresh rank=%d ENTER (size=%d) — closing device "
      "contexts and rebuilding\n",
      rank_,
      size_);
  fflush(stderr);

  // 1. Tear down device-side state in reverse dependency order. Clearing the
  //    buffer vectors runs SharedBuffer dtors, which deregister every MR —
  //    this MUST happen before the owning PDs are dealloc'd. Clearing the
  //    connection vectors then runs Connection dtors: destroy_qp,
  //    destroy_cq, dealloc_pd, and (data connections own their ctx)
  //    close_device. Ack connections borrow the data ctxs (owns_ctx=false),
  //    so they are dropped FIRST while their ctx is still open.
  buffers_.clear();
  ack_send_buffers_.clear();
  ack_recv_buffers_.clear();
  p2p_retry_send_buffers_.clear();
  p2p_retry_recv_buffers_.clear();
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();
  ack_connections_.clear();
  pool_connections_.clear();
  p2p_retry_connections_.clear();
  connections_.clear();

  // 2. Reopen the devices — fresh ibv contexts. This is the whole point:
  //    the dead-path wedge lives somewhere in the preserved ctx/PD/CQ/MR
  //    layer and only a fresh ibv_open_device clears it.
  connections_ = create_connections(device_names_);
  ack_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    ack_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }
  // Rebuild pool_connections_ too (see ctor comment) — must be reborn
  // against the fresh ctxs just like ack_connections_, or reconnect_fresh
  // leaves it dangling against destroyed contexts.
  pool_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    pool_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }
  // Same for p2p_retry_connections_ -- reborn against the fresh ctxs, or
  // send()/recv()'s retry exchange is left pointing at destroyed contexts.
  p2p_retry_connections_.reserve(static_cast<size_t>(size_));
  for (auto& data_conn : connections_) {
    p2p_retry_connections_.emplace_back(data_conn.ctx, /*owns_ctx=*/false);
  }

  // 3. Full bring-up, identical to the ctor: PD/CQ/QP creation, buffer
  //    allocation + MR registration (BEFORE the INIT transition — macOS
  //    librdma locks the QP's MR table there), INIT → exchange → RTR/RTS
  //    over the surviving TCP side channel, then a barrier so no rank
  //    proceeds before every rank finished QP setup.
  initialize([this](const std::vector<Destination>& info) {
    return side_channel_->all_gather(info);
  });
  side_channel_->all_gather<int>(0);

  // 4. Rebuild the impl views — MeshImpl/RingImpl hold spans over the
  //    vectors we just rebuilt, so the old instances dangle. A fresh
  //    MeshImpl also starts with clean ACK bookkeeping (the equivalent of
  //    reset_ack_state() on the soft path).
  mesh_ = MeshImpl(
      rank_,
      size_,
      connections_,
      ack_connections_,
      pool_connections_,
      p2p_retry_connections_,
      buffers_,
      ack_send_buffers_,
      ack_recv_buffers_,
      p2p_retry_send_buffers_,
      p2p_retry_recv_buffers_);
  mesh_.set_coordinator(&*side_channel_);
  // Rebuild p2p_channel_ same as reconnect() -- it is a separate TCP socket
  // from the RDMA transport rebuilt above, so it does NOT get cleared by
  // steps 1-2 and can carry a stale/desynced stream across from whatever
  // fault triggered this reconnect_fresh() (was previously left as-is here,
  // which produced a "p2p_retry_barrier DESYNC" throw on the very next p2p
  // call after an otherwise-successful reconnect_fresh -- see
  // rebuild_p2p_channel()'s comment for the mechanism). mesh_ is a
  // brand-new MeshImpl either way, so its pointer needs re-wiring just
  // like set_coordinator above regardless of whether the channel itself
  // was rebuilt.
  if (p2p_channel_.has_value()) {
    rebuild_p2p_channel();
    mesh_.set_p2p_channel(&*p2p_channel_);
  }
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);

  mesh_.post_ack_recvs(0);
  mesh_.post_p2p_retry_recvs();

  // Bootstrap barrier — same rationale as the ctor: neither rank may fire
  // its first ack_sync_pre before BOTH have posted their ACK recv pool
  // (UC silently drops sends into an empty recv queue).
  side_channel_->all_gather<int>(0);

  fprintf(stderr, "[jaccl] reconnect_fresh rank=%d COMPLETE\n", rank_);
  fflush(stderr);
}

void MeshGroup::allocate_buffers() {
  buffers_.clear();
  ack_send_buffers_.clear();
  ack_recv_buffers_.clear();
  p2p_retry_send_buffers_.clear();
  p2p_retry_recv_buffers_.clear();
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
  // Per-peer p2p-retry buffers. BOTH send and recv are P2P_RETRY_NUM_SLOTS-
  // deep rotating pools per peer, flattened as peer * P2P_RETRY_NUM_SLOTS +
  // slot -- the exact indexing p2p_retry_send_bitmask/post_p2p_retry_recvs/
  // p2p_retry_exchange use in mesh_impl.h. A single shared send slot per
  // peer was an earlier draft that turned out to be unsafe: reusing one
  // buffer for every frame of a multi-frame bitmask requires blocking to
  // drain each frame's completion before reusing it, and that blocking
  // drain polls the SAME CQ the main exchange loop polls for RECV
  // completions -- silently dropping any interleaved peer frame seen while
  // waiting (a consult review caught this). Per-frame rotating pools on
  // BOTH sides make sends fire-and-forget in the common case instead --
  // see p2p_retry_send_bitmask's own comment in mesh_impl.h for the full
  // incident this fixes. Self slots are allocated for index alignment and
  // left unused.
  for (int j = 0; j < size_; j++) {
    for (int s = 0; s < MeshImpl::P2P_RETRY_NUM_SLOTS; s++) {
      p2p_retry_send_buffers_.emplace_back(FRAME_SIZE);
      p2p_retry_recv_buffers_.emplace_back(FRAME_SIZE);
    }
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
          // ROOT-CAUSE FIX (2026-07-17): ALSO register to every peer's
          // pool_connections_ PD. buffers_ is the SAME array send()/
          // recv()/all_reduce()/all_gather() use AND that
          // reliable_all_reduce_v2 (jaccl-v2) now posts via
          // pool_connections_ (its own dedicated QP, isolated from
          // connections_ to fix an IBV_WC_LOC_LEN_ERR collision — see
          // pool_connections_ member comment in mesh.h). SharedBuffer's
          // lkey lookup is keyed by ibv_pd* (register_to_protection_domain
          // populates a per-PD map); post_send/post_recv on
          // pool_connections_[peer] calls to_scatter_gather_entry(
          // pool_connections_[peer].protection_domain) — WITHOUT this
          // registration that lookup throws (unregistered PD), so v2
          // traffic on the new QP would fail immediately on first use.
          if (!pool_connections_.empty()) {
            for (auto& conn : pool_connections_) {
              if (conn.ctx != nullptr) {
                buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                    .register_to_protection_domain(conn.protection_domain);
              }
            }
          }
        } else {
          // Recv buffer from rank j: register to rank j's PD.
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
          // See comment above: also register to rank j's pool QP PD.
          if (!pool_connections_.empty() &&
              pool_connections_[j].ctx != nullptr) {
            buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                .register_to_protection_domain(
                    pool_connections_[j].protection_domain);
          }
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

    // Same shape for the p2p-retry buffers: register to the dedicated
    // p2p_retry QP's PD when that QP exists for this peer (top-level
    // groups), else fall back to the data conn's PD.
    bool has_p2p_retry_for_peer = !p2p_retry_connections_.empty() &&
        p2p_retry_connections_[j].ctx != nullptr;
    auto* p2p_retry_pd = has_p2p_retry_for_peer
        ? p2p_retry_connections_[j].protection_domain
        : connections_[j].protection_domain;
    for (int s = 0; s < MeshImpl::P2P_RETRY_NUM_SLOTS; s++) {
      p2p_retry_send_buffers_[j * MeshImpl::P2P_RETRY_NUM_SLOTS + s]
          .register_to_protection_domain(p2p_retry_pd);
      p2p_retry_recv_buffers_[j * MeshImpl::P2P_RETRY_NUM_SLOTS + s]
          .register_to_protection_domain(p2p_retry_pd);
    }
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
