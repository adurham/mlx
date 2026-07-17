// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>

#include "jaccl/group.h"
#include "jaccl/mesh_impl.h"
#include "jaccl/rdma.h"
#include "jaccl/ring_impl.h"

namespace jaccl {

/**
 * The JACCL communication group for a fully connected mesh. We expect one
 * connection per peer and it should be the lowest latency communication group
 * for small to medium size messages.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class MeshGroup : public Group {
 public:
  // QP-destination exchange callback used by initialize(). Top-level
  // groups pass a lambda that uses the local SideChannel; subgroups
  // built by split() pass a lambda that uses the PARENT's SideChannel
  // under the parent's collective_mutex_.
  using ExchangeFn =
      std::function<std::vector<std::vector<Destination>>(
          const std::vector<Destination>&)>;

  MeshGroup(
      int rank,
      const std::vector<std::string>& device_names,
      const std::string& coordinator_addr);

  // Subgroup ctor used by split(). Builds Connections that BORROW the
  // parent's per-peer ibv_context (one open per device on the system —
  // macOS librdma's second ibv_open_device on the same device does
  // not return a fully-isolated context), then runs the same init as
  // the top-level path: allocate PD/CQ/QP per peer, register MRs to
  // those PDs (via allocate_buffers), then INIT/RTR/RTS the QPs with
  // destinations exchanged through `exchange` (which the caller wires
  // up to the parent's SideChannel under the parent's collective
  // mutex). The order is critical: register MRs BEFORE INIT/RTR/RTS,
  // since macOS librdma locks the QP's MR table at the INIT
  // transition.
  MeshGroup(
      int rank,
      int size,
      std::vector<ibv_context*> ctxs,
      std::vector<std::string> device_names,
      bool owns_ctxs,
      const ExchangeFn& exchange,
      int color = 0);

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_gather(const void* input, void* output, size_t n_bytes) override;

  void send(const void* input, size_t n_bytes, int dst) override;
  void recv(void* output, size_t n_bytes, int src) override;

  void barrier() override;

  // In-place recovery: reset + re-establish the QPs (top-level group only)
  // without destroying PD/CQ/MRs, to clear a UC transport wedge without a
  // full runner re-place. Both ranks must call it.
  void reconnect() override;

  // split(color, key) — current implementation requires all ranks to
  // call with the same color (single subgroup per call). Cannot be called
  // on a subgroup (subgroup has no SideChannel for the destination exchange).
  std::shared_ptr<Group> split(int color, int key = -1) override;

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      uint32_t call_id,
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  void initialize(const ExchangeFn& exchange);

  // Hard recovery: close and reopen the ibv device contexts and rebuild
  // EVERYTHING device-side (PD/CQ/QP, buffer MR registrations, MeshImpl/
  // RingImpl views) — the in-process equivalent of a runner respawn. The
  // dead-UC-data-path wedge observed on Apple librdma (2026-07-06: stuck
  // send CQE, all_recv=0, one or both directions) survives
  // queue_pair_reset() — which preserves PD/CQ/MRs/ctx — but clears with
  // a fresh ibv_open_device, so this is the recovery that actually works.
  // Top-level, PRE-SPLIT groups only: subgroups borrow our contexts, so
  // closing them under an existing subgroup would leave it dangling
  // (reconnect() falls back to the QP-only reset in that case). Gated by
  // MLX_JACCL_RECONNECT_FRESH=1 inside reconnect().
  void reconnect_fresh();

  void allocate_buffers();

  void open_trace_file_if_enabled();
  void trace_call(uint32_t call_id, const char* op, int64_t msg_bytes);
  void trace_hash(uint32_t call_id, const void* data, int64_t n_bytes);

  int rank_;
  int size_;
  int color_ = 0;
  std::optional<SideChannel> side_channel_;
  std::vector<std::string> device_names_;
  std::vector<Connection> connections_;
  std::vector<Connection> ack_connections_;
  // ROOT-CAUSE FIX (2026-07-17): dedicated QP for the jaccl-v2 reliable-ARQ
  // optimistic path (reliable_all_reduce_v2's standing POOL_RECV_WR pool,
  // gated by MLX_JACCL_RELIABLE_OPTIMISTIC/MLX_JACCL_RELIABLE_DATA -- built
  // because Apple's RDMA stack doesn't support hardware RC connections, so
  // this software layer provides reliability over UC for TP's collectives).
  // It used to share connections_ (the same QP raw send()/recv() posts on
  // for exo's Pipeline-Parallel p2p handoff). The pool's recv buffers use
  // ONE uniform size class; send()/recv() post buffers sized per-message.
  // When a send()/recv() work request landed on a QP slot the hardware
  // still associated with one of the pool's differently-sized posted
  // recvs (or vice versa), it threw IBV_WC_LOC_LEN_ERR -- and since both
  // paths only filtered completions by call_id (not work_type), an errored
  // pool slot was silently discarded without being re-armed, eventually
  // exhausting the pool and wedging BOTH paths. Isolating the pool onto
  // its own QP (same pattern as ack_connections_ below: borrows the peer's
  // ibv_context, owns its own PD/CQ/QP) removes the collision entirely --
  // same fix shape as the ACK-QP split that solved an identical problem
  // for collective barriers sharing the data QP (see MeshGroup ctor
  // comment, "2026-05-17: restore dedicated ACK QP").
  std::vector<Connection> pool_connections_;
  std::vector<SharedBuffer> ack_send_buffers_;
  std::vector<SharedBuffer> ack_recv_buffers_;
  std::vector<SharedBuffer> buffers_;
  std::vector<SharedBuffer> ring_send_buffers_;
  std::vector<SharedBuffer> ring_recv_buffers_;

  MeshImpl mesh_;
  RingImpl ring_;

  std::mutex collective_mutex_;

  std::atomic<uint32_t> next_call_id_{1};
  uint32_t next_call_id() {
    return next_call_id_.fetch_add(1, std::memory_order_relaxed);
  }

  FILE* trace_file_ = nullptr;
  bool hash_enabled_ = false;
  // Set by split(): subgroups borrow our ibv contexts, which makes
  // reconnect_fresh() (device close + reopen) unsafe from then on.
  bool has_split_ = false;
};

} // namespace jaccl
