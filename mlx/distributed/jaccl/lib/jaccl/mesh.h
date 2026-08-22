// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

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
      int color = 0,
      const std::string& coord_addr = std::string());

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

  // ROOT-CAUSE FIX (2026-08-21): QP-less, TCP-only coord subgroup. Replaces
  // split() for exo's get_coord_group() control-plane path, which could
  // never work over RDMA on this hardware (max_qp=3, all three already held
  // by the top-level group under Tensor sharding). Reserves an ephemeral
  // port on rank 0, publishes it over THIS group's side_channel_ (under
  // collective_mutex_, same borrow-under-the-parent's-lock pattern split()
  // uses for the QP destination exchange), and returns a CoordGroup that
  // owns a dedicated SideChannel and NOTHING device-side.
  //
  // Deliberately does NOT set has_split_ and does NOT register in
  // subgroups_: the returned group borrows no ibv_context, so the parent
  // keeps unrestricted reconnect_fresh() capability -- the only recovery
  // that clears a dead-UC wedge (see reconnect_fresh()'s comment above).
  // The RDMA split() path permanently forfeited that.
  std::shared_ptr<Group> split_tcp_coord(int color) override;

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      uint32_t call_id,
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  void initialize(const ExchangeFn& exchange);

  // (Re)build p2p_channel_ on the fixed derived port (coordinator_addr_'s
  // port + 1). Used by the ctor AND by reconnect()/reconnect_fresh() --
  // see p2p_channel_'s member comment for why the p2p retry-barrier
  // protocol needs its own TCP socket, and coordinator_addr_'s comment for
  // why a stale/desynced p2p_channel_ from a pre-fault stream must be torn
  // down and rebuilt rather than reused across a reconnect. Assigning to
  // an engaged std::optional destroys the old SideChannel first (closing
  // its fd via ~TCPSocket) before the new one binds/connects, so this is
  // safe to call with p2p_channel_ already populated.
  void rebuild_p2p_channel();

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

  // ROOT-CAUSE FIX (2026-08-20, bug #3 of the jaccl-v2 chain): the two
  // halves of making reconnect_fresh() safe when live subgroups borrow our
  // ibv contexts. Called by the PARENT's reconnect_fresh() on each live
  // borrowing subgroup, around the parent's device close/reopen:
  //
  //   release_borrowed_device_state()  -- BEFORE the parent closes its
  //     devices. Drops every MR (SharedBuffer dtors) and every PD/CQ/QP
  //     (Connection dtors) this subgroup built on top of the parent's
  //     contexts, so nothing dangles when those contexts go away. The
  //     subgroup object itself stays alive and usable; only its
  //     device-side state is torn down.
  //
  //   rebuild_on_contexts(ctxs, exchange) -- AFTER the parent has reopened
  //     its devices. Re-adopts the parent's NEW per-peer ibv_context*s and
  //     re-runs the full subgroup bring-up (PD/CQ/QP -> MR registration ->
  //     INIT -> destination exchange -> RTR/RTS), exactly as the subgroup
  //     ctor does, then re-arms the ACK recv pool behind a barrier.
  //
  // Without this pair, reconnect() saw has_split_ == true, refused
  // reconnect_fresh(), and silently degraded to the QP-only reset -- which
  // mesh.h's own reconnect_fresh() comment above documents as INSUFFICIENT
  // to clear a dead-UC wedge. It then printed "COMPLETE" on a still-dead
  // transport, and the next collective on the parent group deadlocked with
  // zero progress in both directions (all_recv=0/N, peer_in_call=0,
  // symmetric on both ranks).
  void release_borrowed_device_state();
  void rebuild_on_contexts(
      const std::vector<ibv_context*>& ctxs,
      const ExchangeFn& exchange);

  // (Re)build coord_channel_ -- the SUBGROUP's OWN dedicated TCP coordinator.
  // See coord_channel_'s member comment below for why a subgroup must not
  // share the parent's side_channel_. Called from the subgroup ctor and from
  // rebuild_on_contexts() (post-reconnect_fresh). No-op on top-level groups
  // and whenever coord_addr_ is empty (feature disabled / no address given).
  void rebuild_coord_channel();

  void allocate_buffers();

  void open_trace_file_if_enabled();
  void trace_call(uint32_t call_id, const char* op, int64_t msg_bytes);
  void trace_hash(uint32_t call_id, const void* data, int64_t n_bytes);
  // 2026-08-21: env-gated (JACCL_TRACE_TIMING=1, requires JACCL_TRACE_CALLS=1
  // to also be set since it reuses that trace file) real steady_clock
  // duration around the transport call inside a collective, appended to
  // the same per-call trace line trace_call() already writes. Distinct
  // from trace_call's own timestamp (that's just a call-ordering log
  // line, written once at dispatch with no duration). This exists to
  // decompose the moe.all_sum 34x software-overhead gap (raw jaccl wire
  // floor ~120us vs in-model sync-span average ~4094us, see
  // docs/offline-collective-microbenchmark-2026-08-21.md) into real
  // per-call jaccl-internal transport duration vs whatever overhead sits
  // OUTSIDE this call (MLX dispatch/eval-fence/rank-skew) -- a live
  // NOP-ablation attempt at getting this same signal was found unsafe
  // (destabilized the cluster) so this read-only timing probe is the
  // safer replacement path. Overhead when disabled: one env var read at
  // MeshGroup construction, then a single relaxed bool check per call.
  void trace_duration(uint32_t call_id, double transport_us);

  int rank_;
  int size_;
  int color_ = 0;
  std::optional<SideChannel> side_channel_;
  // ROOT-CAUSE FIX (2026-07-17): dedicated TCP side-channel for send()/
  // recv()'s drop-recovery retry protocol (p2p_retry_barrier), isolated
  // from side_channel_/coordinator_. side_channel_ is ALSO used every
  // single forward pass by mlx-lm's model-level mx.distributed.all_gather
  // (deepseek_v4.py, unconditional when pipeline_size>1) -- which routes
  // through reliable_all_reduce/v2 and posts its own framed messages on
  // that same socket. A retry-protocol message interleaving with an
  // all_gather message on one shared socket corrupts BOTH streams' framing
  // (confirmed: a peer's frame showed our own MAGIC bytes landed in the
  // wrong field, meaning the two logical operations' bytes literally
  // interleaved on the wire). Same isolation principle as pool_connections_
  // (the RDMA QP split earlier tonight) applied to the TCP side-channel:
  // separate socket entirely, not a shared-socket mutex+demux (which only
  // fixes local write ordering, not cross-rank operation-identity
  // confusion -- verified with Fable before implementing). Built eagerly
  // in the ctor, right after side_channel_, on a port derived
  // deterministically from the coordinator's own port (+1) -- see ctor.
  std::optional<SideChannel> p2p_channel_;
  // ROOT-CAUSE FIX (2026-08-20, bug #8 of the jaccl-v2 chain): a SUBGROUP's
  // OWN reliable TCP coordinator, used for confirmed_coord_barrier().
  //
  // Every ordered/reliable path in mesh_impl.h -- all_reduce's confirmed
  // barrier, all_gather's confirmed barrier (0a8d8a4ee), the reliable and
  // reliable-optimistic v2 paths -- is gated on `coordinator_ != nullptr`.
  // Only the TOP-LEVEL ctor ever called mesh_.set_coordinator(), so on a
  // subgroup coordinator_ stayed nullptr FOREVER and every one of those
  // gates was structurally dead: MLX_JACCL_CONFIRMED_BARRIER_PRE=1 could
  // not engage no matter what. exo's warmup control-plane all_gather
  // (generate.py, "control-plane sync" after warmup token generation) runs
  // on exactly such a subgroup via get_coord_group()/split(), so it kept
  // taking the OLD ack_sync_pre-then-immediate-send path and kept hitting
  // the send-before-peer-recv-posted UC drop that 0a8d8a4ee fixed for the
  // parent group ("all_gather STALLED ... UC completion lost").
  //
  // The subgroup must NOT share the parent's side_channel_. Two reasons,
  // both fatal:
  //   1. SideChannel is a plain framed TCP stream with NO operation-id
  //      demux, and the parent and subgroup hold SEPARATE collective
  //      mutexes -- so a subgroup barrier and a concurrent parent
  //      collective are two independent writers to one socket. That is
  //      byte-for-byte the corruption p2p_channel_ was split out to fix on
  //      2026-07-17 (see its comment above); repeating it here would just
  //      move the bug.
  //   2. call_id namespaces are per-group (each MeshGroup has its own
  //      next_call_id_), so confirmed_coord_barrier's self-verifying
  //      "all ranks report the same call_id" check would compare a
  //      subgroup call_id against a parent call_id and throw spurious
  //      DESYNC.
  // Hence: a separate socket, on its own port.
  //
  // The port is NOT derived arithmetically from the parent's (that was the
  // first design; it collides across co-hosted model instances whose base
  // ports are near each other, and it fights TIME_WAIT on rebuild).
  // Instead subgroup rank 0 reserves an ephemeral port
  // (reserve_ephemeral_port()) and split() publishes "<ip>:<port>" to every
  // rank over the PARENT's side_channel_ while the parent's
  // collective_mutex_ is already held -- the same borrow-the-parent's-
  // channel-under-the-parent's-lock pattern the QP destination exchange
  // already uses, so it introduces no new concurrency.
  std::optional<SideChannel> coord_channel_;
  // Address ("<ip>:<port>") coord_channel_ binds/connects on, stashed from
  // the subgroup ctor arg so rebuild_on_contexts() can rebuild the channel
  // after the parent's reconnect_fresh() without re-plumbing a parameter.
  // Empty on top-level groups (they use side_channel_ directly).
  std::string coord_addr_;
  // Stashed verbatim from the ctor arg so reconnect()/reconnect_fresh() can
  // rebuild p2p_channel_ on the SAME derived port (coordinator_addr's port
  // + 1) without threading a new parameter through every call site. Empty
  // on subgroups (which never have a p2p_channel_ to rebuild in the first
  // place -- see reconnect()'s has_split_ guard).
  std::string coordinator_addr_;
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
  // DESIGN DOC SECTION 37 PHASE 1 (2026-08-10): dedicated QP for send()/
  // recv()'s got-bitmask retry exchange (p2p_retry_exchange in
  // mesh_impl.h), migrated off the TCP p2p_retry_barrier (p2p_channel_
  // below, now otherwise unused -- kept alive only because bootstrap and
  // reconnect_fresh() rebuild it as part of the standard side-channel
  // lifecycle; not calling it from send()/recv() anymore is enough).
  // ISOLATED onto its own QP rather than sharing ack_connections_ or
  // pool_connections_, following the SAME established pattern that fixed
  // two prior IBV_WC_LOC_LEN_ERR collisions in this file (see
  // ack_connections_ and pool_connections_'s own member comments): a
  // differently-sized/differently-timed completion landing on a QP another
  // path also polls is exactly the class of bug this codebase has already
  // paid to learn not to reintroduce. A standing recv pool (like
  // pool_connections_'s POOL_RECV_WR) is pre-posted per peer at QP setup
  // and replenished on consumption; routing is by the in-band P2PFrameHdr
  // wire header (magic/direction_tag/seq/round/frame_index), not call_id.
  std::vector<Connection> p2p_retry_connections_;
  std::vector<SharedBuffer> ack_send_buffers_;
  std::vector<SharedBuffer> ack_recv_buffers_;
  // Buffer pools for p2p_retry_connections_ above. BOTH send and recv are
  // rotating per-peer pools, P2P_RETRY_NUM_SLOTS deep (flattened as
  // peer * P2P_RETRY_NUM_SLOTS + slot) -- NOT a single fixed slot like
  // ack_send/recv_buffers_. A single shared per-peer send slot was an
  // earlier draft that turned out to be unsafe (a consult review caught
  // it): reusing one send buffer for every frame of a multi-frame
  // bitmask requires blocking to drain each frame's completion before
  // the next post, and that blocking drain polls the SAME CQ the main
  // exchange loop polls for RECV completions -- silently dropping any
  // interleaved peer frame it saw while waiting. A per-frame rotating
  // pool (mirroring the recv side) lets sends be fire-and-forget in the
  // common case instead. See p2p_retry_send_bitmask's own comment in
  // mesh_impl.h for the full incident this fixes.
  std::vector<SharedBuffer> p2p_retry_send_buffers_;
  std::vector<SharedBuffer> p2p_retry_recv_buffers_;
  // ROOT-CAUSE FIX (2026-08-16, exo design doc Sections 112/115): DEDICATED
  // buffers for the standing data-QP recv pool (post_data_recv_pool). The
  // pool used to post into buffers_ at recv_buffer(sz=0, b, peer) -- the same
  // slots a SMALL collective lands in, because buffer_size_from_message maps
  // a tiny message (e.g. mx_barrier's all_sum(1.0)) to size class 0. Two
  // writers, one slot: the pool's standing recv and the collective's own
  // recv both target that buffer, so the peer's collective payload could DMA
  // into a slot the pool had armed (or vice versa). Under Tensor sharding
  // that corrupted the FIRST all_gather of warmup -- "all_gather STALLED ...
  // UC completion lost" -> reconnect -> segfault, 100% reproducible,
  // confirmed by live gate-toggle A/B.
  //
  // Same isolation shape as p2p_retry_recv_buffers_ / ack_recv_buffers_
  // above: the pool gets its own storage so no collective can ever share a
  // slot with it. Paired with DATA_POOL_RECV_WR (rdma.h) so completion
  // filtering can route the pool's CQEs unambiguously.
  std::vector<SharedBuffer> data_pool_recv_buffers_;
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
  // See trace_duration() above. Independent of hash_enabled_ -- both are
  // opt-in add-ons to the base JACCL_TRACE_CALLS trace file.
  bool timing_enabled_ = false;
  // Set by split(): subgroups borrow our ibv contexts.
  //
  // Historically this permanently disabled reconnect_fresh() (device close
  // + reopen), because closing a context out from under a live subgroup
  // leaves its PD/CQ/QP/MRs dangling. As of 2026-08-20 that veto is gone:
  // subgroups_ below tracks the live borrowers, and reconnect_fresh() now
  // tears their device state down before the close and rebuilds it against
  // the reopened contexts afterwards. has_split_ is retained purely as a
  // cheap "do we need to walk subgroups_" predicate.
  bool has_split_ = false;

  // Live subgroups created by split() that BORROW our per-peer
  // ibv_context*s (i.e. JACCL_SPLIT_FRESH_CTX unset -- the production
  // default). weak_ptr, not shared_ptr: split() hands ownership to the
  // Python/MLX caller, and a parent->child shared_ptr would both leak the
  // subgroup for the process lifetime and create a cycle. Expired entries
  // are pruned as they're encountered.
  //
  // Subgroups built with JACCL_SPLIT_FRESH_CTX=1 own their own contexts
  // and are deliberately NOT tracked here -- the parent's device close
  // cannot affect them.
  std::vector<std::weak_ptr<MeshGroup>> subgroups_;
};

} // namespace jaccl
