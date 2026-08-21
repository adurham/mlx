// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jaccl/group.h"
#include "jaccl/rdma.h" // SideChannel
#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

namespace jaccl {

/**
 * A QP-LESS, TCP-ONLY collective group for CONTROL-PLANE traffic.
 *
 * ROOT-CAUSE FIX (2026-08-21). Background:
 *
 * exo's `get_coord_group()` builds an isolated "coord subgroup" so that its
 * control-plane collectives (warmup sync, the per-decode-step `mx_any` task
 * agreement, MTP token-count agreement, KV-pressure all_gather) do NOT share
 * the model TP group's `next_call_id_` counter and UC FIFOs with live
 * forward-pass traffic. It did that with `MeshGroup::split()`.
 *
 * On Apple's Thunderbolt RDMA HCA `ibv_devinfo -v` reports **max_qp = 3 per
 * device**, device-wide. Under Tensor sharding the top-level MeshGroup
 * already allocates all three (`connections_`, `ack_connections_`,
 * `pool_connections_` for the reliable-optimistic v2 path), so a subgroup's
 * own `connections_` + `ack_connections_` have ZERO budget left and
 * `split()` throws `[jaccl] Couldn't create queue pair` — every single time,
 * deterministically. That is a hard hardware ceiling, not a leak, not a
 * transient: no retry/backoff can ever make it succeed.
 *
 * exo's `get_coord_group()` then swallowed that RuntimeError and fell back to
 * returning the PARENT group as the "coord group" — so in production every
 * coord-subgroup collective has always actually been running on the shared
 * model group, colliding call_ids with live model traffic. That is the root
 * cause of the deterministic warmup `all_gather STALLED ... call_id=2375`
 * and a latent correctness/contention hazard on the per-decode-step
 * collectives.
 *
 * The fix is to stop asking for RDMA resources that provably do not exist.
 * A subgroup used PURELY for tiny control-plane collectives moves a handful
 * of bytes a few times per token; it has no business owning queue pairs at
 * all. CoordGroup therefore implements the full `jaccl::Group` collective
 * surface on ONE dedicated, reliable, framed TCP socket (`SideChannel`) —
 * the same primitive `side_channel_`/`coord_channel_`/`p2p_channel_` already
 * use — and calls **no ibverbs verb of any kind**. It can never hit the
 * max_qp ceiling regardless of sharding mode or what the parent already
 * allocated.
 *
 * Properties this buys, beyond just "it works":
 *   - Its own `next_call_id_` namespace (the whole point of the subgroup),
 *     with the id carried in-band and cross-checked on every op, so a
 *     desync is a loud throw rather than a silent mispair.
 *   - Its own socket, so control-plane traffic can never interleave with
 *     the parent's framed side_channel_ stream (the corruption class that
 *     `p2p_channel_` was split out to fix on 2026-07-17).
 *   - TCP is reliable and ordered, so none of the UC silent-drop machinery
 *     (standing recv pools, ACK QPs, confirmed barriers, soft-RC ARQ,
 *     retransmit deadlines) is needed or even meaningful here.
 *   - It borrows NOTHING from the parent: no ibv_context, no PD/CQ/QP, no
 *     MRs. So creating one does NOT set the parent's `has_split_`, which
 *     means the parent keeps full `reconnect_fresh()` capability — the only
 *     recovery that actually clears a dead-UC wedge (see mesh.h:118-127).
 *     The RDMA `split()` path permanently forfeited that.
 *
 * NOT for bulk data. `send()`/`recv()` throw, and payloads are bounded by
 * MAX_BYTES below: this is a control plane, and routing a large tensor
 * through it would be a silent throughput cliff rather than an error.
 *
 * Concurrency: every op takes `mutex_`, exactly like MeshGroup's
 * `collective_mutex_`. All ranks must call the same ops in the same order —
 * the standard collective contract — and the in-band header verifies it.
 */
class CoordGroup : public Group {
 public:
  // Wire header prefixed to every payload: {MAGIC, opcode, call_id,
  // n_bytes}. All ranks all_gather their own header+payload, so each rank
  // sees every peer's header and can verify the whole group is executing the
  // SAME operation with the SAME call_id on the SAME byte count. Mirrors
  // reliable_barrier()'s framing rationale in rdma.h: a desynced stream must
  // be DETECTED and throw (=> clean re-place) rather than silently read the
  // wrong bytes forever.
  static constexpr uint32_t MAGIC = 0x4a43475fu; // "JCG_"
  static constexpr uint32_t HEADER_BYTES = 16;

  enum Op : uint32_t {
    OP_ALL_SUM = 1,
    OP_ALL_MAX = 2,
    OP_ALL_MIN = 3,
    OP_ALL_GATHER = 4,
    OP_BARRIER = 5,
  };

  // Sanity bound on a single control-plane payload. exo's real coord call
  // sites move at most a few hundred bytes (a 3-int32 agreement vector, a
  // per-request presence array, a task-id all_gather of 36-byte UUIDs, an
  // MTP draft-token broadcast). 1 MiB is orders of magnitude above that
  // while still turning "somebody routed a model tensor through the control
  // plane" into an immediate, explanatory error instead of a mystery stall.
  static constexpr size_t MAX_BYTES = 1u << 20;

  CoordGroup(int rank, int size, const std::string& addr)
      : rank_(rank), size_(size), channel_(rank, size, addr.c_str()) {
    // Same generous elapsed-no-progress deadline coord_channel_ uses: this
    // socket legitimately blocks while waiting for the peer to reach the
    // same collective, which under load can be much longer than a data-path
    // transfer. See MeshGroup::rebuild_coord_channel().
    const double data_path_deadline = [] {
      const char* e = std::getenv("MLX_JACCL_RECV_RETRY_DEADLINE_SECS");
      return e ? std::atof(e) : 60.0;
    }();
    const double coord_deadline = [&] {
      const char* e = std::getenv("MLX_JACCL_COORD_RECV_RETRY_DEADLINE_SECS");
      return e ? std::atof(e) : 2.0 * data_path_deadline + 30.0;
    }();
    channel_.set_recv_retry_deadline_secs(coord_deadline);
    std::fprintf(
        stderr,
        "[jaccl] tcp coord group rank=%d size=%d ready on %s (no QPs "
        "allocated)\n",
        rank_,
        size_,
        addr.c_str());
    std::fflush(stderr);
    // Fence construction on both ranks before either can return and issue a
    // collective. TCP wouldn't silently drop an early message the way UC
    // does, but keeping the same bring-up shape as every other group in this
    // file means the first real op can't be the thing that surfaces a
    // half-built channel.
    channel_.barrier();
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override {
    reduce(OP_ALL_SUM, input, output, n_bytes, dtype);
  }

  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override {
    reduce(OP_ALL_MAX, input, output, n_bytes, dtype);
  }

  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override {
    reduce(OP_ALL_MIN, input, output, n_bytes, dtype);
  }

  void all_gather(const void* input, void* output, size_t n_bytes) override {
    std::lock_guard<std::mutex> guard(mutex_);
    auto peers = exchange(OP_ALL_GATHER, input, n_bytes);
    // Concatenation in rank order, matching MeshImpl::all_gather's contract
    // (output is size_ * n_bytes, rank i's contribution at i * n_bytes).
    char* out = static_cast<char*>(output);
    for (int i = 0; i < size_; i++) {
      std::memcpy(out + static_cast<size_t>(i) * n_bytes, peers[i].data(), n_bytes);
    }
  }

  // Control plane only: point-to-point bulk transfer has no business here,
  // and silently servicing it over a single coordinator socket would be a
  // throughput cliff rather than an error. Fail loudly instead.
  void send(const void* input, size_t n_bytes, int dst) override {
    throw std::runtime_error(
        "[jaccl] send() is not supported on a TCP-only coord group -- it "
        "exists solely for small control-plane collectives (all_sum / "
        "all_max / all_min / all_gather / barrier). Use the top-level "
        "group for point-to-point data transfer.");
  }

  void recv(void* output, size_t n_bytes, int src) override {
    throw std::runtime_error(
        "[jaccl] recv() is not supported on a TCP-only coord group -- see "
        "send()'s message.");
  }

  void barrier() override {
    std::lock_guard<std::mutex> guard(mutex_);
    char dummy = 0;
    (void)exchange(OP_BARRIER, &dummy, 1);
  }

  // Nothing to recover: TCP is reliable and ordered, so there is no UC-wedge
  // analogue here. A genuinely broken socket surfaces as a throw from
  // recv()/send() rather than a silent stall, and the correct response to
  // that is a re-place, not an in-place QP reset. Deliberately a no-op (the
  // Group base default) rather than a throw, so a caller that reconnects the
  // whole cluster doesn't have to special-case us.
  void reconnect() override {}

  std::shared_ptr<Group> split(int color, int key = -1) override {
    throw std::runtime_error(
        "[jaccl] Cannot split a TCP-only coord group further.");
  }

 private:
  // Build header+payload, all_gather it over the dedicated socket, verify
  // every peer's header agrees with ours, and return the peers' payloads.
  std::vector<std::vector<char>>
  exchange(Op op, const void* input, size_t n_bytes) {
    if (n_bytes > MAX_BYTES) {
      std::ostringstream m;
      m << "[jaccl] TCP-only coord group payload of " << n_bytes
        << " bytes exceeds the control-plane limit of " << MAX_BYTES
        << " bytes. This group is for small control-plane collectives; a "
           "payload this large belongs on the top-level RDMA group.";
      throw std::runtime_error(m.str());
    }
    const uint32_t call_id = next_call_id_++;
    std::vector<char> frame(HEADER_BYTES + n_bytes);
    const uint32_t hdr[4] = {
        MAGIC,
        static_cast<uint32_t>(op),
        call_id,
        static_cast<uint32_t>(n_bytes)};
    std::memcpy(frame.data(), hdr, HEADER_BYTES);
    if (n_bytes) {
      std::memcpy(frame.data() + HEADER_BYTES, input, n_bytes);
    }

    auto gathered = channel_.all_gather(frame);

    std::vector<std::vector<char>> payloads;
    payloads.reserve(static_cast<size_t>(size_));
    for (int i = 0; i < size_; i++) {
      auto& g = gathered[i];
      if (g.size() != frame.size()) {
        throw desync(i, op, call_id, n_bytes, "frame size", g.size());
      }
      uint32_t rhdr[4];
      std::memcpy(rhdr, g.data(), HEADER_BYTES);
      if (rhdr[0] != MAGIC || rhdr[1] != static_cast<uint32_t>(op) ||
          rhdr[2] != call_id || rhdr[3] != static_cast<uint32_t>(n_bytes)) {
        std::ostringstream m;
        m << "[jaccl] TCP coord group DESYNC: rank " << rank_
          << " is executing op=" << static_cast<uint32_t>(op)
          << " call_id=" << call_id << " n_bytes=" << n_bytes
          << " but rank " << i << " reported magic=0x" << std::hex << rhdr[0]
          << std::dec << " op=" << rhdr[1] << " call_id=" << rhdr[2]
          << " n_bytes=" << rhdr[3]
          << ". The ranks are not executing the same sequence of coord "
             "collectives -- treating as a hard fault rather than silently "
             "mispairing control-plane values.";
        throw std::runtime_error(m.str());
      }
      payloads.emplace_back(g.begin() + HEADER_BYTES, g.end());
    }
    return payloads;
  }

  std::runtime_error desync(
      int peer,
      Op op,
      uint32_t call_id,
      size_t n_bytes,
      const char* what,
      size_t got) {
    std::ostringstream m;
    m << "[jaccl] TCP coord group DESYNC on rank " << rank_ << ": peer "
      << peer << " " << what << "=" << got << " for op="
      << static_cast<uint32_t>(op) << " call_id=" << call_id
      << " n_bytes=" << n_bytes;
    return std::runtime_error(m.str());
  }

  void
  reduce(Op op, const void* input, void* output, size_t n_bytes, int dtype) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto peers = exchange(op, input, n_bytes);
    // Seed the output with our own contribution, then fold in every peer's
    // via the SAME reduce functors MeshGroup uses (SumOp/MaxOp/MinOp from
    // reduction_ops.h) so all_sum/all_max/all_min semantics -- including the
    // bf16 specializations -- stay identical to the RDMA path by
    // construction rather than by reimplementation.
    std::memcpy(output, input, n_bytes);
    dispatch_all_types(dtype, [&](auto type_tag) {
      using T = JACCL_GET_TYPE(type_tag);
      const size_t n = n_bytes / sizeof(T);
      T* out = static_cast<T*>(output);
      for (int i = 0; i < size_; i++) {
        if (i == rank_) {
          continue;
        }
        const T* in = reinterpret_cast<const T*>(peers[i].data());
        switch (op) {
          case OP_ALL_SUM:
            SumOp<T>{}(in, out, n);
            break;
          case OP_ALL_MAX:
            MaxOp<T>{}(in, out, n);
            break;
          case OP_ALL_MIN:
            MinOp<T>{}(in, out, n);
            break;
          default:
            throw std::runtime_error(
                "[jaccl] TCP coord group: reduce() called with a non-reduce "
                "opcode.");
        }
      }
    });
  }

  int rank_;
  int size_;
  SideChannel channel_;
  std::mutex mutex_;
  // Private call_id namespace -- the entire reason exo wants a coord
  // subgroup in the first place. Purely local; correctness comes from every
  // rank incrementing it in lockstep, which exchange() verifies in-band.
  uint32_t next_call_id_ = 1;
};

} // namespace jaccl
