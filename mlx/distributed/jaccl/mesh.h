// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <mutex>

#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/jaccl/mesh_impl.h"
#include "mlx/distributed/jaccl/ring_impl.h"
#include "mlx/distributed/jaccl/utils.h"

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

namespace mlx::core::distributed::jaccl {

/**
 * The JACCL communication group for a fully connected mesh. We expect one
 * connection per peer and it should be the lowest latency communication group
 * for small to medium size messages.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class MeshGroup : public GroupImpl {
 public:
  MeshGroup(
      int rank,
      const std::vector<std::string>& device_names,
      const char* coordinator_addr);

  Stream communication_stream(StreamOrDevice s) override {
    // Pin every collective on this group to ONE shared process-wide
    // CPU stream — NOT the caller's stream and NOT default_stream()
    // (which is thread_local: different caller threads get different
    // streams, defeating the whole point of pinning).
    //
    // Why pinning at all: cpu::get_command_encoder is keyed by
    // stream.index. If model-attention all_sums dispatch on a
    // GPU-derived stream and agree_on_tasks runs on the CPU default
    // stream, those go to different encoder worker threads. The
    // per-group mutex still serializes mesh_.X() bodies, but the
    // order in which lambdas reach the mutex is decided by which
    // encoder thread races faster — which varies between rank 0 and
    // rank 1. The QP then sees a different post_send / post_recv
    // interleaving on each rank, and UC's per-QP FIFO matching
    // corrupts cross-collective sends into the wrong recv buffers
    // (the c=2+γ=2 MTP corruption mechanism, surfaced by our
    // wc.status=IBV_WC_LOC_LEN_ERR diagnostic).
    //
    // Pinning to a single owned stream forces one FIFO dispatch
    // queue per rank; both ranks run the same user code in the same
    // order, so their queues match.
    (void)s;
    return communication_stream_;
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input, array& output, Stream stream) override;
  void all_max(const array& input, array& output, Stream stream) override;
  void all_min(const array& input, array& output, Stream stream) override;
  void all_gather(const array& input, array& output, Stream stream) override;
  void send(const array& input, int dst, Stream stream) override;
  void recv(array& out, int src, Stream stream) override;

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] sum_scatter not supported.");
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const array& input,
      array& output,
      Stream stream,
      ReduceOp reduce_op);

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state and all buffers
   * should have been allocated.
   */
  void initialize();

  /**
   * Allocate all the buffers that we will use in the communication group.
   */
  void allocate_buffers();

  int rank_;
  int size_;
  // One shared CPU stream owned by this group. Created once at
  // construction so every caller thread gets the SAME stream from
  // communication_stream(); all collectives funnel through one
  // cpu::CommandEncoder and serialize FIFO. See the comment on
  // communication_stream() for why default_stream() doesn't work.
  Stream communication_stream_;
  SideChannel side_channel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
  std::vector<SharedBuffer> ring_send_buffers_;
  std::vector<SharedBuffer> ring_recv_buffers_;

  MeshImpl mesh_;
  RingImpl ring_;

  // MeshImpl and RingImpl are NOT thread-safe: they share `connections_`
  // (one ibv_cq per peer) and `buffers_` (one (sz, buff) slot per peer).
  // MLX dispatches each collective on the CPU command encoder for the
  // input array's stream, and `cpu::get_command_encoder` keyes its
  // encoder map by stream.index — so two collectives issued from
  // different streams (e.g. a CPU-stream `all_gather` of task counts
  // racing a GPU-stream `all_reduce` from MTP's verify forward) run
  // on different worker threads. Without serialization the polling
  // loops drain each other's completions out of the shared CQ and
  // post into each other's buffer slots, producing the float-bit-
  // pattern garbage seen at c=2 + MTP. DSB barriers (added in the
  // earlier coherency fix) only protect a single collective's CPU-NIC
  // handshake; they do not address this concurrent-call race. Serialize
  // every collective on this group behind one mutex.
  std::mutex collective_mutex_;

  // Per-collective monotonic id, encoded in the high 32 bits of every
  // wr_id we post. Ensures that any completion that surfaces in the
  // shared CQ from a prior call (whether because the prior call leaked
  // a completion, or because a recv WR it posted was matched by a
  // peer's send after the prior call returned) is detected by id
  // mismatch and skipped during polling — instead of being interpreted
  // as ours and read out of a buffer slot we never wrote to. Starts at
  // 1 so 0 stays an obvious sentinel for "no call yet."
  std::atomic<uint32_t> next_call_id_{1};

  uint32_t next_call_id() {
    return next_call_id_.fetch_add(1, std::memory_order_relaxed);
  }
};

} // namespace mlx::core::distributed::jaccl
