// Copyright © 2024 Apple Inc.

#pragma once

#include <memory>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

// Forward declaration of the base group implementation.
namespace detail {
class GroupImpl;
};

/* Check if a communication backend is available */
MLX_API bool is_available();
MLX_API bool is_available(const std::string& bk);

/**
 * A distributed::Group represents a group of independent mlx processes that
 * can communicate. We must also be able to create sub-groups from a group in
 * order to define more granular communication.
 */
struct MLX_API Group {
  Group(std::shared_ptr<detail::GroupImpl> group) : group_(std::move(group)) {}

  int rank() const;
  int size() const;

  /**
   * Split the group according to the provided color. Namely processes that use
   * the same color will go to the same group.
   *
   * The key defines the rank of the processes in the new group. The smaller
   * the key the smaller the rank. If the provided key is negative, then the
   * rank in the current group is used.
   */
  Group split(int color, int key = -1) const;

  /**
   * Create a QP-less, TCP-backed sibling group intended purely for small
   * CONTROL-PLANE collectives (all_sum / all_max / all_min / all_gather on
   * tiny payloads), with its own isolated call_id namespace and its own
   * dedicated socket.
   *
   * Unlike ``split``, this allocates NO RDMA queue pairs and borrows no
   * device state from the parent group. That matters on hardware whose
   * queue-pair budget is already fully consumed by the top-level group
   * (Apple's Thunderbolt HCA reports max_qp=3 per device), where ``split``
   * can never succeed at all. Point-to-point ``send``/``recv`` and bulk
   * payloads are deliberately unsupported on the returned group.
   *
   * Only the jaccl backend supports this; other backends throw.
   */
  Group split_tcp_coord(int color) const;

  /**
   * In-place recovery of a wedged distributed transport: reset and
   * re-establish the underlying connections without tearing the process down.
   * All ranks must call this together. No-op for backends that don't support
   * it. Intended to be called after a collective raised a transport fault, to
   * resume serving without reloading anything.
   */
  void reconnect() const;

  const std::shared_ptr<detail::GroupImpl>& raw_group() const {
    return group_;
  }

 private:
  std::shared_ptr<detail::GroupImpl> group_{nullptr};
};

/**
 * Initialize the distributed backend and return the group containing all
 * discoverable processes.
 *
 * If strict is true then throw an error if we couldn't initialize the
 * distributed subsystem. Otherwise simply return a singleton group which will
 * render communication operations as no-op.
 */
MLX_API Group init(bool strict = false, const std::string& bk = "any");

} // namespace mlx::core::distributed
