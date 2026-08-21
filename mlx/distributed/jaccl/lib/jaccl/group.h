// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace jaccl {

/**
 * Abstract base class for a JACCL communication group.
 */
class Group {
 public:
  virtual ~Group() {}

  virtual int rank() = 0;
  virtual int size() = 0;

  virtual void
  all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
  virtual void barrier() = 0;

  // In-place recovery of a wedged UC transport: reset + re-establish QPs
  // without a full re-place. Default no-op (only the top-level mesh supports
  // it); overridden by MeshGroup.
  virtual void reconnect() {}

  virtual std::shared_ptr<Group> split(int color, int key = -1) {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

  // Create a QP-LESS, TCP-only sibling group for small CONTROL-PLANE
  // collectives (all_sum / all_max / all_min / all_gather / barrier on tiny
  // payloads), with its own call_id namespace and its own dedicated socket.
  //
  // Unlike split(), this allocates NO RDMA queue pairs and borrows no device
  // state from the parent. That is the entire point: Apple's Thunderbolt HCA
  // reports max_qp=3 per device, and under Tensor sharding the top-level
  // group already holds all three -- so split() can NEVER succeed there.
  // See CoordGroup's class comment in coord_group.h for the full root cause.
  //
  // Default: not supported (backends without a TCP coordinator).
  virtual std::shared_ptr<Group> split_tcp_coord(int color) {
    throw std::runtime_error(
        "[jaccl] TCP-only coord group not supported by this backend.");
  }
};

/**
 * Type IDs for dispatch in the standalone JACCL library.
 *
 * Users pass one of these to all_sum/all_max/all_min so JACCL knows how to
 * interpret the data for typed reduction operations.
 */
enum Dtype {
  Bool = 0,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  BFloat16,
  Float32,
  Float64,
  Complex64,
};

} // namespace jaccl
