// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Delegate to CPU implementation.
  // On Apple Silicon unified memory, GPU and CPU share the same physical
  // memory so no copy is needed — only synchronization.  The JACCL/Ring
  // backends perform RDMA on the CPU; this delegation lets the MLX scheduler
  // use fence-based GPU↔CPU stream sync instead of refusing the operation.
  eval_cpu(inputs, outputs);
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval_cpu(inputs, outputs);
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval_cpu(inputs, outputs);
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval_cpu(inputs, outputs);
}

void ReduceScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval_cpu(inputs, outputs);
}

} // namespace mlx::core::distributed
