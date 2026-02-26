#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

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
