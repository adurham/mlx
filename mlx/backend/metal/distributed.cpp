#include "mlx/backend/metal/device.h"
#include "mlx/distributed/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
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
