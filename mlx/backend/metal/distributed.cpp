#include "mlx/backend/metal/device.h"
#include "mlx/distributed/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (env::metal_fast_synch()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[AllReduce::eval_gpu] has no GPU implementation.");
  }
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (env::metal_fast_synch()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[AllGather::eval_gpu] has no GPU implementation.");
  }
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (env::metal_fast_synch()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[Send::eval_gpu] has no GPU implementation.");
  }
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (env::metal_fast_synch()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[Recv::eval_gpu] has no GPU implementation.");
  }
}

void ReduceScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (env::metal_fast_synch()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[ReduceScatter::eval_gpu] has no GPU implementation.");
  }
}

} // namespace mlx::core::distributed
