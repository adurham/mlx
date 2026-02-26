#include "mlx/backend/metal/device.h"
#include "mlx/distributed/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

// Helper to dynamically check if we should force distributed ops onto the GPU stream.
// By default (or if MLX_FORCE_DISTRIBUTED_GPU=1), we wrap CPU network calls inside the GPU 
// evaluation graph to achieve ~1ms orchestration latency.
// However, for massive contexts (>50K tokens), other nodes might take several seconds to compute,
// causing this GPU-blocking wait to exceed macOS's strict Metal Command Buffer Timeout (2-5s),
// resulting in a fatal kIOGPUCommandBufferCallbackErrorTimeout.
// When MLX_FORCE_DISTRIBUTED_GPU=0, we throw an error, causing MLX to gracefully fall back
// to pure CPU execution for the network primitive. This avoids blocking the GPU, 
// at the cost of ~10-30ms of orchestration overhead.
inline bool force_distributed_gpu() {
  // Read dynamically every time to allow on-the-fly toggling during execution
  if (const char* env_p = std::getenv("MLX_FORCE_DISTRIBUTED_GPU")) {
    return std::stoi(env_p) != 0;
  }
  return true; // Default to fast GPU delegation
}

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (force_distributed_gpu()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[AllReduce::eval_gpu] Safe Sync fallback requested.");
  }
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (force_distributed_gpu()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[AllGather::eval_gpu] Safe Sync fallback requested.");
  }
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (force_distributed_gpu()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[Send::eval_gpu] Safe Sync fallback requested.");
  }
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (force_distributed_gpu()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[Recv::eval_gpu] Safe Sync fallback requested.");
  }
}

void ReduceScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (force_distributed_gpu()) {
    eval_cpu(inputs, outputs);
  } else {
    throw std::runtime_error("[ReduceScatter::eval_gpu] Safe Sync fallback requested.");
  }
}

} // namespace mlx::core::distributed
