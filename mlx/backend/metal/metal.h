// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Total number of GPU kernel dispatches issued since process start or the
 * last reset. Counts `dispatch_threadgroups` and `dispatch_threads` calls
 * across all command encoders. Intended for fused-kernel validation /
 * dispatch-scheduling diagnostics.
 *
 * Gated on the ``MLX_DISPATCH_COUNT`` env var (default off) so the dispatch
 * hot path stays branch-predicted-free in production. When the env var is
 * unset or 0, ``dispatch_count()`` always returns 0. */
MLX_API uint64_t dispatch_count();
MLX_API void reset_dispatch_count();

/** Total GPU-busy time accumulated across all completed command buffers,
 * in nanoseconds. Sum of (GPUEndTime - GPUStartTime) per command buffer
 * since the last reset. Designed for cycle-level utilization profiling:
 * cycle_busy_pct = gpu_time_ns() / wall_ns.
 *
 * Gated on the ``MLX_GPU_TIME`` env var (default off) so completion
 * handlers stay branch-predicted-free in production. When unset or 0,
 * ``gpu_time_ns()`` always returns 0. */
MLX_API uint64_t gpu_time_ns();
MLX_API void reset_gpu_time();
MLX_API bool gpu_time_enabled();
MLX_API void accumulate_gpu_time_ns(uint64_t ns);

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

} // namespace mlx::core::metal
