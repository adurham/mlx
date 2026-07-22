// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <cstdio>
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

/** exo-stall-diag (2026-07-21): dumps recent command-buffer commit/schedule/
 * complete timestamps + status to `out`, for the PP+DSpark GPU-idle-stall
 * investigation (repeated 15-47s decode stalls where `sample`/ioreg polling
 * showed the GPU genuinely idle and the main thread parked in
 * Event::wait()/wait_for_one(), with no prior visibility into whether a
 * command buffer had even been submitted for the awaited event).
 *
 * Gated by the ``EXO_CMDBUF_RING_DIAG`` env var (opt-in, zero overhead when
 * unset -- both the ring-buffer bookkeeping in CommandEncoder::commit() and
 * this dump no-op; callers don't need their own gate). Intended to be
 * called from Event::wait()'s existing "slow wait" diagnostic
 * (backend/metal/event.cpp) once a wait has been stuck for several
 * seconds, to answer in one shot: was a command buffer even committed for
 * the stuck event; if so, is it stuck at Committed (handed to Metal's
 * queue but the driver hasn't scheduled it), Scheduled-but-not-running
 * (driver has it, GPU hasn't started), or genuinely mid-execution
 * (GPUStartTime set, no GPUEndTime yet)? Answering this distinguishes an
 * MLX/graph-scheduling stall from a Metal-queue/driver-level stall from
 * true (if unusually slow) GPU execution, without needing separate tools'
 * output correlated after the fact via wall-clock timestamps. */
MLX_API void dump_recent_command_buffers(FILE* out);

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

} // namespace mlx::core::metal
