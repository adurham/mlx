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
 * last call to ``reset_dispatch_count()``. Counts every
 * ``dispatch_threadgroups`` and ``dispatch_threads`` call across all
 * command encoders. Intended for fused-kernel validation and dispatch
 * diagnostics. Call ``eval()`` first so in-flight work is flushed. */
MLX_API uint64_t dispatch_count();
MLX_API void reset_dispatch_count();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

} // namespace mlx::core::metal
