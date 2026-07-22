// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core {

namespace metal {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}
uint64_t dispatch_count() {
  return 0;
}
void reset_dispatch_count() {}
uint64_t gpu_time_ns() {
  return 0;
}
void reset_gpu_time() {}
bool gpu_time_enabled() {
  return false;
}
void accumulate_gpu_time_ns(uint64_t) {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

void set_metallib_path(const std::string& path) {}

const std::string& get_metallib_path() {
  throw std::runtime_error(
      "[metal::get_metallib_path] Cannot get metallib path without metal backend");
}

} // namespace metal

} // namespace mlx::core
