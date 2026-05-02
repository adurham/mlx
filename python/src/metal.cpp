// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def(
      "dispatch_count",
      &mx::metal::dispatch_count,
      R"pbdoc(
      Total number of GPU kernel dispatches issued since process start or
      the last reset. Counts every ``dispatch_threadgroups`` and
      ``dispatch_threads`` call across all command encoders. Intended for
      fused-kernel validation and dispatch-scheduling diagnostics.

      Gated on the ``MLX_DISPATCH_COUNT=1`` env var (default off). When
      unset, always returns 0 — the dispatch hot path stays free of
      counter traffic in production. Set the env var before importing
      ``mlx`` to enable.

      Call ``mx.eval(...)`` before reading so in-flight work is flushed.
      )pbdoc");
  metal.def(
      "reset_dispatch_count",
      &mx::metal::reset_dispatch_count,
      R"pbdoc(
      Reset the global dispatch counter to zero.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });
  metal.def(
      "live_array_desc_count",
      []() { return mx::live_array_desc_count(); },
      R"pbdoc(
      Diagnostic: snapshot the live ``ArrayDesc`` instance counter (fork-only).

      Returns the current value of an atomic counter that is incremented on
      each ``ArrayDesc`` construction and decremented on destruction. Intended
      for memory-leak hunts where we need to compare a "live count" reading
      against heap snapshots taken at the same moment.

      The counter is process-global and lock-free. Reading it is cheap; safe
      to call from the decode hot loop.
      )pbdoc");
  metal.def(
      "live_array_desc_count_by_type",
      []() {
        // Convert (string, int64) pairs into a Python dict so callers can
        // do `d.get("mlx::core::SliceUpdate", 0)` directly.
        auto pairs = mx::live_array_desc_count_by_type();
        std::unordered_map<std::string, int64_t> out;
        out.reserve(pairs.size());
        for (auto& p : pairs) {
          out.emplace(std::move(p.first), p.second);
        }
        return out;
      },
      R"pbdoc(
      Diagnostic: per-primitive-type live ``ArrayDesc`` counts (fork-only).

      Returns a ``dict[str, int]`` mapping demangled primitive class name
      (e.g. ``"mlx::core::SliceUpdate"``) to live ArrayDesc count for every
      primitive type that has been constructed since process start.

      Empty when neither ``MLX_PER_TYPE_TRACK`` nor
      ``MLX_PER_TYPE_DUMP_INTERVAL`` were set in the environment at startup.

      Snapshot via a brief mutex; safe to call periodically (e.g. once per
      memory-profile sample) but not in the per-step decode hot loop.
      )pbdoc");
}
