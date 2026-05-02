// Copyright © 2023-2024 Apple Inc.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

array::array(const std::complex<float>& val, Dtype dtype /* = complex64 */)
    : array_desc_(std::make_shared<ArrayDesc>(Shape{}, dtype)) {
  auto cval = static_cast<complex64_t>(val);
  init(&cval);
}

array::array(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              std::move(shape),
              dtype,
              std::move(primitive),
              std::move(inputs))) {
  if (has_primitive() && this->primitive().stream().device == Device::gpu) {
    for (auto& in : this->inputs()) {
      if (in.dtype() == float64) {
        throw std::invalid_argument("float64 is not supported on the GPU");
      }
    }
    if (this->dtype() == float64) {
      throw std::invalid_argument("float64 is not supported on the GPU");
    }
  }
}

std::vector<array> array::make_arrays(
    std::vector<Shape> shapes,
    const std::vector<Dtype>& dtypes,
    const std::shared_ptr<Primitive>& primitive,
    const std::vector<array>& inputs) {
  std::vector<array> outputs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    outputs.emplace_back(std::move(shapes[i]), dtypes[i], primitive, inputs);
  }
  // For each node in |outputs|, its siblings are the other nodes.
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto siblings = outputs;
    siblings.erase(siblings.begin() + i);
    outputs[i].set_siblings(std::move(siblings), i);
  }
  return outputs;
}

array array::unsafe_weak_copy(const array& other) {
  auto cpy = array(other.shape(), other.dtype(), nullptr, {});
  cpy.set_data(
      other.buffer(),
      other.data_size(),
      other.strides(),
      other.flags(),
      [](auto) {});
  cpy.array_desc_->offset = other.array_desc_->offset;
  return cpy;
}

array::array(std::initializer_list<float> data)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              float32)) {
  init(data.begin());
}

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              dtype)) {
  init(data.begin());
}

array::array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  auto buffer = allocator::make_buffer(data, nbytes());
  if (buffer.ptr() == nullptr) {
    set_data(allocator::malloc(nbytes()));
    auto ptr = static_cast<char*>(data);
    std::copy(ptr, ptr + nbytes(), this->data<char>());
    deleter(data);
  } else {
    auto wrapped_deleter = [deleter](allocator::Buffer buffer) {
      auto ptr = buffer.raw_ptr();
      allocator::release(buffer);
      return deleter(ptr);
    };
    set_data(buffer, std::move(wrapped_deleter));
  }
}

/* Build an array from a shared buffer */
array::array(allocator::Buffer data, Shape shape, Dtype dtype, Deleter deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  set_data(data, deleter);
}

void array::detach() {
  array_desc_->primitive = nullptr;
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->primitive = nullptr;
  }
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->inputs.clear();
    s.array_desc_->siblings.clear();
    s.array_desc_->position = 0;
  }
  array_desc_->inputs.clear();
  array_desc_->siblings.clear();
  array_desc_->position = 0;
}

bool array::is_available() const {
  if (status() == Status::available) {
    return true;
  } else if (
      status() == Status::evaluated &&
      (!event().valid() || event().is_signaled())) {
    detach_event();
    set_status(Status::available);
    return true;
  }
  return false;
}

void array::wait() {
  if (!is_available()) {
    if (event().valid()) {
      event().wait();
      detach_event();
    }
    set_status(Status::available);
  }
}

void array::eval() {
  // Ensure the array is ready to be read
  if (status() == Status::unscheduled) {
    mlx::core::eval({*this});
  } else {
    wait();
  }
}

bool array::is_tracer() const {
  return (array_desc_->is_tracer && detail::in_tracing()) ||
      detail::retain_graph();
}

void array::set_data(allocator::Buffer buffer, Deleter d) {
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->offset = 0;
  array_desc_->data_size = size();
  array_desc_->flags.contiguous = true;
  array_desc_->flags.row_contiguous = true;
  auto max_dim = std::max_element(shape().begin(), shape().end());
  array_desc_->flags.col_contiguous = size() <= 1 || size() == *max_dim;
}

void array::set_data(
    allocator::Buffer buffer,
    size_t data_size,
    Strides strides,
    Flags flags,
    Deleter d) {
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->offset = 0;
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const Strides& strides,
    Flags flags,
    size_t data_size,
    int64_t offset /* = 0 */) {
  array_desc_->data = other.array_desc_->data;
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  array_desc_->offset =
      sizeof(char) * itemsize() * offset + other.array_desc_->offset;
}

void array::copy_shared_buffer(const array& other) {
  copy_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::~array() {
  if (array_desc_ == nullptr) {
    return;
  }

  // Detached/detaching
  if (array_desc_->primitive == nullptr) {
    return;
  }

  // Break circular reference for non-detached arrays with siblings
  if (auto n = siblings().size(); n > 0) {
    bool do_detach = true;
    // If all siblings have siblings.size() references except
    // the one we are currently destroying (which has siblings.size() + 1)
    // then there are no more external references
    do_detach &= (array_desc_.use_count() == (n + 1));
    for (auto& s : siblings()) {
      do_detach &= (s.array_desc_.use_count() == n);
      if (!do_detach) {
        break;
      }
    }
    if (do_detach) {
      for (auto& s : siblings()) {
        for (auto& ss : s.siblings()) {
          // Set to null here to avoid descending into array destructor
          // for siblings
          ss.array_desc_ = nullptr;
        }
        s.array_desc_->siblings.clear();
      }
    }
  }
}

void array::ArrayDesc::init() {
  strides.resize(shape.size());
  size = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = size;
    size *= shape[i];
  }
  for (const auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

// Read MLX_LOG_ARRAY_DESC_COUNT_INTERVAL once at first use.
static int64_t array_desc_log_interval() {
  static int64_t interval = []() -> int64_t {
    if (const char* env = std::getenv("MLX_LOG_ARRAY_DESC_COUNT_INTERVAL")) {
      return std::strtoll(env, nullptr, 10);
    }
    return 0;
  }();
  return interval;
}

// Read MLX_PER_TYPE_DUMP_INTERVAL once at first use. When > 0, the
// stderr dump fires every Nth ArrayDesc construction (the dump path is
// independent of the counting path; counting is gated separately on
// per_type_tracking_enabled).
static int64_t per_type_dump_interval() {
  static int64_t interval = []() -> int64_t {
    if (const char* env = std::getenv("MLX_PER_TYPE_DUMP_INTERVAL")) {
      return std::strtoll(env, nullptr, 10);
    }
    return 0;
  }();
  return interval;
}

// Read MLX_PER_TYPE_TRACK once at first use. When set (or when
// MLX_PER_TYPE_DUMP_INTERVAL > 0), the per-type live-count map is
// populated on every ArrayDesc construction and the
// `live_array_desc_count_by_type()` accessor returns non-empty data.
// Cheap-but-not-free: each construction does a mutex-protected map
// lookup the first time it sees a primitive type, and one lock-free
// atomic increment thereafter. Off by default.
static bool per_type_tracking_enabled() {
  static bool enabled = []() -> bool {
    if (per_type_dump_interval() > 0) {
      return true;
    }
    if (const char* env = std::getenv("MLX_PER_TYPE_TRACK")) {
      return std::strcmp(env, "0") != 0 && env[0] != '\0';
    }
    return false;
  }();
  return enabled;
}

// Per-primitive-type live counter. The map only gains entries when a new
// primitive type is first seen; lookups go through a mutex on the slow
// path of construction but the resulting pointer is then stored in the
// ArrayDesc so destruction is mutex-free. The atomic counter inside the
// struct gives lock-free inc/dec from any thread.
struct PrimitiveTypeCounter {
  std::string demangled_name;
  std::atomic<int64_t> live_count{0};
};

static std::mutex& per_type_mutex() {
  static std::mutex m;
  return m;
}

// Stable storage: unique_ptrs so the map can grow without invalidating
// pointers we've handed out to ArrayDescs.
static std::vector<std::unique_ptr<PrimitiveTypeCounter>>& per_type_storage() {
  static std::vector<std::unique_ptr<PrimitiveTypeCounter>> v;
  return v;
}

static std::unordered_map<std::string, PrimitiveTypeCounter*>&
per_type_index() {
  static std::unordered_map<std::string, PrimitiveTypeCounter*> m;
  return m;
}

// Demangle a typeid().name() into a human-readable type name. Returns the
// mangled string on demangle failure.
static std::string demangle_type_name(const char* mangled) {
  int status = 0;
  char* d = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && d != nullptr) {
    std::string out(d);
    std::free(d);
    return out;
  }
  if (d != nullptr) {
    std::free(d);
  }
  return std::string(mangled);
}

// Look up (or create) a stable counter for a given primitive type name.
// Slow path: takes the mutex, allocates on first sighting. Fast path
// (after the storage exists): pure map lookup. The returned pointer is
// stable for process lifetime.
static PrimitiveTypeCounter* get_or_create_type_counter(
    const std::string& name) {
  std::lock_guard<std::mutex> lk(per_type_mutex());
  auto& idx = per_type_index();
  auto it = idx.find(name);
  if (it != idx.end()) {
    return it->second;
  }
  per_type_storage().push_back(std::make_unique<PrimitiveTypeCounter>());
  auto* c = per_type_storage().back().get();
  c->demangled_name = name;
  idx[name] = c;
  return c;
}

// Dump the top-K live primitive-type counters (sorted descending by
// live_count) to stderr. Called from the count-printer hook.
static void dump_per_type_counts(int64_t cur, size_t top_k = 30) {
  std::vector<std::pair<std::string, int64_t>> snapshot;
  {
    std::lock_guard<std::mutex> lk(per_type_mutex());
    snapshot.reserve(per_type_storage().size());
    for (const auto& c : per_type_storage()) {
      snapshot.emplace_back(
          c->demangled_name,
          c->live_count.load(std::memory_order_relaxed));
    }
  }
  std::sort(
      snapshot.begin(),
      snapshot.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(
      stderr,
      "[mlx ArrayDesc per-type] cur=%lld types=%zu\n",
      static_cast<long long>(cur),
      snapshot.size());
  size_t shown = std::min(top_k, snapshot.size());
  for (size_t i = 0; i < shown; ++i) {
    std::fprintf(
        stderr,
        "  %8lld  %s\n",
        static_cast<long long>(snapshot[i].second),
        snapshot[i].first.c_str());
  }
  std::fflush(stderr);
}

static void log_array_desc_count_if_due(int64_t cur) {
  int64_t interval = array_desc_log_interval();
  if (interval > 0 && (cur % interval) == 0) {
    std::fprintf(
        stderr,
        "[mlx ArrayDesc] live=%lld\n",
        static_cast<long long>(cur));
    std::fflush(stderr);
  }
  int64_t per_type_interval = per_type_dump_interval();
  if (per_type_interval > 0 && (cur % per_type_interval) == 0) {
    dump_per_type_counts(cur);
  }
}

array::ArrayDesc::ArrayDesc(Shape shape, Dtype dtype)
    : shape(std::move(shape)), dtype(dtype), status(Status::available) {
  init();
  log_array_desc_count_if_due(++live_array_desc_count());
}

array::ArrayDesc::ArrayDesc(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : shape(std::move(shape)),
      dtype(dtype),
      primitive(std::move(primitive)),
      status(Status::unscheduled),
      inputs(std::move(inputs)) {
  init();
  // Per-primitive-type live counter. Only populated when
  // MLX_PER_TYPE_TRACK or MLX_PER_TYPE_DUMP_INTERVAL is set, to keep the
  // production hot path free of the mutex-protected lookup. The counter
  // pointer, once assigned, is stable for process lifetime.
  if (per_type_tracking_enabled() && this->primitive != nullptr) {
    auto name = demangle_type_name(typeid(*this->primitive).name());
    primitive_type_counter_ = get_or_create_type_counter(name);
    static_cast<PrimitiveTypeCounter*>(primitive_type_counter_)
        ->live_count.fetch_add(1, std::memory_order_relaxed);
  }
  log_array_desc_count_if_due(++live_array_desc_count());
}

// Live ArrayDesc count diagnostic (fork-only). Returns a reference to a
// thread-safe atomic counter incremented on each ArrayDesc construction
// and decremented on destruction.
std::atomic<int64_t>& array::ArrayDesc::live_array_desc_count() {
  static std::atomic<int64_t> count{0};
  return count;
}

// Public read-only accessor for the live ArrayDesc counter. Wraps the
// private inner-class static so external callers (Python bindings, tests)
// can sample the value without touching the private type.
int64_t live_array_desc_count() {
  return array::ArrayDesc::live_array_desc_count().load(
      std::memory_order_relaxed);
}

// Public snapshot accessor for the per-primitive-type live counters.
// Returns an empty vector when MLX_PER_TYPE_DUMP_INTERVAL was not set
// at runtime startup (storage stays empty in that case).
std::vector<std::pair<std::string, int64_t>>
live_array_desc_count_by_type() {
  std::vector<std::pair<std::string, int64_t>> out;
  std::lock_guard<std::mutex> lk(per_type_mutex());
  out.reserve(per_type_storage().size());
  for (const auto& c : per_type_storage()) {
    out.emplace_back(
        c->demangled_name,
        c->live_count.load(std::memory_order_relaxed));
  }
  return out;
}

array::ArrayDesc::~ArrayDesc() {
  --live_array_desc_count();
  if (primitive_type_counter_ != nullptr) {
    static_cast<PrimitiveTypeCounter*>(primitive_type_counter_)
        ->live_count.fetch_sub(1, std::memory_order_relaxed);
  }
  // When an array description is destroyed it will delete a bunch of arrays
  // that may also destroy their corresponding descriptions and so on and so
  // forth.
  //
  // This calls recursively the destructor and can result in stack overflow, we
  // instead put them in a vector and destroy them one at a time resulting in a
  // max stack depth of 2.
  if (inputs.empty()) {
    return;
  }

  std::vector<std::shared_ptr<ArrayDesc>> for_deletion;

  auto append_deletable_inputs = [&for_deletion](ArrayDesc& ad) {
    std::unordered_map<std::uintptr_t, array> input_map;
    for (array& a : ad.inputs) {
      if (a.array_desc_) {
        input_map.insert({a.id(), a});
        for (auto& s : a.siblings()) {
          input_map.insert({s.id(), s});
        }
      }
    }
    ad.inputs.clear();
    for (auto& [_, a] : input_map) {
      bool is_deletable =
          (a.array_desc_.use_count() <= a.siblings().size() + 1);
      // An array with siblings is deletable only if all of its siblings
      // are deletable
      for (auto& s : a.siblings()) {
        if (!is_deletable) {
          break;
        }
        int is_input = (input_map.find(s.id()) != input_map.end());
        is_deletable &=
            s.array_desc_.use_count() <= a.siblings().size() + is_input;
      }
      if (is_deletable) {
        for_deletion.push_back(std::move(a.array_desc_));
      }
    }
  };

  append_deletable_inputs(*this);

  while (!for_deletion.empty()) {
    // top is going to be deleted at the end of the block *after* the arrays
    // with inputs have been moved into the vector
    auto top = std::move(for_deletion.back());
    for_deletion.pop_back();
    append_deletable_inputs(*top);

    // Clear out possible siblings to break circular references
    for (auto& s : top->siblings) {
      // Set to null here to avoid descending into top-level
      // array destructor for siblings
      s.array_desc_ = nullptr;
    }
    top->siblings.clear();
  }
}

array::ArrayIterator::ArrayIterator(const array& arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

array::ArrayIterator::reference array::ArrayIterator::operator*() const {
  auto start = Shape(arr.ndim(), 0);
  auto end = arr.shape();
  auto shape = arr.shape();
  shape.erase(shape.begin());
  start[0] = idx;
  end[0] = idx + 1;
  return reshape(slice(arr, start, end), shape);
};

} // namespace mlx::core
