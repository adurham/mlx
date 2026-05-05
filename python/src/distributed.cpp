// Copyright  © 2024 Apple Inc.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>

#include <unistd.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Diagnostic Python-stack-capture for JACCL collectives (Phase A of
// next_session_plan_jaccl_c2_v3.md). Active only when
// JACCL_TRACE_CALLS=1. Each call to mx.distributed.{all_sum,all_max,
// all_min,all_gather,send,recv,recv_like,sum_scatter} captures the
// current Python stack (top frames) and writes it to a per-rank,
// per-pid log alongside a monotonic submit_id. The submit_id ordering
// in the Python log mirrors the call_id ordering in the existing
// /tmp/jaccl_trace_rank_${rank}_pid${pid}.log emitted from
// MeshGroup::trace_call (because the runner main loop is
// single-threaded), so the Nth pystack entry corresponds to the Nth
// trace_call entry. Diff rank0 vs rank1 pystack at the divergent line
// to find the offending call site.

std::mutex g_pystack_mutex;
FILE* g_pystack_file = nullptr;
bool g_pystack_open_attempted = false;
bool g_pystack_enabled = false;
std::atomic<uint64_t> g_pystack_submit_id{0};

bool tracing_enabled() {
  const char* env = std::getenv("JACCL_TRACE_CALLS");
  return env != nullptr && std::string_view(env) == "1";
}

// Must be called with GIL held.
std::string capture_python_stack(int max_frames) {
  PyObject* tb_module = PyImport_ImportModule("traceback");
  if (tb_module == nullptr) {
    PyErr_Clear();
    return "<traceback-import-failed>\n";
  }
  PyObject* fmt_stack = PyObject_GetAttrString(tb_module, "format_stack");
  Py_DECREF(tb_module);
  if (fmt_stack == nullptr) {
    PyErr_Clear();
    return "<format_stack-getattr-failed>\n";
  }
  PyObject* lines = PyObject_CallObject(fmt_stack, nullptr);
  Py_DECREF(fmt_stack);
  if (lines == nullptr || !PyList_Check(lines)) {
    Py_XDECREF(lines);
    PyErr_Clear();
    return "<format_stack-call-failed>\n";
  }
  Py_ssize_t n = PyList_Size(lines);
  // Drop the top-most frame (this very function) and any nanobind shim
  // frames by trimming the tail; the user-relevant frames are above.
  Py_ssize_t end = n; // entries are oldest-first
  Py_ssize_t start = end - max_frames > 0 ? end - max_frames : 0;
  std::string result;
  for (Py_ssize_t i = start; i < end; ++i) {
    PyObject* item = PyList_GetItem(lines, i); // borrowed
    if (item == nullptr) {
      continue;
    }
    Py_ssize_t len = 0;
    const char* s = PyUnicode_AsUTF8AndSize(item, &len);
    if (s != nullptr && len > 0) {
      result.append(s, static_cast<size_t>(len));
    }
  }
  Py_DECREF(lines);
  if (result.empty()) {
    result = "<empty-stack>\n";
  }
  return result;
}

void open_pystack_log(int rank) {
  // g_pystack_mutex must be held by caller.
  if (g_pystack_open_attempted) {
    return;
  }
  g_pystack_open_attempted = true;
  if (!tracing_enabled()) {
    return;
  }
  char path[160];
  std::snprintf(
      path,
      sizeof(path),
      "/tmp/jaccl_pystack_rank_%d_pid%d.log",
      rank,
      static_cast<int>(::getpid()));
  g_pystack_file = std::fopen(path, "w");
  if (g_pystack_file == nullptr) {
    return;
  }
  std::fprintf(
      g_pystack_file,
      "# submit_id\top\telem_size\tmsg_bytes\tpython_stack\n");
  std::fflush(g_pystack_file);
  g_pystack_enabled = true;
}

// GIL must be held.
void log_pystack(
    const char* op,
    int rank,
    int elem_size,
    int64_t msg_bytes) {
  if (!tracing_enabled()) {
    return;
  }
  std::string stack = capture_python_stack(10);
  uint64_t submit_id =
      g_pystack_submit_id.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_pystack_mutex);
  open_pystack_log(rank);
  if (!g_pystack_enabled || g_pystack_file == nullptr) {
    return;
  }
  std::fprintf(
      g_pystack_file,
      "=== submit_id=%llu op=%s elem=%d bytes=%lld ===\n%s",
      static_cast<unsigned long long>(submit_id),
      op,
      elem_size,
      static_cast<long long>(msg_bytes),
      stack.c_str());
  std::fflush(g_pystack_file);
}

mx::distributed::Group resolve_group(
    const std::optional<mx::distributed::Group>& group) {
  if (group.has_value()) {
    return group.value();
  }
  return mx::distributed::init();
}

// Returns elem_size in bytes (0 for unknown dtype).
int dtype_size(const mx::array& a) {
  return static_cast<int>(a.itemsize());
}

} // namespace

void init_distributed(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "distributed", "mlx.core.distributed: Communication operations");

  nb::class_<mx::distributed::Group>(
      m,
      "Group",
      R"pbcopy(
        An :class:`mlx.core.distributed.Group` represents a group of independent mlx
        processes that can communicate.
      )pbcopy")
      .def(
          "rank", &mx::distributed::Group::rank, "Get the rank of this process")
      .def("size", &mx::distributed::Group::size, "Get the size of the group")
      .def(
          "split",
          &mx::distributed::Group::split,
          "color"_a,
          "key"_a = -1,
          nb::sig("def split(self, color: int, key: int = -1) -> Group"),
          R"pbdoc(
            Split the group to subgroups based on the provided color.

            Processes that use the same color go to the same group. The ``key``
            argument defines the rank in the new group. The smaller the key the
            smaller the rank. If the key is negative then the rank in the
            current group is used.

            Args:
              color (int): A value to group processes into subgroups.
              key (int, optional): A key to optionally change the rank ordering
                of the processes.
          )pbdoc");

  m.def(
      "is_available",
      [](const std::string& backend) {
        return mx::distributed::is_available(backend);
      },
      "backend"_a = "any",
      nb::sig("def is_available(backend: str = 'any') -> bool"),
      R"pbdoc(
      Check if a communication backend is available.

      Note, this function returns whether MLX has the capability of
      instantiating that distributed backend not whether it is possible to
      create a communication group. For that purpose one should use
      ``init(strict=True)``.

      Args:
        backend (str, optional): The name of the backend to check for availability.
          It takes the same values as :func:`init()`. Default: ``"any"``.

      Returns:
        bool: Whether the distributed backend is available.
      )pbdoc");

  m.def(
      "init",
      &mx::distributed::init,
      "strict"_a = false,
      "backend"_a = "any",
      nb::sig("def init(strict: bool = False, backend: str = 'any') -> Group"),
      R"pbdoc(
        Initialize the communication backend and create the global communication group.

        Example:

          .. code:: python

            import mlx.core as mx

            group = mx.distributed.init(backend="ring")

        Args:
          strict (bool, optional): If set to False it returns a singleton group
            in case ``mx.distributed.is_available()`` returns False otherwise
            it throws a runtime error. Default: ``False``
          backend (str, optional): Which distributed backend to initialize.
            Possible values ``mpi``, ``ring``, ``nccl``, ``jaccl``, ``any``. If
            set to ``any`` all available backends are tried and the first one
            that succeeds becomes the global group which will be returned in
            subsequent calls. Default: ``any``

        Returns:
          Group: The group representing all the launched processes.
      )pbdoc");

  m.def(
      "all_sum",
      [](const ScalarOrArray& x,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          log_pystack("all_sum", eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::all_sum(x_arr, eg, s);
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_sum(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        All reduce sum.

        Sum the ``x`` arrays from all processes in the group.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            reduction. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The sum of all ``x`` arrays.
      )pbdoc");
  m.def(
      "all_max",
      [](const ScalarOrArray& x,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          log_pystack("all_max", eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::all_max(x_arr, eg, s);
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_max(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        All reduce max.

        Find the maximum of the ``x`` arrays from all processes in the group.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            reduction. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The maximum of all ``x`` arrays.
      )pbdoc");
  m.def(
      "all_min",
      [](const ScalarOrArray& x,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          log_pystack("all_min", eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::all_min(x_arr, eg, s);
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_min(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      All reduce min.

      Find the minimum of the ``x`` arrays from all processes in the group.

      Args:
        x (array): Input array.
        group (Group): The group of processes that will participate in the
          reduction. If set to ``None`` the global group is used. Default:
          ``None``.
        stream (Stream, optional): Stream or device. Defaults to ``None``
          in which case the default stream of the default device is used.

      Returns:
        array: The minimum of all ``x`` arrays.
    )pbdoc");
  m.def(
      "all_gather",
      [](const ScalarOrArray& x,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          log_pystack(
              "all_gather", eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::all_gather(x_arr, eg, s);
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_gather(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Gather arrays from all processes.

        Gather the ``x`` arrays from all processes in the group and concatenate
        them along the first axis. The arrays should all have the same shape.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            gather. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The concatenation of all ``x`` arrays.
      )pbdoc");

  m.def(
      "send",
      [](const ScalarOrArray& x,
         int dst,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          char op[24];
          std::snprintf(op, sizeof(op), "send_dst%d", dst);
          log_pystack(op, eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::send(x_arr, dst, eg, s);
      },
      "x"_a,
      "dst"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def send(x: array, dst: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Send an array from the current process to the process that has rank
        ``dst`` in the group.

        Args:
          x (array): Input array.
          dst (int): Rank of the destination process in the group.
          group (Group): The group of processes that will participate in the
            send. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: An array identical to ``x`` which when evaluated the send is performed.
      )pbdoc");

  m.def(
      "recv",
      [](mx::Shape shape,
         mx::Dtype dtype,
         int src,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          char op[24];
          std::snprintf(op, sizeof(op), "recv_src%d", src);
          int64_t n_elem = 1;
          for (auto d : shape) {
            n_elem *= d;
          }
          int elem = static_cast<int>(mx::size_of(dtype));
          log_pystack(op, eg.rank(), elem, n_elem * elem);
        }
        return mx::distributed::recv(std::move(shape), dtype, src, eg, s);
      },
      "shape"_a,
      "dtype"_a,
      "src"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def recv(shape: Sequence[int], dtype: Dtype, src: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Recv an array with shape ``shape`` and dtype ``dtype`` from process
        with rank ``src``.

        Args:
          shape (Tuple[int]): The shape of the array we are receiving.
          dtype (Dtype): The data type of the array we are receiving.
          src (int): Rank of the source process in the group.
          group (Group): The group of processes that will participate in the
            recv. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The array that was received from ``src``.
      )pbdoc");

  m.def(
      "recv_like",
      [](const ScalarOrArray& x,
         int src,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          char op[24];
          std::snprintf(op, sizeof(op), "recv_src%d", src);
          log_pystack(op, eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::recv_like(x_arr, src, eg, s);
      },
      "x"_a,
      "src"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def recv_like(x: array, src: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Recv an array with shape and type like ``x`` from process with rank
        ``src``.

        It is equivalent to calling ``mx.distributed.recv(x.shape, x.dtype, src)``.

        Args:
          x (array): An array defining the shape and dtype of the array we are
            receiving.
          src (int): Rank of the source process in the group.
          group (Group): The group of processes that will participate in the
            recv. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The array that was received from ``src``.
      )pbdoc");

  m.def(
      "sum_scatter",
      [](const ScalarOrArray& x,
         std::optional<mx::distributed::Group> group,
         mx::StreamOrDevice s) {
        auto x_arr = to_array(x);
        auto eg = resolve_group(group);
        if (eg.size() > 1) {
          log_pystack(
              "sum_scatter", eg.rank(), dtype_size(x_arr), x_arr.nbytes());
        }
        return mx::distributed::sum_scatter(x_arr, eg, s);
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sum_scatter(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Sum ``x`` across all processes in the group and shard the result along the first axis across ranks.
      ``x.shape[0]`` must be divisible by the group size.

      The result is equivalent to ``all_sum(x)[rank*chunk_size:(rank+1)*chunk_size]``, where ``chunk_size = x.shape[0] // group.size()`` and ``rank`` is the rank of this process in the group.
      Note: ``all_sum`` is mentioned only for illustration; the actual implementation does not perform ``all_sum`` and uses a single reduce-scatter collective instead.
      Currently supported only for the NCCL backend.

      Args:
        x (array): Input array.
        group (Group): The group of processes that will participate in the
          sum scatter. If set to ``None`` the global group is used. Default:
          ``None``.
        stream (Stream, optional): Stream or device. Defaults to ``None``
          in which case the default stream of the default device is used.
      Returns:
        array: The output array with shape ``[x.shape[0] // group.size(), *x.shape[1:]]``.
    )pbdoc");
}
