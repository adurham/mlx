// Loopback test for the QP-less TCP-only CoordGroup.
//
// Forks two ranks that connect over 127.0.0.1 and exercises every collective
// the coord group actually supports, plus the desync tripwire. No RDMA
// device, no ibverbs, no queue pairs -- which is the entire point.
//
// Build/run:
//   c++ -std=c++20 -I<jaccl lib dir> coord_group_loopback_test.cpp \
//       <jaccl lib dir>/jaccl/tcp.cpp -o /tmp/cg_test && /tmp/cg_test

#include <sys/wait.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "jaccl/coord_group.h"

using namespace jaccl;

static int failures = 0;

#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::fprintf(stderr, "FAIL[rank %d]: %s\n", rank, (msg));      \
      failures++;                                                    \
    }                                                                \
  } while (0)

static int run_rank(int rank, const std::string& addr) {
  CoordGroup g(rank, 2, addr);

  CHECK(g.rank() == rank, "rank()");
  CHECK(g.size() == 2, "size()");

  // --- all_sum on float32 (mx_any's shape: a 1-element vote) ---
  {
    float in = (rank == 0) ? 1.0f : 0.0f;
    float out = -1.0f;
    g.all_sum(&in, &out, sizeof(float), Dtype::Float32);
    CHECK(std::fabs(out - 1.0f) < 1e-6, "all_sum float32 == 1.0");
  }

  // --- all_sum on int32 vector (MTP presence array shape) ---
  {
    int32_t in[4] = {rank, rank * 2, rank * 3, 7};
    int32_t out[4] = {0, 0, 0, 0};
    g.all_sum(in, out, sizeof(in), Dtype::Int32);
    // rank0 {0,0,0,7} + rank1 {1,2,3,7} = {1,2,3,14}
    CHECK(out[0] == 1 && out[1] == 2 && out[2] == 3 && out[3] == 14,
          "all_sum int32 vector");
  }

  // --- all_max / all_min (mx_min_int's shape) ---
  {
    int32_t in = (rank == 0) ? 5 : 9;
    int32_t mx_out = 0, mn_out = 0;
    g.all_max(&in, &mx_out, sizeof(int32_t), Dtype::Int32);
    g.all_min(&in, &mn_out, sizeof(int32_t), Dtype::Int32);
    CHECK(mx_out == 9, "all_max int32");
    CHECK(mn_out == 5, "all_min int32");
  }

  // --- all_gather (warmup control-plane sync + task-id gather shape) ---
  {
    int32_t in[2] = {100 + rank, 200 + rank};
    int32_t out[4] = {0, 0, 0, 0};
    g.all_gather(in, out, sizeof(in));
    CHECK(out[0] == 100 && out[1] == 200 && out[2] == 101 && out[3] == 201,
          "all_gather concatenation in rank order");
  }

  // --- barrier ---
  g.barrier();

  // --- in-place aliasing safety: input == output buffer ---
  {
    int32_t buf = rank + 1; // rank0: 1, rank1: 2
    g.all_sum(&buf, &buf, sizeof(int32_t), Dtype::Int32);
    CHECK(buf == 3, "all_sum with input aliasing output");
  }

  // --- send/recv must refuse loudly ---
  {
    bool threw = false;
    char b = 0;
    try {
      g.send(&b, 1, 1 - rank);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "send() throws on a coord group");
  }

  // --- oversized payload must refuse loudly ---
  {
    bool threw = false;
    std::vector<char> big(CoordGroup::MAX_BYTES + 1, 0);
    std::vector<char> bigout(big.size());
    try {
      g.all_gather(big.data(), bigout.data(), big.size());
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "oversized payload throws");
    // NOTE: the throw happens BEFORE any wire I/O (the size check is the
    // first thing exchange() does), so both ranks throw symmetrically and
    // the stream stays in sync -- verified by the ops below still working.
  }

  // --- stream still healthy after the two refusals ---
  {
    int32_t in = 1, out = 0;
    g.all_sum(&in, &out, sizeof(int32_t), Dtype::Int32);
    CHECK(out == 2, "collectives still work after refused ops");
  }

  // --- DESYNC tripwire: ranks execute DIFFERENT ops at the same step ---
  {
    bool threw = false;
    int32_t in = 1, out = 0;
    try {
      if (rank == 0) {
        g.all_sum(&in, &out, sizeof(int32_t), Dtype::Int32);
      } else {
        g.all_max(&in, &out, sizeof(int32_t), Dtype::Int32);
      }
    } catch (const std::exception& e) {
      threw = true;
      if (std::string(e.what()).find("DESYNC") == std::string::npos) {
        std::fprintf(stderr, "FAIL[rank %d]: wrong exception: %s\n", rank, e.what());
        failures++;
      }
    }
    CHECK(threw, "mismatched ops across ranks throw DESYNC");
  }

  if (failures == 0) {
    std::fprintf(stderr, "rank %d: ALL CHECKS PASSED\n", rank);
  }
  return failures == 0 ? 0 : 1;
}

int main() {
  // Pick a free port the same way MeshGroup::split_tcp_coord does.
  int port = reserve_ephemeral_port("127.0.0.1");
  std::string addr = "127.0.0.1:" + std::to_string(port);

  pid_t child = fork();
  if (child == 0) {
    // Give rank 0 a moment to listen; TCPSocket::connect also retries.
    usleep(200000);
    _exit(run_rank(1, addr));
  }
  int rc0 = run_rank(0, addr);
  int status = 0;
  waitpid(child, &status, 0);
  int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  if (rc0 == 0 && rc1 == 0) {
    std::fprintf(stderr, "\nCoordGroup loopback test: PASS\n");
    return 0;
  }
  std::fprintf(stderr, "\nCoordGroup loopback test: FAIL (rank0=%d rank1=%d)\n", rc0, rc1);
  return 1;
}
