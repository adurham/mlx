// HopID / QP budget probe for the jaccl dual-model work.
//
// QUESTION THIS ANSWERS:
//   How many UC queue pairs can a single node bring to RTS over one
//   Thunderbolt RDMA device before the macOS NHI driver runs out of HopIDs
//   (hardware rings)? The whole dual-model ACK-coalesce problem rests on the
//   claim "data QP (2 HopIDs) + ACK QP (2 HopIDs) per model => 8 for dual
//   model => overflows a ~6-7 budget". That number was inferred from a past
//   kernel log on a different machine, never measured on THIS build/hardware.
//   This probe measures it directly.
//
// METHOD (2 ranks, mirror of how MeshGroup brings up QPs):
//   Loop creating UC QPs to the peer. For each QP: PD/CQ/QP alloc -> INIT ->
//   exchange dest over the side channel -> RTR -> RTS. RTS is the transition
//   where the NHI driver backs each direction with a hardware ring (HopID),
//   so a HopID exhaustion shows up as an RTS failure (errno) or a wedge.
//   Report the count of QPs that reached RTS successfully. Both ranks run in
//   lockstep (a side-channel barrier per QP) so neither races ahead.
//
// SAFETY:
//   - Sets IBV_FORK_SAFE via env in the launcher (REQUIRED; without it QP
//     bring-up wedges in uninterruptible kernel state -> needs reboot).
//   - Stops at MAX_QPS regardless, so it can't spin forever.
//   - Each RTS is wrapped: on failure we report the count and exit cleanly
//     rather than letting the jaccl helper throw into a wedge.
//
// USAGE (rank 0 on m4-1 first, then rank 1 on m4-2):
//   rank 0: ./jaccl_hopid_budget_probe 0 0.0.0.0:48555 rdma_en3 16
//   rank 1: ./jaccl_hopid_budget_probe 1 192.168.86.201:48555 rdma_en3 16
//   (last arg = max QPs to attempt)

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "jaccl/rdma.h"

using namespace jaccl;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <rank> <coordinator addr:port> <rdma_device> [max_qps=16]\n";
    return 2;
  }
  int rank = std::atoi(argv[1]);
  const char* coordinator = argv[2];
  std::string device = argv[3];
  int max_qps = (argc >= 5) ? std::atoi(argv[4]) : 16;
  int size = 2;
  int peer = 1 - rank;

  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  std::cout << "[hopid] rank=" << rank << " device=" << device
            << " max_qps=" << max_qps << " coordinator=" << coordinator << "\n";

  if (!ibv().is_available()) {
    std::cerr << "[hopid] FAIL: librdma.dylib not available\n";
    return 3;
  }

  SideChannel sc(rank, size, coordinator);
  std::cout << "[hopid] side channel up\n";

  // device_names[i] = device to reach rank i; empty for self.
  std::vector<std::string> device_names(size);
  for (int i = 0; i < size; i++) {
    device_names[i] = (i == rank) ? std::string() : device;
  }

  // Keep all Connections alive for the duration (HopIDs are held until the
  // QP is destroyed). Use unique_ptr so addresses are stable in the vector.
  std::vector<std::unique_ptr<Connection>> conns;
  int reached_rts = 0;

  for (int q = 0; q < max_qps; q++) {
    // Each QP needs its own ibv_context (mirrors how separate model
    // processes each open the device). create_connections opens a fresh
    // context per non-empty device entry.
    auto fresh = create_connections(device_names);
    auto conn = std::make_unique<Connection>(std::move(fresh[peer]));
    if (conn->ctx == nullptr) {
      std::cerr << "[hopid] qp#" << q << " FAIL: could not open device\n";
      break;
    }

    bool setup_ok = true;
    try {
      conn->allocate_protection_domain();
      conn->create_completion_queue(16);
      conn->create_queue_pair();
      conn->queue_pair_init();
    } catch (const std::exception& e) {
      std::cerr << "[hopid] qp#" << q << " setup FAILED: " << e.what() << "\n";
      setup_ok = false;
    }

    // Exchange destinations for THIS qp. Layout: slot[peer] holds our QP
    // toward the peer; read back all_dest[peer][rank].
    std::vector<Destination> my_dest(size);
    if (setup_ok) {
      my_dest[peer] = conn->info();
    }
    auto all_dest = sc.all_gather(my_dest);
    Destination peer_dest = all_dest[peer][rank];

    if (!setup_ok) {
      break;
    }

    // RTR then RTS — RTS is where the HopID/hardware ring is allocated.
    bool rts_ok = true;
    std::string err;
    try {
      conn->queue_pair_rtr(peer_dest);
    } catch (const std::exception& e) {
      rts_ok = false;
      err = std::string("RTR: ") + e.what();
    }
    if (rts_ok) {
      try {
        conn->queue_pair_rts();
      } catch (const std::exception& e) {
        rts_ok = false;
        err = std::string("RTS: ") + e.what();
      }
    }

    // Tell the peer whether we succeeded, so both sides agree on the count
    // and tear down together (a half-open QP set can wedge the link).
    auto results = sc.all_gather<int>(rts_ok ? 1 : 0);
    bool both_ok = results[0] == 1 && results[1] == 1;

    if (rts_ok && both_ok) {
      conns.push_back(std::move(conn));
      reached_rts++;
      std::cout << "[hopid] qp#" << q << " RTS ok (total live QPs="
                << reached_rts << ", ~" << (reached_rts * 2)
                << " HopIDs assuming 2/QP)\n";
    } else {
      std::cerr << "[hopid] qp#" << q
                << " STOPPED: this_rank_rts=" << (rts_ok ? "ok" : "fail")
                << " peer_rts=" << (results[peer] == 1 ? "ok" : "fail");
      if (!err.empty()) {
        std::cerr << " err=[" << err << "]";
      }
      std::cerr << "\n";
      break;
    }
  }

  std::cout << "[hopid] RESULT rank=" << rank
            << ": brought " << reached_rts
            << " UC QPs to RTS over " << device << " before failure/limit ("
            << "max_qps=" << max_qps << ")\n";
  std::cout << "[hopid] => budget interpretation: data+ack per model = 2 QPs; "
            << (reached_rts / 2) << " model(s) fit; dual-model needs 4 QPs.\n";

  // Barrier before teardown so neither rank destroys QPs while the other is
  // still polling/using the link.
  sc.barrier();
  return 0;
}
