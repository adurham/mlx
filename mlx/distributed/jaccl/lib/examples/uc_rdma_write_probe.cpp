// UC RDMA_WRITE feasibility probe for the jaccl ACK-coalesce work.
//
// QUESTION THIS ANSWERS:
//   Can an IBV_QPT_UC queue pair (the only QP type jaccl uses) complete a
//   one-sided IBV_WR_RDMA_WRITE over the macOS Thunderbolt librdma path,
//   landing bytes directly in peer memory WITHOUT the peer consuming a recv
//   WR? If yes, the ACK barrier can ride the existing data QP as a one-sided
//   write (0 extra HopIDs) instead of a dedicated ACK QP (2 extra HopIDs),
//   which is what overflows the Thunderbolt HopID budget at dual-model.
//
// WHY A STANDALONE PROBE:
//   The whole jaccl codebase only ever uses IBV_WR_SEND. There is zero
//   in-tree evidence that Apple's closed librdma.dylib implements RDMA_WRITE
//   on UC. A ~30-line probe is far cheaper than rewriting mesh.cpp + a full
//   cluster restart only to discover the opcode is unsupported.
//
// WHAT IT DOES (2 ranks):
//   1. SideChannel TCP bootstrap (rank 0 listens, rank 1 connects).
//   2. One UC QP per rank to the peer over the given RDMA device.
//   3. Register a local recv-target buffer with REMOTE_WRITE access; the
//      peer will RDMA_WRITE directly into it. NO recv WR is ever posted on
//      it — that is the whole point.
//   4. Exchange QP Destination + (buffer addr, rkey) over the side channel.
//   5. RTR/RTS, barrier.
//   6. Each rank RDMA_WRITEs a known sentinel payload into the peer's target
//      buffer, polls its OWN send CQ for the write completion, then spins on
//      its local target buffer waiting for the peer's sentinel to appear.
//   7. Reports PASS only if BOTH the send completion fired AND the peer's
//      bytes landed (verified by content) with no recv WR consumed.
//
// USAGE (run on each studio node; rank 0 first):
//   rank 0 (m4-1): ./jaccl_uc_rdma_write_probe 0 0.0.0.0:48999 rdma_en3
//   rank 1 (m4-2): ./jaccl_uc_rdma_write_probe 1 192.168.86.201:48999 rdma_en3
//
// EXIT CODE: 0 = PASS (RDMA_WRITE on UC works), non-zero = FAIL/unsupported.

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "jaccl/rdma.h"

using namespace jaccl;

namespace {

// Buffer the peer RDMA_WRITEs into. Page-sized to satisfy macOS librdma's
// minimum SGE constraints (same FRAME_SIZE rationale as the ack buffers).
constexpr size_t PROBE_BUF = 4096;
// Sentinel marker the writer stamps so the reader can confirm the bytes are
// the peer's write and not leftover/zeroed memory.
constexpr uint32_t SENTINEL_MAGIC = 0xAC4B0001u; // "ACK" + version

struct RemoteMem {
  uint64_t addr;
  uint32_t rkey;
  uint32_t pad;
};

// macOS librdma rejects ibv_reg_mr on non-page-aligned buffers (jaccl's own
// SharedBuffer uses posix_memalign to the page size for exactly this reason —
// see rdma.cpp page_aligned_alloc). A plain std::vector is NOT page-aligned.
uint8_t* page_aligned(size_t n) {
  static size_t page = sysconf(_SC_PAGESIZE);
  void* p = nullptr;
  if (posix_memalign(&p, page, n) != 0 || p == nullptr) {
    return nullptr;
  }
  std::memset(p, 0, n);
  return static_cast<uint8_t*>(p);
}

// One-sided write: opcode IBV_WR_RDMA_WRITE into peer's (addr, rkey).
// Returns after posting; caller polls the send CQ for the completion.
void post_rdma_write(
    ibv_qp* qp,
    const ibv_sge& sge,
    uint64_t remote_addr,
    uint32_t remote_rkey,
    uint64_t wr_id) {
  JACCL_DMA_BARRIER(); // ensure our CPU writes are visible to the NIC

  ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = const_cast<ibv_sge*>(&sge);
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.next = nullptr;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;

  ibv_send_wr* bad = nullptr;
  int status = ibv_post_send(qp, &wr, &bad);
  if (status != 0) {
    std::ostringstream msg;
    msg << "ibv_post_send(RDMA_WRITE) failed status=" << status
        << " (errno=" << errno << " " << std::strerror(errno) << ")";
    throw std::runtime_error(msg.str());
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <rank> <coordinator addr:port> <rdma_device> [size=2]\n";
    return 2;
  }
  int rank = std::atoi(argv[1]);
  const char* coordinator = argv[2];
  std::string device = argv[3];
  int size = (argc >= 5) ? std::atoi(argv[4]) : 2;
  int peer = 1 - rank; // 2-rank probe

  std::cout << std::unitbuf; // flush every insertion — visibility when redirected
  std::cerr << std::unitbuf;

  std::cout << "[probe] rank=" << rank << " size=" << size
            << " device=" << device << " coordinator=" << coordinator << "\n";

  if (!ibv().is_available()) {
    std::cerr << "[probe] FAIL: librdma.dylib not available\n";
    return 3;
  }

  // device_names[i] = device used to reach rank i; empty for self.
  std::vector<std::string> device_names(size);
  for (int i = 0; i < size; i++) {
    device_names[i] = (i == rank) ? std::string() : device;
  }

  // TCP side channel for out-of-band exchange + barriers.
  SideChannel sc(rank, size, coordinator);
  std::cout << "[probe] side channel up\n";

  // One Connection (UC QP) to the peer.
  std::cout << "[probe] opening device...\n";
  auto connections = create_connections(device_names);
  Connection& conn = connections[peer];
  if (conn.ctx == nullptr) {
    std::cerr << "[probe] FAIL: could not open device " << device << "\n";
    return 4;
  }
  std::cout << "[probe] device open; alloc PD...\n";
  conn.allocate_protection_domain();
  std::cout << "[probe] PD ok; create CQ...\n";
  conn.create_completion_queue(16);
  std::cout << "[probe] CQ ok; create QP...\n";
  conn.create_queue_pair();
  std::cout << "[probe] QP created\n";

  // Allocate + register the target buffer the PEER will RDMA_WRITE into.
  // REMOTE_WRITE is the access flag that authorizes one-sided writes.
  // NOTE: no recv WR is ever posted on this buffer.
  uint8_t* target = page_aligned(PROBE_BUF);
  uint8_t* source = page_aligned(PROBE_BUF);
  if (target == nullptr || source == nullptr) {
    std::cerr << "[probe] FAIL: page_aligned alloc failed\n";
    return 5;
  }
  // Diagnostic sweep: register the target MR under several access-flag
  // combinations and report the rkey each yields. macOS librdma returned
  // rkey=0 for the full LOCAL|REMOTE_WRITE|REMOTE_READ combo; this isolates
  // whether ANY flag combination produces a usable (nonzero) remote key.
  {
    struct FlagCase {
      const char* name;
      int flags;
    };
    FlagCase cases[] = {
        {"REMOTE_WRITE only", IBV_ACCESS_REMOTE_WRITE},
        {"LOCAL|REMOTE_WRITE",
         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE},
        {"REMOTE_READ only", IBV_ACCESS_REMOTE_READ},
        {"LOCAL|REMOTE_WRITE|REMOTE_READ",
         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
             IBV_ACCESS_REMOTE_READ},
    };
    for (auto& c : cases) {
      uint8_t* b = page_aligned(PROBE_BUF);
      ibv_mr* m = ibv().reg_mr(conn.protection_domain, b, PROBE_BUF, c.flags);
      std::cout << "[probe] flagsweep '" << c.name << "': "
                << (m ? "ok" : "NULL");
      if (m) {
        std::cout << " lkey=0x" << std::hex << m->lkey << " rkey=0x"
                  << m->rkey << std::dec;
        ibv().dereg_mr(m);
      }
      std::cout << "\n";
      std::free(b);
    }
  }

  ibv_mr* target_mr = ibv().reg_mr(
      conn.protection_domain,
      target,
      PROBE_BUF,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ);
  ibv_mr* source_mr = ibv().reg_mr(
      conn.protection_domain,
      source,
      PROBE_BUF,
      IBV_ACCESS_LOCAL_WRITE);
  if (target_mr == nullptr || source_mr == nullptr) {
    std::cerr << "[probe] FAIL: reg_mr returned null (target_mr="
              << (void*)target_mr << " source_mr=" << (void*)source_mr
              << ") errno=" << errno << " " << std::strerror(errno) << "\n";
    return 5;
  }
  std::cout << "[probe] MRs registered (REMOTE_WRITE accepted): "
            << "target_mr lkey=0x" << std::hex << target_mr->lkey
            << " rkey=0x" << target_mr->rkey << " source_mr lkey=0x"
            << source_mr->lkey << " rkey=0x" << source_mr->rkey << std::dec
            << "\n";

  conn.queue_pair_init();
  std::cout << "[probe] QP INIT; exchanging destinations...\n";

  // Exchange QP destinations. Layout mirrors MeshGroup::initialize: slot i
  // holds the resource THIS rank created to talk to rank i. Our single QP
  // targets `peer`, so it goes in slot [peer]. After all_gather, the peer's
  // QP-toward-us is at all_dest[peer][rank].
  std::vector<Destination> my_dest(size);
  my_dest[peer] = conn.info();
  auto all_dest = sc.all_gather(my_dest);
  Destination peer_dest = all_dest[peer][rank];
  std::cout << "[probe] destinations exchanged\n";

  // Exchange remote buffer (addr, rkey) for the one-sided write target.
  // Same slot convention: the buffer the peer should write into lives in
  // slot [peer] (it's our resource dedicated to the peer connection).
  std::vector<RemoteMem> my_mem(size);
  my_mem[peer] = RemoteMem{
      reinterpret_cast<uint64_t>(target), target_mr->rkey, 0};
  auto all_mem = sc.all_gather(my_mem);
  RemoteMem peer_mem = all_mem[peer][rank];

  std::cout << "[probe] exchanged: peer_qpn="
            << peer_dest.queue_pair_number << " peer_addr=0x" << std::hex
            << peer_mem.addr << " peer_rkey=0x" << peer_mem.rkey << std::dec
            << "\n";

  // RTR/RTS.
  std::cout << "[probe] modify QP -> RTR...\n";
  conn.queue_pair_rtr(peer_dest);
  std::cout << "[probe] RTR ok; modify QP -> RTS...\n";
  conn.queue_pair_rts();
  std::cout << "[probe] RTS ok; barrier...\n";
  sc.barrier();
  std::cout << "[probe] QP RTS, barrier passed\n";

  // Stamp our source buffer with a rank-identifying sentinel.
  std::memset(source, 0, PROBE_BUF);
  uint32_t magic = SENTINEL_MAGIC;
  uint32_t tag = static_cast<uint32_t>(rank + 1) * 0x1111u; // 0x1111 / 0x2222
  std::memcpy(source, &magic, sizeof(magic));
  std::memcpy(source + 4, &tag, sizeof(tag));
  JACCL_DMA_BARRIER();

  ibv_sge sge;
  sge.addr = reinterpret_cast<uintptr_t>(source);
  sge.length = PROBE_BUF;
  sge.lkey = source_mr->lkey;

  // Post the one-sided RDMA_WRITE to the peer's target buffer.
  post_rdma_write(
      conn.queue_pair, sge, peer_mem.addr, peer_mem.rkey, /*wr_id=*/0xAC0001u);

  // Poll OUR send CQ for the write completion.
  bool send_done = false;
  auto t0 = std::chrono::steady_clock::now();
  while (!send_done) {
    ibv_wc wc;
    int n = ibv_poll_cq(conn.completion_queue, 1, &wc);
    if (n > 0) {
      if (wc.status != IBV_WC_SUCCESS) {
        std::cerr << "[probe] FAIL: RDMA_WRITE completion status=" << wc.status
                  << " (opcode unsupported on UC?)\n";
        return 6;
      }
      std::cout << "[probe] RDMA_WRITE send completion OK (opcode="
                << wc.opcode << ")\n";
      send_done = true;
    }
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
      std::cerr << "[probe] FAIL: timed out waiting for send completion\n";
      return 7;
    }
  }

  // Spin on our LOCAL target buffer waiting for the PEER's sentinel to land.
  // This is the crux: peer wrote into us via RDMA_WRITE with NO recv WR.
  uint32_t expect_tag = static_cast<uint32_t>(peer + 1) * 0x1111u;
  bool recv_landed = false;
  auto t1 = std::chrono::steady_clock::now();
  while (!recv_landed) {
    JACCL_DMA_BARRIER(); // make NIC's DMA writes visible to the CPU
    uint32_t got_magic, got_tag;
    std::memcpy(&got_magic, target, sizeof(got_magic));
    std::memcpy(&got_tag, target + 4, sizeof(got_tag));
    if (got_magic == SENTINEL_MAGIC && got_tag == expect_tag) {
      std::cout << "[probe] peer sentinel landed via one-sided write: magic=0x"
                << std::hex << got_magic << " tag=0x" << got_tag << std::dec
                << " (NO recv WR consumed)\n";
      recv_landed = true;
    }
    if (std::chrono::steady_clock::now() - t1 > std::chrono::seconds(10)) {
      std::cerr << "[probe] FAIL: peer's RDMA_WRITE never landed in local "
                   "memory (got magic=0x"
                << std::hex << got_magic << " tag=0x" << got_tag << std::dec
                << ")\n";
      return 8;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  sc.barrier(); // don't tear down MRs while peer is still reading
  std::cout << "[probe] PASS: UC IBV_WR_RDMA_WRITE works over this path "
               "(one-sided ack is viable)\n";
  return 0;
}
