// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

#include <iostream>

constexpr int MAX_PEERS = 8;

namespace mlx::core::distributed::jaccl {

namespace {

// Calculate total bytes needed for all JACCL buffers.
// Sum over k=0..BUFFER_SIZES-1: FRAME_SIZE * (1 << k) * NUM_BUFFERS * num_peers
size_t calculate_pool_size(int num_peers) {
  size_t total = 0;
  for (int k = 0; k < BUFFER_SIZES; k++) {
    total +=
        static_cast<size_t>(FRAME_SIZE) * (1 << k) * NUM_BUFFERS * num_peers;
  }
  return total;
}

} // namespace

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const char* coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      coordinator_addr_(coordinator_addr),
      device_names_(device_names),
      side_channel_(rank_, size_, coordinator_addr),
      connections_(create_connections(device_names)),
      pool_(calculate_pool_size(device_names.size())) {
  if (size_ > MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);
}

void MeshGroup::initialize() {
  // Create the queue pairs
  for (int i = 0; i < connections_.size(); i++) {
    auto& conn = connections_[i];
    if (conn.ctx == nullptr) {
      std::cerr << "[jaccl] Rank " << rank_ << " skipping peer " << i
                << " (self)" << std::endl;
      continue;
    }
    std::cerr << "[jaccl] Rank " << rank_ << " initializing peer " << i
              << " device=" << device_names_[i] << std::endl;
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  allocate_buffers();

  // First init all connections
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier so
  // that all peers have initialized their connections before attempting to
  // transition to RTS.
  std::vector<Destination> info;
  for (auto& conn : connections_) {
    info.emplace_back(conn.info());
  }
  auto all_infos = side_channel_.all_gather(info);

  // Transition queue pairs to RTS
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_info = all_infos[peer][rank_];
    connections_[peer].queue_pair_rtr(peer_info);
    connections_[peer].queue_pair_rts();
  }
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();

  // Register the pool with all active protection domains
  for (auto& conn : connections_) {
    if (conn.ctx != nullptr) {
      pool_.register_to_protection_domain(conn.protection_domain);
    }
  }

  // Allocate buffers from the pool (single MR, pointer arithmetic)
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        buffers_.push_back(pool_.allocate(FRAME_SIZE * (1 << k)));
      }
    }
  }

  // Pool-backed buffers don't need individual registration —
  // the pool's MR covers all of them. register_to_protection_domain()
  // on a pool-backed SharedBuffer is a no-op.
}

void MeshGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{});
  });
}

void MeshGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
  });
}

void MeshGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{});
  });
}

void MeshGroup::all_gather(const array& input, array& output, Stream stream) {
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  size_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    // Fully connected all gather
    char* data = out_ptr;
    char* our_data = out_ptr + rank_ * n_bytes;
    auto [sz, N] = buffer_size_from_message(n_bytes);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(n_bytes);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int write_offset[MAX_PEERS] = {0};

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);
      std::copy(
          our_data + read_offset,
          our_data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<char>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        // Send completed. If all sends completed then send the next chunk.
        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                our_data + read_offset,
                our_data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<char>());
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        // Recv completed. If we have more chunks then post another recv.
        else if (work_type == RECV_WR) {
          std::copy(
              recv_buffer(sz, buff, rank).begin<char>(),
              recv_buffer(sz, buff, rank).begin<char>() +
                  std::min(N, total - write_offset[rank]),
              data + rank * n_bytes + write_offset[rank]);
          write_offset[rank] += N;
          if (write_offset[rank] + N * (PIPELINE - 1) < total) {
            recv_from(sz, rank, buff);
            in_flight++;
          }
        }
      }
    }
  });
}

void MeshGroup::send(const array& input, int dst, Stream stream) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([data, n_bytes, dst, this]() {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < n_bytes && buff < PIPELINE) {
      std::copy(
          data + read_offset,
          data + std::min(read_offset + N, n_bytes),
          send_buffer(sz, buff).begin<char>());
      send_to(sz, dst, buff);

      buff++;
      read_offset += N;
      in_flight++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed and we have more data to send then go ahead
      // and send them.
      ibv_wc wc[WC_NUM];
      int n = connections_[dst].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (read_offset < n_bytes) {
          std::copy(
              data + read_offset,
              data + std::min(read_offset + N, n_bytes),
              send_buffer(sz, buff).begin<char>());
          send_to(sz, dst, buff);

          read_offset += N;
          in_flight++;
        }
      }
    }
  });
}

void MeshGroup::recv(array& out, int src, Stream stream) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([data, n_bytes, src, this]() {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t write_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(sz, src, buff);

      in_flight++;
      buff++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
      ibv_wc wc[WC_NUM];
      int n = connections_[src].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        std::copy(
            recv_buffer(sz, buff, src).begin<char>(),
            recv_buffer(sz, buff, src).begin<char>() +
                std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
            data + write_offset);
        write_offset += N;

        if (write_offset + (PIPELINE - 1) * N < n_bytes) {
          recv_from(sz, src, buff);

          in_flight++;
        }
      }
    }
  });
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, size = input.size(), this, reduce_op]() {
    // If not inplace all reduce then copy the input to the output first
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }

    // Fully connected all reduce
    T* data = out_ptr;
    auto [sz, buffer_size] = buffer_size_from_message(size * sizeof(T));
    int64_t N = buffer_size / sizeof(T);
    constexpr int PIPELINE = NUM_BUFFERS;
    constexpr int WC_NUM = PIPELINE * MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(size);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int completed_recv_begin[MAX_PEERS] = {0};
    int completed_recv_end[MAX_PEERS] = {0};

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);
      std::copy(
          data + read_offset,
          data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<T>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed mark how many completions we have received
      // for that buffer. If we have sent the buffer to all peers we can
      // reuse the buffer so copy the next chunk of data and send it to all.
      //
      // If a receive is completed then advance the pointer of completed
      // receives.
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                data + read_offset,
                data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<T>());
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        else if (work_type == RECV_WR) {
          completed_recv_end[rank]++;
        }
      }

      // Process the completed recv
      //
      // For each rank we have a range of completed recv defined by a begin
      // and end inclusive and exlusive in standard C++ fashion.
      //
      // When there is an unprocessed receive we first check if we have
      // finished sending the write location. If so then we reduce in-place
      // and then check if there is more to be received and post a recv.
      for (int r = 0; r < size_; r++) {
        int s = completed_recv_begin[r];
        int e = completed_recv_end[r];
        int w = s * N;
        while (w < read_offset && e - s > 0) {
          int buff = s % PIPELINE;
          reduce_op(
              recv_buffer(sz, buff, r).begin<T>(),
              data + w,
              std::min(N, total - w));
          w += N;
          s++;
          if (w + (PIPELINE - 1) * N < total) {
            recv_from(sz, r, buff);
            in_flight++;
          }
        }
        completed_recv_begin[r] = s;
      }
    }
  });
}

} // namespace mlx::core::distributed::jaccl

// split() implementation — must be outside the template-heavy section
namespace mlx::core::distributed::jaccl {

std::shared_ptr<GroupImpl> MeshGroup::split(int color, int key) {
  // Use negative key as sentinel for "use current rank"
  if (key < 0) {
    key = rank_;
  }

  // Step 1: Coordinate across all ranks to determine sub-group membership.
  // Pack (color, key) into a single struct for all_gather.
  struct SplitInfo {
    int color;
    int key;
  };
  SplitInfo my_info{color, key};
  auto all_info = side_channel_.all_gather(my_info);

  // Step 2: Find peers with matching color, sorted by key for rank assignment.
  struct PeerEntry {
    int key;
    int global_rank;
    bool operator<(const PeerEntry& o) const {
      return key < o.key;
    }
  };
  std::vector<PeerEntry> peers;
  for (int i = 0; i < size_; i++) {
    if (all_info[i].color == color) {
      peers.push_back({all_info[i].key, i});
    }
  }
  std::sort(peers.begin(), peers.end());

  int new_size = static_cast<int>(peers.size());

  // Singleton group — just return an EmptyGroup-like wrapper (no comm needed)
  if (new_size <= 1) {
    // Return a single-node MeshGroup. Build a device_names with one entry
    // (self, empty string) and use a dedicated coordinator port.
    // For a size-1 group, communication ops are no-ops at the caller level.
    //
    // Actually, we can construct a MeshGroup with just ourselves.
    // The coordinator negotiation below handles this correctly.
    // Fall through to the general case.
  }

  // Step 3: Determine new local rank and build filtered device names.
  int new_rank = -1;
  // Build a map: new_rank -> global_rank
  std::vector<int> global_ranks(new_size);
  for (int i = 0; i < new_size; i++) {
    global_ranks[i] = peers[i].global_rank;
    if (peers[i].global_rank == rank_) {
      new_rank = i;
    }
  }

  if (new_rank < 0) {
    throw std::runtime_error(
        "[jaccl] split: current rank not found in sub-group");
  }

  // Build new device_names: for each sub-group member, use the original
  // device name that connects us to that global rank.
  std::vector<std::string> new_device_names(new_size);
  for (int i = 0; i < new_size; i++) {
    int g = global_ranks[i];
    if (g == rank_) {
      new_device_names[i] = ""; // Self — no connection needed
    } else {
      new_device_names[i] = device_names_[g];
    }
  }

  // Step 4: Negotiate a new coordinator address for the sub-group.
  // The lowest-ranked process in each sub-group (new_rank == 0) will
  // start a new coordinator.
  //
  // Strategy: offset the coordinator port by (color + 1) * 100 to avoid
  // collisions between sub-groups. Each node broadcasts a "connect"
  // address (its reachable IP + new port). Non-leaders use the leader's
  // address to connect. The leader itself always binds to 0.0.0.0.
  std::string coordinator_addr = coordinator_addr_;

  // Parse the original coordinator address to extract host and port
  auto colon_pos = coordinator_addr_.rfind(':');
  if (colon_pos != std::string::npos) {
    std::string host = coordinator_addr_.substr(0, colon_pos);
    int base_port = std::atoi(coordinator_addr_.substr(colon_pos + 1).c_str());
    int new_port = base_port + (color + 1) * 100;
    coordinator_addr = host + ":" + std::to_string(new_port);
  }

  // Broadcast each node's proposed coordinator address via all_gather.
  auto all_coords = side_channel_.all_gather(coordinator_addr);

  // Non-leaders use the leader's address (which has the leader's
  // reachable IP). The leader overrides to 0.0.0.0 for binding.
  int leader_global_rank = global_ranks[0];
  if (new_rank == 0) {
    // Leader: extract the port from our address and bind to 0.0.0.0
    auto leader_colon = coordinator_addr.rfind(':');
    if (leader_colon != std::string::npos) {
      coordinator_addr = "0.0.0.0" + coordinator_addr.substr(leader_colon);
    }
  } else {
    // Non-leader: use the leader's address to connect
    coordinator_addr = all_coords[leader_global_rank];
    // If the leader's address is 0.0.0.0, we need a reachable IP.
    // Use our original coordinator host as the leader IS the original
    // coordinator in the typical case.
    if (coordinator_addr.find("0.0.0.0") == 0) {
      auto leader_colon = coordinator_addr.rfind(':');
      auto our_colon = coordinator_addr_.rfind(':');
      if (leader_colon != std::string::npos && our_colon != std::string::npos) {
        std::string our_host = coordinator_addr_.substr(0, our_colon);
        coordinator_addr = our_host + coordinator_addr.substr(leader_colon);
      }
    }
  }

  std::cerr << "[jaccl] split: rank=" << rank_ << " new_rank=" << new_rank
            << " color=" << color << " coordinator=" << coordinator_addr
            << std::endl;

  // Step 5: Construct the sub-group MeshGroup.
  return std::make_shared<MeshGroup>(
      new_rank, new_device_names, coordinator_addr.c_str());
}

} // namespace mlx::core::distributed::jaccl
