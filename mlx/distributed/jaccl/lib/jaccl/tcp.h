// Copyright © 2025 Apple Inc.

#pragma once

#include <sys/socket.h>
#include <functional>
#include <string>

namespace jaccl {

struct address_t {
  sockaddr_storage addr;
  socklen_t len;

  const sockaddr* get() const {
    return (struct sockaddr*)&addr;
  }
};

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port);

/**
 * Parse a sockaddr provided as an <ip>:<port> string.
 */
address_t parse_address(const std::string& ip_port);

/**
 * Reserve an ephemeral TCP port on `ip` and return it.
 *
 * Binds a throwaway socket to <ip>:0, reads back the kernel-assigned port
 * with getsockname(), then closes the socket. The port is NOT held open --
 * the caller is expected to publish it to its peers and then bind it for
 * real. That leaves a small window in which an unrelated process could take
 * the port, but the socket is closed without ever having accepted a
 * connection (so it leaves no TIME_WAIT entry) and TCPSocket::listen() sets
 * SO_REUSEADDR/SO_REUSEPORT, so the realistic failure mode is a clean
 * bind error rather than silent cross-talk.
 *
 * Used by MeshGroup::split() to pick a collision-free port for a subgroup's
 * dedicated coordinator SideChannel, instead of deriving one arithmetically
 * from the parent's port (which collides across co-hosted model instances).
 */
int reserve_ephemeral_port(const std::string& ip);

/**
 * Small wrapper over a TCP socket to simplify initiating connections.
 */
class TCPSocket {
 public:
  TCPSocket(const char* tag);
  TCPSocket(const TCPSocket&) = delete;
  TCPSocket& operator=(const TCPSocket&) = delete;
  TCPSocket(TCPSocket&& s);
  TCPSocket& operator=(TCPSocket&&);
  ~TCPSocket();

  void listen(const char* tag, const address_t& addr);
  TCPSocket accept(const char* tag);

  void send(const char* tag, const void* data, size_t len);
  void recv(const char* tag, void* data, size_t len);

  // Bound blocking recv() with SO_RCVTIMEO so a stuck peer fails cleanly
  // (recv throws) instead of hanging forever. 0 = no timeout (default).
  void set_recv_timeout_secs(int secs);

  // Override the ELAPSED no-progress retry deadline recv() enforces on top
  // of the per-syscall SO_RCVTIMEO above (see recv()'s own comment for the
  // two-timer distinction). -1 (default) means "no override -- fall back to
  // MLX_JACCL_RECV_RETRY_DEADLINE_SECS / the 60s hardcoded default", i.e.
  // unchanged legacy behavior. Added 2026-08-09 (design doc Section 30) so
  // the coordinator/recovery side channel can be given a LONGER deadline
  // than ordinary data-path sockets -- see recv()'s comment for why the two
  // must differ.
  void set_recv_retry_deadline_secs(double secs);

  int detach();

  operator int() const {
    return sock_;
  }

  static TCPSocket connect(
      const char* tag,
      const address_t& addr,
      int num_retries = 1,
      int wait = 0,
      std::function<void(int, int)> cb = nullptr);

 private:
  TCPSocket(int sock);

  int sock_;
  // -1 = unset (recv() falls back to MLX_JACCL_RECV_RETRY_DEADLINE_SECS /
  // 60.0 default). See set_recv_retry_deadline_secs().
  double recv_retry_deadline_secs_override_ = -1.0;
};

} // namespace jaccl
