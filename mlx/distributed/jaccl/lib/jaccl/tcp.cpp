// Copyright © 2025 Apple Inc.

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "jaccl/tcp.h"

namespace jaccl {

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port) {
  struct addrinfo hints, *res;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  if (status != 0) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

  address_t result;
  memcpy(&result.addr, res->ai_addr, res->ai_addrlen);
  result.len = res->ai_addrlen;
  freeaddrinfo(res);

  return result;
}

/**
 * Parse a sockaddr provided as an <ip>:<port> string.
 */
address_t parse_address(const std::string& ip_port) {
  auto colon = ip_port.find(":");
  if (colon == std::string::npos) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip_port;
    throw std::runtime_error(msg.str());
  }
  std::string ip(ip_port.begin(), ip_port.begin() + colon);
  std::string port(ip_port.begin() + colon + 1, ip_port.end());

  return parse_address(ip, port);
}

int reserve_ephemeral_port(const std::string& ip) {
  auto addr = parse_address(ip, "0");
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    throw std::runtime_error(
        "[jaccl] reserve_ephemeral_port: couldn't create socket");
  }
  int enable = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
  setsockopt(s, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
  if (bind(s, addr.get(), addr.len) < 0) {
    int e = errno;
    close(s);
    std::ostringstream msg;
    msg << "[jaccl] reserve_ephemeral_port: bind failed on " << ip
        << ":0 (errno=" << e << ")";
    throw std::runtime_error(msg.str());
  }
  sockaddr_storage bound{};
  socklen_t len = sizeof(bound);
  if (getsockname(s, reinterpret_cast<sockaddr*>(&bound), &len) < 0) {
    int e = errno;
    close(s);
    std::ostringstream msg;
    msg << "[jaccl] reserve_ephemeral_port: getsockname failed (errno=" << e
        << ")";
    throw std::runtime_error(msg.str());
  }
  close(s);
  if (bound.ss_family == AF_INET6) {
    return ntohs(reinterpret_cast<sockaddr_in6*>(&bound)->sin6_port);
  }
  return ntohs(reinterpret_cast<sockaddr_in*>(&bound)->sin_port);
}

TCPSocket::TCPSocket(const char* tag) {
  sock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_ < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't create socket (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
}

TCPSocket::TCPSocket(TCPSocket&& s) {
  sock_ = s.sock_;
  recv_retry_deadline_secs_override_ = s.recv_retry_deadline_secs_override_;
  s.sock_ = -1;
}

TCPSocket& TCPSocket::operator=(TCPSocket&& s) {
  if (this != &s) {
    sock_ = s.sock_;
    recv_retry_deadline_secs_override_ = s.recv_retry_deadline_secs_override_;
    s.sock_ = -1;
  }
  return *this;
}

TCPSocket::TCPSocket(int s) : sock_(s) {}

TCPSocket::~TCPSocket() {
  if (sock_ > 0) {
    shutdown(sock_, 2);
    close(sock_);
  }
}

int TCPSocket::detach() {
  int s = sock_;
  sock_ = -1;
  return s;
}

void TCPSocket::listen(const char* tag, const address_t& addr) {
  int success;

  // Make sure we can launch immediately after shutdown by setting the
  // reuseaddr option so that we don't get address already in use errors
  int enable = 1;
  success = setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't enable reuseaddr (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
  success = setsockopt(sock_, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't enable reuseport (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  // Bind the socket to the address and port
  success = bind(sock_, addr.get(), addr.len);
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't bind socket (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  // Prepare waiting for connections
  success = ::listen(sock_, 0);
  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't listen (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }
}

TCPSocket TCPSocket::accept(const char* tag) {
  int peer = ::accept(sock_, nullptr, nullptr);
  if (peer < 0) {
    std::ostringstream msg;
    msg << tag << " Accept failed (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  return TCPSocket(peer);
}

void TCPSocket::send(const char* tag, const void* data, size_t len) {
  while (len > 0) {
    auto n = ::send(sock_, data, len, 0);
    if (n <= 0) {
      std::ostringstream msg;
      msg << tag << " Send failed with errno=" << errno;
      throw std::runtime_error(msg.str());
    }
    len -= n;
    data = static_cast<const char*>(data) + n;
  }
}

void TCPSocket::set_recv_timeout_secs(int secs) {
  timeval tv{};
  tv.tv_sec = secs;
  tv.tv_usec = 0;
  // Bound both recv and send so a stuck/backpressured coordinator op fails
  // cleanly (throws) instead of hanging forever.
  setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void TCPSocket::set_recv_retry_deadline_secs(double secs) {
  recv_retry_deadline_secs_override_ = secs;
}

void TCPSocket::recv(const char* tag, void* data, size_t len) {
  // RETRY-ON-EAGAIN FIX (2026-07-18): previously any recv() <= 0 threw
  // immediately, including errno=EAGAIN/EWOULDBLOCK from a plain
  // SO_RCVTIMEO expiry -- "peer hasn't sent yet this round" is NOT the
  // same condition as a dead/closed connection, but was treated
  // identically. Under the higher-message-frequency PP speculative-
  // decode protocols (multiple small p2p_retry_barrier round-trips per
  // decode cycle instead of one), transient scheduling jitter on either
  // rank (GC pause, Metal command-buffer queueing delay, thermal
  // throttling, etc.) could exceed one 35s SO_RCVTIMEO window without
  // the peer being remotely unhealthy -- confirmed empirically: this
  // threw mid p2p_retry_barrier's 16-byte ack recv (remaining=16, i.e.
  // the FIRST byte of this round's reply, not a stall mid-message),
  // self-healed the FIRST time (group.reconnect() succeeded), then
  // recurred and this time ALSO hit during reconnect's own side_channel_
  // recv (same TCPSocket::recv), causing the recovery attempt itself to
  // throw -> RunnerFailed (full re-place) instead of the intended
  // in-place recovery.
  //
  // Fix distinguishes:
  //   n == 0            -> peer sent FIN (orderly close). FATAL,
  //                        immediate throw -- errno is not meaningful
  //                        here (stale from a prior call), so this
  //                        check must come BEFORE any errno inspection.
  //   n < 0, EAGAIN/EWOULDBLOCK -> SO_RCVTIMEO expired, zero bytes
  //                        consumed this call (kernel doesn't discard
  //                        partial TCP stream data on timeout -- safe
  //                        to retry into the same buffer offset).
  //                        Retryable, bounded by an ELAPSED deadline
  //                        that resets on every partial-progress recv
  //                        (n > 0), not a raw retry count -- a peer
  //                        that's slow-but-alive and trickling data
  //                        keeps getting fresh budget; one that's
  //                        genuinely wedged for the full deadline with
  //                        zero progress still throws.
  //   n < 0, EINTR       -> interrupted by a signal, not a real fault.
  //                        Retry unconditionally, doesn't consume
  //                        retry-deadline budget.
  //   n < 0, anything else (ECONNRESET, EPIPE, ETIMEDOUT, ENOTCONN,
  //                        EBADF, ...) -> genuinely fatal transport
  //                        fault. Immediate throw, unchanged behavior.
  const char* deadline_env = std::getenv("MLX_JACCL_RECV_RETRY_DEADLINE_SECS");
  // Per-socket override (set_recv_retry_deadline_secs()) takes priority over
  // the env var / 60.0 hardcoded default -- see tcp.h's comment on why the
  // coordinator/recovery side channel needs a DIFFERENT (longer) deadline
  // than ordinary data-path sockets.
  const double deadline_secs = recv_retry_deadline_secs_override_ >= 0.0
      ? recv_retry_deadline_secs_override_
      : (deadline_env ? std::atof(deadline_env) : 60.0);
  auto deadline_start = std::chrono::steady_clock::now();
  const auto _wait_t0 = deadline_start;
  bool warned_once = false;
  // 2026-08-09 (design doc Section 38, real-hardware finding): a recv()
  // that eventually SUCCEEDS after one or more EAGAIN retries used to be
  // invisible -- only the DEADLINE-EXCEEDED case ever got logged. That
  // means every prior deadline-sizing decision (including this comment's
  // own p2p_channel_ default below) was made from survivorship-biased
  // data: we only ever saw waits that exceeded whatever deadline was set
  // at the time, never the FULL distribution of legitimate wait times
  // that stayed under it. Log any recv() that took longer than
  // MLX_JACCL_RECV_WAIT_LOG_THRESHOLD_SECS (default 5s) to eventually
  // succeed, so the NEXT deadline-tuning pass has real data instead of
  // guessing from deadline-exceeded throws alone.
  const double _wait_log_threshold_secs = [] {
    const char* e = std::getenv("MLX_JACCL_RECV_WAIT_LOG_THRESHOLD_SECS");
    return e ? std::atof(e) : 5.0;
  }();

  while (len > 0) {
    auto n = ::recv(sock_, data, len, 0);
    if (n > 0) {
      len -= n;
      data = static_cast<char*>(data) + n;
      deadline_start = std::chrono::steady_clock::now(); // progress: reset
      if (warned_once) {
        const double _total_wait =
            std::chrono::duration<double>(deadline_start - _wait_t0).count();
        if (_total_wait >= _wait_log_threshold_secs) {
          fprintf(
              stderr,
              "[jaccl] %s recv succeeded after a %.1fs stall (deadline was "
              "%.1fs) -- logged for future deadline-sizing data, not an "
              "error\n",
              tag,
              _total_wait,
              deadline_secs);
          fflush(stderr);
        }
      }
      warned_once = false;
      continue;
    }
    if (n == 0) {
      // Peer closed the connection (EOF). errno is NOT meaningful for a
      // 0-return -- do not inspect it. Always fatal.
      std::ostringstream msg;
      msg << tag << " Recv failed: peer closed connection (EOF) fd=" << sock_
          << " remaining=" << len;
      throw std::runtime_error(msg.str());
    }
    // n < 0 from here on.
    int e = errno;
    if (e == EINTR) {
      continue; // Signal interruption, not a fault -- no deadline cost.
    }
    if (e == EAGAIN || e == EWOULDBLOCK) {
      auto elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - deadline_start)
                         .count();
      if (deadline_secs > 0 && elapsed >= deadline_secs) {
        int fl = ::fcntl(sock_, F_GETFL);
        std::ostringstream msg;
        msg << tag << " Recv failed with errno=" << e << " n=" << n
            << " fd=" << sock_ << " remaining=" << len << " flags=0x"
            << std::hex << fl << std::dec << " nonblock="
            << ((fl >= 0 && (fl & O_NONBLOCK)) ? 1 : 0)
            << " (no progress for " << elapsed << "s, retry deadline "
            << deadline_secs << "s exceeded)";
        throw std::runtime_error(msg.str());
      }
      if (!warned_once) {
        // One heartbeat per stall onset (not per SO_RCVTIMEO retry --
        // that would spam at one line per 35s timeout window) so a
        // genuinely stuck peer is visible in logs before the deadline
        // fires, distinguishing "still waiting, within budget" from a
        // silent multi-minute hang.
        fprintf(
            stderr,
            "[jaccl] %s recv EAGAIN, retrying (remaining=%zu, elapsed=%.1fs, "
            "deadline=%.1fs)\n",
            tag,
            len,
            elapsed,
            deadline_secs);
        fflush(stderr);
        warned_once = true;
      }
      continue; // Retry the same ::recv() call -- no bytes were consumed.
    }
    // Any other errno: genuinely fatal (ECONNRESET, EPIPE, ETIMEDOUT,
    // ENOTCONN, EBADF, ...). Unchanged from the original behavior.
    int fl = ::fcntl(sock_, F_GETFL);
    std::ostringstream msg;
    msg << tag << " Recv failed with errno=" << e << " n=" << n
        << " fd=" << sock_ << " remaining=" << len << " flags=0x" << std::hex
        << fl << std::dec << " nonblock="
        << ((fl >= 0 && (fl & O_NONBLOCK)) ? 1 : 0);
    throw std::runtime_error(msg.str());
  }
}

TCPSocket TCPSocket::connect(
    const char* tag,
    const address_t& addr,
    int num_retries,
    int wait,
    std::function<void(int, int)> cb) {
  int sock, success;

  // Attempt to connect `num_retries` times with exponential backoff.
  for (int attempt = 0; attempt < num_retries; attempt++) {
    // Create the socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::ostringstream msg;
      msg << tag << " Couldn't create socket to connect (error: " << errno
          << ")";
      throw std::runtime_error(msg.str());
    }

    success = ::connect(sock, addr.get(), addr.len);
    if (success == 0) {
      break;
    }

    if (cb != nullptr) {
      cb(attempt, wait);
    }
    if (wait > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(wait));
    }

    wait <<= 1;
  }

  if (success < 0) {
    std::ostringstream msg;
    msg << tag << " Couldn't connect (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  return TCPSocket(sock);
}

} // namespace jaccl
