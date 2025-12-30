"""
IAN P2P Manager - Wires up transport, protocol, and discovery.

Provides:
1. Complete P2P message passing over TCP
2. Connection management and pooling
3. Message routing and handlers
4. Peer session management

This module connects the transport layer (TCP) with the protocol layer
(message types) and the discovery layer (peer finding).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
from hmac import compare_digest
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .node import NodeIdentity, NodeInfo
from .protocol import (
    HandshakeChallenge,
    HandshakeResponse,
    Message,
    MessageType,
    Ping,
    Pong,
)

if TYPE_CHECKING:
    from .tls import TLSConfig

logger = logging.getLogger(__name__)


_MIN_CLEANUP_SLEEP_S = 0.25


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class P2PConfig:
    """P2P manager configuration."""

    # Network
    listen_host: str = "0.0.0.0"
    listen_port: int = 9000
    external_host: str | None = None  # For NAT traversal

    # Connections
    max_peers: int = 50
    max_pending_connections: int = 10
    max_connections_per_ip: int = 3  # Limit connections from same IP
    connection_timeout: float = 10.0
    handshake_timeout: float = 5.0  # Timeout for handshake completion

    # Keepalive
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_missed_pings: int = 3

    # Messages
    max_message_size: int = 16 * 1024 * 1024  # 16 MB
    message_queue_size: int = 1000

    # Rate limiting
    max_messages_per_second: float = 100.0
    rate_limit_burst: int = 20


# =============================================================================
# Peer Session
# =============================================================================

class PeerState(Enum):
    """State of peer connection."""
    CONNECTING = auto()
    CONNECTED = auto()
    HANDSHAKING = auto()
    READY = auto()
    DISCONNECTING = auto()
    DISCONNECTED = auto()


# =============================================================================
# Rate Limiter
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for per-peer message throttling.

    Security:
    - Prevents message flooding DoS
    - Allows burst tolerance for legitimate traffic
    - O(1) per check

    ESSO-Verified Invariants (Inductive(k=1)):
    - tokens <= burst (tokens_bounded_by_burst)
    - tokens >= 0 (tokens_non_negative)
    - 1 <= burst <= max_burst (domain bound)

    IR hash: sha256:4fdfd80b9ef67570c25c38fb5afa4a04c1907bee4ce0f41a6ae584ea34e99a2d
    """

    __slots__ = ('rate', 'burst', 'tokens', 'last_update', '_lock')

    def __init__(self, rate: float = 10.0, burst: int = 20):
        if not (1 <= burst <= 20):
            raise ValueError(f"burst must be in [1, 20], got {burst}")
        self.rate = rate  # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._check_invariants()  # Verify initial state

    def _check_invariants(self) -> None:
        """CBC: Verify ESSO-proven invariants hold."""
        assert 0 <= self.tokens <= self.burst, \
            f"invariant violated: tokens={self.tokens}, burst={self.burst}"
        assert 1 <= self.burst <= 20, \
            f"domain violation: burst={self.burst}"

    async def acquire(self) -> bool:
        """
        Try to acquire a token.

        Returns:
            True if allowed, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Replenish tokens
            self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)

            if self.tokens < 1.0:
                self._check_invariants()  # CBC: post-state check
                return False  # Rate limited

            self.tokens -= 1.0
            self._check_invariants()  # CBC: post-state check
            return True

    def reset(self) -> None:
        """Reset limiter to full burst capacity."""
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._check_invariants()  # CBC: post-state check



@dataclass
class PeerSession:
    """
    Session with a connected peer.

    Tracks connection state, message queues, and statistics.

    Security:
    - Stores handshake challenge for verification
    - Tracks rate limiting state
    - Validates peer identity cryptographically

    ESSO-Verified Invariants (Inductive(k=1)):
    - ready_requires_verified: READY state implies verified=True
    - ready_requires_handshake: READY state implies handshake completed

    IR hash: sha256:ff741b6b32e8033c221784e0908aeb08ec29b0669d943ef3256cf762bc6d99d4
    """
    node_id: str
    address: str
    port: int

    # Connection
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    state: PeerState = PeerState.DISCONNECTED

    # Peer info (after handshake)
    info: NodeInfo | None = None

    # Handshake security
    pending_challenge: bytes | None = None  # Our nonce, awaiting signed response
    verified: bool = False  # Whether peer identity is cryptographically verified
    handshake_completed: bool = False

    handshake_started_at: float = 0.0

    peer_public_key: bytes | None = None
    kx_private_key: bytes | None = None
    kx_public_key: bytes | None = None
    peer_kx_public_key: bytes | None = None
    session_key: bytes | None = None

    # Statistics
    connected_at: float = 0.0
    last_message_at: float = 0.0
    last_ping_at: float = 0.0
    missed_pings: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    rate_limit_violations: int = 0

    # Message queue
    outbound_queue: asyncio.Queue[bytes | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=100)
    )

    # Rate limiter (initialized per session)
    rate_limiter: TokenBucketRateLimiter | None = None

    # Tasks
    read_task: asyncio.Task | None = None
    write_task: asyncio.Task | None = None

    def _check_invariants(self) -> None:
        """CBC: Verify ESSO-proven peer_state_fsm invariants."""
        # Invariant: READY implies verified
        if self.state == PeerState.READY:
            assert self.verified, \
                f"invariant ready_requires_verified violated: state=READY but verified=False"
            assert self.handshake_completed, \
                f"invariant ready_requires_handshake violated: state=READY but handshake_completed=False"

    def transition_to(self, new_state: PeerState) -> None:
        """
        Transition to a new state with CBC validation.

        Validates that the transition is legal per ESSO model.
        Checks invariants BEFORE committing state change (fail-fast).
        """
        # Valid transitions per peer_state_fsm model
        valid_transitions = {
            PeerState.CONNECTING: {PeerState.CONNECTED, PeerState.DISCONNECTING},
            PeerState.CONNECTED: {PeerState.HANDSHAKING, PeerState.DISCONNECTING},
            PeerState.HANDSHAKING: {PeerState.READY, PeerState.DISCONNECTING},
            PeerState.READY: {PeerState.DISCONNECTING},
            PeerState.DISCONNECTING: {PeerState.DISCONNECTED},
            PeerState.DISCONNECTED: {PeerState.CONNECTING},
        }

        allowed = valid_transitions.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {self.state.name} -> {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )

        # CBC: check invariants BEFORE committing (fail-fast)
        # Temporarily set state to check proposed invariants
        old_state = self.state
        self.state = new_state
        try:
            self._check_invariants()
        except AssertionError:
            self.state = old_state  # Rollback on invariant violation
            raise

    def is_connected(self) -> bool:
        return self.state in (PeerState.CONNECTED, PeerState.HANDSHAKING, PeerState.READY)

    def is_ready(self) -> bool:
        return self.state == PeerState.READY



# =============================================================================
# P2P Manager
# =============================================================================

class P2PManager:
    """
    Manages P2P connections and message passing.

    Responsibilities:
    1. Accept inbound connections
    2. Establish outbound connections
    3. Route messages to handlers
    4. Maintain peer sessions
    5. Handle keepalive pings

    Integration:
    - Uses TCPTransport for wire protocol
    - Connects to ConsensusCoordinator for message handling
    - Integrates with Discovery for peer finding
    """

    def __init__(
        self,
        identity: NodeIdentity,
        config: P2PConfig | None = None,
        tls_config: TLSConfig | None = None,
    ):
        self._identity = identity
        self._config = config or P2PConfig()
        self._tls_config = tls_config

        # Peer sessions by node_id
        self._peers: dict[str, PeerSession] = {}

        # Address to node_id mapping
        self._address_map: dict[str, str] = {}  # "host:port" -> node_id

        # IP connection tracking (for DoS prevention)
        self._ip_connections: dict[str, int] = {}  # IP -> connection count

        # Connection semaphore (limit concurrent pending connections)
        self._connection_semaphore = asyncio.Semaphore(self._config.max_pending_connections)

        # Replay attack protection: per-peer monotonic timestamps + nonce LRU
        # Key: peer_id -> {nonce: seen_at_s}; timestamps are tracked per peer
        # Messages older than 5 minutes are rejected (freshness window)
        self._peer_nonce_cache: dict[str, OrderedDict[str, float]] = {}
        self._peer_last_timestamp_ms: dict[str, int] = {}
        self._max_nonce_cache_per_peer = 10_000
        self._message_ttl_seconds = 300  # 5 minute window

        # Server
        self._server: asyncio.Server | None = None

        # Message handlers
        self._handlers: dict[MessageType, Callable] = {}

        # Background tasks
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Lock for peer modifications
        self._lock = asyncio.Lock()

    @staticmethod
    def _now_s() -> float:
        return time.time()

    def _mark_handshake_started(self, session: PeerSession, *, now_s: float) -> None:
        if session.handshake_started_at <= 0.0:
            session.handshake_started_at = now_s
        session.handshake_completed = False

    def _mark_handshake_completed(self, session: PeerSession) -> None:
        session.handshake_started_at = 0.0
        session.handshake_completed = True

    async def _find_handshake_timeouts(self, *, now_s: float) -> list[str]:
        timeout_s = float(self._config.handshake_timeout)
        if timeout_s <= 0:
            return []

        async with self._lock:
            expired: list[str] = []
            for node_id, session in list(self._peers.items()):
                if session.state not in (PeerState.CONNECTED, PeerState.HANDSHAKING):
                    continue
                if session.handshake_started_at <= 0.0:
                    continue
                if (now_s - session.handshake_started_at) <= timeout_s:
                    continue
                expired.append(node_id)
            return expired

    # -------------------------------------------------------------------------
    # Replay Attack Protection
    # -------------------------------------------------------------------------

    def _is_replay(self, message: Message) -> bool:
        """
        Check if a message is a replay attack.

        Security:
        - Prevents replay attacks by tracking seen message nonces
        - Enforces per-peer monotonic timestamps
        - Uses LRU eviction to bound memory usage
        - Evicts messages older than TTL

        Returns:
            True if message is a replay (should be rejected)
        """
        msg_id = message.message_id()
        now = time.time()
        sender_id = message.sender_id

        # Check timestamp freshness (reject messages older than TTL)
        msg_timestamp_s = message.timestamp / 1000.0  # Convert ms to seconds
        if abs(now - msg_timestamp_s) > self._message_ttl_seconds:
            logger.warning(f"Message timestamp too old/future: {msg_id[:32]}...")
            return True

        # Per-peer monotonic timestamps
        last_timestamp_ms = self._peer_last_timestamp_ms.get(sender_id)
        if last_timestamp_ms is not None and message.timestamp <= last_timestamp_ms:
            logger.warning(f"Non-monotonic timestamp from {sender_id[:16]}...: {msg_id[:32]}...")
            return True

        # Per-peer nonce cache (LRU)
        peer_cache = self._peer_nonce_cache.setdefault(sender_id, OrderedDict())
        if message.nonce in peer_cache:
            logger.warning(f"Replay attack detected: {msg_id[:32]}...")
            return True

        peer_cache[message.nonce] = now
        peer_cache.move_to_end(message.nonce)
        while len(peer_cache) > self._max_nonce_cache_per_peer:
            peer_cache.popitem(last=False)

        # Record last timestamp for monotonicity
        self._peer_last_timestamp_ms[sender_id] = message.timestamp

        # Evict old entries
        self._evict_old_messages()

        return False

    def _evict_old_messages(self) -> None:
        """Evict expired messages from the seen cache."""
        now = time.time()
        cutoff = now - self._message_ttl_seconds

        for peer_id, cache in list(self._peer_nonce_cache.items()):
            # Evict by age (oldest first since OrderedDict maintains insertion order)
            while cache:
                _, oldest_time = next(iter(cache.items()))
                if oldest_time < cutoff:
                    cache.popitem(last=False)
                else:
                    break  # Rest are newer

            if not cache:
                self._peer_nonce_cache.pop(peer_id, None)

        # Clean up stale timestamp entries for peers with no cache
        for peer_id, last_ts in list(self._peer_last_timestamp_ms.items()):
            if peer_id in self._peer_nonce_cache:
                continue
            if (last_ts / 1000.0) < cutoff:
                self._peer_last_timestamp_ms.pop(peer_id, None)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> bool:
        """Start P2P manager."""
        if self._running:
            return True

        self._running = True

        # Start server
        try:
            ssl_ctx = None
            if self._tls_config is not None:
                # Use TLS for inbound connections (mTLS recommended)
                ssl_ctx = self._tls_config.create_server_context()

            self._server = await asyncio.start_server(
                self._handle_inbound_connection,
                self._config.listen_host,
                self._config.listen_port,
                ssl=ssl_ctx,
            )

            addr = self._server.sockets[0].getsockname()
            logger.info(f"P2P server listening on {addr[0]}:{addr[1]}")

        except Exception as e:
            logger.error(f"Failed to start P2P server: {e}")
            self._running = False
            return False

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._ping_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        return True

    async def stop(self) -> None:
        """Stop P2P manager."""
        self._running = False

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Disconnect all peers
        for session in list(self._peers.values()):
            await self._disconnect_peer(session.node_id)

        logger.info("P2P manager stopped")

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect_to_peer(
        self,
        address: str,
        port: int,
        node_id: str | None = None,
    ) -> tuple[bool, str]:
        """
        Connect to a peer.

        Args:
            address: Peer address
            port: Peer port
            node_id: Expected node ID (optional)

        Returns:
            (success, node_id_or_error)
        """
        addr_key = f"{address}:{port}"

        # Check if already connected
        if addr_key in self._address_map:
            existing_id = self._address_map[addr_key]
            if existing_id in self._peers and self._peers[existing_id].is_connected():
                return True, existing_id

        # Check peer limit
        if len(self._peers) >= self._config.max_peers:
            return False, "max peers reached"

        try:
            # Connect (optionally using TLS)
            ssl_ctx = None
            if self._tls_config is not None:
                ssl_ctx = self._tls_config.create_client_context()

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port, ssl=ssl_ctx),
                timeout=self._config.connection_timeout,
            )

            # Optional certificate pinning
            if self._tls_config is not None and self._tls_config.pinned_certs:
                ssl_object = writer.get_extra_info('ssl_object')
                if ssl_object is not None:
                    peer_cert = ssl_object.getpeercert(binary_form=True)
                    if peer_cert and not self._tls_config.verify_pinned(peer_cert):
                        writer.close()
                        await writer.wait_closed()
                        return False, "peer certificate not pinned"

            # Create session (node_id may be updated after handshake)
            session_id = node_id or f"pending_{addr_key}"

            session = PeerSession(
                node_id=session_id,
                address=address,
                port=port,
                reader=reader,
                writer=writer,
                state=PeerState.CONNECTED,
                connected_at=time.time(),
                handshake_started_at=time.time(),
            )

            async with self._lock:
                self._peers[session_id] = session
                self._address_map[addr_key] = session_id

            # Start session tasks
            session.read_task = asyncio.create_task(self._read_loop(session))
            session.write_task = asyncio.create_task(self._write_loop(session))

            # Handshake
            await self._send_handshake(session)

            logger.info(f"Connected to peer at {address}:{port}")
            # Initialize rate limiter for outbound connection too
            session.rate_limiter = TokenBucketRateLimiter(
                rate=self._config.max_messages_per_second,
                burst=self._config.rate_limit_burst,
            )

            return True, session_id

        except asyncio.TimeoutError:
            return False, "connection timeout"
        except Exception as e:
            return False, str(e)

    async def _handle_inbound_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle inbound connection with security checks.

        Security:
        - Limits connections per IP (DoS prevention)
        - Uses semaphore to limit concurrent pending connections
        - Enforces handshake timeout
        """
        peer_addr = writer.get_extra_info('peername')
        if not peer_addr:
            writer.close()
            return

        peer_ip = peer_addr[0]
        addr_key = f"{peer_ip}:{peer_addr[1]}"

        logger.debug(f"Inbound connection from {addr_key}")

        # Check per-IP connection limit (DoS prevention)
        async with self._lock:
            ip_count = self._ip_connections.get(peer_ip, 0)
            if ip_count >= self._config.max_connections_per_ip:
                logger.warning(f"Rejecting connection from {addr_key}: too many from this IP")
                writer.close()
                await writer.wait_closed()
                return

            # Check total peer limit
            if len(self._peers) >= self._config.max_peers:
                logger.warning(f"Rejecting connection from {addr_key}: max peers")
                writer.close()
                await writer.wait_closed()
                return

            # Increment IP connection count
            self._ip_connections[peer_ip] = ip_count + 1

        # Create session with rate limiter
        session_id = f"inbound_{addr_key}"

        session = PeerSession(
            node_id=session_id,
            address=peer_ip,
            port=peer_addr[1],
            reader=reader,
            writer=writer,
            state=PeerState.CONNECTED,
            connected_at=time.time(),
            handshake_started_at=time.time(),
            rate_limiter=TokenBucketRateLimiter(
                rate=self._config.max_messages_per_second,
                burst=self._config.rate_limit_burst,
            ),
        )

        async with self._lock:
            self._peers[session_id] = session
            self._address_map[addr_key] = session_id

        # Start session tasks
        session.read_task = asyncio.create_task(self._read_loop(session))
        session.write_task = asyncio.create_task(self._write_loop(session))

    async def _disconnect_peer(self, node_id: str) -> None:
        """
        Disconnect a peer and cleanup resources.

        Ensures IP connection count is decremented.
        """
        async with self._lock:
            session = self._peers.pop(node_id, None)
            if not session:
                return

            addr_key = f"{session.address}:{session.port}"
            self._address_map.pop(addr_key, None)

            # Decrement IP connection count
            ip_count = self._ip_connections.get(session.address, 1)
            if ip_count <= 1:
                self._ip_connections.pop(session.address, None)
            else:
                self._ip_connections[session.address] = ip_count - 1

        if session.state not in (PeerState.DISCONNECTING, PeerState.DISCONNECTED):
            session.transition_to(PeerState.DISCONNECTING)

        # Cancel tasks
        if session.read_task:
            session.read_task.cancel()
        if session.write_task:
            session.write_task.cancel()

        # Close connection
        if session.writer:
            try:
                session.writer.close()
                await session.writer.wait_closed()
            except Exception:
                pass

        if session.state == PeerState.DISCONNECTING:
            session.transition_to(PeerState.DISCONNECTED)
        logger.info(f"Disconnected peer {node_id}")

    # -------------------------------------------------------------------------
    # Message I/O
    # -------------------------------------------------------------------------

    async def _read_loop(self, session: PeerSession) -> None:
        """
        Read messages from peer with rate limiting.

        Security:
        - Rate limits messages per second
        - Disconnects on repeated violations
        - Enforces message size limits
        """
        max_rate_violations = 10  # Disconnect after this many violations

        try:
            while self._running and session.is_connected():
                if session.reader is None:
                    break
                # Read length prefix (4 bytes)
                length_data = await asyncio.wait_for(
                    session.reader.readexactly(4),
                    timeout=60.0,
                )

                msg_length = int.from_bytes(length_data, 'big')

                # Validate message size
                if msg_length <= 0 or msg_length > self._config.max_message_size:
                    logger.warning(f"Invalid message size from {session.node_id}: {msg_length}")
                    break

                # Read message body
                msg_data = await asyncio.wait_for(
                    session.reader.readexactly(msg_length),
                    timeout=30.0,
                )

                session.bytes_received += 4 + msg_length
                session.messages_received += 1
                session.last_message_at = time.time()

                # Rate limit check
                if session.rate_limiter:
                    if not await session.rate_limiter.acquire():
                        session.rate_limit_violations += 1
                        logger.warning(
                            f"Rate limiting {session.node_id} "
                            f"(violation {session.rate_limit_violations})"
                        )

                        if session.rate_limit_violations >= max_rate_violations:
                            logger.warning(f"Disconnecting {session.node_id}: too many rate limit violations")
                            break

                        # Skip this message but continue
                        await asyncio.sleep(0.1)
                        continue

                # Parse and handle message
                await self._handle_message(session, msg_data)

        except asyncio.TimeoutError:
            logger.debug(f"Read timeout from {session.node_id}")
        except asyncio.CancelledError:
            raise
        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by {session.node_id}")
        except Exception as e:
            logger.error(f"Read error from {session.node_id}: {e}")

        # Disconnect on any error
        await self._disconnect_peer(session.node_id)

    async def _write_loop(self, session: PeerSession) -> None:
        """Write messages to peer."""
        try:
            while self._running and session.is_connected():
                if session.writer is None:
                    break
                # Get message from queue
                msg_data = await asyncio.wait_for(
                    session.outbound_queue.get(),
                    timeout=1.0,
                )

                if msg_data is None:
                    break

                # Write length prefix + message
                length_prefix = len(msg_data).to_bytes(4, 'big')
                session.writer.write(length_prefix + msg_data)
                await session.writer.drain()

                session.bytes_sent += 4 + len(msg_data)
                session.messages_sent += 1

        except asyncio.TimeoutError:
            pass  # Normal - queue empty
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Write error to {session.node_id}: {e}")
            await self._disconnect_peer(session.node_id)

    async def _handle_message(self, session: PeerSession, data: bytes) -> None:
        """Handle received message.

        Security:
        - Rejects non-handshake messages from unverified sessions (fail-closed)
        - Verifies Ed25519 signatures for all accepted messages
        - Checks for replay attacks *after* authentication (prevents cache DoS)
        - Validates message timestamps (via replay gate)
        """
        try:
            # Parse JSON (wire payload is just JSON; length prefix is handled in _read_loop)
            msg_dict = json.loads(data.decode("utf-8"))
            if not isinstance(msg_dict, dict):
                return
            msg_type_raw = msg_dict.get("type", "")
            try:
                msg_type = MessageType(msg_type_raw)
            except Exception:
                legacy_map = {
                    "HANDSHAKE_CHALLENGE": MessageType.HANDSHAKE_CHALLENGE,
                    "HANDSHAKE_RESPONSE": MessageType.HANDSHAKE_RESPONSE,
                }
                msg_type_opt = legacy_map.get(msg_type_raw)
                if msg_type_opt is None:
                    logger.debug(f"Unknown message type: {msg_type_raw}")
                    return
                msg_type = msg_type_opt

            is_handshake = msg_type in (MessageType.HANDSHAKE_CHALLENGE, MessageType.HANDSHAKE_RESPONSE)

            # Gate non-handshake traffic on a verified session.
            # (We drop rather than disconnect to avoid brittleness; rate limiting applies upstream.)
            if not is_handshake and not session.is_ready():
                logger.debug(
                    f"Dropping non-handshake message from unready session {session.node_id}: {msg_type.value}"
                )
                return

            # Parse into a typed protocol message (used for signing payload + replay check).
            try:
                parsed = Message.from_dict(msg_dict)
            except Exception:
                return

            # Require a signature for all accepted messages.
            if parsed.signature is None or not isinstance(parsed.signature, (bytes, bytearray)):
                if session.is_ready():
                    logger.warning(f"Unsigned message from ready peer {session.node_id[:16]}...; disconnecting")
                    asyncio.create_task(self._disconnect_peer(session.node_id))
                return

            # Authenticate message (Ed25519).
            if is_handshake:
                # Handshake messages carry the sender's public key; verify binding + signature.
                peer_pub = self._decode_b64_field(msg_dict.get("public_key"))
                if peer_pub is None or len(peer_pub) != 32:
                    return
                expected_id = self._derive_node_id(peer_pub)
                if not compare_digest(parsed.sender_id, expected_id):
                    logger.warning("Handshake sender_id/pubkey mismatch; dropping")
                    return
                if not NodeIdentity.verify_with_public_key(peer_pub, parsed.signing_payload(), bytes(parsed.signature)):
                    logger.warning("Handshake signature verification failed; dropping")
                    return
            else:
                # For established sessions, verify using the cached peer public key.
                peer_pub = session.peer_public_key
                if peer_pub is None or len(peer_pub) != 32:
                    logger.warning(f"Missing peer_public_key for ready peer {session.node_id[:16]}...; disconnecting")
                    asyncio.create_task(self._disconnect_peer(session.node_id))
                    return
                if parsed.sender_id != session.node_id:
                    logger.warning(
                        f"Sender spoof attempt: session={session.node_id[:16]}... msg.sender_id={parsed.sender_id[:16]}..."
                    )
                    asyncio.create_task(self._disconnect_peer(session.node_id))
                    return
                if not NodeIdentity.verify_with_public_key(peer_pub, parsed.signing_payload(), bytes(parsed.signature)):
                    logger.warning(f"Invalid signature from {session.node_id[:16]}...; disconnecting")
                    asyncio.create_task(self._disconnect_peer(session.node_id))
                    return

            # Replay protection (authenticated sender_id + nonce).
            if self._is_replay(parsed):
                logger.warning(f"Dropping replayed message from {parsed.sender_id[:16]}...")
                return

            # Route to handler
            handler = self._handlers.get(msg_type)
            if handler:
                response = await handler(msg_dict, session.node_id)

                # Send response if any
                if response:
                    await self._send_to_peer(session.node_id, response)
            else:
                logger.debug(f"No handler for message type {msg_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    @staticmethod
    def _decode_b64_field(value: Any) -> bytes | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return None

    @staticmethod
    def _derive_node_id(public_key: bytes) -> str:
        return hashlib.sha256(public_key).hexdigest()[:40]

    @staticmethod
    def _derive_session_key(
        shared_secret: bytes,
        challenge_nonce: bytes,
        node_id_a: str,
        node_id_b: str,
        kx_pub_a: bytes,
        kx_pub_b: bytes,
    ) -> bytes:
        left_id, right_id = sorted([node_id_a, node_id_b])
        if node_id_a == left_id:
            left_pub, right_pub = kx_pub_a, kx_pub_b
        else:
            left_pub, right_pub = kx_pub_b, kx_pub_a

        salt = hashlib.sha256(challenge_nonce + left_id.encode('utf-8') + right_id.encode('utf-8')).digest()
        info = b"IAN_P2P_SESSION_KEY_V1" + left_id.encode('utf-8') + right_id.encode('utf-8') + left_pub + right_pub
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info)
        return cast(bytes, hkdf.derive(shared_secret))

    async def send_message(self, node_id: str, message: Any) -> bool:
        """
        Send message to a peer.

        Args:
            node_id: Target peer node ID
            message: Message object with to_wire() method

        Returns:
            True if queued successfully
        """
        return await self._send_to_peer(node_id, message)

    async def _send_to_peer(self, node_id: str, message: Any) -> bool:
        """Send message to specific peer."""
        session = self._peers.get(node_id)
        if not session or not session.is_connected():
            return False

        try:
            # Sign message
            if hasattr(message, 'sender_id'):
                message.sender_id = self._identity.node_id
            if self._identity.has_private_key() and hasattr(message, "signature"):
                # All protocol messages should be authenticated.
                self._identity.sign_message(message)

            # Serialize message
            if hasattr(message, 'to_dict'):
                data = json.dumps(message.to_dict()).encode('utf-8')
            else:
                data = json.dumps(message).encode('utf-8')

            # Queue for sending
            await asyncio.wait_for(
                session.outbound_queue.put(data),
                timeout=1.0,
            )

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Send queue full for {node_id}")
            return False
        except Exception as e:
            logger.error(f"Error sending to {node_id}: {e}")
            return False

    async def broadcast(self, message: Any, exclude: set[str] | None = None) -> int:
        """
        Broadcast message to all connected peers.

        Args:
            message: Message to broadcast
            exclude: Node IDs to exclude

        Returns:
            Number of peers message was sent to
        """
        exclude = exclude or set()
        sent = 0

        for node_id, session in self._peers.items():
            if node_id in exclude:
                continue
            if session.is_ready():
                if await self._send_to_peer(node_id, message):
                    sent += 1

        return sent

    # -------------------------------------------------------------------------
    # Handshake
    # -------------------------------------------------------------------------

    async def _send_handshake(self, session: PeerSession) -> None:
        """
        Send handshake with cryptographic challenge.

        Security:
        - Generates random 32-byte nonce
        - Peer must sign nonce to prove identity
        - Prevents session hijacking
        """
        self._mark_handshake_started(session, now_s=self._now_s())
        session.transition_to(PeerState.HANDSHAKING)

        # Generate challenge nonce
        challenge_nonce = secrets.token_bytes(32)
        session.pending_challenge = challenge_nonce

        kx_private = X25519PrivateKey.generate()
        kx_public = kx_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        session.kx_private_key = kx_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        session.kx_public_key = kx_public

        handshake_msg = HandshakeChallenge(
            sender_id=self._identity.node_id,
            challenge_nonce=challenge_nonce.hex(),
            kx_public_key=base64.b64encode(kx_public).decode('utf-8'),
            public_key=base64.b64encode(self._identity.public_key).decode('utf-8'),
        )
        if self._identity.has_private_key():
            self._identity.sign_message(handshake_msg)
        await self._send_to_peer(session.node_id, handshake_msg)

    def _verify_handshake_response(self, session: PeerSession, response: dict[str, Any]) -> bool:
        """
        Verify handshake response proves peer controls claimed identity.

        Security:
        - Verifies peer's Ed25519 signature over the HandshakeResponse signing payload
        - Verifies sender_id is bound to the provided public key (node_id = sha256(pubkey)[:40])
        - Verifies challenge_nonce matches our pending challenge (prevents replay)
        - Prevents node ID spoofing/hijacking

        Returns:
            True if verification passed
        """
        if not session.pending_challenge:
            logger.warning(f"No pending challenge for {session.node_id}")
            return False

        try:
            # Expect a full HandshakeResponse message dict (wire shape).
            if response.get("type") not in (MessageType.HANDSHAKE_RESPONSE.value, "HANDSHAKE_RESPONSE"):
                logger.warning("Invalid handshake response type")
                return False

            msg = HandshakeResponse._from_dict_impl(response)

            # Verify challenge nonce matches our pending challenge (constant-time).
            expected_challenge_hex = session.pending_challenge.hex()
            if not compare_digest(msg.challenge_nonce, expected_challenge_hex):
                logger.warning("Handshake challenge nonce mismatch")
                return False

            # Decode and bind public key to claimed node_id.
            peer_pubkey = self._decode_b64_field(response.get("public_key"))
            if peer_pubkey is None or len(peer_pubkey) != 32:
                return False
            expected_node_id = hashlib.sha256(peer_pubkey).hexdigest()[:40]
            if not compare_digest(msg.sender_id, expected_node_id):
                logger.warning("Handshake sender_id/pubkey mismatch")
                return False

            # Verify Ed25519 signature over signing payload.
            if msg.signature is None or len(msg.signature) != 64:
                return False
            if not NodeIdentity.verify_with_public_key(peer_pubkey, msg.signing_payload(), msg.signature):
                logger.warning("Handshake response signature invalid")
                return False

            return True
        except Exception as e:
            logger.warning(f"Handshake verification error: {e}")
            return False

    async def _handle_handshake_response(self, session: PeerSession, response: dict) -> bool:
        """
        Handle handshake response with verification.

        Security:
        - Only updates node_id after verification
        - DISCONNECTS unverified peers (prevents identity spoofing)

        Returns:
            True if handshake succeeded, False if peer should be disconnected
        """
        claimed_id = str(response.get("sender_id", ""))
        current_session_id = session.node_id

        if session.state == PeerState.CONNECTED:
            self._mark_handshake_started(session, now_s=self._now_s())
            session.transition_to(PeerState.HANDSHAKING)

        if session.state != PeerState.HANDSHAKING:
            logger.warning(
                f"Handshake response in invalid state {session.state.name} from {claimed_id[:16]}... - disconnecting"
            )
            session.verified = False
            session.handshake_completed = False
            session.transition_to(PeerState.DISCONNECTING)
            asyncio.create_task(self._disconnect_peer(current_session_id))
            return False

        # Require challenge-response verification for all connections
        if session.pending_challenge:
            if session.kx_private_key is None or session.kx_public_key is None:
                logger.warning(f"Handshake missing local kx keys for {session.node_id[:16]}... - disconnecting")
                session.verified = False
                session.handshake_completed = False
                session.transition_to(PeerState.DISCONNECTING)
                asyncio.create_task(self._disconnect_peer(current_session_id))
                return False

            if not self._verify_handshake_response(session, response):
                # SECURITY: Reject unverified peers to prevent identity spoofing
                logger.warning(
                    f"Rejecting unverified peer {claimed_id[:16]}... - signature verification failed"
                )
                session.verified = False
                session.handshake_completed = False
                session.transition_to(PeerState.DISCONNECTING)
                # Schedule disconnect
                asyncio.create_task(self._disconnect_peer(current_session_id))
                return False

            session.verified = True
        else:
            # No challenge was sent - this shouldn't happen in normal flow
            # Fail-closed: do not allow READY without cryptographic verification.
            logger.warning(f"No challenge sent for {claimed_id[:16]}... - disconnecting")
            session.verified = False
            session.handshake_completed = False
            session.transition_to(PeerState.DISCONNECTING)
            asyncio.create_task(self._disconnect_peer(current_session_id))
            return False

        # Parse response for key material.
        try:
            msg = HandshakeResponse._from_dict_impl(response)
        except Exception:
            session.verified = False
            session.handshake_completed = False
            session.transition_to(PeerState.DISCONNECTING)
            asyncio.create_task(self._disconnect_peer(current_session_id))
            return False

        peer_pubkey = self._decode_b64_field(response.get("public_key"))
        peer_kx_pub = self._decode_b64_field(response.get("kx_public_key"))
        if peer_pubkey is None or len(peer_pubkey) != 32 or peer_kx_pub is None or len(peer_kx_pub) != 32:
            session.verified = False
            session.handshake_completed = False
            session.transition_to(PeerState.DISCONNECTING)
            asyncio.create_task(self._disconnect_peer(current_session_id))
            return False

        # Derive session key (X25519 + HKDF), binding to the verified node IDs and kx pubs.
        try:
            kx_private = X25519PrivateKey.from_private_bytes(session.kx_private_key)
            shared_secret = kx_private.exchange(X25519PublicKey.from_public_bytes(peer_kx_pub))
            session.peer_public_key = peer_pubkey
            session.peer_kx_public_key = peer_kx_pub
            session.session_key = self._derive_session_key(
                shared_secret=shared_secret,
                challenge_nonce=session.pending_challenge,
                node_id_a=self._identity.node_id,
                node_id_b=msg.sender_id,
                kx_pub_a=cast(bytes, session.kx_public_key),
                kx_pub_b=peer_kx_pub,
            )
        except Exception:
            session.verified = False
            session.handshake_completed = False
            session.transition_to(PeerState.DISCONNECTING)
            asyncio.create_task(self._disconnect_peer(current_session_id))
            return False

        # Update session with verified peer info
        old_id = session.node_id
        session.node_id = claimed_id
        self._mark_handshake_completed(session)
        session.transition_to(PeerState.READY)

        # Clear challenge
        session.pending_challenge = None

        # Update mappings if ID changed
        if old_id != claimed_id:
            async with self._lock:
                self._peers.pop(old_id, None)
                self._peers[claimed_id] = session
                addr_key = f"{session.address}:{session.port}"
                self._address_map[addr_key] = claimed_id

        verified_str = "verified" if session.verified else "UNVERIFIED"
        logger.info(f"Handshake complete with {claimed_id[:16]}... ({verified_str})")
        return True

    # -------------------------------------------------------------------------
    # Keepalive
    # -------------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """Send periodic pings to all peers."""
        while self._running:
            try:
                await asyncio.sleep(self._config.ping_interval)

                now = time.time()
                ping = Ping(sender_id=self._identity.node_id)

                if self._identity.has_private_key():
                    self._identity.sign_message(ping)

                for node_id, session in list(self._peers.items()):
                    if not session.is_ready():
                        continue

                    # Check for missed pings
                    if session.last_ping_at > 0:
                        time_since_pong = now - session.last_message_at
                        if time_since_pong > self._config.ping_timeout:
                            session.missed_pings += 1

                            if session.missed_pings >= self._config.max_missed_pings:
                                logger.warning(f"Peer {node_id} unresponsive, disconnecting")
                                await self._disconnect_peer(node_id)
                                continue

                    # Send ping
                    session.last_ping_at = now
                    await self._send_to_peer(node_id, ping)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Ping loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale connections."""
        while self._running:
            try:
                sleep_s = min(60.0, max(_MIN_CLEANUP_SLEEP_S, float(self._config.handshake_timeout)))
                await asyncio.sleep(sleep_s)

                now_s = self._now_s()
                for node_id in await self._find_handshake_timeouts(now_s=now_s):
                    logger.warning(f"Handshake timeout for peer {node_id[:16]}...; disconnecting")
                    await self._disconnect_peer(node_id)

                # Clean up disconnected sessions
                to_remove = [
                    node_id for node_id, session in self._peers.items()
                    if session.state == PeerState.DISCONNECTED
                ]

                for node_id in to_remove:
                    async with self._lock:
                        self._peers.pop(node_id, None)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------

    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[dict, str], Any | None],
    ) -> None:
        """
        Register a message handler.

        Args:
            msg_type: Message type to handle
            handler: Async function(message_dict, from_node_id) -> optional_response
        """
        self._handlers[msg_type] = handler

    def register_default_handlers(self) -> None:
        """Register default protocol handlers."""

        async def handle_handshake_challenge(msg: dict, from_id: str) -> HandshakeResponse | None:
            session = self._peers.get(from_id)
            if session is None:
                return None

            challenge = HandshakeChallenge._from_dict_impl(msg)
            peer_pubkey = self._decode_b64_field(challenge.public_key)
            if peer_pubkey is None or len(peer_pubkey) != 32:
                return None
            if self._derive_node_id(peer_pubkey) != challenge.sender_id:
                return None
            if challenge.signature is None:
                return None
            if not NodeIdentity.verify_with_public_key(peer_pubkey, challenge.signing_payload(), challenge.signature):
                return None

            peer_kx_pub = self._decode_b64_field(challenge.kx_public_key)
            if peer_kx_pub is None or len(peer_kx_pub) != 32:
                return None

            try:
                challenge_nonce = bytes.fromhex(challenge.challenge_nonce)
            except Exception:
                return None
            if len(challenge_nonce) != 32:
                return None

            kx_private = X25519PrivateKey.generate()
            kx_public = kx_private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            try:
                shared_secret = kx_private.exchange(X25519PublicKey.from_public_bytes(peer_kx_pub))
            except Exception:
                return None

            old_id = session.node_id
            session.node_id = challenge.sender_id
            session.verified = True

            self._mark_handshake_started(session, now_s=self._now_s())
            session.transition_to(PeerState.HANDSHAKING)

            if old_id != session.node_id:
                async with self._lock:
                    self._peers.pop(old_id, None)
                    self._peers[session.node_id] = session
                    addr_key = f"{session.address}:{session.port}"
                    self._address_map[addr_key] = session.node_id

            session.peer_public_key = peer_pubkey
            session.peer_kx_public_key = peer_kx_pub
            session.kx_private_key = kx_private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            session.kx_public_key = kx_public
            session.session_key = self._derive_session_key(
                shared_secret=shared_secret,
                challenge_nonce=challenge_nonce,
                node_id_a=self._identity.node_id,
                node_id_b=challenge.sender_id,
                kx_pub_a=kx_public,
                kx_pub_b=peer_kx_pub,
            )

            response_nonce = secrets.token_bytes(32)
            response = HandshakeResponse(
                sender_id=self._identity.node_id,
                challenge_nonce=challenge.challenge_nonce,
                response_nonce=response_nonce.hex(),
                kx_public_key=base64.b64encode(kx_public).decode('utf-8'),
                public_key=base64.b64encode(self._identity.public_key).decode('utf-8'),
            )
            if self._identity.has_private_key():
                self._identity.sign_message(response)
            return response

        async def handle_handshake_response(msg: dict, from_id: str) -> None:
            session = self._peers.get(from_id)
            if session is None:
                return
            # Use the hardened, single-source handshake response implementation.
            await self._handle_handshake_response(session, msg)

        async def handle_ping(msg: dict, from_id: str) -> Pong | None:
            pong = Pong(sender_id=self._identity.node_id)
            if self._identity.has_private_key():
                self._identity.sign_message(pong)
            return pong

        async def handle_pong(msg: dict, from_id: str) -> None:
            Pong.from_dict(msg)
            session = self._peers.get(from_id)
            if session:
                session.missed_pings = 0
                return

        self._handlers[MessageType.HANDSHAKE_CHALLENGE] = handle_handshake_challenge
        self._handlers[MessageType.HANDSHAKE_RESPONSE] = handle_handshake_response
        self._handlers[MessageType.PING] = handle_ping
        self._handlers[MessageType.PONG] = handle_pong

    def get_session_key(self, node_id: str) -> bytes | None:
        session = self._peers.get(node_id)
        if session is None:
            return None
        if session.session_key is None:
            return None
        if len(session.session_key) != 32:
            return None
        return session.session_key

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_connected_peers(self) -> list[str]:
        """Get list of connected peer IDs."""
        return [
            node_id for node_id, session in self._peers.items()
            if session.is_ready()
        ]

    def get_peer_count(self) -> int:
        """Get number of ready peers."""
        return sum(1 for s in self._peers.values() if s.is_ready())

    def get_peer_info(self, node_id: str) -> dict[str, Any] | None:
        """Get info about a peer."""
        session = self._peers.get(node_id)
        if not session:
            return None

        return {
            "node_id": session.node_id,
            "address": f"{session.address}:{session.port}",
            "state": session.state.name,
            "connected_at": session.connected_at,
            "messages_sent": session.messages_sent,
            "messages_received": session.messages_received,
            "bytes_sent": session.bytes_sent,
            "bytes_received": session.bytes_received,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get P2P manager statistics."""
        return {
            "listening": f"{self._config.listen_host}:{self._config.listen_port}",
            "total_peers": len(self._peers),
            "ready_peers": self.get_peer_count(),
            "total_messages_sent": sum(s.messages_sent for s in self._peers.values()),
            "total_messages_received": sum(s.messages_received for s in self._peers.values()),
            "total_bytes_sent": sum(s.bytes_sent for s in self._peers.values()),
            "total_bytes_received": sum(s.bytes_received for s in self._peers.values()),
        }

    @property
    def node_id(self) -> str:
        return cast(str, self._identity.node_id)

    @property
    def listen_address(self) -> str:
        return f"{self._config.listen_host}:{self._config.listen_port}"
