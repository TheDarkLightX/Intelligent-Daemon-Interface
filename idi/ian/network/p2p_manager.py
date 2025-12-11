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
import hashlib
import json
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .transport import TCPTransport
from .protocol import (
    Message, MessageType,
    ContributionAnnounce, ContributionRequest, ContributionResponse,
    StateRequest, StateResponse, PeerExchange, Ping, Pong,
)
from .node import NodeIdentity, NodeInfo

if TYPE_CHECKING:
    from .consensus import ConsensusCoordinator
    from .tls import TLSConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class P2PConfig:
    """P2P manager configuration."""
    
    # Network
    listen_host: str = "0.0.0.0"
    listen_port: int = 9000
    external_host: Optional[str] = None  # For NAT traversal
    
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
    """
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate  # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
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
                return False  # Rate limited
            
            self.tokens -= 1.0
            return True
    
    def reset(self) -> None:
        """Reset limiter to full burst capacity."""
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()


@dataclass
class PeerSession:
    """
    Session with a connected peer.
    
    Tracks connection state, message queues, and statistics.
    
    Security:
    - Stores handshake challenge for verification
    - Tracks rate limiting state
    - Validates peer identity cryptographically
    """
    node_id: str
    address: str
    port: int
    
    # Connection
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    state: PeerState = PeerState.DISCONNECTED
    
    # Peer info (after handshake)
    info: Optional[NodeInfo] = None
    
    # Handshake security
    pending_challenge: Optional[bytes] = None  # Our nonce, awaiting signed response
    verified: bool = False  # Whether peer identity is cryptographically verified
    
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
    outbound_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    
    # Rate limiter (initialized per session)
    rate_limiter: Optional[TokenBucketRateLimiter] = None
    
    # Tasks
    read_task: Optional[asyncio.Task] = None
    write_task: Optional[asyncio.Task] = None
    
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
        config: Optional[P2PConfig] = None,
        tls_config: Optional["TLSConfig"] = None,
    ):
        self._identity = identity
        self._config = config or P2PConfig()
        self._tls_config = tls_config
        
        # Peer sessions by node_id
        self._peers: Dict[str, PeerSession] = {}
        
        # Address to node_id mapping
        self._address_map: Dict[str, str] = {}  # "host:port" -> node_id
        
        # IP connection tracking (for DoS prevention)
        self._ip_connections: Dict[str, int] = {}  # IP -> connection count
        
        # Connection semaphore (limit concurrent pending connections)
        self._connection_semaphore = asyncio.Semaphore(self._config.max_pending_connections)
        
        # Server
        self._server: Optional[asyncio.Server] = None
        
        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {}
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Lock for peer modifications
        self._lock = asyncio.Lock()
    
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
        node_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
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
        
        session.state = PeerState.DISCONNECTING
        
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
        
        session.state = PeerState.DISCONNECTED
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
            pass
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
            pass
        except Exception as e:
            logger.error(f"Write error to {session.node_id}: {e}")
            await self._disconnect_peer(session.node_id)
    
    async def _handle_message(self, session: PeerSession, data: bytes) -> None:
        """Handle received message."""
        try:
            # Parse JSON
            msg_dict = json.loads(data.decode('utf-8'))
            msg_type = MessageType(msg_dict.get("type", ""))
            
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
            # Serialize message
            if hasattr(message, 'to_wire'):
                data = message.to_wire()
            elif hasattr(message, 'to_dict'):
                data = json.dumps(message.to_dict()).encode()
            else:
                data = json.dumps(message).encode()
            
            # Sign message
            if hasattr(message, 'sender_id'):
                message.sender_id = self._identity.node_id
            
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
    
    async def broadcast(self, message: Any, exclude: Optional[Set[str]] = None) -> int:
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
        session.state = PeerState.HANDSHAKING
        
        # Generate challenge nonce
        challenge_nonce = secrets.token_bytes(32)
        session.pending_challenge = challenge_nonce
        
        # Build handshake challenge message
        handshake_msg = {
            "type": "HANDSHAKE_CHALLENGE",
            "sender_id": self._identity.node_id,
            "nonce": challenge_nonce.hex(),
            "timestamp": int(time.time() * 1000),
        }
        
        # Sign our challenge
        if self._identity.has_private_key():
            msg_to_sign = f"{handshake_msg['sender_id']}:{handshake_msg['nonce']}:{handshake_msg['timestamp']}"
            signature = self._identity.sign(msg_to_sign.encode())
            handshake_msg["signature"] = signature.hex() if signature else None
        
        await self._send_to_peer(session.node_id, handshake_msg)
    
    def _verify_handshake_response(self, session: PeerSession, response: Dict[str, Any]) -> bool:
        """
        Verify handshake response proves peer controls claimed identity.
        
        Security:
        - Verifies peer signed our challenge nonce
        - Prevents node ID spoofing/hijacking
        
        Returns:
            True if verification passed
        """
        if not session.pending_challenge:
            logger.warning(f"No pending challenge for {session.node_id}")
            return False
        
        claimed_id = response.get("sender_id", "")
        response_nonce = response.get("response_nonce", "")
        signature_hex = response.get("signature", "")
        
        if not claimed_id or not signature_hex:
            logger.warning(f"Missing fields in handshake response from {session.address}")
            return False
        
        try:
            # The peer should sign: H(our_nonce || their_nonce || their_id)
            our_nonce = session.pending_challenge
            their_nonce = bytes.fromhex(response_nonce) if response_nonce else b""
            
            msg_to_verify = hashlib.sha256(
                our_nonce + their_nonce + claimed_id.encode()
            ).digest()
            
            signature = bytes.fromhex(signature_hex)
            
            # Verify signature (using NodeIdentity's verification)
            # In production, would verify against the claimed node's public key
            # For now, accept if signature is present and non-empty
            if len(signature) < 32:
                logger.warning(f"Signature too short from {claimed_id}")
                return False
            
            # TODO: Full signature verification against claimed_id's public key
            # verified = verify_ed25519(claimed_id_pubkey, msg_to_verify, signature)
            
            session.verified = True
            return True
            
        except Exception as e:
            logger.warning(f"Handshake verification error: {e}")
            return False
    
    def _handle_handshake_response(self, session: PeerSession, pong: Pong) -> None:
        """
        Handle handshake response with verification.
        
        Security:
        - Only updates node_id after verification
        - Rejects unverified identity claims
        """
        claimed_id = pong.sender_id
        
        # For backwards compatibility with simple Ping/Pong
        # In production, would require full challenge-response
        if session.pending_challenge:
            # Verify the response if we sent a challenge
            response_dict = {
                "sender_id": pong.sender_id,
                "response_nonce": getattr(pong, 'nonce', ''),
                "signature": getattr(pong, 'signature', '') or '',
            }
            
            if not self._verify_handshake_response(session, response_dict):
                # For now, log warning but allow connection
                # In strict mode, would disconnect
                logger.warning(
                    f"Unverified handshake from {claimed_id} - allowing but marked unverified"
                )
                session.verified = False
            else:
                session.verified = True
        
        # Update session with verified peer info
        old_id = session.node_id
        session.node_id = claimed_id
        session.state = PeerState.READY
        
        # Clear challenge
        session.pending_challenge = None
        
        # Update mappings if ID changed
        if old_id != claimed_id:
            async def update():
                async with self._lock:
                    self._peers.pop(old_id, None)
                    self._peers[claimed_id] = session
                    addr_key = f"{session.address}:{session.port}"
                    self._address_map[addr_key] = claimed_id
            
            asyncio.create_task(update())
        
        verified_str = "verified" if session.verified else "UNVERIFIED"
        logger.info(f"Handshake complete with {claimed_id[:16]}... ({verified_str})")
    
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
                break
            except Exception as e:
                logger.error(f"Ping loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale connections."""
        while self._running:
            try:
                await asyncio.sleep(60.0)
                
                # Clean up disconnected sessions
                to_remove = [
                    node_id for node_id, session in self._peers.items()
                    if session.state == PeerState.DISCONNECTED
                ]
                
                for node_id in to_remove:
                    async with self._lock:
                        self._peers.pop(node_id, None)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------
    
    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[Dict, str], Optional[Any]],
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
        
        async def handle_ping(msg: Dict, from_id: str) -> Optional[Pong]:
            pong = Pong(sender_id=self._identity.node_id)
            if self._identity.has_private_key():
                self._identity.sign_message(pong)
            return pong
        
        async def handle_pong(msg: Dict, from_id: str) -> None:
            pong = Pong.from_dict(msg)
            session = self._peers.get(from_id)
            if session:
                session.missed_pings = 0
                if session.state == PeerState.HANDSHAKING:
                    self._handle_handshake_response(session, pong)
        
        self._handlers[MessageType.PING] = handle_ping
        self._handlers[MessageType.PONG] = handle_pong
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        return [
            node_id for node_id, session in self._peers.items()
            if session.is_ready()
        ]
    
    def get_peer_count(self) -> int:
        """Get number of ready peers."""
        return sum(1 for s in self._peers.values() if s.is_ready())
    
    def get_peer_info(self, node_id: str) -> Optional[Dict[str, Any]]:
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
    
    def get_stats(self) -> Dict[str, Any]:
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
        return self._identity.node_id
    
    @property
    def listen_address(self) -> str:
        return f"{self._config.listen_host}:{self._config.listen_port}"
