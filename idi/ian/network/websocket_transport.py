"""
IAN WebSocket Transport - WebSocket-based transport for web clients.

Provides:
1. WebSocket server for browser clients
2. WebSocket client for connecting to other nodes
3. JSON message framing compatible with P2P protocol
4. TLS support for secure connections

Features:
- Browser-compatible (no TCP sockets needed)
- Automatic reconnection
- Heartbeat/keepalive
- Message compression (optional)

Integration:
- Uses same message protocol as TCP transport
- Can be used alongside TCP transport
- Suitable for REST API and real-time updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

try:
    import aiohttp
    from aiohttp import web, WSMsgType
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .protocol import Message, MessageType
from .node import NodeIdentity

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WebSocketConfig:
    """WebSocket transport configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 9001
    path: str = "/ws"
    
    # TLS (optional)
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    
    # Limits
    max_connections: int = 1000
    max_message_size: int = 16 * 1024 * 1024  # 16 MB
    
    # Timeouts
    heartbeat_interval: float = 30.0
    receive_timeout: float = 60.0
    close_timeout: float = 5.0
    
    # Reconnection (client mode)
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 10
    
    # Compression
    compress: bool = True


# =============================================================================
# Client Connection
# =============================================================================

@dataclass
class WSClientConnection:
    """Represents a connected WebSocket client."""
    id: str  # Connection ID
    ws: Any  # WebSocket object (aiohttp.WebSocketResponse)
    node_id: Optional[str] = None  # Node ID after handshake
    connected_at: float = field(default_factory=time.time)
    last_message_at: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0
    subscriptions: Set[str] = field(default_factory=set)  # Subscribed topics
    
    def is_authenticated(self) -> bool:
        return self.node_id is not None


# =============================================================================
# WebSocket Server
# =============================================================================

class WebSocketServer:
    """
    WebSocket server for IAN.
    
    Provides real-time communication with web clients.
    Supports:
    - P2P message protocol over WebSocket
    - Event subscriptions (leaderboard updates, new contributions)
    - API streaming (log entries, state changes)
    """
    
    def __init__(
        self,
        identity: NodeIdentity,
        config: Optional[WebSocketConfig] = None,
    ):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for WebSocket transport")
        
        self._identity = identity
        self._config = config or WebSocketConfig()
        
        # Connections
        self._connections: Dict[str, WSClientConnection] = {}
        self._connection_counter = 0
        
        # Server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        
        # Message handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Event subscriptions
        self._topic_subscribers: Dict[str, Set[str]] = {}  # topic -> {connection_ids}
        
        # Running state
        self._running = False
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    async def start(self) -> bool:
        """Start WebSocket server."""
        if self._running:
            return True
        
        try:
            self._app = web.Application()
            self._app.router.add_get(self._config.path, self._handle_websocket)
            self._app.router.add_get("/health", self._handle_health)
            
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            
            # TLS setup if configured
            ssl_context = None
            if self._config.ssl_cert and self._config.ssl_key:
                import ssl
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(self._config.ssl_cert, self._config.ssl_key)
            
            self._site = web.TCPSite(
                self._runner,
                self._config.host,
                self._config.port,
                ssl_context=ssl_context,
            )
            await self._site.start()
            
            self._running = True
            
            protocol = "wss" if ssl_context else "ws"
            logger.info(
                f"WebSocket server started at "
                f"{protocol}://{self._config.host}:{self._config.port}{self._config.path}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop WebSocket server."""
        self._running = False
        
        # Close all connections
        for conn in list(self._connections.values()):
            try:
                await conn.ws.close()
            except Exception:
                pass
        
        self._connections.clear()
        
        # Stop server
        if self._site:
            await self._site.stop()
        
        if self._runner:
            await self._runner.cleanup()
        
        logger.info("WebSocket server stopped")
    
    # -------------------------------------------------------------------------
    # Connection Handling
    # -------------------------------------------------------------------------
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle new WebSocket connection."""
        # Check connection limit
        if len(self._connections) >= self._config.max_connections:
            return web.Response(status=503, text="Too many connections")
        
        ws = web.WebSocketResponse(
            heartbeat=self._config.heartbeat_interval,
            max_msg_size=self._config.max_message_size,
            compress=self._config.compress,
        )
        await ws.prepare(request)
        
        # Create connection
        self._connection_counter += 1
        conn_id = f"ws_{self._connection_counter}"
        
        conn = WSClientConnection(id=conn_id, ws=ws)
        self._connections[conn_id] = conn
        
        logger.debug(f"WebSocket client connected: {conn_id}")
        
        try:
            # Send welcome message
            await self._send_to_connection(conn, {
                "type": "welcome",
                "node_id": self._identity.node_id,
                "connection_id": conn_id,
            })
            
            # Message loop
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    conn.last_message_at = time.time()
                    conn.messages_received += 1
                    
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(conn, data)
                    except json.JSONDecodeError:
                        await self._send_error(conn, "Invalid JSON")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        await self._send_error(conn, str(e))
                
                elif msg.type == WSMsgType.BINARY:
                    # Binary messages not supported
                    await self._send_error(conn, "Binary messages not supported")
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                    
        except asyncio.CancelledError:
            pass
        finally:
            # Cleanup
            self._remove_connection(conn_id)
            logger.debug(f"WebSocket client disconnected: {conn_id}")
        
        return ws
    
    def _remove_connection(self, conn_id: str) -> None:
        """Remove connection and cleanup subscriptions."""
        conn = self._connections.pop(conn_id, None)
        if not conn:
            return
        
        # Remove from all subscriptions
        for topic, subscribers in self._topic_subscribers.items():
            subscribers.discard(conn_id)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "connections": len(self._connections),
            "node_id": self._identity.node_id,
        })
    
    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------
    
    async def _handle_message(self, conn: WSClientConnection, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")
        
        # Built-in handlers
        if msg_type == "ping":
            await self._send_to_connection(conn, {"type": "pong"})
            return
        
        if msg_type == "subscribe":
            await self._handle_subscribe(conn, data)
            return
        
        if msg_type == "unsubscribe":
            await self._handle_unsubscribe(conn, data)
            return
        
        if msg_type == "authenticate":
            await self._handle_authenticate(conn, data)
            return
        
        # Custom handlers
        handler = self._handlers.get(msg_type)
        if handler:
            try:
                response = await handler(data, conn)
                if response:
                    await self._send_to_connection(conn, response)
            except Exception as e:
                await self._send_error(conn, str(e))
        else:
            await self._send_error(conn, f"Unknown message type: {msg_type}")
    
    async def _handle_subscribe(self, conn: WSClientConnection, data: Dict[str, Any]) -> None:
        """Handle subscription request."""
        topics = data.get("topics", [])
        
        for topic in topics:
            if topic not in self._topic_subscribers:
                self._topic_subscribers[topic] = set()
            self._topic_subscribers[topic].add(conn.id)
            conn.subscriptions.add(topic)
        
        await self._send_to_connection(conn, {
            "type": "subscribed",
            "topics": list(conn.subscriptions),
        })
    
    async def _handle_unsubscribe(self, conn: WSClientConnection, data: Dict[str, Any]) -> None:
        """Handle unsubscription request."""
        topics = data.get("topics", [])
        
        for topic in topics:
            if topic in self._topic_subscribers:
                self._topic_subscribers[topic].discard(conn.id)
            conn.subscriptions.discard(topic)
        
        await self._send_to_connection(conn, {
            "type": "unsubscribed",
            "topics": topics,
        })
    
    async def _handle_authenticate(self, conn: WSClientConnection, data: Dict[str, Any]) -> None:
        """Handle authentication."""
        node_id = data.get("node_id")
        signature = data.get("signature")
        
        # In a full implementation, would verify signature
        if node_id:
            conn.node_id = node_id
            await self._send_to_connection(conn, {
                "type": "authenticated",
                "node_id": node_id,
            })
        else:
            await self._send_error(conn, "Invalid authentication")
    
    # -------------------------------------------------------------------------
    # Sending Messages
    # -------------------------------------------------------------------------
    
    async def _send_to_connection(self, conn: WSClientConnection, data: Dict[str, Any]) -> bool:
        """Send message to specific connection."""
        try:
            await conn.ws.send_json(data)
            conn.messages_sent += 1
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {conn.id}: {e}")
            return False
    
    async def _send_error(self, conn: WSClientConnection, error: str) -> None:
        """Send error message."""
        await self._send_to_connection(conn, {
            "type": "error",
            "error": error,
        })
    
    async def broadcast(self, data: Dict[str, Any]) -> int:
        """Broadcast message to all connections."""
        sent = 0
        for conn in list(self._connections.values()):
            if await self._send_to_connection(conn, data):
                sent += 1
        return sent
    
    async def publish(self, topic: str, data: Dict[str, Any]) -> int:
        """Publish message to topic subscribers."""
        subscribers = self._topic_subscribers.get(topic, set())
        sent = 0
        
        message = {
            "type": "event",
            "topic": topic,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        
        for conn_id in list(subscribers):
            conn = self._connections.get(conn_id)
            if conn and await self._send_to_connection(conn, message):
                sent += 1
        
        return sent
    
    async def send_to_node(self, node_id: str, data: Dict[str, Any]) -> bool:
        """Send message to specific node (by node_id, not connection_id)."""
        for conn in self._connections.values():
            if conn.node_id == node_id:
                return await self._send_to_connection(conn, data)
        return False
    
    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------
    
    def register_handler(
        self,
        msg_type: str,
        handler: Callable[[Dict, WSClientConnection], Optional[Dict]],
    ) -> None:
        """Register message handler."""
        self._handlers[msg_type] = handler
    
    # -------------------------------------------------------------------------
    # Event Publishing
    # -------------------------------------------------------------------------
    
    async def publish_contribution(
        self,
        goal_id: str,
        pack_hash: str,
        contributor_id: str,
        score: float,
        log_index: int,
    ) -> int:
        """Publish new contribution event."""
        return await self.publish(f"contributions:{goal_id}", {
            "goal_id": goal_id,
            "pack_hash": pack_hash,
            "contributor_id": contributor_id,
            "score": score,
            "log_index": log_index,
        })
    
    async def publish_leaderboard_update(
        self,
        goal_id: str,
        leaderboard: List[Dict[str, Any]],
    ) -> int:
        """Publish leaderboard update event."""
        return await self.publish(f"leaderboard:{goal_id}", {
            "goal_id": goal_id,
            "leaderboard": leaderboard,
        })
    
    async def publish_upgrade(
        self,
        goal_id: str,
        pack_hash: str,
        score: float,
    ) -> int:
        """Publish policy upgrade event."""
        return await self.publish(f"upgrades:{goal_id}", {
            "goal_id": goal_id,
            "pack_hash": pack_hash,
            "score": score,
        })
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_connection_count(self) -> int:
        """Get number of connected clients."""
        return len(self._connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "connections": len(self._connections),
            "authenticated": sum(1 for c in self._connections.values() if c.is_authenticated()),
            "topics": len(self._topic_subscribers),
            "total_subscriptions": sum(len(s) for s in self._topic_subscribers.values()),
            "total_messages_sent": sum(c.messages_sent for c in self._connections.values()),
            "total_messages_received": sum(c.messages_received for c in self._connections.values()),
        }


# =============================================================================
# WebSocket Client
# =============================================================================

class WebSocketClient:
    """
    WebSocket client for connecting to IAN nodes.
    
    Used by:
    - Web applications
    - Monitoring tools
    - Light clients
    """
    
    def __init__(
        self,
        url: str,
        identity: Optional[NodeIdentity] = None,
        config: Optional[WebSocketConfig] = None,
    ):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for WebSocket transport")
        
        self._url = url
        self._identity = identity
        self._config = config or WebSocketConfig()
        
        # Connection
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connected = False
        self._reconnect_attempts = 0
        
        # Message handling
        self._handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_counter = 0
        
        # Background tasks
        self._read_task: Optional[asyncio.Task] = None
        self._running = False
    
    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        if self._connected:
            return True
        
        try:
            self._session = aiohttp.ClientSession()
            
            self._ws = await self._session.ws_connect(
                self._url,
                heartbeat=self._config.heartbeat_interval,
                max_msg_size=self._config.max_message_size,
                compress=self._config.compress,
            )
            
            self._connected = True
            self._reconnect_attempts = 0
            self._running = True
            
            # Start read loop
            self._read_task = asyncio.create_task(self._read_loop())
            
            # Authenticate if we have identity
            if self._identity:
                await self.authenticate()
            
            logger.info(f"WebSocket connected to {self._url}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._cleanup()
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._running = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        await self._cleanup()
        logger.info("WebSocket disconnected")
    
    async def _cleanup(self) -> None:
        """Clean up connection resources."""
        self._connected = False
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        if not self._config.auto_reconnect:
            return
        
        while (
            self._running and
            self._reconnect_attempts < self._config.max_reconnect_attempts
        ):
            self._reconnect_attempts += 1
            delay = self._config.reconnect_delay * self._reconnect_attempts
            
            logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
            await asyncio.sleep(delay)
            
            if await self.connect():
                return
        
        logger.error("Max reconnect attempts reached")
    
    # -------------------------------------------------------------------------
    # Message Loop
    # -------------------------------------------------------------------------
    
    async def _read_loop(self) -> None:
        """Read messages from server."""
        try:
            async for msg in self._ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
                
                elif msg.type == WSMsgType.CLOSED:
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Read loop error: {e}")
        
        # Reconnect if needed
        self._connected = False
        if self._running:
            await self._reconnect()
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming message."""
        msg_type = data.get("type", "")
        
        # Check for response to pending request
        request_id = data.get("request_id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data)
            return
        
        # Call registered handler
        handler = self._handlers.get(msg_type)
        if handler:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")
    
    # -------------------------------------------------------------------------
    # Sending
    # -------------------------------------------------------------------------
    
    async def send(self, data: Dict[str, Any]) -> bool:
        """Send message to server."""
        if not self._connected or not self._ws:
            return False
        
        try:
            await self._ws.send_json(data)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    async def request(
        self,
        data: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send request and wait for response."""
        self._request_counter += 1
        request_id = f"req_{self._request_counter}"
        
        data["request_id"] = request_id
        
        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            if not await self.send(data):
                return None
            
            return await asyncio.wait_for(future, timeout=timeout)
            
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            logger.warning(f"Request timeout: {request_id}")
            return None
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    async def authenticate(self) -> bool:
        """Authenticate with server."""
        if not self._identity:
            return False
        
        response = await self.request({
            "type": "authenticate",
            "node_id": self._identity.node_id,
            # Would include signature in production
        })
        
        return response and response.get("type") == "authenticated"
    
    async def subscribe(self, topics: List[str]) -> bool:
        """Subscribe to topics."""
        response = await self.request({
            "type": "subscribe",
            "topics": topics,
        })
        return response and response.get("type") == "subscribed"
    
    async def unsubscribe(self, topics: List[str]) -> bool:
        """Unsubscribe from topics."""
        response = await self.request({
            "type": "unsubscribe",
            "topics": topics,
        })
        return response and response.get("type") == "unsubscribed"
    
    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------
    
    def on(self, msg_type: str, handler: Callable[[Dict], None]) -> None:
        """Register message handler."""
        self._handlers[msg_type] = handler
    
    def on_event(self, handler: Callable[[str, Dict], None]) -> None:
        """Register event handler."""
        async def wrapper(data: Dict):
            topic = data.get("topic", "")
            event_data = data.get("data", {})
            await handler(topic, event_data)
        
        self._handlers["event"] = wrapper
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def is_connected(self) -> bool:
        return self._connected
