"""
IAN Transport Layer - Network transport abstraction.

Provides:
1. Abstract transport interface
2. TCP transport implementation
3. Connection management
4. Message framing

Future extensions:
- WebSocket transport
- QUIC transport
- Tor/I2P transport
"""

from __future__ import annotations

import asyncio
import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .protocol import Message


logger = logging.getLogger(__name__)


# =============================================================================
# Transport Interface
# =============================================================================

class Transport(ABC):
    """Abstract transport interface."""
    
    @abstractmethod
    async def start(self, host: str, port: int) -> None:
        """Start listening for connections."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop transport."""
        pass
    
    @abstractmethod
    async def connect(self, address: str) -> str:
        """Connect to peer, return connection ID."""
        pass
    
    @abstractmethod
    async def disconnect(self, conn_id: str) -> None:
        """Disconnect from peer."""
        pass
    
    @abstractmethod
    async def send(self, conn_id: str, message: Message) -> None:
        """Send message to peer."""
        pass
    
    @abstractmethod
    def set_message_handler(self, handler: Callable[[str, Message], None]) -> None:
        """Set handler for incoming messages."""
        pass


# =============================================================================
# TCP Transport
# =============================================================================

@dataclass
class TCPConnection:
    """TCP connection state."""
    conn_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    address: str
    connected: bool = True


class TCPTransport(Transport):
    """
    TCP-based transport.
    
    Wire format:
    - 4-byte big-endian length prefix
    - JSON-encoded message body
    """
    
    MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MB
    
    def __init__(
        self,
        max_connections: int = 1024,
        max_connections_per_ip: int = 64,
    ) -> None:
        self._server: Optional[asyncio.Server] = None
        self._connections: Dict[str, TCPConnection] = {}
        self._address_to_conn: Dict[str, str] = {}
        self._message_handler: Optional[Callable[[str, Message], None]] = None
        self._conn_counter = 0
        self._running = False
        self._tasks: list = []
        self._max_connections = max_connections
        self._max_connections_per_ip = max_connections_per_ip
        self._ip_connection_counts: Dict[str, int] = {}
    
    async def start(self, host: str, port: int) -> None:
        """Start TCP server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            host,
            port,
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"TCP transport listening on {addr[0]}:{addr[1]}")
    
    async def stop(self) -> None:
        """Stop TCP server and close connections."""
        self._running = False
        
        # Close all connections
        for conn_id in list(self._connections.keys()):
            await self.disconnect(conn_id)
        
        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("TCP transport stopped")
    
    async def connect(self, address: str) -> str:
        """Connect to peer at address (host:port)."""
        # Check if already connected
        if address in self._address_to_conn:
            return self._address_to_conn[address]
        
        if len(self._connections) >= self._max_connections:
            raise ConnectionError("Too many connections")
        
        # Parse address
        if "://" in address:
            # tcp://host:port
            address = address.split("://")[1]
        
        host, port = address.rsplit(":", 1)
        port = int(port)
        ip = host
        current_for_ip = self._ip_connection_counts.get(ip, 0)
        if current_for_ip >= self._max_connections_per_ip:
            raise ConnectionError(f"Too many connections for {ip}")
        
        # Connect
        reader, writer = await asyncio.open_connection(host, port)
        
        # Create connection
        self._conn_counter += 1
        conn_id = f"conn_{self._conn_counter}"
        
        conn = TCPConnection(
            conn_id=conn_id,
            reader=reader,
            writer=writer,
            address=address,
        )
        
        self._connections[conn_id] = conn
        self._address_to_conn[address] = conn_id
        self._ip_connection_counts[ip] = current_for_ip + 1
        
        # Start reader task
        task = asyncio.create_task(self._read_loop(conn))
        self._tasks.append(task)
        
        logger.info(f"Connected to {address}")
        return conn_id
    
    async def disconnect(self, conn_id: str) -> None:
        """Disconnect from peer."""
        if conn_id not in self._connections:
            return
        
        conn = self._connections[conn_id]
        conn.connected = False
        # Handle IPv6 addresses like [::1]:8080
        address = conn.address
        if address.startswith("["):
            # IPv6: [host]:port
            ip = address.split("]:")[0] + "]"
        else:
            # IPv4: host:port
            ip = address.rsplit(":", 1)[0]
        
        try:
            conn.writer.close()
            await conn.writer.wait_closed()
        except Exception:
            pass
        
        # Clean up
        if conn.address in self._address_to_conn:
            del self._address_to_conn[conn.address]
        del self._connections[conn_id]
        current_for_ip = self._ip_connection_counts.get(ip)
        if current_for_ip is not None:
            if current_for_ip <= 1:
                self._ip_connection_counts.pop(ip, None)
            else:
                self._ip_connection_counts[ip] = current_for_ip - 1
        
        logger.info(f"Disconnected from {conn.address}")
    
    async def send(self, conn_id: str, message: Message) -> None:
        """Send message to peer."""
        if conn_id not in self._connections:
            raise ConnectionError(f"Not connected: {conn_id}")
        
        conn = self._connections[conn_id]
        if not conn.connected:
            raise ConnectionError(f"Connection closed: {conn_id}")
        
        # Serialize message
        data = message.to_wire()
        
        # Send
        conn.writer.write(data)
        await conn.writer.drain()
    
    def set_message_handler(self, handler: Callable[[str, Message], None]) -> None:
        """Set handler for incoming messages."""
        self._message_handler = handler
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming connection."""
        addr = writer.get_extra_info('peername')
        address = f"{addr[0]}:{addr[1]}"
        ip = addr[0]
        
        if len(self._connections) >= self._max_connections:
            logger.warning("Max connections reached; rejecting connection from %s", address)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return
        
        current_for_ip = self._ip_connection_counts.get(ip, 0)
        if current_for_ip >= self._max_connections_per_ip:
            logger.warning("Per-IP connection limit reached for %s; rejecting", address)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return
        
        self._conn_counter += 1
        conn_id = f"conn_{self._conn_counter}"
        
        conn = TCPConnection(
            conn_id=conn_id,
            reader=reader,
            writer=writer,
            address=address,
        )
        
        self._connections[conn_id] = conn
        self._address_to_conn[address] = conn_id
        
        logger.info(f"Accepted connection from {address}")
        
        await self._read_loop(conn)
    
    async def _read_loop(self, conn: TCPConnection) -> None:
        """Read messages from connection."""
        # Timeout for reading the header/body to prevent slow-loris attacks
        READ_TIMEOUT = 60.0 
        
        try:
            while conn.connected and self._running:
                # Read length prefix with timeout
                try:
                    length_bytes = await asyncio.wait_for(
                        conn.reader.readexactly(4),
                        timeout=READ_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout (header) from {conn.address}")
                    break
                
                length = struct.unpack(">I", length_bytes)[0]
                
                if length > self.MAX_MESSAGE_SIZE:
                    logger.warning(f"Message too large: {length}")
                    break
                
                # Read message body with timeout
                try:
                    data = await asyncio.wait_for(
                        conn.reader.readexactly(length),
                        timeout=READ_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout (body) from {conn.address}")
                    break
                
                # Parse message
                try:
                    # Reconstruct wire format for parsing
                    wire_data = length_bytes + data
                    message = Message.from_wire(wire_data)
                    
                    # Dispatch to handler
                    if self._message_handler:
                        self._message_handler(conn.conn_id, message)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
                    
        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by peer: {conn.address}")
        except Exception as e:
            logger.warning(f"Connection error: {e}")
        finally:
            conn.connected = False
            await self.disconnect(conn.conn_id)
    
    def get_connection_by_address(self, address: str) -> Optional[str]:
        """Get connection ID by address."""
        return self._address_to_conn.get(address)
    
    def get_connected_addresses(self) -> list:
        """Get list of connected addresses."""
        return list(self._address_to_conn.keys())
