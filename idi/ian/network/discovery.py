"""
IAN Peer Discovery - Find and maintain peer connections.

Discovery Methods:
1. Seed nodes - Bootstrap from known nodes
2. Peer exchange - Learn peers from connected nodes
3. DHT (future) - Kademlia-style distributed hash table

Peer Management:
- Track peer health (latency, uptime)
- Automatic reconnection
- Peer scoring and eviction
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .node import NodeInfo, NodeIdentity, verify_message_signature, HAS_CRYPTO
from .protocol import PeerExchange, Ping, Pong


logger = logging.getLogger(__name__)


# =============================================================================
# Peer State
# =============================================================================

@dataclass
class PeerState:
    """State tracking for a peer."""
    info: NodeInfo
    connected: bool = False
    last_seen: float = field(default_factory=time.time)
    last_ping: Optional[float] = None
    latency_ms: Optional[float] = None
    failures: int = 0
    score: float = 1.0
    
    def update_seen(self) -> None:
        self.last_seen = time.time()
        self.failures = 0
    
    def record_failure(self) -> None:
        self.failures += 1
        self.score *= 0.9  # Decay score on failure
    
    def record_success(self, latency_ms: float) -> None:
        self.latency_ms = latency_ms
        self.score = min(1.0, self.score * 1.1)  # Increase score on success
        self.update_seen()
    
    def is_stale(self, max_age_seconds: float = 3600) -> bool:
        """Check if peer info is stale."""
        return time.time() - self.last_seen > max_age_seconds


# =============================================================================
# Discovery Interface
# =============================================================================

class PeerDiscovery(ABC):
    """Abstract base for peer discovery."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start discovery."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop discovery."""
        pass
    
    @abstractmethod
    def get_peers(self) -> List[PeerState]:
        """Get known peers."""
        pass
    
    @abstractmethod
    def add_peer(self, info: NodeInfo) -> None:
        """Add a peer."""
        pass
    
    @abstractmethod
    def remove_peer(self, node_id: str) -> None:
        """Remove a peer."""
        pass


# =============================================================================
# Seed Node Discovery
# =============================================================================

class SeedNodeDiscovery(PeerDiscovery):
    """
    Simple seed-node based discovery.
    
    Bootstrap from a list of known seed nodes, then learn
    more peers via peer exchange.
    """
    
    def __init__(
        self,
        identity: NodeIdentity,
        seed_addresses: List[str],
        max_peers: int = 50,
        exchange_interval: float = 60.0,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize seed node discovery.
        
        Args:
            identity: This node's identity
            seed_addresses: List of seed node addresses
            max_peers: Maximum peers to maintain
            exchange_interval: Seconds between peer exchanges
            health_check_interval: Seconds between health checks
        """
        self._identity = identity
        self._seed_addresses = seed_addresses
        self._max_peers = max_peers
        self._exchange_interval = exchange_interval
        self._health_check_interval = health_check_interval
        
        self._peers: Dict[str, PeerState] = {}
        self._seen_node_ids: Set[str] = set()
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Callbacks
        self._on_peer_discovered: Optional[Callable[[NodeInfo], None]] = None
        self._on_peer_lost: Optional[Callable[[str], None]] = None
        self._send_message: Optional[Callable[[str, Any], asyncio.Future]] = None
    
    def set_callbacks(
        self,
        on_peer_discovered: Optional[Callable[[NodeInfo], None]] = None,
        on_peer_lost: Optional[Callable[[str], None]] = None,
        send_message: Optional[Callable[[str, Any], asyncio.Future]] = None,
    ) -> None:
        """Set discovery callbacks."""
        self._on_peer_discovered = on_peer_discovered
        self._on_peer_lost = on_peer_lost
        self._send_message = send_message
    
    async def start(self) -> None:
        """Start discovery process."""
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting peer discovery with {len(self._seed_addresses)} seeds")
        
        # Bootstrap from seeds
        await self._bootstrap()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._exchange_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]
    
    async def stop(self) -> None:
        """Stop discovery process."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("Peer discovery stopped")
    
    async def _bootstrap(self) -> None:
        """Bootstrap from seed nodes."""
        for address in self._seed_addresses:
            try:
                # For now, create placeholder peer info
                # In production, would connect and fetch actual info
                logger.info(f"Bootstrapping from seed: {address}")
                
                # TODO: Actually connect and fetch node info
                
            except Exception as e:
                logger.warning(f"Failed to bootstrap from {address}: {e}")
    
    async def _exchange_loop(self) -> None:
        """Periodically exchange peers with connected nodes."""
        while self._running:
            try:
                await asyncio.sleep(self._exchange_interval)
                await self._do_peer_exchange()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Peer exchange error: {e}")
    
    async def _do_peer_exchange(self) -> None:
        """Exchange peers with a random connected peer."""
        connected = [p for p in self._peers.values() if p.connected]
        if not connected:
            return
        
        # Pick random peer
        peer = random.choice(connected)
        
        # Send peer exchange
        if self._send_message:
            exchange = PeerExchange(
                sender_id=self._identity.node_id,
                peers=[p.info.to_dict() for p in self._peers.values()],
            )
            if HAS_CRYPTO and self._identity.has_private_key():
                self._identity.sign_message(exchange)
            
            try:
                await self._send_message(peer.info.node_id, exchange)
            except Exception as e:
                logger.debug(f"Peer exchange failed: {e}")
    
    async def _health_check_loop(self) -> None:
        """Periodically check peer health."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_peer_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_peer_health(self) -> None:
        """Check health of connected peers."""
        now = time.time()
        to_remove = []
        
        for node_id, peer in self._peers.items():
            # Remove stale peers
            if peer.is_stale():
                to_remove.append(node_id)
                continue
            
            # Ping connected peers
            if peer.connected and self._send_message:
                ping = Ping(sender_id=self._identity.node_id)
                if HAS_CRYPTO and self._identity.has_private_key():
                    self._identity.sign_message(ping)
                peer.last_ping = now
                
                try:
                    await self._send_message(node_id, ping)
                except Exception:
                    peer.record_failure()
                    
                    if peer.failures > 3:
                        to_remove.append(node_id)
        
        # Remove failed peers
        for node_id in to_remove:
            self.remove_peer(node_id)
    
    def get_peers(self) -> List[PeerState]:
        """Get all known peers."""
        return list(self._peers.values())
    
    def get_connected_peers(self) -> List[PeerState]:
        """Get connected peers."""
        return [p for p in self._peers.values() if p.connected]
    
    def add_peer(self, info: NodeInfo) -> None:
        """Add a discovered peer."""
        if info.node_id == self._identity.node_id:
            return  # Don't add self
        
        if info.node_id in self._peers:
            # Update existing peer
            self._peers[info.node_id].info = info
            self._peers[info.node_id].update_seen()
        else:
            # Add new peer
            if len(self._peers) >= self._max_peers:
                self._evict_peer()
            
            self._peers[info.node_id] = PeerState(info=info)
            self._seen_node_ids.add(info.node_id)
            
            logger.info(f"Discovered peer: {info.node_id[:16]}...")
            
            if self._on_peer_discovered:
                self._on_peer_discovered(info)
    
    def remove_peer(self, node_id: str) -> None:
        """Remove a peer."""
        if node_id in self._peers:
            del self._peers[node_id]
            
            logger.info(f"Removed peer: {node_id[:16]}...")
            
            if self._on_peer_lost:
                self._on_peer_lost(node_id)
    
    def _evict_peer(self) -> None:
        """Evict the lowest-scored peer."""
        if not self._peers:
            return
        
        worst_id = min(self._peers.keys(), key=lambda x: self._peers[x].score)
        self.remove_peer(worst_id)
    
    def handle_peer_exchange(self, exchange: PeerExchange) -> None:
        """Handle incoming peer exchange."""
        sender = self._peers.get(exchange.sender_id)
        if HAS_CRYPTO:
            if sender is None:
                logger.debug(f"Peer exchange from unknown sender {exchange.sender_id[:16]}...; ignoring")
                return
            if exchange.signature is None:
                logger.debug(f"Unsigned peer exchange from {exchange.sender_id[:16]}...; ignoring")
                return
            if not verify_message_signature(sender.info, exchange):
                logger.debug(f"Invalid signature on peer exchange from {exchange.sender_id[:16]}...; ignoring")
                return

        for peer_dict in exchange.peers:
            try:
                info = NodeInfo.from_dict(peer_dict)
                
                # Don't add ourselves or duplicates
                if info.node_id != self._identity.node_id:
                    self.add_peer(info)
            except Exception as e:
                logger.debug(f"Invalid peer in exchange: {e}")
    
    def handle_pong(self, pong: Pong) -> None:
        """Handle pong response."""
        peer = self._peers.get(pong.sender_id)
        if not peer:
            return
        if HAS_CRYPTO:
            if pong.signature is None:
                logger.debug(f"Unsigned pong from {pong.sender_id[:16]}...; ignoring")
                return
            if not verify_message_signature(peer.info, pong):
                logger.debug(f"Invalid pong signature from {pong.sender_id[:16]}...; ignoring")
                return
        if peer.last_ping:
            latency = (time.time() - peer.last_ping) * 1000
            peer.record_success(latency)
    
    def get_random_peers(self, count: int = 3) -> List[NodeInfo]:
        """Get random peers for gossip."""
        connected = [p for p in self._peers.values() if p.connected]
        if not connected:
            return []
        
        count = min(count, len(connected))
        return [p.info for p in random.sample(connected, count)]
