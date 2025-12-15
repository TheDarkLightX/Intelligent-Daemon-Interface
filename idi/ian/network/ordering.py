"""
IAN Contribution Ordering - Deterministic ordering for multi-node consensus.

Provides:
1. Deterministic contribution ordering (timestamp + hash)
2. Contribution mempool with ordering guarantees
3. Gossip protocol integration for contribution propagation
4. Ordering proofs for fraud detection

Design Principles:
- Same inputs → same order on all honest nodes
- Bounded mempool to prevent DoS
- Cryptographic ordering proofs for verification

Security:
- Order is determined by (timestamp_ms, pack_hash) - no manipulation
- Contributions must be signed by contributor
- Ordering proofs allow detection of ordering fraud
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import heapq
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from idi.ian.models import Contribution

logger = logging.getLogger(__name__)


_U64_MODULUS = 1 << 64
_U64_MAX = _U64_MODULUS - 1


# =============================================================================
# Ordering Key
# =============================================================================

@dataclass(frozen=True, order=True)
class OrderingKey:
    """
    Deterministic ordering key for contributions.
    
    Order is determined by:
    1. timestamp_ms (ascending) - earlier contributions first
    2. pack_hash (ascending) - tie-breaker for same timestamp
    
    This ensures all nodes arrive at the same order for the same set
    of contributions, regardless of the order they were received.
    
    Invariants:
    - key(A) < key(B) iff A should be processed before B
    - key(A) == key(B) iff A and B are the same contribution
    """
    timestamp_ms: int
    pack_hash: bytes
    
    def to_bytes(self) -> bytes:
        """Serialize for hashing/signing."""
        return self.timestamp_ms.to_bytes(8, 'big') + self.pack_hash
    
    def __str__(self) -> str:
        return f"OrderKey({self.timestamp_ms}, {self.pack_hash.hex()[:16]}...)"
    
    @classmethod
    def from_contribution(cls, contrib: "Contribution") -> "OrderingKey":
        """Create ordering key from contribution."""
        # Use submission timestamp.
        #
        # Contract:
        # - If contrib.seed is non-zero, treat it as a deterministic timestamp for ordering.
        # - Otherwise, fall back to wall-clock.
        seed_ms = int(contrib.seed)
        if 0 < seed_ms < _U64_MODULUS:
            return cls(timestamp_ms=seed_ms, pack_hash=contrib.pack_hash)
        return cls(timestamp_ms=_U64_MAX, pack_hash=contrib.pack_hash)


# =============================================================================
# Mempool Entry
# =============================================================================

@dataclass
class MempoolEntry:
    """
    Entry in the contribution mempool.
    
    Tracks contribution metadata and ordering information.
    """
    key: OrderingKey
    contribution_hash: bytes  # SHA-256 of full contribution
    contributor_id: str
    goal_id: str
    received_at: float = field(default_factory=time.time)
    from_peer: Optional[str] = None  # Node ID of peer who sent it
    
    # Full contribution (may be None if we only have announcement)
    contribution: Optional["Contribution"] = None
    
    # Processing state
    processed: bool = False
    accepted: bool = False
    
    def __lt__(self, other: "MempoolEntry") -> bool:
        return self.key < other.key
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MempoolEntry):
            return False
        return self.key == other.key


# =============================================================================
# Contribution Mempool
# =============================================================================

class ContributionMempool:
    """
    Ordered mempool for pending contributions.
    
    Maintains contributions in deterministic order for processing.
    All nodes with the same mempool contents will have the same order.
    
    Features:
    - Bounded size to prevent DoS
    - LRU eviction for old contributions
    - Duplicate detection
    - Ordering proofs
    
    Complexity:
    - Insert: O(log N)
    - Pop next: O(log N)
    - Contains: O(1)
    """
    
    def __init__(
        self,
        max_size: int = 10_000,
        max_age_seconds: float = 3600,
        goal_id: Optional[str] = None,
    ):
        """
        Initialize mempool.
        
        Args:
            max_size: Maximum entries in mempool
            max_age_seconds: Maximum age before eviction
            goal_id: Filter for specific goal (None = all goals)
        """
        self._max_size = max_size
        self._max_age_seconds = max_age_seconds
        self._goal_id = goal_id
        
        # Priority queue for ordered processing
        self._heap: List[MempoolEntry] = []
        
        # Index by contribution hash for O(1) lookup
        self._by_hash: Dict[bytes, MempoolEntry] = {}
        
        # Index by ordering key
        self._by_key: Dict[OrderingKey, MempoolEntry] = {}
        
        # Processed contributions (for dedup)
        self._processed: Set[bytes] = set()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def add(
        self,
        contribution: "Contribution",
        from_peer: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Add contribution to mempool.
        
        Args:
            contribution: Contribution to add
            from_peer: Node ID of peer who sent it
            
        Returns:
            (success, reason)
            
        Preconditions:
        - Contribution must be valid (caller responsibility)
        
        Postconditions:
        - If success, contribution is in mempool
        - Order is maintained
        """
        async with self._lock:
            # Check goal filter
            if self._goal_id and str(contribution.goal_id) != self._goal_id:
                return False, "wrong goal_id"
            
            # Create ordering key
            key = OrderingKey.from_contribution(contribution)
            
            # Check for duplicate
            if key in self._by_key:
                return False, "duplicate"
            
            # Compute contribution hash
            contrib_bytes = json.dumps(contribution.to_dict(), sort_keys=True).encode()
            contrib_hash = hashlib.sha256(contrib_bytes).digest()
            
            if contrib_hash in self._by_hash:
                return False, "duplicate"
            
            if contrib_hash in self._processed:
                return False, "already processed"
            
            # Check capacity
            if len(self._heap) >= self._max_size:
                self._evict_oldest()
            
            # Create entry
            entry = MempoolEntry(
                key=key,
                contribution_hash=contrib_hash,
                contributor_id=contribution.contributor_id,
                goal_id=str(contribution.goal_id),
                from_peer=from_peer,
                contribution=contribution,
            )
            
            # Add to data structures
            heapq.heappush(self._heap, entry)
            self._by_hash[contrib_hash] = entry
            self._by_key[key] = entry
            
            logger.debug(f"Mempool: added {key}, size={len(self._heap)}")
            return True, "added"
    
    async def pop_next(self) -> Optional[MempoolEntry]:
        """
        Pop the next contribution to process (lowest ordering key).
        
        Returns:
            Next entry, or None if mempool empty
            
        Postconditions:
        - Entry is removed from mempool
        - Entry is marked as processed
        """
        async with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                
                # Clean up indexes
                self._by_hash.pop(entry.contribution_hash, None)
                self._by_key.pop(entry.key, None)
                
                # Mark as processed
                self._processed.add(entry.contribution_hash)
                entry.processed = True
                
                return entry
            
            return None
    
    async def peek_next(self) -> Optional[MempoolEntry]:
        """Peek at next contribution without removing."""
        async with self._lock:
            if self._heap:
                return self._heap[0]
            return None
    
    async def contains(self, contribution_hash: bytes) -> bool:
        """Check if contribution is in mempool."""
        async with self._lock:
            if contribution_hash in self._by_hash:
                return True
            return contribution_hash in self._processed
    
    async def get_ordered_batch(self, max_count: int = 100) -> List[MempoolEntry]:
        """
        Get a batch of contributions in order without removing.
        
        Used for:
        - State sync with peers
        - Generating ordering proofs
        """
        async with self._lock:
            # Get smallest N entries
            sorted_entries = sorted(self._heap)[:max_count]
            return sorted_entries
    
    async def remove_processed(self, contrib_hashes: List[bytes]) -> int:
        """
        Remove contributions that have been processed externally.
        
        Used when syncing state from peers.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            removed = 0
            for h in contrib_hashes:
                if h in self._by_hash:
                    entry = self._by_hash.pop(h)
                    self._by_key.pop(entry.key, None)
                    # Note: entry remains in heap but will be skipped
                    self._processed.add(h)
                    removed += 1
            
            # Rebuild heap without removed entries
            self._heap = [e for e in self._heap if e.contribution_hash in self._by_hash]
            heapq.heapify(self._heap)
            
            return removed
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry by received_at time."""
        if not self._by_hash:
            return
        
        # Find oldest by received_at
        oldest = min(self._by_hash.values(), key=lambda e: e.received_at)
        
        self._by_hash.pop(oldest.contribution_hash, None)
        self._by_key.pop(oldest.key, None)
        
        # Mark as processed (to prevent re-add)
        self._processed.add(oldest.contribution_hash)
        
        logger.debug(f"Mempool: evicted {oldest.key}")
    
    async def cleanup_old(self) -> int:
        """Remove entries older than max_age_seconds."""
        async with self._lock:
            now = time.time()
            cutoff = now - self._max_age_seconds
            
            to_remove = [
                entry for entry in self._by_hash.values()
                if entry.received_at < cutoff
            ]
            
            for entry in to_remove:
                self._by_hash.pop(entry.contribution_hash, None)
                self._by_key.pop(entry.key, None)
                self._processed.add(entry.contribution_hash)
            
            # Rebuild heap
            self._heap = [e for e in self._heap if e.contribution_hash in self._by_hash]
            heapq.heapify(self._heap)
            
            return len(to_remove)
    
    def __len__(self) -> int:
        return len(self._by_hash)
    
    @property
    def size(self) -> int:
        return len(self._by_hash)
    
    # -------------------------------------------------------------------------
    # Ordering Proofs
    # -------------------------------------------------------------------------
    
    async def get_ordering_proof(self, count: int = 10) -> "OrderingProof":
        """
        Generate proof of current ordering.
        
        This allows other nodes to verify that we're processing
        contributions in the correct order.
        """
        async with self._lock:
            entries = sorted(self._heap)[:count]
            
            return OrderingProof(
                goal_id=self._goal_id or "",
                timestamp_ms=int(time.time() * 1000),
                entries=[
                    OrderingProofEntry(
                        key=e.key,
                        contribution_hash=e.contribution_hash,
                    )
                    for e in entries
                ],
                mempool_size=len(self._by_hash),
            )


# =============================================================================
# Ordering Proofs
# =============================================================================

@dataclass
class OrderingProofEntry:
    """Single entry in an ordering proof."""
    key: OrderingKey
    contribution_hash: bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.key.timestamp_ms,
            "pack_hash": self.key.pack_hash.hex(),
            "contribution_hash": self.contribution_hash.hex(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderingProofEntry":
        return cls(
            key=OrderingKey(
                timestamp_ms=data["timestamp_ms"],
                pack_hash=bytes.fromhex(data["pack_hash"]),
            ),
            contribution_hash=bytes.fromhex(data["contribution_hash"]),
        )


@dataclass
class OrderingProof:
    """
    Proof of contribution ordering at a point in time.
    
    Used to:
    1. Verify nodes are processing in correct order
    2. Detect ordering manipulation
    3. Generate fraud proofs
    """
    goal_id: str
    timestamp_ms: int
    entries: List[OrderingProofEntry]
    mempool_size: int
    signature: Optional[bytes] = None
    signer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "timestamp_ms": self.timestamp_ms,
            "entries": [e.to_dict() for e in self.entries],
            "mempool_size": self.mempool_size,
            "signature": self.signature.hex() if self.signature else None,
            "signer_id": self.signer_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderingProof":
        return cls(
            goal_id=data["goal_id"],
            timestamp_ms=data["timestamp_ms"],
            entries=[OrderingProofEntry.from_dict(e) for e in data["entries"]],
            mempool_size=data["mempool_size"],
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None,
            signer_id=data.get("signer_id"),
        )
    
    def signing_payload(self) -> bytes:
        """Get payload for signing."""
        data = {
            "goal_id": self.goal_id,
            "timestamp_ms": self.timestamp_ms,
            "entries": [e.to_dict() for e in self.entries],
            "mempool_size": self.mempool_size,
        }
        return json.dumps(data, sort_keys=True).encode()
    
    def verify_ordering(self) -> Tuple[bool, str]:
        """
        Verify that entries are in correct order.
        
        Returns:
            (valid, reason)
        """
        for i in range(1, len(self.entries)):
            if self.entries[i].key < self.entries[i-1].key:
                return False, f"entry {i} is out of order"
        return True, "valid"
    
    def compute_root(self) -> bytes:
        """Compute Merkle root of ordering proof."""
        if not self.entries:
            return b'\x00' * 32
        
        leaves = [
            hashlib.sha256(e.key.to_bytes() + e.contribution_hash).digest()
            for e in self.entries
        ]
        
        # Simple Merkle tree
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            leaves = [
                hashlib.sha256(leaves[i] + leaves[i+1]).digest()
                for i in range(0, len(leaves), 2)
            ]
        
        return leaves[0]


# =============================================================================
# Gossip Manager
# =============================================================================

class ContributionGossip:
    """
    Gossip protocol for contribution propagation.
    
    Ensures all nodes receive contributions in a timely manner
    for consistent ordering and processing.
    
    Features:
    - Push-based gossip to random peers
    - Pull-based sync for missing contributions
    - Deduplication to prevent amplification
    """
    
    def __init__(
        self,
        mempool: ContributionMempool,
        node_id: str,
        fanout: int = 3,
        gossip_interval: float = 1.0,
    ):
        """
        Initialize gossip manager.
        
        Args:
            mempool: Contribution mempool
            node_id: This node's ID
            fanout: Number of peers to gossip to
            gossip_interval: Seconds between gossip rounds
        """
        self._mempool = mempool
        self._node_id = node_id
        self._fanout = fanout
        self._gossip_interval = gossip_interval
        
        # Seen contribution hashes (dedup)
        self._seen: Set[bytes] = set()
        self._max_seen = 100_000
        
        # Callbacks
        self._get_peers: Optional[Callable[[], List[str]]] = None
        self._send_message: Optional[Callable[[str, Any], asyncio.Future]] = None
        
        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def set_callbacks(
        self,
        get_peers: Callable[[], List[str]],
        send_message: Callable[[str, Any], asyncio.Future],
    ) -> None:
        """Set gossip callbacks."""
        self._get_peers = get_peers
        self._send_message = send_message
    
    async def start(self) -> None:
        """Start gossip protocol."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._gossip_loop())
        logger.info("Contribution gossip started")
    
    async def stop(self) -> None:
        """Stop gossip protocol."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Contribution gossip stopped")
    
    async def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self._running:
            try:
                await asyncio.sleep(self._gossip_interval)
                await self._do_gossip_round()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip error: {e}")
    
    async def _do_gossip_round(self) -> None:
        """Perform one gossip round."""
        if not self._get_peers or not self._send_message:
            return
        
        # Get pending contributions
        batch = await self._mempool.get_ordered_batch(10)
        if not batch:
            return
        
        # Get random peers
        peers = self._get_peers()
        if not peers:
            return
        
        # Security: Use secrets for unpredictable peer selection to prevent
        # adversaries from predicting gossip targets and partitioning the network
        import secrets
        peer_list = list(peers)
        secrets.SystemRandom().shuffle(peer_list)
        targets = peer_list[:min(self._fanout, len(peer_list))]
        
        # Gossip to each peer
        for peer_id in targets:
            for entry in batch:
                if entry.contribution_hash not in self._seen:
                    try:
                        await self._gossip_contribution(peer_id, entry)
                    except Exception as e:
                        logger.debug(f"Gossip to {peer_id[:16]}... failed: {e}")
    
    async def _gossip_contribution(self, peer_id: str, entry: MempoolEntry) -> None:
        """Gossip single contribution to peer."""
        from .protocol import ContributionAnnounce, ContributionMeta
        
        if entry.contribution is None:
            return
        
        # Create announcement
        announce = ContributionAnnounce(
            sender_id=self._node_id,
            goal_id=entry.goal_id,
            contribution_hash=entry.contribution_hash.hex(),
            contributor_id=entry.contributor_id,
            score=None,  # Not yet processed
            log_index=None,
        )
        
        await self._send_message(peer_id, announce)
    
    async def handle_announcement(
        self,
        announce: "ContributionAnnounce",
        from_peer: str,
    ) -> bool:
        """
        Handle incoming contribution announcement.
        
        Returns:
            True if we need the full contribution
        """
        contrib_hash = bytes.fromhex(announce.contribution_hash)
        
        # Check if already seen
        if contrib_hash in self._seen:
            return False
        
        # Mark as seen
        self._seen.add(contrib_hash)
        self._trim_seen()
        
        # Check if in mempool
        if await self._mempool.contains(contrib_hash):
            return False
        
        # Need full contribution
        return True
    
    def _trim_seen(self) -> None:
        """Trim seen set to max size."""
        if len(self._seen) > self._max_seen:
            # Remove oldest (arbitrary since set has no order)
            excess = len(self._seen) - self._max_seen
            to_remove = list(self._seen)[:excess]
            for h in to_remove:
                self._seen.discard(h)
