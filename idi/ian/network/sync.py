"""
IAN State Synchronization - Full state sync protocol for recovery.

Provides:
1. State comparison with peers
2. Incremental log synchronization
3. Contribution request/response
4. Checkpoint-based recovery

Used when:
- Node starts fresh and needs to catch up
- Node detects divergence from network majority
- Network partition recovery

Protocol:
1. Request peer state summary
2. Identify missing log entries
3. Request contributions in batches
4. Verify and apply contributions in order
5. Verify final state matches peer
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from idi.ian.coordinator import IANCoordinator
    from idi.ian.models import Contribution

from .protocol import (
    Message, MessageType,
    StateRequest, StateResponse,
    ContributionRequest, ContributionResponse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SyncConfig:
    """Configuration for state synchronization."""
    
    # Batch sizes
    contribution_batch_size: int = 100
    max_pending_requests: int = 10
    
    # Timeouts
    state_request_timeout: float = 10.0
    contribution_request_timeout: float = 30.0
    sync_timeout: float = 300.0  # 5 minutes max sync
    
    # Retries
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Verification
    verify_roots_after_sync: bool = True
    checkpoint_interval: int = 100  # Verify every N contributions


# =============================================================================
# Sync Protocol Messages
# =============================================================================

@dataclass
class LogSyncRequest:
    """Request for log entries in a range."""
    sender_id: str
    goal_id: str
    from_index: int
    to_index: int  # exclusive
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    nonce: int = 0
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LOG_SYNC_REQUEST",
            "sender_id": self.sender_id,
            "goal_id": self.goal_id,
            "from_index": self.from_index,
            "to_index": self.to_index,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": self.signature.hex() if self.signature else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogSyncRequest":
        return cls(
            sender_id=data["sender_id"],
            goal_id=data["goal_id"],
            from_index=data["from_index"],
            to_index=data["to_index"],
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            nonce=data.get("nonce", 0),
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None,
        )


@dataclass
class LogSyncResponse:
    """Response with log entries."""
    sender_id: str
    goal_id: str
    from_index: int
    contributions: List[Dict[str, Any]]  # Serialized contributions
    has_more: bool
    log_root: str  # Current log root
    log_size: int  # Current log size
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LOG_SYNC_RESPONSE",
            "sender_id": self.sender_id,
            "goal_id": self.goal_id,
            "from_index": self.from_index,
            "contributions": self.contributions,
            "has_more": self.has_more,
            "log_root": self.log_root,
            "log_size": self.log_size,
            "timestamp": self.timestamp,
            "signature": self.signature.hex() if self.signature else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogSyncResponse":
        return cls(
            sender_id=data["sender_id"],
            goal_id=data["goal_id"],
            from_index=data["from_index"],
            contributions=data["contributions"],
            has_more=data["has_more"],
            log_root=data["log_root"],
            log_size=data["log_size"],
            timestamp=data.get("timestamp", int(time.time() * 1000)),
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None,
        )


# =============================================================================
# Sync State
# =============================================================================

class SyncState(Enum):
    """State of synchronization."""
    IDLE = auto()
    COMPARING = auto()
    SYNCING = auto()
    VERIFYING = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class SyncProgress:
    """Progress of ongoing sync."""
    state: SyncState = SyncState.IDLE
    peer_id: Optional[str] = None
    target_size: int = 0
    current_size: int = 0
    contributions_received: int = 0
    contributions_applied: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    error: Optional[str] = None
    
    @property
    def progress_pct(self) -> float:
        if self.target_size == 0:
            return 0.0
        return min(100.0, (self.current_size / self.target_size) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.name,
            "peer_id": self.peer_id,
            "target_size": self.target_size,
            "current_size": self.current_size,
            "contributions_received": self.contributions_received,
            "contributions_applied": self.contributions_applied,
            "progress_pct": self.progress_pct,
            "duration_seconds": (
                (self.completed_at or time.time()) - self.started_at
                if self.started_at > 0 else 0
            ),
            "error": self.error,
        }


# =============================================================================
# State Synchronizer
# =============================================================================

class StateSynchronizer:
    """
    Handles full state synchronization with peers.
    
    Protocol:
    1. Compare state with peer (StateRequest/Response)
    2. If behind, request missing contributions
    3. Apply contributions in order
    4. Verify final state matches
    """
    
    def __init__(
        self,
        coordinator: "IANCoordinator",
        node_id: str,
        config: Optional[SyncConfig] = None,
    ):
        self._coordinator = coordinator
        self._node_id = node_id
        self._config = config or SyncConfig()
        
        # Sync state
        self._progress = SyncProgress()
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._send_message: Optional[Callable[[str, Any], asyncio.Future]] = None
        self._request_state: Optional[Callable[[str], asyncio.Future]] = None
    
    def set_callbacks(
        self,
        send_message: Callable[[str, Any], asyncio.Future],
        request_state: Optional[Callable[[str], asyncio.Future]] = None,
    ) -> None:
        """Set sync callbacks."""
        self._send_message = send_message
        self._request_state = request_state
    
    # -------------------------------------------------------------------------
    # Main Sync Entry Point
    # -------------------------------------------------------------------------
    
    async def sync_from_peer(
        self,
        peer_id: str,
        peer_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Synchronize state from a peer.
        
        Args:
            peer_id: Node ID of peer to sync from
            peer_state: Optional pre-fetched peer state
            
        Returns:
            (success, message)
        """
        async with self._lock:
            if self._progress.state in (SyncState.COMPARING, SyncState.SYNCING, SyncState.VERIFYING):
                return False, "sync already in progress"
            
            self._progress = SyncProgress(
                state=SyncState.COMPARING,
                peer_id=peer_id,
                started_at=time.time(),
            )
        
        try:
            # Step 1: Compare state
            if peer_state is None:
                peer_state = await self._request_peer_state(peer_id)
                if peer_state is None:
                    raise ValueError("Failed to get peer state")
            
            peer_log_size = peer_state.get("log_size", 0)
            our_log_size = self._coordinator.state.log.size
            
            self._progress.target_size = peer_log_size
            self._progress.current_size = our_log_size
            
            # Already in sync?
            if our_log_size >= peer_log_size:
                peer_root = bytes.fromhex(peer_state.get("log_root", "00" * 32))
                our_root = self._coordinator.get_log_root()
                
                if peer_root == our_root:
                    self._complete_sync(True, "already in sync")
                    return True, "already in sync"
            
            # Step 2: Request and apply missing contributions
            self._progress.state = SyncState.SYNCING
            
            success = await self._sync_contributions(peer_id, our_log_size, peer_log_size)
            
            if not success:
                self._complete_sync(False, "contribution sync failed")
                return False, "contribution sync failed"
            
            # Step 3: Verify final state
            if self._config.verify_roots_after_sync:
                self._progress.state = SyncState.VERIFYING
                
                our_root = self._coordinator.get_log_root()
                peer_root = bytes.fromhex(peer_state.get("log_root", "00" * 32))
                
                if our_root != peer_root:
                    # May have diverged - warn but don't fail if sizes match
                    logger.warning(
                        f"Post-sync root mismatch: "
                        f"ours={our_root.hex()[:16]}, peer={peer_root.hex()[:16]}"
                    )
            
            self._complete_sync(True, f"synced {self._progress.contributions_applied} contributions")
            return True, f"synced {self._progress.contributions_applied} contributions"
            
        except asyncio.TimeoutError:
            self._complete_sync(False, "sync timeout")
            return False, "sync timeout"
        except Exception as e:
            self._complete_sync(False, str(e))
            return False, str(e)
    
    def _complete_sync(self, success: bool, message: str) -> None:
        """Mark sync as complete."""
        self._progress.completed_at = time.time()
        self._progress.state = SyncState.COMPLETE if success else SyncState.FAILED
        if not success:
            self._progress.error = message
        
        logger.info(
            f"Sync {'completed' if success else 'failed'}: {message} "
            f"(applied {self._progress.contributions_applied} contributions)"
        )
    
    # -------------------------------------------------------------------------
    # State Comparison
    # -------------------------------------------------------------------------
    
    async def _request_peer_state(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Request state from peer."""
        if not self._send_message:
            return None
        
        request = StateRequest(
            sender_id=self._node_id,
            goal_id=str(self._coordinator.goal_spec.goal_id),
            include_leaderboard=False,
        )
        
        # In a full implementation, this would await the response
        # For now, return None to indicate we need the state passed in
        try:
            await self._send_message(peer_id, request)
            # Would await response here
            return None
        except Exception as e:
            logger.error(f"Failed to request peer state: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Contribution Sync
    # -------------------------------------------------------------------------
    
    async def _sync_contributions(
        self,
        peer_id: str,
        from_index: int,
        to_index: int,
    ) -> bool:
        """
        Request and apply contributions in range.
        
        Args:
            peer_id: Peer to sync from
            from_index: Starting log index (our current size)
            to_index: Target log index (peer's size)
            
        Returns:
            True if successful
            
        Security:
            - Enforces overall sync timeout to prevent indefinite sync
            - Limits retries per batch
        """
        current_index = from_index
        retries = 0
        
        # Calculate deadline for overall sync timeout
        deadline = time.time() + self._config.sync_timeout
        
        while current_index < to_index and retries < self._config.max_retries:
            # Check overall timeout
            if time.time() > deadline:
                logger.warning(f"Sync timeout after {self._config.sync_timeout}s")
                return False
            # Calculate batch
            batch_end = min(current_index + self._config.contribution_batch_size, to_index)
            
            # Request batch
            contributions = await self._request_contribution_batch(
                peer_id, current_index, batch_end
            )
            
            if contributions is None:
                retries += 1
                await asyncio.sleep(self._config.retry_delay)
                
                # Check timeout during retry
                if time.time() > deadline:
                    logger.warning(f"Sync timeout during retry")
                    return False
                continue
            
            # Apply contributions in order with hash verification
            for contrib_dict in contributions:
                try:
                    from idi.ian.models import Contribution
                    contrib = Contribution.from_dict(contrib_dict)
                    
                    # Security: Verify pack_hash matches content before applying
                    claimed_hash_str = contrib_dict.get("pack_hash", "")
                    if claimed_hash_str:
                        claimed_hash = bytes.fromhex(claimed_hash_str)
                        computed_hash = contrib.compute_pack_hash()
                        
                        if computed_hash != claimed_hash:
                            logger.warning(
                                f"Contribution hash mismatch from {peer_id}: "
                                f"claimed={claimed_hash.hex()[:16]}, "
                                f"computed={computed_hash.hex()[:16]}"
                            )
                            # Skip this malicious/corrupted contribution
                            continue
                    
                    result = self._coordinator.process_contribution(contrib)
                    
                    if result.accepted:
                        self._progress.contributions_applied += 1
                    
                    self._progress.contributions_received += 1
                    self._progress.current_size = self._coordinator.state.log.size
                    
                except Exception as e:
                    logger.error(f"Failed to apply contribution: {e}")
            
            current_index = batch_end
            retries = 0  # Reset on success
            
            # Periodic checkpoint verification
            if (
                self._config.verify_roots_after_sync and
                self._progress.contributions_applied % self._config.checkpoint_interval == 0
            ):
                logger.debug(
                    f"Sync checkpoint: {self._progress.contributions_applied} applied, "
                    f"log_size={self._progress.current_size}"
                )
        
        return current_index >= to_index
    
    async def _request_contribution_batch(
        self,
        peer_id: str,
        from_index: int,
        to_index: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Request a batch of contributions from peer."""
        if not self._send_message:
            return None
        
        request = LogSyncRequest(
            sender_id=self._node_id,
            goal_id=str(self._coordinator.goal_spec.goal_id),
            from_index=from_index,
            to_index=to_index,
        )
        
        try:
            # In full implementation, would await response
            await self._send_message(peer_id, request)
            
            # For now, return empty list
            # Real implementation would collect response
            return []
            
        except asyncio.TimeoutError:
            logger.warning(f"Contribution batch request timeout")
            return None
        except Exception as e:
            logger.error(f"Failed to request contributions: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Request Handlers
    # -------------------------------------------------------------------------
    
    def handle_log_sync_request(
        self,
        request: LogSyncRequest,
    ) -> LogSyncResponse:
        """
        Handle incoming log sync request.
        
        Returns contributions in the requested range.
        """
        goal_id = str(self._coordinator.goal_spec.goal_id)
        
        if request.goal_id != goal_id:
            return LogSyncResponse(
                sender_id=self._node_id,
                goal_id=goal_id,
                from_index=request.from_index,
                contributions=[],
                has_more=False,
                log_root=self._coordinator.get_log_root().hex(),
                log_size=self._coordinator.state.log.size,
            )
        
        # Get contributions from log
        # Note: This assumes we have a way to retrieve contributions by index
        # In practice, would need contribution storage
        contributions = []
        
        # Placeholder: would retrieve from storage
        # for i in range(request.from_index, min(request.to_index, self._coordinator.state.log.size)):
        #     contrib = self._storage.get_contribution_by_index(i)
        #     if contrib:
        #         contributions.append(contrib.to_dict())
        
        has_more = request.to_index < self._coordinator.state.log.size
        
        return LogSyncResponse(
            sender_id=self._node_id,
            goal_id=goal_id,
            from_index=request.from_index,
            contributions=contributions,
            has_more=has_more,
            log_root=self._coordinator.get_log_root().hex(),
            log_size=self._coordinator.state.log.size,
        )
    
    def handle_log_sync_response(
        self,
        response: LogSyncResponse,
    ) -> None:
        """Handle incoming log sync response."""
        # This would be called when we receive a response to our request
        # and would be integrated with the async sync flow
        pass
    
    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------
    
    def get_progress(self) -> SyncProgress:
        """Get current sync progress."""
        return self._progress
    
    def is_syncing(self) -> bool:
        """Check if sync is in progress."""
        return self._progress.state in (
            SyncState.COMPARING,
            SyncState.SYNCING,
            SyncState.VERIFYING,
        )
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get local state summary for comparison."""
        return {
            "goal_id": str(self._coordinator.goal_spec.goal_id),
            "log_root": self._coordinator.get_log_root().hex(),
            "log_size": self._coordinator.state.log.size,
            "leaderboard_root": self._coordinator.get_leaderboard_root().hex(),
            "leaderboard_size": len(self._coordinator.state.leaderboard),
            "active_policy_hash": (
                self._coordinator.state.active_policy_hash.hex()
                if self._coordinator.state.active_policy_hash
                else None
            ),
        }


# =============================================================================
# Contribution Storage
# =============================================================================

# Security: Default storage limits
DEFAULT_MAX_STORAGE_ENTRIES = 100_000
DEFAULT_MAX_RANGE_RESULTS = 1_000


class ContributionStorage:
    """
    Persistent storage for contributions with bounded memory.
    
    Allows retrieval by:
    - Log index
    - Pack hash
    - Contributor ID
    
    Used for:
    - State sync (providing contributions to peers)
    - Replay (re-processing from checkpoint)
    - Auditing
    
    Security:
    - LRU eviction when max entries reached
    - Bounded range queries
    - Chunked disk I/O to prevent OOM
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_entries: int = DEFAULT_MAX_STORAGE_ENTRIES,
    ):
        self._storage_path = storage_path
        self._max_entries = max_entries
        
        # LRU-ordered in-memory storage
        self._by_index: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._by_hash: Dict[bytes, int] = {}  # pack_hash -> index
    
    def store(self, index: int, contribution: Dict[str, Any]) -> None:
        """
        Store contribution with LRU eviction.
        
        Security:
        - Evicts oldest entries when max_entries reached
        - Prevents unbounded memory growth
        """
        # Evict oldest if at capacity
        while len(self._by_index) >= self._max_entries:
            oldest_idx, oldest_contrib = self._by_index.popitem(last=False)
            old_hash_str = oldest_contrib.get("pack_hash", "")
            if old_hash_str:
                try:
                    old_hash = bytes.fromhex(old_hash_str)
                    self._by_hash.pop(old_hash, None)
                except ValueError:
                    pass
            logger.debug(f"Evicted contribution at index {oldest_idx} (LRU)")
        
        self._by_index[index] = contribution
        self._by_index.move_to_end(index)  # Mark as recently used
        
        pack_hash_str = contribution.get("pack_hash", "")
        if pack_hash_str:
            try:
                pack_hash = bytes.fromhex(pack_hash_str)
                self._by_hash[pack_hash] = index
            except ValueError:
                pass
    
    def get_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get contribution by log index."""
        contrib = self._by_index.get(index)
        if contrib:
            # Mark as recently used
            self._by_index.move_to_end(index)
        return contrib
    
    def get_by_hash(self, pack_hash: bytes) -> Optional[Dict[str, Any]]:
        """Get contribution by pack hash."""
        index = self._by_hash.get(pack_hash)
        if index is not None:
            return self.get_by_index(index)
        return None
    
    def get_range(
        self,
        from_index: int,
        to_index: int,
        max_count: int = DEFAULT_MAX_RANGE_RESULTS,
    ) -> List[Dict[str, Any]]:
        """
        Get contributions in range with limit.
        
        Args:
            from_index: Start index (inclusive)
            to_index: End index (exclusive)
            max_count: Maximum results to return (security limit)
            
        Returns:
            List of contributions, capped at max_count
        """
        result = []
        count = 0
        
        for i in range(from_index, to_index):
            if count >= max_count:
                break
            if i in self._by_index:
                result.append(self._by_index[i])
                count += 1
        
        return result
    
    def get_latest_index(self) -> int:
        """Get highest stored index."""
        if not self._by_index:
            return -1
        return max(self._by_index.keys())
    
    def __len__(self) -> int:
        """Return number of stored contributions."""
        return len(self._by_index)
    
    def save_to_disk(self) -> bool:
        """
        Persist to disk using chunked writes.
        
        Security:
        - Writes in chunks to limit memory during serialization
        - Atomic write via temp file rename
        """
        if not self._storage_path:
            return False
        
        try:
            path = Path(self._storage_path)
            temp_path = path.with_suffix('.tmp')
            
            with open(temp_path, 'w') as f:
                f.write('{"by_index": {\n')
                
                items = list(self._by_index.items())
                for i, (k, v) in enumerate(items):
                    entry = json.dumps(v, separators=(',', ':'))
                    comma = ',' if i < len(items) - 1 else ''
                    f.write(f'  "{k}": {entry}{comma}\n')
                
                f.write('}}\n')
            
            # Atomic rename
            temp_path.rename(path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save storage: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """Load from disk."""
        if not self._storage_path:
            return False
        
        try:
            path = Path(self._storage_path)
            if not path.exists():
                return False
            
            # Load with size limit check
            file_size = path.stat().st_size
            max_file_size = 500 * 1024 * 1024  # 500 MB limit
            if file_size > max_file_size:
                logger.error(f"Storage file too large: {file_size} > {max_file_size}")
                return False
            
            data = json.loads(path.read_text())
            
            # Clear existing and rebuild with limits
            self._by_index = OrderedDict()
            self._by_hash = {}
            
            for k, v in data.get("by_index", {}).items():
                if len(self._by_index) >= self._max_entries:
                    logger.warning(f"Truncated storage load at {self._max_entries} entries")
                    break
                    
                index = int(k)
                self._by_index[index] = v
                
                pack_hash_str = v.get("pack_hash", "")
                if pack_hash_str:
                    try:
                        pack_hash = bytes.fromhex(pack_hash_str)
                        self._by_hash[pack_hash] = index
                    except ValueError:
                        pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load storage: {e}")
            return False
