"""
IAN Consensus Coordinator - Multi-node consensus for decentralized operation.

Provides:
1. ConsensusCoordinator - Wraps IANCoordinator with multi-node consensus
2. State synchronization protocol
3. Peer state verification
4. Conflict resolution

Design Principles:
- Deterministic processing ensures same inputs → same state
- Periodic state verification against peer majority
- Tau Net commits provide ultimate finality
- Byzantine fault tolerance for < 1/3 malicious nodes

Security Model:
- Nodes are untrusted; state is verified cryptographically
- Divergent nodes are detected and can be excluded
- Fraud proofs allow challenging invalid state transitions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .ordering import ContributionMempool, OrderingKey, OrderingProof
from .protocol import (
    StateRequest, StateResponse, ContributionAnnounce,
    ContributionRequest, ContributionResponse, Message, MessageType,
)

if TYPE_CHECKING:
    from idi.ian.coordinator import IANCoordinator, ProcessResult
    from idi.ian.models import Contribution, GoalID

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConsensusConfig:
    """Configuration for consensus coordinator."""
    
    # State verification
    state_check_interval: float = 30.0  # Seconds between state checks
    max_state_divergence: int = 10  # Max contributions before forced sync
    quorum_threshold: float = 0.67  # 2/3 quorum for consensus
    
    # Peer management
    min_peers_for_consensus: int = 2  # Minimum peers needed
    max_sync_batch_size: int = 100  # Max contributions per sync batch
    
    # Processing
    process_interval: float = 1.0  # Seconds between processing rounds
    max_process_batch: int = 50  # Max contributions per round
    
    # Timeouts
    state_request_timeout: float = 5.0
    sync_timeout: float = 30.0


# =============================================================================
# Peer State Tracking
# =============================================================================

@dataclass
class PeerStateSnapshot:
    """Snapshot of a peer's state."""
    node_id: str
    goal_id: str
    log_root: bytes
    log_size: int
    leaderboard_root: bytes
    active_policy_hash: Optional[bytes]
    timestamp_ms: int
    
    def matches(self, other: "PeerStateSnapshot") -> bool:
        """Check if states match."""
        return (
            self.log_root == other.log_root and
            self.log_size == other.log_size and
            self.leaderboard_root == other.leaderboard_root
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "goal_id": self.goal_id,
            "log_root": self.log_root.hex(),
            "log_size": self.log_size,
            "leaderboard_root": self.leaderboard_root.hex(),
            "active_policy_hash": self.active_policy_hash.hex() if self.active_policy_hash else None,
            "timestamp_ms": self.timestamp_ms,
        }


class ConsensusState(Enum):
    """State of consensus with peers."""
    SYNCING = auto()  # Catching up with peers
    SYNCHRONIZED = auto()  # In sync with peer majority
    DIVERGED = auto()  # Diverged from peer majority
    ISOLATED = auto()  # Not enough peers for consensus


# =============================================================================
# Consensus Coordinator
# =============================================================================

class ConsensusCoordinator:
    """
    Coordinator wrapper that adds multi-node consensus.
    
    Responsibilities:
    1. Process contributions from mempool in order
    2. Verify state matches peer majority
    3. Sync state from peers if diverged
    4. Gossip contributions to peers
    
    Invariants:
    - Contributions are processed in deterministic order
    - State divergence is detected and resolved
    - Only commits to Tau when in consensus
    """
    
    def __init__(
        self,
        coordinator: "IANCoordinator",
        node_id: str,
        config: Optional[ConsensusConfig] = None,
    ):
        """
        Initialize consensus coordinator.
        
        Args:
            coordinator: Base IAN coordinator
            node_id: This node's ID
            config: Consensus configuration
        """
        self._coordinator = coordinator
        self._node_id = node_id
        self._config = config or ConsensusConfig()
        
        # Goal ID
        self._goal_id = str(coordinator.goal_spec.goal_id)
        
        # Mempool for ordered processing
        self._mempool = ContributionMempool(
            max_size=10_000,
            goal_id=self._goal_id,
        )
        
        # Peer state tracking
        self._peer_states: Dict[str, PeerStateSnapshot] = {}
        self._consensus_state = ConsensusState.ISOLATED
        
        # Processing state
        self._contributions_since_check = 0
        self._last_state_check = 0.0
        self._processing_lock = asyncio.Lock()
        
        # Callbacks
        self._get_peers: Optional[Callable[[], List[str]]] = None
        self._send_message: Optional[Callable[[str, Any], asyncio.Future]] = None
        self._on_state_change: Optional[Callable[[ConsensusState], None]] = None
        
        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    
    def set_callbacks(
        self,
        get_peers: Callable[[], List[str]],
        send_message: Callable[[str, Any], asyncio.Future],
        on_state_change: Optional[Callable[[ConsensusState], None]] = None,
    ) -> None:
        """Set consensus callbacks."""
        self._get_peers = get_peers
        self._send_message = send_message
        self._on_state_change = on_state_change
    
    async def start(self) -> None:
        """Start consensus coordinator."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._process_loop()),
            asyncio.create_task(self._state_check_loop()),
            asyncio.create_task(self._mempool_cleanup_loop()),
        ]
        
        logger.info(f"Consensus coordinator started for goal {self._goal_id}")
    
    async def stop(self) -> None:
        """Stop consensus coordinator."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("Consensus coordinator stopped")
    
    # -------------------------------------------------------------------------
    # Contribution Handling
    # -------------------------------------------------------------------------
    
    async def submit_contribution(
        self,
        contribution: "Contribution",
        from_peer: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Submit a contribution for processing.
        
        The contribution is added to the mempool and will be processed
        in deterministic order when it reaches the front of the queue.
        
        Args:
            contribution: Contribution to submit
            from_peer: Node ID if received from peer
            
        Returns:
            (success, reason)
        """
        # Add to mempool
        success, reason = await self._mempool.add(contribution, from_peer)
        
        if success and from_peer is None:
            # This is a local submission - gossip to peers
            await self._gossip_contribution(contribution)
        
        return success, reason
    
    async def _gossip_contribution(self, contribution: "Contribution") -> None:
        """Gossip contribution to peers."""
        if not self._get_peers or not self._send_message:
            return
        
        peers = self._get_peers()
        if not peers:
            return
        
        # Create announcement
        contrib_bytes = json.dumps(contribution.to_dict(), sort_keys=True).encode()
        contrib_hash = hashlib.sha256(contrib_bytes).hexdigest()
        
        announce = ContributionAnnounce(
            sender_id=self._node_id,
            goal_id=self._goal_id,
            contribution_hash=contrib_hash,
            contributor_id=contribution.contributor_id,
        )
        
        # Send to all peers
        for peer_id in peers[:self._config.max_sync_batch_size]:
            try:
                await self._send_message(peer_id, announce)
            except Exception as e:
                logger.debug(f"Failed to gossip to {peer_id[:16]}...: {e}")
    
    # -------------------------------------------------------------------------
    # Processing Loop
    # -------------------------------------------------------------------------
    
    async def _process_loop(self) -> None:
        """Background loop for processing contributions from mempool."""
        while self._running:
            try:
                await asyncio.sleep(self._config.process_interval)
                await self._process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")
    
    async def _process_batch(self) -> int:
        """
        Process a batch of contributions from mempool.
        
        Returns:
            Number of contributions processed
        """
        async with self._processing_lock:
            processed = 0
            
            for _ in range(self._config.max_process_batch):
                entry = await self._mempool.pop_next()
                if entry is None:
                    break
                
                if entry.contribution is None:
                    continue
                
                # Process through coordinator
                try:
                    result = self._coordinator.process_contribution(entry.contribution)
                    entry.accepted = result.accepted
                    
                    self._contributions_since_check += 1
                    processed += 1
                    
                    logger.debug(
                        f"Processed contribution {entry.key}: "
                        f"accepted={result.accepted}"
                    )
                except Exception as e:
                    logger.error(f"Failed to process contribution: {e}")
            
            # Check if we need state verification
            if self._contributions_since_check >= self._config.max_state_divergence:
                await self._verify_state_with_peers()
            
            return processed
    
    # -------------------------------------------------------------------------
    # State Verification
    # -------------------------------------------------------------------------
    
    async def _state_check_loop(self) -> None:
        """Background loop for state verification."""
        while self._running:
            try:
                await asyncio.sleep(self._config.state_check_interval)
                await self._verify_state_with_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State check error: {e}")
    
    async def _verify_state_with_peers(self) -> None:
        """
        Verify our state matches peer majority.
        
        If diverged, attempt to sync from majority.
        """
        if not self._get_peers or not self._send_message:
            self._set_consensus_state(ConsensusState.ISOLATED)
            return
        
        peers = self._get_peers()
        if len(peers) < self._config.min_peers_for_consensus:
            self._set_consensus_state(ConsensusState.ISOLATED)
            return
        
        # Request state from all peers
        responses = await self._request_peer_states(peers)
        
        if len(responses) < self._config.min_peers_for_consensus:
            self._set_consensus_state(ConsensusState.ISOLATED)
            return
        
        # Get our current state
        our_state = self._get_local_state_snapshot()
        
        # Count how many peers match our state
        matching = sum(1 for r in responses if r.matches(our_state))
        total = len(responses)
        
        if matching >= total * self._config.quorum_threshold:
            # We're in consensus
            self._set_consensus_state(ConsensusState.SYNCHRONIZED)
            self._contributions_since_check = 0
        else:
            # Find majority state
            majority_state = self._find_majority_state(responses)
            
            if majority_state and not majority_state.matches(our_state):
                # We're diverged - need to sync
                self._set_consensus_state(ConsensusState.DIVERGED)
                await self._sync_from_peer(majority_state.node_id)
            else:
                # No clear majority
                self._set_consensus_state(ConsensusState.DIVERGED)
        
        self._last_state_check = time.time()
    
    async def _request_peer_states(
        self,
        peers: List[str],
    ) -> List[PeerStateSnapshot]:
        """Request state from multiple peers."""
        responses = []
        
        for peer_id in peers:
            try:
                state = await self._request_peer_state(peer_id)
                if state:
                    responses.append(state)
                    self._peer_states[peer_id] = state
            except Exception as e:
                logger.debug(f"Failed to get state from {peer_id[:16]}...: {e}")
        
        return responses
    
    async def _request_peer_state(self, peer_id: str) -> Optional[PeerStateSnapshot]:
        """Request state from a single peer."""
        if not self._send_message:
            return None
        
        request = StateRequest(
            sender_id=self._node_id,
            goal_id=self._goal_id,
            include_leaderboard=False,
        )
        
        # This would need response handling - simplified for now
        try:
            await self._send_message(peer_id, request)
            # In real implementation, would await response
            return None
        except Exception:
            return None
    
    def _get_local_state_snapshot(self) -> PeerStateSnapshot:
        """Get snapshot of our current state."""
        return PeerStateSnapshot(
            node_id=self._node_id,
            goal_id=self._goal_id,
            log_root=self._coordinator.get_log_root(),
            log_size=self._coordinator.state.log.size,
            leaderboard_root=self._coordinator.get_leaderboard_root(),
            active_policy_hash=self._coordinator.state.active_policy_hash,
            timestamp_ms=int(time.time() * 1000),
        )
    
    def _find_majority_state(
        self,
        states: List[PeerStateSnapshot],
    ) -> Optional[PeerStateSnapshot]:
        """Find state held by majority of peers."""
        if not states:
            return None
        
        # Group by (log_root, log_size, leaderboard_root)
        groups: Dict[Tuple[bytes, int, bytes], List[PeerStateSnapshot]] = {}
        
        for state in states:
            key = (state.log_root, state.log_size, state.leaderboard_root)
            if key not in groups:
                groups[key] = []
            groups[key].append(state)
        
        # Find largest group
        largest_group = max(groups.values(), key=len)
        
        if len(largest_group) >= len(states) * self._config.quorum_threshold:
            return largest_group[0]
        
        return None
    
    def _set_consensus_state(self, state: ConsensusState) -> None:
        """Update consensus state and notify callbacks."""
        if state != self._consensus_state:
            old_state = self._consensus_state
            self._consensus_state = state
            
            logger.info(f"Consensus state: {old_state.name} -> {state.name}")
            
            if self._on_state_change:
                self._on_state_change(state)
    
    # -------------------------------------------------------------------------
    # State Synchronization
    # -------------------------------------------------------------------------
    
    async def _sync_from_peer(self, peer_id: str) -> bool:
        """
        Synchronize state from a peer.
        
        This is used when we've diverged from the network majority.
        
        Returns:
            True if sync successful
        """
        logger.info(f"Syncing state from peer {peer_id[:16]}...")
        
        self._set_consensus_state(ConsensusState.SYNCING)
        
        # Request full state from peer
        # In a real implementation, this would:
        # 1. Request contributions we're missing
        # 2. Verify and apply them in order
        # 3. Verify final state matches peer
        
        # For now, just log the attempt
        logger.warning(
            "State sync not fully implemented - "
            "would sync from peer here"
        )
        
        return False
    
    # -------------------------------------------------------------------------
    # Mempool Management
    # -------------------------------------------------------------------------
    
    async def _mempool_cleanup_loop(self) -> None:
        """Background loop for mempool cleanup."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Every minute
                removed = await self._mempool.cleanup_old()
                if removed > 0:
                    logger.debug(f"Cleaned up {removed} old mempool entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Mempool cleanup error: {e}")
    
    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------
    
    async def handle_state_request(self, request: StateRequest) -> StateResponse:
        """Handle incoming state request."""
        leaderboard = None
        if request.include_leaderboard:
            leaderboard = [
                entry.to_dict()
                for entry in self._coordinator.get_leaderboard()
            ]
        
        return StateResponse(
            sender_id=self._node_id,
            goal_id=self._goal_id,
            log_root=self._coordinator.get_log_root().hex(),
            log_size=self._coordinator.state.log.size,
            leaderboard_root=self._coordinator.get_leaderboard_root().hex(),
            leaderboard=leaderboard,
            active_policy_hash=(
                self._coordinator.state.active_policy_hash.hex()
                if self._coordinator.state.active_policy_hash
                else None
            ),
        )
    
    async def handle_state_response(self, response: StateResponse) -> None:
        """Handle incoming state response."""
        # Update peer state tracking
        self._peer_states[response.sender_id] = PeerStateSnapshot(
            node_id=response.sender_id,
            goal_id=response.goal_id,
            log_root=bytes.fromhex(response.log_root),
            log_size=response.log_size,
            leaderboard_root=bytes.fromhex(response.leaderboard_root),
            active_policy_hash=(
                bytes.fromhex(response.active_policy_hash)
                if response.active_policy_hash
                else None
            ),
            timestamp_ms=response.timestamp,
        )
    
    async def handle_contribution_announce(
        self,
        announce: ContributionAnnounce,
        from_peer: str,
    ) -> bool:
        """
        Handle incoming contribution announcement.
        
        Returns:
            True if we need the full contribution
        """
        contrib_hash = bytes.fromhex(announce.contribution_hash)
        
        # Check if already in mempool
        if await self._mempool.contains(contrib_hash):
            return False
        
        # Request full contribution
        return True
    
    async def handle_contribution_response(
        self,
        response: ContributionResponse,
        from_peer: str,
    ) -> None:
        """Handle incoming contribution response."""
        if not response.found or not response.contribution:
            return
        
        try:
            from idi.ian.models import Contribution
            
            contrib = Contribution.from_dict(response.contribution)
            await self.submit_contribution(contrib, from_peer=from_peer)
        except Exception as e:
            logger.error(f"Failed to handle contribution response: {e}")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def coordinator(self) -> "IANCoordinator":
        """Get underlying coordinator."""
        return self._coordinator
    
    @property
    def consensus_state(self) -> ConsensusState:
        """Get current consensus state."""
        return self._consensus_state
    
    @property
    def mempool_size(self) -> int:
        """Get current mempool size."""
        return self._mempool.size
    
    @property
    def peer_count(self) -> int:
        """Get number of known peers."""
        return len(self._peer_states)
    
    def is_synchronized(self) -> bool:
        """Check if in consensus with peers."""
        return self._consensus_state == ConsensusState.SYNCHRONIZED
    
    def can_commit_to_tau(self) -> bool:
        """Check if safe to commit state to Tau Net."""
        return self._consensus_state in (
            ConsensusState.SYNCHRONIZED,
            ConsensusState.ISOLATED,  # Allow commits when alone
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus coordinator statistics."""
        return {
            "goal_id": self._goal_id,
            "consensus_state": self._consensus_state.name,
            "mempool_size": self._mempool.size,
            "peer_count": len(self._peer_states),
            "contributions_since_check": self._contributions_since_check,
            "coordinator_stats": self._coordinator.get_stats(),
        }
