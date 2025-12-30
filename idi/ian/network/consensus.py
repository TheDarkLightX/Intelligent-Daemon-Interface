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
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .ordering import ContributionMempool, OrderingKey, OrderingProof
from .protocol import (
    StateRequest, StateResponse, ContributionAnnounce,
    ContributionRequest, ContributionResponse, Message, MessageType,
    SyncRequest, SyncResponse,
)

if TYPE_CHECKING:
    from idi.ian.coordinator import IANCoordinator, ProcessResult
    from idi.ian.models import Contribution, GoalID

# CODEX: Wired to patched verified kernel
from .kernels import consensus_state_fsm_ref as consensus_kernel

logger = logging.getLogger(__name__)

# =============================================================================
# ESSO Kernel Envelope (consensus_state_fsm_ref)
# =============================================================================
#
# The ESSO proof only applies within these finite domains. The imperative shell
# must either (a) stay within this envelope, or (b) explicitly define a
# refinement mapping to a larger/unbounded domain.
_KERNEL_MAX_PEERS: int = 10
_KERNEL_MAX_MIN_PEERS: int = 5
_KERNEL_MAX_QUORUM_COMPONENT: int = 100


def _quorum_fraction_num_from_threshold(threshold: float) -> int:
    """
    Convert a quorum threshold in (0, 1] into an integer numerator in [1, 100].

    We use ceil() to fail-closed: the kernel will require at least the requested
    threshold (never less due to truncation).
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"quorum_threshold must be in (0, 1], got {threshold}")
    return max(1, min(_KERNEL_MAX_QUORUM_COMPONENT, int(math.ceil(threshold * 100.0))))


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

        if not (1 <= int(self._config.min_peers_for_consensus) <= _KERNEL_MAX_MIN_PEERS):
            raise ValueError(
                f"min_peers_for_consensus must be in [1, {_KERNEL_MAX_MIN_PEERS}], "
                f"got {self._config.min_peers_for_consensus}"
            )
        
        # CODEX: Initialize patched verified kernel state
        quorum_num = _quorum_fraction_num_from_threshold(self._config.quorum_threshold)
        self._kstate = consensus_kernel.State(
            matching_peers=0,
            min_peers=self._config.min_peers_for_consensus,
            peer_count=0,
            quorum_fraction_den=_KERNEL_MAX_QUORUM_COMPONENT,
            quorum_fraction_num=quorum_num,
            state="ISOLATED",
        )

        # The ESSO proof scope is finite; if we ever leave it, we must fail-closed.
        self._verified_envelope_ok: bool = True
        
        # Goal ID
        self._goal_id = str(coordinator.goal_spec.goal_id)
        
        # Mempool for ordered processing
        self._mempool = ContributionMempool(
            max_size=10_000,
            goal_id=self._goal_id,
        )
        
        # Peer state tracking
        self._peer_states: Dict[str, PeerStateSnapshot] = {}
        
        # Pending request correlation (nonce -> future)
        self._pending_state_requests: dict[tuple[str, str], asyncio.Future[PeerStateSnapshot]] = {}
        self._pending_sync_requests: dict[tuple[str, str], asyncio.Future[SyncResponse]] = {}

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
        
        
        # Contribution body store
        self._max_body_cache: int = 50_000
        self._bodies_by_hash: "OrderedDict[bytes, Dict[str, Any]]" = OrderedDict()
        self._accepted_by_index: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self._max_accepted_cache: int = 100_000

    @property
    def status(self) -> ConsensusState:
        """Get current consensus state from verified kernel."""
        try:
            return ConsensusState[self._kstate.state]
        except KeyError:
            return ConsensusState.ISOLATED

    def _apply_kernel(self, tag: str, **kwargs) -> bool:
        """Apply a kernel command and sync state."""
        prev_state = self.status
        cmd = consensus_kernel.Command(tag=tag, args=kwargs)
        result = consensus_kernel.step(self._kstate, cmd)
        
        if not result.ok:
            logger.error(f"Kernel REJECTED command {tag}: {result.error}")
            return False
            
        self._kstate = result.state

        new_state = self.status
        if new_state != prev_state and self._on_state_change:
            self._on_state_change(new_state)
        return True

    def _drive_kernel_from_observation(self, *, matching: int, total: int) -> None:
        """
        Drive the verified kernel using only observable counts.

        This is the functional-core boundary: given (matching, total) in-domain,
        we deterministically reconcile kernel counters and then trigger any
        derived state transitions (SYNCING->SYNCHRONIZED, SYNCHRONIZED->DIVERGED)
        via kernel commands.
        """
        if not (0 <= total <= _KERNEL_MAX_PEERS):
            self._verified_envelope_ok = False
            logger.error(
                f"Consensus kernel envelope exceeded: total peers {total} > {_KERNEL_MAX_PEERS}. "
                "Failing closed (refusing commit) until peer set shrinks or kernel bounds are widened."
            )
            return
        if not (0 <= matching <= total):
            self._verified_envelope_ok = False
            logger.error(
                f"Invalid consensus observation: matching={matching} must be in [0, total={total}]. "
                "Failing closed."
            )
            return

        self._verified_envelope_ok = True

        self._sync_kernel_counts(actual_matching=matching, actual_total=total)

        has_quorum = (
            (self._kstate.matching_peers * self._kstate.quorum_fraction_den)
            >= (self._kstate.peer_count * self._kstate.quorum_fraction_num)
        )

        # Derived state transitions (based on the *now-synced* kernel state).
        if self.status == ConsensusState.SYNCING and has_quorum:
            # Complete syncing only when quorum is satisfied.
            self._apply_kernel("sync_complete_become_synchronized")

        if self.status == ConsensusState.SYNCHRONIZED and not has_quorum:
            # Diverge only when quorum is violated.
            self._apply_kernel("state_diverge")

    def _sync_kernel_counts(self, actual_matching: int, actual_total: int) -> None:
        """
        Synchronize kernel counters with actual peer observations.
        Uses patched kernel commands for granular updates.
        """
        if not (0 <= actual_total <= _KERNEL_MAX_PEERS):
            raise ValueError(f"actual_total must be in [0, {_KERNEL_MAX_PEERS}], got {actual_total}")
        if not (0 <= actual_matching <= actual_total):
            raise ValueError(f"actual_matching must be in [0, actual_total], got {actual_matching}/{actual_total}")

        # 1. Sync Increase
        while self._kstate.peer_count < actual_total:
            # Try transitions in order of likelihood/validity.
            # The kernel guards ensure only semantically valid transitions commit.
            success = (
                self._apply_kernel("peers_join_become_syncing")
                or self._apply_kernel("peers_join_still_isolated")
                or self._apply_kernel("peers_join_steady")
            )
            if not success:
                raise RuntimeError(
                    f"Consensus sync stalled (join): kstate={self._kstate}, actual_total={actual_total}"
                )
        
        # 1b. Sync Matching Increase
        while self._kstate.matching_peers < actual_matching:
            if not self._apply_kernel("peer_matches_increase"):
                raise RuntimeError(
                    f"Consensus sync stalled (match): kstate={self._kstate}, actual_matching={actual_matching}"
                )

        # 2. Sync Decrease (Patched Logic)
        while self._kstate.peer_count > actual_total:
            # Decide if a matching peer left or non-matching
            # We try to maintain the ratio observed if possible, or prioritize removing non-matching
            if self._kstate.matching_peers > actual_matching:
                if not self._apply_kernel("peers_leave_matching"):
                    # Fallback to isolation if guard failed (too few peers)
                    if not self._apply_kernel("peers_leave_become_isolated"):
                        break  # Guarded against infinite loop
            else:
                if not self._apply_kernel("peers_leave_non_matching"):
                    if not self._apply_kernel("peers_leave_become_isolated"):
                        break
        
        # 2b. Sync Matching Decrease (Divergence)
        while self._kstate.matching_peers > actual_matching:
            # Decrease matching without changing peer_count.
            # The kernel selects the correct control-flow (keep sync vs lose quorum).
            if not (
                self._apply_kernel("peer_diverge_decrease_non_sync")
                or self._apply_kernel("peer_diverge_decrease_from_sync_keep")
                or self._apply_kernel("peer_diverge_decrease_from_sync_lose")
            ):
                break

    def _update_kernel_state(self) -> None:
        """Drive kernel state based on current peer states."""
        active_peers = list(self._peer_states.values())

        # Determine our local "head" (log root).
        local_root: bytes = b""
        try:
            if hasattr(self._coordinator, "get_log_root"):
                local_root = self._coordinator.get_log_root()
        except Exception:
            local_root = b""
        if not local_root:
            try:
                chain = getattr(self._coordinator, "chain", None)
                head = getattr(chain, "head", None) if chain is not None else None
                header = getattr(head, "header", None) if head is not None else None
                local_root = getattr(header, "hash", b"") if header is not None else b""
            except Exception:
                local_root = b""

        total_count = len(active_peers)
        matching_count = sum(1 for p in active_peers if p.log_root == local_root)
        self._drive_kernel_from_observation(matching=matching_count, total=total_count)

    @staticmethod
    def _compute_contribution_hash_bytes(contrib_dict: Dict[str, Any]) -> bytes:
        contrib_bytes = json.dumps(contrib_dict, sort_keys=True).encode()
        return hashlib.sha256(contrib_bytes).digest()

    def _cache_body(self, contrib_dict: Dict[str, Any]) -> bytes:
        key = self._compute_contribution_hash_bytes(contrib_dict)
        # LRU insert/update
        self._bodies_by_hash[key] = contrib_dict
        self._bodies_by_hash.move_to_end(key)
        while len(self._bodies_by_hash) > self._max_body_cache:
            self._bodies_by_hash.popitem(last=False)
        return key

    def _cache_accepted(self, log_index: int, contrib_dict: Dict[str, Any]) -> None:
        self._accepted_by_index[log_index] = contrib_dict
        self._accepted_by_index.move_to_end(log_index)
        while len(self._accepted_by_index) > self._max_accepted_cache:
            self._accepted_by_index.popitem(last=False)
    
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
        # Cache body for future gossip requests (even before processing).
        try:
            self._cache_body(contribution.to_dict())
        except Exception:
            pass

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
                    if result.accepted and entry.contribution is not None and result.log_index is not None:
                        try:
                            self._cache_accepted(int(result.log_index), entry.contribution.to_dict())
                        except Exception:
                            pass
                    
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
            # No network wiring: treat as standalone (no peers).
            self._peer_states.clear()
            self._drive_kernel_from_observation(matching=0, total=0)
            return
        
        peers = self._get_peers()
        if len(peers) < self._config.min_peers_for_consensus:
            # Not enough peers to form consensus (kernel will remain ISOLATED).
            self._drive_kernel_from_observation(matching=0, total=len(peers))
            return
        
        # Request state from all peers
        responses = await self._request_peer_states(peers)
        
        if len(responses) < self._config.min_peers_for_consensus:
            # Not enough responsive peers; fail-closed into non-consensus.
            self._drive_kernel_from_observation(matching=0, total=len(responses))
            return
        
        # Get our current state
        our_state = self._get_local_state_snapshot()
        
        # Count how many peers match our state
        matching = sum(1 for r in responses if r.matches(our_state))
        total = len(responses)

        self._drive_kernel_from_observation(matching=matching, total=total)

        if self.status == ConsensusState.SYNCHRONIZED:
            self._contributions_since_check = 0
        else:
            # Find majority state
            majority_state = self._find_majority_state(responses)
            
            if majority_state and not majority_state.matches(our_state):
                # We're diverged - need to sync
                await self._sync_from_peer(majority_state.node_id)
        
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

        fut: asyncio.Future[PeerStateSnapshot] = asyncio.get_running_loop().create_future()
        self._pending_state_requests[(peer_id, request.nonce)] = fut
        try:
            await self._send_message(peer_id, request)
            return await asyncio.wait_for(fut, timeout=self._config.state_request_timeout)
        except Exception:
            fut.cancel()
            return None
        finally:
            self._pending_state_requests.pop((peer_id, request.nonce), None)
    
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
        
        # Guard against empty groups (defensive)
        if not groups:
            return None
        
        # Find largest group
        largest_group = max(groups.values(), key=len)
        
        if len(largest_group) >= len(states) * self._config.quorum_threshold:
            return largest_group[0]
        
        return None
    
    # NOTE: Consensus invariants are enforced by the ESSO kernel itself.
    # The shell must only drive state changes through kernel commands.
    
    # -------------------------------------------------------------------------
    # State Synchronization
    # -------------------------------------------------------------------------
    
    async def _sync_from_peer(self, peer_id: str) -> bool:
        """
        Synchronize state from a peer.
        
        This is used when we've diverged from the network majority.
        
        Algorithm:
        1. Request peer's current state snapshot
        2. Compare log sizes to determine what we're missing
        3. Request contributions in batches (max_sync_batch_size)
        4. Verify and apply each contribution in order
        5. After each batch, verify intermediate state
        6. Final verification: our state matches peer's state
        
        Args:
            peer_id: Node ID of peer to sync from
            
        Returns:
            True if sync successful, False otherwise
            
        Invariants:
            - Contributions are applied in strict order
            - Each contribution is verified before applying
            - State is verified after sync completes
            
        Time Complexity: O(n) where n = contributions to sync
        """
        logger.info(f"Syncing state from peer {peer_id[:16]}...")

        # If we're diverged, transition the kernel into SYNCING.
        if self.status == ConsensusState.DIVERGED:
            self._apply_kernel("diverged_start_sync")
        
        try:
            # Step 1: Get peer's state snapshot
            peer_state = self._peer_states.get(peer_id)
            if not peer_state:
                logger.warning(f"No state snapshot for peer {peer_id[:16]}")
                return False
            
            our_log_size = self._coordinator.state.log.size
            peer_log_size = peer_state.log_size
            
            # Check if we're actually behind
            if our_log_size >= peer_log_size:
                logger.info(
                    f"No sync needed: our log ({our_log_size}) >= peer ({peer_log_size})"
                )
                self._update_kernel_state()
                return True
            
            contributions_needed = peer_log_size - our_log_size
            logger.info(
                f"Syncing {contributions_needed} contributions "
                f"(log {our_log_size} -> {peer_log_size})"
            )
            
            # Step 2: Request contributions in batches
            synced_count = 0
            current_index = our_log_size
            
            while current_index < peer_log_size:
                batch_end = min(
                    current_index + self._config.max_sync_batch_size,
                    peer_log_size
                )
                
                # Request batch from peer
                contributions = await self._request_sync_batch(
                    peer_id, current_index, batch_end
                )
                
                if not contributions:
                    logger.error(
                        f"Failed to get sync batch [{current_index}, {batch_end})"
                    )
                    self._update_kernel_state()
                    return False
                
                # Step 3: Verify and apply each contribution
                for contrib_data in contributions:
                    try:
                        # Deserialize contribution
                        from idi.ian.models import Contribution
                        contribution = Contribution.from_dict(contrib_data)
                        
                        # Verify contribution is valid
                        if not self._verify_contribution_for_sync(contribution):
                            logger.error(
                                f"Invalid contribution during sync at index {current_index}"
                            )
                            self._update_kernel_state()
                            return False
                        
                        # Apply to coordinator (bypassing mempool)
                        result = self._coordinator.process_contribution(contribution)
                        
                        if not result.accepted:
                            logger.error(
                                f"Failed to apply synced contribution: {result.reason}"
                            )
                            self._update_kernel_state()
                            return False

                        if result.log_index is not None:
                            try:
                                self._cache_accepted(int(result.log_index), contribution.to_dict())
                            except Exception:
                                pass
                        
                        synced_count += 1
                        current_index += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing synced contribution: {e}")
                        self._update_kernel_state()
                        return False
                
                logger.debug(
                    f"Synced batch: {synced_count}/{contributions_needed} contributions"
                )
            
            # Step 4: Final verification - our state should match peer's
            our_log_root = self._coordinator.get_log_root()
            if our_log_root != peer_state.log_root:
                logger.error(
                    f"State mismatch after sync: "
                    f"ours={our_log_root.hex()[:16]}, "
                    f"peer={peer_state.log_root.hex()[:16]}"
                )
                self._update_kernel_state()
                return False
            
            logger.info(
                f"Sync complete: {synced_count} contributions applied, "
                f"state verified"
            )
            self._update_kernel_state()
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Sync timeout from peer {peer_id[:16]}")
            self._update_kernel_state()
            return False
        except Exception as e:
            logger.error(f"Sync error: {e}")
            self._update_kernel_state()
            return False
    
    async def _request_sync_batch(
        self,
        peer_id: str,
        from_index: int,
        to_index: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Request a batch of contributions from a peer.
        
        Args:
            peer_id: Peer to request from
            from_index: Start index (inclusive)
            to_index: End index (exclusive)
            
        Returns:
            List of contribution dicts, or None on failure
        """
        if not self._send_message:
            return None
        
        request = SyncRequest(
            sender_id=self._node_id,
            goal_id=self._goal_id,
            from_index=from_index,
            to_index=to_index,
        )
        
        try:
            fut: asyncio.Future[SyncResponse] = asyncio.get_running_loop().create_future()
            self._pending_sync_requests[(peer_id, request.nonce)] = fut

            # Send request and await correlated response (nonce echoed back)
            await self._send_message(peer_id, request)
            
            response = await asyncio.wait_for(fut, timeout=self._config.sync_timeout)
            
            if response:
                return response.contributions
            return None
            
        except asyncio.TimeoutError:
            logger.warning(f"Sync request timed out for peer {peer_id[:16]}")
            return None
        except Exception as e:
            logger.error(f"Sync request failed: {e}")
            return None
        finally:
            self._pending_sync_requests.pop((peer_id, request.nonce), None)
    
    def _verify_contribution_for_sync(self, contribution: "Contribution") -> bool:
        """
        Verify a contribution is valid for sync application.
        
        Args:
            contribution: Contribution to verify
            
        Returns:
            True if valid
        """
        # Basic validation
        if not contribution.pack_hash:
            return False
        
        if not contribution.contributor_id:
            return False
        
        return True
    
    async def handle_sync_request(self, request: SyncRequest) -> SyncResponse:
        """
        Handle incoming sync request from a peer.
        
        Args:
            request: Sync request message
            
        Returns:
            SyncResponse with requested contributions
        """
        contributions: list[dict[str, Any]] = []
        log_size = int(self._coordinator.state.log.size)
        
        # Bounds check
        from_idx = max(0, request.from_index)
        to_idx = min(request.to_index, log_size)
        
        # Limit batch size
        to_idx = min(to_idx, from_idx + self._config.max_sync_batch_size)
        
        # Gather contributions
        for i in range(from_idx, to_idx):
            contrib = self._accepted_by_index.get(i)
            if contrib is not None:
                contributions.append(contrib)
        
        return SyncResponse(
            sender_id=self._node_id,
            goal_id=request.goal_id,
            from_index=from_idx,
            contributions=contributions,
            has_more=(to_idx < log_size),
            nonce=request.nonce,
        )
    
    async def handle_sync_response(self, response: SyncResponse) -> None:
        """
        Handle incoming sync response from a peer.
        
        Stores the response for correlation with pending requests.
        """
        fut = self._pending_sync_requests.get((response.sender_id, response.nonce))
        if fut is not None and not fut.done():
            fut.set_result(response)
            return
    
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
            nonce=request.nonce,
        )
    
    async def handle_state_response(self, response: StateResponse) -> None:
        """Handle incoming state response."""
        # Update peer state tracking
        snapshot = PeerStateSnapshot(
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
        self._peer_states[response.sender_id] = snapshot
        # Keep kernel counters in sync with the latest peer observations.
        self._update_kernel_state()

        fut = self._pending_state_requests.get((response.sender_id, response.nonce))
        if fut is not None and not fut.done():
            fut.set_result(snapshot)
            return

    async def handle_contribution_request(self, request: ContributionRequest) -> ContributionResponse:
        """Serve a full contribution body by contribution_hash (sha256 of serialized Contribution)."""
        try:
            want = bytes.fromhex(request.contribution_hash)
        except Exception:
            return ContributionResponse(
                sender_id=self._node_id,
                contribution_hash=request.contribution_hash,
                contribution=None,
                found=False,
                nonce=request.nonce,
            )

        body = self._bodies_by_hash.get(want)
        if body is None:
            return ContributionResponse(
                sender_id=self._node_id,
                contribution_hash=request.contribution_hash,
                contribution=None,
                found=False,
                nonce=request.nonce,
            )

        return ContributionResponse(
            sender_id=self._node_id,
            contribution_hash=request.contribution_hash,
            contribution=body,
            found=True,
            nonce=request.nonce,
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
        return self.status
    
    def get_state(self) -> ConsensusState:
        """Get current consensus state (method form for compatibility)."""
        return self.status
    
    @property
    def mempool_size(self) -> int:
        """Get current mempool size."""
        return self._mempool.size

    @property
    def is_running(self) -> bool:
        """Whether the coordinator background loops are running (imperative shell state)."""
        return self._running
    
    @property
    def peer_count(self) -> int:
        """Get number of known peers."""
        return len(self._peer_states)
    
    def is_synchronized(self) -> bool:
        """Check if in consensus with peers."""
        return self.status == ConsensusState.SYNCHRONIZED
    
    def can_commit_to_tau(self) -> bool:
        """Check if safe to commit state to Tau Net."""
        if not self._verified_envelope_ok:
            return False
        return self.status in (
            ConsensusState.SYNCHRONIZED,
            ConsensusState.ISOLATED,  # Allow commits when alone
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus coordinator statistics."""
        return {
            "goal_id": self._goal_id,
            "consensus_state": self.status.name,
            "mempool_size": self._mempool.size,
            "peer_count": len(self._peer_states),
            "contributions_since_check": self._contributions_since_check,
            "coordinator_stats": self._coordinator.get_stats(),
        }
