"""
IAN Decentralized Node - Unified L2 node implementation.

Brings together:
1. Consensus Coordinator - Multi-node state consensus
2. Contribution Ordering - Deterministic ordering and mempool
3. Fraud Proofs - Detection and verification
4. Economic Security - Bonding and slashing
5. Evaluation Quorum - Distributed evaluation
6. Tau Bridge - L1 integration

This is the main entry point for running a fully decentralized IAN node.

Usage:
    node = DecentralizedNode(
        goal_spec=goal_spec,
        identity=NodeIdentity.generate(),
        tau_sender=TauNetSender(...),
    )
    await node.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .consensus import ConsensusCoordinator, ConsensusConfig, ConsensusState
from .ordering import ContributionMempool, ContributionGossip
from .fraud import FraudProofGenerator, FraudProofVerifier, ChallengeManager
from .economics import EconomicManager, EconomicConfig
from .evaluation import EvaluationQuorumManager, EvaluationQuorumConfig
from .node import NodeIdentity, NodeInfo, NodeCapabilities
from .discovery import SeedNodeDiscovery
from .protocol import Message, MessageType
from .production import (
    TaskSupervisor,
    NodeMetrics,
    HealthServer,
    HealthCheck,
    HealthStatus,
    PeerScoreManager,
    GracefulShutdown,
    NodeState,
    backoff_with_jitter,
)

if TYPE_CHECKING:
    from idi.ian.coordinator import IANCoordinator
    from idi.ian.models import GoalSpec, Contribution
    from idi.ian.tau_bridge import TauBridge

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DecentralizedNodeConfig:
    """Configuration for decentralized node."""
    
    # Network
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    seed_addresses: List[str] = field(default_factory=list)
    max_peers: int = 50
    
    # Consensus
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    
    # Economics
    economics: EconomicConfig = field(default_factory=EconomicConfig)
    
    # Evaluation
    evaluation: EvaluationQuorumConfig = field(default_factory=EvaluationQuorumConfig)
    
    # Tau integration
    tau_commit_interval: float = 300.0  # 5 minutes
    tau_commit_threshold: int = 100  # Or after 100 contributions
    
    # Node capabilities
    accept_contributions: bool = True
    serve_evaluations: bool = False
    commit_to_tau: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Production
    health_port: int = 8080
    enable_health_server: bool = True
    state_persist_path: Optional[str] = None  # Path for state persistence
    peer_scores_path: Optional[str] = None  # Path for peer scores
    min_peers_for_ready: int = 3
    max_sync_lag_for_ready: int = 10


# =============================================================================
# Decentralized Node
# =============================================================================

class DecentralizedNode:
    """
    Fully decentralized IAN L2 node.
    
    Responsibilities:
    1. Process contributions in deterministic order
    2. Maintain consensus with peer nodes
    3. Detect and prove fraud
    4. Commit state to Tau Net (if authorized)
    5. Participate in distributed evaluation
    
    Lifecycle:
    1. Initialize with goal spec and identity
    2. Connect to network and discover peers
    3. Sync state from peers if needed
    4. Process contributions and maintain state
    5. Periodically commit to Tau Net
    """
    
    def __init__(
        self,
        goal_spec: "GoalSpec",
        identity: NodeIdentity,
        coordinator: Optional["IANCoordinator"] = None,
        tau_bridge: Optional["TauBridge"] = None,
        config: Optional[DecentralizedNodeConfig] = None,
    ):
        """
        Initialize decentralized node.
        
        Args:
            goal_spec: Goal specification this node serves
            identity: Node cryptographic identity
            coordinator: Base coordinator (created if None)
            tau_bridge: Tau Net bridge (created if None)
            config: Node configuration
        """
        self._config = config or DecentralizedNodeConfig()
        self._identity = identity
        self._goal_spec = goal_spec
        self._goal_id = str(goal_spec.goal_id)
        
        # Create coordinator if not provided
        if coordinator is None:
            from idi.ian.coordinator import IANCoordinator
            coordinator = IANCoordinator(goal_spec)
        
        self._base_coordinator = coordinator
        
        # Create Tau bridge if not provided
        if tau_bridge is None:
            from idi.ian.tau_bridge import TauBridge
            tau_bridge = TauBridge()
        
        self._tau_bridge = tau_bridge
        
        # Initialize components
        self._consensus = ConsensusCoordinator(
            coordinator=self._base_coordinator,
            node_id=identity.node_id,
            config=self._config.consensus,
        )
        
        self._economics = EconomicManager(
            node_id=identity.node_id,
            config=self._config.economics,
        )
        
        self._evaluation = EvaluationQuorumManager(
            node_id=identity.node_id,
            config=self._config.evaluation,
        )
        
        self._fraud_generator = FraudProofGenerator(
            node_id=identity.node_id,
            coordinator=self._base_coordinator,
        )
        
        self._fraud_verifier = FraudProofVerifier()
        
        self._challenges = ChallengeManager(
            node_id=identity.node_id,
        )
        
        # Discovery
        self._discovery = SeedNodeDiscovery(
            identity=identity,
            seed_addresses=self._config.seed_addresses,
            max_peers=self._config.max_peers,
        )
        
        # Production infrastructure
        self._supervisor = TaskSupervisor(name=f"node-{identity.node_id[:8]}")
        self._metrics = NodeMetrics(
            node_id=identity.node_id,
            goal_id=self._goal_id,
        )
        self._health_server = HealthServer(
            port=self._config.health_port,
            metrics=self._metrics,
        ) if self._config.enable_health_server else None
        
        self._peer_scores = PeerScoreManager(
            persist_path=self._config.peer_scores_path,
        )
        self._shutdown = GracefulShutdown()
        
        # State
        self._running = False
        self._last_tau_commit = 0.0
        self._contributions_since_commit = 0
        self._start_time = 0.0
        
        # Peer connections (would be managed by transport layer)
        self._connected_peers: Dict[str, NodeInfo] = {}
        
        # Load persisted state if available
        self._load_persisted_state()
        
        # Setup callbacks
        self._setup_callbacks()
        self._setup_health_checks()

        # Register info/peers providers for health server
        if self._health_server:
            self._health_server.set_info_provider(self._build_node_info)
            self._health_server.set_peers_provider(self._build_peers_info)
    
    def _setup_callbacks(self) -> None:
        """Wire up component callbacks."""
        # Consensus callbacks
        self._consensus.set_callbacks(
            get_peers=self._get_peer_ids,
            send_message=self._send_message,
            on_state_change=self._on_consensus_state_change,
        )
        
        # Discovery callbacks
        self._discovery.set_callbacks(
            on_peer_discovered=self._on_peer_discovered,
            on_peer_lost=self._on_peer_lost,
            send_message=self._send_message_to_address,
        )
        
        # Challenge callbacks
        self._challenges.set_tau_callback(self._submit_challenge_to_tau)
        
        # Economics callbacks
        self._economics.set_tx_callback(self._submit_tx_to_tau)
    
    def _setup_health_checks(self) -> None:
        """Register health checks for readiness endpoint."""
        if not self._health_server:
            return
        
        # Sync status check
        def check_sync() -> HealthCheck:
            lag = self.get_sync_lag()
            if lag > self._config.max_sync_lag_for_ready:
                return HealthCheck(
                    name="sync",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Sync lag: {lag} blocks",
                    details={"lag": lag},
                )
            elif lag > self._config.max_sync_lag_for_ready // 2:
                return HealthCheck(
                    name="sync",
                    status=HealthStatus.DEGRADED,
                    message=f"Sync lag: {lag} blocks",
                    details={"lag": lag},
                )
            return HealthCheck(
                name="sync",
                status=HealthStatus.HEALTHY,
                message="Synced",
                details={"lag": lag},
            )
        
        # Peer count check
        def check_peers() -> HealthCheck:
            count = len(self._connected_peers)
            if count < self._config.min_peers_for_ready:
                return HealthCheck(
                    name="peers",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Low peer count: {count}",
                    details={"count": count, "min": self._config.min_peers_for_ready},
                )
            return HealthCheck(
                name="peers",
                status=HealthStatus.HEALTHY,
                message=f"{count} peers connected",
                details={"count": count},
            )
        
        # Consensus check
        def check_consensus() -> HealthCheck:
            state = self._consensus.get_state()
            if state == ConsensusState.SYNCING:
                return HealthCheck(
                    name="consensus",
                    status=HealthStatus.DEGRADED,
                    message="Consensus syncing",
                )
            elif state == ConsensusState.STOPPED:
                return HealthCheck(
                    name="consensus",
                    status=HealthStatus.UNHEALTHY,
                    message="Consensus stopped",
                )
            return HealthCheck(
                name="consensus",
                status=HealthStatus.HEALTHY,
                message=f"Consensus: {state.name}",
            )
        
        self._health_server.register_check("sync", check_sync)
        self._health_server.register_check("peers", check_peers)
        self._health_server.register_check("consensus", check_consensus)
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from disk if available."""
        if not self._config.state_persist_path:
            return
        
        state = NodeState.load(self._config.state_persist_path)
        if state:
            logger.info(
                f"Loaded persisted state: log_size={state.log_size}, "
                f"last_commit={state.last_tau_commit}"
            )
            self._last_tau_commit = state.last_tau_commit
            self._contributions_since_commit = state.contributions_since_commit
        
        # Load peer scores
        self._peer_scores.load()
    
    async def _persist_state(self) -> None:
        """Persist critical state to disk."""
        if not self._config.state_persist_path:
            return
        
        state = NodeState(
            node_id=self._identity.node_id,
            goal_id=self._goal_id,
            last_log_root=self._base_coordinator.get_log_root().hex(),
            log_size=self._base_coordinator.state.log.size,
            last_tau_commit=self._last_tau_commit,
            contributions_since_commit=self._contributions_since_commit,
        )
        state.save(self._config.state_persist_path)
        
        # Save peer scores
        self._peer_scores.save()
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    async def start(self) -> None:
        """
        Start the decentralized node.
        
        Sets up signal handlers for graceful shutdown, starts all components,
        and spawns supervised background tasks.
        """
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        
        logger.info(
            f"Starting decentralized node {self._identity.node_id[:16]}... "
            f"for goal {self._goal_id}"
        )
        
        # Setup graceful shutdown handlers
        self._shutdown.register_handler(self._persist_state)
        self._shutdown.register_handler(self._consensus.stop)
        self._shutdown.register_handler(self._discovery.stop)
        if self._health_server:
            self._shutdown.register_handler(self._health_server.stop)
        self._shutdown.setup_signals()
        
        # Start health server
        if self._health_server:
            await self._health_server.start()
        
        # Start discovery
        await self._discovery.start()
        
        # Start consensus
        await self._consensus.start()
        
        # Spawn supervised background tasks
        self._supervisor.spawn(
            lambda: self._tau_commit_loop(),
            name="tau_commit",
            restart_on_failure=True,
            restart_delay=5.0,
        )
        self._supervisor.spawn(
            lambda: self._peer_state_check_loop(),
            name="peer_state_check",
            restart_on_failure=True,
            restart_delay=10.0,
        )
        self._supervisor.spawn(
            lambda: self._challenge_finalization_loop(),
            name="challenge_finalization",
            restart_on_failure=True,
            restart_delay=5.0,
        )
        self._supervisor.spawn(
            lambda: self._metrics_update_loop(),
            name="metrics_update",
            restart_on_failure=True,
            restart_delay=1.0,
        )
        self._supervisor.spawn(
            lambda: self._peer_score_decay_loop(),
            name="peer_score_decay",
            restart_on_failure=True,
            restart_delay=60.0,
        )
        
        logger.info(f"Node started: {self._identity.node_id[:16]}...")
    
    async def stop(self) -> None:
        """
        Stop the decentralized node gracefully.
        
        Persists state, stops all tasks, and cleans up resources.
        """
        logger.info(f"Stopping node {self._identity.node_id[:16]}...")
        self._running = False
        
        # Persist state before shutdown
        await self._persist_state()
        
        # Shutdown supervisor (cancels all tasks)
        await self._supervisor.shutdown(timeout=30.0)
        
        # Stop health server
        if self._health_server:
            await self._health_server.stop()
        
        # Stop components
        await self._consensus.stop()
        await self._discovery.stop()
        
        logger.info(f"Node stopped: {self._identity.node_id[:16]}...")
    
    # -------------------------------------------------------------------------
    # Contribution Handling
    # -------------------------------------------------------------------------
    
    async def submit_contribution(
        self,
        contribution: "Contribution",
    ) -> Tuple[bool, str]:
        """
        Submit a contribution to the network.
        
        The contribution will be:
        1. Added to local mempool
        2. Gossiped to peers
        3. Processed in deterministic order
        
        Args:
            contribution: Contribution to submit
            
        Returns:
            (success, reason)
        """
        if not self._config.accept_contributions:
            return False, "this node does not accept contributions"
        
        # Submit to consensus coordinator
        success, reason = await self._consensus.submit_contribution(contribution)
        
        if success:
            self._contributions_since_commit += 1
        
        return success, reason
    
    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------
    
    async def handle_message(
        self,
        message: Message,
        from_peer: str,
    ) -> Optional[Message]:
        """
        Handle incoming P2P message.
        
        Args:
            message: Incoming message
            from_peer: Node ID of sender
            
        Returns:
            Response message if applicable
        """
        # Bind or propagate trace ID for this message
        try:
            from .resilience import set_correlation_id, new_correlation_id
            # Use message nonce as trace seed if available, else generate new
            if hasattr(message, 'nonce') and message.nonce:
                set_correlation_id(message.nonce[:8])
            else:
                new_correlation_id()
        except ImportError:
            pass  # Tracing not available
        
        try:
            if message.type == MessageType.CONTRIBUTION_ANNOUNCE:
                from .protocol import ContributionAnnounce
                announce = message  # Already parsed
                need_full = await self._consensus.handle_contribution_announce(
                    announce, from_peer
                )
                if need_full:
                    # Request full contribution
                    from .protocol import ContributionRequest
                    return ContributionRequest(
                        sender_id=self._identity.node_id,
                        contribution_hash=announce.contribution_hash,
                    )
            
            elif message.type == MessageType.CONTRIBUTION_RESPONSE:
                from .protocol import ContributionResponse
                await self._consensus.handle_contribution_response(message, from_peer)
            
            elif message.type == MessageType.STATE_REQUEST:
                from .protocol import StateRequest
                return await self._consensus.handle_state_request(message)
            
            elif message.type == MessageType.STATE_RESPONSE:
                await self._consensus.handle_state_response(message)
            
            elif message.type == MessageType.PEER_EXCHANGE:
                self._discovery.handle_peer_exchange(message)
            
            elif message.type == MessageType.PONG:
                self._discovery.handle_pong(message)
            
        except Exception as e:
            logger.error(f"Error handling message from {from_peer[:16]}...: {e}")
        
        return None
    
    # -------------------------------------------------------------------------
    # Tau Net Integration
    # -------------------------------------------------------------------------
    
    async def _tau_commit_loop(self) -> None:
        """Background loop for Tau Net commits."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Check every 10s
                
                if not self._config.commit_to_tau:
                    continue
                
                # Check if commit needed
                should_commit = (
                    time.time() - self._last_tau_commit >= self._config.tau_commit_interval or
                    self._contributions_since_commit >= self._config.tau_commit_threshold
                )
                
                if should_commit and self._consensus.can_commit_to_tau():
                    # Generate trace ID for this commit operation
                    try:
                        from .resilience import new_correlation_id, get_correlation_id
                        trace_id = new_correlation_id()
                        logger.info(
                            f"Initiating Tau commit (trace={trace_id}, "
                            f"contributions={self._contributions_since_commit})"
                        )
                    except ImportError:
                        pass  # Tracing not available
                    
                    await self._commit_to_tau()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tau commit loop error: {e}")
    
    async def _commit_to_tau(self) -> Tuple[bool, str]:
        """
        Commit current state to Tau Net.
        
        Returns:
            (success, tx_hash_or_error)
        """
        # Check if we're authorized (have bond)
        can_commit, reason = self._economics.can_commit(
            self._identity.node_id,
            self._goal_id,
        )
        
        if not can_commit:
            logger.debug(f"Cannot commit to Tau: {reason}")
            return False, reason
        
        # Get state
        log_root = self._base_coordinator.get_log_root()
        log_size = self._base_coordinator.state.log.size
        lb_root = self._base_coordinator.get_leaderboard_root()
        lb_size = len(self._base_coordinator.state.leaderboard)
        
        # Commit via bridge
        success, result = self._tau_bridge.commit_log(
            goal_id=self._goal_spec.goal_id,
            log_root=log_root,
            log_size=log_size,
            leaderboard_root=lb_root,
            leaderboard_size=lb_size,
        )
        
        if success:
            self._last_tau_commit = time.time()
            self._contributions_since_commit = 0
            self._economics.record_commit(self._identity.node_id, self._goal_id)
            
            logger.info(f"Committed to Tau: log_root={log_root.hex()[:16]}...")
        else:
            logger.error(f"Tau commit failed: {result}")
        
        return success, result
    
    def _submit_tx_to_tau(self, tx_data: bytes) -> Tuple[bool, str]:
        """Submit arbitrary transaction to Tau."""
        return self._tau_bridge.sender.send_tx(tx_data)
    
    def _submit_challenge_to_tau(self, proof_data: bytes) -> Tuple[bool, str]:
        """Submit fraud proof challenge to Tau."""
        # Wrap in challenge transaction
        import json
        tx = {
            "type": "IAN_CHALLENGE",
            "goal_id": self._goal_id,
            "proof_data": proof_data.hex(),
            "challenger_id": self._identity.node_id,
            "timestamp_ms": int(time.time() * 1000),
        }
        return self._tau_bridge.sender.send_tx(json.dumps(tx).encode())
    
    # -------------------------------------------------------------------------
    # Fraud Detection
    # -------------------------------------------------------------------------
    
    async def _peer_state_check_loop(self) -> None:
        """Background loop for detecting peer fraud."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                await self._check_peer_states_for_fraud()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Peer state check error: {e}")
    
    async def _check_peer_states_for_fraud(self) -> None:
        """Check peer states for potential fraud and generate proofs."""
        our_log_root = self._base_coordinator.get_log_root()
        our_log_size = self._base_coordinator.state.log.size
        our_lb_root = self._base_coordinator.get_leaderboard_root()
        
        for peer_id, state in self._consensus._peer_states.items():
            # Skip if peer matches our state
            if state.log_root == our_log_root and state.leaderboard_root == our_lb_root:
                continue
            
            # Log root divergence at same size = potential fraud
            if state.log_size == our_log_size and state.log_root != our_log_root:
                logger.warning(
                    f"Fraud detected: peer {peer_id[:16]}... has different "
                    f"log root at size {state.log_size} "
                    f"(ours: {our_log_root.hex()[:16]}, theirs: {state.log_root.hex()[:16]})"
                )
                
                # Generate fraud proof
                proof = self._fraud_generator.generate_invalid_log_root_proof(
                    commit_hash=state.log_root,  # Use peer's root as commit hash
                    claimed_root=state.log_root,
                    goal_id=self._goal_id,
                )
                
                if proof:
                    # Verify locally before submitting
                    valid, reason = self._fraud_verifier.verify(proof)
                    if valid:
                        # Sign the proof
                        if self._identity.has_private_key():
                            proof.challenger_signature = self._identity.sign(
                                proof.signing_payload()
                            )
                        
                        # Submit to challenge queue
                        await self._challenges.submit_challenge(proof)
                        self._metrics.increment("fraud_proofs_generated")
                        logger.info(
                            f"Fraud proof queued for peer {peer_id[:16]}... "
                            f"(type: {proof.fraud_type.value})"
                        )
                    else:
                        logger.debug(f"Generated proof failed verification: {reason}")
                
                # Record negative event for peer
                self._peer_scores.record_event(peer_id, "invalid_message")
    
    async def _challenge_finalization_loop(self) -> None:
        """Background loop for finalizing challenges."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30s
                
                for proof in self._challenges.get_pending_challenges():
                    # Try to finalize if enough confirmations
                    success, result = await self._challenges.finalize_challenge(
                        proof.challenged_commit_hash
                    )
                    if success:
                        logger.info(f"Challenge finalized: {result}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Challenge finalization error: {e}")
    
    # -------------------------------------------------------------------------
    # Peer Management
    # -------------------------------------------------------------------------
    
    def _get_peer_ids(self) -> List[str]:
        """Get list of connected peer IDs."""
        return list(self._connected_peers.keys())
    
    async def _send_message(self, peer_id: str, message: Any) -> None:
        """Send message to peer by ID."""
        if peer_id not in self._connected_peers:
            raise ValueError(f"peer not connected: {peer_id}")
        
        # Sign message
        if hasattr(message, 'sender_id'):
            message.sender_id = self._identity.node_id
        
        if self._identity.has_private_key() and hasattr(message, 'signature'):
            self._identity.sign_message(message)
        
        # Would send via transport layer
        pass
    
    async def _send_message_to_address(self, address: str, message: Any) -> None:
        """Send message to peer by address."""
        # Would send via transport layer
        pass
    
    def _on_peer_discovered(self, info: NodeInfo) -> None:
        """Handle peer discovery."""
        self._connected_peers[info.node_id] = info
        logger.info(f"Peer connected: {info.node_id[:16]}...")
    
    def _on_peer_lost(self, node_id: str) -> None:
        """Handle peer disconnection."""
        self._connected_peers.pop(node_id, None)
        logger.info(f"Peer disconnected: {node_id[:16]}...")
    
    def _on_consensus_state_change(self, state: ConsensusState) -> None:
        """Handle consensus state change."""
        logger.info(f"Consensus state changed to: {state.name}")
    
    # -------------------------------------------------------------------------
    # Metrics & Monitoring
    # -------------------------------------------------------------------------
    
    async def _metrics_update_loop(self) -> None:
        """Background loop for updating metrics."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    def _update_metrics(self) -> None:
        """Update all metrics gauges."""
        # Log state
        self._metrics.set_gauge("log_size", self._base_coordinator.state.log.size)
        self._metrics.set_gauge(
            "leaderboard_size",
            len(self._base_coordinator.state.leaderboard),
        )
        
        # Network state
        self._metrics.set_gauge("peer_count", len(self._connected_peers))
        
        # Sync state
        self._metrics.set_gauge("sync_lag", self.get_sync_lag())
        
        # Uptime
        if self._start_time > 0:
            self._metrics.set_gauge("uptime_seconds", time.time() - self._start_time)
        
        # Peer score stats
        score_stats = self._peer_scores.get_stats()
        self._metrics.set_gauge("peer_score_avg", score_stats.get("avg_score", 0))
    
    async def _peer_score_decay_loop(self) -> None:
        """Background loop for decaying peer scores."""
        last_decay = time.time()
        while self._running:
            try:
                await asyncio.sleep(3600.0)  # Decay every hour
                
                hours = (time.time() - last_decay) / 3600.0
                self._peer_scores.decay_all(hours)
                last_decay = time.time()
                
                # Disconnect bad peers
                for peer_id in self._peer_scores.get_peers_to_disconnect():
                    logger.warning(f"Disconnecting low-score peer: {peer_id[:16]}...")
                    self._on_peer_lost(peer_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Peer score decay error: {e}")
    
    def get_sync_lag(self) -> int:
        """
        Get current sync lag (blocks behind network).
        
        Returns:
            Number of blocks behind the network head
        """
        # Get our log size
        our_size = self._base_coordinator.state.log.size
        
        # Get max peer log size
        max_peer_size = our_size
        for peer_id, state in self._consensus._peer_states.items():
            if state.log_size > max_peer_size:
                max_peer_size = state.log_size
        
        return max_peer_size - our_size
    
    # -------------------------------------------------------------------------
    # Node Info
    # -------------------------------------------------------------------------
    
    def get_node_info(self) -> NodeInfo:
        """Get this node's public info."""
        capabilities = NodeCapabilities(
            accepts_contributions=self._config.accept_contributions,
            serves_leaderboard=True,
            serves_log_proofs=True,
            goal_ids=[self._goal_id],
        )
        
        addresses = [
            f"tcp://{self._config.listen_address}:{self._config.listen_port}"
        ]
        
        return self._identity.create_node_info(
            addresses=addresses,
            capabilities=capabilities,
        )

    def _build_node_info(self) -> Dict[str, Any]:
        """Provider used by HealthServer for /info."""
        info = self.get_node_info().to_dict()
        info.update(
            {
                "consensus_state": self._consensus.get_state().name,
                "running": self._running,
                "goal_id": self._goal_id,
            }
        )
        return info

    def _build_peers_info(self) -> Dict[str, Any]:
        """Provider used by HealthServer for /peers."""
        peers: List[Dict[str, Any]] = []
        for node_id, node_info in self._connected_peers.items():
            entry = node_info.to_dict()
            score = self._peer_scores.get_score(node_id)
            entry["peer_score"] = {
                "score": score.score,
                "banned": score.is_banned(),
                "trusted": score.is_trusted(),
            }
            peers.append(entry)
        
        return {
            "goal_id": self._goal_id,
            "total": len(peers),
            "peers": peers,
            "stats": self._peer_scores.get_stats(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self._identity.node_id,
            "goal_id": self._goal_id,
            "consensus": self._consensus.get_stats(),
            "connected_peers": len(self._connected_peers),
            "last_tau_commit": self._last_tau_commit,
            "contributions_since_commit": self._contributions_since_commit,
            "economic_state": self._economics.to_dict(),
        }
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def node_id(self) -> str:
        return self._identity.node_id
    
    @property
    def goal_id(self) -> str:
        return self._goal_id
    
    @property
    def coordinator(self) -> "IANCoordinator":
        return self._base_coordinator
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def consensus_state(self) -> ConsensusState:
        return self._consensus.consensus_state


# =============================================================================
# Factory Functions
# =============================================================================

def create_decentralized_node(
    goal_spec: "GoalSpec",
    seed_addresses: Optional[List[str]] = None,
    identity_path: Optional[str] = None,
    config: Optional[DecentralizedNodeConfig] = None,
) -> DecentralizedNode:
    """
    Factory function to create a decentralized node.
    
    Args:
        goal_spec: Goal specification
        seed_addresses: Seed node addresses for discovery
        identity_path: Path to load/save identity
        config: Node configuration
        
    Returns:
        Configured DecentralizedNode
    """
    from pathlib import Path
    
    # Load or create identity
    if identity_path:
        path = Path(identity_path)
        if path.exists():
            identity = NodeIdentity.load(path)
        else:
            identity = NodeIdentity.generate()
            identity.save(path)
    else:
        identity = NodeIdentity.generate()
    
    # Create config with seeds
    if config is None:
        config = DecentralizedNodeConfig()
    
    if seed_addresses:
        config.seed_addresses = seed_addresses
    
    return DecentralizedNode(
        goal_spec=goal_spec,
        identity=identity,
        config=config,
    )


async def run_decentralized_node(
    goal_spec: "GoalSpec",
    seed_addresses: Optional[List[str]] = None,
    **config_kwargs,
) -> None:
    """
    Run a decentralized node until interrupted.
    
    This is a convenience function for running a node in a script.
    
    Args:
        goal_spec: Goal specification
        seed_addresses: Seed node addresses
        **config_kwargs: Additional config options
    """
    config = DecentralizedNodeConfig(**config_kwargs)
    
    node = create_decentralized_node(
        goal_spec=goal_spec,
        seed_addresses=seed_addresses,
        config=config,
    )
    
    await node.start()
    
    try:
        # Run until cancelled
        while node.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await node.stop()
