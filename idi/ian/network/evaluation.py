"""
IAN Decentralized Evaluation - Multi-evaluator quorum for trustless evaluation.

Provides:
1. Distributed evaluation requests across multiple evaluators
2. Quorum-based result consensus
3. Evaluation result verification
4. Economic incentives for evaluators

Design Principles:
- No single evaluator is trusted
- Results require quorum agreement
- Evaluators are economically incentivized
- Fraud proofs can challenge wrong evaluations

Security Model:
- Evaluators must stake bonds
- Disagreeing evaluators can be slashed
- Results within tolerance are accepted
- ZK proofs can provide trustless verification (future)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from idi.ian.models import AgentPack, GoalSpec, Metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvaluationQuorumConfig:
    """Configuration for decentralized evaluation."""
    
    # Quorum requirements
    min_evaluators: int = 3  # Minimum evaluators per request
    quorum_threshold: float = 0.67  # 2/3 must agree
    
    # Tolerance for metric agreement
    reward_tolerance: float = 0.05  # 5% tolerance
    risk_tolerance: float = 0.05
    complexity_tolerance: float = 0.05
    
    # Timeouts
    evaluation_timeout: float = 300.0  # 5 minutes
    response_timeout: float = 60.0  # Per-evaluator timeout
    
    # Economic
    evaluation_fee: int = 1_000_000  # 1 TAU per evaluation
    evaluator_stake: int = 100_000_000  # 100 TAU stake required
    
    # Retries
    max_retries: int = 2


# =============================================================================
# Evaluator Registry
# =============================================================================

class EvaluatorStatus(Enum):
    """Status of an evaluator."""
    ACTIVE = "active"
    BUSY = "busy"
    OFFLINE = "offline"
    SLASHED = "slashed"


@dataclass
class EvaluatorInfo:
    """Information about a registered evaluator."""
    node_id: str
    address: str  # Network address
    stake_amount: int
    status: EvaluatorStatus = EvaluatorStatus.ACTIVE
    registered_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Performance metrics
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    avg_latency_ms: float = 0.0
    disagreement_count: int = 0
    
    # Capabilities
    supported_goal_ids: List[str] = field(default_factory=list)
    max_concurrent: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "stake_amount": self.stake_amount,
            "status": self.status.value,
            "registered_at_ms": self.registered_at_ms,
            "evaluations_completed": self.evaluations_completed,
            "evaluations_failed": self.evaluations_failed,
            "avg_latency_ms": self.avg_latency_ms,
            "disagreement_count": self.disagreement_count,
            "supported_goal_ids": self.supported_goal_ids,
            "max_concurrent": self.max_concurrent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluatorInfo":
        return cls(
            node_id=data["node_id"],
            address=data["address"],
            stake_amount=data["stake_amount"],
            status=EvaluatorStatus(data["status"]),
            registered_at_ms=data.get("registered_at_ms", int(time.time() * 1000)),
            evaluations_completed=data.get("evaluations_completed", 0),
            evaluations_failed=data.get("evaluations_failed", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            disagreement_count=data.get("disagreement_count", 0),
            supported_goal_ids=data.get("supported_goal_ids", []),
            max_concurrent=data.get("max_concurrent", 5),
        )
    
    @property
    def reputation_score(self) -> float:
        """Calculate reputation score for evaluator selection."""
        total = self.evaluations_completed + self.evaluations_failed
        if total == 0:
            return 0.5  # Neutral for new evaluators
        
        success_rate = self.evaluations_completed / total
        disagreement_rate = self.disagreement_count / max(1, self.evaluations_completed)
        
        # Score: high success, low disagreement, low latency
        latency_factor = max(0, 1 - (self.avg_latency_ms / 60000))  # Penalize > 60s
        
        return 0.5 * success_rate + 0.3 * (1 - disagreement_rate) + 0.2 * latency_factor


# =============================================================================
# Evaluation Request/Response
# =============================================================================

@dataclass
class EvaluationRequest:
    """Request for distributed evaluation."""
    request_id: str  # Unique request ID
    goal_id: str
    agent_pack_hash: bytes
    agent_pack_data: bytes  # Serialized AgentPack
    seed: int
    requester_id: str
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "goal_id": self.goal_id,
            "agent_pack_hash": self.agent_pack_hash.hex(),
            "agent_pack_data": self.agent_pack_data.hex(),
            "seed": self.seed,
            "requester_id": self.requester_id,
            "timestamp_ms": self.timestamp_ms,
        }
    
    def signing_payload(self) -> bytes:
        """Get payload for signing."""
        return json.dumps(self.to_dict(), sort_keys=True).encode()


@dataclass
class EvaluationResponse:
    """Response from an evaluator."""
    request_id: str
    evaluator_id: str
    
    # Results
    success: bool
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    
    # Timing
    start_time_ms: int = 0
    end_time_ms: int = 0
    
    # Signature
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "evaluator_id": self.evaluator_id,
            "success": self.success,
            "metrics": self.metrics,
            "error": self.error,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "signature": self.signature.hex() if self.signature else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResponse":
        return cls(
            request_id=data["request_id"],
            evaluator_id=data["evaluator_id"],
            success=data["success"],
            metrics=data.get("metrics"),
            error=data.get("error"),
            start_time_ms=data.get("start_time_ms", 0),
            end_time_ms=data.get("end_time_ms", 0),
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None,
        )
    
    @property
    def latency_ms(self) -> int:
        """Get evaluation latency."""
        if self.end_time_ms and self.start_time_ms:
            return self.end_time_ms - self.start_time_ms
        return 0


# =============================================================================
# Quorum Result
# =============================================================================

@dataclass
class QuorumResult:
    """Result of quorum evaluation."""
    request_id: str
    success: bool
    
    # Consensus metrics (median of agreeing evaluators)
    metrics: Optional[Dict[str, float]] = None
    
    # Participation
    total_evaluators: int = 0
    responding_evaluators: int = 0
    agreeing_evaluators: int = 0
    
    # Disagreements
    disagreeing_evaluator_ids: List[str] = field(default_factory=list)
    
    # Error if failed
    error: Optional[str] = None
    
    # Individual responses
    responses: List[EvaluationResponse] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "metrics": self.metrics,
            "total_evaluators": self.total_evaluators,
            "responding_evaluators": self.responding_evaluators,
            "agreeing_evaluators": self.agreeing_evaluators,
            "disagreeing_evaluator_ids": self.disagreeing_evaluator_ids,
            "error": self.error,
        }


# =============================================================================
# Evaluation Quorum Manager
# =============================================================================

class EvaluationQuorumManager:
    """
    Manage decentralized evaluation with quorum consensus.
    
    Flow:
    1. Receive evaluation request
    2. Select evaluators from registry
    3. Fan out request to evaluators
    4. Collect responses with timeout
    5. Check quorum agreement
    6. Return consensus result or fail
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[EvaluationQuorumConfig] = None,
    ):
        self._node_id = node_id
        self._config = config or EvaluationQuorumConfig()
        
        # Evaluator registry
        self._evaluators: Dict[str, EvaluatorInfo] = {}
        
        # Pending requests
        self._pending: Dict[str, EvaluationRequest] = {}
        self._responses: Dict[str, List[EvaluationResponse]] = {}
        
        # Callbacks
        self._send_request: Optional[Callable[[str, EvaluationRequest], asyncio.Future]] = None
        
        # Local evaluation (if this node is also an evaluator)
        self._local_evaluator: Optional[Callable[["AgentPack", "GoalSpec", int], Optional["Metrics"]]] = None
    
    def set_callbacks(
        self,
        send_request: Callable[[str, EvaluationRequest], asyncio.Future],
        local_evaluator: Optional[Callable[["AgentPack", "GoalSpec", int], Optional["Metrics"]]] = None,
    ) -> None:
        """Set callbacks for evaluation."""
        self._send_request = send_request
        self._local_evaluator = local_evaluator
    
    # -------------------------------------------------------------------------
    # Evaluator Management
    # -------------------------------------------------------------------------
    
    def register_evaluator(
        self,
        info: EvaluatorInfo,
    ) -> Tuple[bool, str]:
        """
        Register an evaluator.
        
        Args:
            info: Evaluator information
            
        Returns:
            (success, reason)
        """
        if info.stake_amount < self._config.evaluator_stake:
            return False, f"insufficient stake: {info.stake_amount} < {self._config.evaluator_stake}"
        
        self._evaluators[info.node_id] = info
        logger.info(f"Evaluator registered: {info.node_id[:16]}...")
        
        return True, "registered"
    
    def unregister_evaluator(self, node_id: str) -> bool:
        """Unregister an evaluator."""
        if node_id in self._evaluators:
            del self._evaluators[node_id]
            return True
        return False
    
    def get_evaluators(
        self,
        goal_id: str,
        count: int,
    ) -> List[EvaluatorInfo]:
        """
        Select evaluators for a goal.
        
        Selection criteria:
        1. Supports the goal
        2. Active status
        3. Sorted by reputation
        """
        eligible = [
            e for e in self._evaluators.values()
            if (
                e.status == EvaluatorStatus.ACTIVE and
                (not e.supported_goal_ids or goal_id in e.supported_goal_ids)
            )
        ]
        
        # Sort by reputation
        eligible.sort(key=lambda e: e.reputation_score, reverse=True)
        
        return eligible[:count]
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    
    async def request_evaluation(
        self,
        goal_id: str,
        agent_pack: "AgentPack",
        seed: int,
    ) -> QuorumResult:
        """
        Request distributed evaluation with quorum.
        
        Args:
            goal_id: Goal ID for evaluation
            agent_pack: Agent pack to evaluate
            seed: Evaluation seed
            
        Returns:
            QuorumResult with consensus metrics or error
        """
        # Create request ID
        request_id = hashlib.sha256(
            f"{goal_id}:{agent_pack.pack_hash.hex()}:{seed}:{time.time_ns()}".encode()
        ).hexdigest()[:32]
        
        # Serialize agent pack
        pack_data = json.dumps(agent_pack.to_dict(), sort_keys=True).encode()
        
        request = EvaluationRequest(
            request_id=request_id,
            goal_id=goal_id,
            agent_pack_hash=agent_pack.pack_hash,
            agent_pack_data=pack_data,
            seed=seed,
            requester_id=self._node_id,
        )
        
        # Select evaluators
        evaluators = self.get_evaluators(goal_id, self._config.min_evaluators * 2)
        
        if len(evaluators) < self._config.min_evaluators:
            return QuorumResult(
                request_id=request_id,
                success=False,
                error=f"insufficient evaluators: {len(evaluators)} < {self._config.min_evaluators}",
                total_evaluators=len(evaluators),
            )
        
        # Store pending request
        self._pending[request_id] = request
        self._responses[request_id] = []
        
        # Fan out requests
        tasks = []
        for evaluator in evaluators[:self._config.min_evaluators * 2]:
            task = asyncio.create_task(
                self._request_single_evaluation(evaluator, request)
            )
            tasks.append(task)
        
        # Wait for responses with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.evaluation_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Evaluation request {request_id} timed out")
        
        # Process results
        responses = self._responses.get(request_id, [])
        result = self._process_quorum(request_id, responses, len(evaluators))
        
        # Cleanup
        self._pending.pop(request_id, None)
        self._responses.pop(request_id, None)
        
        return result
    
    async def _request_single_evaluation(
        self,
        evaluator: EvaluatorInfo,
        request: EvaluationRequest,
    ) -> Optional[EvaluationResponse]:
        """Request evaluation from a single evaluator."""
        if not self._send_request:
            return None
        
        try:
            start_time = time.time()
            
            # Send request
            response = await asyncio.wait_for(
                self._send_request(evaluator.address, request),
                timeout=self._config.response_timeout,
            )
            
            if response:
                # Record response
                self._responses[request.request_id].append(response)
                
                # Update evaluator stats
                latency = (time.time() - start_time) * 1000
                self._update_evaluator_stats(evaluator.node_id, True, latency)
            
            return response
            
        except asyncio.TimeoutError:
            logger.debug(f"Evaluator {evaluator.node_id[:16]}... timed out")
            self._update_evaluator_stats(evaluator.node_id, False, 0)
            return None
        except Exception as e:
            logger.debug(f"Evaluator {evaluator.node_id[:16]}... failed: {e}")
            self._update_evaluator_stats(evaluator.node_id, False, 0)
            return None
    
    def _update_evaluator_stats(
        self,
        node_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Update evaluator statistics."""
        evaluator = self._evaluators.get(node_id)
        if not evaluator:
            return
        
        if success:
            evaluator.evaluations_completed += 1
            # Running average for latency
            n = evaluator.evaluations_completed
            evaluator.avg_latency_ms = (
                (evaluator.avg_latency_ms * (n - 1) + latency_ms) / n
            )
        else:
            evaluator.evaluations_failed += 1
    
    # -------------------------------------------------------------------------
    # Quorum Processing
    # -------------------------------------------------------------------------
    
    def _process_quorum(
        self,
        request_id: str,
        responses: List[EvaluationResponse],
        total_evaluators: int,
    ) -> QuorumResult:
        """
        Process responses and determine quorum result.
        
        Returns:
            QuorumResult with consensus or failure
        """
        # Filter successful responses
        successful = [r for r in responses if r.success and r.metrics]
        
        if len(successful) < self._config.min_evaluators:
            return QuorumResult(
                request_id=request_id,
                success=False,
                error=f"insufficient successful responses: {len(successful)} < {self._config.min_evaluators}",
                total_evaluators=total_evaluators,
                responding_evaluators=len(responses),
                responses=responses,
            )
        
        # Check metric agreement
        agreeing, disagreeing, consensus_metrics = self._check_agreement(successful)
        
        quorum_size = len(successful) * self._config.quorum_threshold
        
        if len(agreeing) < quorum_size:
            return QuorumResult(
                request_id=request_id,
                success=False,
                error=f"no quorum: {len(agreeing)} agreeing < {quorum_size} required",
                total_evaluators=total_evaluators,
                responding_evaluators=len(responses),
                agreeing_evaluators=len(agreeing),
                disagreeing_evaluator_ids=[r.evaluator_id for r in disagreeing],
                responses=responses,
            )
        
        # Record disagreements
        for response in disagreeing:
            evaluator = self._evaluators.get(response.evaluator_id)
            if evaluator:
                evaluator.disagreement_count += 1
        
        return QuorumResult(
            request_id=request_id,
            success=True,
            metrics=consensus_metrics,
            total_evaluators=total_evaluators,
            responding_evaluators=len(responses),
            agreeing_evaluators=len(agreeing),
            disagreeing_evaluator_ids=[r.evaluator_id for r in disagreeing],
            responses=responses,
        )
    
    def _check_agreement(
        self,
        responses: List[EvaluationResponse],
    ) -> Tuple[List[EvaluationResponse], List[EvaluationResponse], Dict[str, float]]:
        """
        Check if responses agree within tolerance.
        
        Returns:
            (agreeing_responses, disagreeing_responses, consensus_metrics)
        """
        if not responses:
            return [], [], {}
        
        # Extract metrics
        rewards = [r.metrics.get("reward", 0) for r in responses if r.metrics]
        risks = [r.metrics.get("risk", 0) for r in responses if r.metrics]
        complexities = [r.metrics.get("complexity", 0) for r in responses if r.metrics]
        
        # Calculate medians
        median_reward = statistics.median(rewards) if rewards else 0
        median_risk = statistics.median(risks) if risks else 0
        median_complexity = statistics.median(complexities) if complexities else 0
        
        # Check each response against median
        agreeing = []
        disagreeing = []
        
        for response in responses:
            if not response.metrics:
                disagreeing.append(response)
                continue
            
            r = response.metrics.get("reward", 0)
            k = response.metrics.get("risk", 0)
            c = response.metrics.get("complexity", 0)
            
            # Check within tolerance
            reward_ok = abs(r - median_reward) <= self._config.reward_tolerance * max(abs(median_reward), 0.01)
            risk_ok = abs(k - median_risk) <= self._config.risk_tolerance * max(abs(median_risk), 0.01)
            complexity_ok = abs(c - median_complexity) <= self._config.complexity_tolerance * max(abs(median_complexity), 0.01)
            
            if reward_ok and risk_ok and complexity_ok:
                agreeing.append(response)
            else:
                disagreeing.append(response)
        
        consensus_metrics = {
            "reward": median_reward,
            "risk": median_risk,
            "complexity": median_complexity,
        }
        
        return agreeing, disagreeing, consensus_metrics
    
    # -------------------------------------------------------------------------
    # Response Handling
    # -------------------------------------------------------------------------
    
    async def handle_evaluation_response(
        self,
        response: EvaluationResponse,
    ) -> None:
        """Handle incoming evaluation response."""
        if response.request_id in self._responses:
            self._responses[response.request_id].append(response)
    
    # -------------------------------------------------------------------------
    # Local Evaluation
    # -------------------------------------------------------------------------
    
    async def handle_evaluation_request(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResponse:
        """
        Handle incoming evaluation request (as an evaluator).
        
        Args:
            request: Evaluation request
            
        Returns:
            Evaluation response
        """
        start_time = int(time.time() * 1000)
        
        if not self._local_evaluator:
            return EvaluationResponse(
                request_id=request.request_id,
                evaluator_id=self._node_id,
                success=False,
                error="no local evaluator configured",
                start_time_ms=start_time,
                end_time_ms=int(time.time() * 1000),
            )
        
        try:
            # Deserialize agent pack
            from idi.ian.models import AgentPack, GoalSpec
            
            pack_data = json.loads(request.agent_pack_data.decode())
            agent_pack = AgentPack.from_dict(pack_data)
            
            # Verify hash
            if agent_pack.pack_hash != request.agent_pack_hash:
                return EvaluationResponse(
                    request_id=request.request_id,
                    evaluator_id=self._node_id,
                    success=False,
                    error="agent pack hash mismatch",
                    start_time_ms=start_time,
                    end_time_ms=int(time.time() * 1000),
                )
            
            # Run evaluation
            # Note: Would need GoalSpec from somewhere
            # For now, just return the pack hash verification
            metrics = None
            if self._local_evaluator:
                # This would call the actual evaluation harness
                # metrics = self._local_evaluator(agent_pack, goal_spec, request.seed)
                pass
            
            end_time = int(time.time() * 1000)
            
            if metrics:
                return EvaluationResponse(
                    request_id=request.request_id,
                    evaluator_id=self._node_id,
                    success=True,
                    metrics={
                        "reward": metrics.reward,
                        "risk": metrics.risk,
                        "complexity": metrics.complexity,
                    },
                    start_time_ms=start_time,
                    end_time_ms=end_time,
                )
            else:
                return EvaluationResponse(
                    request_id=request.request_id,
                    evaluator_id=self._node_id,
                    success=False,
                    error="evaluation returned no metrics",
                    start_time_ms=start_time,
                    end_time_ms=end_time,
                )
                
        except Exception as e:
            return EvaluationResponse(
                request_id=request.request_id,
                evaluator_id=self._node_id,
                success=False,
                error=str(e),
                start_time_ms=start_time,
                end_time_ms=int(time.time() * 1000),
            )
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize quorum manager state."""
        return {
            "evaluators": {
                k: v.to_dict() for k, v in self._evaluators.items()
            },
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        node_id: str,
        config: Optional[EvaluationQuorumConfig] = None,
    ) -> "EvaluationQuorumManager":
        """Deserialize quorum manager state."""
        manager = cls(node_id=node_id, config=config)
        
        for k, v in data.get("evaluators", {}).items():
            manager._evaluators[k] = EvaluatorInfo.from_dict(v)
        
        return manager
