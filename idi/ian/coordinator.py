"""
IAN Coordinator - Core contribution processing logic.

The coordinator implements the deterministic, auditable pipeline for
processing agent contributions:

1. Dedup check (O(1) via BloomFilter + HashIndex)
2. Structure validation
3. Invariant checks (fail-fast)
4. Proof verification
5. Sandboxed evaluation
6. Threshold check
7. Log append (O(log N) via MMR)
8. Dedup update
9. Leaderboard update (O(log K))

Design Principles:
- Pure: No side effects outside GoalState
- Deterministic: Same input always produces same output
- Terminating: Bounded by evaluation limits
- Auditable: Every step is logged

Invariants:
- After processing contribution i, GoalState reflects all valid contributions [0..i]
- The log root uniquely identifies the contribution history
- The leaderboard contains exactly the top-K valid contributions
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .models import (
    AgentPack,
    Contribution,
    ContributionMeta,
    GoalID,
    GoalSpec,
    GoalState,
    Metrics,
    Thresholds,
)
from .mmr import MerkleMountainRange
from .leaderboard import Leaderboard, ParetoFrontier
from .dedup import DedupService
from .hooks import (
    CoordinatorHooks,
    ContributionAcceptedEvent,
    ContributionRejectedEvent,
    LeaderboardUpdatedEvent,
)


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Rejection Reasons
# -----------------------------------------------------------------------------

class RejectionReason(Enum):
    """Reasons why a contribution may be rejected."""
    DUPLICATE = auto()
    INVALID_STRUCTURE = auto()
    INVARIANT_VIOLATION = auto()
    PROOF_FAILURE = auto()
    EVALUATION_TIMEOUT = auto()
    EVALUATION_ERROR = auto()
    BELOW_THRESHOLD = auto()
    # Security-related rejections
    VALIDATION_ERROR = auto()
    RATE_LIMITED = auto()
    POW_REQUIRED = auto()
    POW_INVALID = auto()


@dataclass(frozen=True)
class ProcessResult:
    """Result of processing a contribution."""
    accepted: bool
    reason: str
    rejection_type: Optional[RejectionReason] = None
    metrics: Optional[Metrics] = None
    log_index: Optional[int] = None
    score: Optional[float] = None
    
    @classmethod
    def success(
        cls,
        metrics: Metrics,
        log_index: int,
        score: float,
    ) -> "ProcessResult":
        return cls(
            accepted=True,
            reason="accepted",
            metrics=metrics,
            log_index=log_index,
            score=score,
        )
    
    @classmethod
    def reject(
        cls,
        rejection_type: RejectionReason,
        reason: str,
    ) -> "ProcessResult":
        return cls(
            accepted=False,
            reason=reason,
            rejection_type=rejection_type,
        )


# -----------------------------------------------------------------------------
# Pluggable Interfaces
# -----------------------------------------------------------------------------

class InvariantChecker(Protocol):
    """Protocol for invariant checking."""
    
    def check(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """
        Check if agent_pack satisfies invariants in goal_spec.
        
        Returns:
            (passed, reason) where reason explains failure if passed=False
        """
        ...


class ProofVerifier(Protocol):
    """Protocol for proof verification."""
    
    def verify(
        self,
        agent_pack: AgentPack,
        proofs: Dict[str, bytes],
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        """
        Verify proofs for an agent_pack.
        
        Returns:
            (valid, reason) where reason explains failure if valid=False
        """
        ...


class EvaluationHarness(Protocol):
    """Protocol for sandboxed evaluation."""
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        """
        Evaluate agent_pack in a sandboxed environment.
        
        Returns:
            Metrics if evaluation succeeded, None if timeout/error
        """
        ...


# -----------------------------------------------------------------------------
# Default Implementations
# -----------------------------------------------------------------------------

class PassthroughInvariantChecker:
    """Default invariant checker that always passes (for testing)."""
    
    def check(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        return True, "passed (no invariants configured)"


class PassthroughProofVerifier:
    """Default proof verifier that always passes (for testing)."""
    
    def verify(
        self,
        agent_pack: AgentPack,
        proofs: Dict[str, bytes],
        goal_spec: GoalSpec,
    ) -> Tuple[bool, str]:
        return True, "passed (no proofs required)"


class DummyEvaluationHarness:
    """Dummy evaluation harness for testing (returns mock metrics)."""
    
    def evaluate(
        self,
        agent_pack: AgentPack,
        goal_spec: GoalSpec,
        seed: int,
    ) -> Optional[Metrics]:
        # Deterministic mock metrics based on pack hash
        h = agent_pack.pack_hash
        reward = (h[0] / 255.0) * 0.5 + 0.5  # 0.5 to 1.0
        risk = (h[1] / 255.0) * 0.3  # 0.0 to 0.3
        complexity = (h[2] / 255.0) * 0.2 + 0.1  # 0.1 to 0.3
        
        return Metrics(
            reward=reward,
            risk=risk,
            complexity=complexity,
            episodes_run=goal_spec.eval_limits.max_episodes,
            steps_run=goal_spec.eval_limits.max_episodes * 100,
        )


# -----------------------------------------------------------------------------
# Coordinator
# -----------------------------------------------------------------------------

# Configurable limits (avoid magic numbers)
MAX_PARAM_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max agent parameters


@dataclass
class CoordinatorConfig:
    """Configuration for IANCoordinator."""
    leaderboard_capacity: int = 100
    use_pareto: bool = False
    expected_contributions: int = 100_000
    bloom_fp_rate: float = 0.01
    max_param_size: int = MAX_PARAM_SIZE_BYTES  # Configurable


class IANCoordinator:
    """
    Main coordinator for processing IAN contributions.
    
    Maintains per-goal state and processes contributions through
    the deterministic pipeline.
    """
    
    def __init__(
        self,
        goal_spec: GoalSpec,
        config: Optional[CoordinatorConfig] = None,
        invariant_checker: Optional[InvariantChecker] = None,
        proof_verifier: Optional[ProofVerifier] = None,
        evaluation_harness: Optional[EvaluationHarness] = None,
        hooks: Optional[CoordinatorHooks] = None,
    ) -> None:
        """
        Initialize coordinator for a goal.
        
        Args:
            goal_spec: Specification of the goal
            config: Coordinator configuration
            invariant_checker: Plugin for invariant checking
            proof_verifier: Plugin for proof verification
            evaluation_harness: Plugin for sandboxed evaluation
            hooks: Optional event hooks for real-time notifications
        """
        self.config = config or CoordinatorConfig()
        self.goal_spec = goal_spec
        
        # Initialize state
        self.state = GoalState(
            goal_id=goal_spec.goal_id,
            spec=goal_spec,
            log=MerkleMountainRange(),
            leaderboard=self._create_leaderboard(),
            dedup=DedupService(
                expected_contributions=self.config.expected_contributions,
                bloom_fp_rate=self.config.bloom_fp_rate,
            ),
        )
        
        # Plugins
        self._invariant_checker = invariant_checker or PassthroughInvariantChecker()
        self._proof_verifier = proof_verifier or PassthroughProofVerifier()
        self._evaluation_harness = evaluation_harness or DummyEvaluationHarness()
        self._hooks = hooks
        
        logger.info(
            f"IANCoordinator initialized for goal {goal_spec.goal_id}, "
            f"leaderboard_capacity={self.config.leaderboard_capacity}, "
            f"use_pareto={self.config.use_pareto}"
        )
    
    def _create_leaderboard(self) -> Leaderboard | ParetoFrontier:
        """Create appropriate leaderboard based on config."""
        if self.config.use_pareto or self.goal_spec.use_pareto:
            return ParetoFrontier(max_size=self.config.leaderboard_capacity)
        return Leaderboard(capacity=self.config.leaderboard_capacity)
    
    def process_contribution(self, contrib: Contribution) -> ProcessResult:
        """
        Process a contribution through the full pipeline.
        
        This is the main entry point for the coordinator.
        
        Steps:
        1. Dedup check
        2. Structure validation
        3. Invariant checks
        4. Proof verification
        5. Sandboxed evaluation
        6. Threshold check
        7. Log append
        8. Dedup update
        9. Leaderboard update
        
        Args:
            contrib: The contribution to process
            
        Returns:
            ProcessResult with acceptance status and details
            
        Complexity: O(log N) for log append + O(log K) for leaderboard
        """
        self.state.total_contributions += 1
        pack_hash = contrib.pack_hash
        
        # Step 1: Dedup check (O(1))
        if self.state.dedup.is_duplicate(pack_hash):
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: duplicate")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.DUPLICATE,
                "duplicate contribution",
            )
        
        # Step 2: Structure validation
        valid, reason = self._validate_structure(contrib)
        if not valid:
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: {reason}")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.INVALID_STRUCTURE,
                reason,
            )
        
        # Step 3: Invariant checks (fail-fast)
        passed, reason = self._invariant_checker.check(
            contrib.agent_pack,
            self.goal_spec,
        )
        if not passed:
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: invariant violation - {reason}")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.INVARIANT_VIOLATION,
                f"invariant violation: {reason}",
            )
        
        # Step 4: Proof verification
        valid, reason = self._proof_verifier.verify(
            contrib.agent_pack,
            contrib.proofs,
            self.goal_spec,
        )
        if not valid:
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: proof failure - {reason}")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.PROOF_FAILURE,
                f"proof verification failed: {reason}",
            )
        
        # Step 5: Sandboxed evaluation
        metrics = self._evaluation_harness.evaluate(
            contrib.agent_pack,
            self.goal_spec,
            contrib.seed,
        )
        if metrics is None:
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: evaluation failed")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.EVALUATION_ERROR,
                "evaluation failed (timeout or error)",
            )
        
        # Step 6: Threshold check
        passed, reason = self.goal_spec.thresholds.check(metrics)
        if not passed:
            logger.debug(f"Contribution {pack_hash.hex()[:16]}... rejected: below threshold - {reason}")
            self.state.rejected_contributions += 1
            return ProcessResult.reject(
                RejectionReason.BELOW_THRESHOLD,
                f"below threshold: {reason}",
            )
        
        # Step 7: Log append (O(log N))
        log_index = self.state.log.append(pack_hash)
        
        # Step 8: Dedup update
        self.state.dedup.add(pack_hash, log_index)
        
        # Step 9: Leaderboard update (O(log K))
        score = self.goal_spec.compute_score(metrics)
        meta = ContributionMeta(
            pack_hash=pack_hash,
            metrics=metrics,
            score=score,
            contributor_id=contrib.contributor_id,
            timestamp_ms=int(time.time() * 1000),
            log_index=log_index,
        )
        
        added_to_leaderboard = self.state.leaderboard.add(meta)
        
        # Update state roots
        self.state.update_roots()
        self.state.accepted_contributions += 1
        
        # Check if this becomes the active policy
        active = self.get_active_policy()
        if active and active.pack_hash == pack_hash:
            self.state.active_policy_hash = pack_hash
        
        logger.info(
            f"Contribution {pack_hash.hex()[:16]}... accepted: "
            f"score={score:.4f}, log_index={log_index}, "
            f"on_leaderboard={added_to_leaderboard}"
        )
        
        # Notify hooks (exception-safe to preserve determinism)
        if self._hooks:
            try:
                # Compute leaderboard position
                leaderboard = self.get_leaderboard()
                position = next(
                    (i + 1 for i, m in enumerate(leaderboard) if m.pack_hash == pack_hash),
                    None,
                )
                is_new_leader = position == 1 and active and active.pack_hash == pack_hash
                
                self._hooks.on_contribution_accepted(
                    ContributionAcceptedEvent(
                        goal_id=str(self.goal_spec.goal_id),
                        pack_hash=pack_hash,
                        contributor_id=contrib.contributor_id,
                        score=score,
                        log_index=log_index,
                        metrics=metrics,
                        leaderboard_position=position,
                        is_new_leader=is_new_leader,
                    )
                )
                
                # Also emit leaderboard update if entry was added
                if added_to_leaderboard:
                    self._hooks.on_leaderboard_updated(
                        LeaderboardUpdatedEvent(
                            goal_id=str(self.goal_spec.goal_id),
                            entries=leaderboard,
                            active_policy_hash=self.state.active_policy_hash,
                        )
                    )
            except Exception as e:
                logger.warning(f"Hook exception (ignored for determinism): {e}")
        
        return ProcessResult.success(
            metrics=metrics,
            log_index=log_index,
            score=score,
        )
    
    def _validate_structure(self, contrib: Contribution) -> Tuple[bool, str]:
        """
        Validate contribution structure.
        
        Checks:
        - Goal ID matches
        - Agent pack version is present
        - Parameters size within limits (DoS prevention)
        - Contributor ID is valid
        """
        # Check goal ID matches
        if contrib.goal_id != self.goal_spec.goal_id:
            return False, f"goal_id mismatch: expected {self.goal_spec.goal_id}, got {contrib.goal_id}"
        
        # Check agent pack version
        if not contrib.agent_pack.version:
            return False, "agent_pack.version is empty"
        
        # Check version format (basic validation)
        if len(contrib.agent_pack.version) > 64:
            return False, f"agent_pack.version too long: {len(contrib.agent_pack.version)} > 64"
        
        # Check parameters size (prevent DoS) - use configurable limit
        if len(contrib.agent_pack.parameters) > self.config.max_param_size:
            return False, f"agent_pack.parameters too large: {len(contrib.agent_pack.parameters)} > {self.config.max_param_size}"
        
        # Check contributor ID length
        if len(contrib.contributor_id) > 256:
            return False, f"contributor_id too long: {len(contrib.contributor_id)} > 256"
        
        return True, "valid"
    
    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------
    
    def get_leaderboard(self, sorted_desc: bool = True) -> List[ContributionMeta]:
        """Get current leaderboard entries."""
        if isinstance(self.state.leaderboard, Leaderboard):
            return self.state.leaderboard.top_k(sorted_desc=sorted_desc)
        elif isinstance(self.state.leaderboard, ParetoFrontier):
            return self.state.leaderboard.frontier()
        return []
    
    def get_active_policy(self) -> Optional[ContributionMeta]:
        """Get the current best (active) policy."""
        if isinstance(self.state.leaderboard, Leaderboard):
            return self.state.leaderboard.get_active_policy()
        elif isinstance(self.state.leaderboard, ParetoFrontier):
            # For Pareto, return the one with highest reward
            frontier = self.state.leaderboard.frontier()
            if not frontier:
                return None
            return max(frontier, key=lambda m: m.metrics.reward)
        return None
    
    def get_log_root(self) -> bytes:
        """Get current experiment log root."""
        return self.state.log.get_root()
    
    def get_leaderboard_root(self) -> bytes:
        """Get current leaderboard state root."""
        return self.state.leaderboard.get_root()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "goal_id": str(self.state.goal_id),
            "total_contributions": self.state.total_contributions,
            "accepted_contributions": self.state.accepted_contributions,
            "rejected_contributions": self.state.rejected_contributions,
            "log_size": self.state.log.size,
            "leaderboard_size": len(self.state.leaderboard),
            "log_root": self.state.log_root.hex(),
            "leaderboard_root": self.state.leaderboard_root.hex(),
            "active_policy_hash": self.state.active_policy_hash.hex() if self.state.active_policy_hash else None,
        }
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize coordinator state for persistence."""
        return {
            "goal_spec": self.goal_spec.to_dict(),
            "config": {
                "leaderboard_capacity": self.config.leaderboard_capacity,
                "use_pareto": self.config.use_pareto,
                "expected_contributions": self.config.expected_contributions,
                "bloom_fp_rate": self.config.bloom_fp_rate,
            },
            "state": {
                "log": self.state.log.to_dict(),
                "leaderboard": self.state.leaderboard.to_dict(),
                "dedup": self.state.dedup.to_dict(),
                "total_contributions": self.state.total_contributions,
                "accepted_contributions": self.state.accepted_contributions,
                "rejected_contributions": self.state.rejected_contributions,
                "active_policy_hash": self.state.active_policy_hash.hex() if self.state.active_policy_hash else None,
                "last_upgrade_timestamp_ms": self.state.last_upgrade_timestamp_ms,
            },
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        invariant_checker: Optional[InvariantChecker] = None,
        proof_verifier: Optional[ProofVerifier] = None,
        evaluation_harness: Optional[EvaluationHarness] = None,
    ) -> "IANCoordinator":
        """Deserialize coordinator from stored state."""
        from .models import EvaluationLimits, Thresholds
        
        # Reconstruct GoalSpec
        spec_data = data["goal_spec"]
        goal_spec = GoalSpec(
            goal_id=GoalID(spec_data["goal_id"]),
            name=spec_data["name"],
            description=spec_data.get("description", ""),
            invariant_ids=spec_data.get("invariant_ids", []),
            mpb_bytecode=bytes.fromhex(spec_data["mpb_bytecode"]) if spec_data.get("mpb_bytecode") else None,
            eval_harness_id=spec_data.get("eval_harness_id", "default"),
            eval_limits=EvaluationLimits(
                max_episodes=spec_data["eval_limits"]["max_episodes"],
                max_steps_per_episode=spec_data["eval_limits"]["max_steps_per_episode"],
                timeout_seconds=spec_data["eval_limits"]["timeout_seconds"],
                max_memory_mb=spec_data["eval_limits"]["max_memory_mb"],
            ),
            thresholds=Thresholds(
                min_reward=spec_data["thresholds"]["min_reward"],
                max_risk=spec_data["thresholds"]["max_risk"],
                max_complexity=spec_data["thresholds"]["max_complexity"],
                min_sharpe_ratio=spec_data["thresholds"].get("min_sharpe_ratio"),
                max_drawdown=spec_data["thresholds"].get("max_drawdown"),
            ),
            ranking_weights=spec_data.get("ranking_weights", {}),
            use_pareto=spec_data.get("use_pareto", False),
            requires_governance_approval=spec_data.get("requires_governance_approval", False),
            governance_quorum=spec_data.get("governance_quorum", 0.5),
            upgrade_cooldown_seconds=spec_data.get("upgrade_cooldown_seconds", 86400),
        )
        
        # Reconstruct config
        config_data = data["config"]
        config = CoordinatorConfig(
            leaderboard_capacity=config_data["leaderboard_capacity"],
            use_pareto=config_data["use_pareto"],
            expected_contributions=config_data["expected_contributions"],
            bloom_fp_rate=config_data["bloom_fp_rate"],
        )
        
        # Create coordinator
        coordinator = cls(
            goal_spec=goal_spec,
            config=config,
            invariant_checker=invariant_checker,
            proof_verifier=proof_verifier,
            evaluation_harness=evaluation_harness,
        )
        
        # Restore state
        state_data = data["state"]
        coordinator.state.log = MerkleMountainRange.from_dict(state_data["log"])
        
        if config.use_pareto:
            coordinator.state.leaderboard = ParetoFrontier.from_dict(state_data["leaderboard"])
        else:
            coordinator.state.leaderboard = Leaderboard.from_dict(state_data["leaderboard"])
        
        coordinator.state.dedup = DedupService.from_dict(state_data["dedup"])
        coordinator.state.total_contributions = state_data["total_contributions"]
        coordinator.state.accepted_contributions = state_data["accepted_contributions"]
        coordinator.state.rejected_contributions = state_data["rejected_contributions"]
        coordinator.state.active_policy_hash = (
            bytes.fromhex(state_data["active_policy_hash"])
            if state_data.get("active_policy_hash")
            else None
        )
        coordinator.state.last_upgrade_timestamp_ms = state_data.get("last_upgrade_timestamp_ms", 0)
        
        # Update roots
        coordinator.state.update_roots()
        
        return coordinator
    
    def __repr__(self) -> str:
        return (
            f"IANCoordinator(goal={self.goal_spec.goal_id}, "
            f"log_size={self.state.log.size}, "
            f"leaderboard_size={len(self.state.leaderboard)})"
        )
