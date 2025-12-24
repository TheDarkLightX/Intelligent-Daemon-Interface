"""
IAN Data Models.

Defines core data structures for the Intelligent Augmentation Network:
- GoalID, GoalSpec: Define tasks/archetypes
- AgentPack, Contribution, ContributionMeta: Candidate policies and submissions
- Metrics, Thresholds, EvaluationLimits: Evaluation configuration
- GoalState: Per-goal coordinator state

Design Principles:
- Immutable where possible (frozen dataclasses)
- Content-addressable via deterministic hashing
- Serializable to JSON/MessagePack
- Type-safe with explicit validation
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------

# Pre-compiled regex for performance
_GOAL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_goal_id(value: str) -> str:
    """Validate GoalID format: alphanumeric + underscores, 1-64 chars."""
    if not value:
        raise ValueError("GoalID cannot be empty")
    if len(value) > 64:
        raise ValueError(f"GoalID too long: {len(value)} > 64")
    if not _GOAL_ID_PATTERN.match(value):
        raise ValueError(f"GoalID contains invalid characters: {value}")
    return value


def _compute_hash(data: bytes) -> bytes:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).digest()


def _deterministic_timestamp() -> int:
    """Return current Unix timestamp in milliseconds (deterministic for testing)."""
    return int(time.time() * 1000)


# -----------------------------------------------------------------------------
# Core Identifiers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalID:
    """
    Unique identifier for a goal/task in IAN.
    
    Examples: "OWNERLESS_VC_AGENT", "MARKET_MAKER_V1", "INDEX_REBALANCER"
    
    Invariants:
    - value is non-empty, alphanumeric + underscores, max 64 chars
    """
    value: str
    
    def __post_init__(self) -> None:
        _validate_goal_id(self.value)
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def to_bytes(self) -> bytes:
        return self.value.encode("utf-8")


# -----------------------------------------------------------------------------
# Metrics & Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Metrics:
    """
    Evaluation metrics for a candidate agent.
    
    All values are normalized floats. Higher reward is better;
    lower risk and complexity are better.
    """
    reward: float
    risk: float
    complexity: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    episodes_run: int = 0
    steps_run: int = 0
    
    def __post_init__(self) -> None:
        # Basic sanity checks
        if not isinstance(self.reward, (int, float)):
            raise TypeError(f"reward must be numeric, got {type(self.reward)}")
        if not isinstance(self.risk, (int, float)):
            raise TypeError(f"risk must be numeric, got {type(self.risk)}")
        if not isinstance(self.complexity, (int, float)):
            raise TypeError(f"complexity must be numeric, got {type(self.complexity)}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "risk": self.risk,
            "complexity": self.complexity,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "episodes_run": self.episodes_run,
            "steps_run": self.steps_run,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metrics":
        return cls(
            reward=data["reward"],
            risk=data["risk"],
            complexity=data["complexity"],
            sharpe_ratio=data.get("sharpe_ratio"),
            max_drawdown=data.get("max_drawdown"),
            episodes_run=data.get("episodes_run", 0),
            steps_run=data.get("steps_run", 0),
        )


@dataclass(frozen=True)
class Thresholds:
    """
    Minimum thresholds for accepting a contribution.
    
    A contribution must meet ALL thresholds to be accepted.
    """
    min_reward: float = 0.0
    max_risk: float = 1.0
    max_complexity: float = float("inf")
    min_sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    def _check_finite(self, value: float, name: str) -> Optional[str]:
        """Return error message if value is not finite, else None."""
        import math
        if not math.isfinite(value):
            return f"{name} is not finite: {value}"
        return None

    def _check_core_metrics(self, metrics: Metrics) -> Optional[str]:
        """Check required metrics (reward, risk, complexity). Return error or None."""
        if err := self._check_finite(metrics.reward, "reward"):
            return err
        if err := self._check_finite(metrics.risk, "risk"):
            return err
        if err := self._check_finite(metrics.complexity, "complexity"):
            return err
        if metrics.reward < self.min_reward:
            return f"reward {metrics.reward:.4f} < min {self.min_reward:.4f}"
        if metrics.risk > self.max_risk:
            return f"risk {metrics.risk:.4f} > max {self.max_risk:.4f}"
        if metrics.complexity > self.max_complexity:
            return f"complexity {metrics.complexity:.4f} > max {self.max_complexity:.4f}"
        return None

    def _check_optional_metrics(self, metrics: Metrics) -> Optional[str]:
        """Check optional metrics (sharpe_ratio, max_drawdown). Return error or None."""
        if self.min_sharpe_ratio is not None and metrics.sharpe_ratio is not None:
            if err := self._check_finite(metrics.sharpe_ratio, "sharpe_ratio"):
                return err
            if metrics.sharpe_ratio < self.min_sharpe_ratio:
                return f"sharpe {metrics.sharpe_ratio:.4f} < min {self.min_sharpe_ratio:.4f}"
        if self.max_drawdown is not None and metrics.max_drawdown is not None:
            if err := self._check_finite(metrics.max_drawdown, "max_drawdown"):
                return err
            if metrics.max_drawdown > self.max_drawdown:
                return f"drawdown {metrics.max_drawdown:.4f} > max {self.max_drawdown:.4f}"
        return None

    def check(self, metrics: Metrics) -> Tuple[bool, str]:
        """
        Check if metrics meet thresholds.
        
        Returns:
            (passed, reason) where reason explains failure if passed=False
        """
        if err := self._check_core_metrics(metrics):
            return False, err
        if err := self._check_optional_metrics(metrics):
            return False, err
        return True, "passed"


@dataclass(frozen=True)
class EvaluationLimits:
    """
    Resource limits for sandboxed evaluation.
    
    These limits prevent DoS via expensive evaluations.
    """
    max_episodes: int = 1000
    max_steps_per_episode: int = 500
    timeout_seconds: float = 300.0  # 5 minutes
    max_memory_mb: int = 1024  # 1 GB
    
    def __post_init__(self) -> None:
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


# -----------------------------------------------------------------------------
# Agent Pack & Contribution
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentPack:
    """
    A concrete learned policy/configuration submitted as a candidate.
    
    The pack is content-addressed by its hash, computed from parameters.
    """
    version: str
    parameters: bytes  # Serialized Q-table, network weights, or config
    metadata: Dict[str, Any] = field(default_factory=dict)
    tau_spec_hash: Optional[bytes] = None  # Hash of associated Tau spec
    mpb_invariants_hash: Optional[bytes] = None  # Hash of MPB invariants
    
    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("AgentPack parameters cannot be empty")
    
    @property
    def pack_hash(self) -> bytes:
        """
        Compute deterministic hash of pack contents.
        
        Uses JSON serialization for metadata to ensure deterministic
        hashing of nested structures.
        """
        hasher = hashlib.sha256()
        hasher.update(self.version.encode("utf-8"))
        hasher.update(self.parameters)
        # FIXED: Use JSON for deterministic nested dict serialization
        hasher.update(json.dumps(self.metadata, sort_keys=True, separators=(',', ':')).encode("utf-8"))
        if self.tau_spec_hash:
            hasher.update(self.tau_spec_hash)
        if self.mpb_invariants_hash:
            hasher.update(self.mpb_invariants_hash)
        return hasher.digest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "parameters": self.parameters.hex(),
            "metadata": self.metadata,
            "tau_spec_hash": self.tau_spec_hash.hex() if self.tau_spec_hash else None,
            "mpb_invariants_hash": self.mpb_invariants_hash.hex() if self.mpb_invariants_hash else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPack":
        return cls(
            version=data["version"],
            parameters=bytes.fromhex(data["parameters"]),
            metadata=data.get("metadata", {}),
            tau_spec_hash=bytes.fromhex(data["tau_spec_hash"]) if data.get("tau_spec_hash") else None,
            mpb_invariants_hash=bytes.fromhex(data["mpb_invariants_hash"]) if data.get("mpb_invariants_hash") else None,
        )


@dataclass(frozen=True)
class Contribution:
    """
    A submission to IAN: an AgentPack with proofs and contributor info.
    
    This is the input to `process_contribution`.
    """
    goal_id: GoalID
    agent_pack: AgentPack
    proofs: Dict[str, bytes] = field(default_factory=dict)  # proof_type -> proof_data
    contributor_id: str = "anonymous"
    seed: int = 0  # Deterministic seed for evaluation
    
    def __post_init__(self) -> None:
        if not self.contributor_id:
            object.__setattr__(self, "contributor_id", "anonymous")
    
    @property
    def pack_hash(self) -> bytes:
        return self.agent_pack.pack_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id),
            "agent_pack": self.agent_pack.to_dict(),
            "proofs": {key: value.hex() for key, value in self.proofs.items()},
            "contributor_id": self.contributor_id,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contribution":
        proofs_raw = data.get("proofs", {})
        proofs: Dict[str, bytes] = {}
        if isinstance(proofs_raw, dict):
            proofs = {key: bytes.fromhex(value) for key, value in proofs_raw.items()}

        return cls(
            goal_id=GoalID(data["goal_id"]),
            agent_pack=AgentPack.from_dict(data["agent_pack"]),
            proofs=proofs,
            contributor_id=data.get("contributor_id", "anonymous"),
            seed=int(data.get("seed", 0)),
        )


@dataclass(frozen=True)
class ContributionMeta:
    """
    Metadata stored in the experiment log and leaderboard.
    
    This is a compact summary of a contribution, not the full pack.
    """
    pack_hash: bytes
    metrics: Metrics
    score: float  # Computed by ranking function
    contributor_id: str
    timestamp_ms: int
    log_index: int = -1  # Position in experiment log (-1 if not yet logged)
    
    def __post_init__(self) -> None:
        if len(self.pack_hash) != 32:
            raise ValueError(f"pack_hash must be 32 bytes, got {len(self.pack_hash)}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_hash": self.pack_hash.hex(),
            "metrics": self.metrics.to_dict(),
            "score": self.score,
            "contributor_id": self.contributor_id,
            "timestamp_ms": self.timestamp_ms,
            "log_index": self.log_index,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContributionMeta":
        return cls(
            pack_hash=bytes.fromhex(data["pack_hash"]),
            metrics=Metrics.from_dict(data["metrics"]),
            score=data["score"],
            contributor_id=data["contributor_id"],
            timestamp_ms=data["timestamp_ms"],
            log_index=data.get("log_index", -1),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for MMR leaf."""
        return json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ContributionMeta":
        return cls.from_dict(json.loads(data.decode("utf-8")))


# -----------------------------------------------------------------------------
# Goal Specification
# -----------------------------------------------------------------------------

@dataclass
class GoalSpec:
    """
    Full specification of a goal/task in IAN.
    
    Bundles objectives, invariants, evaluation config, and thresholds.
    """
    goal_id: GoalID
    name: str
    description: str = ""
    
    # Invariants (references or inline)
    invariant_ids: List[str] = field(default_factory=list)  # IDs of invariants to check
    mpb_bytecode: Optional[bytes] = None  # Compiled MPB program for invariants
    
    # Evaluation
    eval_harness_id: str = "default"  # ID of evaluation harness to use
    eval_limits: EvaluationLimits = field(default_factory=EvaluationLimits)
    
    # Acceptance criteria
    thresholds: Thresholds = field(default_factory=Thresholds)
    
    # Ranking
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "reward": 1.0,
        "risk": -0.5,
        "complexity": -0.01,
    })
    use_pareto: bool = False  # If True, use Pareto frontier instead of scalar ranking
    
    # Governance
    requires_governance_approval: bool = False
    governance_quorum: float = 0.5  # Fraction of votes needed
    upgrade_cooldown_seconds: int = 86400  # 24 hours between upgrades
    
    def compute_score(self, metrics: Metrics) -> float:
        """Compute scalar score from metrics using ranking weights.
        
        All metrics with configured weights are included in the score.
        Missing optional metrics (None values) are skipped.
        """
        score = 0.0
        if "reward" in self.ranking_weights:
            score += self.ranking_weights["reward"] * metrics.reward
        if "risk" in self.ranking_weights:
            score += self.ranking_weights["risk"] * metrics.risk
        if "complexity" in self.ranking_weights:
            score += self.ranking_weights["complexity"] * metrics.complexity
        if "sharpe_ratio" in self.ranking_weights and metrics.sharpe_ratio is not None:
            score += self.ranking_weights["sharpe_ratio"] * metrics.sharpe_ratio
        # Include max_drawdown if weight is configured and metric is available
        if "max_drawdown" in self.ranking_weights and metrics.max_drawdown is not None:
            score += self.ranking_weights["max_drawdown"] * metrics.max_drawdown
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id.value,
            "name": self.name,
            "description": self.description,
            "invariant_ids": self.invariant_ids,
            "mpb_bytecode": self.mpb_bytecode.hex() if self.mpb_bytecode else None,
            "eval_harness_id": self.eval_harness_id,
            "eval_limits": {
                "max_episodes": self.eval_limits.max_episodes,
                "max_steps_per_episode": self.eval_limits.max_steps_per_episode,
                "timeout_seconds": self.eval_limits.timeout_seconds,
                "max_memory_mb": self.eval_limits.max_memory_mb,
            },
            "thresholds": {
                "min_reward": self.thresholds.min_reward,
                "max_risk": self.thresholds.max_risk,
                "max_complexity": self.thresholds.max_complexity,
                "min_sharpe_ratio": self.thresholds.min_sharpe_ratio,
                "max_drawdown": self.thresholds.max_drawdown,
            },
            "ranking_weights": self.ranking_weights,
            "use_pareto": self.use_pareto,
            "requires_governance_approval": self.requires_governance_approval,
            "governance_quorum": self.governance_quorum,
            "upgrade_cooldown_seconds": self.upgrade_cooldown_seconds,
        }


# -----------------------------------------------------------------------------
# Goal State (Mutable, held by Coordinator)
# -----------------------------------------------------------------------------

@dataclass
class GoalState:
    """
    Per-goal state maintained by an IAN coordinator.
    
    This is mutable and updated by `process_contribution`.
    Contains references to MMR log, leaderboard, and dedup structures.
    """
    goal_id: GoalID
    spec: GoalSpec
    
    # These are initialized by the coordinator
    log: Any = None  # MerkleMountainRange
    leaderboard: Any = None  # Leaderboard or ParetoFrontier
    dedup: Any = None  # DedupService
    
    # State roots (for Tau anchoring)
    log_root: bytes = field(default_factory=lambda: b"\x00" * 32)
    leaderboard_root: bytes = field(default_factory=lambda: b"\x00" * 32)
    
    # Active policy
    active_policy_hash: Optional[bytes] = None
    last_upgrade_timestamp_ms: int = 0
    
    # Statistics
    total_contributions: int = 0
    accepted_contributions: int = 0
    rejected_contributions: int = 0
    
    def update_roots(self) -> None:
        """Update cached state roots from underlying structures."""
        if self.log is not None:
            self.log_root = self.log.get_root()
        if self.leaderboard is not None:
            self.leaderboard_root = self.leaderboard.get_root()
