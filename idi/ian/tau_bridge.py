"""
IAN-Tau Bridge - Transaction types and state synchronization.

Implements the bridge between IAN coordinator and Tau Net:
- Transaction types: IAN_UPGRADE, IAN_LOG_COMMIT, IAN_GOAL_REGISTER
- Wire format serialization compatible with Tau Testnet sendtx.py
- State stream definitions for active_policy_hash, ian_log_root, ian_lb_root

Design Principles:
- All transactions are deterministic and verifiable
- State roots are cryptographically bound to Tau consensus
- Upgrades require governance approval (optional)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .models import GoalID, GoalSpec, ContributionMeta


logger = logging.getLogger(__name__)


# =============================================================================
# Transaction Types
# =============================================================================

class IanTxType(Enum):
    """IAN transaction types for Tau Net."""
    GOAL_REGISTER = "IAN_GOAL_REGISTER"
    LOG_COMMIT = "IAN_LOG_COMMIT"
    UPGRADE = "IAN_UPGRADE"


@dataclass(frozen=True)
class IanGoalRegisterTx:
    """
    Transaction to register a new goal on Tau Net.
    
    Wire format:
    {
        "type": "IAN_GOAL_REGISTER",
        "goal_id": "<string>",
        "goal_spec_hash": "<hex>",
        "name": "<string>",
        "description": "<string>",
        "invariant_ids": ["I1", "I2", ...],
        "thresholds": {...},
        "timestamp_ms": <int>,
        "signature": "<hex>"
    }
    """
    goal_id: GoalID
    goal_spec_hash: bytes  # SHA-256 of serialized GoalSpec
    name: str
    description: str
    invariant_ids: List[str]
    thresholds: Dict[str, float]
    timestamp_ms: int
    signature: Optional[bytes] = None  # Governance signature
    
    @classmethod
    def from_goal_spec(cls, spec: GoalSpec, timestamp_ms: Optional[int] = None) -> "IanGoalRegisterTx":
        """Create registration transaction from GoalSpec."""
        spec_bytes = json.dumps(spec.to_dict(), sort_keys=True).encode("utf-8")
        spec_hash = hashlib.sha256(spec_bytes).digest()
        
        return cls(
            goal_id=spec.goal_id,
            goal_spec_hash=spec_hash,
            name=spec.name,
            description=spec.description,
            invariant_ids=spec.invariant_ids,
            thresholds={
                "min_reward": spec.thresholds.min_reward,
                "max_risk": spec.thresholds.max_risk,
                "max_complexity": spec.thresholds.max_complexity,
            },
            timestamp_ms=timestamp_ms or int(time.time() * 1000),
        )
    
    def to_wire(self) -> bytes:
        """Serialize to wire format (JSON bytes)."""
        data = {
            "type": IanTxType.GOAL_REGISTER.value,
            "goal_id": str(self.goal_id),
            "goal_spec_hash": self.goal_spec_hash.hex(),
            "name": self.name,
            "description": self.description,
            "invariant_ids": self.invariant_ids,
            "thresholds": self.thresholds,
            "timestamp_ms": self.timestamp_ms,
        }
        if self.signature:
            data["signature"] = self.signature.hex()
        return json.dumps(data, sort_keys=True, separators=(',', ':')).encode("utf-8")
    
    @classmethod
    def from_wire(cls, data: bytes) -> "IanGoalRegisterTx":
        """Deserialize from wire format."""
        obj = json.loads(data.decode("utf-8"))
        
        if obj.get("type") != IanTxType.GOAL_REGISTER.value:
            raise ValueError(f"Invalid tx type: {obj.get('type')}")
        
        return cls(
            goal_id=GoalID(obj["goal_id"]),
            goal_spec_hash=bytes.fromhex(obj["goal_spec_hash"]),
            name=obj["name"],
            description=obj.get("description", ""),
            invariant_ids=obj.get("invariant_ids", []),
            thresholds=obj.get("thresholds", {}),
            timestamp_ms=obj["timestamp_ms"],
            signature=bytes.fromhex(obj["signature"]) if obj.get("signature") else None,
        )
    
    @property
    def tx_hash(self) -> bytes:
        """Compute transaction hash."""
        return hashlib.sha256(self.to_wire()).digest()


@dataclass(frozen=True)
class IanLogCommitTx:
    """
    Transaction to commit IAN log state to Tau Net.
    
    Published periodically (e.g., every N contributions or every T seconds)
    to anchor IAN log root on-chain.
    
    Wire format:
    {
        "type": "IAN_LOG_COMMIT",
        "goal_id": "<string>",
        "log_root": "<hex>",
        "log_size": <int>,
        "leaderboard_root": "<hex>",
        "leaderboard_size": <int>,
        "prev_commit_hash": "<hex>",
        "timestamp_ms": <int>,
        "signature": "<hex>"
    }
    """
    goal_id: GoalID
    log_root: bytes  # MMR root hash
    log_size: int  # Number of contributions in log
    leaderboard_root: bytes  # Leaderboard state hash
    leaderboard_size: int  # Number of entries on leaderboard
    prev_commit_hash: Optional[bytes]  # Hash of previous commit (chain)
    timestamp_ms: int
    signature: Optional[bytes] = None
    
    def to_wire(self) -> bytes:
        """Serialize to wire format."""
        data = {
            "type": IanTxType.LOG_COMMIT.value,
            "goal_id": str(self.goal_id),
            "log_root": self.log_root.hex(),
            "log_size": self.log_size,
            "leaderboard_root": self.leaderboard_root.hex(),
            "leaderboard_size": self.leaderboard_size,
            "prev_commit_hash": self.prev_commit_hash.hex() if self.prev_commit_hash else None,
            "timestamp_ms": self.timestamp_ms,
        }
        if self.signature:
            data["signature"] = self.signature.hex()
        return json.dumps(data, sort_keys=True, separators=(',', ':')).encode("utf-8")
    
    @classmethod
    def from_wire(cls, data: bytes) -> "IanLogCommitTx":
        """Deserialize from wire format."""
        obj = json.loads(data.decode("utf-8"))
        
        if obj.get("type") != IanTxType.LOG_COMMIT.value:
            raise ValueError(f"Invalid tx type: {obj.get('type')}")
        
        return cls(
            goal_id=GoalID(obj["goal_id"]),
            log_root=bytes.fromhex(obj["log_root"]),
            log_size=obj["log_size"],
            leaderboard_root=bytes.fromhex(obj["leaderboard_root"]),
            leaderboard_size=obj["leaderboard_size"],
            prev_commit_hash=bytes.fromhex(obj["prev_commit_hash"]) if obj.get("prev_commit_hash") else None,
            timestamp_ms=obj["timestamp_ms"],
            signature=bytes.fromhex(obj["signature"]) if obj.get("signature") else None,
        )
    
    @property
    def tx_hash(self) -> bytes:
        """Compute transaction hash."""
        return hashlib.sha256(self.to_wire()).digest()


@dataclass(frozen=True)
class IanUpgradeTx:
    """
    Transaction to upgrade active policy on Tau Net.
    
    Triggered when a new contribution becomes the top-ranked policy.
    May require governance approval based on GoalSpec.
    
    Wire format:
    {
        "type": "IAN_UPGRADE",
        "goal_id": "<string>",
        "pack_hash": "<hex>",
        "prev_pack_hash": "<hex>",
        "score": <float>,
        "metrics": {...},
        "log_index": <int>,
        "log_root": "<hex>",
        "contributor_id": "<string>",
        "timestamp_ms": <int>,
        "governance_signatures": ["<hex>", ...],
        "cooldown_ok": <bool>
    }
    """
    goal_id: GoalID
    pack_hash: bytes  # Hash of new active policy
    prev_pack_hash: Optional[bytes]  # Hash of previous policy (None if first)
    score: float  # Score of new policy
    metrics: Dict[str, float]  # Evaluation metrics
    log_index: int  # Index in contribution log
    log_root: bytes  # Log root at time of upgrade
    contributor_id: str  # Contributor who submitted
    timestamp_ms: int
    governance_signatures: List[bytes] = field(default_factory=list)
    cooldown_ok: bool = True  # Upgrade cooldown respected
    
    @classmethod
    def from_contribution_meta(
        cls,
        meta: ContributionMeta,
        goal_id: GoalID,
        log_root: bytes,
        prev_pack_hash: Optional[bytes] = None,
        timestamp_ms: Optional[int] = None,
    ) -> "IanUpgradeTx":
        """Create upgrade transaction from ContributionMeta."""
        return cls(
            goal_id=goal_id,
            pack_hash=meta.pack_hash,
            prev_pack_hash=prev_pack_hash,
            score=meta.score,
            metrics={
                "reward": meta.metrics.reward,
                "risk": meta.metrics.risk,
                "complexity": meta.metrics.complexity,
            },
            log_index=meta.log_index,
            log_root=log_root,
            contributor_id=meta.contributor_id,
            timestamp_ms=timestamp_ms or int(time.time() * 1000),
        )
    
    def to_wire(self) -> bytes:
        """Serialize to wire format."""
        data = {
            "type": IanTxType.UPGRADE.value,
            "goal_id": str(self.goal_id),
            "pack_hash": self.pack_hash.hex(),
            "prev_pack_hash": self.prev_pack_hash.hex() if self.prev_pack_hash else None,
            "score": self.score,
            "metrics": self.metrics,
            "log_index": self.log_index,
            "log_root": self.log_root.hex(),
            "contributor_id": self.contributor_id,
            "timestamp_ms": self.timestamp_ms,
            "cooldown_ok": self.cooldown_ok,
        }
        if self.governance_signatures:
            data["governance_signatures"] = [sig.hex() for sig in self.governance_signatures]
        return json.dumps(data, sort_keys=True, separators=(',', ':')).encode("utf-8")
    
    @classmethod
    def from_wire(cls, data: bytes) -> "IanUpgradeTx":
        """Deserialize from wire format."""
        obj = json.loads(data.decode("utf-8"))
        
        if obj.get("type") != IanTxType.UPGRADE.value:
            raise ValueError(f"Invalid tx type: {obj.get('type')}")
        
        return cls(
            goal_id=GoalID(obj["goal_id"]),
            pack_hash=bytes.fromhex(obj["pack_hash"]),
            prev_pack_hash=bytes.fromhex(obj["prev_pack_hash"]) if obj.get("prev_pack_hash") else None,
            score=obj["score"],
            metrics=obj.get("metrics", {}),
            log_index=obj["log_index"],
            log_root=bytes.fromhex(obj["log_root"]),
            contributor_id=obj.get("contributor_id", ""),
            timestamp_ms=obj["timestamp_ms"],
            governance_signatures=[
                bytes.fromhex(sig) for sig in obj.get("governance_signatures", [])
            ],
            cooldown_ok=obj.get("cooldown_ok", True),
        )
    
    @property
    def tx_hash(self) -> bytes:
        """Compute transaction hash."""
        return hashlib.sha256(self.to_wire()).digest()


# Union type for all IAN transactions
IanTransaction = IanGoalRegisterTx | IanLogCommitTx | IanUpgradeTx


def parse_ian_tx(data: bytes) -> IanTransaction:
    """Parse any IAN transaction from wire format."""
    obj = json.loads(data.decode("utf-8"))
    tx_type = obj.get("type")
    
    if tx_type == IanTxType.GOAL_REGISTER.value:
        return IanGoalRegisterTx.from_wire(data)
    elif tx_type == IanTxType.LOG_COMMIT.value:
        return IanLogCommitTx.from_wire(data)
    elif tx_type == IanTxType.UPGRADE.value:
        return IanUpgradeTx.from_wire(data)
    else:
        raise ValueError(f"Unknown IAN transaction type: {tx_type}")


# =============================================================================
# Tau State Streams
# =============================================================================

@dataclass
class IanTauState:
    """
    IAN state as represented in Tau streams.
    
    Corresponds to Tau state variables:
    - ian_goal_registered[goal_id] : bool
    - ian_active_policy[goal_id] : bytes (pack_hash)
    - ian_log_root[goal_id] : bytes
    - ian_lb_root[goal_id] : bytes
    - ian_upgrade_count[goal_id] : int
    - ian_last_commit_hash[goal_id] : bytes
    """
    goal_id: GoalID
    registered: bool = False
    active_policy_hash: Optional[bytes] = None
    log_root: bytes = field(default_factory=lambda: b'\x00' * 32)
    leaderboard_root: bytes = field(default_factory=lambda: b'\x00' * 32)
    upgrade_count: int = 0
    last_commit_hash: Optional[bytes] = None
    last_upgrade_timestamp_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "goal_id": str(self.goal_id),
            "registered": self.registered,
            "active_policy_hash": self.active_policy_hash.hex() if self.active_policy_hash else None,
            "log_root": self.log_root.hex(),
            "leaderboard_root": self.leaderboard_root.hex(),
            "upgrade_count": self.upgrade_count,
            "last_commit_hash": self.last_commit_hash.hex() if self.last_commit_hash else None,
            "last_upgrade_timestamp_ms": self.last_upgrade_timestamp_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IanTauState":
        """Deserialize from dictionary."""
        return cls(
            goal_id=GoalID(data["goal_id"]),
            registered=data.get("registered", False),
            active_policy_hash=bytes.fromhex(data["active_policy_hash"]) if data.get("active_policy_hash") else None,
            log_root=bytes.fromhex(data["log_root"]) if data.get("log_root") else b'\x00' * 32,
            leaderboard_root=bytes.fromhex(data["leaderboard_root"]) if data.get("leaderboard_root") else b'\x00' * 32,
            upgrade_count=data.get("upgrade_count", 0),
            last_commit_hash=bytes.fromhex(data["last_commit_hash"]) if data.get("last_commit_hash") else None,
            last_upgrade_timestamp_ms=data.get("last_upgrade_timestamp_ms", 0),
        )


# =============================================================================
# Tau Bridge
# =============================================================================

class TauTransactionSender(Protocol):
    """Protocol for sending transactions to Tau Net."""
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """
        Send a transaction to Tau Net.
        
        Returns:
            (success, tx_hash_or_error)
        """
        ...


class MockTauSender:
    """Mock transaction sender for testing."""
    
    def __init__(self) -> None:
        self.sent_txs: List[bytes] = []
    
    def send_tx(self, tx_data: bytes) -> Tuple[bool, str]:
        """Store transaction and return success."""
        self.sent_txs.append(tx_data)
        tx_hash = hashlib.sha256(tx_data).hexdigest()
        logger.info(f"MockTauSender: sent tx {tx_hash[:16]}...")
        return True, tx_hash


@dataclass
class TauBridgeConfig:
    """Configuration for TauBridge."""
    commit_interval_seconds: int = 300  # Commit log every 5 minutes
    commit_threshold_contributions: int = 100  # Or every 100 contributions
    require_governance_for_upgrade: bool = False
    upgrade_cooldown_seconds: int = 86400  # 24 hours between upgrades
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class TauBridge:
    """
    Bridge between IAN coordinator and Tau Net.
    
    Responsibilities:
    - Send IAN transactions to Tau Net
    - Track Tau state for each goal
    - Manage upgrade governance
    - Handle transaction retries
    """
    
    def __init__(
        self,
        sender: Optional[TauTransactionSender] = None,
        config: Optional[TauBridgeConfig] = None,
    ) -> None:
        """
        Initialize TauBridge.
        
        Args:
            sender: Transaction sender (defaults to mock for testing)
            config: Bridge configuration
        """
        self.sender = sender or MockTauSender()
        self.config = config or TauBridgeConfig()
        self._states: Dict[str, IanTauState] = {}
        self._pending_commits: Dict[str, IanLogCommitTx] = {}
        self._last_commit_time: Dict[str, float] = {}
        self._contributions_since_commit: Dict[str, int] = {}
    
    def get_state(self, goal_id: GoalID) -> IanTauState:
        """Get Tau state for a goal."""
        key = str(goal_id)
        if key not in self._states:
            self._states[key] = IanTauState(goal_id=goal_id)
        return self._states[key]
    
    def register_goal(self, spec: GoalSpec) -> Tuple[bool, str]:
        """
        Register a goal on Tau Net.
        
        Args:
            spec: Goal specification to register
            
        Returns:
            (success, tx_hash_or_error)
        """
        tx = IanGoalRegisterTx.from_goal_spec(spec)
        
        success, result = self._send_with_retry(tx.to_wire())
        
        if success:
            state = self.get_state(spec.goal_id)
            state.registered = True
            self._contributions_since_commit[str(spec.goal_id)] = 0
            self._last_commit_time[str(spec.goal_id)] = time.time()
            logger.info(f"Goal {spec.goal_id} registered on Tau Net: {result}")
        
        return success, result
    
    def commit_log(
        self,
        goal_id: GoalID,
        log_root: bytes,
        log_size: int,
        leaderboard_root: bytes,
        leaderboard_size: int,
    ) -> Tuple[bool, str]:
        """
        Commit IAN log state to Tau Net.
        
        Args:
            goal_id: Goal identifier
            log_root: Current MMR root
            log_size: Number of contributions
            leaderboard_root: Current leaderboard hash
            leaderboard_size: Leaderboard entry count
            
        Returns:
            (success, tx_hash_or_error)
        """
        state = self.get_state(goal_id)
        
        tx = IanLogCommitTx(
            goal_id=goal_id,
            log_root=log_root,
            log_size=log_size,
            leaderboard_root=leaderboard_root,
            leaderboard_size=leaderboard_size,
            prev_commit_hash=state.last_commit_hash,
            timestamp_ms=int(time.time() * 1000),
        )
        
        success, result = self._send_with_retry(tx.to_wire())
        
        if success:
            state.log_root = log_root
            state.leaderboard_root = leaderboard_root
            state.last_commit_hash = tx.tx_hash
            self._last_commit_time[str(goal_id)] = time.time()
            self._contributions_since_commit[str(goal_id)] = 0
            logger.info(f"Log commit for {goal_id}: root={log_root.hex()[:16]}...")
        
        return success, result
    
    def upgrade_policy(
        self,
        goal_id: GoalID,
        new_policy: ContributionMeta,
        log_root: bytes,
        governance_signatures: Optional[List[bytes]] = None,
    ) -> Tuple[bool, str]:
        """
        Upgrade active policy on Tau Net.
        
        Args:
            goal_id: Goal identifier
            new_policy: New top-ranked policy
            log_root: Current log root
            governance_signatures: Required if governance approval needed
            
        Returns:
            (success, tx_hash_or_error)
        """
        state = self.get_state(goal_id)
        
        # Check cooldown - always enforced regardless of governance setting
        # to prevent rapid policy churn that could destabilize the network
        now_ms = int(time.time() * 1000)
        cooldown_ms = self.config.upgrade_cooldown_seconds * 1000
        cooldown_ok = (now_ms - state.last_upgrade_timestamp_ms) >= cooldown_ms
        
        if not cooldown_ok:
            return False, f"Upgrade cooldown not elapsed ({self.config.upgrade_cooldown_seconds}s)"
        
        tx = IanUpgradeTx.from_contribution_meta(
            meta=new_policy,
            goal_id=goal_id,
            log_root=log_root,
            prev_pack_hash=state.active_policy_hash,
            timestamp_ms=now_ms,
        )
        
        # Replace with governance signatures if provided
        if governance_signatures:
            tx = IanUpgradeTx(
                **{**tx.__dict__, "governance_signatures": governance_signatures, "cooldown_ok": cooldown_ok}
            )
        
        success, result = self._send_with_retry(tx.to_wire())
        
        if success:
            state.active_policy_hash = new_policy.pack_hash
            state.upgrade_count += 1
            state.last_upgrade_timestamp_ms = now_ms
            logger.info(
                f"Policy upgrade for {goal_id}: "
                f"pack={new_policy.pack_hash.hex()[:16]}... "
                f"score={new_policy.score:.4f}"
            )
        
        return success, result
    
    def should_commit(self, goal_id: GoalID) -> bool:
        """
        Check if a log commit should be triggered.
        
        Based on:
        - Time since last commit
        - Number of contributions since last commit
        """
        key = str(goal_id)
        
        # Time-based
        last_commit = self._last_commit_time.get(key, 0)
        if time.time() - last_commit >= self.config.commit_interval_seconds:
            return True
        
        # Contribution count-based
        count = self._contributions_since_commit.get(key, 0)
        if count >= self.config.commit_threshold_contributions:
            return True
        
        return False
    
    def record_contribution(self, goal_id: GoalID) -> None:
        """Record a contribution for commit tracking."""
        key = str(goal_id)
        self._contributions_since_commit[key] = self._contributions_since_commit.get(key, 0) + 1
    
    def _send_with_retry(self, tx_data: bytes) -> Tuple[bool, str]:
        """Send transaction with retry logic."""
        last_error = "unknown error"
        
        for attempt in range(self.config.retry_attempts):
            try:
                success, result = self.sender.send_tx(tx_data)
                if success:
                    return True, result
                last_error = result
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tx send attempt {attempt + 1} failed: {e}")
            
            if attempt < self.config.retry_attempts - 1:
                time.sleep(self.config.retry_delay_seconds * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Tx send failed after {self.config.retry_attempts} attempts: {last_error}")
        return False, last_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize bridge state."""
        return {
            "states": {k: v.to_dict() for k, v in self._states.items()},
            "last_commit_time": self._last_commit_time,
            "contributions_since_commit": self._contributions_since_commit,
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        sender: Optional[TauTransactionSender] = None,
        config: Optional[TauBridgeConfig] = None,
    ) -> "TauBridge":
        """Deserialize bridge state."""
        bridge = cls(sender=sender, config=config)
        
        for key, state_data in data.get("states", {}).items():
            bridge._states[key] = IanTauState.from_dict(state_data)
        
        bridge._last_commit_time = data.get("last_commit_time", {})
        bridge._contributions_since_commit = data.get("contributions_since_commit", {})
        
        return bridge


# =============================================================================
# Integration with IANCoordinator
# =============================================================================

class TauIntegratedCoordinator:
    """
    Wrapper that integrates IANCoordinator with TauBridge.
    
    Automatically:
    - Commits log to Tau periodically
    - Triggers upgrades when policy changes
    - Tracks Tau state
    """
    
    def __init__(
        self,
        coordinator: "IANCoordinator",  # Forward reference
        bridge: Optional[TauBridge] = None,
    ) -> None:
        from .coordinator import IANCoordinator
        
        self.coordinator = coordinator
        self.bridge = bridge or TauBridge()
        self._last_active_policy: Optional[bytes] = None
    
    def process_contribution(self, contrib: "Contribution") -> "ProcessResult":
        """
        Process contribution and handle Tau integration.
        
        Steps:
        1. Delegate to coordinator
        2. Record contribution for commit tracking
        3. Check if commit needed
        4. Check if upgrade triggered
        """
        from .coordinator import ProcessResult
        from .models import Contribution
        
        result = self.coordinator.process_contribution(contrib)
        
        if result.accepted:
            goal_id = self.coordinator.goal_spec.goal_id
            
            # Record for commit tracking
            self.bridge.record_contribution(goal_id)
            
            # Check if commit needed
            if self.bridge.should_commit(goal_id):
                self.bridge.commit_log(
                    goal_id=goal_id,
                    log_root=self.coordinator.get_log_root(),
                    log_size=self.coordinator.state.log.size,
                    leaderboard_root=self.coordinator.get_leaderboard_root(),
                    leaderboard_size=len(self.coordinator.state.leaderboard),
                )
            
            # Check if upgrade triggered
            active = self.coordinator.get_active_policy()
            if active and active.pack_hash != self._last_active_policy:
                self.bridge.upgrade_policy(
                    goal_id=goal_id,
                    new_policy=active,
                    log_root=self.coordinator.get_log_root(),
                )
                self._last_active_policy = active.pack_hash
        
        return result
    
    def register_on_tau(self) -> Tuple[bool, str]:
        """Register the goal on Tau Net."""
        return self.bridge.register_goal(self.coordinator.goal_spec)
    
    def force_commit(self) -> Tuple[bool, str]:
        """Force a log commit to Tau Net."""
        goal_id = self.coordinator.goal_spec.goal_id
        return self.bridge.commit_log(
            goal_id=goal_id,
            log_root=self.coordinator.get_log_root(),
            log_size=self.coordinator.state.log.size,
            leaderboard_root=self.coordinator.get_leaderboard_root(),
            leaderboard_size=len(self.coordinator.state.leaderboard),
        )


def create_tau_integrated_coordinator(
    goal_spec: GoalSpec,
    tau_sender: Optional[TauTransactionSender] = None,
    bridge_config: Optional[TauBridgeConfig] = None,
    **coordinator_kwargs,
) -> TauIntegratedCoordinator:
    """
    Factory function for creating a Tau-integrated coordinator.
    
    Args:
        goal_spec: Goal specification
        tau_sender: Transaction sender for Tau Net
        bridge_config: TauBridge configuration
        **coordinator_kwargs: Additional kwargs for IANCoordinator
        
    Returns:
        TauIntegratedCoordinator instance
    """
    from .idi_integration import create_idi_coordinator
    
    coordinator = create_idi_coordinator(goal_spec, **coordinator_kwargs)
    bridge = TauBridge(sender=tau_sender, config=bridge_config)
    
    return TauIntegratedCoordinator(coordinator=coordinator, bridge=bridge)
