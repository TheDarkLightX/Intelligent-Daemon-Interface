"""
IAN Economic Security - Bonding, staking, and slashing for Tau Net integration.

Provides:
1. Committer bonding for commit authorization
2. Slashing conditions and execution
3. Reward distribution for challengers
4. Economic configuration for Tau Net

Design Principles:
- Economic incentives align node behavior with network goals
- Slashing makes fraud economically irrational
- Bonds create skin-in-the-game for committers

Integration with Tau Net:
- Bonds are held in Tau Net state streams
- Slashing is executed via Tau transactions
- Reward distribution is handled by Tau rules
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .fraud import FraudProof, FraudType

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EconomicConfig:
    """
    Economic parameters for IAN on Tau Net.
    
    All amounts are in Tau's base unit (e.g., 1 TAU = 1_000_000 units).
    """
    
    # Committer bonds
    min_committer_bond: int = 1_000_000_000  # 1000 TAU
    max_committer_bond: int = 100_000_000_000  # 100,000 TAU
    
    # Challenger bonds
    challenge_bond: int = 100_000_000  # 100 TAU
    
    # Slashing
    slash_percentage: float = 0.50  # 50% of bond
    slash_escalation: float = 1.5  # Multiply for repeat offenders
    max_slash_percentage: float = 1.0  # 100% max
    
    # Rewards
    challenger_reward_percentage: float = 0.25  # 25% of slashed amount
    burn_percentage: float = 0.25  # 25% burned
    # Remaining 50% goes to protocol treasury
    
    # Timeouts
    bond_lock_period_seconds: int = 86400 * 7  # 7 days
    challenge_period_seconds: int = 3600  # 1 hour
    
    # Limits
    max_commits_per_epoch: int = 100
    epoch_duration_seconds: int = 3600  # 1 hour epochs


# =============================================================================
# Bond State
# =============================================================================

class BondStatus(Enum):
    """Status of a bond."""
    ACTIVE = "active"
    LOCKED = "locked"  # During challenge period
    SLASHED = "slashed"
    WITHDRAWN = "withdrawn"


@dataclass
class CommitterBond:
    """
    Bond posted by a committer.
    
    Bonds are required to submit IAN_LOG_COMMIT and IAN_UPGRADE
    transactions to Tau Net.
    """
    committer_id: str  # Node ID / public key
    goal_id: str
    amount: int
    status: BondStatus = BondStatus.ACTIVE
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    locked_until_ms: Optional[int] = None
    
    # Tau Net reference
    bond_tx_hash: Optional[bytes] = None
    
    # History
    slash_count: int = 0
    total_slashed: int = 0
    commits_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "committer_id": self.committer_id,
            "goal_id": self.goal_id,
            "amount": self.amount,
            "status": self.status.value,
            "created_at_ms": self.created_at_ms,
            "locked_until_ms": self.locked_until_ms,
            "bond_tx_hash": self.bond_tx_hash.hex() if self.bond_tx_hash else None,
            "slash_count": self.slash_count,
            "total_slashed": self.total_slashed,
            "commits_count": self.commits_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitterBond":
        return cls(
            committer_id=data["committer_id"],
            goal_id=data["goal_id"],
            amount=data["amount"],
            status=BondStatus(data["status"]),
            created_at_ms=data["created_at_ms"],
            locked_until_ms=data.get("locked_until_ms"),
            bond_tx_hash=bytes.fromhex(data["bond_tx_hash"]) if data.get("bond_tx_hash") else None,
            slash_count=data.get("slash_count", 0),
            total_slashed=data.get("total_slashed", 0),
            commits_count=data.get("commits_count", 0),
        )
    
    def can_commit(self) -> bool:
        """Check if this bond allows committing."""
        return self.status == BondStatus.ACTIVE
    
    def effective_bond(self) -> int:
        """Get effective bond amount after slashing."""
        return self.amount - self.total_slashed
    
    def lock(self, duration_ms: int) -> None:
        """Lock bond during challenge period."""
        self.status = BondStatus.LOCKED
        self.locked_until_ms = int(time.time() * 1000) + duration_ms
    
    def unlock(self) -> None:
        """Unlock bond after challenge period."""
        if self.status == BondStatus.LOCKED:
            self.status = BondStatus.ACTIVE
            self.locked_until_ms = None


@dataclass
class ChallengeBond:
    """
    Bond posted by a challenger.
    
    Required to submit fraud proof challenges.
    Returned if challenge is valid, slashed if invalid.
    """
    challenger_id: str
    goal_id: str
    challenged_commit_hash: bytes
    amount: int
    status: BondStatus = BondStatus.ACTIVE
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Resolution
    challenge_valid: Optional[bool] = None
    resolution_tx_hash: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenger_id": self.challenger_id,
            "goal_id": self.goal_id,
            "challenged_commit_hash": self.challenged_commit_hash.hex(),
            "amount": self.amount,
            "status": self.status.value,
            "created_at_ms": self.created_at_ms,
            "challenge_valid": self.challenge_valid,
            "resolution_tx_hash": self.resolution_tx_hash.hex() if self.resolution_tx_hash else None,
        }


# =============================================================================
# Slashing
# =============================================================================

@dataclass
class SlashEvent:
    """
    Record of a slashing event.
    """
    committer_id: str
    goal_id: str
    commit_hash: bytes
    fraud_type: str
    amount_slashed: int
    challenger_id: str
    challenger_reward: int
    burn_amount: int
    treasury_amount: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    tau_tx_hash: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "committer_id": self.committer_id,
            "goal_id": self.goal_id,
            "commit_hash": self.commit_hash.hex(),
            "fraud_type": self.fraud_type,
            "amount_slashed": self.amount_slashed,
            "challenger_id": self.challenger_id,
            "challenger_reward": self.challenger_reward,
            "burn_amount": self.burn_amount,
            "treasury_amount": self.treasury_amount,
            "timestamp_ms": self.timestamp_ms,
            "tau_tx_hash": self.tau_tx_hash.hex() if self.tau_tx_hash else None,
        }


# =============================================================================
# Economic Manager
# =============================================================================

class EconomicManager:
    """
    Manage economic security for IAN.
    
    Responsibilities:
    - Track committer and challenger bonds
    - Calculate slashing amounts
    - Distribute rewards
    - Generate Tau Net transactions
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[EconomicConfig] = None,
    ):
        self._node_id = node_id
        self._config = config or EconomicConfig()
        
        # Bond tracking
        self._committer_bonds: Dict[str, CommitterBond] = {}  # committer_id -> bond
        self._challenger_bonds: Dict[bytes, ChallengeBond] = {}  # commit_hash -> bond
        
        # History
        self._slash_events: List[SlashEvent] = []
        
        # Callbacks
        self._submit_tx: Optional[Callable[[bytes], Tuple[bool, str]]] = None
    
    def set_tx_callback(
        self,
        submit_tx: Callable[[bytes], Tuple[bool, str]],
    ) -> None:
        """Set callback for Tau transaction submission."""
        self._submit_tx = submit_tx
    
    # -------------------------------------------------------------------------
    # Committer Bonds
    # -------------------------------------------------------------------------
    
    def register_bond(
        self,
        committer_id: str,
        goal_id: str,
        amount: int,
        bond_tx_hash: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Register a committer bond.
        
        Args:
            committer_id: Node ID of committer
            goal_id: Goal this bond covers
            amount: Bond amount
            bond_tx_hash: Tau transaction hash
            
        Returns:
            (success, reason)
            
        Preconditions:
        - Amount >= min_committer_bond
        
        Postconditions:
        - Bond is registered and active
        """
        if amount < self._config.min_committer_bond:
            return False, f"bond below minimum: {amount} < {self._config.min_committer_bond}"
        
        if amount > self._config.max_committer_bond:
            return False, f"bond exceeds maximum: {amount} > {self._config.max_committer_bond}"
        
        key = f"{committer_id}:{goal_id}"
        
        if key in self._committer_bonds:
            # Increase existing bond
            self._committer_bonds[key].amount += amount
        else:
            # New bond
            self._committer_bonds[key] = CommitterBond(
                committer_id=committer_id,
                goal_id=goal_id,
                amount=amount,
                bond_tx_hash=bond_tx_hash,
            )
        
        logger.info(f"Bond registered: {committer_id[:16]}... for {goal_id}, amount={amount}")
        return True, "bond registered"
    
    def get_bond(self, committer_id: str, goal_id: str) -> Optional[CommitterBond]:
        """Get committer bond."""
        key = f"{committer_id}:{goal_id}"
        return self._committer_bonds.get(key)
    
    def can_commit(self, committer_id: str, goal_id: str) -> Tuple[bool, str]:
        """
        Check if committer can submit a commit.
        
        Returns:
            (can_commit, reason)
        """
        bond = self.get_bond(committer_id, goal_id)
        
        if not bond:
            return False, "no bond registered"
        
        if not bond.can_commit():
            return False, f"bond status: {bond.status.value}"
        
        if bond.effective_bond() < self._config.min_committer_bond:
            return False, "effective bond below minimum (too much slashed)"
        
        return True, "authorized"
    
    def record_commit(self, committer_id: str, goal_id: str) -> None:
        """Record a successful commit."""
        bond = self.get_bond(committer_id, goal_id)
        if bond:
            bond.commits_count += 1
    
    # -------------------------------------------------------------------------
    # Challenger Bonds
    # -------------------------------------------------------------------------
    
    def register_challenge_bond(
        self,
        challenger_id: str,
        goal_id: str,
        challenged_commit_hash: bytes,
    ) -> Tuple[bool, str]:
        """
        Register a challenger bond.
        
        Returns:
            (success, reason)
        """
        if challenged_commit_hash in self._challenger_bonds:
            return False, "challenge already exists for this commit"
        
        self._challenger_bonds[challenged_commit_hash] = ChallengeBond(
            challenger_id=challenger_id,
            goal_id=goal_id,
            challenged_commit_hash=challenged_commit_hash,
            amount=self._config.challenge_bond,
        )
        
        return True, "challenge bond registered"
    
    # -------------------------------------------------------------------------
    # Slashing
    # -------------------------------------------------------------------------
    
    def calculate_slash_amount(
        self,
        committer_id: str,
        goal_id: str,
    ) -> int:
        """
        Calculate slash amount for a committer.
        
        Takes into account:
        - Base slash percentage
        - Repeat offender escalation
        """
        bond = self.get_bond(committer_id, goal_id)
        if not bond:
            return 0
        
        # Base slash
        slash_pct = self._config.slash_percentage
        
        # Escalate for repeat offenders
        for _ in range(bond.slash_count):
            slash_pct *= self._config.slash_escalation
        
        # Cap at max
        slash_pct = min(slash_pct, self._config.max_slash_percentage)
        
        return int(bond.effective_bond() * slash_pct)
    
    def execute_slash(
        self,
        committer_id: str,
        goal_id: str,
        commit_hash: bytes,
        fraud_type: str,
        challenger_id: str,
    ) -> Optional[SlashEvent]:
        """
        Execute slashing for proven fraud.
        
        Args:
            committer_id: ID of committer being slashed
            goal_id: Goal ID
            commit_hash: Hash of fraudulent commit
            fraud_type: Type of fraud proven
            challenger_id: ID of challenger who proved fraud
            
        Returns:
            SlashEvent if successful, None otherwise
        """
        bond = self.get_bond(committer_id, goal_id)
        if not bond:
            logger.error(f"Cannot slash: no bond for {committer_id}")
            return None
        
        # Calculate amounts
        slash_amount = self.calculate_slash_amount(committer_id, goal_id)
        challenger_reward = int(slash_amount * self._config.challenger_reward_percentage)
        burn_amount = int(slash_amount * self._config.burn_percentage)
        treasury_amount = slash_amount - challenger_reward - burn_amount
        
        # Update bond
        bond.total_slashed += slash_amount
        bond.slash_count += 1
        
        if bond.effective_bond() < self._config.min_committer_bond:
            bond.status = BondStatus.SLASHED
        
        # Create event
        event = SlashEvent(
            committer_id=committer_id,
            goal_id=goal_id,
            commit_hash=commit_hash,
            fraud_type=fraud_type,
            amount_slashed=slash_amount,
            challenger_id=challenger_id,
            challenger_reward=challenger_reward,
            burn_amount=burn_amount,
            treasury_amount=treasury_amount,
        )
        
        self._slash_events.append(event)
        
        logger.warning(
            f"SLASH: {committer_id[:16]}... slashed {slash_amount} "
            f"for {fraud_type} on goal {goal_id}"
        )
        
        return event
    
    def generate_slash_transaction(
        self,
        event: SlashEvent,
    ) -> bytes:
        """
        Generate Tau Net transaction for slashing.
        
        Returns:
            Serialized transaction bytes
        """
        tx_data = {
            "type": "IAN_SLASH",
            "committer_id": event.committer_id,
            "goal_id": event.goal_id,
            "commit_hash": event.commit_hash.hex(),
            "fraud_type": event.fraud_type,
            "amount_slashed": event.amount_slashed,
            "challenger_id": event.challenger_id,
            "challenger_reward": event.challenger_reward,
            "burn_amount": event.burn_amount,
            "treasury_amount": event.treasury_amount,
            "timestamp_ms": event.timestamp_ms,
        }
        
        return json.dumps(tx_data).encode()
    
    # -------------------------------------------------------------------------
    # Reward Distribution
    # -------------------------------------------------------------------------
    
    def distribute_challenger_reward(
        self,
        event: SlashEvent,
    ) -> Tuple[bool, str]:
        """
        Distribute reward to challenger.
        
        Returns:
            (success, tx_hash_or_error)
        """
        if not self._submit_tx:
            return False, "no transaction callback configured"
        
        tx_data = {
            "type": "IAN_REWARD",
            "recipient_id": event.challenger_id,
            "amount": event.challenger_reward,
            "reason": f"fraud proof for {event.commit_hash.hex()[:16]}...",
            "timestamp_ms": int(time.time() * 1000),
        }
        
        tx_bytes = json.dumps(tx_data).encode()
        return self._submit_tx(tx_bytes)
    
    # -------------------------------------------------------------------------
    # Bond Withdrawal
    # -------------------------------------------------------------------------
    
    def request_withdrawal(
        self,
        committer_id: str,
        goal_id: str,
    ) -> Tuple[bool, str]:
        """
        Request bond withdrawal.
        
        Bond will be locked for bond_lock_period_seconds before
        it can be withdrawn.
        
        Returns:
            (success, reason)
        """
        bond = self.get_bond(committer_id, goal_id)
        
        if not bond:
            return False, "no bond found"
        
        if bond.status != BondStatus.ACTIVE:
            return False, f"bond not active: {bond.status.value}"
        
        # Lock bond
        lock_duration_ms = self._config.bond_lock_period_seconds * 1000
        bond.lock(lock_duration_ms)
        
        return True, f"bond locked until {bond.locked_until_ms}"
    
    def finalize_withdrawal(
        self,
        committer_id: str,
        goal_id: str,
    ) -> Tuple[bool, str]:
        """
        Finalize bond withdrawal after lock period.
        
        Returns:
            (success, reason)
        """
        bond = self.get_bond(committer_id, goal_id)
        
        if not bond:
            return False, "no bond found"
        
        if bond.status != BondStatus.LOCKED:
            return False, f"bond not locked: {bond.status.value}"
        
        now_ms = int(time.time() * 1000)
        if bond.locked_until_ms and now_ms < bond.locked_until_ms:
            remaining = (bond.locked_until_ms - now_ms) // 1000
            return False, f"lock period not elapsed: {remaining}s remaining"
        
        # Finalize
        bond.status = BondStatus.WITHDRAWN
        withdrawal_amount = bond.effective_bond()
        
        logger.info(
            f"Bond withdrawn: {committer_id[:16]}... for {goal_id}, "
            f"amount={withdrawal_amount}"
        )
        
        return True, f"withdrawn {withdrawal_amount}"
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_active_committers(self, goal_id: str) -> List[str]:
        """Get list of active committers for a goal."""
        return [
            bond.committer_id
            for bond in self._committer_bonds.values()
            if bond.goal_id == goal_id and bond.can_commit()
        ]
    
    def get_slash_history(
        self,
        committer_id: Optional[str] = None,
        goal_id: Optional[str] = None,
    ) -> List[SlashEvent]:
        """Get slash history with optional filters."""
        events = self._slash_events
        
        if committer_id:
            events = [e for e in events if e.committer_id == committer_id]
        
        if goal_id:
            events = [e for e in events if e.goal_id == goal_id]
        
        return events
    
    def get_total_bonded(self, goal_id: str) -> int:
        """Get total bonded amount for a goal."""
        return sum(
            bond.effective_bond()
            for bond in self._committer_bonds.values()
            if bond.goal_id == goal_id and bond.status in (BondStatus.ACTIVE, BondStatus.LOCKED)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize economic state."""
        return {
            "committer_bonds": {
                k: v.to_dict() for k, v in self._committer_bonds.items()
            },
            "challenger_bonds": {
                k.hex(): v.to_dict() for k, v in self._challenger_bonds.items()
            },
            "slash_events": [e.to_dict() for e in self._slash_events],
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        node_id: str,
        config: Optional[EconomicConfig] = None,
    ) -> "EconomicManager":
        """Deserialize economic state."""
        manager = cls(node_id=node_id, config=config)
        
        for k, v in data.get("committer_bonds", {}).items():
            manager._committer_bonds[k] = CommitterBond.from_dict(v)
        
        for k, v in data.get("challenger_bonds", {}).items():
            manager._challenger_bonds[bytes.fromhex(k)] = ChallengeBond(
                challenger_id=v["challenger_id"],
                goal_id=v["goal_id"],
                challenged_commit_hash=bytes.fromhex(v["challenged_commit_hash"]),
                amount=v["amount"],
                status=BondStatus(v["status"]),
                created_at_ms=v["created_at_ms"],
            )
        
        return manager
