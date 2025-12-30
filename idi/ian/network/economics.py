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

from .kernels.bond_status_fsm_ref import (
    State as BondKernelState,
    Command as BondKernelCommand,
    StepResult as BondKernelStepResult,
    check_invariants as bond_check_invariants,
    step as bond_step,
)
from .kernels.challenge_bond_fsm_ref import (
    State as ChallengeKernelState,
    Command as ChallengeKernelCommand,
    StepResult as ChallengeKernelStepResult,
    check_invariants as challenge_check_invariants,
    step as challenge_step,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EconomicConfig:
    """
    Economic parameters for IAN on Tau Net.

    IMPORTANT (verification envelope):
    The production implementation is wired to ESSO-verified kernels whose state
    variables are *bounded* (see `internal/esso/requirements/*.req.yaml`).
    These defaults are chosen to stay within the kernel domains so the formal
    claims apply without silent truncation.
    """
    
    # Committer bonds
    min_committer_bond: int = 1_000  # Kernel domain: min_bond <= 100_000
    max_committer_bond: int = 1_000_000  # Kernel domain: amount <= 1_000_000
    
    # Challenger bonds
    challenge_bond: int = 1_000  # Kernel domain: min_challenge_bond <= 100_000
    
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
    
    Wired to ESSO-verified kernel: bond_status_fsm_ref.
    The kernel is the single source of truth for status and total_slashed.
    """
    committer_id: str  # Node ID / public key
    goal_id: str
    amount: int
    _status: BondStatus = BondStatus.ACTIVE  # Internal, synced from kernel
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    locked_until_ms: Optional[int] = None
    
    # Tau Net reference
    bond_tx_hash: Optional[bytes] = None
    
    # Config (set by EconomicManager)
    min_bond: int = 0  # For active_has_valid_bond invariant
    
    # History (total_slashed is now derived from kernel)
    slash_count: int = 0
    commits_count: int = 0
    
    # Kernel state (source of truth)
    _kstate: BondKernelState = field(default=None, repr=False)  # type: ignore
    
    def __post_init__(self) -> None:
        """Initialize kernel state."""
        if not (0 <= int(self.amount) <= 1_000_000):
            raise ValueError(f"Bond amount out of kernel domain [0, 1_000_000]: {self.amount}")
        if not (0 <= int(self.min_bond) <= 100_000):
            raise ValueError(f"min_bond out of kernel domain [0, 100_000]: {self.min_bond}")

        # Map shell status to kernel status
        status_map = {
            BondStatus.ACTIVE: 'ACTIVE',
            BondStatus.LOCKED: 'LOCKED',
            BondStatus.SLASHED: 'SLASHED',
            BondStatus.WITHDRAWN: 'WITHDRAWN',
        }
        self._kstate = BondKernelState(
            amount=int(self.amount),
            min_bond=int(self.min_bond),
            status=status_map.get(self._status, 'ACTIVE'),
            total_slashed=0,
        )
        # Keep shell and kernel amounts consistent.
        self.amount = self._kstate.amount
    
    @property
    def status(self) -> BondStatus:
        """Read-through property: derive status from kernel."""
        kernel_status_map = {
            'ACTIVE': BondStatus.ACTIVE,
            'LOCKED': BondStatus.LOCKED,
            'SLASHED': BondStatus.SLASHED,
            'WITHDRAWN': BondStatus.WITHDRAWN,
        }
        return kernel_status_map.get(self._kstate.status, BondStatus.ACTIVE)
    
    @property
    def total_slashed(self) -> int:
        """Read-through property: derive total_slashed from kernel."""
        return self._kstate.total_slashed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "committer_id": self.committer_id,
            "goal_id": self.goal_id,
            "amount": self.amount,
            "min_bond": self.min_bond,
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
        bond = cls(
            committer_id=data["committer_id"],
            goal_id=data["goal_id"],
            amount=data["amount"],
            _status=BondStatus(data["status"]),
            created_at_ms=data["created_at_ms"],
            locked_until_ms=data.get("locked_until_ms"),
            bond_tx_hash=bytes.fromhex(data["bond_tx_hash"]) if data.get("bond_tx_hash") else None,
            slash_count=data.get("slash_count", 0),
            commits_count=data.get("commits_count", 0),
            min_bond=int(data.get("min_bond", 0)),
        )
        # Reconstruct kernel state from persisted fields (source of truth).
        status_map = {
            BondStatus.ACTIVE: 'ACTIVE',
            BondStatus.LOCKED: 'LOCKED',
            BondStatus.SLASHED: 'SLASHED',
            BondStatus.WITHDRAWN: 'WITHDRAWN',
        }
        total_slashed = int(data.get("total_slashed", 0))
        bond._kstate = BondKernelState(
            amount=int(bond.amount),
            min_bond=int(bond.min_bond),
            status=status_map.get(bond._status, 'ACTIVE'),
            total_slashed=total_slashed,
        )
        bond.amount = bond._kstate.amount
        bond._check_invariants()
        return bond
    
    def can_commit(self) -> bool:
        """Check if this bond allows committing."""
        return self.status == BondStatus.ACTIVE
    
    def effective_bond(self) -> int:
        """Get effective bond amount after slashing."""
        return self.amount - self.total_slashed
    
    def _apply_kernel(self, tag: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Apply a kernel command. Returns True if accepted, False if rejected.
        
        Fail-closed: if kernel rejects, shell state is unchanged.
        """
        cmd = BondKernelCommand(tag=tag, args=args or {})
        result = bond_step(self._kstate, cmd)
        if result.ok and result.state is not None:
            self._kstate = result.state
            logger.debug(f"Bond kernel accepted {tag}: {result.effects}")
            return True
        else:
            logger.warning(f"Bond kernel REJECTED {tag}: {result.error}")
            return False

    def set_min_bond(self, min_bond: int) -> None:
        """
        Update the configured `min_bond` (kernel data var).

        This is a configuration/environment update, not an action transition,
        so we reconstruct kernel state and re-check invariants.
        """
        if not (0 <= int(min_bond) <= 100_000):
            raise ValueError(f"min_bond out of kernel domain [0, 100_000]: {min_bond}")
        self.min_bond = int(min_bond)
        self._kstate = BondKernelState(
            amount=self._kstate.amount,
            min_bond=self.min_bond,
            status=self._kstate.status,
            total_slashed=self._kstate.total_slashed,
        )
        self._check_invariants()
    
    def lock(self, duration_ms: int) -> bool:
        """Lock bond during challenge period. Returns True if accepted."""
        if self._apply_kernel('lock_bond'):
            self.locked_until_ms = int(time.time() * 1000) + duration_ms
            return True
        return False
    
    def unlock(self) -> bool:
        """Unlock bond after challenge period. Returns True if accepted."""
        if self._apply_kernel('unlock_bond'):
            self.locked_until_ms = None
            return True
        return False
    
    def slash(self, slash_amount: int) -> bool:
        """
        Slash the bond by the given amount. Returns True if accepted.
        
        The kernel determines whether this causes a state transition to SLASHED
        based on whether effective_bond drops below min_bond.
        """
        if slash_amount <= 0:
            return False
        # Kernel domain for slash_amount is [0, 1_000_000].
        kernel_slash = min(int(slash_amount), 1_000_000)
        
        # Try slash_partial first (stays in current state)
        if self._apply_kernel('slash_partial', {'slash_amount': kernel_slash}):
            self.slash_count += 1
            return True
        
        # If partial failed, try slash_below_min (transitions to SLASHED)
        if self._apply_kernel('slash_below_min', {'slash_amount': kernel_slash}):
            self.slash_count += 1
            return True
        
        return False
    
    def withdraw(self) -> bool:
        """Withdraw the bond (terminal state). Returns True if accepted."""
        if self.status == BondStatus.LOCKED:
            return self._apply_kernel('withdraw_locked')
        elif self.status == BondStatus.SLASHED:
            return self._apply_kernel('withdraw_slashed')
        return False
    
    def topup(self, topup_amount: int) -> bool:
        """
        Add funds to the bond. Returns True if accepted.
        
        Available via Foundry-generated kernel.
        """
        if topup_amount <= 0:
            return False
        if topup_amount > 1_000_000:
            return False
        if self._apply_kernel('topup', {'topup_amount': int(topup_amount)}):
            # Update shell amount to match kernel
            self.amount = self._kstate.amount
            return True
        return False
    
    def _check_invariants(self) -> None:
        """
        Check ESSO-verified invariants via kernel.
        
        Invariants from bond_status_fsm.json:
        - slashed_not_exceeds_amount: total_slashed <= amount
        - active_has_valid_bond: ACTIVE => effective_bond >= min_bond
        - withdrawn_is_terminal: WITHDRAWN => effective_bond == 0
        """
        ok, failed = bond_check_invariants(self._kstate)
        if not ok:
            raise RuntimeError(
                f"ESSO kernel invariant violation: {failed} - "
                f"state={self._kstate}"
            )


@dataclass
class ChallengeBond:
    """
    Bond posted by a challenger.
    
    Required to submit fraud proof challenges.
    Returned if challenge is valid, slashed if invalid.
    
    Wired to ESSO-verified kernel: challenge_bond_fsm_ref.
    """
    challenger_id: str
    goal_id: str
    challenged_commit_hash: bytes
    amount: int
    _status: BondStatus = BondStatus.ACTIVE  # Internal, derived from kernel
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Config
    min_challenge_bond: int = 0  # Must be <= 100_000 (kernel domain)
    
    # Resolution
    challenge_valid: Optional[bool] = None
    resolution_tx_hash: Optional[bytes] = None
    
    # Kernel state
    _kstate: ChallengeKernelState = field(default=None, repr=False)  # type: ignore

    def __post_init__(self) -> None:
        """Initialize kernel state."""
        if not (0 <= int(self.amount) <= 1_000_000):
            raise ValueError(f"ChallengeBond amount out of kernel domain [0, 1_000_000]: {self.amount}")
        if not (0 <= int(self.min_challenge_bond) <= 100_000):
            raise ValueError(
                f"min_challenge_bond out of kernel domain [0, 100_000]: {self.min_challenge_bond}"
            )
        status_map = {
            BondStatus.ACTIVE: 'ACTIVE',
            BondStatus.LOCKED: 'LOCKED',
            BondStatus.SLASHED: 'SLASHED',
            BondStatus.WITHDRAWN: 'WITHDRAWN',
        }
        self._kstate = ChallengeKernelState(
            amount=int(self.amount),
            min_challenge_bond=int(self.min_challenge_bond),
            status=status_map.get(self._status, 'ACTIVE'),
        )
        self.amount = self._kstate.amount

    @property
    def status(self) -> BondStatus:
        """Read-through property: derive status from kernel."""
        kernel_status_map = {
            'ACTIVE': BondStatus.ACTIVE,
            'LOCKED': BondStatus.LOCKED,
            'SLASHED': BondStatus.SLASHED,
            'WITHDRAWN': BondStatus.WITHDRAWN,
        }
        return kernel_status_map.get(self._kstate.status, BondStatus.ACTIVE)

    def _apply_kernel(self, tag: str, args: Optional[Dict[str, Any]] = None) -> bool:
        cmd = ChallengeKernelCommand(tag=tag, args=args or {})
        result = challenge_step(self._kstate, cmd)
        if result.ok and result.state is not None:
            self._kstate = result.state
            # Sync amount from kernel
            self.amount = self._kstate.amount
            logger.debug(f"ChallengeBond kernel accepted {tag}")
            return True
        logger.warning(f"ChallengeBond kernel REJECTED {tag}: {result.error}")
        return False

    def lock(self) -> bool:
        """Lock bond for challenge."""
        return self._apply_kernel('lock_for_challenge')

    def resolve(self, valid: bool) -> bool:
        """Resolve challenge. Valid -> Unlock, Invalid -> Slash."""
        self.challenge_valid = valid
        if valid:
            return self._apply_kernel('resolve_valid')
        else:
            return self._apply_kernel('resolve_invalid', {'slash_amount': self.amount})

    def withdraw(self) -> bool:
        """Withdraw bond if possible."""
        if self.status == BondStatus.ACTIVE:
            return self._apply_kernel('withdraw_active')
        elif self.status == BondStatus.SLASHED:
            return self._apply_kernel('withdraw_remainder')
        return False

    def _check_invariants(self) -> None:
        """Check ESSO kernel invariants."""
        ok, failed = challenge_check_invariants(self._kstate)
        if not ok:
            raise RuntimeError(f"ESSO invariant failed: {failed}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenger_id": self.challenger_id,
            "goal_id": self.goal_id,
            "challenged_commit_hash": self.challenged_commit_hash.hex(),
            "amount": self.amount,
            "min_challenge_bond": self.min_challenge_bond,
            "status": self.status.value,
            "created_at_ms": self.created_at_ms,
            "challenge_valid": self.challenge_valid,
            "resolution_tx_hash": self.resolution_tx_hash.hex() if self.resolution_tx_hash else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChallengeBond":
        return cls(
            challenger_id=data["challenger_id"],
            goal_id=data["goal_id"],
            challenged_commit_hash=bytes.fromhex(data["challenged_commit_hash"]),
            amount=data["amount"],
            min_challenge_bond=int(data.get("min_challenge_bond", 0)),
            _status=BondStatus(data["status"]),
            created_at_ms=data["created_at_ms"],
            challenge_valid=data.get("challenge_valid"),
            resolution_tx_hash=bytes.fromhex(data["resolution_tx_hash"]) if data.get("resolution_tx_hash") else None,
        )


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

        # Enforce the ESSO kernel domains up-front (fail-fast / fail-closed).
        if not (0 <= int(self._config.min_committer_bond) <= 100_000):
            raise ValueError(
                f"min_committer_bond must be in [0, 100_000] for the verified kernel, "
                f"got {self._config.min_committer_bond}"
            )
        if not (0 <= int(self._config.max_committer_bond) <= 1_000_000):
            raise ValueError(
                f"max_committer_bond must be in [0, 1_000_000] for the verified kernel, "
                f"got {self._config.max_committer_bond}"
            )
        if self._config.min_committer_bond > self._config.max_committer_bond:
            raise ValueError(
                f"min_committer_bond ({self._config.min_committer_bond}) must be <= "
                f"max_committer_bond ({self._config.max_committer_bond})"
            )
        if not (0 <= int(self._config.challenge_bond) <= 100_000):
            raise ValueError(
                f"challenge_bond must be in [0, 100_000] for the verified kernel, "
                f"got {self._config.challenge_bond}"
            )
        
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
            bond = self._committer_bonds[key]
            if not bond.topup(amount):
                return False, "bond topup rejected by kernel (would exceed bounds)"
        else:
            # New bond
            self._committer_bonds[key] = CommitterBond(
                committer_id=committer_id,
                goal_id=goal_id,
                amount=amount,
                bond_tx_hash=bond_tx_hash,
                min_bond=self._config.min_committer_bond,  # Propagate config
            )
        
        # After creation/update, always check invariants
        self._committer_bonds[key]._check_invariants()
        
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
            min_challenge_bond=self._config.challenge_bond,
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
        # Update bond via kernel
        if not bond.slash(slash_amount):
            logger.error(f"Slashing rejected by kernel for {committer_id}")
            return None
        
        bond._check_invariants()  # CBC: verify after slash
        
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
        
        # Finalize - kernel requires total_slashed = amount for WITHDRAWN
        withdrawal_amount = bond.effective_bond()
        # Finalize - kernel requires total_slashed = amount for WITHDRAWN
        withdrawal_amount = bond.effective_bond()
        if not bond.withdraw():
            return False, "withdrawal rejected by kernel"
        bond._check_invariants()  # CBC: verify withdrawn_is_terminal
        
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
            bond = CommitterBond.from_dict(v)
            if manager._config:
                bond.set_min_bond(manager._config.min_committer_bond)
            manager._committer_bonds[k] = bond
        
        for k, v in data.get("challenger_bonds", {}).items():
            manager._challenger_bonds[bytes.fromhex(k)] = ChallengeBond.from_dict(v)
        
        return manager
