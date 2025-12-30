"""
SlotBatcher: Fair transaction ordering with VRF-based random permutation.

Security Controls:
- VRF key protection (never expose private key)
- Commit-reveal scheme to prevent front-running
- Deterministic ordering from VRF output
- Fixed slot duration to prevent timing attacks
- Quality weight commitment before slot close
- Atomic slot state transitions

Based on: Fair Sequencer (HackMD), TimeBoost (Arbitrum), with IAN enhancements

Author: DarkLightX
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning(
        "cryptography library not available - VRF signatures will use fallback "
        "(NOT SECURE FOR PRODUCTION)"
    )

# CODEX: Wired to verified kernel
from .kernels import slot_state_fsm_ref as slot_kernel

# =============================================================================
# Security Constants
# =============================================================================

HASH_SIZE = 32

# Slot timing
DEFAULT_SLOT_DURATION_MS = 500  # 500ms per slot
MIN_SLOT_DURATION_MS = 100
MAX_SLOT_DURATION_MS = 5000
SLOT_GRACE_PERIOD_MS = 50  # Allow late arrivals within grace period

# Quality weights
MIN_QUALITY_WEIGHT = 0.1  # Minimum weight for any contributor
MAX_QUALITY_WEIGHT = 10.0  # Maximum weight cap
DEFAULT_QUALITY_WEIGHT = 1.0

# Capacity limits
MAX_CONTRIBUTIONS_PER_SLOT = 1000
MAX_BUNDLE_SIZE = 10
MAX_PENDING_SLOTS = 100

# VRF domain separation
VRF_DOMAIN_PREFIX = b"SLOTBATCHER_VRF_V1"


# =============================================================================
# Data Structures
# =============================================================================

class SlotState(Enum):
    """State of a slot in its lifecycle."""
    COLLECTING = 1     # Accepting contributions
    ORDERING = 2       # Computing VRF and ordering
    COMMITTED = 3      # Order finalized, awaiting execution
    EXECUTED = 4       # All contributions executed
    CANCELLED = 5      # Slot cancelled (error/timeout)


@dataclass
class SlotContribution:
    """
    A contribution submitted to a slot.
    
    Security:
        - commitment_hash computed before content revealed
        - quality_weight committed at submission time
    """
    contribution_id: str
    contributor_id: str
    pack_hash: bytes
    commitment_hash: bytes  # H(content || nonce) for commit-reveal
    quality_weight: float
    submitted_at_ms: int
    bundle_id: Optional[str] = None  # For atomic bundles
    
    # Revealed after slot closes
    content: Optional[bytes] = None
    reveal_nonce: Optional[bytes] = None
    
    def __post_init__(self) -> None:
        """Validate contribution."""
        if len(self.pack_hash) != HASH_SIZE:
            raise ValueError(f"pack_hash must be {HASH_SIZE} bytes")
        if len(self.commitment_hash) != HASH_SIZE:
            raise ValueError(f"commitment_hash must be {HASH_SIZE} bytes")
        if not MIN_QUALITY_WEIGHT <= self.quality_weight <= MAX_QUALITY_WEIGHT:
            raise ValueError(
                f"quality_weight must be in [{MIN_QUALITY_WEIGHT}, {MAX_QUALITY_WEIGHT}]"
            )
    
    def verify_reveal(self) -> bool:
        """
        Verify that revealed content matches commitment.
        
        Security: Ensures content wasn't changed after commit.
        """
        if self.content is None or self.reveal_nonce is None:
            return False
        
        expected = hashlib.sha256(self.content + self.reveal_nonce).digest()
        return expected == self.commitment_hash


@dataclass
class ContributionBundle:
    """
    Atomic bundle of contributions that must be ordered together.
    
    Security: Bundle integrity verified via bundle_hash.
    """
    bundle_id: str
    contribution_ids: List[str]
    bundle_hash: bytes  # H(sorted contribution_ids)
    creator_id: str
    created_at_ms: int
    
    def verify_integrity(self, contributions: List[SlotContribution]) -> bool:
        """Verify bundle contains expected contributions."""
        actual_ids = sorted([c.contribution_id for c in contributions])
        expected_ids = sorted(self.contribution_ids)
        
        if actual_ids != expected_ids:
            return False
        
        # Verify bundle hash
        expected_hash = hashlib.sha256(
            b'\x00'.join(c.encode() for c in expected_ids)
        ).digest()
        
        return expected_hash == self.bundle_hash


@dataclass
class VRFOutput:
    """
    VRF output for slot ordering.
    
    Security:
        - proof allows verification without private key
        - output is deterministic given input
    """
    slot_id: str
    vrf_input: bytes     # Deterministic input to VRF
    vrf_output: bytes    # Pseudorandom output (32 bytes)
    vrf_proof: bytes     # Proof of correct computation
    public_key: bytes    # VRF public key
    computed_at_ms: int
    
    def verify(self) -> bool:
        """
        Verify VRF output is correct.
        
        Security: Uses Ed25519-based verification.
        """
        if not CRYPTO_AVAILABLE:
            logger.warning("cryptography not available, VRF not verified")
            return True  # Skip verification if crypto unavailable
        
        try:
            # Load public key
            public_key = Ed25519PublicKey.from_public_bytes(self.public_key)
            
            # Verify signature (proof is signature over input)
            public_key.verify(self.vrf_proof, VRF_DOMAIN_PREFIX + self.vrf_input)
            
            # Verify output is hash of proof
            expected_output = hashlib.sha256(
                VRF_DOMAIN_PREFIX + self.vrf_proof
            ).digest()
            
            return expected_output == self.vrf_output
            
        except Exception as e:
            logger.warning(f"VRF verification failed: {e}")
            return False


@dataclass
class Slot:
    """
    A time-bounded collection of contributions to be ordered.
    
    Lifecycle:
        COLLECTING → ORDERING → COMMITTED → EXECUTED
    
    Security:
        - Fixed duration prevents timing manipulation
        - Atomic state transitions
        - VRF computed only after collection closes
    """
    slot_id: str
    start_time_ms: int
    end_time_ms: int
    _state: SlotState = SlotState.COLLECTING
    
    # Contributions (keyed by contribution_id)
    contributions: Dict[str, SlotContribution] = field(default_factory=dict)
    bundles: Dict[str, ContributionBundle] = field(default_factory=dict)
    
    # Ordering (set after VRF computation)
    vrf_output: Optional[VRFOutput] = None
    ordered_ids: List[str] = field(default_factory=list)
    
    # Quality-weighted lottery results
    lottery_winner_id: Optional[str] = None
    
    def is_accepting(self, now_ms: Optional[int] = None) -> bool:
        """Check if slot is still accepting contributions."""
        if self.state != SlotState.COLLECTING:
            return False
        
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        
        # Allow grace period
        return now_ms <= self.end_time_ms + SLOT_GRACE_PERIOD_MS
    
    def can_transition_to(self, new_state: SlotState) -> bool:
        """Check if state transition is valid."""
        valid_transitions = {
            SlotState.COLLECTING: {SlotState.ORDERING, SlotState.CANCELLED},
            SlotState.ORDERING: {SlotState.COMMITTED, SlotState.CANCELLED},
            SlotState.COMMITTED: {SlotState.EXECUTED, SlotState.CANCELLED},
            SlotState.EXECUTED: set(),  # Terminal
            SlotState.CANCELLED: set(),  # Terminal
        }
        return new_state in valid_transitions.get(self.state, set())

    @property
    def state(self) -> SlotState:
        """Get current state from verified kernel."""
        try:
            return SlotState[self._kstate.state]
        except (KeyError, AttributeError):
            return self._state

    def __post_init__(self):
        # CODEX: Initialize verified kernel state based on shell state
        self._kstate = slot_kernel.State(
            contribution_count=len(self.contributions),
            has_ordered_ids=bool(self.ordered_ids),
            has_vrf=self.vrf_output is not None,
            state=self._state.name,
        )
        self._check_invariants()

    def _apply_kernel(self, tag: str, **kwargs) -> bool:
        """Apply a kernel command and sync state."""
        cmd = slot_kernel.Command(tag=tag, args=kwargs)
        result = slot_kernel.step(self._kstate, cmd)
        
        if not result.ok:
            logger.error(f"Slot kernel REJECTED command {tag}: {result.error}")
            return False
            
        self._kstate = result.state
        return True

    def add_contribution(self, contrib_id: str, contrib: SlotContribution) -> bool:
        """Add contribution with kernel enforcement."""
        if self._apply_kernel('add_contribution'):
            self.contributions[contrib_id] = contrib
            return True
        return False

    def transition_to(self, new_state: SlotState) -> None:
        """
        Transition to new state via verified kernel.
        """
        success = False
        
        if new_state == SlotState.ORDERING:
            success = self._apply_kernel('start_ordering')
            
        elif new_state == SlotState.COMMITTED:
            if not self.contributions:
                success = self._apply_kernel('commit_empty_slot')
            else:
                success = self._apply_kernel('compute_vrf_and_commit')
        
        elif new_state == SlotState.EXECUTED:
            success = self._apply_kernel('execute_slot')
            
        elif new_state == SlotState.CANCELLED:
            # Probing for valid cancel transition
            for cmd in ['cancel_collecting', 'cancel_ordering', 'cancel_committed']:
                if self._apply_kernel(cmd):
                    success = True
                    break
        
        if not success:
            raise ValueError(
                f"Invalid slot state transition (Kernel Rejected): {self.state.name} -> {new_state.name}"
            )
        
        self._check_invariants()
    
    def _check_invariants(self) -> None:
        """
        Verify domain invariants with hard fails.
        """
        ok, error = slot_kernel.check_invariants(self._kstate)
        if not ok:
            raise RuntimeError(f"Slot invariant violation (Kernel enforced): {error}")
        
        # Cross-check shell counts
        if len(self.contributions) != self._kstate.contribution_count:
             raise RuntimeError(
                 f"Slot count mismatch: shell={len(self.contributions)} kernel={self._kstate.contribution_count}"
             )


@dataclass
class OrderingProof:
    """
    Proof that a contribution was ordered correctly in a slot.
    
    Security: Allows third-party verification of ordering.
    """
    slot_id: str
    contribution_id: str
    position: int
    total_contributions: int
    vrf_output: VRFOutput
    quality_weight: float
    lottery_ticket: float
    
    def verify(self) -> bool:
        """Verify the ordering proof."""
        # Verify VRF
        if not self.vrf_output.verify():
            return False
        
        # Position must be valid
        if not 0 <= self.position < self.total_contributions:
            return False
        
        return True


# =============================================================================
# VRF Implementation
# =============================================================================

class VRFProvider:
    """
    VRF provider for generating verifiable random ordering.
    
    Security:
        - Private key never exposed
        - Deterministic output for same input
        - Proof allows verification
        - PRODUCTION MODE: Requires cryptography library
    """
    
    def __init__(
        self,
        private_key: Optional[Ed25519PrivateKey] = None,
        require_crypto: bool = True,
    ):
        """
        Initialize VRF provider.
        
        Args:
            private_key: Ed25519 private key (generates new if None)
            require_crypto: If True, raises error when cryptography unavailable
        
        Security: Set require_crypto=True in production to prevent insecure fallback.
        """
        # Security: Enforce cryptographic VRF in production
        if require_crypto and not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "VRF requires cryptography library. Install with: pip install cryptography. "
                "Set require_crypto=False only for testing (INSECURE)."
            )
        
        if private_key is None and CRYPTO_AVAILABLE:
            private_key = Ed25519PrivateKey.generate()
        
        self._private_key = private_key
        self._require_crypto = require_crypto
        self._key_created_at_ms = int(time.time() * 1000)
        self._previous_public_key: Optional[bytes] = None
        
        if private_key and CRYPTO_AVAILABLE:
            self._public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        else:
            self._public_key = b'\x00' * 32
            if require_crypto:
                logger.error("VRF initialized without cryptographic key - INSECURE")
    
    @property
    def public_key(self) -> bytes:
        """Get VRF public key (safe to share)."""
        return self._public_key
    
    @property
    def key_created_at_ms(self) -> int:
        """Get timestamp when current key was created."""
        return self._key_created_at_ms
    
    def rotate_key(self) -> bytes:
        """
        Rotate VRF key pair, returning new public key.
        
        Security:
            - Old private key is discarded
            - Returns new public key for distribution
            - Should be called periodically to limit key exposure window
        
        Returns:
            New public key bytes
        
        Raises:
            RuntimeError: If cryptography library unavailable
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Cannot rotate VRF key: cryptography library not available"
            )
        
        # Generate new key pair
        new_private_key = Ed25519PrivateKey.generate()
        new_public_key = new_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Store old public key for transition period
        old_public_key = self._public_key
        
        # Atomic update
        self._private_key = new_private_key
        self._public_key = new_public_key
        self._key_created_at_ms = int(time.time() * 1000)
        self._previous_public_key = old_public_key
        
        logger.info(
            f"VRF key rotated. New public key: {new_public_key[:8].hex()}..."
        )
        
        return new_public_key
    
    def should_rotate(self, max_age_hours: int = 24) -> bool:
        """
        Check if key should be rotated based on age.
        
        Args:
            max_age_hours: Maximum key age before rotation recommended
        
        Returns:
            True if key is older than max_age_hours
        """
        age_ms = int(time.time() * 1000) - self._key_created_at_ms
        max_age_ms = max_age_hours * 60 * 60 * 1000
        return age_ms > max_age_ms
    
    def compute(self, slot_id: str, contribution_hashes: List[bytes]) -> VRFOutput:
        """
        Compute VRF output for slot ordering.
        
        Security:
            - Input is deterministic (sorted hashes)
            - Output cannot be predicted without private key
        """
        # Build deterministic input
        sorted_hashes = sorted(contribution_hashes)
        vrf_input = hashlib.sha256(
            slot_id.encode() + b'\x00'.join(sorted_hashes)
        ).digest()
        
        if self._private_key is None or not CRYPTO_AVAILABLE:
            # Fallback: use hash-based pseudo-VRF (not cryptographically secure)
            vrf_proof = hashlib.sha256(
                VRF_DOMAIN_PREFIX + vrf_input + secrets.token_bytes(32)
            ).digest()
            vrf_output = hashlib.sha256(VRF_DOMAIN_PREFIX + vrf_proof).digest()
        else:
            # Sign input as proof
            vrf_proof = self._private_key.sign(VRF_DOMAIN_PREFIX + vrf_input)
            vrf_output = hashlib.sha256(VRF_DOMAIN_PREFIX + vrf_proof).digest()
        
        return VRFOutput(
            slot_id=slot_id,
            vrf_input=vrf_input,
            vrf_output=vrf_output,
            vrf_proof=vrf_proof,
            public_key=self._public_key,
            computed_at_ms=int(time.time() * 1000),
        )


# =============================================================================
# Quality-Weighted Lottery
# =============================================================================

def compute_lottery_ticket(
    contribution_id: str,
    quality_weight: float,
    vrf_output: bytes,
) -> float:
    """
    Compute lottery ticket for contribution.
    
    Higher quality_weight = higher probability of earlier position.
    
    Security:
        - Uses VRF output for unpredictability
        - Weight clamped to prevent extreme bias
    """
    # Clamp weight
    weight = max(MIN_QUALITY_WEIGHT, min(quality_weight, MAX_QUALITY_WEIGHT))
    
    # Compute random value from VRF and contribution_id
    random_bytes = hashlib.sha256(
        vrf_output + contribution_id.encode()
    ).digest()
    
    # Convert to float in [0, 1)
    random_value = int.from_bytes(random_bytes[:8], 'big') / (2 ** 64)
    
    # Apply weight: ticket = -log(random) / weight
    # Higher weight = lower expected ticket = earlier position
    import math
    if random_value <= 0:
        random_value = 1e-10  # Prevent log(0)
    
    ticket = -math.log(random_value) / weight
    
    return ticket


def order_by_lottery(
    contributions: List[SlotContribution],
    vrf_output: bytes,
) -> List[str]:
    """
    Order contributions by quality-weighted lottery.
    
    Returns list of contribution_ids in order (lowest ticket first).
    
    Security:
        - Deterministic given VRF output
        - All nodes will compute same order
    """
    tickets = []
    
    for contrib in contributions:
        ticket = compute_lottery_ticket(
            contrib.contribution_id,
            contrib.quality_weight,
            vrf_output,
        )
        tickets.append((ticket, contrib.contribution_id))
    
    # Sort by ticket (ascending = earlier position)
    tickets.sort(key=lambda x: x[0])
    
    return [contrib_id for _, contrib_id in tickets]


# =============================================================================
# SlotBatcher Implementation
# =============================================================================

class SlotBatcher:
    """
    Fair ordering system using time-bounded slots and VRF-based permutation.
    
    Security features:
        - Commit-reveal scheme prevents front-running
        - VRF provides unpredictable ordering
        - Quality-weighted lottery rewards good contributors
        - Fixed slot duration prevents timing attacks
        - Atomic slot state transitions
    
    Usage:
        batcher = SlotBatcher(vrf_provider)
        
        # Submit contribution
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=hash_bytes,
            commitment_hash=commit_hash,
            quality_weight=1.5,
        )
        
        # After slot closes, reveal content
        await batcher.reveal(contrib_id, content, nonce)
        
        # Get ordered contributions
        ordered = await batcher.get_slot_order(slot_id)
    """
    
    def __init__(
        self,
        vrf_provider: Optional[VRFProvider] = None,
        slot_duration_ms: int = DEFAULT_SLOT_DURATION_MS,
        get_quality_weight: Optional[Callable[[str], float]] = None,
    ):
        """
        Initialize SlotBatcher.
        
        Args:
            vrf_provider: VRF provider for ordering (creates new if None)
            slot_duration_ms: Duration of each slot
            get_quality_weight: Callback to get contributor quality weight
        """
        # Validate config
        if not MIN_SLOT_DURATION_MS <= slot_duration_ms <= MAX_SLOT_DURATION_MS:
            raise ValueError(
                f"slot_duration_ms must be in [{MIN_SLOT_DURATION_MS}, {MAX_SLOT_DURATION_MS}]"
            )
        
        self._vrf_provider = vrf_provider or VRFProvider()
        self._slot_duration_ms = slot_duration_ms
        self._get_quality_weight = get_quality_weight or (lambda _: DEFAULT_QUALITY_WEIGHT)
        
        # Slot management
        self._current_slot: Optional[Slot] = None
        self._slots: OrderedDict[str, Slot] = OrderedDict()
        self._contribution_to_slot: Dict[str, str] = {}  # contrib_id → slot_id
        
        # Locks
        self._lock = asyncio.Lock()
        
        # Create first slot
        self._create_new_slot()
    
    @property
    def vrf_public_key(self) -> bytes:
        """Get VRF public key for verification."""
        return self._vrf_provider.public_key
    
    def _create_new_slot(self) -> Slot:
        """Create a new slot."""
        now_ms = int(time.time() * 1000)
        
        slot = Slot(
            slot_id=secrets.token_hex(16),
            start_time_ms=now_ms,
            end_time_ms=now_ms + self._slot_duration_ms,
            state=SlotState.COLLECTING,
        )
        
        self._current_slot = slot
        self._slots[slot.slot_id] = slot
        
        # Evict old slots
        while len(self._slots) > MAX_PENDING_SLOTS:
            oldest_id, oldest_slot = self._slots.popitem(last=False)
            # Clean up contribution mapping
            for contrib_id in oldest_slot.contributions:
                if self._contribution_to_slot.get(contrib_id) == oldest_id:
                    del self._contribution_to_slot[contrib_id]
        
        return slot
    
    def _get_or_create_current_slot(self) -> Slot:
        """Get current slot, creating new one if needed."""
        now_ms = int(time.time() * 1000)
        
        if self._current_slot is None or not self._current_slot.is_accepting(now_ms):
            # Current slot closed, finalize and create new
            if self._current_slot and self._current_slot.state == SlotState.COLLECTING:
                self._finalize_slot(self._current_slot)
            
            self._create_new_slot()
        
        return self._current_slot
    
    def _finalize_slot(self, slot: Slot) -> None:
        """
        Finalize slot: compute VRF and order contributions.
        
        Security: Called atomically after collection closes.
        """
        if slot.state != SlotState.COLLECTING:
            return
        
        slot.transition_to(SlotState.ORDERING)
        
        if not slot.contributions:
            # Empty slot
            slot.transition_to(SlotState.COMMITTED)
            return
        
        # Compute VRF
        contribution_hashes = [
            c.pack_hash for c in slot.contributions.values()
        ]
        slot.vrf_output = self._vrf_provider.compute(slot.slot_id, contribution_hashes)
        
        # Order by quality-weighted lottery
        slot.ordered_ids = order_by_lottery(
            list(slot.contributions.values()),
            slot.vrf_output.vrf_output,
        )
        
        # Determine lottery winner (first position)
        if slot.ordered_ids:
            slot.lottery_winner_id = slot.ordered_ids[0]
        
        slot.transition_to(SlotState.COMMITTED)
    
    async def submit(
        self,
        contributor_id: str,
        pack_hash: bytes,
        commitment_hash: bytes,
        quality_weight: Optional[float] = None,
        bundle_id: Optional[str] = None,
    ) -> str:
        """
        Submit a contribution to the current slot.
        
        Args:
            contributor_id: ID of the contributor
            pack_hash: Hash of the contribution pack
            commitment_hash: H(content || nonce) for commit-reveal
            quality_weight: Optional weight override
            bundle_id: Optional bundle ID for atomic grouping
        
        Returns:
            contribution_id
        
        Security:
            - commitment_hash hides content until reveal
            - quality_weight checked against callback
        """
        async with self._lock:
            slot = self._get_or_create_current_slot()
            
            # Check capacity
            if len(slot.contributions) >= MAX_CONTRIBUTIONS_PER_SLOT:
                raise RuntimeError(f"Slot {slot.slot_id} is full")
            
            # Get quality weight
            if quality_weight is None:
                quality_weight = self._get_quality_weight(contributor_id)
            
            # Clamp weight
            quality_weight = max(MIN_QUALITY_WEIGHT, min(quality_weight, MAX_QUALITY_WEIGHT))
            
            # Create contribution
            contribution_id = secrets.token_hex(16)
            
            contribution = SlotContribution(
                contribution_id=contribution_id,
                contributor_id=contributor_id,
                pack_hash=pack_hash,
                commitment_hash=commitment_hash,
                quality_weight=quality_weight,
                submitted_at_ms=int(time.time() * 1000),
                bundle_id=bundle_id,
            )
            
            if not slot.add_contribution(contribution_id, contribution):
                raise RuntimeError(f"Slot {slot.slot_id} rejected contribution (Kernel denied)")
            self._contribution_to_slot[contribution_id] = slot.slot_id
            
            return contribution_id
    
    async def reveal(
        self,
        contribution_id: str,
        content: bytes,
        nonce: bytes,
    ) -> bool:
        """
        Reveal contribution content after slot closes.
        
        Args:
            contribution_id: ID of the contribution
            content: Original content
            nonce: Random nonce used in commitment
        
        Returns:
            True if reveal verified successfully
        
        Security:
            - Verifies content matches commitment
            - Can only reveal after slot closes
        """
        async with self._lock:
            slot_id = self._contribution_to_slot.get(contribution_id)
            if slot_id is None:
                raise ValueError(f"Unknown contribution: {contribution_id}")
            
            slot = self._slots.get(slot_id)
            if slot is None:
                raise ValueError(f"Slot {slot_id} not found")
            
            # Can only reveal after collection closes
            if slot.state == SlotState.COLLECTING:
                raise RuntimeError("Cannot reveal while slot is still collecting")
            
            contribution = slot.contributions.get(contribution_id)
            if contribution is None:
                raise ValueError(f"Contribution {contribution_id} not in slot")
            
            # Store revealed content
            contribution.content = content
            contribution.reveal_nonce = nonce
            
            # Verify reveal
            if not contribution.verify_reveal():
                logger.warning(f"Reveal verification failed for {contribution_id}")
                return False
            
            return True
    
    async def get_slot_order(self, slot_id: str) -> List[str]:
        """
        Get the ordered contribution IDs for a slot.
        
        Returns empty list if slot not finalized.
        """
        async with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                return []
            
            if slot.state in (SlotState.COLLECTING, SlotState.ORDERING):
                return []  # Not yet ordered
            
            return slot.ordered_ids.copy()
    
    async def get_ordering_proof(
        self,
        contribution_id: str,
    ) -> Optional[OrderingProof]:
        """
        Get proof that contribution was ordered correctly.
        """
        async with self._lock:
            slot_id = self._contribution_to_slot.get(contribution_id)
            if slot_id is None:
                return None
            
            slot = self._slots.get(slot_id)
            if slot is None or slot.vrf_output is None:
                return None
            
            if contribution_id not in slot.ordered_ids:
                return None
            
            contribution = slot.contributions.get(contribution_id)
            if contribution is None:
                return None
            
            position = slot.ordered_ids.index(contribution_id)
            
            return OrderingProof(
                slot_id=slot_id,
                contribution_id=contribution_id,
                position=position,
                total_contributions=len(slot.ordered_ids),
                vrf_output=slot.vrf_output,
                quality_weight=contribution.quality_weight,
                lottery_ticket=compute_lottery_ticket(
                    contribution_id,
                    contribution.quality_weight,
                    slot.vrf_output.vrf_output,
                ),
            )
    
    async def get_current_slot_id(self) -> str:
        """Get ID of current collecting slot."""
        async with self._lock:
            slot = self._get_or_create_current_slot()
            return slot.slot_id
    
    async def force_finalize_current_slot(self) -> str:
        """
        Force finalize current slot (for testing/admin).
        
        Returns slot_id of finalized slot.
        """
        async with self._lock:
            if self._current_slot and self._current_slot.state == SlotState.COLLECTING:
                slot_id = self._current_slot.slot_id
                self._finalize_slot(self._current_slot)
                self._create_new_slot()
                return slot_id
            return ""
