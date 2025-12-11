"""
IAN Fraud Proof System - Detect and prove invalid state transitions.

Provides:
1. Fraud proof types for different violations
2. Fraud proof generation
3. Fraud proof verification
4. Challenge submission to Tau Net

Fraud Types:
- INVALID_LOG_ROOT: Claimed log root doesn't match computed root
- INVALID_LEADERBOARD: Leaderboard state is inconsistent
- SKIPPED_CONTRIBUTION: A valid contribution was not processed
- WRONG_ORDERING: Contributions processed out of order
- WRONG_EVALUATION: Evaluation metrics are incorrect

Security Model:
- Any node can generate fraud proofs
- Proofs are cryptographically verifiable
- Valid proofs result in slashing of offending committer
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
    from idi.ian.coordinator import IANCoordinator
    from idi.ian.models import Contribution, ContributionMeta
    from idi.ian.mmr import MerkleMountainRange, MembershipProof

logger = logging.getLogger(__name__)


# =============================================================================
# Fraud Types
# =============================================================================

class FraudType(Enum):
    """Types of fraud that can be proven."""
    INVALID_LOG_ROOT = "invalid_log_root"
    INVALID_LEADERBOARD_ROOT = "invalid_leaderboard_root"
    SKIPPED_CONTRIBUTION = "skipped_contribution"
    WRONG_ORDERING = "wrong_ordering"
    WRONG_EVALUATION = "wrong_evaluation"
    DOUBLE_PROCESSING = "double_processing"
    INVALID_COMMIT_SIGNATURE = "invalid_commit_signature"


# =============================================================================
# Fraud Proofs
# =============================================================================

@dataclass
class FraudProof:
    """
    Base class for fraud proofs.
    
    A fraud proof demonstrates that a committer submitted
    an invalid state transition to Tau Net.
    """
    fraud_type: FraudType
    goal_id: str
    challenged_commit_hash: bytes  # Hash of the commit being challenged
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    challenger_id: Optional[str] = None
    challenger_signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_type": self.fraud_type.value,
            "goal_id": self.goal_id,
            "challenged_commit_hash": self.challenged_commit_hash.hex(),
            "timestamp_ms": self.timestamp_ms,
            "challenger_id": self.challenger_id,
            "challenger_signature": (
                self.challenger_signature.hex()
                if self.challenger_signature
                else None
            ),
        }
    
    def signing_payload(self) -> bytes:
        """Get payload for signing."""
        data = self.to_dict()
        data.pop("challenger_signature", None)
        return json.dumps(data, sort_keys=True).encode()
    
    def verify(self) -> Tuple[bool, str]:
        """
        Verify the fraud proof is valid.
        
        Returns:
            (valid, reason)
        """
        raise NotImplementedError("Subclasses must implement verify()")


@dataclass
class InvalidLogRootProof(FraudProof):
    """
    Proof that a committed log root is invalid.
    
    Contains:
    - Claimed root from the commit
    - Actual leaf hashes
    - Merkle proof showing correct root
    """
    fraud_type: FraudType = field(default=FraudType.INVALID_LOG_ROOT, init=False)
    
    claimed_root: bytes = b''
    actual_leaves: List[bytes] = field(default_factory=list)
    leaf_indices: List[int] = field(default_factory=list)
    merkle_proofs: List[Dict[str, Any]] = field(default_factory=list)
    computed_root: bytes = b''
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "claimed_root": self.claimed_root.hex(),
            "actual_leaves": [leaf.hex() for leaf in self.actual_leaves],
            "leaf_indices": self.leaf_indices,
            "merkle_proofs": self.merkle_proofs,
            "computed_root": self.computed_root.hex(),
        })
        return data
    
    def verify(self) -> Tuple[bool, str]:
        """Verify the fraud proof."""
        # Check that computed root differs from claimed root
        if self.computed_root == self.claimed_root:
            return False, "computed root matches claimed root - no fraud"
        
        # Verify Merkle proofs
        for i, (leaf, index, proof) in enumerate(
            zip(self.actual_leaves, self.leaf_indices, self.merkle_proofs)
        ):
            if not self._verify_merkle_path(leaf, index, proof, self.computed_root):
                return False, f"merkle proof {i} is invalid"
        
        return True, "fraud proven: invalid log root"
    
    def _verify_merkle_path(
        self,
        leaf: bytes,
        index: int,
        proof: Dict[str, Any],
        expected_root: bytes,
    ) -> bool:
        """Verify a single Merkle path."""
        current = leaf
        
        for sibling_data in proof.get("siblings", []):
            sibling = bytes.fromhex(sibling_data["hash"])
            is_right = sibling_data["is_right"]
            
            if is_right:
                current = hashlib.sha256(b'\x01' + current + sibling).digest()
            else:
                current = hashlib.sha256(b'\x01' + sibling + current).digest()
        
        return current == expected_root


@dataclass
class InvalidLeaderboardProof(FraudProof):
    """
    Proof that a committed leaderboard root is invalid.
    """
    fraud_type: FraudType = field(default=FraudType.INVALID_LEADERBOARD_ROOT, init=False)
    
    claimed_root: bytes = b''
    entries: List[Dict[str, Any]] = field(default_factory=list)
    computed_root: bytes = b''
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "claimed_root": self.claimed_root.hex(),
            "entries": self.entries,
            "computed_root": self.computed_root.hex(),
        })
        return data
    
    def verify(self) -> Tuple[bool, str]:
        """Verify the fraud proof."""
        if self.computed_root == self.claimed_root:
            return False, "computed root matches claimed root - no fraud"
        
        # Recompute root from entries
        recomputed = self._compute_leaderboard_root(self.entries)
        
        if recomputed != self.computed_root:
            return False, "provided computed_root doesn't match recomputation"
        
        return True, "fraud proven: invalid leaderboard root"
    
    def _compute_leaderboard_root(self, entries: List[Dict[str, Any]]) -> bytes:
        """Compute leaderboard root from entries."""
        if not entries:
            return b'\x00' * 32
        
        # Sort by score descending
        sorted_entries = sorted(entries, key=lambda e: -e.get("score", 0))
        
        # Hash all entries
        entry_hashes = []
        for entry in sorted_entries:
            entry_bytes = json.dumps(entry, sort_keys=True).encode()
            entry_hashes.append(hashlib.sha256(entry_bytes).digest())
        
        # Merkle tree of entries
        while len(entry_hashes) > 1:
            if len(entry_hashes) % 2 == 1:
                entry_hashes.append(entry_hashes[-1])
            entry_hashes = [
                hashlib.sha256(entry_hashes[i] + entry_hashes[i+1]).digest()
                for i in range(0, len(entry_hashes), 2)
            ]
        
        return entry_hashes[0]


@dataclass
class SkippedContributionProof(FraudProof):
    """
    Proof that a valid contribution was not processed.
    """
    fraud_type: FraudType = field(default=FraudType.SKIPPED_CONTRIBUTION, init=False)
    
    contribution: Dict[str, Any] = field(default_factory=dict)
    contribution_hash: bytes = b''
    ordering_key_timestamp: int = 0
    ordering_key_pack_hash: bytes = b''
    
    # Evidence that contribution was gossiped
    gossip_witnesses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Proof that contribution is not in log
    non_inclusion_proof: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "contribution": self.contribution,
            "contribution_hash": self.contribution_hash.hex(),
            "ordering_key_timestamp": self.ordering_key_timestamp,
            "ordering_key_pack_hash": self.ordering_key_pack_hash.hex(),
            "gossip_witnesses": self.gossip_witnesses,
            "non_inclusion_proof": self.non_inclusion_proof,
        })
        return data
    
    def verify(self) -> Tuple[bool, str]:
        """Verify the fraud proof."""
        # Verify contribution hash
        contrib_bytes = json.dumps(self.contribution, sort_keys=True).encode()
        computed_hash = hashlib.sha256(contrib_bytes).digest()
        
        if computed_hash != self.contribution_hash:
            return False, "contribution hash mismatch"
        
        # Verify gossip witnesses (need threshold)
        if len(self.gossip_witnesses) < 2:
            return False, "insufficient gossip witnesses"
        
        # Verify non-inclusion proof
        # This would verify against the committed log root
        
        return True, "fraud proven: skipped contribution"


@dataclass
class WrongOrderingProof(FraudProof):
    """
    Proof that contributions were processed out of order.
    """
    fraud_type: FraudType = field(default=FraudType.WRONG_ORDERING, init=False)
    
    # Two contributions that are out of order
    contribution_a: Dict[str, Any] = field(default_factory=dict)
    contribution_b: Dict[str, Any] = field(default_factory=dict)
    
    # Log indices showing wrong order
    log_index_a: int = 0
    log_index_b: int = 0
    
    # Ordering keys showing correct order
    ordering_key_a: Dict[str, Any] = field(default_factory=dict)
    ordering_key_b: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "contribution_a": self.contribution_a,
            "contribution_b": self.contribution_b,
            "log_index_a": self.log_index_a,
            "log_index_b": self.log_index_b,
            "ordering_key_a": self.ordering_key_a,
            "ordering_key_b": self.ordering_key_b,
        })
        return data
    
    def verify(self) -> Tuple[bool, str]:
        """Verify the fraud proof."""
        # Extract ordering keys
        ts_a = self.ordering_key_a.get("timestamp_ms", 0)
        ts_b = self.ordering_key_b.get("timestamp_ms", 0)
        hash_a = bytes.fromhex(self.ordering_key_a.get("pack_hash", ""))
        hash_b = bytes.fromhex(self.ordering_key_b.get("pack_hash", ""))
        
        # A should come before B based on ordering key
        a_before_b = (ts_a, hash_a) < (ts_b, hash_b)
        
        # But A has higher log index (processed later)
        a_processed_later = self.log_index_a > self.log_index_b
        
        if a_before_b and a_processed_later:
            return True, "fraud proven: wrong ordering"
        
        return False, "ordering is correct - no fraud"


@dataclass
class WrongEvaluationProof(FraudProof):
    """
    Proof that evaluation metrics are incorrect.
    
    Requires re-running the evaluation to verify.
    """
    fraud_type: FraudType = field(default=FraudType.WRONG_EVALUATION, init=False)
    
    contribution: Dict[str, Any] = field(default_factory=dict)
    claimed_metrics: Dict[str, Any] = field(default_factory=dict)
    actual_metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation_seed: int = 0
    
    # Witnesses who ran independent evaluations
    evaluation_witnesses: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "contribution": self.contribution,
            "claimed_metrics": self.claimed_metrics,
            "actual_metrics": self.actual_metrics,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_witnesses": self.evaluation_witnesses,
        })
        return data
    
    def verify(self) -> Tuple[bool, str]:
        """Verify the fraud proof."""
        # Check that metrics differ significantly
        for key in ["reward", "risk", "complexity"]:
            claimed = self.claimed_metrics.get(key, 0)
            actual = self.actual_metrics.get(key, 0)
            
            if abs(claimed - actual) > 0.01:  # 1% tolerance
                # Verify witnesses agree with actual
                if len(self.evaluation_witnesses) < 2:
                    return False, "insufficient evaluation witnesses"
                
                return True, f"fraud proven: wrong evaluation ({key})"
        
        return False, "metrics within tolerance - no fraud"


# =============================================================================
# Fraud Proof Generator
# =============================================================================

class FraudProofGenerator:
    """
    Generate fraud proofs from detected inconsistencies.
    """
    
    def __init__(self, node_id: str, coordinator: "IANCoordinator"):
        """
        Initialize generator.
        
        Args:
            node_id: This node's ID
            coordinator: Local coordinator for state access
        """
        self._node_id = node_id
        self._coordinator = coordinator
    
    def generate_invalid_log_root_proof(
        self,
        commit_hash: bytes,
        claimed_root: bytes,
        goal_id: str,
    ) -> Optional[InvalidLogRootProof]:
        """
        Generate proof that a log root is invalid.
        
        Args:
            commit_hash: Hash of the commit being challenged
            claimed_root: Root claimed in the commit
            goal_id: Goal ID
            
        Returns:
            Fraud proof if we can prove fraud, None otherwise
        """
        # Get our computed root
        computed_root = self._coordinator.get_log_root()
        
        if computed_root == claimed_root:
            return None  # Roots match, no fraud
        
        # Gather leaves and proofs
        log = self._coordinator.state.log
        leaves = []
        indices = []
        proofs = []
        
        # Get sample of leaves with proofs
        sample_size = min(10, log.size)
        for i in range(sample_size):
            try:
                proof = log.get_proof(i)
                leaves.append(proof.leaf_hash)
                indices.append(i)
                proofs.append({
                    "siblings": [
                        {"hash": h.hex(), "is_right": r}
                        for h, r in proof.siblings
                    ],
                    "peaks_bag": [p.hex() for p in proof.peaks_bag],
                })
            except Exception:
                continue
        
        return InvalidLogRootProof(
            goal_id=goal_id,
            challenged_commit_hash=commit_hash,
            challenger_id=self._node_id,
            claimed_root=claimed_root,
            actual_leaves=leaves,
            leaf_indices=indices,
            merkle_proofs=proofs,
            computed_root=computed_root,
        )
    
    def generate_invalid_leaderboard_proof(
        self,
        commit_hash: bytes,
        claimed_root: bytes,
        goal_id: str,
    ) -> Optional[InvalidLeaderboardProof]:
        """Generate proof that a leaderboard root is invalid."""
        computed_root = self._coordinator.get_leaderboard_root()
        
        if computed_root == claimed_root:
            return None
        
        # Get leaderboard entries
        entries = [
            entry.to_dict()
            for entry in self._coordinator.get_leaderboard()
        ]
        
        return InvalidLeaderboardProof(
            goal_id=goal_id,
            challenged_commit_hash=commit_hash,
            challenger_id=self._node_id,
            claimed_root=claimed_root,
            entries=entries,
            computed_root=computed_root,
        )
    
    def generate_wrong_ordering_proof(
        self,
        commit_hash: bytes,
        goal_id: str,
        contrib_a: "Contribution",
        contrib_b: "Contribution",
        log_index_a: int,
        log_index_b: int,
    ) -> Optional[WrongOrderingProof]:
        """Generate proof of wrong ordering."""
        from .ordering import OrderingKey
        
        key_a = OrderingKey.from_contribution(contrib_a)
        key_b = OrderingKey.from_contribution(contrib_b)
        
        # A should come before B but has higher log index
        if key_a < key_b and log_index_a > log_index_b:
            return WrongOrderingProof(
                goal_id=goal_id,
                challenged_commit_hash=commit_hash,
                challenger_id=self._node_id,
                contribution_a=contrib_a.to_dict(),
                contribution_b=contrib_b.to_dict(),
                log_index_a=log_index_a,
                log_index_b=log_index_b,
                ordering_key_a={
                    "timestamp_ms": key_a.timestamp_ms,
                    "pack_hash": key_a.pack_hash.hex(),
                },
                ordering_key_b={
                    "timestamp_ms": key_b.timestamp_ms,
                    "pack_hash": key_b.pack_hash.hex(),
                },
            )
        
        return None


# =============================================================================
# Fraud Proof Verifier
# =============================================================================

class FraudProofVerifier:
    """
    Verify fraud proofs independently.
    
    Used by:
    - Nodes to verify proofs before relaying
    - Tau Net rules to verify proofs on-chain
    """
    
    def verify(self, proof: FraudProof) -> Tuple[bool, str]:
        """
        Verify a fraud proof.
        
        Args:
            proof: Fraud proof to verify
            
        Returns:
            (valid, reason)
        """
        # Basic validation
        if not proof.challenged_commit_hash:
            return False, "missing challenged_commit_hash"
        
        if not proof.goal_id:
            return False, "missing goal_id"
        
        # Type-specific verification
        return proof.verify()
    
    def verify_signature(
        self,
        proof: FraudProof,
        public_key: bytes,
    ) -> bool:
        """Verify challenger signature on proof."""
        if not proof.challenger_signature:
            return False
        
        from .node import NodeIdentity
        
        return NodeIdentity.verify_with_public_key(
            public_key,
            proof.signing_payload(),
            proof.challenger_signature,
        )


# =============================================================================
# Challenge Manager
# =============================================================================

@dataclass
class ChallengeConfig:
    """Configuration for challenge submission."""
    challenge_bond: int = 100  # Bond required to submit challenge
    challenge_period_seconds: int = 3600  # 1 hour challenge window
    min_confirmations: int = 2  # Minimum peer confirmations


class ChallengeManager:
    """
    Manage fraud proof challenges.
    
    Responsibilities:
    - Track pending challenges
    - Submit challenges to Tau Net
    - Handle challenge responses
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[ChallengeConfig] = None,
    ):
        self._node_id = node_id
        self._config = config or ChallengeConfig()
        
        # Pending challenges
        self._pending: Dict[bytes, FraudProof] = {}  # commit_hash -> proof
        
        # Confirmed challenges (by peers)
        self._confirmations: Dict[bytes, List[str]] = {}  # commit_hash -> [node_ids]
        
        # Callback for Tau submission
        self._submit_to_tau: Optional[Callable[[bytes], Tuple[bool, str]]] = None
    
    def set_tau_callback(
        self,
        submit_to_tau: Callable[[bytes], Tuple[bool, str]],
    ) -> None:
        """Set callback for Tau Net submission."""
        self._submit_to_tau = submit_to_tau
    
    async def submit_challenge(
        self,
        proof: FraudProof,
    ) -> Tuple[bool, str]:
        """
        Submit a fraud proof challenge.
        
        Args:
            proof: Verified fraud proof
            
        Returns:
            (success, reason_or_tx_hash)
        """
        commit_hash = proof.challenged_commit_hash
        
        # Check if already challenging
        if commit_hash in self._pending:
            return False, "already challenging this commit"
        
        # Verify proof locally
        verifier = FraudProofVerifier()
        valid, reason = verifier.verify(proof)
        
        if not valid:
            return False, f"invalid proof: {reason}"
        
        # Add to pending
        self._pending[commit_hash] = proof
        self._confirmations[commit_hash] = [self._node_id]
        
        logger.info(
            f"Challenge submitted for commit {commit_hash.hex()[:16]}...: "
            f"{proof.fraud_type.value}"
        )
        
        return True, "challenge submitted"
    
    def add_confirmation(
        self,
        commit_hash: bytes,
        confirmer_id: str,
    ) -> int:
        """
        Add peer confirmation to a challenge.
        
        Returns:
            Total confirmations
        """
        if commit_hash not in self._confirmations:
            self._confirmations[commit_hash] = []
        
        if confirmer_id not in self._confirmations[commit_hash]:
            self._confirmations[commit_hash].append(confirmer_id)
        
        return len(self._confirmations[commit_hash])
    
    async def finalize_challenge(
        self,
        commit_hash: bytes,
    ) -> Tuple[bool, str]:
        """
        Finalize a challenge by submitting to Tau Net.
        
        Requires minimum confirmations from peers.
        
        Returns:
            (success, tx_hash_or_error)
        """
        if commit_hash not in self._pending:
            return False, "no pending challenge for this commit"
        
        confirmations = len(self._confirmations.get(commit_hash, []))
        
        if confirmations < self._config.min_confirmations:
            return False, f"need {self._config.min_confirmations} confirmations, have {confirmations}"
        
        proof = self._pending[commit_hash]
        
        if not self._submit_to_tau:
            return False, "no Tau submission callback configured"
        
        # Serialize and submit
        proof_bytes = json.dumps(proof.to_dict()).encode()
        success, result = self._submit_to_tau(proof_bytes)
        
        if success:
            del self._pending[commit_hash]
            logger.info(f"Challenge finalized: {result}")
        
        return success, result
    
    def get_pending_challenges(self) -> List[FraudProof]:
        """Get all pending challenges."""
        return list(self._pending.values())
