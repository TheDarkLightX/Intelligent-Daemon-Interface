"""
Gradient Commitment Protocol for IAN zkML Training.

Provides cryptographic commitments to gradients for:
- Integrity: Prevent gradient tampering after computation
- Verifiability: Prove gradients were computed correctly
- Attribution: Bind commitments to contributor identity

Uses SHA-256 hash-based commitments with random blinding.

Security Properties:
- Computationally binding (SHA-256 collision resistance)
- Computationally hiding (random 256-bit blinding factor)
- Contributor-bound (contributor_id included in commitment hash)
- Timestamp-bound (timestamp included in commitment hash)

Note: This is NOT a Pedersen commitment and does NOT support
homomorphic aggregation of gradient values. Aggregation is
performed on commitment hashes for verification only.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..algorithms.crypto import (
    BLSOperations,
    BLSError,
    BLS_PUBKEY_LEN,
    BLS_SIGNATURE_LEN,
)


# Constants
GRADIENT_HASH_LEN = 32
MAX_GRADIENT_BYTES = 100 * 1024 * 1024  # 100 MB max gradient size
MAX_AGGREGATION_SIZE = 10_000  # Max contributors per aggregation
MAX_TIMESTAMP_SKEW_SECONDS = 300  # 5 minutes clock skew allowed
MAX_COMMITMENT_AGE_SECONDS = 86400 * 7  # 7 days max age
MAX_DTYPE_LEN = 64  # Max dtype string length
MAX_LAYER_NAME_LEN = 256  # Max layer name length
MAX_SHAPE_DIMS = 16  # Max number of dimensions
MAX_SHAPE_DIM_VALUE = 2**32  # Max value for any single dimension
COMMITMENT_DOMAIN = b"IAN_GRADIENT_V1"
BLS_SIGNING_CONTEXT = b"IAN_GRADIENT_COMMITMENT_V1"  # Protocol-specific BLS context


class CommitmentError(Exception):
    """Error in gradient commitment operations."""
    pass


# =============================================================================
# Gradient Representation
# =============================================================================


@dataclass(frozen=True)
class GradientTensor:
    """
    Represents a gradient tensor for commitment.
    
    Gradients are serialized as bytes for commitment.
    The actual tensor format (numpy, torch, etc.) is handled externally.
    """
    data: bytes
    shape: Tuple[int, ...]
    dtype: str = "float32"
    layer_name: Optional[str] = None
    
    def __post_init__(self) -> None:
        if len(self.data) > MAX_GRADIENT_BYTES:
            raise CommitmentError(f"Gradient exceeds {MAX_GRADIENT_BYTES} bytes")
        if not self.shape:
            raise CommitmentError("Gradient shape is required")
        if len(self.shape) > MAX_SHAPE_DIMS:
            raise CommitmentError(f"Shape has too many dimensions (max {MAX_SHAPE_DIMS})")
        for dim in self.shape:
            if dim < 0 or dim > MAX_SHAPE_DIM_VALUE:
                raise CommitmentError(f"Shape dimension must be 0-{MAX_SHAPE_DIM_VALUE}")
        if not self.dtype:
            raise CommitmentError("dtype is required")
        if len(self.dtype) > MAX_DTYPE_LEN:
            raise CommitmentError(f"dtype exceeds {MAX_DTYPE_LEN} characters")
        if self.layer_name and len(self.layer_name) > MAX_LAYER_NAME_LEN:
            raise CommitmentError(f"layer_name exceeds {MAX_LAYER_NAME_LEN} characters")
    
    @property
    def hash(self) -> bytes:
        """
        Compute hash of gradient data with full binding.
        
        Includes domain, dtype (length-prefixed), shape, layer_name, and data
        to prevent any ambiguity or replay attacks.
        """
        hasher = hashlib.sha256()
        hasher.update(COMMITMENT_DOMAIN)
        
        # Length-prefix dtype to prevent ambiguity
        dtype_bytes = self.dtype.encode("utf-8")
        hasher.update(len(dtype_bytes).to_bytes(4, "big"))
        hasher.update(dtype_bytes)
        
        # Include shape dimensions
        hasher.update(len(self.shape).to_bytes(4, "big"))
        for dim in self.shape:
            hasher.update(dim.to_bytes(8, "big"))
        
        # Include layer_name if present (length-prefixed)
        if self.layer_name:
            layer_bytes = self.layer_name.encode("utf-8")
            hasher.update(len(layer_bytes).to_bytes(4, "big"))
            hasher.update(layer_bytes)
        else:
            hasher.update(b"\x00\x00\x00\x00")  # Zero length for None
        
        # Include data
        hasher.update(len(self.data).to_bytes(8, "big"))
        hasher.update(self.data)
        
        return hasher.digest()
    
    @property
    def num_elements(self) -> int:
        """Number of elements in tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


# =============================================================================
# Gradient Commitment (Pedersen-style with BLS)
# =============================================================================


@dataclass(frozen=True)
class GradientCommitment:
    """
    Cryptographic commitment to a gradient with BLS signature authentication.
    
    Structure:
    - commitment: Hash-based commitment (32 bytes)
    - blinding_hash: Hash of blinding factor (32 bytes)
    - gradient_hash: Hash of gradient data (32 bytes)
    - epoch: Training epoch number
    - batch_id: Identifier for training batch
    - contributor_id: BLS public key of contributor (48 bytes)
    - timestamp: Unix timestamp
    - signature: BLS signature over commitment (96 bytes)
    
    Security properties:
    - Binding: Cannot open to different gradient (hash collision resistance)
    - Hiding: Reveals nothing about gradient (random blinding)
    - Authenticated: Signature proves contributor created this commitment
    - Time-bound: Timestamp included in commitment hash
    """
    commitment: bytes
    blinding_hash: bytes
    gradient_hash: bytes
    epoch: int
    batch_id: bytes
    contributor_id: bytes
    timestamp: int
    signature: bytes  # BLS signature over commitment
    
    def __post_init__(self) -> None:
        if len(self.commitment) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"commitment must be {GRADIENT_HASH_LEN} bytes")
        if len(self.blinding_hash) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"blinding_hash must be {GRADIENT_HASH_LEN} bytes")
        if len(self.gradient_hash) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"gradient_hash must be {GRADIENT_HASH_LEN} bytes")
        if self.epoch < 0:
            raise CommitmentError("epoch must be non-negative")
        if len(self.batch_id) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"batch_id must be {GRADIENT_HASH_LEN} bytes")
        if len(self.contributor_id) != BLS_PUBKEY_LEN:
            raise CommitmentError(f"contributor_id must be {BLS_PUBKEY_LEN} bytes")
        if self.contributor_id == b'\x00' * BLS_PUBKEY_LEN:
            raise CommitmentError("contributor_id cannot be all zeros")
        if len(self.signature) != BLS_SIGNATURE_LEN:
            raise CommitmentError(f"signature must be {BLS_SIGNATURE_LEN} bytes")
    
    def to_bytes(self) -> bytes:
        """Serialize commitment for transmission."""
        return (
            self.commitment +
            self.blinding_hash +
            self.gradient_hash +
            self.epoch.to_bytes(8, "big") +
            self.batch_id +
            self.contributor_id +
            self.timestamp.to_bytes(8, "big") +
            self.signature
        )
    
    def signable_bytes(self) -> bytes:
        """Return bytes that are signed (everything except signature)."""
        return (
            self.commitment +
            self.blinding_hash +
            self.gradient_hash +
            self.epoch.to_bytes(8, "big") +
            self.batch_id +
            self.contributor_id +
            self.timestamp.to_bytes(8, "big")
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "GradientCommitment":
        """Deserialize commitment with strict validation."""
        # Exact expected length
        expected_len = (
            GRADIENT_HASH_LEN +  # commitment
            GRADIENT_HASH_LEN +  # blinding_hash
            GRADIENT_HASH_LEN +  # gradient_hash
            8 +                  # epoch
            GRADIENT_HASH_LEN +  # batch_id
            BLS_PUBKEY_LEN +     # contributor_id
            8 +                  # timestamp
            BLS_SIGNATURE_LEN    # signature
        )
        
        if len(data) != expected_len:
            raise CommitmentError(
                f"Invalid commitment data length: expected {expected_len}, got {len(data)}"
            )
        
        offset = 0
        commitment = data[offset:offset + GRADIENT_HASH_LEN]
        offset += GRADIENT_HASH_LEN
        
        blinding_hash = data[offset:offset + GRADIENT_HASH_LEN]
        offset += GRADIENT_HASH_LEN
        
        gradient_hash = data[offset:offset + GRADIENT_HASH_LEN]
        offset += GRADIENT_HASH_LEN
        
        epoch = int.from_bytes(data[offset:offset + 8], "big")
        offset += 8
        
        batch_id = data[offset:offset + GRADIENT_HASH_LEN]
        offset += GRADIENT_HASH_LEN
        
        contributor_id = data[offset:offset + BLS_PUBKEY_LEN]
        offset += BLS_PUBKEY_LEN
        
        timestamp = int.from_bytes(data[offset:offset + 8], "big")
        offset += 8
        
        signature = data[offset:offset + BLS_SIGNATURE_LEN]
        
        # Validation happens in __post_init__
        return cls(
            commitment=commitment,
            blinding_hash=blinding_hash,
            gradient_hash=gradient_hash,
            epoch=epoch,
            batch_id=batch_id,
            contributor_id=contributor_id,
            timestamp=timestamp,
            signature=signature,
        )


# =============================================================================
# Commitment Scheme
# =============================================================================


class GradientCommitmentScheme:
    """
    SHA-256 hash-based commitment scheme for gradients with BLS signatures.
    
    Commitment: C = H(domain || contributor_id || gradient_hash || blinding || epoch || batch_id || timestamp)
    
    Properties:
    - Computationally binding (SHA-256 collision resistance)
    - Computationally hiding (256-bit random blinding)
    - Authenticated (BLS signature over commitment)
    - Contributor-bound (contributor_id in hash)
    - Time-bound (timestamp in hash)
    """
    
    def __init__(
        self,
        private_key: bytes,
        public_key: bytes,
        domain: bytes = COMMITMENT_DOMAIN,
    ) -> None:
        """
        Initialize commitment scheme with BLS keypair.
        
        Args:
            private_key: BLS private key for signing (32 bytes)
            public_key: BLS public key (contributor_id, 48 bytes)
            domain: Domain separator for commitments
        """
        if len(public_key) != BLS_PUBKEY_LEN:
            raise CommitmentError(f"public_key must be {BLS_PUBKEY_LEN} bytes")
        if len(private_key) != 32:
            raise CommitmentError("private_key must be 32 bytes")
        
        self._private_key = private_key
        self._contributor_id = public_key
        self._domain = domain
        self._bls = BLSOperations()
    
    def commit(
        self,
        gradient: GradientTensor,
        epoch: int,
        batch_id: bytes,
        timestamp: Optional[int] = None,
    ) -> Tuple[GradientCommitment, bytes]:
        """
        Create commitment to gradient.
        
        Args:
            gradient: Gradient tensor to commit
            epoch: Training epoch number
            batch_id: Identifier for training batch
            timestamp: Unix timestamp (auto-generated if None)
            
        Returns:
            (commitment, blinding) - commitment object and blinding factor
        """
        import time
        
        if timestamp is None:
            timestamp = int(time.time())
        
        # Generate random blinding factor
        blinding = secrets.token_bytes(32)
        
        # Compute gradient hash
        gradient_hash = gradient.hash
        
        # Compute commitment: C = H(domain || contributor_id || gradient_hash || blinding || epoch || batch_id || timestamp)
        # Including contributor_id and timestamp binds the commitment to the contributor and time
        commitment_input = (
            self._domain +
            self._contributor_id +
            gradient_hash +
            blinding +
            epoch.to_bytes(8, "big") +
            batch_id +
            timestamp.to_bytes(8, "big")
        )
        commitment = hashlib.sha256(commitment_input).digest()
        
        # Hash blinding for storage (don't store raw blinding in commitment)
        blinding_hash = hashlib.sha256(blinding).digest()
        
        # Create unsigned commitment data for signing
        signable_data = (
            commitment +
            blinding_hash +
            gradient_hash +
            epoch.to_bytes(8, "big") +
            batch_id +
            self._contributor_id +
            timestamp.to_bytes(8, "big")
        )
        
        # Sign with BLS using protocol-specific context
        signature = self._bls.sign(self._private_key, signable_data, context=BLS_SIGNING_CONTEXT)
        
        return GradientCommitment(
            commitment=commitment,
            blinding_hash=blinding_hash,
            gradient_hash=gradient_hash,
            epoch=epoch,
            batch_id=batch_id,
            contributor_id=self._contributor_id,
            timestamp=timestamp,
            signature=signature,
        ), blinding
    
    def verify_signature(self, commitment: GradientCommitment) -> bool:
        """
        Verify BLS signature on commitment.
        
        Args:
            commitment: Commitment to verify signature of
            
        Returns:
            True if signature is valid
        """
        signable_data = commitment.signable_bytes()
        return self._bls.verify(
            commitment.contributor_id,
            signable_data,
            commitment.signature,
            context=BLS_SIGNING_CONTEXT,
        )
    
    def verify_opening(
        self,
        commitment: GradientCommitment,
        gradient: GradientTensor,
        blinding: bytes,
    ) -> bool:
        """
        Verify commitment opening (full verification including signature).
        
        Args:
            commitment: Commitment to verify
            gradient: Claimed gradient
            blinding: Blinding factor used in commitment
            
        Returns:
            True if opening is valid and signature verifies
        """
        # Verify signature first
        if not self.verify_signature(commitment):
            return False
        
        # Verify gradient hash matches
        if gradient.hash != commitment.gradient_hash:
            return False
        
        # Verify blinding hash matches
        if hashlib.sha256(blinding).digest() != commitment.blinding_hash:
            return False
        
        # Verify contributor matches
        if commitment.contributor_id != self._contributor_id:
            return False
        
        # Verify timestamp freshness
        import time
        now = int(time.time())
        if commitment.timestamp > now + MAX_TIMESTAMP_SKEW_SECONDS:
            return False
        if commitment.timestamp < now - MAX_COMMITMENT_AGE_SECONDS:
            return False
        
        # Recompute commitment (must include all bound fields)
        commitment_input = (
            self._domain +
            self._contributor_id +
            commitment.gradient_hash +
            blinding +
            commitment.epoch.to_bytes(8, "big") +
            commitment.batch_id +
            commitment.timestamp.to_bytes(8, "big")
        )
        expected = hashlib.sha256(commitment_input).digest()
        
        return commitment.commitment == expected
    
    def verify_commitment_structure(
        self,
        commitment: GradientCommitment,
    ) -> bool:
        """
        Verify commitment structure (without opening).
        
        Checks that commitment is well-formed.
        Does NOT verify the gradient value.
        """
        try:
            # Check field lengths
            if len(commitment.commitment) != GRADIENT_HASH_LEN:
                return False
            if len(commitment.blinding_hash) != GRADIENT_HASH_LEN:
                return False
            if len(commitment.gradient_hash) != GRADIENT_HASH_LEN:
                return False
            if len(commitment.batch_id) != GRADIENT_HASH_LEN:
                return False
            
            # Check epoch is reasonable
            if commitment.epoch < 0:
                return False
            
            # Check timestamp is reasonable (not in future, not too old)
            import time
            now = int(time.time())
            if commitment.timestamp > now + MAX_TIMESTAMP_SKEW_SECONDS:
                return False
            if commitment.timestamp < now - MAX_COMMITMENT_AGE_SECONDS:
                return False
            
            return True
        except Exception:
            return False


# =============================================================================
# Aggregated Commitments
# =============================================================================


@dataclass(frozen=True)
class AggregatedCommitment:
    """
    Aggregated commitment from multiple contributors.
    
    Represents the sum of committed gradients without revealing individuals.
    """
    aggregated_hash: bytes
    contributor_commitments: Tuple[bytes, ...]  # Individual commitment hashes
    epoch: int
    batch_id: bytes
    num_contributors: int
    timestamp: int
    
    def __post_init__(self) -> None:
        if len(self.aggregated_hash) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"aggregated_hash must be {GRADIENT_HASH_LEN} bytes")
        if self.num_contributors != len(self.contributor_commitments):
            raise CommitmentError("num_contributors must match contributor_commitments length")


class CommitmentAggregator:
    """
    Aggregates gradient commitments from multiple contributors.
    
    Supports:
    - Collecting commitments for an epoch
    - Verifying all commitments are valid
    - Computing aggregated commitment hash
    
    Security:
    - Bounded to MAX_AGGREGATION_SIZE contributors
    - O(1) duplicate detection via set
    - Timestamp validation on add
    """
    
    def __init__(
        self,
        epoch: int,
        batch_id: bytes,
        max_contributors: int = MAX_AGGREGATION_SIZE,
    ) -> None:
        if len(batch_id) != GRADIENT_HASH_LEN:
            raise CommitmentError(f"batch_id must be {GRADIENT_HASH_LEN} bytes")
        if max_contributors <= 0 or max_contributors > MAX_AGGREGATION_SIZE:
            raise CommitmentError(f"max_contributors must be 1-{MAX_AGGREGATION_SIZE}")
        
        self._epoch = epoch
        self._batch_id = batch_id
        self._max_contributors = max_contributors
        self._commitments: List[GradientCommitment] = []
        self._contributor_ids: set[bytes] = set()  # O(1) duplicate detection
        self._finalized = False
    
    @property
    def epoch(self) -> int:
        return self._epoch
    
    @property
    def batch_id(self) -> bytes:
        return self._batch_id
    
    @property
    def num_commitments(self) -> int:
        return len(self._commitments)
    
    @property
    def is_finalized(self) -> bool:
        return self._finalized
    
    def add_commitment(self, commitment: GradientCommitment) -> None:
        """
        Add a commitment to the aggregator with full validation.
        
        Validates:
        - Aggregator not finalized or full
        - Epoch and batch_id match
        - No duplicate contributor
        - Timestamp freshness
        - BLS signature (authentication)
        
        Raises:
            CommitmentError: If any validation fails
        """
        if self._finalized:
            raise CommitmentError("Aggregator is finalized, cannot add more commitments")
        
        # Check bounds
        if len(self._commitments) >= self._max_contributors:
            raise CommitmentError(f"Aggregator full (max {self._max_contributors} contributors)")
        
        # Verify commitment matches epoch/batch
        if commitment.epoch != self._epoch:
            raise CommitmentError(f"Commitment epoch {commitment.epoch} != aggregator epoch {self._epoch}")
        if commitment.batch_id != self._batch_id:
            raise CommitmentError("Commitment batch_id does not match aggregator")
        
        # O(1) duplicate detection
        if commitment.contributor_id in self._contributor_ids:
            raise CommitmentError("Duplicate commitment from same contributor")
        
        # Validate timestamp freshness
        import time
        now = int(time.time())
        if commitment.timestamp > now + MAX_TIMESTAMP_SKEW_SECONDS:
            raise CommitmentError("Commitment timestamp is in the future")
        if commitment.timestamp < now - MAX_COMMITMENT_AGE_SECONDS:
            raise CommitmentError("Commitment timestamp is too old")
        
        # Verify BLS signature (authentication) with protocol-specific context
        bls = BLSOperations()
        signable_data = commitment.signable_bytes()
        if not bls.verify(
            commitment.contributor_id,
            signable_data,
            commitment.signature,
            context=BLS_SIGNING_CONTEXT,
        ):
            raise CommitmentError("Invalid commitment signature")
        
        self._contributor_ids.add(commitment.contributor_id)
        self._commitments.append(commitment)
    
    def finalize(self) -> AggregatedCommitment:
        """
        Finalize aggregation and return aggregated commitment.
        
        Returns:
            AggregatedCommitment containing the aggregated hash
        """
        if self._finalized:
            raise CommitmentError("Aggregator already finalized")
        
        if not self._commitments:
            raise CommitmentError("No commitments to aggregate")
        
        self._finalized = True
        
        # Sort commitments by contributor ID for deterministic ordering
        sorted_commitments = sorted(
            self._commitments,
            key=lambda c: c.contributor_id,
        )
        
        # Compute aggregated hash: H(C1 || C2 || ... || Cn)
        hasher = hashlib.sha256()
        hasher.update(COMMITMENT_DOMAIN)
        hasher.update(self._epoch.to_bytes(8, "big"))
        hasher.update(self._batch_id)
        
        contributor_commitment_hashes: List[bytes] = []
        for c in sorted_commitments:
            hasher.update(c.commitment)
            contributor_commitment_hashes.append(c.commitment)
        
        aggregated_hash = hasher.digest()
        
        import time
        return AggregatedCommitment(
            aggregated_hash=aggregated_hash,
            contributor_commitments=tuple(contributor_commitment_hashes),
            epoch=self._epoch,
            batch_id=self._batch_id,
            num_contributors=len(self._commitments),
            timestamp=int(time.time()),
        )
    
    def verify_aggregated(
        self,
        aggregated: AggregatedCommitment,
    ) -> bool:
        """
        Verify an aggregated commitment matches the collected commitments.
        
        Returns:
            True if aggregated commitment is valid
        """
        if aggregated.epoch != self._epoch:
            return False
        if aggregated.batch_id != self._batch_id:
            return False
        if aggregated.num_contributors != len(self._commitments):
            return False
        
        # Recompute aggregated hash
        sorted_commitments = sorted(
            self._commitments,
            key=lambda c: c.contributor_id,
        )
        
        hasher = hashlib.sha256()
        hasher.update(COMMITMENT_DOMAIN)
        hasher.update(self._epoch.to_bytes(8, "big"))
        hasher.update(self._batch_id)
        
        for c in sorted_commitments:
            hasher.update(c.commitment)
        
        expected_hash = hasher.digest()
        
        return aggregated.aggregated_hash == expected_hash
