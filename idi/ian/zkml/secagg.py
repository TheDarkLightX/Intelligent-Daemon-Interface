"""
Secure Aggregation Primitives for IAN zkML Training.

This module provides BUILDING BLOCKS for secure aggregation:
- Shamir (t,n) threshold secret sharing
- Pairwise mask generation using X25519/ECDH shared secrets
- XOR-based gradient aggregation

IMPORTANT: This is NOT a complete secure aggregation protocol.

Current limitations:
- No dropout recovery: All participants must complete the round
- No key confirmation or transcript binding (ephemeral keys are signed)
- XOR-based masking is illustrative, not a full SecAgg protocol

What this module DOES provide:
- Correct Shamir secret sharing with validation
- Real X25519/ECDH pairwise mask derivation (server cannot compute masks)
- Protocol structure for secure aggregation flows

Use cases:
- Learning secure aggregation concepts
- Integration testing with real ECDH-based masking
- Foundation for a full SecAgg protocol
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from ..algorithms.crypto import (
    BLSError,
    BLSOperations,
    BLS_PRIVKEY_LEN,
    BLS_PUBKEY_LEN,
    BLS_SIGNATURE_LEN,
)

# =============================================================================
# Constants
# =============================================================================

# Security parameters
SECAGG_DOMAIN = b"IAN_SECAGG_V1"
SECAGG_BLS_CONTEXT = b"IAN_SECAGG_EPHEMERAL_V1"
MIN_PARTICIPANTS = 3
MAX_PARTICIPANTS = 1000
DEFAULT_THRESHOLD = 2  # t-of-n threshold (minimum survivors)
MAX_GRADIENT_SIZE = 100 * 1024 * 1024  # 100 MB
MASK_SEED_LEN = 32
SHARE_LEN = 32

# Protocol timeouts (seconds)
SETUP_TIMEOUT = 60
MASKING_TIMEOUT = 120
SHARING_TIMEOUT = 120
AGGREGATION_TIMEOUT = 300


class SecAggError(Exception):
    """Error in secure aggregation protocol."""
    pass


class SecAggPhase(Enum):
    """Protocol phases."""
    SETUP = auto()
    KEY_EXCHANGE = auto()
    MASKING = auto()
    SHARING = auto()
    AGGREGATION = auto()
    UNMASKING = auto()
    COMPLETE = auto()
    FAILED = auto()


# =============================================================================
# Shamir Secret Sharing
# =============================================================================


class ShamirSecretSharing:
    """
    Shamir's (t, n) threshold secret sharing over a prime field.
    
    Uses a 256-bit prime (2^256 - 189 is prime) for compatibility with SHA-256 outputs.
    """
    
    # 2^256 - 189 is a verified prime number
    PRIME = 2**256 - 189
    
    @classmethod
    def split(
        cls,
        secret: bytes,
        n: int,
        t: int,
    ) -> List[Tuple[int, bytes]]:
        """
        Split secret into n shares with threshold t.
        
        Args:
            secret: 32-byte secret to split
            n: Number of shares to create
            t: Threshold (minimum shares to reconstruct)
            
        Returns:
            List of (index, share) tuples where index is 1-based
        """
        if len(secret) != SHARE_LEN:
            raise SecAggError(f"Secret must be {SHARE_LEN} bytes")
        if t < 2:
            raise SecAggError("Threshold must be >= 2")
        if n < t:
            raise SecAggError("n must be >= t")
        if n > MAX_PARTICIPANTS:
            raise SecAggError(f"n must be <= {MAX_PARTICIPANTS}")
        
        # Convert secret to integer (already < 2^256, so < PRIME)
        secret_int = int.from_bytes(secret, "big")
        if secret_int >= cls.PRIME:
            raise SecAggError("Secret value exceeds field prime")
        
        # Generate random coefficients for polynomial
        coefficients = [secret_int]
        for _ in range(t - 1):
            # Generate coefficient strictly less than PRIME
            while True:
                coef = int.from_bytes(secrets.token_bytes(32), "big")
                if coef < cls.PRIME:
                    break
            coefficients.append(coef)
        
        # Evaluate polynomial at points 1..n
        shares: List[Tuple[int, bytes]] = []
        for x in range(1, n + 1):
            y = cls._eval_poly(coefficients, x)
            share_bytes = y.to_bytes(32, "big")
            shares.append((x, share_bytes))
        
        return shares
    
    @classmethod
    def reconstruct(
        cls,
        shares: List[Tuple[int, bytes]],
    ) -> bytes:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (index, share) tuples where index is 1-based
            
        Returns:
            Reconstructed 32-byte secret
            
        Raises:
            SecAggError: If shares are invalid (duplicates, zero indices, etc.)
        """
        if len(shares) < 2:
            raise SecAggError("Need at least 2 shares to reconstruct")
        
        # Convert shares to integers and validate
        points: List[Tuple[int, int]] = []
        seen_indices: Set[int] = set()
        
        for x, share_bytes in shares:
            # Validate index
            if x <= 0:
                raise SecAggError(f"Share index must be positive, got {x}")
            if x in seen_indices:
                raise SecAggError(f"Duplicate share index: {x}")
            seen_indices.add(x)
            
            # Validate share bytes
            if len(share_bytes) != SHARE_LEN:
                raise SecAggError(f"Share must be {SHARE_LEN} bytes")
            
            y = int.from_bytes(share_bytes, "big")
            if y >= cls.PRIME:
                raise SecAggError("Share value exceeds field prime")
            
            points.append((x, y))
        
        # Lagrange interpolation at x=0
        secret_int = 0
        for i, (xi, yi) in enumerate(points):
            numerator = 1
            denominator = 1
            for j, (xj, _) in enumerate(points):
                if i != j:
                    numerator = (numerator * (-xj)) % cls.PRIME
                    denominator = (denominator * (xi - xj)) % cls.PRIME
            
            # denominator cannot be 0 because we validated no duplicate indices
            # Modular inverse using Fermat's little theorem (PRIME is prime)
            lagrange = (numerator * pow(denominator, cls.PRIME - 2, cls.PRIME)) % cls.PRIME
            secret_int = (secret_int + yi * lagrange) % cls.PRIME
        
        return secret_int.to_bytes(32, "big")
    
    @classmethod
    def _eval_poly(cls, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coef in reversed(coefficients):
            result = (result * x + coef) % cls.PRIME
        return result


# =============================================================================
# Pairwise Masking
# =============================================================================


@dataclass(frozen=True)
class ParticipantKeys:
    """Ephemeral keys for a participant in secure aggregation."""
    participant_id: bytes  # BLS public key (identity)
    ephemeral_public: bytes  # Ephemeral X25519 public key (32 bytes)
    round_id: bytes  # Round identifier (32 bytes)
    ephemeral_signature: bytes  # BLS signature over ephemeral key material
    
    def __post_init__(self) -> None:
        if len(self.participant_id) != BLS_PUBKEY_LEN:
            raise SecAggError(f"participant_id must be {BLS_PUBKEY_LEN} bytes")
        if len(self.ephemeral_public) != 32:
            raise SecAggError("ephemeral_public must be 32 bytes")
        if len(self.round_id) != 32:
            raise SecAggError("round_id must be 32 bytes")
        if len(self.ephemeral_signature) != BLS_SIGNATURE_LEN:
            raise SecAggError(f"ephemeral_signature must be {BLS_SIGNATURE_LEN} bytes")


def _ephemeral_signing_payload(participant_id: bytes, round_id: bytes, ephemeral_public: bytes) -> bytes:
    """Build the signable payload for ephemeral key authentication."""
    return (
        SECAGG_DOMAIN +
        b"EPHEMERAL_KEY" +
        participant_id +
        round_id +
        ephemeral_public
    )


class PairwiseMasking:
    """
    Generates pairwise masks using Diffie-Hellman key agreement.
    
    Each pair of participants computes a shared secret and derives
    masks that cancel out when aggregated.
    """
    
    def __init__(self, participant_id: bytes, round_id: bytes) -> None:
        """
        Initialize masking for a participant.
        
        Args:
            participant_id: BLS public key of this participant
            round_id: Unique identifier for this aggregation round
        """
        if len(participant_id) != BLS_PUBKEY_LEN:
            raise SecAggError(f"participant_id must be {BLS_PUBKEY_LEN} bytes")
        if len(round_id) != 32:
            raise SecAggError("round_id must be 32 bytes")
        
        self._participant_id = participant_id
        self._round_id = round_id
        
        # Generate ephemeral X25519 keypair
        self._ephemeral_private = X25519PrivateKey.generate()
        self._ephemeral_public = self._ephemeral_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    
    @property
    def participant_id(self) -> bytes:
        return self._participant_id
    
    @property
    def round_id(self) -> bytes:
        return self._round_id
    
    @property
    def ephemeral_public(self) -> bytes:
        return self._ephemeral_public
    
    def get_keys(self, signature: bytes) -> ParticipantKeys:
        """Get participant keys for sharing with others."""
        return ParticipantKeys(
            participant_id=self._participant_id,
            ephemeral_public=self._ephemeral_public,
            round_id=self._round_id,
            ephemeral_signature=signature,
        )
    
    def compute_shared_secret(self, other_id: bytes, other_public: bytes) -> bytes:
        """
        Compute shared secret with another participant.
        
        IMPORTANT: Uses X25519 ECDH so only participants with private keys
        can compute the shared secret (the server cannot).
        """
        if len(other_id) != BLS_PUBKEY_LEN:
            raise SecAggError(f"other_id must be {BLS_PUBKEY_LEN} bytes")
        if len(other_public) != 32:
            raise SecAggError("other_public must be 32 bytes")
        
        peer_public = X25519PublicKey.from_public_bytes(other_public)
        shared = self._ephemeral_private.exchange(peer_public)
        if shared == b"\x00" * len(shared):
            raise SecAggError("low-order point detected")
        return shared
    
    def compute_pairwise_mask(
        self,
        other_id: bytes,
        other_public: bytes,
        gradient_size: int,
    ) -> bytes:
        """
        Compute pairwise mask for gradient.
        
        The mask is deterministic and symmetric. Both parties derive the
        SAME mask from their shared secret.
        
        For XOR-based aggregation, when both parties XOR the same mask:
        (g_A ⊕ m) ⊕ (g_B ⊕ m) = g_A ⊕ g_B (masks cancel)
        
        Returns:
            The mask bytes (same for both parties in the pair)
        """
        if gradient_size <= 0 or gradient_size > MAX_GRADIENT_SIZE:
            raise SecAggError(f"gradient_size must be 1-{MAX_GRADIENT_SIZE}")
        
        shared_secret = self.compute_shared_secret(other_id, other_public)
        
        # Bind masks to round and identities while preserving ECDH secrecy.
        if self._participant_id < other_id:
            id_pair = self._participant_id + other_id
        else:
            id_pair = other_id + self._participant_id
        mask_seed = hashlib.sha256(
            SECAGG_DOMAIN +
            b"PAIRWISE_MASK" +
            shared_secret +
            id_pair +
            self._round_id
        ).digest()
        
        # Generate mask using PRF (expand mask seed to gradient size)
        return self._expand_mask(mask_seed, gradient_size)
    
    def _expand_mask(self, seed: bytes, size: int) -> bytes:
        """Expand seed to mask of given size using HKDF-style expansion."""
        result = bytearray()
        counter = 0
        while len(result) < size:
            block = hashlib.sha256(
                SECAGG_DOMAIN +
                b"MASK" +
                seed +
                counter.to_bytes(4, "big")
            ).digest()
            result.extend(block)
            counter += 1
        return bytes(result[:size])


# =============================================================================
# Masked Gradient
# =============================================================================


@dataclass(frozen=True)
class MaskedGradient:
    """
    A gradient masked for secure aggregation.
    
    The masked value hides the original gradient from the aggregator.
    Only the aggregate of all masked gradients reveals useful information.
    """
    participant_id: bytes
    round_id: bytes
    epoch: int
    masked_data: bytes
    commitment_hash: bytes  # Hash of original gradient for verification
    
    def __post_init__(self) -> None:
        if len(self.participant_id) != BLS_PUBKEY_LEN:
            raise SecAggError(f"participant_id must be {BLS_PUBKEY_LEN} bytes")
        if len(self.round_id) != 32:
            raise SecAggError("round_id must be 32 bytes")
        if len(self.masked_data) > MAX_GRADIENT_SIZE:
            raise SecAggError(f"masked_data exceeds {MAX_GRADIENT_SIZE} bytes")
        if len(self.commitment_hash) != 32:
            raise SecAggError("commitment_hash must be 32 bytes")


# =============================================================================
# Secure Aggregation Session
# =============================================================================


@dataclass
class SecAggSession:
    """
    Manages a secure aggregation session for one round.
    
    Coordinates the protocol phases and tracks participant state.
    """
    round_id: bytes
    epoch: int
    threshold: int
    gradient_size: int
    
    # Participant tracking
    participants: Dict[bytes, ParticipantKeys] = field(default_factory=dict)
    masked_gradients: Dict[bytes, MaskedGradient] = field(default_factory=dict)
    
    # Phase tracking
    phase: SecAggPhase = SecAggPhase.SETUP
    
    def __post_init__(self) -> None:
        if len(self.round_id) != 32:
            raise SecAggError("round_id must be 32 bytes")
        if self.threshold < DEFAULT_THRESHOLD:
            raise SecAggError(f"threshold must be >= {DEFAULT_THRESHOLD}")
        if self.gradient_size <= 0 or self.gradient_size > MAX_GRADIENT_SIZE:
            raise SecAggError(f"gradient_size must be 1-{MAX_GRADIENT_SIZE}")
    
    @property
    def num_participants(self) -> int:
        return len(self.participants)
    
    @property
    def num_submitted(self) -> int:
        return len(self.masked_gradients)
    
    @property
    def can_aggregate(self) -> bool:
        """Check if all participants have submitted (simplified: no dropout support)."""
        return len(self.masked_gradients) == len(self.participants)
    
    def register_participant(self, keys: ParticipantKeys) -> None:
        """Register a participant for this round."""
        if self.phase != SecAggPhase.SETUP:
            raise SecAggError("Cannot register participants outside SETUP phase")
        if keys.round_id != self.round_id:
            raise SecAggError("Participant round_id does not match session")
        if keys.participant_id in self.participants:
            raise SecAggError("Participant already registered")
        if len(self.participants) >= MAX_PARTICIPANTS:
            raise SecAggError(f"Maximum {MAX_PARTICIPANTS} participants reached")
        
        self.participants[keys.participant_id] = keys
    
    def start_key_exchange(self) -> None:
        """Transition to key exchange phase."""
        if self.phase != SecAggPhase.SETUP:
            raise SecAggError("Must be in SETUP phase")
        if len(self.participants) < MIN_PARTICIPANTS:
            raise SecAggError(f"Need at least {MIN_PARTICIPANTS} participants")
        
        self.phase = SecAggPhase.KEY_EXCHANGE
    
    def submit_masked_gradient(self, masked: MaskedGradient) -> None:
        """Submit a masked gradient from a participant."""
        if self.phase not in (SecAggPhase.KEY_EXCHANGE, SecAggPhase.MASKING, SecAggPhase.AGGREGATION):
            raise SecAggError("Cannot submit gradients in current phase")
        if masked.participant_id not in self.participants:
            raise SecAggError("Unknown participant")
        if masked.round_id != self.round_id:
            raise SecAggError("round_id does not match session")
        if len(masked.masked_data) != self.gradient_size:
            raise SecAggError(f"masked_data must be {self.gradient_size} bytes")
        if masked.epoch != self.epoch:
            raise SecAggError("epoch does not match session")
        if len(masked.commitment_hash) != 32:
            raise SecAggError("commitment_hash must be 32 bytes")
        
        self.masked_gradients[masked.participant_id] = masked
    
    
    def finalize(self) -> Optional[bytes]:
        """
        Finalize aggregation and compute aggregate gradient.
        
        Requires ALL participants to submit masked gradients.
        
        Returns:
            XOR of all gradients (pairwise masks cancel), or None if incomplete
        """
        # Require ALL participants to have submitted
        if len(self.masked_gradients) != len(self.participants):
            self.phase = SecAggPhase.FAILED
            return None
        
        # XOR all masked gradients
        # Pairwise masks cancel: (g_A ⊕ m_AB) ⊕ (g_B ⊕ m_AB) = g_A ⊕ g_B
        aggregate = bytearray(self.gradient_size)
        for masked in self.masked_gradients.values():
            for i, b in enumerate(masked.masked_data):
                aggregate[i] ^= b
        
        self.phase = SecAggPhase.COMPLETE
        return bytes(aggregate)


# =============================================================================
# Participant-side SecAgg Client
# =============================================================================


class SecAggParticipant:
    """
    Client-side secure aggregation participant.
    
    Handles key generation, masking, and share distribution.
    """
    
    def __init__(
        self,
        participant_id: bytes,
        round_id: bytes,
        gradient_size: int,
        bls_private_key: bytes,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        if len(participant_id) != BLS_PUBKEY_LEN:
            raise SecAggError(f"participant_id must be {BLS_PUBKEY_LEN} bytes")
        if len(bls_private_key) != BLS_PRIVKEY_LEN:
            raise SecAggError(f"bls_private_key must be {BLS_PRIVKEY_LEN} bytes")
        
        self._participant_id = participant_id
        self._round_id = round_id
        self._gradient_size = gradient_size
        self._threshold = threshold
        self._bls_private_key = bls_private_key
        self._bls = BLSOperations()
        
        # Initialize masking
        self._masking = PairwiseMasking(participant_id, round_id)
        self._ephemeral_signature = self._sign_ephemeral_public()
        
        # Store other participants' keys
        self._other_keys: Dict[bytes, ParticipantKeys] = {}
    
    @property
    def keys(self) -> ParticipantKeys:
        """Get this participant's keys."""
        return self._masking.get_keys(self._ephemeral_signature)

    def _sign_ephemeral_public(self) -> bytes:
        """Sign the ephemeral public key with the participant identity key."""
        payload = _ephemeral_signing_payload(
            self._participant_id,
            self._round_id,
            self._masking.ephemeral_public,
        )
        try:
            signature = self._bls.sign(self._bls_private_key, payload, context=SECAGG_BLS_CONTEXT)
        except BLSError as exc:
            raise SecAggError(f"Failed to sign ephemeral key: {exc}")
        if not self._bls.verify(
            self._participant_id,
            payload,
            signature,
            context=SECAGG_BLS_CONTEXT,
        ):
            raise SecAggError("BLS private key does not match participant_id")
        return signature
    
    def add_peer(self, keys: ParticipantKeys) -> None:
        """Add another participant's keys."""
        if keys.participant_id == self._participant_id:
            raise SecAggError("Cannot add self as peer")
        if keys.round_id != self._round_id:
            raise SecAggError("Peer round_id does not match")
        payload = _ephemeral_signing_payload(
            keys.participant_id,
            keys.round_id,
            keys.ephemeral_public,
        )
        if not self._bls.verify(
            keys.participant_id,
            payload,
            keys.ephemeral_signature,
            context=SECAGG_BLS_CONTEXT,
        ):
            raise SecAggError("Invalid signature for peer ephemeral key")
        
        self._other_keys[keys.participant_id] = keys
    
    
    def mask_gradient(self, gradient: bytes, epoch: int = 0) -> MaskedGradient:
        """
        Mask gradient for secure aggregation.
        
        XOR-based masking: Each pair of participants XORs the SAME mask.
        When all masked gradients are XORed together, pairwise masks cancel:
        (g_A ⊕ m_AB) ⊕ (g_B ⊕ m_AB) = g_A ⊕ g_B (m_AB cancels)
        
        This simplified implementation does NOT use self-masks, so the
        aggregate is the true XOR of all gradients (usable for training).
        
        Args:
            gradient: Raw gradient bytes
            epoch: Training epoch number
            
        Returns:
            MaskedGradient with pairwise masks applied
        """
        if len(gradient) != self._gradient_size:
            raise SecAggError(f"gradient must be {self._gradient_size} bytes")
        
        # Start with raw gradient
        masked = bytearray(gradient)
        
        # Apply pairwise masks with all other participants
        # Both parties XOR the same mask, so masks cancel on aggregation
        for peer_id, peer_keys in self._other_keys.items():
            mask = self._masking.compute_pairwise_mask(
                peer_id,
                peer_keys.ephemeral_public,
                self._gradient_size,
            )
            for i, b in enumerate(mask):
                masked[i] ^= b
        
        # NOTE: No self-mask in simplified version (no dropout recovery)
        # The aggregate will be the true XOR of all gradients
        
        # Compute commitment hash of original gradient
        commitment_hash = hashlib.sha256(
            SECAGG_DOMAIN + b"COMMIT" + gradient
        ).digest()
        
        return MaskedGradient(
            participant_id=self._participant_id,
            round_id=self._round_id,
            epoch=epoch,
            masked_data=bytes(masked),
            commitment_hash=commitment_hash,
        )
    
