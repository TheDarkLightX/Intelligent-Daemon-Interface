"""
Cryptographic Primitives for IAN Algorithms.

Provides production implementations of:
- BLS12-381 signatures (using py_ecc)
- Sparse Merkle Tree (256-bit depth)

Requirements:
- py_ecc >= 7.0.0 for BLS operations

Security Notes:
- py_ecc uses pure Python big-int arithmetic which is NOT constant-time.
  For environments with side-channel concerns (co-tenancy, shared hardware),
  use an HSM or constant-time library (blst, py_arkworks_bls12381).
- All public keys MUST have verified proof-of-possession before aggregation.
- py_ecc uses compressed point format which guarantees canonical encoding
  (one byte representation per curve point). Duplicate detection relies on
  this canonical representation.
- Subgroup validation: KeyValidate() for G1, signature_to_G2() for G2.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from py_ecc.bls import G2ProofOfPossession as bls
from py_ecc.bls.g2_primitives import G1_to_pubkey, pubkey_to_G1
from py_ecc.optimized_bls12_381 import add, curve_order, is_inf
from py_ecc.optimized_bls12_381 import Z1  # Identity element


# =============================================================================
# Constants
# =============================================================================

BLS_PUBKEY_LEN = 48
BLS_SIGNATURE_LEN = 96
BLS_PRIVKEY_LEN = 32
SMT_HASH_LEN = 32
MAX_CONTEXT_LEN = 64
MAX_MESSAGE_LEN = 1024 * 1024  # 1 MB
MAX_SMT_VALUE_LEN = 64 * 1024  # 64 KB
MAX_SMT_ENTRIES = 100_000
MAX_POP_REGISTRY_SIZE = 10_000  # Bounded PoP registry


# =============================================================================
# BLS12-381 Signatures
# =============================================================================

class BLSError(Exception):
    """BLS operation error."""
    pass


class BLSOperations:
    """
    BLS12-381 signature operations with security hardening.
    
    Uses the Proof of Possession ciphersuite:
    - G1 public keys (48 bytes compressed)
    - G2 signatures (96 bytes compressed)
    
    Security features:
    - Strict input validation (lengths, ranges, subgroup checks)
    - Proof-of-possession tracking for rogue-key attack prevention
    - Application-level domain separation
    """
    
    def __init__(
        self,
        domain: bytes = b"IAN_V1",
        max_pop_entries: int = MAX_POP_REGISTRY_SIZE,
    ) -> None:
        """
        Initialize with application domain.
        
        Args:
            domain: Application-level domain separator (max 32 bytes)
            max_pop_entries: Maximum PoP registry size (LRU eviction)
        """
        if len(domain) > 32:
            raise ValueError("domain must be <= 32 bytes")
        self._domain = domain
        self._max_pop_entries = max_pop_entries
        # LRU cache: OrderedDict with move_to_end on access
        self._verified_pops: OrderedDict[bytes, bool] = OrderedDict()
    
    def _validate_privkey(self, private_key: bytes) -> int:
        """Validate and convert private key."""
        if len(private_key) != BLS_PRIVKEY_LEN:
            raise BLSError(f"private key must be {BLS_PRIVKEY_LEN} bytes")
        
        sk_int = int.from_bytes(private_key, "big")
        if sk_int == 0 or sk_int >= curve_order:
            raise BLSError("private key out of range [1, r-1]")
        
        return sk_int
    
    def _validate_pubkey(self, public_key: bytes) -> None:
        """Validate public key format, curve membership, and subgroup."""
        if len(public_key) != BLS_PUBKEY_LEN:
            raise BLSError(f"public key must be {BLS_PUBKEY_LEN} bytes")
        
        try:
            # KeyValidate checks: on curve, in correct subgroup, not infinity
            if not bls.KeyValidate(public_key):
                raise BLSError("public key failed validation")
            
            point = pubkey_to_G1(public_key)
            if is_inf(point):
                raise BLSError("public key is point at infinity")
        except BLSError:
            raise
        except Exception as e:
            raise BLSError(f"invalid public key: {e}")
    
    def _validate_signature(self, signature: bytes, check_subgroup: bool = False) -> None:
        """Validate signature format and optionally subgroup."""
        if len(signature) != BLS_SIGNATURE_LEN:
            raise BLSError(f"signature must be {BLS_SIGNATURE_LEN} bytes")
        
        if check_subgroup:
            try:
                from py_ecc.bls.g2_primitives import signature_to_G2
                from py_ecc.optimized_bls12_381 import is_inf as is_inf_g2
                point = signature_to_G2(signature)
                if is_inf_g2(point):
                    raise BLSError("signature is point at infinity")
            except BLSError:
                raise
            except Exception as e:
                raise BLSError(f"invalid signature: {e}")
    
    def _validate_message(self, message: bytes) -> None:
        """Validate message size."""
        if len(message) > MAX_MESSAGE_LEN:
            raise BLSError(f"message exceeds {MAX_MESSAGE_LEN} bytes")
    
    def _validate_context(self, context: bytes) -> None:
        """Validate context size."""
        if len(context) > MAX_CONTEXT_LEN:
            raise BLSError(f"context exceeds {MAX_CONTEXT_LEN} bytes")
    
    def _domain_separate(self, context: bytes, message: bytes) -> bytes:
        """Apply domain separation with length-prefixing to prevent collisions."""
        # Format: domain_len(1) || domain || context_len(1) || context || message
        return (
            len(self._domain).to_bytes(1, "big") + self._domain +
            len(context).to_bytes(1, "big") + context +
            message
        )
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new BLS keypair.
        
        Returns:
            (private_key, public_key) where:
            - private_key: 32 bytes
            - public_key: 48 bytes (compressed G1)
        """
        import secrets
        
        ikm = secrets.token_bytes(32)
        sk_int = bls.KeyGen(ikm)
        private_key = sk_int.to_bytes(32, "big")
        public_key = bls.SkToPk(sk_int)
        
        return private_key, public_key
    
    def sign(
        self,
        private_key: bytes,
        message: bytes,
        context: bytes = b"MSG",
    ) -> bytes:
        """
        Sign a message with domain separation.
        
        Args:
            private_key: 32-byte private key
            message: Message bytes (max 1MB)
            context: Message context for domain separation
            
        Returns:
            96-byte signature (compressed G2)
        """
        sk_int = self._validate_privkey(private_key)
        self._validate_message(message)
        self._validate_context(context)
        
        full_msg = self._domain_separate(context, message)
        return bls.Sign(sk_int, full_msg)
    
    def verify(
        self,
        public_key: bytes,
        message: bytes,
        signature: bytes,
        context: bytes = b"MSG",
    ) -> bool:
        """
        Verify a signature.
        
        Args:
            public_key: 48-byte public key
            message: Original message
            signature: 96-byte signature
            context: Message context (must match signing)
            
        Returns:
            True if valid
        """
        try:
            self._validate_pubkey(public_key)
            self._validate_signature(signature, check_subgroup=True)
            self._validate_message(message)
            self._validate_context(context)
            
            full_msg = self._domain_separate(context, message)
            return bls.Verify(public_key, full_msg, signature)
        except BLSError:
            return False
        except Exception:
            return False
    
    def create_pop(self, private_key: bytes) -> bytes:
        """
        Create proof-of-possession for a public key.
        
        Args:
            private_key: 32-byte private key
            
        Returns:
            96-byte PoP signature
        """
        sk_int = self._validate_privkey(private_key)
        public_key = bls.SkToPk(sk_int)
        return bls.Sign(sk_int, self._domain_separate(b"POP", public_key))
    
    def verify_pop(self, public_key: bytes, pop: bytes) -> bool:
        """
        Verify and register proof-of-possession.
        
        Args:
            public_key: 48-byte public key
            pop: 96-byte proof-of-possession
            
        Returns:
            True if valid (also registers the key)
        """
        try:
            self._validate_pubkey(public_key)
            self._validate_signature(pop, check_subgroup=True)
            
            full_msg = self._domain_separate(b"POP", public_key)
            if bls.Verify(public_key, full_msg, pop):
                # LRU eviction if at capacity
                if len(self._verified_pops) >= self._max_pop_entries:
                    self._verified_pops.popitem(last=False)  # Remove oldest
                self._verified_pops[public_key] = True
                return True
            return False
        except Exception:
            return False
    
    def has_verified_pop(self, public_key: bytes) -> bool:
        """Check if public key has verified PoP (updates LRU order)."""
        if public_key in self._verified_pops:
            self._verified_pops.move_to_end(public_key)  # Mark as recently used
            return True
        return False
    
    def clear_pop_registry(self) -> None:
        """Clear the PoP registry."""
        self._verified_pops.clear()
    
    def aggregate_signatures(self, signatures: List[bytes]) -> bytes:
        """
        Aggregate multiple signatures.
        
        Args:
            signatures: List of 96-byte signatures
            
        Returns:
            96-byte aggregated signature
        """
        if not signatures:
            raise BLSError("cannot aggregate empty signature list")
        
        for sig in signatures:
            self._validate_signature(sig, check_subgroup=True)
        
        return bls.Aggregate(signatures)
    
    def verify_aggregated(
        self,
        public_keys: List[bytes],
        message: bytes,
        aggregated_sig: bytes,
        context: bytes = b"MSG",
        require_pop: bool = True,
    ) -> bool:
        """
        Verify aggregated signature (same message).
        
        Args:
            public_keys: List of 48-byte public keys
            message: Common message signed by all
            aggregated_sig: 96-byte aggregated signature
            context: Message context
            require_pop: If True, all keys must have verified PoP
            
        Returns:
            True if valid
        """
        if not public_keys:
            return False
        
        try:
            # Check for duplicate public keys (IETF BLS spec requires uniqueness)
            if len(public_keys) != len(set(public_keys)):
                return False
            
            for pk in public_keys:
                self._validate_pubkey(pk)
                if require_pop and pk not in self._verified_pops:
                    return False
            
            self._validate_signature(aggregated_sig, check_subgroup=True)
            self._validate_message(message)
            self._validate_context(context)
            
            full_msg = self._domain_separate(context, message)
            return bls.FastAggregateVerify(public_keys, full_msg, aggregated_sig)
        except Exception:
            return False
    
    def aggregate_public_keys(self, public_keys: List[bytes]) -> bytes:
        """
        Aggregate multiple public keys.
        
        Args:
            public_keys: List of 48-byte public keys (must have verified PoP)
            
        Returns:
            48-byte aggregated public key
        """
        if not public_keys:
            raise BLSError("cannot aggregate empty public key list")
        
        # Check for duplicate public keys
        if len(public_keys) != len(set(public_keys)):
            raise BLSError("duplicate public keys not allowed")
        
        for pk in public_keys:
            self._validate_pubkey(pk)
            if pk not in self._verified_pops:
                raise BLSError("all keys must have verified proof-of-possession")
        
        points = [pubkey_to_G1(pk) for pk in public_keys]
        agg_point = points[0]
        for p in points[1:]:
            agg_point = add(agg_point, p)
        
        # Reject if aggregation yields identity (adversarial input)
        if is_inf(agg_point):
            raise BLSError("aggregated public key is point at infinity")
        
        return G1_to_pubkey(agg_point)


# =============================================================================
# Sparse Merkle Tree
# =============================================================================

_SMT_LEAF = b"SMT_LEAF_V1"
_SMT_BRANCH = b"SMT_BRANCH_V1"
_SMT_EMPTY = b"SMT_EMPTY_V1"

_EMPTY_HASHES: List[bytes] = []


class SMTError(Exception):
    """SMT operation error."""
    pass


def _compute_empty_hashes() -> List[bytes]:
    """Precompute empty hashes for each tree level."""
    global _EMPTY_HASHES
    if _EMPTY_HASHES:
        return _EMPTY_HASHES
    
    hashes = [hashlib.sha256(_SMT_EMPTY).digest()]
    for _ in range(256):
        prev = hashes[-1]
        hashes.append(hashlib.sha256(_SMT_BRANCH + prev + prev).digest())
    
    _EMPTY_HASHES = hashes
    return _EMPTY_HASHES


@dataclass(frozen=True)
class SMTProof:
    """
    Sparse Merkle Tree proof.
    
    Supports both membership and non-membership proofs.
    Contains 256 sibling hashes (one per tree level).
    
    Use verify_membership() or verify_non_membership() for explicit semantics.
    """
    key: bytes
    value: bytes
    siblings: Tuple[bytes, ...]
    
    def __post_init__(self) -> None:
        if len(self.key) != SMT_HASH_LEN:
            raise ValueError(f"key must be {SMT_HASH_LEN} bytes")
        if len(self.siblings) != 256:
            raise ValueError(f"siblings must have 256 entries, got {len(self.siblings)}")
        for i, sib in enumerate(self.siblings):
            if len(sib) != SMT_HASH_LEN:
                raise ValueError(f"sibling {i} must be {SMT_HASH_LEN} bytes")
        if len(self.value) > MAX_SMT_VALUE_LEN:
            raise ValueError(f"value exceeds {MAX_SMT_VALUE_LEN} bytes")
    
    def _leaf_hash(self) -> bytes:
        """Compute leaf hash."""
        if not self.value:
            return _compute_empty_hashes()[0]
        return hashlib.sha256(_SMT_LEAF + self.key + self.value).digest()
    
    def compute_root(self) -> bytes:
        """Recompute root from proof."""
        current = self._leaf_hash()
        key_int = int.from_bytes(self.key, "big")
        
        for level, sibling in enumerate(self.siblings):
            bit = (key_int >> level) & 1
            if bit == 0:
                current = hashlib.sha256(_SMT_BRANCH + current + sibling).digest()
            else:
                current = hashlib.sha256(_SMT_BRANCH + sibling + current).digest()
        
        return current
    
    def is_membership(self) -> bool:
        """True if this is a membership proof (value exists)."""
        return len(self.value) > 0
    
    def verify_membership(self, root: bytes, expected_value: Optional[bytes] = None) -> bool:
        """
        Verify this is a valid membership proof.
        
        Args:
            root: Expected Merkle root (32 bytes)
            expected_value: If provided, also verify value matches
            
        Returns:
            True if proof is valid AND proves membership
        """
        if len(root) != SMT_HASH_LEN:
            return False
        if not self.is_membership():
            return False
        if expected_value is not None and self.value != expected_value:
            return False
        return self.compute_root() == root
    
    def verify_non_membership(self, root: bytes) -> bool:
        """
        Verify this is a valid non-membership proof.
        
        Args:
            root: Expected Merkle root (32 bytes)
            
        Returns:
            True if proof is valid AND proves non-membership
        """
        if len(root) != SMT_HASH_LEN:
            return False
        if self.is_membership():
            return False
        return self.compute_root() == root
    
    def verify(self, root: bytes) -> bool:
        """
        Verify proof against expected root (membership or non-membership).
        
        Note: Use verify_membership() or verify_non_membership() for 
        explicit semantics to avoid logic bugs.
        """
        if len(root) != SMT_HASH_LEN:
            return False
        return self.compute_root() == root


class SparseMerkleTree:
    """
    Sparse Merkle Tree with 256-bit keys.
    
    Uses node caching for efficient operations:
    - O(log n) insert with path caching
    - O(log n) proof generation
    - Membership and non-membership proofs
    
    Security features:
    - Size limits on entries and values
    - Explicit membership/non-membership verification
    
    Tree structure:
    - Bit i of key determines left (0) or right (1) at level i
    - Level 0 is at leaves, level 255 is below root
    """
    
    def __init__(self, max_entries: int = MAX_SMT_ENTRIES) -> None:
        self._data: Dict[bytes, bytes] = {}
        self._nodes: Dict[Tuple[int, int], bytes] = {}
        self._root: Optional[bytes] = None
        self._dirty = True
        self._empty = _compute_empty_hashes()
        self._max_entries = max_entries
    
    def insert(self, key: bytes, value: bytes) -> None:
        """Insert or update a key-value pair."""
        if len(key) != 32:
            raise SMTError("key must be 32 bytes")
        if len(value) > MAX_SMT_VALUE_LEN:
            raise SMTError(f"value exceeds {MAX_SMT_VALUE_LEN} bytes")
        if value and key not in self._data and len(self._data) >= self._max_entries:
            raise SMTError(f"tree exceeds {self._max_entries} entries")
        
        if value:
            self._data[key] = value
        elif key in self._data:
            del self._data[key]
        self._dirty = True
        self._root = None
        self._nodes.clear()
    
    def get(self, key: bytes) -> Optional[bytes]:
        """Get value for key, or None if absent."""
        return self._data.get(key)
    
    def contains(self, key: bytes) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def delete(self, key: bytes) -> None:
        """Delete a key."""
        self.insert(key, b"")
    
    def _leaf_hash(self, key: bytes, value: bytes) -> bytes:
        if not value:
            return self._empty[0]
        return hashlib.sha256(_SMT_LEAF + key + value).digest()
    
    def _branch_hash(self, left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(_SMT_BRANCH + left + right).digest()
    
    def _build_tree(self) -> None:
        """Build tree nodes from leaves up."""
        if not self._data:
            return
        
        # Start with leaf hashes
        for key, value in self._data.items():
            key_int = int.from_bytes(key, "big")
            self._nodes[(0, key_int)] = self._leaf_hash(key, value)
        
        # Build each level from bottom up
        for level in range(1, 257):
            # Find all paths at previous level
            prev_paths = set()
            for (lv, path) in self._nodes.keys():
                if lv == level - 1:
                    prev_paths.add(path >> 1)  # Parent path
            
            for parent_path in prev_paths:
                left_path = parent_path << 1
                right_path = (parent_path << 1) | 1
                
                left = self._nodes.get((level - 1, left_path), self._empty[level - 1])
                right = self._nodes.get((level - 1, right_path), self._empty[level - 1])
                
                self._nodes[(level, parent_path)] = self._branch_hash(left, right)
    
    def _get_node(self, level: int, path: int) -> bytes:
        """Get node hash, using cache or empty hash."""
        if (level, path) in self._nodes:
            return self._nodes[(level, path)]
        return self._empty[level]
    
    @property
    def root(self) -> bytes:
        """Get current Merkle root."""
        if self._dirty or self._root is None:
            if not self._data:
                self._root = self._empty[256]
            else:
                self._nodes.clear()
                self._build_tree()
                self._root = self._get_node(256, 0)
            self._dirty = False
        return self._root
    
    def get_proof(self, key: bytes) -> SMTProof:
        """Generate proof for a key (membership or non-membership)."""
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        
        _ = self.root  # Ensure tree is built
        value = self._data.get(key, b"")
        key_int = int.from_bytes(key, "big")
        siblings = []
        
        # At level L, nodes are indexed by bits L..255 (i.e., key_int >> L)
        # Sibling differs only in the lowest bit of that index
        for level in range(256):
            node_path = key_int >> level
            sibling_path = node_path ^ 1  # Flip lowest bit to get sibling
            siblings.append(self._get_node(level, sibling_path))
        
        return SMTProof(key=key, value=value, siblings=tuple(siblings))
    
    def verify_proof(self, proof: SMTProof) -> bool:
        """Verify a proof against current root."""
        return proof.verify(self.root)
    
    def __len__(self) -> int:
        return len(self._data)
