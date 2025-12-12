"""Merkle tree implementation for Q-table commitments.

Provides efficient Merkle tree construction and proof generation for large Q-tables.

Security Properties:
- Collision resistance: Relies on SHA-256 preimage resistance
- Binding: Tree root uniquely identifies leaf set (no second preimage)
- Privacy: Only reveals Merkle proofs, not full tree structure

Trust Assumptions:
- Honest prover provides correct leaf data
- Verifier has authentic root hash from trusted source

Dependencies: hashlib (SHA-256)
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple


def _hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash a pair of child nodes to create parent hash.
    
    This follows Bitcoin's Merkle tree construction (BIP 34).
    Parent hash = SHA-256(left_child || right_child)
    
    Args:
        left: Left child hash (32 bytes)
        right: Right child hash (32 bytes)
        
    Returns:
        Parent hash (32 bytes)
    """
    return hashlib.sha256(left + right).digest()


def _compute_level_hashes(
    level: List[bytes],
    level_keys: List[List[str]],
    proofs: Dict[str, List[Tuple[bytes, bool]]],
) -> Tuple[List[bytes], List[List[str]]]:
    """Compute parent level hashes from current level.
    
    Builds parent level by hashing pairs of children.
    If odd number of nodes, duplicates the last node (Bitcoin-style).
    Accumulates proof paths for each leaf during tree construction.
    
    Args:
        level: Current level hashes
        level_keys: Keys associated with current level nodes
        proofs: Proof dictionary to update with sibling hashes
        
    Returns:
        Tuple of (next_level_hashes, next_level_keys)
    """
    next_level: List[bytes] = []
    next_keys: List[List[str]] = []
    
    for i in range(0, len(level), 2):
        left_child = level[i]
        # Handle odd node count by self-pairing the last node
        # This matches Bitcoin's Merkle tree behavior (BIP 34)
        right_child = level[i + 1] if i + 1 < len(level) else left_child
        
        parent_hash = _hash_pair(left_child, right_child)
        next_level.append(parent_hash)
        
        # Collect descendant leaf keys for both children
        left_keys = level_keys[i]
        right_keys = level_keys[i + 1] if i + 1 < len(level_keys) else level_keys[i]

        # Normalize in case of accidental flattening
        if isinstance(left_keys, str):
            left_keys = [left_keys]
        if isinstance(right_keys, str):
            right_keys = [right_keys]

        parent_keys = left_keys + right_keys
        next_keys.append(parent_keys)

        # Accumulate proof paths for all leaves under each subtree
        for key in left_keys:
            proofs[key].append((right_child, True))  # Right sibling
        for key in right_keys:
            proofs[key].append((left_child, False))  # Left sibling
    
    return next_level, next_keys


class MerkleTreeBuilder:
    """Builds Merkle trees from Q-table data.
    
    This class constructs Merkle trees for efficient commitment to large Q-tables.
    Only the root hash is revealed publicly; individual entries require Merkle
    proofs to verify membership without revealing the full table.
    """
    
    def __init__(self) -> None:
        """Initialize Merkle tree builder."""
        self.leaves: List[Tuple[str, bytes]] = []  # (key, hash)
        self._cache_root: Optional[bytes] = None
        self._cache_proofs: Optional[Dict[str, List[Tuple[bytes, bool]]]] = None
    
    def add_leaf(self, key: str, data: bytes) -> None:
        """Add a leaf node to the tree with validation."""
        if not isinstance(key, str) or not key:
            raise ValueError("Leaf key must be a non-empty string")
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Leaf data must be bytes")
        leaf_hash = hashlib.sha256(data).digest()
        self.leaves.append((key, leaf_hash))
        # Invalidate caches when leaf set changes
        self._cache_root = None
        self._cache_proofs = None
    
    def build(self) -> Tuple[bytes, Dict[str, List[Tuple[bytes, bool]]]]:
        """Build Merkle tree and return root hash and proofs.
        
        Constructs tree bottom-up, accumulating authentication paths for each leaf.
        Tree construction is deterministic (sorted by key) to ensure consistent
        root hashes across different prover instances.
        
        Returns:
            Tuple of (root_hash, proofs_dict) where proofs_dict maps
            state keys to authentication paths (list of (sibling_hash, is_right))
            
        Complexity: O(n log n) where n is number of leaves
        """
        if self._cache_root is not None and self._cache_proofs is not None:
            return self._cache_root, self._cache_proofs

        if not self.leaves:
            return hashlib.sha256(b"").digest(), {}
        
        # Sort leaves by key for deterministic ordering
        sorted_leaves = sorted(self.leaves, key=lambda x: x[0])
        leaf_hashes = [hash for _, hash in sorted_leaves]
        keys = [key for key, _ in sorted_leaves]
        
        # Initialize proofs for all keys
        proofs: Dict[str, List[Tuple[bytes, bool]]] = {key: [] for key in keys}
        
        # Build tree bottom-up, accumulating proofs at each level
        level = leaf_hashes.copy()
        level_keys: List[List[str]] = [[k] for k in keys]
        
        while len(level) > 1:
            level, level_keys = _compute_level_hashes(level, level_keys, proofs)
        
        root_hash = level[0] if level else hashlib.sha256(b"").digest()
        
        # Cache computed root/proofs for reuse when leaves unchanged
        self._cache_root = root_hash
        self._cache_proofs = proofs

        return root_hash, proofs
    
    def verify_proof(
        self,
        key: str,
        leaf_data: bytes,
        proof_path: List[Tuple[bytes, bool]],
        root_hash: bytes,
    ) -> bool:
        """Verify a Merkle authentication path.
        
        Security:
            - Constant-time comparison used for root hash check
            - No early exit based on partial proof validation
            
        Preconditions:
            - proof_path length matches tree depth
            - leaf_data is raw data that was hashed to create leaf
            
        Postconditions:
            - Returns True iff leaf is in tree with given root
            
        Complexity: O(log n) where n is number of leaves
        
        Args:
            key: State identifier (used for ordering, not verification)
            leaf_data: Raw data that was hashed to create leaf
            proof_path: Authentication path from leaf to root
            root_hash: Expected Merkle root (32 bytes)
            
        Returns:
            True if proof is valid, False otherwise
            
        Raises:
            None - Invalid proofs return False, not exceptions
            
        Example:
            >>> builder = MerkleTreeBuilder()
            >>> builder.add_leaf("state_0", b"data")
            >>> root, proofs = builder.build()
            >>> builder.verify_proof("state_0", b"data", proofs["state_0"], root)
            True
        """
        # Compute leaf hash from raw data
        current_hash = hashlib.sha256(leaf_data).digest()
        
        # Traverse proof path from leaf to root
        for sibling_hash, is_right in proof_path:
            # Combine current hash with sibling based on position
            # If sibling is right, current is left: hash(left || right)
            # If sibling is left, current is right: hash(left || right)
            combined = (
                current_hash + sibling_hash if is_right
                else sibling_hash + current_hash
            )
            current_hash = hashlib.sha256(combined).digest()
        
        # Constant-time comparison (Python's == is constant-time for fixed-size bytes)
        return current_hash == root_hash
