"""Merkle tree implementation for Q-table commitments.

Provides efficient Merkle tree construction and proof generation for large Q-tables.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple


class MerkleTreeBuilder:
    """Builds Merkle trees from Q-table data."""
    
    def __init__(self):
        """Initialize Merkle tree builder."""
        self.leaves: List[Tuple[str, bytes]] = []  # (key, hash)
    
    def add_leaf(self, key: str, data: bytes) -> None:
        """Add a leaf node to the tree.
        
        Args:
            key: State key or identifier
            data: Serialized Q-table entry data
        """
        leaf_hash = hashlib.sha256(data).digest()
        self.leaves.append((key, leaf_hash))
    
    def build(self) -> Tuple[bytes, Dict[str, List[Tuple[bytes, bool]]]]:
        """Build Merkle tree and return root hash and proofs.
        
        Returns:
            Tuple of (root_hash, proofs_dict) where proofs_dict maps
            state keys to authentication paths (list of (sibling_hash, is_right))
        """
        if not self.leaves:
            return hashlib.sha256(b"").digest(), {}
        
        # Sort leaves by key for deterministic ordering
        sorted_leaves = sorted(self.leaves, key=lambda x: x[0])
        leaf_hashes = [hash for _, hash in sorted_leaves]
        keys = [key for key, _ in sorted_leaves]
        
        # Initialize proofs for all keys
        proofs: Dict[str, List[Tuple[bytes, bool]]] = {key: [] for key in keys}
        
        # Build tree bottom-up
        level = leaf_hashes.copy()
        level_keys = keys.copy()  # Track which keys are at each level
        
        while len(level) > 1:
            next_level = []
            next_keys = []
            
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                    sibling_hash = level[i + 1]
                    is_right = True
                    # Both children contribute to parent
                    parent_key = level_keys[i]  # Use left child's key
                else:
                    combined = level[i] + level[i]  # Duplicate odd node
                    sibling_hash = level[i]
                    is_right = False
                    parent_key = level_keys[i]
                
                parent_hash = hashlib.sha256(combined).digest()
                next_level.append(parent_hash)
                next_keys.append(parent_key)
                
                # Store proof for left child
                if i < len(level_keys):
                    key = level_keys[i]
                    proofs[key].append((sibling_hash, is_right))
                
                # Store proof for right child (if exists)
                if i + 1 < len(level_keys):
                    key = level_keys[i + 1]
                    proofs[key].append((level[i], False))  # Left sibling
            
            level = next_level
            level_keys = next_keys
        
        root_hash = level[0] if level else hashlib.sha256(b"").digest()
        
        return root_hash, proofs
    
    def verify_proof(
        self,
        key: str,
        leaf_data: bytes,
        proof_path: List[Tuple[bytes, bool]],
        root_hash: bytes,
    ) -> bool:
        """Verify a Merkle proof.
        
        Args:
            key: State key
            leaf_data: Original leaf data
            proof_path: Authentication path (list of (sibling_hash, is_right))
            root_hash: Expected root hash
        
        Returns:
            True if proof is valid
        """
        current_hash = hashlib.sha256(leaf_data).digest()
        
        for sibling_hash, is_right in proof_path:
            if is_right:
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = hashlib.sha256(combined).digest()
        
        return current_hash == root_hash

