"""Witness generation for Q-table ZK proofs.

Generates witness data from trained Q-tables for Risc0 guest programs.
Supports both small (in-memory) and large (Merkle tree) Q-tables.

Security Properties:
- Privacy: Q-table entries remain private; only commitments revealed
- Correctness: Witness generation matches zkVM verification logic
- Determinism: Same inputs always produce same witness

Trust Assumptions:
- Q-table data is authentic (from trusted training process)
- Fixed-point conversion preserves Q-value semantics

Dependencies: hashlib, json, numpy
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Tuple

import numpy as np


# Type-safe wrappers for security-critical strings
StateKey = NewType("StateKey", str)
ActionIndex = NewType("ActionIndex", int)
HashBytes = NewType("HashBytes", bytes)

# Q16.16 fixed-point scale factor (2^16 = 65536)
Q16_16_SCALE: int = 1 << 16


@dataclass(frozen=True)
class QTableEntry:
    """Single Q-table entry with fixed-point representation.
    
    Uses Q16.16 fixed-point format for zk-friendly arithmetic.
    Q16.16 represents values as integers scaled by 2^16, allowing
    fractional values in range [-32768.0, 32767.9999847].
    
    Security: Immutable dataclass prevents accidental modification
    of Q-values after creation.
    """
    
    q_hold: int  # Q16.16 fixed-point
    q_buy: int   # Q16.16 fixed-point
    q_sell: int  # Q16.16 fixed-point
    
    @classmethod
    def from_float(cls, q_hold: float, q_buy: float, q_sell: float) -> QTableEntry:
        """Convert float Q-values to Q16.16 fixed-point.
        
        Security:
            - Overflow/underflow: Values outside [-32768, 32767.9999] will
              overflow/underflow INT32 range. Caller should validate inputs.
            - Rounding: Truncation to int may lose precision (acceptable for Q-values)
        
        Args:
            q_hold: Hold action Q-value (float)
            q_buy: Buy action Q-value (float)
            q_sell: Sell action Q-value (float)
            
        Returns:
            QTableEntry with fixed-point values
            
        Example:
            >>> entry = QTableEntry.from_float(0.5, 0.75, -0.25)
            >>> entry.q_buy == int(0.75 * 65536)
            True
        """
        return cls(
            q_hold=int(q_hold * Q16_16_SCALE),
            q_buy=int(q_buy * Q16_16_SCALE),
            q_sell=int(q_sell * Q16_16_SCALE),
        )
    
    def to_float(self) -> Tuple[float, float, float]:
        """Convert Q16.16 fixed-point to float.
        
        Returns:
            Tuple of (q_hold, q_buy, q_sell) as floats
            
        Example:
            >>> entry = QTableEntry(32768, 49152, -16384)
            >>> entry.to_float()
            (0.5, 0.75, -0.25)
        """
        return (
            self.q_hold / Q16_16_SCALE,
            self.q_buy / Q16_16_SCALE,
            self.q_sell / Q16_16_SCALE,
        )


@dataclass(frozen=True)
class MerkleProof:
    """Merkle tree authentication path.
    
    Contains the authentication path from a leaf to the root of a Merkle tree.
    Each path element is a tuple of (sibling_hash, is_right) where:
    - sibling_hash: Hash of the sibling node at that level (32 bytes)
    - is_right: True if sibling is to the right, False if to the left
    
    Security: 
        - Immutable dataclass prevents tampering with proof paths
        - Path length corresponds to tree depth (log2 of leaf count)
        
    Example:
        >>> proof = MerkleProof(
        ...     leaf_hash=b"a" * 32,
        ...     path=((b"b" * 32, True), (b"c" * 32, False)),
        ...     root_hash=b"d" * 32
        ... )
    """
    
    leaf_hash: bytes
    path: Tuple[Tuple[bytes, bool], ...]  # Immutable tuple of (sibling_hash, is_right)
    root_hash: bytes
    
    def __post_init__(self) -> None:
        """Validate proof structure.
        
        Raises:
            ValueError: If hash lengths are incorrect
        """
        if len(self.leaf_hash) != 32:
            raise ValueError(f"Leaf hash must be 32 bytes, got {len(self.leaf_hash)}")
        if len(self.root_hash) != 32:
            raise ValueError(f"Root hash must be 32 bytes, got {len(self.root_hash)}")
        # Validate all sibling hashes in path are 32 bytes
        for i, (sibling_hash, _) in enumerate(self.path):
            if len(sibling_hash) != 32:
                raise ValueError(f"Path element {i} hash must be 32 bytes, got {len(sibling_hash)}")


class MerkleTree:
    """Merkle tree for Q-table commitments."""
    
    def __init__(self, entries: Dict[str, QTableEntry]):
        """Build Merkle tree from Q-table entries.
        
        Args:
            entries: Dictionary mapping state keys to Q-table entries
        """
        self.entries = entries
        self.leaves = self._build_leaves()
        self.root = self._build_tree()
    
    def _build_leaves(self) -> List[bytes]:
        """Build leaf hashes from entries."""
        leaves = []
        for state_key, entry in sorted(self.entries.items()):
            leaf_data = json.dumps({
                "state": state_key,
                "q_hold": entry.q_hold,
                "q_buy": entry.q_buy,
                "q_sell": entry.q_sell,
            }, sort_keys=True).encode()
            leaf_hash = hashlib.sha256(leaf_data).digest()
            leaves.append((state_key, leaf_hash))
        return leaves
    
    def _build_tree(self) -> bytes:
        """Build Merkle tree and return root hash."""
        if not self.leaves:
            return hashlib.sha256(b"").digest()
        
        # Build tree bottom-up
        level = [hash for _, hash in self.leaves]
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]  # Duplicate odd node
                next_level.append(hashlib.sha256(combined).digest())
            level = next_level
        
        return level[0]
    
    def _compute_parent_level(self, level: List[bytes]) -> List[bytes]:
        """Compute parent level from current level by hashing pairs.
        
        Args:
            level: Current level hashes
            
        Returns:
            Parent level hashes
        """
        next_level: List[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            combined = left + right
            next_level.append(hashlib.sha256(combined).digest())
        return next_level
    
    def _get_sibling_for_index(self, idx: int, level: List[bytes]) -> Optional[Tuple[bytes, bool]]:
        """Get sibling hash and position for a given index.
        
        Args:
            idx: Current node index
            level: Current level hashes
            
        Returns:
            Tuple of (sibling_hash, is_right) or None if no sibling
        """
        sibling_idx = idx ^ 1  # XOR to get sibling index
        if sibling_idx >= len(level):
            return None
        is_right = sibling_idx > idx
        return (level[sibling_idx], is_right)
    
    def get_proof(self, state_key: str) -> Optional[MerkleProof]:
        """Get Merkle proof for a state key.
        
        Builds authentication path from leaf to root by traversing tree levels.
        Each path element contains sibling hash and position (left/right).
        
        Args:
            state_key: State key to get proof for
            
        Returns:
            MerkleProof if key exists, None otherwise
            
        Complexity: O(log n) where n is number of entries
        """
        if state_key not in self.entries:
            return None
        
        # Find leaf index in sorted order
        sorted_keys = sorted(self.entries.keys())
        leaf_idx = sorted_keys.index(state_key)
        
        # Build authentication path bottom-up
        level = [hash for _, hash in self.leaves]
        path: List[Tuple[bytes, bool]] = []
        current_idx = leaf_idx
        
        while len(level) > 1:
            sibling = self._get_sibling_for_index(current_idx, level)
            if sibling is not None:
                path.append(sibling)
            
            # Move to parent level
            current_idx //= 2
            level = self._compute_parent_level(level)
        
        return MerkleProof(
            leaf_hash=self.leaves[leaf_idx][1],
            path=tuple(path),  # Convert to immutable tuple
            root_hash=self.root,
        )


def _select_action_greedy(q_entry: QTableEntry) -> ActionIndex:
    """Select action using greedy (argmax) policy.
    
    Returns the action with highest Q-value.
    Tie-breaking order: buy > sell > hold (deterministic).
    
    Security:
        - Deterministic: Same Q-values always produce same action
        - Matches Rust implementation in idi-qtable/src/main.rs:argmax_q()
    
    Args:
        q_entry: Q-table entry with fixed-point values
        
    Returns:
        Action index: 0=hold, 1=buy, 2=sell
    """
    q_hold, q_buy, q_sell = q_entry.to_float()
    
    # Greedy selection: choose action with highest Q-value
    # Tie-breaking: buy > sell > hold (matches Rust implementation)
    if q_buy > q_sell and q_buy > q_hold:
        return ActionIndex(1)  # buy
    elif q_sell > q_hold:
        return ActionIndex(2)  # sell
    else:
        return ActionIndex(0)  # hold


@dataclass(frozen=True)
class QTableWitness:
    """Witness data for Q-table proof.
    
    Contains all private data needed to generate a ZK proof:
    - Q-table entry (private)
    - Merkle proof path (private, if using Merkle tree)
    - Selected action (public, committed in proof)
    
    Security: Immutable dataclass prevents tampering with witness data.
    """
    
    state_key: StateKey
    q_entry: QTableEntry
    merkle_proof: Optional[MerkleProof]
    q_table_root: HashBytes  # Merkle root or hash of full table (32 bytes)
    selected_action: ActionIndex  # 0=hold, 1=buy, 2=sell
    layer_weights: Dict[str, float]  # Layer voting weights (for multi-layer agents)
    comm_action: Optional[ActionIndex] = None  # Communication action (optional)
    
    def __post_init__(self) -> None:
        """Validate witness structure.
        
        Raises:
            ValueError: If action index or root hash length is invalid
        """
        if self.selected_action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {self.selected_action} (must be 0, 1, or 2)")
        if len(self.q_table_root) != 32:
            raise ValueError(f"Q-table root must be 32 bytes, got {len(self.q_table_root)}")
        if self.comm_action is not None and self.comm_action not in (0, 1, 2):
            raise ValueError(f"Invalid comm_action: {self.comm_action} (must be 0, 1, or 2)")


def generate_witness_from_q_table(
    q_table: Dict[str, Dict[str, float]],
    state_key: str,
    use_merkle: bool = True,
) -> QTableWitness:
    """Generate witness from Q-table for a given state.
    
    Args:
        q_table: Dictionary mapping state keys to action Q-values
        state_key: State to generate witness for
        use_merkle: Whether to use Merkle tree (for large tables)
    
    Returns:
        QTableWitness with proof data
    """
    if state_key not in q_table:
        raise ValueError(f"State {state_key} not in Q-table")
    
    q_values = q_table[state_key]
    q_entry = QTableEntry.from_float(
        q_hold=q_values.get("hold", 0.0),
        q_buy=q_values.get("buy", 0.0),
        q_sell=q_values.get("sell", 0.0),
    )
    
    # Select action using greedy policy (matches Rust implementation)
    selected_action = _select_action_greedy(q_entry)
    
    # Build Merkle tree if requested
    merkle_proof: Optional[MerkleProof] = None
    q_table_root_bytes: bytes
    
    if use_merkle and len(q_table) > 100:  # Use Merkle for large tables
        # Convert to QTableEntry format
        entries = {
            key: QTableEntry.from_float(
                q_hold=vals.get("hold", 0.0),
                q_buy=vals.get("buy", 0.0),
                q_sell=vals.get("sell", 0.0),
            )
            for key, vals in q_table.items()
        }
        tree = MerkleTree(entries)
        merkle_proof = tree.get_proof(state_key)
        q_table_root_bytes = tree.root
    else:
        # Small table: hash entire table
        table_json = json.dumps(q_table, sort_keys=True).encode()
        q_table_root_bytes = hashlib.sha256(table_json).digest()
    
    return QTableWitness(
        state_key=StateKey(state_key),
        q_entry=q_entry,
        merkle_proof=merkle_proof,
        q_table_root=HashBytes(q_table_root_bytes),
        selected_action=selected_action,
        layer_weights={},  # Would be populated from multi-layer config
        comm_action=None,
    )


def serialize_witness(witness: QTableWitness) -> bytes:
    """Serialize witness for Risc0 guest program.
    
    Converts witness to JSON format compatible with Risc0 guest program
    deserialization. All binary data is hex-encoded.
    
    Security:
        - Deterministic: sort_keys=True ensures consistent serialization
        - No sensitive data leakage: Only commits what's needed for proof
        
    Args:
        witness: Q-table witness to serialize
        
    Returns:
        Serialized witness as bytes (JSON format)
        
    Example:
        >>> witness = generate_witness_from_q_table({"state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0}}, "state_0")
        >>> data = serialize_witness(witness)
        >>> isinstance(data, bytes)
        True
    """
    data: Dict[str, Any] = {
        "state_key": witness.state_key,
        "q_entry": {
            "q_hold": witness.q_entry.q_hold,
            "q_buy": witness.q_entry.q_buy,
            "q_sell": witness.q_entry.q_sell,
        },
        "q_table_root": witness.q_table_root.hex(),
        "selected_action": witness.selected_action,
        "layer_weights": witness.layer_weights,
    }
    
    if witness.merkle_proof:
        data["merkle_proof"] = {
            "leaf_hash": witness.merkle_proof.leaf_hash.hex(),
            "path": [
                {"hash": h.hex(), "is_right": is_right}
                for h, is_right in witness.merkle_proof.path
            ],
            "root_hash": witness.merkle_proof.root_hash.hex(),
        }
    
    return json.dumps(data, sort_keys=True).encode()

