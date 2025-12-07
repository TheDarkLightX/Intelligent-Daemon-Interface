"""Witness generation for Q-table ZK proofs.

Generates witness data from trained Q-tables for Risc0 guest programs.
Supports both small (in-memory) and large (Merkle tree) Q-tables.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QTableEntry:
    """Single Q-table entry with fixed-point representation."""
    
    q_hold: int  # Q16.16 fixed-point
    q_buy: int   # Q16.16 fixed-point
    q_sell: int  # Q16.16 fixed-point
    
    @classmethod
    def from_float(cls, q_hold: float, q_buy: float, q_sell: float) -> QTableEntry:
        """Convert float Q-values to Q16.16 fixed-point."""
        SCALE = 1 << 16  # Q16.16 scale factor
        return cls(
            q_hold=int(q_hold * SCALE),
            q_buy=int(q_buy * SCALE),
            q_sell=int(q_sell * SCALE),
        )
    
    def to_float(self) -> Tuple[float, float, float]:
        """Convert Q16.16 fixed-point to float."""
        SCALE = 1 << 16
        return (
            self.q_hold / SCALE,
            self.q_buy / SCALE,
            self.q_sell / SCALE,
        )


@dataclass
class MerkleProof:
    """Merkle tree authentication path."""
    
    leaf_hash: bytes
    path: List[Tuple[bytes, bool]]  # (sibling_hash, is_right)
    root_hash: bytes


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
    
    def get_proof(self, state_key: str) -> Optional[MerkleProof]:
        """Get Merkle proof for a state key."""
        if state_key not in self.entries:
            return None
        
        # Find leaf index
        sorted_keys = sorted(self.entries.keys())
        leaf_idx = sorted_keys.index(state_key)
        
        # Build authentication path
        level = [hash for _, hash in self.leaves]
        path = []
        current_idx = leaf_idx
        
        while len(level) > 1:
            sibling_idx = current_idx ^ 1  # XOR to get sibling
            if sibling_idx < len(level):
                is_right = sibling_idx > current_idx
                path.append((level[sibling_idx], is_right))
            
            current_idx //= 2
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]
                next_level.append(hashlib.sha256(combined).digest())
            level = next_level
        
        return MerkleProof(
            leaf_hash=self.leaves[leaf_idx][1],
            path=path,
            root_hash=self.root,
        )


@dataclass
class QTableWitness:
    """Witness data for Q-table proof."""
    
    state_key: str
    q_entry: QTableEntry
    merkle_proof: Optional[MerkleProof]
    q_table_root: bytes  # Merkle root or hash of full table
    
    # Action selection data
    selected_action: int  # 0=hold, 1=buy, 2=sell
    layer_weights: Dict[str, float]  # Layer voting weights
    
    # Communication data
    comm_action: Optional[int] = None


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
    
    # Select action (greedy)
    _, q_buy, q_sell = q_entry.to_float()
    if q_buy > q_sell and q_buy > 0.0:
        selected_action = 1
    elif q_sell > 0.0:
        selected_action = 2
    else:
        selected_action = 0
    
    # Build Merkle tree if requested
    merkle_proof = None
    q_table_root = b""
    
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
        q_table_root = tree.root
    else:
        # Small table: hash entire table
        table_json = json.dumps(q_table, sort_keys=True).encode()
        q_table_root = hashlib.sha256(table_json).digest()
    
    return QTableWitness(
        state_key=state_key,
        q_entry=q_entry,
        merkle_proof=merkle_proof,
        q_table_root=q_table_root,
        selected_action=selected_action,
        layer_weights={},  # Would be populated from multi-layer config
        comm_action=None,
    )


def serialize_witness(witness: QTableWitness) -> bytes:
    """Serialize witness for Risc0 guest program."""
    data = {
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

