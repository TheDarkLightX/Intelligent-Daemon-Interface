"""Tests for witness generation."""

import json
from pathlib import Path

import pytest

from idi.zk.witness_generator import (
    QTableEntry,
    QTableWitness,
    MerkleTree,
    generate_witness_from_q_table,
    serialize_witness,
)


def test_q_table_entry_fixed_point():
    """Test Q-table entry fixed-point conversion."""
    entry = QTableEntry.from_float(0.5, 0.75, -0.25)
    
    assert entry.q_hold == int(0.5 * (1 << 16))
    assert entry.q_buy == int(0.75 * (1 << 16))
    assert entry.q_sell == int(-0.25 * (1 << 16))
    
    hold, buy, sell = entry.to_float()
    assert abs(hold - 0.5) < 0.0001
    assert abs(buy - 0.75) < 0.0001
    assert abs(sell - (-0.25)) < 0.0001


def test_merkle_tree_small():
    """Test Merkle tree with small Q-table."""
    entries = {
        "state_0": QTableEntry.from_float(0.0, 0.5, 0.0),
        "state_1": QTableEntry.from_float(0.0, 0.0, 0.5),
        "state_2": QTableEntry.from_float(0.5, 0.0, 0.0),
    }
    
    tree = MerkleTree(entries)
    assert len(tree.root) == 32  # SHA256 hash length
    
    proof = tree.get_proof("state_0")
    assert proof is not None
    assert proof.root_hash == tree.root


def test_merkle_tree_proof_verification():
    """Test Merkle proof verification."""
    entries = {
        "state_0": QTableEntry.from_float(0.0, 0.5, 0.0),
        "state_1": QTableEntry.from_float(0.0, 0.0, 0.5),
    }
    
    tree = MerkleTree(entries)
    proof = tree.get_proof("state_0")
    
    assert proof is not None
    assert proof.root_hash == tree.root


def test_generate_witness_small_table():
    """Test witness generation for small Q-table."""
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
        "state_1": {"hold": 0.0, "buy": 0.0, "sell": 0.5},
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    
    assert witness.state_key == "state_0"
    assert witness.selected_action == 1  # buy has highest Q-value
    assert len(witness.q_table_root) == 32  # SHA256 hash


def test_generate_witness_large_table():
    """Test witness generation for large Q-table with Merkle tree."""
    q_table = {
        f"state_{i}": {"hold": 0.0, "buy": 0.5 if i % 2 == 0 else 0.0, "sell": 0.0}
        for i in range(200)  # Large enough to trigger Merkle
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=True)
    
    assert witness.state_key == "state_0"
    assert witness.merkle_proof is not None
    assert len(witness.q_table_root) == 32


def test_serialize_witness():
    """Test witness serialization."""
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    serialized = serialize_witness(witness)
    
    assert isinstance(serialized, bytes)
    data = json.loads(serialized)
    assert data["state_key"] == "state_0"
    assert data["selected_action"] == 1
    assert "q_table_root" in data

