"""Tests for Q-table proof generation."""

import json
from pathlib import Path

import pytest

from idi.zk.witness_generator import (
    QTableEntry,
    QTableWitness,
    generate_witness_from_q_table,
    serialize_witness,
)
from idi.zk.qtable_prover import verify_qtable_proof


def test_qtable_witness_generation():
    """Test Q-table witness generation."""
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
        "state_1": {"hold": 0.0, "buy": 0.0, "sell": 0.5},
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    
    assert witness.state_key == "state_0"
    assert witness.selected_action == 1  # buy has highest Q-value
    assert len(witness.q_table_root) == 32  # SHA256 hash


def test_qtable_witness_merkle():
    """Test Q-table witness with Merkle tree."""
    q_table = {
        f"state_{i}": {"hold": 0.0, "buy": 0.5 if i % 2 == 0 else 0.0, "sell": 0.0}
        for i in range(200)
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=True)
    
    assert witness.merkle_proof is not None
    assert len(witness.q_table_root) == 32


def test_qtable_proof_verification():
    """Test Q-table proof verification."""
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    
    # Create mock receipt
    receipt_path = Path("/tmp/test_receipt.json")
    receipt = {
        "q_table_root": witness.q_table_root.hex(),
        "selected_action": witness.selected_action,
    }
    receipt_path.write_text(json.dumps(receipt))
    
    # Verify proof
    assert verify_qtable_proof(
        proof_path=Path("/tmp/test_proof.bin"),
        receipt_path=receipt_path,
        expected_q_root=witness.q_table_root,
    )
    
    # Cleanup
    receipt_path.unlink(missing_ok=True)


def test_witness_serialization_roundtrip():
    """Test witness serialization and deserialization."""
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
    }
    
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    serialized = serialize_witness(witness)
    
    # Deserialize
    data = json.loads(serialized)
    
    assert data["state_key"] == "state_0"
    assert data["selected_action"] == 1
    assert "q_table_root" in data
    assert "q_entry" in data

