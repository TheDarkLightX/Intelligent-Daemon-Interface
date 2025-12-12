"""Tests for Merkle tree implementation."""

import hashlib

import pytest

from idi.zk.merkle_tree import MerkleTreeBuilder


def test_merkle_tree_empty():
    """Test Merkle tree with no leaves."""
    builder = MerkleTreeBuilder()
    root, proofs = builder.build()
    
    assert len(root) == 32  # SHA256 hash
    assert len(proofs) == 0


def test_merkle_tree_single_leaf():
    """Test Merkle tree with single leaf."""
    builder = MerkleTreeBuilder()
    builder.add_leaf("state_0", b"test_data")
    root, proofs = builder.build()
    
    assert len(root) == 32
    assert "state_0" in proofs
    assert len(proofs["state_0"]) == 0  # No siblings for single leaf


def test_merkle_tree_multiple_leaves():
    """Test Merkle tree with multiple leaves."""
    builder = MerkleTreeBuilder()
    builder.add_leaf("state_0", b"data_0")
    builder.add_leaf("state_1", b"data_1")
    builder.add_leaf("state_2", b"data_2")
    
    root, proofs = builder.build()
    
    assert len(root) == 32
    assert len(proofs) == 3
    assert all(len(path) > 0 for path in proofs.values())


def test_merkle_proof_verification():
    """Test Merkle proof verification."""
    builder = MerkleTreeBuilder()
    builder.add_leaf("state_0", b"data_0")
    builder.add_leaf("state_1", b"data_1")
    
    root, proofs = builder.build()
    
    # Verify proof for state_0
    proof_path = proofs["state_0"]
    assert builder.verify_proof("state_0", b"data_0", proof_path, root)
    
    # Verify proof for state_1
    proof_path = proof_path = proofs["state_1"]
    assert builder.verify_proof("state_1", b"data_1", proof_path, root)
    
    # Invalid proof should fail
    assert not builder.verify_proof("state_0", b"wrong_data", proof_path, root)


def test_merkle_tree_deterministic():
    """Test that Merkle tree is deterministic."""
    builder1 = MerkleTreeBuilder()
    builder1.add_leaf("state_0", b"data_0")
    builder1.add_leaf("state_1", b"data_1")
    root1, _ = builder1.build()
    
    builder2 = MerkleTreeBuilder()
    builder2.add_leaf("state_0", b"data_0")
    builder2.add_leaf("state_1", b"data_1")
    root2, _ = builder2.build()
    
    assert root1 == root2


def test_merkle_tree_order_independent():
    """Test that Merkle tree root is independent of insertion order."""
    builder1 = MerkleTreeBuilder()
    builder1.add_leaf("state_0", b"data_0")
    builder1.add_leaf("state_1", b"data_1")
    root1, _ = builder1.build()
    
    builder2 = MerkleTreeBuilder()
    builder2.add_leaf("state_1", b"data_1")
    builder2.add_leaf("state_0", b"data_0")
    root2, _ = builder2.build()
    
    # Should be same because we sort by key
    assert root1 == root2

