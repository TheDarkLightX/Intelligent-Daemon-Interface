"""Property-based tests for Merkle tree using Hypothesis.

Tests Merkle tree properties that must hold for all inputs:
- Proof roundtrip (generate → verify)
- Tamper detection
- Determinism
- Consistency
"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, strategies as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed; skipping Merkle property tests", allow_module_level=True)

from idi.zk.merkle_tree import MerkleTreeBuilder


@given(
    entries=st.dictionaries(
        keys=st.text(min_size=1, max_size=32, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        values=st.binary(min_size=1, max_size=128),
        min_size=1,
        max_size=1000,
    )
)
def test_merkle_proof_roundtrip(entries: dict[str, bytes]) -> None:
    """Every leaf has valid proof that verifies.
    
    Property: For any set of entries, all generated proofs verify successfully.
    """
    builder = MerkleTreeBuilder()
    
    # Add all entries
    for key, data in entries.items():
        builder.add_leaf(key, data)
    
    # Build tree and get proofs
    root_hash, proofs = builder.build()
    
    # Verify all proofs
    for key, data in entries.items():
        proof_path = proofs[key]
        assert builder.verify_proof(key, data, proof_path, root_hash), (
            f"Proof verification failed for key: {key}"
        )


@given(
    entries=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.binary(min_size=1, max_size=128),
        min_size=2,
        max_size=100,
    ),
    tamper_key=st.text(min_size=1, max_size=32),
    tamper_byte_idx=st.integers(min_value=0, max_value=127),
    tamper_byte_value=st.integers(min_value=0, max_value=255),
)
def test_merkle_tamper_detection(
    entries: dict[str, bytes],
    tamper_key: str,
    tamper_byte_idx: int,
    tamper_byte_value: int,
) -> None:
    """Tampering any byte in proof causes verification failure.
    
    Property: Any modification to proof path or root hash is detected.
    """
    builder = MerkleTreeBuilder()
    
    # Add entries
    for key, data in entries.items():
        builder.add_leaf(key, data)
    
    root_hash, proofs = builder.build()
    
    # Only test if tamper_key exists in entries
    if tamper_key not in entries:
        return
    
    original_data = entries[tamper_key]
    proof_path = proofs[tamper_key]
    
    # Test 1: Tamper leaf data
    if tamper_byte_idx < len(original_data):
        tampered_data = bytearray(original_data)
        original_byte = tampered_data[tamper_byte_idx]
        next_byte = tamper_byte_value
        if next_byte == original_byte:
            next_byte = (original_byte + 1) % 256
        tampered_data[tamper_byte_idx] = next_byte
        assert not builder.verify_proof(
            tamper_key, bytes(tampered_data), proof_path, root_hash
        ), "Tampered leaf data should fail verification"
    
    # Test 2: Tamper proof path
    if proof_path:
        tampered_path = list(proof_path)
        # Tamper first sibling hash
        tampered_sibling = bytearray(tampered_path[0][0])
        if tampered_sibling:
            tampered_sibling[0] = (tampered_sibling[0] + 1) % 256
            tampered_path[0] = (bytes(tampered_sibling), tampered_path[0][1])
            assert not builder.verify_proof(
                tamper_key, original_data, tampered_path, root_hash
            ), "Tampered proof path should fail verification"
    
    # Test 3: Tamper root hash
    tampered_root = bytearray(root_hash)
    tampered_root[0] = (tampered_root[0] + 1) % 256
    assert not builder.verify_proof(
        tamper_key, original_data, proof_path, bytes(tampered_root)
    ), "Tampered root hash should fail verification"


@given(
    entries1=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.binary(min_size=1, max_size=128),
        min_size=1,
        max_size=100,
    ),
    entries2=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.binary(min_size=1, max_size=128),
        min_size=1,
        max_size=100,
    ),
)
def test_merkle_determinism(entries1: dict[str, bytes], entries2: dict[str, bytes]) -> None:
    """Same entries always produce same root hash.
    
    Property: Merkle tree construction is deterministic.
    """
    builder1 = MerkleTreeBuilder()
    builder2 = MerkleTreeBuilder()
    
    # Build tree twice with same entries (different order)
    for key in sorted(entries1.keys()):
        builder1.add_leaf(key, entries1[key])
        builder2.add_leaf(key, entries1[key])
    
    root1, _ = builder1.build()
    root2, _ = builder2.build()
    
    assert root1 == root2, "Same entries should produce same root hash"
    
    # Test with different order (should still be same due to sorting)
    builder3 = MerkleTreeBuilder()
    for key in reversed(sorted(entries1.keys())):
        builder3.add_leaf(key, entries1[key])
    root3, _ = builder3.build()
    
    assert root1 == root3, "Root hash should be independent of insertion order"


@given(
    entries=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.binary(min_size=1, max_size=128),
        min_size=1,
        max_size=100,
    )
)
def test_merkle_proof_validity(entries: dict[str, bytes]) -> None:
    """All generated proofs verify successfully.
    
    Property: Every leaf has a valid proof that verifies against the root.
    """
    builder = MerkleTreeBuilder()
    
    for key, data in entries.items():
        builder.add_leaf(key, data)
    
    root_hash, proofs = builder.build()
    
    # Verify each proof is valid
    for key, data in entries.items():
        proof_path = proofs[key]
        assert builder.verify_proof(key, data, proof_path, root_hash), (
            f"Proof verification failed for key: {key}"
        )
    
    # Verify all entries have proofs
    assert len(proofs) == len(entries), (
        f"Expected {len(entries)} proofs, got {len(proofs)}"
    )


def test_merkle_empty_tree() -> None:
    """Empty tree produces deterministic root hash."""
    builder = MerkleTreeBuilder()
    root1, proofs1 = builder.build()
    root2, proofs2 = builder.build()
    
    assert root1 == root2, "Empty trees should produce same root"
    assert len(proofs1) == 0, "Empty tree should have no proofs"
    assert len(root1) == 32, "Root hash should be 32 bytes"


@given(
    entries=st.dictionaries(
        keys=st.text(min_size=1, max_size=32),
        values=st.binary(min_size=1, max_size=128),
        min_size=1,
        max_size=100,
    ),
    wrong_data=st.binary(min_size=1, max_size=128),
)
def test_merkle_wrong_data_rejection(entries: dict[str, bytes], wrong_data: bytes) -> None:
    """Wrong leaf data fails verification even with correct proof path.
    
    Property: Proof binds to specific leaf data, not just key.
    """
    builder = MerkleTreeBuilder()
    
    for key, data in entries.items():
        builder.add_leaf(key, data)
    
    root_hash, proofs = builder.build()
    
    # Try to verify wrong data with correct proof
    for key in entries.keys():
        proof_path = proofs[key]
        # Use wrong data
        if wrong_data != entries[key]:
            assert not builder.verify_proof(
                key, wrong_data, proof_path, root_hash
            ), f"Wrong data should fail verification for key: {key}"

