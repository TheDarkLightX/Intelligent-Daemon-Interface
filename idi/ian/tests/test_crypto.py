"""
Tests for cryptographic primitives (BLS12-381 and Sparse Merkle Tree).

Covers:
- BLS key generation, signing, verification
- Input validation (key/signature lengths, ranges)
- Proof-of-possession (PoP) enforcement
- Domain separation
- Aggregation with duplicate key rejection
- Aggregate identity rejection
- Bounded PoP registry with LRU eviction
- SMT membership and non-membership proofs
- SMT size limits and validation
"""

from __future__ import annotations

import secrets
from typing import List, Tuple

import pytest

from idi.ian.algorithms.crypto import (
    BLSOperations,
    BLSError,
    SparseMerkleTree,
    SMTProof,
    SMTError,
    BLS_PUBKEY_LEN,
    BLS_SIGNATURE_LEN,
    BLS_PRIVKEY_LEN,
    SMT_HASH_LEN,
    MAX_CONTEXT_LEN,
    MAX_MESSAGE_LEN,
    MAX_SMT_VALUE_LEN,
    MAX_SMT_ENTRIES,
    MAX_POP_REGISTRY_SIZE,
)


# =============================================================================
# BLS Key Generation Tests
# =============================================================================


class TestBLSKeyGeneration:
    """Tests for BLS keypair generation."""

    def test_generate_keypair_returns_correct_lengths(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        
        assert len(sk) == BLS_PRIVKEY_LEN
        assert len(pk) == BLS_PUBKEY_LEN

    def test_generate_keypair_produces_unique_keys(self) -> None:
        bls = BLSOperations()
        keys = [bls.generate_keypair() for _ in range(5)]
        
        private_keys = [k[0] for k in keys]
        public_keys = [k[1] for k in keys]
        
        assert len(set(private_keys)) == 5
        assert len(set(public_keys)) == 5


# =============================================================================
# BLS Sign/Verify Tests
# =============================================================================


class TestBLSSignVerify:
    """Tests for BLS signing and verification."""

    def test_sign_verify_roundtrip(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        msg = b"test message"
        
        sig = bls.sign(sk, msg)
        assert bls.verify(pk, msg, sig)

    def test_verify_wrong_message_fails(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        
        sig = bls.sign(sk, b"message A")
        assert not bls.verify(pk, b"message B", sig)

    def test_verify_wrong_key_fails(self) -> None:
        bls = BLSOperations()
        sk1, pk1 = bls.generate_keypair()
        _, pk2 = bls.generate_keypair()
        
        sig = bls.sign(sk1, b"test")
        assert not bls.verify(pk2, b"test", sig)

    def test_domain_separation_prevents_cross_domain(self) -> None:
        bls1 = BLSOperations(domain=b"DOMAIN_A")
        bls2 = BLSOperations(domain=b"DOMAIN_B")
        
        sk, pk = bls1.generate_keypair()
        msg = b"test"
        sig = bls1.sign(sk, msg)
        
        # Same key/msg/sig but different domain should fail
        assert bls1.verify(pk, msg, sig)
        assert not bls2.verify(pk, msg, sig)

    def test_context_separation(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        msg = b"test"
        
        sig = bls.sign(sk, msg, context=b"CTX_A")
        
        assert bls.verify(pk, msg, sig, context=b"CTX_A")
        assert not bls.verify(pk, msg, sig, context=b"CTX_B")


# =============================================================================
# BLS Input Validation Tests
# =============================================================================


class TestBLSInputValidation:
    """Tests for BLS input validation."""

    def test_invalid_private_key_length_rejected(self) -> None:
        bls = BLSOperations()
        
        with pytest.raises(BLSError, match="private key must be"):
            bls.sign(b"short", b"msg")
        
        with pytest.raises(BLSError, match="private key must be"):
            bls.sign(b"x" * 64, b"msg")

    def test_zero_private_key_rejected(self) -> None:
        bls = BLSOperations()
        
        with pytest.raises(BLSError, match="out of range"):
            bls.sign(b"\x00" * 32, b"msg")

    def test_invalid_public_key_length_rejected(self) -> None:
        bls = BLSOperations()
        sk, _ = bls.generate_keypair()
        sig = bls.sign(sk, b"test")
        
        assert not bls.verify(b"short", b"test", sig)
        assert not bls.verify(b"x" * 96, b"test", sig)

    def test_invalid_signature_length_rejected(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        
        assert not bls.verify(pk, b"test", b"short")
        assert not bls.verify(pk, b"test", b"x" * 48)

    def test_message_size_limit(self) -> None:
        bls = BLSOperations()
        sk, _ = bls.generate_keypair()
        
        # Should work at limit
        msg = b"x" * MAX_MESSAGE_LEN
        sig = bls.sign(sk, msg)
        assert len(sig) == BLS_SIGNATURE_LEN
        
        # Should fail over limit
        with pytest.raises(BLSError, match="exceeds"):
            bls.sign(sk, b"x" * (MAX_MESSAGE_LEN + 1))

    def test_context_size_limit(self) -> None:
        bls = BLSOperations()
        sk, _ = bls.generate_keypair()
        
        # Should work at limit
        ctx = b"x" * MAX_CONTEXT_LEN
        sig = bls.sign(sk, b"msg", context=ctx)
        assert len(sig) == BLS_SIGNATURE_LEN
        
        # Should fail over limit
        with pytest.raises(BLSError, match="exceeds"):
            bls.sign(sk, b"msg", context=b"x" * (MAX_CONTEXT_LEN + 1))


# =============================================================================
# BLS Proof-of-Possession Tests
# =============================================================================


class TestBLSProofOfPossession:
    """Tests for BLS proof-of-possession."""

    def test_pop_create_verify_roundtrip(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        
        pop = bls.create_pop(sk)
        assert len(pop) == BLS_SIGNATURE_LEN
        assert bls.verify_pop(pk, pop)
        assert bls.has_verified_pop(pk)

    def test_pop_wrong_key_fails(self) -> None:
        bls = BLSOperations()
        sk1, pk1 = bls.generate_keypair()
        _, pk2 = bls.generate_keypair()
        
        pop = bls.create_pop(sk1)
        assert not bls.verify_pop(pk2, pop)
        assert not bls.has_verified_pop(pk2)

    def test_pop_required_for_aggregation(self) -> None:
        bls = BLSOperations()
        sk1, pk1 = bls.generate_keypair()
        sk2, pk2 = bls.generate_keypair()
        
        msg = b"test"
        sig1 = bls.sign(sk1, msg)
        sig2 = bls.sign(sk2, msg)
        agg_sig = bls.aggregate_signatures([sig1, sig2])
        
        # Without PoP, verification should fail
        assert not bls.verify_aggregated([pk1, pk2], msg, agg_sig, require_pop=True)
        
        # Register PoPs
        bls.verify_pop(pk1, bls.create_pop(sk1))
        bls.verify_pop(pk2, bls.create_pop(sk2))
        
        # Now should pass
        assert bls.verify_aggregated([pk1, pk2], msg, agg_sig, require_pop=True)

    def test_pop_bypass_warning(self) -> None:
        """Test that require_pop=False bypasses PoP check (unsafe but allowed)."""
        bls = BLSOperations()
        sk1, pk1 = bls.generate_keypair()
        sk2, pk2 = bls.generate_keypair()
        
        msg = b"test"
        sig1 = bls.sign(sk1, msg)
        sig2 = bls.sign(sk2, msg)
        agg_sig = bls.aggregate_signatures([sig1, sig2])
        
        # Without PoP but with require_pop=False
        assert bls.verify_aggregated([pk1, pk2], msg, agg_sig, require_pop=False)


# =============================================================================
# BLS Bounded PoP Registry Tests
# =============================================================================


class TestBLSBoundedPoPRegistry:
    """Tests for bounded PoP registry with LRU eviction."""

    def test_registry_bounded_by_max_entries(self) -> None:
        bls = BLSOperations(max_pop_entries=3)
        
        keys: List[bytes] = []
        for _ in range(5):
            sk, pk = bls.generate_keypair()
            pop = bls.create_pop(sk)
            bls.verify_pop(pk, pop)
            keys.append(pk)
        
        # Registry should be bounded at 3
        assert len(bls._verified_pops) == 3
        
        # First two should be evicted
        assert not bls.has_verified_pop(keys[0])
        assert not bls.has_verified_pop(keys[1])
        
        # Last three should be present
        assert bls.has_verified_pop(keys[2])
        assert bls.has_verified_pop(keys[3])
        assert bls.has_verified_pop(keys[4])

    def test_lru_access_updates_order(self) -> None:
        bls = BLSOperations(max_pop_entries=3)
        
        keys: List[Tuple[bytes, bytes]] = []
        for _ in range(3):
            sk, pk = bls.generate_keypair()
            pop = bls.create_pop(sk)
            bls.verify_pop(pk, pop)
            keys.append((sk, pk))
        
        # Access first key (makes it most recently used)
        bls.has_verified_pop(keys[0][1])
        
        # Add a new key (should evict second key, not first)
        sk4, pk4 = bls.generate_keypair()
        bls.verify_pop(pk4, bls.create_pop(sk4))
        
        # First should still be present (was accessed)
        assert bls.has_verified_pop(keys[0][1])
        # Second should be evicted (oldest after access)
        assert not bls.has_verified_pop(keys[1][1])
        # Third and fourth should be present
        assert bls.has_verified_pop(keys[2][1])
        assert bls.has_verified_pop(pk4)

    def test_clear_registry(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        bls.verify_pop(pk, bls.create_pop(sk))
        
        assert bls.has_verified_pop(pk)
        
        bls.clear_pop_registry()
        
        assert not bls.has_verified_pop(pk)


# =============================================================================
# BLS Aggregation Tests
# =============================================================================


class TestBLSAggregation:
    """Tests for BLS signature and key aggregation."""

    def test_aggregate_signatures_verify(self) -> None:
        bls = BLSOperations()
        
        keys = [bls.generate_keypair() for _ in range(3)]
        for sk, pk in keys:
            bls.verify_pop(pk, bls.create_pop(sk))
        
        msg = b"common message"
        sigs = [bls.sign(sk, msg) for sk, _ in keys]
        pks = [pk for _, pk in keys]
        
        agg_sig = bls.aggregate_signatures(sigs)
        assert len(agg_sig) == BLS_SIGNATURE_LEN
        assert bls.verify_aggregated(pks, msg, agg_sig)

    def test_duplicate_public_keys_rejected(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        bls.verify_pop(pk, bls.create_pop(sk))
        
        msg = b"test"
        sig = bls.sign(sk, msg)
        agg_sig = bls.aggregate_signatures([sig, sig])
        
        # Duplicate keys should fail
        assert not bls.verify_aggregated([pk, pk], msg, agg_sig)

    def test_aggregate_public_keys_rejects_duplicates(self) -> None:
        bls = BLSOperations()
        sk, pk = bls.generate_keypair()
        bls.verify_pop(pk, bls.create_pop(sk))
        
        with pytest.raises(BLSError, match="duplicate"):
            bls.aggregate_public_keys([pk, pk])

    def test_aggregate_empty_list_rejected(self) -> None:
        bls = BLSOperations()
        
        with pytest.raises(BLSError, match="empty"):
            bls.aggregate_signatures([])
        
        with pytest.raises(BLSError, match="empty"):
            bls.aggregate_public_keys([])

    def test_aggregate_public_keys_requires_pop(self) -> None:
        bls = BLSOperations()
        _, pk = bls.generate_keypair()
        
        with pytest.raises(BLSError, match="proof-of-possession"):
            bls.aggregate_public_keys([pk])


# =============================================================================
# SMT Basic Tests
# =============================================================================


class TestSMTBasic:
    """Tests for basic SMT operations."""

    def test_insert_and_get(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        value = b"test value"
        
        smt.insert(key, value)
        assert smt.get(key) == value

    def test_get_nonexistent_returns_none(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        assert smt.get(key) is None

    def test_root_changes_on_insert(self) -> None:
        smt = SparseMerkleTree()
        root1 = smt.root
        
        smt.insert(secrets.token_bytes(32), b"value")
        root2 = smt.root
        
        assert root1 != root2

    def test_delete_key(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"value")
        assert smt.get(key) == b"value"
        
        smt.insert(key, b"")  # Delete by inserting empty value
        assert smt.get(key) is None


# =============================================================================
# SMT Proof Tests
# =============================================================================


class TestSMTProofs:
    """Tests for SMT membership and non-membership proofs."""

    def test_membership_proof_valid(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        value = b"test value"
        
        smt.insert(key, value)
        proof = smt.get_proof(key)
        
        assert proof.is_membership()
        assert proof.verify_membership(smt.root)
        assert proof.verify_membership(smt.root, expected_value=value)

    def test_membership_proof_wrong_value_fails(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"correct")
        proof = smt.get_proof(key)
        
        assert not proof.verify_membership(smt.root, expected_value=b"wrong")

    def test_non_membership_proof_valid(self) -> None:
        smt = SparseMerkleTree()
        existing_key = secrets.token_bytes(32)
        missing_key = secrets.token_bytes(32)
        
        smt.insert(existing_key, b"value")
        proof = smt.get_proof(missing_key)
        
        assert not proof.is_membership()
        assert proof.verify_non_membership(smt.root)

    def test_membership_proof_fails_non_membership_check(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"value")
        proof = smt.get_proof(key)
        
        assert not proof.verify_non_membership(smt.root)

    def test_non_membership_proof_fails_membership_check(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        proof = smt.get_proof(key)
        
        assert not proof.verify_membership(smt.root)

    def test_proof_invalid_root_fails(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"value")
        proof = smt.get_proof(key)
        
        wrong_root = secrets.token_bytes(32)
        assert not proof.verify(wrong_root)

    def test_proof_invalid_root_length_fails(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"value")
        proof = smt.get_proof(key)
        
        assert not proof.verify(b"short")
        assert not proof.verify(b"x" * 64)


# =============================================================================
# SMT Validation Tests
# =============================================================================


class TestSMTValidation:
    """Tests for SMT input validation."""

    def test_invalid_key_length_rejected(self) -> None:
        smt = SparseMerkleTree()
        
        with pytest.raises(SMTError, match="32 bytes"):
            smt.insert(b"short", b"value")
        
        with pytest.raises(SMTError, match="32 bytes"):
            smt.insert(b"x" * 64, b"value")

    def test_value_size_limit(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        # Should work at limit
        smt.insert(key, b"x" * MAX_SMT_VALUE_LEN)
        
        # Should fail over limit
        with pytest.raises(SMTError, match="exceeds"):
            smt.insert(key, b"x" * (MAX_SMT_VALUE_LEN + 1))

    def test_entry_count_limit(self) -> None:
        smt = SparseMerkleTree(max_entries=3)
        
        for _ in range(3):
            smt.insert(secrets.token_bytes(32), b"value")
        
        with pytest.raises(SMTError, match="exceeds"):
            smt.insert(secrets.token_bytes(32), b"value")

    def test_proof_sibling_length_validation(self) -> None:
        key = secrets.token_bytes(32)
        
        # Valid siblings
        valid_siblings = tuple(secrets.token_bytes(32) for _ in range(256))
        proof = SMTProof(key=key, value=b"test", siblings=valid_siblings)
        assert proof.is_membership()
        
        # Invalid sibling length
        bad_siblings = tuple(b"short" for _ in range(256))
        with pytest.raises(ValueError, match="must be"):
            SMTProof(key=key, value=b"test", siblings=bad_siblings)

    def test_proof_key_length_validation(self) -> None:
        siblings = tuple(secrets.token_bytes(32) for _ in range(256))
        
        with pytest.raises(ValueError, match="must be"):
            SMTProof(key=b"short", value=b"test", siblings=siblings)


# =============================================================================
# SMT Multiple Keys Tests
# =============================================================================


class TestSMTMultipleKeys:
    """Tests for SMT with multiple keys."""

    def test_multiple_inserts_all_verifiable(self) -> None:
        smt = SparseMerkleTree()
        entries = [(secrets.token_bytes(32), f"value_{i}".encode()) for i in range(10)]
        
        for key, value in entries:
            smt.insert(key, value)
        
        root = smt.root
        
        for key, value in entries:
            proof = smt.get_proof(key)
            assert proof.verify_membership(root, expected_value=value)

    def test_update_existing_key(self) -> None:
        smt = SparseMerkleTree()
        key = secrets.token_bytes(32)
        
        smt.insert(key, b"value1")
        root1 = smt.root
        
        smt.insert(key, b"value2")
        root2 = smt.root
        
        assert root1 != root2
        assert smt.get(key) == b"value2"
        
        proof = smt.get_proof(key)
        assert proof.verify_membership(root2, expected_value=b"value2")
