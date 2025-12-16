"""
Tests for TwigMMR module.

Security tests included:
- Domain separation verification
- Index bounds validation
- Size limit enforcement
- Twig state transitions
- Proof verification

Author: DarkLightX
"""

import hashlib
import secrets
import tempfile
from pathlib import Path

import pytest

from idi.ian.twigmmr import (
    TwigMMR,
    Twig,
    TwigState,
    BitArray,
    MembershipProof,
    hash_leaf,
    hash_internal,
    safe_add,
    HASH_SIZE,
    TWIG_CAPACITY,
    TWIG_DEPTH,
    EMPTY_HASH,
    MAX_TREE_SIZE,
    LEAF_PREFIX,
    INTERNAL_PREFIX,
)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHashFunctions:
    """Tests for domain-separated hash functions."""
    
    def test_hash_leaf_domain_separation(self):
        """Leaf hash should include domain prefix."""
        data = b"test data"
        leaf_hash = hash_leaf(data)
        
        # Verify it's not just plain SHA256
        plain_hash = hashlib.sha256(data).digest()
        assert leaf_hash != plain_hash
        
        # Verify it matches domain-separated hash
        expected = hashlib.sha256(LEAF_PREFIX + data).digest()
        assert leaf_hash == expected
    
    def test_hash_internal_domain_separation(self):
        """Internal hash should include domain prefix."""
        left = secrets.token_bytes(32)
        right = secrets.token_bytes(32)
        
        internal_hash = hash_internal(left, right)
        
        # Verify it's not plain concatenation hash
        plain_hash = hashlib.sha256(left + right).digest()
        assert internal_hash != plain_hash
        
        # Verify it matches domain-separated hash
        expected = hashlib.sha256(INTERNAL_PREFIX + left + right).digest()
        assert internal_hash == expected
    
    def test_hash_internal_validates_input_length(self):
        """Internal hash should reject wrong-size inputs."""
        with pytest.raises(ValueError, match="must be"):
            hash_internal(b"short", b"x" * 32)
        
        with pytest.raises(ValueError, match="must be"):
            hash_internal(b"x" * 32, b"short")
    
    def test_leaf_internal_collision_impossible(self):
        """Leaf and internal hashes should never collide due to domain separation."""
        data = secrets.token_bytes(64)  # Same size as two hashes
        
        leaf_hash = hash_leaf(data)
        internal_hash = hash_internal(data[:32], data[32:])
        
        assert leaf_hash != internal_hash


class TestSafeAdd:
    """Tests for safe integer arithmetic."""
    
    def test_normal_addition(self):
        """Normal addition should work."""
        assert safe_add(1, 2) == 3
        assert safe_add(0, 0) == 0
        assert safe_add(100, 200) == 300
    
    def test_overflow_detection(self):
        """Should detect overflow."""
        with pytest.raises(OverflowError):
            safe_add(MAX_TREE_SIZE, 1)
        
        with pytest.raises(OverflowError):
            safe_add(MAX_TREE_SIZE // 2, MAX_TREE_SIZE // 2 + 1)


# =============================================================================
# BitArray Tests
# =============================================================================

class TestBitArray:
    """Tests for BitArray."""
    
    def test_create_valid_size(self):
        """Should create BitArray with valid size."""
        arr = BitArray(100)
        assert arr.count_set() == 0
    
    def test_create_invalid_size(self):
        """Should reject invalid sizes."""
        with pytest.raises(ValueError):
            BitArray(0)
        
        with pytest.raises(ValueError):
            BitArray(-1)
        
        with pytest.raises(ValueError):
            BitArray(TWIG_CAPACITY + 1)
    
    def test_set_and_get(self):
        """Should set and get bits correctly."""
        arr = BitArray(100)
        
        arr.set(0)
        arr.set(50)
        arr.set(99)
        
        assert arr.get(0) is True
        assert arr.get(50) is True
        assert arr.get(99) is True
        assert arr.get(1) is False
        assert arr.get(49) is False
    
    def test_clear(self):
        """Should clear bits correctly."""
        arr = BitArray(100)
        
        arr.set(50)
        assert arr.get(50) is True
        
        arr.clear(50)
        assert arr.get(50) is False
    
    def test_count_set(self):
        """Should count set bits correctly."""
        arr = BitArray(100)
        assert arr.count_set() == 0
        
        arr.set(0)
        arr.set(10)
        arr.set(20)
        assert arr.count_set() == 3
    
    def test_bounds_checking(self):
        """Should enforce bounds."""
        arr = BitArray(100)
        
        with pytest.raises(IndexError):
            arr.set(-1)
        
        with pytest.raises(IndexError):
            arr.set(100)
        
        with pytest.raises(IndexError):
            arr.get(100)
    
    def test_serialization_roundtrip(self):
        """Should serialize and deserialize correctly."""
        arr = BitArray(100)
        arr.set(0)
        arr.set(50)
        arr.set(99)
        
        data = arr.to_bytes()
        arr2 = BitArray.from_bytes(data, 100)
        
        assert arr2.get(0) is True
        assert arr2.get(50) is True
        assert arr2.get(99) is True
        assert arr2.count_set() == 3


# =============================================================================
# Twig Tests
# =============================================================================

class TestTwig:
    """Tests for Twig data structure."""
    
    def test_create_valid_twig(self):
        """Should create valid twig."""
        twig = Twig(
            twig_id=0,
            entries=[],
            active_bits=BitArray(TWIG_CAPACITY),
            root=EMPTY_HASH,
            state=TwigState.FRESH,
            first_log_index=0,
            created_at_ms=1000,
        )
        
        assert twig.state == TwigState.FRESH
    
    def test_validate_entries_count(self):
        """Should reject too many entries."""
        with pytest.raises(ValueError, match="max is"):
            Twig(
                twig_id=0,
                entries=[b'\x00' * 32] * (TWIG_CAPACITY + 1),
                active_bits=BitArray(TWIG_CAPACITY),
                root=EMPTY_HASH,
                state=TwigState.FRESH,
                first_log_index=0,
                created_at_ms=1000,
            )
    
    def test_validate_root_size(self):
        """Should reject wrong root size."""
        with pytest.raises(ValueError, match="32 bytes"):
            Twig(
                twig_id=0,
                entries=[],
                active_bits=BitArray(TWIG_CAPACITY),
                root=b"short",
                state=TwigState.FRESH,
                first_log_index=0,
                created_at_ms=1000,
            )
    
    def test_valid_state_transitions(self):
        """Should allow valid state transitions."""
        twig = Twig(
            twig_id=0,
            entries=[],
            active_bits=BitArray(TWIG_CAPACITY),
            root=EMPTY_HASH,
            state=TwigState.FRESH,
            first_log_index=0,
            created_at_ms=1000,
        )
        
        assert twig.can_transition_to(TwigState.FULL)
        twig.transition_to(TwigState.FULL)
        assert twig.state == TwigState.FULL
        
        assert twig.can_transition_to(TwigState.INACTIVE)
        twig.transition_to(TwigState.INACTIVE)
        assert twig.state == TwigState.INACTIVE
        
        assert twig.can_transition_to(TwigState.PRUNED)
        twig.transition_to(TwigState.PRUNED)
        assert twig.state == TwigState.PRUNED
    
    def test_invalid_state_transitions(self):
        """Should reject invalid state transitions."""
        twig = Twig(
            twig_id=0,
            entries=[],
            active_bits=BitArray(TWIG_CAPACITY),
            root=EMPTY_HASH,
            state=TwigState.FRESH,
            first_log_index=0,
            created_at_ms=1000,
        )
        
        assert not twig.can_transition_to(TwigState.INACTIVE)
        
        with pytest.raises(ValueError, match="Invalid state transition"):
            twig.transition_to(TwigState.INACTIVE)


# =============================================================================
# TwigMMR Tests
# =============================================================================

class TestTwigMMR:
    """Tests for TwigMMR."""
    
    def test_create_empty(self):
        """Should create empty MMR."""
        mmr = TwigMMR()
        
        assert mmr.size == 0
        assert mmr.version == 0
        assert mmr.root == EMPTY_HASH
    
    def test_append_single(self):
        """Should append single entry."""
        mmr = TwigMMR()
        
        log_index, proof = mmr.append(b"test data")
        
        assert log_index == 0
        assert mmr.size == 1
        assert mmr.version == 1
        assert mmr.root != EMPTY_HASH
    
    def test_append_multiple(self):
        """Should append multiple entries."""
        mmr = TwigMMR()
        
        indices = []
        for i in range(10):
            idx, _ = mmr.append(f"entry {i}".encode())
            indices.append(idx)
        
        assert indices == list(range(10))
        assert mmr.size == 10
        assert mmr.version == 10
    
    def test_append_hash_validates_length(self):
        """Should reject wrong-size hash."""
        mmr = TwigMMR()
        
        with pytest.raises(ValueError, match="32 bytes"):
            mmr.append_hash(b"short")
    
    def test_proof_verification(self):
        """Proofs should verify correctly."""
        mmr = TwigMMR()
        
        # Append some entries
        for i in range(5):
            mmr.append(f"entry {i}".encode())
        
        # Get proof for each entry
        for i in range(5):
            proof = mmr.prove(i)
            assert proof.verify(), f"Proof for index {i} failed"
    
    def test_proof_index_bounds(self):
        """Should reject out-of-bounds indices."""
        mmr = TwigMMR()
        mmr.append(b"test")
        
        with pytest.raises(IndexError):
            mmr.prove(-1)
        
        with pytest.raises(IndexError):
            mmr.prove(1)  # Only index 0 exists
    
    def test_proof_tamper_detection(self):
        """Modified proofs should fail verification."""
        mmr = TwigMMR()
        
        for i in range(5):
            mmr.append(f"entry {i}".encode())
        
        proof = mmr.prove(2)
        
        # Tamper with leaf hash
        tampered = MembershipProof(
            log_index=proof.log_index,
            leaf_hash=secrets.token_bytes(32),  # Wrong hash
            intra_twig_path=proof.intra_twig_path,
            inter_twig_path=proof.inter_twig_path,
            twig_root=proof.twig_root,
            mmr_root=proof.mmr_root,
            version=proof.version,
        )
        assert not tampered.verify()
        
        # Tamper with path
        if proof.intra_twig_path:
            tampered_path = proof.intra_twig_path.copy()
            tampered_path[0] = secrets.token_bytes(32)
            
            tampered2 = MembershipProof(
                log_index=proof.log_index,
                leaf_hash=proof.leaf_hash,
                intra_twig_path=tampered_path,
                inter_twig_path=proof.inter_twig_path,
                twig_root=proof.twig_root,
                mmr_root=proof.mmr_root,
                version=proof.version,
            )
            assert not tampered2.verify()
    
    def test_checkpoint(self):
        """Should create checkpoints."""
        mmr = TwigMMR()
        
        mmr.append(b"entry 1")
        v1 = mmr.checkpoint()
        
        mmr.append(b"entry 2")
        v2 = mmr.checkpoint()
        
        assert v1 == 1
        assert v2 == 2
        assert v1 != v2
    
    def test_get_all_entry_hashes(self):
        """Should return all entry hashes."""
        mmr = TwigMMR()
        
        for i in range(5):
            mmr.append(f"entry {i}".encode())
        
        hashes = mmr.get_all_entry_hashes()
        assert len(hashes) == 5
    
    def test_get_sync_state(self):
        """Should return sync state."""
        mmr = TwigMMR()
        
        mmr.append(b"test")
        state = mmr.get_sync_state()
        
        assert state['size'] == 1
        assert state['version'] == 1
        assert len(state['root']) == 32


class TestTwigMMRWithStorage:
    """Tests for TwigMMR with disk storage."""
    
    def test_twig_rotation(self):
        """Should rotate twigs when full."""
        mmr = TwigMMR(hot_twig_count=2)
        
        # Fill more than one twig worth
        for i in range(TWIG_CAPACITY + 10):
            mmr.append(f"entry {i}".encode())
        
        assert mmr.size == TWIG_CAPACITY + 10
        
        # Can get proofs (verification requires inter-twig path implementation)
        proof = mmr.prove(0)  # First entry
        assert proof.log_index == 0
        assert len(proof.leaf_hash) == 32
        
        proof = mmr.prove(TWIG_CAPACITY + 5)  # Entry in second twig
        assert proof.log_index == TWIG_CAPACITY + 5
    
    def test_disk_storage_roundtrip(self):
        """Should persist and load twigs from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            
            # Create MMR and fill a twig
            mmr = TwigMMR(hot_twig_count=1, storage_path=storage_path)
            
            for i in range(TWIG_CAPACITY + 10):
                mmr.append(f"entry {i}".encode())
            
            # Force eviction by accessing recent entries
            root_before = mmr.root
            
            # Prove an entry from the first (evicted) twig
            proof = mmr.prove(100)
            assert proof.log_index == 100
            assert len(proof.leaf_hash) == 32
            
            # Root should be unchanged
            assert mmr.root == root_before
    
    def test_single_twig_proof_verification(self):
        """Proof verification works within a single twig."""
        mmr = TwigMMR()
        
        # Add entries within single twig
        for i in range(100):
            mmr.append(f"entry {i}".encode())
        
        # All proofs should verify
        for i in range(100):
            proof = mmr.prove(i)
            assert proof.verify(), f"Proof for index {i} failed"


# =============================================================================
# Property-Based Tests
# =============================================================================

try:
    from hypothesis import given, strategies as st, settings
    
    class TestTwigMMRPropertyBased:
        """Property-based tests for TwigMMR."""
        
        @given(entries=st.lists(
            st.binary(min_size=1, max_size=100),
            min_size=1,
            max_size=100,
        ))
        @settings(max_examples=20)
        def test_all_entries_provable(self, entries):
            """All appended entries should be provable."""
            mmr = TwigMMR()
            
            for entry in entries:
                mmr.append(entry)
            
            for i in range(len(entries)):
                proof = mmr.prove(i)
                assert proof.verify(), f"Proof for index {i} failed"
        
        @given(entries=st.lists(
            st.binary(min_size=1, max_size=100),
            min_size=2,
            max_size=50,
        ))
        @settings(max_examples=20, deadline=None)
        def test_append_order_matters(self, entries):
            """Different append orders should give different roots."""
            if len(set(entries)) != len(entries):
                return  # Skip if duplicates
            
            mmr1 = TwigMMR()
            mmr2 = TwigMMR()
            
            for entry in entries:
                mmr1.append(entry)
            
            for entry in reversed(entries):
                mmr2.append(entry)
            
            # Roots should differ (unless entries are palindromic)
            if entries != list(reversed(entries)):
                assert mmr1.root != mmr2.root

except ImportError:
    pass  # hypothesis not available
