"""
Tests for FrontierSync and IBLT modules.

Security tests included:
- IBLT bounds validation
- Replay attack prevention
- Timestamp freshness enforcement
- Nonce uniqueness
- Input validation

Author: DarkLightX
"""

import asyncio
import hashlib
import secrets
import time
from typing import Any, List, Optional, Set

import pytest

from idi.ian.network.iblt import (
    IBLT,
    IBLTCell,
    IBLTConfig,
    HASH_SIZE,
    MAX_IBLT_CELLS,
    MIN_IBLT_CELLS,
    estimate_iblt_size,
)
from idi.ian.network.frontiersync import (
    FrontierSync,
    SyncState,
    SyncStatus,
    SyncResult,
    WitnessSignature,
    CosignedSyncState,
    ForkProof,
    SyncSession,
    NonceCache,
    SyncRateLimiter,
    WitnessCoordinator,
    FRESHNESS_WINDOW_MS,
    MAX_CLOCK_SKEW_MS,
)


# =============================================================================
# IBLT Tests
# =============================================================================

class TestIBLTConfig:
    """Tests for IBLTConfig validation."""
    
    def test_valid_config(self):
        """Valid config should be accepted."""
        config = IBLTConfig(num_cells=1000, num_hashes=3)
        assert config.num_cells == 1000
        assert config.num_hashes == 3
        assert len(config.hash_seed) == 32
    
    def test_config_bounds_min_cells(self):
        """Config with too few cells should raise."""
        with pytest.raises(ValueError, match="num_cells must be"):
            IBLTConfig(num_cells=MIN_IBLT_CELLS - 1)
    
    def test_config_bounds_max_cells(self):
        """Config with too many cells should raise."""
        with pytest.raises(ValueError, match="num_cells must be"):
            IBLTConfig(num_cells=MAX_IBLT_CELLS + 1)
    
    def test_config_bounds_num_hashes(self):
        """Config with invalid num_hashes should raise."""
        with pytest.raises(ValueError, match="num_hashes must be"):
            IBLTConfig(num_cells=100, num_hashes=0)
        with pytest.raises(ValueError, match="num_hashes must be"):
            IBLTConfig(num_cells=100, num_hashes=11)
    
    def test_config_hash_seed_length(self):
        """Config with wrong hash_seed length should raise."""
        with pytest.raises(ValueError, match="hash_seed must be 32 bytes"):
            IBLTConfig(num_cells=100, hash_seed=b"short")


class TestIBLTCell:
    """Tests for IBLTCell."""
    
    def test_empty_cell(self):
        """Empty cell should be detected."""
        cell = IBLTCell()
        assert cell.is_empty()
        assert not cell.is_pure()
    
    def test_pure_cell_positive(self):
        """Pure cell with count=1 should be detected."""
        key = secrets.token_bytes(32)
        key_hash = hashlib.sha256(key).digest()
        cell = IBLTCell(count=1, key_sum=key, hash_sum=key_hash)
        assert cell.is_pure()
        assert not cell.is_empty()
    
    def test_pure_cell_negative(self):
        """Pure cell with count=-1 should be detected."""
        key = secrets.token_bytes(32)
        key_hash = hashlib.sha256(key).digest()
        cell = IBLTCell(count=-1, key_sum=key, hash_sum=key_hash)
        assert cell.is_pure()
    
    def test_impure_cell_wrong_count(self):
        """Cell with count != ±1 is not pure."""
        key = secrets.token_bytes(32)
        key_hash = hashlib.sha256(key).digest()
        cell = IBLTCell(count=2, key_sum=key, hash_sum=key_hash)
        assert not cell.is_pure()
    
    def test_impure_cell_wrong_hash(self):
        """Cell with mismatched hash is not pure."""
        key = secrets.token_bytes(32)
        wrong_hash = secrets.token_bytes(32)
        cell = IBLTCell(count=1, key_sum=key, hash_sum=wrong_hash)
        assert not cell.is_pure()


class TestIBLT:
    """Tests for IBLT operations."""
    
    @pytest.fixture
    def config(self):
        """Shared IBLT config for tests - sized for test workloads."""
        # Need ~1.5x * num_hashes * expected_elements cells for reliable decode
        return IBLTConfig(num_cells=1000, num_hashes=4)
    
    def test_insert_key_validation(self, config):
        """Insert should validate key length."""
        iblt = IBLT(config)
        with pytest.raises(ValueError, match="Key must be"):
            iblt.insert(b"short")
    
    def test_insert_and_decode_single(self, config):
        """Single inserted key should decode correctly."""
        iblt = IBLT(config)
        key = secrets.token_bytes(32)
        iblt.insert(key)
        
        # Create empty IBLT and subtract
        empty = IBLT(config)
        diff = iblt.subtract(empty)
        only_in_iblt, only_in_empty, success = diff.decode()
        
        assert success
        assert key in only_in_iblt
        assert len(only_in_empty) == 0
    
    def test_insert_and_decode_multiple(self, config):
        """Multiple inserted keys should decode correctly."""
        iblt = IBLT(config)
        keys = [secrets.token_bytes(32) for _ in range(10)]
        for key in keys:
            iblt.insert(key)
        
        empty = IBLT(config)
        diff = iblt.subtract(empty)
        only_in_iblt, only_in_empty, success = diff.decode()
        
        assert success
        assert set(keys) == only_in_iblt
        assert len(only_in_empty) == 0
    
    def test_symmetric_difference(self, config):
        """Symmetric difference should be computed correctly."""
        iblt_a = IBLT(config)
        iblt_b = IBLT(config)
        
        # Common keys
        common = [secrets.token_bytes(32) for _ in range(5)]
        for key in common:
            iblt_a.insert(key)
            iblt_b.insert(key)
        
        # Keys only in A
        only_a = [secrets.token_bytes(32) for _ in range(3)]
        for key in only_a:
            iblt_a.insert(key)
        
        # Keys only in B
        only_b = [secrets.token_bytes(32) for _ in range(4)]
        for key in only_b:
            iblt_b.insert(key)
        
        diff = iblt_a.subtract(iblt_b)
        decoded_a, decoded_b, success = diff.decode()
        
        assert success
        assert set(only_a) == decoded_a
        assert set(only_b) == decoded_b
    
    def test_serialize_deserialize(self, config):
        """Serialization should preserve IBLT state."""
        iblt = IBLT(config)
        keys = [secrets.token_bytes(32) for _ in range(10)]
        for key in keys:
            iblt.insert(key)
        
        # Serialize and deserialize
        data = iblt.serialize()
        iblt2 = IBLT.deserialize(data, config)
        
        # Subtract should give empty difference
        diff = iblt.subtract(iblt2)
        only_a, only_b, success = diff.decode()
        
        assert success
        assert len(only_a) == 0
        assert len(only_b) == 0
    
    def test_deserialize_config_mismatch(self, config):
        """Deserialize with wrong config should raise."""
        iblt = IBLT(config)
        data = iblt.serialize()
        
        # Different config
        other_config = IBLTConfig(
            num_cells=config.num_cells,
            hash_seed=secrets.token_bytes(32),  # Different seed
        )
        
        with pytest.raises(ValueError, match="Config mismatch"):
            IBLT.deserialize(data, other_config)
    
    def test_deserialize_truncated_data(self, config):
        """Deserialize with truncated data should raise."""
        iblt = IBLT(config)
        data = iblt.serialize()
        
        with pytest.raises(ValueError, match="Data"):
            IBLT.deserialize(data[:100], config)


class TestEstimateIBLTSize:
    """Tests for IBLT size estimation."""
    
    def test_small_diff(self):
        """Small diff should give reasonable size."""
        size = estimate_iblt_size(10)
        assert MIN_IBLT_CELLS <= size <= MAX_IBLT_CELLS
    
    def test_large_diff(self):
        """Large diff should be bounded by MAX."""
        size = estimate_iblt_size(1_000_000)
        assert size == MAX_IBLT_CELLS
    
    def test_zero_diff(self):
        """Zero diff should give MIN."""
        size = estimate_iblt_size(0)
        assert size == MIN_IBLT_CELLS


# =============================================================================
# SyncState Tests
# =============================================================================

class TestSyncState:
    """Tests for SyncState."""
    
    def test_root_hash_empty(self):
        """Empty frontier should give zero hash."""
        state = SyncState(goal_id="test", size=0, frontier=[], version=0)
        assert state.root_hash() == b'\x00' * HASH_SIZE
    
    def test_root_hash_single_peak(self):
        """Single peak frontier should return that peak."""
        peak = secrets.token_bytes(32)
        state = SyncState(goal_id="test", size=1, frontier=[peak], version=0)
        assert state.root_hash() == peak
    
    def test_root_hash_multiple_peaks(self):
        """Multiple peaks should be bagged correctly."""
        peaks = [secrets.token_bytes(32) for _ in range(3)]
        state = SyncState(goal_id="test", size=7, frontier=peaks, version=0)
        
        # Manual calculation: bag right-to-left
        expected = peaks[2]
        expected = hashlib.sha256(peaks[1] + expected).digest()
        expected = hashlib.sha256(peaks[0] + expected).digest()
        
        assert state.root_hash() == expected
    
    def test_serialize_deterministic(self):
        """Serialization should be deterministic."""
        state = SyncState(
            goal_id="test",
            size=100,
            frontier=[b'\x01' * 32, b'\x02' * 32],
            version=5,
            timestamp_ms=1234567890,
        )
        
        s1 = state.serialize()
        s2 = state.serialize()
        assert s1 == s2
    
    def test_hash_deterministic(self):
        """Hash should be deterministic."""
        state = SyncState(
            goal_id="test",
            size=100,
            frontier=[b'\x01' * 32],
            version=1,
            timestamp_ms=1000,
        )
        
        h1 = state.hash()
        h2 = state.hash()
        assert h1 == h2
        assert len(h1) == 32


# =============================================================================
# Security Tests
# =============================================================================

class TestNonceCache:
    """Tests for NonceCache replay prevention."""
    
    @pytest.mark.asyncio
    async def test_fresh_nonce_accepted(self):
        """Fresh nonce should be accepted."""
        cache = NonceCache(max_size=100)
        nonce = secrets.token_bytes(32)
        
        result = await cache.check_and_add(nonce, int(time.time() * 1000))
        assert result is True
    
    @pytest.mark.asyncio
    async def test_duplicate_nonce_rejected(self):
        """Duplicate nonce should be rejected."""
        cache = NonceCache(max_size=100)
        nonce = secrets.token_bytes(32)
        now_ms = int(time.time() * 1000)
        
        # First use
        result1 = await cache.check_and_add(nonce, now_ms)
        assert result1 is True
        
        # Replay attempt
        result2 = await cache.check_and_add(nonce, now_ms)
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Cache should evict oldest entries when full."""
        cache = NonceCache(max_size=10)
        
        # Fill cache
        for i in range(15):
            nonce = i.to_bytes(32, 'big')
            await cache.check_and_add(nonce, 1000 + i)
        
        # Oldest should be evicted, so replay of it should succeed
        oldest = (0).to_bytes(32, 'big')
        result = await cache.check_and_add(oldest, 2000)
        assert result is True  # Not in cache anymore


class TestSyncRateLimiter:
    """Tests for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_under_limit_allowed(self):
        """Requests under limit should be allowed."""
        limiter = SyncRateLimiter(max_requests_per_minute=10)
        
        for _ in range(5):
            allowed, _ = await limiter.check_rate_limit("peer1")
            assert allowed is True
    
    @pytest.mark.asyncio
    async def test_over_limit_rejected(self):
        """Requests over limit should be rejected."""
        limiter = SyncRateLimiter(max_requests_per_minute=5)
        
        # Use up limit
        for _ in range(5):
            await limiter.check_rate_limit("peer1")
        
        # Next should be rejected
        allowed, reason = await limiter.check_rate_limit("peer1")
        assert allowed is False
        assert "Rate limit" in reason
    
    @pytest.mark.asyncio
    async def test_different_peers_independent(self):
        """Rate limits should be per-peer."""
        limiter = SyncRateLimiter(max_requests_per_minute=2)
        
        # Exhaust peer1 limit
        await limiter.check_rate_limit("peer1")
        await limiter.check_rate_limit("peer1")
        allowed1, _ = await limiter.check_rate_limit("peer1")
        assert allowed1 is False
        
        # peer2 should still be allowed
        allowed2, _ = await limiter.check_rate_limit("peer2")
        assert allowed2 is True


class TestCosignedSyncState:
    """Tests for witness cosigning validation."""
    
    def _make_witness_signature(
        self, 
        state: SyncState,
        witness_id: str = "witness1",
        timestamp_offset_ms: int = 0,
    ) -> WitnessSignature:
        """Create a mock witness signature."""
        return WitnessSignature(
            witness_id=witness_id,
            public_key=b'\x00' * 32,  # Mock
            signature=b'\x00' * 64,   # Mock
            session_nonce=secrets.token_bytes(32),
            timestamp_ms=int(time.time() * 1000) + timestamp_offset_ms,
        )
    
    def test_insufficient_witnesses(self):
        """Should reject if below threshold."""
        state = SyncState(goal_id="test", size=10, frontier=[b'\x01' * 32], version=1)
        cosigned = CosignedSyncState(
            state=state,
            witnesses=[],
            threshold=3,
        )
        
        valid, error = cosigned.is_valid()
        assert valid is False
        assert "Insufficient witnesses" in error
    
    def test_duplicate_witnesses_rejected(self):
        """Should reject duplicate witness IDs."""
        state = SyncState(goal_id="test", size=10, frontier=[b'\x01' * 32], version=1)
        w1 = self._make_witness_signature(state, "witness1")
        w2 = self._make_witness_signature(state, "witness1")  # Same ID!
        
        cosigned = CosignedSyncState(
            state=state,
            witnesses=[w1, w2],
            threshold=2,
        )
        
        valid, error = cosigned.is_valid()
        assert valid is False
        assert "Duplicate" in error


class TestForkProof:
    """Tests for fork proof validation."""
    
    def test_different_sizes_invalid(self):
        """Fork proof with different sizes is invalid."""
        state_a = SyncState(goal_id="test", size=10, frontier=[b'\x01' * 32], version=1)
        state_b = SyncState(goal_id="test", size=20, frontier=[b'\x02' * 32], version=1)
        
        cosigned_a = CosignedSyncState(state=state_a, witnesses=[], threshold=0)
        cosigned_b = CosignedSyncState(state=state_b, witnesses=[], threshold=0)
        
        proof = ForkProof(
            state_a=cosigned_a,
            state_b=cosigned_b,
            conflicting_witnesses=[],
        )
        
        valid, error = proof.is_valid()
        assert valid is False
        assert "different sizes" in error
    
    def test_same_frontier_invalid(self):
        """Fork proof with same frontier is invalid (no fork)."""
        frontier = [b'\x01' * 32]
        state_a = SyncState(goal_id="test", size=10, frontier=frontier, version=1)
        state_b = SyncState(goal_id="test", size=10, frontier=frontier, version=2)
        
        cosigned_a = CosignedSyncState(state=state_a, witnesses=[], threshold=0)
        cosigned_b = CosignedSyncState(state=state_b, witnesses=[], threshold=0)
        
        proof = ForkProof(
            state_a=cosigned_a,
            state_b=cosigned_b,
            conflicting_witnesses=[],
        )
        
        valid, error = proof.is_valid()
        assert valid is False
        assert "identical" in error


class TestSyncSession:
    """Tests for sync session management."""
    
    def test_session_expiry(self):
        """Session should expire after max duration."""
        session = SyncSession(
            session_id="test",
            peer_id="peer1",
            goal_id="goal1",
            direction="NONE",
        )
        
        # Not expired immediately
        assert not session.is_expired()
        
        # Expired after max duration
        future_ms = session.started_ms + 400_000  # 400 seconds (> 300s max)
        assert session.is_expired(future_ms)
    
    def test_session_touch(self):
        """Touch should update last activity."""
        session = SyncSession(
            session_id="test",
            peer_id="peer1",
            goal_id="goal1",
            direction="NONE",
        )
        
        old_activity = session.last_activity_ms
        time.sleep(0.01)  # Small delay
        session.touch()
        
        assert session.last_activity_ms >= old_activity


# =============================================================================
# Property-Based Tests (if hypothesis available)
# =============================================================================

try:
    from hypothesis import given, strategies as st, settings
    
    class TestIBLTPropertyBased:
        """Property-based tests for IBLT."""
        
        @given(keys=st.lists(
            st.binary(min_size=32, max_size=32),
            min_size=0,
            max_size=50,
            unique=True,
        ))
        @settings(max_examples=50)
        def test_insert_decode_roundtrip(self, keys: List[bytes]):
            """All inserted keys should be recoverable."""
            config = IBLTConfig(num_cells=500)
            iblt = IBLT(config)
            
            for key in keys:
                iblt.insert(key)
            
            empty = IBLT(config)
            diff = iblt.subtract(empty)
            decoded, _, success = diff.decode()
            
            if success:
                assert decoded == set(keys)
        
        @given(
            keys_a=st.lists(st.binary(min_size=32, max_size=32), min_size=0, max_size=20, unique=True),
            keys_b=st.lists(st.binary(min_size=32, max_size=32), min_size=0, max_size=20, unique=True),
        )
        @settings(max_examples=30)
        def test_symmetric_difference_property(self, keys_a: List[bytes], keys_b: List[bytes]):
            """Symmetric difference should match set difference."""
            config = IBLTConfig(num_cells=500)
            
            iblt_a = IBLT(config)
            iblt_b = IBLT(config)
            
            for key in keys_a:
                iblt_a.insert(key)
            for key in keys_b:
                iblt_b.insert(key)
            
            diff = iblt_a.subtract(iblt_b)
            decoded_a, decoded_b, success = diff.decode()
            
            if success:
                expected_a = set(keys_a) - set(keys_b)
                expected_b = set(keys_b) - set(keys_a)
                assert decoded_a == expected_a
                assert decoded_b == expected_b

except ImportError:
    pass  # hypothesis not available


# =============================================================================
# Security Feature Tests (P1/P2)
# =============================================================================

class TestIBLTHMACAuthentication:
    """Tests for IBLT HMAC authentication (P1 fix)."""
    
    def test_serialize_with_hmac(self):
        """Serialization with auth_key should append HMAC."""
        config = IBLTConfig(num_cells=100)
        iblt = IBLT(config)
        iblt.insert(secrets.token_bytes(32))
        
        auth_key = secrets.token_bytes(32)
        
        # Without HMAC
        data_no_mac = iblt.serialize()
        # With HMAC
        data_with_mac = iblt.serialize(auth_key=auth_key)
        
        # HMAC adds 32 bytes
        assert len(data_with_mac) == len(data_no_mac) + 32
    
    def test_deserialize_with_valid_hmac(self):
        """Deserialization with valid HMAC should succeed."""
        config = IBLTConfig(num_cells=100)
        iblt = IBLT(config)
        test_key = secrets.token_bytes(32)
        iblt.insert(test_key)
        
        auth_key = secrets.token_bytes(32)
        data = iblt.serialize(auth_key=auth_key)
        
        # Should deserialize without error
        restored = IBLT.deserialize(data, config, auth_key=auth_key)
        assert restored is not None
    
    def test_deserialize_with_invalid_hmac_fails(self):
        """Deserialization with wrong auth_key should fail."""
        config = IBLTConfig(num_cells=100)
        iblt = IBLT(config)
        iblt.insert(secrets.token_bytes(32))
        
        auth_key = secrets.token_bytes(32)
        wrong_key = secrets.token_bytes(32)
        
        data = iblt.serialize(auth_key=auth_key)
        
        with pytest.raises(ValueError, match="HMAC verification failed"):
            IBLT.deserialize(data, config, auth_key=wrong_key)
    
    def test_deserialize_tampered_data_fails(self):
        """Deserialization of tampered data should fail HMAC check."""
        config = IBLTConfig(num_cells=100)
        iblt = IBLT(config)
        iblt.insert(secrets.token_bytes(32))
        
        auth_key = secrets.token_bytes(32)
        data = bytearray(iblt.serialize(auth_key=auth_key))
        
        # Tamper with data (flip a bit in the middle)
        data[len(data) // 2] ^= 0x01
        
        with pytest.raises(ValueError, match="HMAC verification failed"):
            IBLT.deserialize(bytes(data), config, auth_key=auth_key)
    
    def test_auth_key_must_be_32_bytes(self):
        """Auth key must be exactly 32 bytes."""
        config = IBLTConfig(num_cells=100)
        iblt = IBLT(config)
        
        with pytest.raises(ValueError, match="auth_key must be 32 bytes"):
            iblt.serialize(auth_key=b"short")
        
        with pytest.raises(ValueError, match="auth_key must be 32 bytes"):
            IBLT.deserialize(b"x" * 100, config, auth_key=b"short")


class TestWitnessDiversityValidation:
    """Tests for witness diversity validation (P1 fix)."""
    
    def _make_witness(self, witness_id: str, public_key: bytes) -> WitnessSignature:
        """Create a mock witness signature."""
        return WitnessSignature(
            witness_id=witness_id,
            public_key=public_key,
            signature=b'\x00' * 64,
            session_nonce=secrets.token_bytes(32),
            timestamp_ms=int(time.time() * 1000),
        )
    
    def test_sufficient_diversity_passes(self):
        """Cosigned state with diverse witnesses should pass."""
        state = SyncState(
            goal_id="test-goal",
            size=100,
            frontier=[secrets.token_bytes(32)],
            version=1,
        )
        
        # Two different public keys
        pk1 = secrets.token_bytes(32)
        pk2 = secrets.token_bytes(32)
        
        witnesses = [
            self._make_witness("w1", pk1),
            self._make_witness("w2", pk2),
        ]
        
        cosigned = CosignedSyncState(
            state=state,
            witnesses=witnesses,
            threshold=1,
            min_unique_entities=2,
        )
        
        # Diversity check passes (signatures won't verify without crypto)
        # but diversity check happens before signature verification
        is_valid, msg = cosigned.is_valid()
        # Will fail on signature, not diversity
        assert "diversity" not in msg.lower()
    
    def test_insufficient_diversity_fails(self):
        """Cosigned state with same public key should fail diversity check."""
        state = SyncState(
            goal_id="test-goal",
            size=100,
            frontier=[secrets.token_bytes(32)],
            version=1,
        )
        
        # Same public key for both witnesses (Sybil attempt)
        pk_same = secrets.token_bytes(32)
        
        witnesses = [
            self._make_witness("w1", pk_same),
            self._make_witness("w2", pk_same),  # Same key!
        ]
        
        cosigned = CosignedSyncState(
            state=state,
            witnesses=witnesses,
            threshold=1,
            min_unique_entities=2,
        )
        
        is_valid, msg = cosigned.is_valid()
        assert not is_valid
        assert "diversity" in msg.lower()


class TestSyncStateValidation:
    """Tests for SyncState input validation (P2 fix)."""
    
    def test_valid_state(self):
        """Valid state should be accepted."""
        state = SyncState(
            goal_id="valid-goal-123",
            size=1000,
            frontier=[secrets.token_bytes(32)],
            version=1,
        )
        assert state.goal_id == "valid-goal-123"
    
    def test_invalid_goal_id_chars(self):
        """Goal ID with invalid characters should be rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            SyncState(
                goal_id="invalid/path/../escape",
                size=100,
                frontier=[],
                version=1,
            )
    
    def test_goal_id_too_long(self):
        """Goal ID exceeding max length should be rejected."""
        with pytest.raises(ValueError, match="256 chars"):
            SyncState(
                goal_id="x" * 300,
                size=100,
                frontier=[],
                version=1,
            )
    
    def test_negative_size_rejected(self):
        """Negative size should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SyncState(
                goal_id="test",
                size=-1,
                frontier=[],
                version=1,
            )
    
    def test_frontier_peak_wrong_size(self):
        """Frontier peak with wrong size should be rejected."""
        with pytest.raises(ValueError, match="32 bytes"):
            SyncState(
                goal_id="test",
                size=100,
                frontier=[b"short"],
                version=1,
            )


class TestWitnessSignatureValidation:
    """Tests for WitnessSignature input validation (P2 fix)."""
    
    def test_valid_witness(self):
        """Valid witness signature should be accepted."""
        ws = WitnessSignature(
            witness_id="witness-1",
            public_key=b'\x00' * 32,
            signature=b'\x00' * 64,
            session_nonce=b'\x00' * 32,
            timestamp_ms=int(time.time() * 1000),
        )
        assert ws.witness_id == "witness-1"
    
    def test_invalid_public_key_size(self):
        """Public key must be 32 bytes (Ed25519)."""
        with pytest.raises(ValueError, match="32 bytes"):
            WitnessSignature(
                witness_id="w1",
                public_key=b"short",
                signature=b'\x00' * 64,
                session_nonce=b'\x00' * 32,
                timestamp_ms=1000,
            )
    
    def test_invalid_signature_size(self):
        """Signature must be 64 bytes (Ed25519)."""
        with pytest.raises(ValueError, match="64 bytes"):
            WitnessSignature(
                witness_id="w1",
                public_key=b'\x00' * 32,
                signature=b"short",
                session_nonce=b'\x00' * 32,
                timestamp_ms=1000,
            )
    
    def test_invalid_nonce_size(self):
        """Session nonce must be 32 bytes."""
        with pytest.raises(ValueError, match="32 bytes"):
            WitnessSignature(
                witness_id="w1",
                public_key=b'\x00' * 32,
                signature=b'\x00' * 64,
                session_nonce=b"short",
                timestamp_ms=1000,
            )


class _DummyMMR:
    def __init__(self, hashes: List[bytes]):
        self._hashes = hashes
        self.size = len(hashes)
        self.frontier = []
        self.version = 0
        self.appended: List[bytes] = []

    def get_all_entry_hashes(self) -> List[bytes]:
        return list(self._hashes)

    def append(self, entry: bytes) -> None:
        self.appended.append(entry)


class _DummyTransport:
    def __init__(self, session_key: bytes):
        self._session_key = session_key

    def get_session_key(self, peer_id: str) -> bytes:
        return self._session_key


class _TestFrontierSync(FrontierSync):
    def __init__(self, mmr: Any, transport: Any):
        super().__init__(mmr=mmr, transport=transport, goal_id="test")
        self.last_sent_iblt: Optional[bytes] = None
        self.reply_mode: str = "echo"

    async def _exchange_states(self, peer_id: str, local_state: SyncState) -> SyncState:
        raise NotImplementedError

    async def _exchange_iblts(self, peer_id: str, local_iblt_data: bytes) -> bytes:
        self.last_sent_iblt = local_iblt_data
        if self.reply_mode == "echo":
            return local_iblt_data
        tampered = bytearray(local_iblt_data)
        tampered[len(tampered) // 2] ^= 0x01
        return bytes(tampered)

    async def _request_entries_by_hash(self, peer_id: str, hashes: List[bytes]) -> List[bytes]:
        raise NotImplementedError

    async def _get_entries_by_hash(self, hashes: List[bytes]) -> List[bytes]:
        raise NotImplementedError

    async def _send_entries(self, peer_id: str, entries: List[bytes]) -> None:
        raise NotImplementedError

    async def _pull_from_peer(self, session: SyncSession) -> SyncResult:
        raise NotImplementedError

    async def _push_to_peer(self, session: SyncSession) -> SyncResult:
        raise NotImplementedError


class TestFrontierSyncIBLTAuthentication:
    def test_iblt_payload_is_hmac_authenticated_when_session_key_available(self):
        mmr = _DummyMMR([secrets.token_bytes(32) for _ in range(5)])
        transport = _DummyTransport(secrets.token_bytes(32))
        sync = _TestFrontierSync(mmr=mmr, transport=transport)

        async def run():
            result = await sync.sync_with_iblt("peer")
            assert result.status == SyncStatus.SUCCESS
            assert sync.last_sent_iblt is not None
            local_iblt = await sync._build_iblt_from_log()
            unauth_len = len(local_iblt.serialize())
            assert len(sync.last_sent_iblt) == unauth_len + 32

        asyncio.run(run())

    def test_tampered_iblt_fails_closed_when_hmac_enabled(self):
        mmr = _DummyMMR([secrets.token_bytes(32) for _ in range(5)])
        transport = _DummyTransport(secrets.token_bytes(32))
        sync = _TestFrontierSync(mmr=mmr, transport=transport)
        sync.reply_mode = "tamper"

        async def run():
            result = await sync.sync_with_iblt("peer")
            assert result.status == SyncStatus.ERROR
            assert result.error is not None
            assert "HMAC verification failed" in result.error

        asyncio.run(run())
