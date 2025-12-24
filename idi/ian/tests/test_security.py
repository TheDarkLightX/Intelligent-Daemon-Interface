"""
Tests for IAN Security Module.

Tests cover:
- Input validation (size limits, format)
- Rate limiting (token bucket behavior)
- Proof-of-work (challenge/verify)
- Timing attack mitigation
- SecureCoordinator integration
"""

import hashlib
import time
from unittest.mock import patch

import pytest

from idi.ian.models import (
    AgentPack,
    Contribution,
    ContributionMeta,
    EvaluationLimits,
    GoalID,
    GoalSpec,
    Thresholds,
)
from idi.ian.security import (
    DEFAULT_LIMITS,
    InputValidator,
    ProofOfWork,
    RateLimiter,
    SecurityLimits,
    SybilResistance,
    TokenBucket,
    ValidationResult,
    constant_time_compare,
    pad_execution_time,
)


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for InputValidator."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @pytest.fixture
    def valid_pack(self):
        return AgentPack(
            version="1.0.0",
            parameters=b"test_params",
            metadata={"key": "value"},
        )
    
    @pytest.fixture
    def valid_contribution(self, valid_pack):
        return Contribution(
            goal_id=GoalID("TEST_GOAL"),
            agent_pack=valid_pack,
            proofs={},
            contributor_id="test_contributor",
            seed=12345,
        )
    
    def test_valid_pack_passes(self, validator, valid_pack):
        """Valid AgentPack passes validation."""
        result = validator.validate_agent_pack(valid_pack)
        assert result.valid
    
    def test_empty_parameters_rejected(self, validator):
        """Empty parameters are rejected at AgentPack construction."""
        # AgentPack validates in __post_init__, so empty params raise immediately
        with pytest.raises(ValueError, match="cannot be empty"):
            AgentPack(version="1.0", parameters=b"")

    def test_empty_version_rejected(self, validator):
        """Empty version is rejected and reports the correct field."""
        pack = AgentPack(
            version="",
            parameters=b"test",
        )
        result = validator.validate_agent_pack(pack)
        assert not result.valid
        assert result.field == "version"
        assert isinstance(result.error, str)
        assert result.error
    
    def test_version_too_long(self, validator):
        """Version string exceeding limit is rejected."""
        pack = AgentPack(
            version="v" * (DEFAULT_LIMITS.MAX_PACK_VERSION_LEN + 1),
            parameters=b"test",
        )
        result = validator.validate_agent_pack(pack)
        assert not result.valid
        assert result.field == "version"
        assert isinstance(result.error, str)
        assert result.error

    def test_version_at_limit_accepted(self):
        """Version string at size limit is accepted."""
        limits = SecurityLimits(MAX_PACK_VERSION_LEN=5)
        validator = InputValidator(limits)

        pack = AgentPack(
            version="v" * 5,
            parameters=b"test",
        )
        result = validator.validate_agent_pack(pack)
        assert result.valid
    
    def test_parameters_too_large(self, validator):
        """Parameters exceeding size limit are rejected."""
        # Create oversized parameters
        limits = SecurityLimits(MAX_PACK_PARAMETERS_SIZE=100)
        validator = InputValidator(limits)
        
        pack = AgentPack(
            version="1.0",
            parameters=b"x" * 101,
        )
        result = validator.validate_agent_pack(pack)
        assert not result.valid
        assert result.field == "parameters"
        assert isinstance(result.error, str)
        assert result.error

    def test_parameters_at_size_limit_accepted(self):
        """Parameters at size limit are accepted."""
        limits = SecurityLimits(MAX_PACK_PARAMETERS_SIZE=100)
        validator = InputValidator(limits)

        pack = AgentPack(
            version="1.0",
            parameters=b"x" * 100,
        )
        result = validator.validate_agent_pack(pack)
        assert result.valid
    
    def test_too_many_metadata_keys(self, validator):
        """Too many metadata keys are rejected."""
        limits = SecurityLimits(MAX_PACK_METADATA_KEYS=5)
        validator = InputValidator(limits)
        
        pack = AgentPack(
            version="1.0",
            parameters=b"test",
            metadata={f"key_{i}": f"value_{i}" for i in range(10)},
        )
        result = validator.validate_agent_pack(pack)
        assert not result.valid
        assert result.field == "metadata"
        assert isinstance(result.error, str)
        assert result.error

    def test_metadata_key_len_boundary(self):
        """Metadata key length boundary is enforced."""
        limits = SecurityLimits(
            MAX_PACK_METADATA_KEYS=10,
            MAX_PACK_METADATA_KEY_LEN=5,
            MAX_PACK_METADATA_VALUE_SIZE=100,
            MAX_PACK_METADATA_SIZE=10_000,
        )
        validator = InputValidator(limits)

        ok_pack = AgentPack(version="1.0", parameters=b"p", metadata={"a" * 5: "v"})
        ok_result = validator.validate_agent_pack(ok_pack)
        assert ok_result.valid

        bad_key = "a" * 6
        bad_pack = AgentPack(version="1.0", parameters=b"p", metadata={bad_key: "v"})
        bad_result = validator.validate_agent_pack(bad_pack)
        assert not bad_result.valid
        assert bad_result.field is not None
        assert bad_result.field.startswith("metadata.")
        assert isinstance(bad_result.error, str)
        assert bad_result.error

    def test_metadata_value_size_boundary(self):
        """Metadata value size boundary is enforced."""
        limits = SecurityLimits(
            MAX_PACK_METADATA_KEYS=10,
            MAX_PACK_METADATA_KEY_LEN=10,
            MAX_PACK_METADATA_VALUE_SIZE=10,
            MAX_PACK_METADATA_SIZE=10_000,
        )
        validator = InputValidator(limits)

        ok_pack = AgentPack(version="1.0", parameters=b"p", metadata={"k": "v" * 10})
        ok_result = validator.validate_agent_pack(ok_pack)
        assert ok_result.valid

        bad_pack = AgentPack(version="1.0", parameters=b"p", metadata={"k": "v" * 11})
        bad_result = validator.validate_agent_pack(bad_pack)
        assert not bad_result.valid
        assert bad_result.field is not None
        assert bad_result.field.startswith("metadata.")
        assert isinstance(bad_result.error, str)
        assert bad_result.error

    def test_total_metadata_size_boundary(self):
        """Total metadata size boundary is enforced."""
        limits = SecurityLimits(
            MAX_PACK_METADATA_KEYS=10,
            MAX_PACK_METADATA_KEY_LEN=10,
            MAX_PACK_METADATA_VALUE_SIZE=10_000,
            MAX_PACK_METADATA_SIZE=10,
        )
        validator = InputValidator(limits)

        ok_pack = AgentPack(version="1.0", parameters=b"p", metadata={"a": "v" * 9})
        ok_result = validator.validate_agent_pack(ok_pack)
        assert ok_result.valid

        bad_pack = AgentPack(version="1.0", parameters=b"p", metadata={"a": "v" * 10})
        bad_result = validator.validate_agent_pack(bad_pack)
        assert not bad_result.valid
        assert bad_result.field == "metadata"
        assert isinstance(bad_result.error, str)
        assert bad_result.error
    
    def test_valid_contribution_passes(self, validator, valid_contribution):
        """Valid contribution passes validation."""
        result = validator.validate_contribution(valid_contribution)
        assert result.valid
    
    def test_contributor_id_too_long(self, validator, valid_pack):
        """Contributor ID exceeding limit is rejected."""
        contrib = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={},
            contributor_id="x" * (DEFAULT_LIMITS.MAX_CONTRIBUTOR_ID_LEN + 1),
            seed=0,
        )
        result = validator.validate_contribution(contrib)
        assert not result.valid
        assert result.field == "contributor_id"
        assert isinstance(result.error, str)
        assert result.error

    def test_contributor_id_at_limit_accepted(self, valid_pack):
        """Contributor ID at length limit is accepted."""
        limits = SecurityLimits(MAX_CONTRIBUTOR_ID_LEN=5)
        validator = InputValidator(limits)

        contrib = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={},
            contributor_id="a" * 5,
            seed=0,
        )
        result = validator.validate_contribution(contrib)
        assert result.valid

    def test_goal_id_length_boundary(self, valid_pack):
        """Goal ID length boundary is enforced by InputValidator limits."""
        limits = SecurityLimits(MAX_GOAL_ID_LEN=4)
        validator = InputValidator(limits)

        ok_contrib = Contribution(
            goal_id=GoalID("ABCD"),
            agent_pack=valid_pack,
            proofs={},
            contributor_id="test",
            seed=0,
        )
        ok_result = validator.validate_contribution(ok_contrib)
        assert ok_result.valid

        bad_contrib = Contribution(
            goal_id=GoalID("ABCDE"),
            agent_pack=valid_pack,
            proofs={},
            contributor_id="test",
            seed=0,
        )
        bad_result = validator.validate_contribution(bad_contrib)
        assert not bad_result.valid
        assert bad_result.field == "goal_id"
        assert isinstance(bad_result.error, str)
        assert bad_result.error
    
    def test_too_many_proofs(self, validator, valid_pack):
        """Too many proofs are rejected."""
        limits = SecurityLimits(MAX_PROOFS_COUNT=2)
        validator = InputValidator(limits)
        
        contrib = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={f"proof_{i}": b"data" for i in range(5)},
            contributor_id="test",
            seed=0,
        )
        result = validator.validate_contribution(contrib)
        assert not result.valid
        assert result.field == "proofs"
        assert isinstance(result.error, str)
        assert result.error

    def test_proofs_at_count_limit_accepted(self, valid_pack):
        """Proof count at limit is accepted."""
        limits = SecurityLimits(MAX_PROOFS_COUNT=2, MAX_PROOF_SIZE=100, MAX_TOTAL_PROOFS_SIZE=1000)
        validator = InputValidator(limits)

        contrib = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={"a": b"x", "b": b"y"},
            contributor_id="test",
            seed=0,
        )
        result = validator.validate_contribution(contrib)
        assert result.valid

    def test_proof_size_boundary(self, valid_pack):
        """Per-proof size boundary is enforced."""
        limits = SecurityLimits(MAX_PROOFS_COUNT=10, MAX_PROOF_SIZE=5, MAX_TOTAL_PROOFS_SIZE=10_000)
        validator = InputValidator(limits)

        ok = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={"p": b"x" * 5},
            contributor_id="test",
            seed=0,
        )
        ok_result = validator.validate_contribution(ok)
        assert ok_result.valid

        bad = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={"p": b"x" * 6},
            contributor_id="test",
            seed=0,
        )
        bad_result = validator.validate_contribution(bad)
        assert not bad_result.valid
        assert bad_result.field == "proofs.p"
        assert isinstance(bad_result.error, str)
        assert bad_result.error

    def test_total_proofs_size_boundary(self, valid_pack):
        """Total proofs size boundary is enforced."""
        limits = SecurityLimits(MAX_PROOFS_COUNT=10, MAX_PROOF_SIZE=10_000, MAX_TOTAL_PROOFS_SIZE=10)
        validator = InputValidator(limits)

        ok = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={"a": b"x" * 5, "b": b"y" * 5},
            contributor_id="test",
            seed=0,
        )
        ok_result = validator.validate_contribution(ok)
        assert ok_result.valid

        bad = Contribution(
            goal_id=GoalID("TEST"),
            agent_pack=valid_pack,
            proofs={"a": b"x" * 5, "b": b"y" * 6},
            contributor_id="test",
            seed=0,
        )
        bad_result = validator.validate_contribution(bad)
        assert not bad_result.valid
        assert bad_result.field == "proofs"
        assert isinstance(bad_result.error, str)
        assert bad_result.error


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for RateLimiter."""
    
    def test_token_bucket_initial_tokens(self):
        """Token bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10, tokens=10.0, refill_rate=1.0)
        assert bucket.tokens == 10.0
    
    def test_token_bucket_consume(self):
        """Tokens can be consumed."""
        bucket = TokenBucket(capacity=10, tokens=10.0, refill_rate=0.0)  # No refill
        
        assert bucket.try_consume(1)
        assert bucket.tokens == 9.0
        
        assert bucket.try_consume(5)
        assert bucket.tokens == 4.0
    
    def test_token_bucket_empty(self):
        """Cannot consume from empty bucket."""
        bucket = TokenBucket(capacity=10, tokens=0.0, refill_rate=1.0)
        
        assert not bucket.try_consume(1)
    
    def test_token_bucket_refill(self):
        """Tokens refill over time."""
        with patch("idi.ian.security.time.monotonic", return_value=0.1):
            bucket = TokenBucket(
                capacity=10,
                tokens=0.0,
                refill_rate=100.0,
                last_refill=0.0,
            )
            bucket.refill()

            assert bucket.tokens > 0

    def test_token_bucket_time_until_available(self):
        """time_until_available computes wait time when not enough tokens."""
        with patch("idi.ian.security.time.monotonic", return_value=0.0):
            bucket = TokenBucket(capacity=10, tokens=0.0, refill_rate=2.0, last_refill=0.0)
            assert bucket.time_until_available(tokens=1) == pytest.approx(0.5, abs=1e-12)

    def test_token_bucket_time_until_available_infinite_when_no_refill(self):
        with patch("idi.ian.security.time.monotonic", return_value=0.0):
            bucket = TokenBucket(capacity=10, tokens=0.0, refill_rate=0.0, last_refill=0.0)
            assert bucket.time_until_available(tokens=1) == float("inf")

    def test_rate_limiter_allows_initial_burst(self):
        """Rate limiter allows initial burst up to capacity."""
        limiter = RateLimiter(capacity=5, refill_rate=0.001)
        
        for i in range(5):
            allowed, wait = limiter.check("contributor_1")
            assert allowed, f"Should allow burst request {i+1}"
            assert wait == 0.0
        
        # 6th should be blocked
        allowed, wait = limiter.check("contributor_1")
        assert not allowed
        assert wait > 0

    def test_rate_limiter_blocked_wait_is_for_one_token(self):
        """RateLimiter.wait_seconds corresponds to 1 token when blocked."""
        limiter = RateLimiter(capacity=1, refill_rate=2.0)
        limiter._last_cleanup = 0.0
        bucket = limiter._get_bucket("c")
        bucket.tokens = 1.0
        bucket.last_refill = 0.0

        with patch("idi.ian.security.time.monotonic", return_value=0.0):
            allowed1, wait1 = limiter.check("c")
            assert allowed1
            assert wait1 == 0.0

            allowed2, wait2 = limiter.check("c")
            assert not allowed2
            assert wait2 == pytest.approx(0.5, abs=1e-12)
    
    def test_rate_limiter_per_contributor(self):
        """Rate limits are per-contributor."""
        limiter = RateLimiter(capacity=2, refill_rate=0.001)
        
        # Exhaust contributor_1
        limiter.check("contributor_1")
        limiter.check("contributor_1")
        allowed1, _ = limiter.check("contributor_1")
        
        # contributor_2 should still be allowed
        allowed2, _ = limiter.check("contributor_2")
        
        assert not allowed1
        assert allowed2
    
    def test_rate_limiter_remaining_tokens(self):
        """Can query remaining tokens."""
        limiter = RateLimiter(capacity=10, refill_rate=0.001)
        
        assert limiter.get_remaining("new_contributor") == 10
        
        limiter.check("new_contributor")
        assert limiter.get_remaining("new_contributor") == 9

    def test_rate_limiter_enforces_max_buckets(self):
        limiter = RateLimiter(capacity=1, refill_rate=0.0, max_buckets=1)
        limiter.check("c1")
        limiter.check("c2")
        assert len(limiter._buckets) == 1

    def test_rate_limiter_evicts_least_recently_refilled_bucket(self):
        limiter = RateLimiter(capacity=1, refill_rate=0.0, max_buckets=2)

        b_old = limiter._get_bucket("old")
        b_new = limiter._get_bucket("new")
        b_old.last_refill = 1.0
        b_new.last_refill = 2.0

        _ = limiter._get_bucket("third")

        assert "old" not in limiter._buckets
        assert "new" in limiter._buckets
        assert "third" in limiter._buckets

    def test_rate_limiter_cleanup_does_not_run_before_interval(self):
        limiter = RateLimiter(capacity=1, refill_rate=0.0, max_buckets=10)
        limiter._cleanup_interval = 100.0
        limiter._last_cleanup = 80.0

        stale = limiter._get_bucket("stale")
        stale.tokens = 1.0
        stale.last_refill = -1000.0

        with patch("idi.ian.security.time.monotonic", return_value=30.0):
            _allowed, _wait = limiter.check("new")

        assert "stale" in limiter._buckets

    def test_rate_limiter_cleanup_runs_after_interval(self):
        """Cleanup runs when now - last_cleanup > cleanup_interval."""
        limiter = RateLimiter(capacity=10, refill_rate=0.0, max_buckets=10)
        limiter._cleanup_interval = 100.0
        limiter._last_cleanup = 0.0

        # Create a stale bucket (idle + full)
        stale = limiter._get_bucket("stale")
        stale.tokens = 10.0  # full (>= capacity)
        stale.last_refill = 0.0  # old

        # Create a fresh bucket
        fresh = limiter._get_bucket("fresh")
        fresh.tokens = 10.0
        fresh.last_refill = 200.0  # recent

        # Trigger cleanup at t=200.1 (200.1 - 0.0 > 100.0)
        with patch("idi.ian.security.time.monotonic", return_value=200.1):
            limiter._maybe_cleanup()

        # Stale bucket should be removed (idle > interval AND tokens >= capacity)
        assert "stale" not in limiter._buckets
        # Fresh bucket should remain
        assert "fresh" in limiter._buckets

    def test_rate_limiter_cleanup_requires_both_idle_and_full(self):
        """Cleanup only removes buckets that are BOTH idle AND full (not OR)."""
        limiter = RateLimiter(capacity=10, refill_rate=0.0, max_buckets=10)
        limiter._cleanup_interval = 100.0
        limiter._last_cleanup = 0.0

        # Bucket that is idle but NOT full (partially consumed)
        idle_not_full = limiter._get_bucket("idle_not_full")
        idle_not_full.tokens = 5.0  # not full
        idle_not_full.last_refill = 0.0  # idle

        # Bucket that is full but NOT idle (recently used)
        full_not_idle = limiter._get_bucket("full_not_idle")
        full_not_idle.tokens = 10.0  # full
        full_not_idle.last_refill = 150.0  # not idle

        with patch("idi.ian.security.time.monotonic", return_value=200.1):
            limiter._maybe_cleanup()

        # Neither should be removed because cleanup requires AND not OR
        assert "idle_not_full" in limiter._buckets
        assert "full_not_idle" in limiter._buckets

    def test_rate_limiter_cleanup_updates_last_cleanup(self):
        """After cleanup runs, _last_cleanup is updated to now."""
        limiter = RateLimiter(capacity=10, refill_rate=0.0, max_buckets=10)
        limiter._cleanup_interval = 100.0
        limiter._last_cleanup = 0.0

        with patch("idi.ian.security.time.monotonic", return_value=500.0):
            limiter._maybe_cleanup()

        assert limiter._last_cleanup == 500.0


# =============================================================================
# Proof of Work Tests
# =============================================================================

class TestProofOfWork:
    """Tests for ProofOfWork."""
    
    def test_create_challenge(self):
        """Challenges are random 32-byte values."""
        with patch("idi.ian.security.secrets.token_bytes", side_effect=[b"\x01" * 32, b"\x02" * 32]):
            c1 = ProofOfWork.create_challenge()
            c2 = ProofOfWork.create_challenge()

            assert len(c1) == 32
            assert len(c2) == 32
            assert c1 != c2
    
    def test_valid_pow_verifies(self):
        """Valid proof-of-work verifies correctly."""
        challenge = b"test_challenge_" + bytes(16)

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
            pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)

            assert pow is not None
            assert pow.nonce == 0
            assert pow.verify()
    
    def test_invalid_nonce_fails(self):
        """Invalid nonce fails verification."""
        challenge = b"test_challenge_" + bytes(16)
        with patch("idi.ian.security.hashlib.sha256", side_effect=lambda _: type("_H", (), {"digest": lambda self: b"\xff" * 32})()):
            pow = ProofOfWork(challenge=challenge, nonce=0, difficulty=20)
            
            # With difficulty 20, nonce=0 is very unlikely to be valid
            # (1 in 2^20 chance)
            assert not pow.verify()
    
    def test_wrong_challenge_fails(self):
        """PoW for wrong challenge fails."""
        challenge1 = b"challenge_1" + bytes(21)
        challenge2 = b"challenge_2" + bytes(21)

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge1) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
            pow = ProofOfWork.solve(challenge1, difficulty=8, max_attempts=2)
            assert pow is not None

            # Change challenge
            wrong_pow = ProofOfWork(challenge=challenge2, nonce=pow.nonce, difficulty=8)
            assert not wrong_pow.verify()


class TestSybilResistance:
    """Tests for SybilResistance."""
    
    def test_disabled_by_default(self):
        """Sybil resistance is disabled by default."""
        sybil = SybilResistance(enabled=False)
        assert not sybil.is_enabled()

    def test_verify_pow_allows_when_disabled(self):
        """When disabled, verify_pow always allows without requiring a challenge."""
        sybil = SybilResistance(enabled=False)
        proof = ProofOfWork(challenge=b"x" * 32, nonce=0, difficulty=0)
        assert sybil.verify_pow("contributor", proof)
    
    def test_get_challenge_consistent(self):
        """Same contributor gets same challenge within TTL."""
        sybil = SybilResistance(enabled=True)
        
        c1 = sybil.get_challenge("contributor_1")
        c2 = sybil.get_challenge("contributor_1")
        
        assert c1 == c2
    
    def test_different_contributors_different_challenges(self):
        """Different contributors get different challenges."""
        with patch("idi.ian.security.secrets.token_bytes", side_effect=[b"\x01" * 32, b"\x02" * 32]):
            sybil = SybilResistance(enabled=True)

            c1 = sybil.get_challenge("contributor_1")
            c2 = sybil.get_challenge("contributor_2")

            assert c1 != c2

    def test_verify_pow_rejects_at_expiry_boundary(self):
        challenge_bytes = b"\x11" * 32
        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=0)

            with patch("idi.ian.security.time.time", return_value=0.0):
                challenge = sybil.get_challenge("c")

            proof = ProofOfWork(challenge=challenge, nonce=0, difficulty=0)

            with patch("idi.ian.security.time.time", return_value=300.0):
                assert sybil.verify_pow("c", proof) is False

            assert "c" not in sybil._pending_challenges

    def test_get_challenge_enforces_max_challenges_by_evicting_oldest(self):
        c1_bytes = b"\x01" * 32
        c2_bytes = b"\x02" * 32
        sybil = SybilResistance(enabled=True, difficulty=0, max_challenges=1)

        with patch("idi.ian.security.time.time", side_effect=[0.0, 1.0]), patch(
            "idi.ian.security.secrets.token_bytes", side_effect=[c1_bytes, c2_bytes]
        ):
            c1 = sybil.get_challenge("c1")
            c2 = sybil.get_challenge("c2")

        assert c1 != c2
        assert len(sybil._pending_challenges) == 1
        assert "c2" in sybil._pending_challenges
    
    def test_verify_valid_pow(self):
        """Valid PoW verifies successfully."""

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        challenge_bytes = b"\x11" * 32

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge_bytes) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes), patch(
            "idi.ian.security.hashlib.sha256", side_effect=fake_sha256
        ):
            sybil = SybilResistance(enabled=True, difficulty=8)

            challenge = sybil.get_challenge("test_contributor")
            pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)

            assert pow is not None
            assert sybil.verify_pow("test_contributor", pow)
    
    def test_verify_clears_challenge(self):
        """Successful verification clears the challenge."""

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        challenge_bytes = b"\x11" * 32

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge_bytes) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes), patch(
            "idi.ian.security.hashlib.sha256", side_effect=fake_sha256
        ):
            sybil = SybilResistance(enabled=True, difficulty=8)

            challenge = sybil.get_challenge("test_contributor")
            pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)

            assert sybil.verify_pow("test_contributor", pow)

            # Second verification should fail (challenge consumed)
            assert not sybil.verify_pow("test_contributor", pow)

    def test_verify_pow_rejects_wrong_challenge(self):
        """verify_pow returns False when proof.challenge doesn't match stored challenge."""
        challenge_bytes = b"\xAA" * 32
        wrong_bytes = b"\xBB" * 32
        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=0)
            _ = sybil.get_challenge("c")

            # Proof with wrong challenge
            wrong_proof = ProofOfWork(challenge=wrong_bytes, nonce=0, difficulty=0)
            assert sybil.verify_pow("c", wrong_proof) is False

    def test_verify_pow_rejects_insufficient_difficulty(self):
        """verify_pow returns False when proof.difficulty < required difficulty."""
        challenge_bytes = b"\xCC" * 32
        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=8)
            challenge = sybil.get_challenge("c")

            # Proof with difficulty less than required
            low_diff_proof = ProofOfWork(challenge=challenge, nonce=0, difficulty=7)
            assert sybil.verify_pow("c", low_diff_proof) is False

    def test_verify_pow_accepts_exact_difficulty(self):
        """verify_pow accepts proof with exactly the required difficulty."""

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        challenge_bytes = b"\xDD" * 32

        def fake_sha256(data: bytes):
            # Return hash with exactly 8 leading zero bits for nonce=0
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge_bytes) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes), patch(
            "idi.ian.security.hashlib.sha256", side_effect=fake_sha256
        ):
            sybil = SybilResistance(enabled=True, difficulty=8)
            challenge = sybil.get_challenge("c")

            pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)
            assert pow is not None
            assert sybil.verify_pow("c", pow) is True

    def test_verify_pow_rejects_failed_verify(self):
        """verify_pow returns False when proof.verify() returns False."""
        challenge_bytes = b"\xEE" * 32
        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=8)
            challenge = sybil.get_challenge("c")

            # Proof with correct challenge and difficulty but will fail verify()
            # (hash won't have enough leading zeros)
            bad_proof = ProofOfWork(challenge=challenge, nonce=999999, difficulty=8)
            assert sybil.verify_pow("c", bad_proof) is False

    def test_verify_pow_rejects_when_expired_after_cleanup(self):
        """verify_pow returns False when challenge expires between cleanup and check.

        This tests the race condition where _cleanup_expired sees t < expiry,
        but the subsequent time.time() > expiry check sees t > expiry.
        """
        challenge_bytes = b"\xFF" * 32

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge_bytes) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=8)

            # Create challenge at t=0, expires at t=300
            with patch("idi.ian.security.time.time", return_value=0.0):
                challenge = sybil.get_challenge("c")

            # Create valid proof
            with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
                pow_proof = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)
            assert pow_proof is not None

            # Simulate time passing during verify_pow:
            # - First call (cleanup_expired): t=299.9 (before expiry)
            # - Second call (expiry check): t=300.1 (after expiry)
            time_values = iter([299.9, 300.1])
            with patch("idi.ian.security.time.time", side_effect=lambda: next(time_values)):
                with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
                    result = sybil.verify_pow("c", pow_proof)

            # Should return False because challenge is expired
            assert result is False

    def test_verify_pow_accepts_just_before_expiry(self):
        """verify_pow accepts valid proof just before expiry (t < expiry)."""
        challenge_bytes = b"\x99" * 32

        class _H:
            def __init__(self, digest_bytes: bytes) -> None:
                self._digest_bytes = digest_bytes

            def digest(self) -> bytes:
                return self._digest_bytes

        def fake_sha256(data: bytes):
            nonce = int.from_bytes(data[-8:], "big")
            if data.startswith(challenge_bytes) and nonce == 0:
                return _H(b"\x00" + (b"\xff" * 31))
            return _H(b"\xff" * 32)

        with patch("idi.ian.security.secrets.token_bytes", return_value=challenge_bytes):
            sybil = SybilResistance(enabled=True, difficulty=8)

            with patch("idi.ian.security.time.time", return_value=0.0):
                challenge = sybil.get_challenge("c")

            with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
                pow_proof = ProofOfWork.solve(challenge, difficulty=8, max_attempts=2)
            assert pow_proof is not None

            # At t=299.9 (just before expiry of 300), should accept
            with patch("idi.ian.security.time.time", return_value=299.9):
                with patch("idi.ian.security.hashlib.sha256", side_effect=fake_sha256):
                    result = sybil.verify_pow("c", pow_proof)

            assert result is True


# =============================================================================
# Timing Attack Mitigation Tests
# =============================================================================

class TestTimingMitigation:
    """Tests for timing attack mitigation."""
    
    def test_constant_time_compare_equal(self):
        """Equal values compare as equal."""
        a = b"secret_value"
        b = b"secret_value"
        assert constant_time_compare(a, b)
    
    def test_constant_time_compare_unequal(self):
        """Unequal values compare as unequal."""
        a = b"secret_value_1"
        b = b"secret_value_2"
        assert not constant_time_compare(a, b)
    
    def test_pad_execution_time(self):
        """Execution is padded to target time."""
        target_ms = 50.0
        start = 0.0

        with patch("idi.ian.security.time.monotonic", return_value=0.0), patch(
            "idi.ian.security.time.sleep"
        ) as sleep:
            _ = hashlib.sha256(b"test").digest()
            pad_execution_time(target_ms, start)

            sleep.assert_called_once()
            assert sleep.call_args[0][0] == pytest.approx(target_ms / 1000.0, abs=1e-12)

    def test_pad_execution_time_accounts_for_elapsed(self):
        """Padding subtracts elapsed time in milliseconds before sleeping."""
        target_ms = 150.0
        start = 0.9

        with patch("idi.ian.security.time.monotonic", return_value=1.0), patch(
            "idi.ian.security.time.sleep"
        ) as sleep:
            pad_execution_time(target_ms, start)

            sleep.assert_called_once()
            assert sleep.call_args[0][0] == pytest.approx(0.05, abs=1e-12)

    def test_pad_execution_time_no_sleep_when_exact(self):
        """No sleep is performed when elapsed time equals target."""
        target_ms = 125.0
        start = 1.0

        with patch("idi.ian.security.time.monotonic", return_value=1.125), patch(
            "idi.ian.security.time.sleep"
        ) as sleep:
            pad_execution_time(target_ms, start)
            sleep.assert_not_called()


# =============================================================================
# SecureCoordinator Integration Tests
# =============================================================================

class TestSecureCoordinator:
    """Integration tests for SecureCoordinator."""
    
    @pytest.fixture
    def goal_spec(self):
        return GoalSpec(
            goal_id=GoalID("SECURE_TEST"),
            name="Secure Test Goal",
            description="For security testing",
            eval_limits=EvaluationLimits(
                max_episodes=10,
                max_steps_per_episode=100,
                timeout_seconds=10,
                max_memory_mb=256,
            ),
            thresholds=Thresholds(
                min_reward=0.1,
                max_risk=0.9,
                max_complexity=0.9,
            ),
        )
    
    @pytest.fixture
    def coordinator(self, goal_spec):
        from idi.ian.coordinator import IANCoordinator, CoordinatorConfig
        return IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=10),
        )
    
    @pytest.fixture
    def secure_coordinator(self, coordinator):
        from idi.ian.security import SecureCoordinator, SecurityLimits
        limits = SecurityLimits(
            RATE_LIMIT_TOKENS=5,
            RATE_LIMIT_REFILL_PER_SECOND=0.001,  # Very slow refill for testing
        )
        secure = SecureCoordinator(coordinator, limits=limits, enable_pow=False)
        secure._min_process_time_ms = 0.0
        return secure
    
    def make_contribution(self, goal_id, seed=None):
        seed = 0 if seed is None else seed
        return Contribution(
            goal_id=goal_id,
            agent_pack=AgentPack(
                version="1.0",
                parameters=hashlib.sha256(f"params_{seed}".encode()).digest(),
            ),
            proofs={},
            contributor_id=f"contributor_{seed % 10}",
            seed=seed,
        )
    
    def test_valid_contribution_accepted(self, secure_coordinator, goal_spec):
        """Valid contribution is accepted."""
        contrib = self.make_contribution(goal_spec.goal_id, seed=1)
        result = secure_coordinator.process_contribution(contrib)
        assert result.accepted
    
    def test_rate_limiting_enforced(self, secure_coordinator, goal_spec):
        """Rate limiting is enforced."""
        from idi.ian.coordinator import RejectionReason
        
        # Same contributor, exhaust rate limit
        for i in range(5):
            contrib = self.make_contribution(goal_spec.goal_id, seed=i)
            contrib = Contribution(
                goal_id=contrib.goal_id,
                agent_pack=contrib.agent_pack,
                proofs=contrib.proofs,
                contributor_id="rate_limit_test",  # Same contributor
                seed=contrib.seed,
            )
            secure_coordinator.process_contribution(contrib)
        
        # Next should be rate limited
        contrib = self.make_contribution(goal_spec.goal_id, seed=100)
        contrib = Contribution(
            goal_id=contrib.goal_id,
            agent_pack=contrib.agent_pack,
            proofs=contrib.proofs,
            contributor_id="rate_limit_test",
            seed=contrib.seed,
        )
        result = secure_coordinator.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.RATE_LIMITED
        assert isinstance(result.reason, str)
        assert result.reason

    def test_rate_limiting_uses_contributor_id(self, goal_spec):
        from idi.ian.coordinator import ProcessResult, RejectionReason
        from idi.ian.security import SecureCoordinator, SecurityLimits

        class _Coordinator:
            def process_contribution(self, _contrib):
                return ProcessResult(accepted=True, reason="accepted")

        limits = SecurityLimits(
            RATE_LIMIT_TOKENS=1,
            RATE_LIMIT_REFILL_PER_SECOND=0.001,
        )
        secure = SecureCoordinator(_Coordinator(), limits=limits, enable_pow=False)
        secure._min_process_time_ms = 0.0

        c1 = self.make_contribution(goal_spec.goal_id, seed=1)
        c2 = self.make_contribution(goal_spec.goal_id, seed=2)
        c2 = Contribution(
            goal_id=c2.goal_id,
            agent_pack=c2.agent_pack,
            proofs=c2.proofs,
            contributor_id="other_contributor",
            seed=c2.seed,
        )

        r1 = secure.process_contribution(c1)
        assert r1.accepted is True

        r2 = secure.process_contribution(c1)
        assert r2.accepted is False
        assert r2.rejection_type == RejectionReason.RATE_LIMITED
        assert isinstance(r2.reason, str)
        assert r2.reason

        r3 = secure.process_contribution(c2)
        assert r3.accepted is True

    def test_pow_required_when_enabled_and_missing(self, goal_spec):
        from idi.ian.coordinator import ProcessResult, RejectionReason
        from idi.ian.security import SecureCoordinator, SecurityLimits

        class _Coordinator:
            def process_contribution(self, _contrib):
                return ProcessResult(accepted=True, reason="accepted")

        limits = SecurityLimits(
            RATE_LIMIT_TOKENS=100,
            RATE_LIMIT_REFILL_PER_SECOND=100.0,
            POW_DIFFICULTY=8,
        )
        secure = SecureCoordinator(_Coordinator(), limits=limits, enable_pow=True)
        secure._min_process_time_ms = 0.0

        contrib = self.make_contribution(goal_spec.goal_id, seed=1)
        with patch("idi.ian.security.secrets.token_bytes", return_value=b"\x11" * 32):
            result = secure.process_contribution(contrib, proof_of_work=None)

        assert result.accepted is False
        assert result.rejection_type == RejectionReason.POW_REQUIRED
        assert isinstance(result.reason, str)
        assert result.reason
        assert contrib.contributor_id in secure._sybil._pending_challenges

    def test_pow_invalid_when_enabled_and_bad_proof(self, goal_spec):
        from idi.ian.coordinator import ProcessResult, RejectionReason
        from idi.ian.security import SecureCoordinator, SecurityLimits

        class _Coordinator:
            def process_contribution(self, _contrib):
                return ProcessResult(accepted=True, reason="accepted")

        limits = SecurityLimits(
            RATE_LIMIT_TOKENS=100,
            RATE_LIMIT_REFILL_PER_SECOND=100.0,
            POW_DIFFICULTY=8,
        )
        secure = SecureCoordinator(_Coordinator(), limits=limits, enable_pow=True)
        secure._min_process_time_ms = 0.0

        contrib = self.make_contribution(goal_spec.goal_id, seed=1)
        proof = ProofOfWork(challenge=b"x" * 32, nonce=0, difficulty=8)

        def verify_pow(contributor_id: str, pow_proof: ProofOfWork) -> bool:
            assert contributor_id == contrib.contributor_id
            assert pow_proof is proof
            return False

        with patch.object(secure._sybil, "verify_pow", side_effect=verify_pow):
            result = secure.process_contribution(contrib, proof_of_work=proof)

        assert result.accepted is False
        assert result.rejection_type == RejectionReason.POW_INVALID
        assert isinstance(result.reason, str)
        assert result.reason
    
    def test_validation_error_rejected(self, secure_coordinator, goal_spec):
        """Invalid contribution is rejected with validation error."""
        from idi.ian.coordinator import RejectionReason
        
        contrib = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=AgentPack(
                version="v" * 1000,  # Too long
                parameters=b"test",
            ),
            proofs={},
            contributor_id="test",
            seed=0,
        )
        
        result = secure_coordinator.process_contribution(contrib)
        
        assert result.accepted is False
        assert result.rejection_type == RejectionReason.VALIDATION_ERROR
        assert isinstance(result.reason, str)
        assert result.reason
    
    def test_get_rate_limit_status(self, secure_coordinator):
        """Can query rate limit status."""
        status = secure_coordinator.get_rate_limit_status("new_contributor")
        
        assert "remaining_tokens" in status
        assert "capacity" in status
        assert "refill_rate" in status
