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
    
    def test_version_too_long(self, validator):
        """Version string exceeding limit is rejected."""
        pack = AgentPack(
            version="v" * (DEFAULT_LIMITS.MAX_PACK_VERSION_LEN + 1),
            parameters=b"test",
        )
        result = validator.validate_agent_pack(pack)
        assert not result.valid
        assert result.field == "version"
    
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
        bucket = TokenBucket(capacity=10, tokens=0.0, refill_rate=100.0)  # Fast refill
        
        time.sleep(0.1)  # Wait for refill
        bucket.refill()
        
        assert bucket.tokens > 0
    
    def test_rate_limiter_allows_initial_burst(self):
        """Rate limiter allows initial burst up to capacity."""
        limiter = RateLimiter(capacity=5, refill_rate=0.001)
        
        for i in range(5):
            allowed, _ = limiter.check("contributor_1")
            assert allowed, f"Should allow burst request {i+1}"
        
        # 6th should be blocked
        allowed, wait = limiter.check("contributor_1")
        assert not allowed
        assert wait > 0
    
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


# =============================================================================
# Proof of Work Tests
# =============================================================================

class TestProofOfWork:
    """Tests for ProofOfWork."""
    
    def test_create_challenge(self):
        """Challenges are random 32-byte values."""
        c1 = ProofOfWork.create_challenge()
        c2 = ProofOfWork.create_challenge()
        
        assert len(c1) == 32
        assert len(c2) == 32
        assert c1 != c2
    
    def test_valid_pow_verifies(self):
        """Valid proof-of-work verifies correctly."""
        # Use low difficulty for fast test
        challenge = b"test_challenge_" + bytes(16)
        pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=100000)
        
        assert pow is not None
        assert pow.verify()
    
    def test_invalid_nonce_fails(self):
        """Invalid nonce fails verification."""
        challenge = b"test_challenge_" + bytes(16)
        pow = ProofOfWork(challenge=challenge, nonce=0, difficulty=20)
        
        # With difficulty 20, nonce=0 is very unlikely to be valid
        # (1 in 2^20 chance)
        assert not pow.verify()
    
    def test_wrong_challenge_fails(self):
        """PoW for wrong challenge fails."""
        challenge1 = b"challenge_1" + bytes(21)
        challenge2 = b"challenge_2" + bytes(21)
        
        pow = ProofOfWork.solve(challenge1, difficulty=8, max_attempts=100000)
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
    
    def test_get_challenge_consistent(self):
        """Same contributor gets same challenge within TTL."""
        sybil = SybilResistance(enabled=True)
        
        c1 = sybil.get_challenge("contributor_1")
        c2 = sybil.get_challenge("contributor_1")
        
        assert c1 == c2
    
    def test_different_contributors_different_challenges(self):
        """Different contributors get different challenges."""
        sybil = SybilResistance(enabled=True)
        
        c1 = sybil.get_challenge("contributor_1")
        c2 = sybil.get_challenge("contributor_2")
        
        assert c1 != c2
    
    def test_verify_valid_pow(self):
        """Valid PoW verifies successfully."""
        sybil = SybilResistance(enabled=True, difficulty=8)
        
        challenge = sybil.get_challenge("test_contributor")
        pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=100000)
        
        assert pow is not None
        assert sybil.verify_pow("test_contributor", pow)
    
    def test_verify_clears_challenge(self):
        """Successful verification clears the challenge."""
        sybil = SybilResistance(enabled=True, difficulty=8)
        
        challenge = sybil.get_challenge("test_contributor")
        pow = ProofOfWork.solve(challenge, difficulty=8, max_attempts=100000)
        
        assert sybil.verify_pow("test_contributor", pow)
        
        # Second verification should fail (challenge consumed)
        assert not sybil.verify_pow("test_contributor", pow)


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
        start = time.monotonic()
        target_ms = 50.0
        
        # Do some quick work
        _ = hashlib.sha256(b"test").digest()
        
        pad_execution_time(target_ms, start)
        
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms >= target_ms * 0.9  # Allow 10% tolerance


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
        return SecureCoordinator(coordinator, limits=limits, enable_pow=False)
    
    def make_contribution(self, goal_id, seed=None):
        import random
        seed = seed or random.randint(0, 2**32)
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
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.VALIDATION_ERROR
    
    def test_get_rate_limit_status(self, secure_coordinator):
        """Can query rate limit status."""
        status = secure_coordinator.get_rate_limit_status("new_contributor")
        
        assert "remaining_tokens" in status
        assert "capacity" in status
        assert "refill_rate" in status
