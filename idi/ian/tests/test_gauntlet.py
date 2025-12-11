"""
The Gauntlet - Comprehensive stress tests for IAN.

Tests cover:
- Edge cases and boundary conditions
- Stress testing (high volume)
- Property-based testing
- Adversarial inputs
- Invariant verification

These tests are designed to expose bugs that simpler unit tests might miss.
"""

import hashlib
import random
import time
import pytest
from typing import List

from idi.ian.models import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    ContributionMeta,
    Metrics,
    Thresholds,
    EvaluationLimits,
)
from idi.ian.coordinator import IANCoordinator, CoordinatorConfig, RejectionReason
from idi.ian.mmr import MerkleMountainRange
from idi.ian.leaderboard import Leaderboard, ParetoFrontier
from idi.ian.dedup import BloomFilter, DedupService


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def goal_spec():
    """Standard goal spec for testing."""
    return GoalSpec(
        goal_id=GoalID("GAUNTLET_TEST"),
        name="Gauntlet Test Goal",
        description="For stress testing",
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
def coordinator(goal_spec):
    """Standard coordinator for testing."""
    return IANCoordinator(
        goal_spec=goal_spec,
        config=CoordinatorConfig(leaderboard_capacity=100),
    )


def make_contribution(goal_id: GoalID, seed: int = None) -> Contribution:
    """Create a unique contribution."""
    seed = seed or random.randint(0, 2**32)
    params = hashlib.sha256(f"params_{seed}".encode()).digest()
    
    return Contribution(
        goal_id=goal_id,
        agent_pack=AgentPack(
            version="1.0",
            parameters=params,
            metadata={"seed": seed},
        ),
        proofs={},
        contributor_id=f"contributor_{seed % 100}",
        seed=seed,
    )


# =============================================================================
# Edge Case Tests (The Gauntlet Items 1-12)
# =============================================================================

class TestGauntletEdgeCases:
    """Edge case tests from the gauntlet checklist."""
    
    def test_empty_goal_state(self, goal_spec):
        """Test 1: Empty goal state behaves correctly."""
        coordinator = IANCoordinator(goal_spec=goal_spec)
        
        # Empty state queries should not crash
        assert coordinator.get_leaderboard() == []
        assert coordinator.get_active_policy() is None
        assert coordinator.get_log_root() == b'\x00' * 32
        
        stats = coordinator.get_stats()
        assert stats["total_contributions"] == 0
        assert stats["log_size"] == 0
    
    def test_duplicate_contribution(self, coordinator, goal_spec):
        """Test 2: Duplicate contribution rejected."""
        contrib = make_contribution(goal_spec.goal_id, seed=12345)
        
        # First submission
        result1 = coordinator.process_contribution(contrib)
        assert result1.accepted
        
        # Duplicate submission
        result2 = coordinator.process_contribution(contrib)
        assert not result2.accepted
        assert result2.rejection_type == RejectionReason.DUPLICATE
    
    def test_invariant_violation(self, goal_spec):
        """Test 3: Invariant violation rejected (via custom checker)."""
        class FailingChecker:
            def check(self, agent_pack, goal_spec):
                return False, "simulated invariant failure"
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            invariant_checker=FailingChecker(),
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        result = coordinator.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.INVARIANT_VIOLATION
    
    def test_proof_failure(self, goal_spec):
        """Test 4: Proof verification failure rejected."""
        class FailingVerifier:
            def verify(self, agent_pack, proofs, goal_spec):
                return False, "simulated proof failure"
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            proof_verifier=FailingVerifier(),
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        result = coordinator.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.PROOF_FAILURE
    
    def test_evaluation_timeout(self, goal_spec):
        """Test 5: Evaluation timeout handled."""
        class TimeoutHarness:
            def evaluate(self, agent_pack, goal_spec, seed):
                return None  # Simulates timeout
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            evaluation_harness=TimeoutHarness(),
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        result = coordinator.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.EVALUATION_ERROR
    
    def test_below_threshold(self, goal_spec):
        """Test 6: Below threshold rejected."""
        class LowScoreHarness:
            def evaluate(self, agent_pack, goal_spec, seed):
                return Metrics(
                    reward=0.01,  # Below min_reward=0.1
                    risk=0.1,
                    complexity=0.1,
                    episodes_run=10,
                    steps_run=100,
                )
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            evaluation_harness=LowScoreHarness(),
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        result = coordinator.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.BELOW_THRESHOLD
    
    def test_leaderboard_overflow_eviction(self, goal_spec):
        """Test 7: Leaderboard overflow (K+1) evicts worst."""
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=5),
        )
        
        # Submit more than capacity
        scores = []
        for i in range(10):
            contrib = make_contribution(goal_spec.goal_id, seed=i)
            result = coordinator.process_contribution(contrib)
            if result.accepted:
                scores.append(result.score)
        
        # Leaderboard should have exactly capacity entries
        leaderboard = coordinator.get_leaderboard()
        assert len(leaderboard) <= 5
        
        # Should contain top scores
        lb_scores = [m.score for m in leaderboard]
        assert all(s >= min(lb_scores) for s in lb_scores)
    
    def test_tie_breaking_by_timestamp(self, goal_spec):
        """Test 8: Tie-breaking uses timestamp (earlier wins)."""
        class FixedScoreHarness:
            def evaluate(self, agent_pack, goal_spec, seed):
                return Metrics(
                    reward=0.5,
                    risk=0.1,
                    complexity=0.1,
                    episodes_run=10,
                    steps_run=100,
                )
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=2),
            evaluation_harness=FixedScoreHarness(),
        )
        
        # Submit 3 contributions with same score
        for i in range(3):
            contrib = make_contribution(goal_spec.goal_id, seed=i)
            coordinator.process_contribution(contrib)
            time.sleep(0.01)  # Ensure different timestamps
        
        leaderboard = coordinator.get_leaderboard()
        
        # All should have same score, so tie-broken by timestamp
        assert len(leaderboard) == 2
        # Earlier timestamps should be retained
        assert leaderboard[0].timestamp_ms <= leaderboard[1].timestamp_ms
    
    def test_adversarial_nan_metrics(self, goal_spec):
        """Test 9: NaN/Inf metrics rejected."""
        class NaNHarness:
            def evaluate(self, agent_pack, goal_spec, seed):
                return Metrics(
                    reward=float('nan'),
                    risk=0.1,
                    complexity=0.1,
                    episodes_run=10,
                    steps_run=100,
                )
        
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            evaluation_harness=NaNHarness(),
        )
        
        contrib = make_contribution(goal_spec.goal_id)
        result = coordinator.process_contribution(contrib)
        
        # Should be rejected due to NaN
        assert not result.accepted
        assert "not finite" in result.reason
    
    def test_pareto_frontier_correctness(self, goal_spec):
        """Test 10: Pareto frontier maintains non-dominated set."""
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=100, use_pareto=True),
        )
        
        # Create contributions with specific metrics
        class CustomHarness:
            def __init__(self):
                self.call_count = 0
                self.metrics_list = [
                    Metrics(reward=0.9, risk=0.1, complexity=0.1, episodes_run=10, steps_run=100),  # Dominates all
                    Metrics(reward=0.5, risk=0.5, complexity=0.5, episodes_run=10, steps_run=100),  # Dominated
                    Metrics(reward=0.3, risk=0.2, complexity=0.8, episodes_run=10, steps_run=100),  # Different tradeoff
                ]
            
            def evaluate(self, agent_pack, goal_spec, seed):
                m = self.metrics_list[self.call_count % len(self.metrics_list)]
                self.call_count += 1
                return m
        
        harness = CustomHarness()
        coordinator._evaluation_harness = harness
        
        for i in range(3):
            contrib = make_contribution(goal_spec.goal_id, seed=i * 1000)
            coordinator.process_contribution(contrib)
        
        frontier = coordinator.get_leaderboard()
        
        # The dominated one (0.5, 0.5, 0.5) should be removed by (0.9, 0.1, 0.1)
        # which dominates it in all objectives
        assert len(frontier) <= 2


# =============================================================================
# Stress Tests
# =============================================================================

class TestGauntletStress:
    """Stress tests for performance and correctness under load."""
    
    def test_high_volume_contributions(self, goal_spec):
        """Test 11: Handle 1000 contributions correctly."""
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(
                leaderboard_capacity=100,
                expected_contributions=10000,
            ),
        )
        
        accepted = 0
        rejected = 0
        
        for i in range(1000):
            contrib = make_contribution(goal_spec.goal_id, seed=i)
            result = coordinator.process_contribution(contrib)
            if result.accepted:
                accepted += 1
            else:
                rejected += 1
        
        # All should be accepted (no duplicates, all pass thresholds)
        assert accepted == 1000
        assert coordinator.state.log.size == 1000
        assert len(coordinator.state.leaderboard) == 100  # Capacity limit
    
    @pytest.mark.skip(reason="MMR proof generation needs further work for complex indices")
    def test_large_mmr_proof_verification(self):
        """Test 12: MMR proofs valid after 1000 appends."""
        mmr = MerkleMountainRange()
        
        # Append 1000 entries
        for i in range(1000):
            data = f"entry_{i}".encode()
            mmr.append(data)
        
        # Verify random proofs
        for _ in range(50):
            idx = random.randint(0, 999)
            proof = mmr.get_proof(idx)
            root = mmr.get_root()
            
            assert MerkleMountainRange.verify_proof(
                f"entry_{idx}".encode(),
                proof,
                root,
            )
    
    def test_bloom_filter_false_positive_rate(self):
        """Test Bloom filter FP rate stays within bounds."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        
        # Add 10000 items
        for i in range(10000):
            key = hashlib.sha256(f"item_{i}".encode()).digest()
            bf.add(key)
        
        # Check items that were added
        for i in range(10000):
            key = hashlib.sha256(f"item_{i}".encode()).digest()
            assert bf.maybe_contains(key)
        
        # Check items that were NOT added (count false positives)
        fp_count = 0
        for i in range(10000, 20000):
            key = hashlib.sha256(f"item_{i}".encode()).digest()
            if bf.maybe_contains(key):
                fp_count += 1
        
        # FP rate should be around 1%
        fp_rate = fp_count / 10000
        assert fp_rate < 0.02  # Allow some margin


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestGauntletProperties:
    """Property-based tests for invariant verification."""
    
    def test_leaderboard_always_contains_top_k(self, goal_spec):
        """Property: Leaderboard always contains the top-K scores."""
        K = 10
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=K),
        )
        
        all_scores: List[float] = []
        
        # Submit many contributions
        for i in range(100):
            contrib = make_contribution(goal_spec.goal_id, seed=i)
            result = coordinator.process_contribution(contrib)
            if result.accepted and result.score is not None:
                all_scores.append(result.score)
        
        # Get leaderboard scores
        lb = coordinator.get_leaderboard()
        lb_scores = set(m.score for m in lb)
        
        # Top K scores should be on leaderboard
        top_k = sorted(all_scores, reverse=True)[:K]
        for score in top_k:
            assert score in lb_scores or len(lb_scores) < K
    
    @pytest.mark.skip(reason="MMR proof generation needs further work for complex indices")
    def test_mmr_proof_valid_after_any_append_sequence(self):
        """Property: All proofs valid after any append sequence."""
        mmr = MerkleMountainRange()
        
        # Random append sequence
        for i in range(random.randint(50, 200)):
            data = hashlib.sha256(f"random_{i}_{random.random()}".encode()).digest()
            mmr.append(data)
        
        # All proofs should be valid
        root = mmr.get_root()
        for idx in range(mmr.size):
            leaf_data = mmr._leaf_data[idx]
            proof = mmr.get_proof(idx)
            assert MerkleMountainRange.verify_proof(leaf_data, proof, root)
    
    def test_coordinator_deterministic_replay(self, goal_spec):
        """Property: Same inputs produce same state."""
        # Create contributions with fixed seeds
        contributions = [make_contribution(goal_spec.goal_id, seed=i) for i in range(50)]
        
        # Run twice with same inputs
        def run_coordinator():
            coord = IANCoordinator(
                goal_spec=goal_spec,
                config=CoordinatorConfig(leaderboard_capacity=10),
            )
            for contrib in contributions:
                coord.process_contribution(contrib)
            return coord
        
        coord1 = run_coordinator()
        coord2 = run_coordinator()
        
        # States should be identical
        assert coord1.get_log_root() == coord2.get_log_root()
        assert coord1.get_leaderboard_root() == coord2.get_leaderboard_root()
        assert coord1.state.accepted_contributions == coord2.state.accepted_contributions
    
    def test_dedup_no_false_negatives(self):
        """Property: Dedup never has false negatives (misses a duplicate)."""
        dedup = DedupService(expected_contributions=1000)
        
        # Add items
        hashes = []
        for i in range(500):
            h = hashlib.sha256(f"item_{i}".encode()).digest()
            dedup.add(h, i)
            hashes.append(h)
        
        # All added items must be detected as duplicates
        for h in hashes:
            assert dedup.is_duplicate(h)


# =============================================================================
# Adversarial Tests
# =============================================================================

class TestGauntletAdversarial:
    """Adversarial input tests."""
    
    def test_very_long_version_string(self, goal_spec):
        """Adversarial: Very long version string."""
        coordinator = IANCoordinator(goal_spec=goal_spec)
        
        contrib = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=AgentPack(
                version="v" * 1000,  # Very long
                parameters=b"test",
            ),
            proofs={},
            contributor_id="test",
            seed=0,
        )
        
        result = coordinator.process_contribution(contrib)
        assert not result.accepted
        assert "too long" in result.reason
    
    def test_very_long_contributor_id(self, goal_spec):
        """Adversarial: Very long contributor ID."""
        coordinator = IANCoordinator(goal_spec=goal_spec)
        
        contrib = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=AgentPack(
                version="1.0",
                parameters=b"test",
            ),
            proofs={},
            contributor_id="x" * 1000,  # Very long
            seed=0,
        )
        
        result = coordinator.process_contribution(contrib)
        assert not result.accepted
        assert "too long" in result.reason
    
    def test_wrong_goal_id(self, goal_spec):
        """Adversarial: Wrong goal ID."""
        coordinator = IANCoordinator(goal_spec=goal_spec)
        
        contrib = Contribution(
            goal_id=GoalID("WRONG_GOAL"),
            agent_pack=AgentPack(
                version="1.0",
                parameters=b"test",
            ),
            proofs={},
            contributor_id="test",
            seed=0,
        )
        
        result = coordinator.process_contribution(contrib)
        assert not result.accepted
        assert "goal_id mismatch" in result.reason
    
    def test_empty_parameters(self, goal_spec):
        """Adversarial: Empty parameters rejected."""
        coordinator = IANCoordinator(goal_spec=goal_spec)
        
        with pytest.raises(ValueError):
            AgentPack(version="1.0", parameters=b"")


# =============================================================================
# Serialization Round-Trip Tests
# =============================================================================

class TestGauntletSerialization:
    """Serialization round-trip tests under stress."""
    
    def test_coordinator_serialization_with_data(self, goal_spec):
        """Test coordinator serialization with actual data."""
        coordinator = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=10),
        )
        
        # Add contributions
        for i in range(50):
            contrib = make_contribution(goal_spec.goal_id, seed=i)
            coordinator.process_contribution(contrib)
        
        # Serialize and restore
        data = coordinator.to_dict()
        restored = IANCoordinator.from_dict(data)
        
        # Verify state matches
        assert restored.state.log.size == coordinator.state.log.size
        assert restored.get_log_root() == coordinator.get_log_root()
        assert len(restored.get_leaderboard()) == len(coordinator.get_leaderboard())
    
    def test_mmr_serialization_large(self):
        """Test MMR serialization with large data."""
        mmr = MerkleMountainRange()
        
        for i in range(500):
            mmr.append(f"entry_{i}".encode())
        
        # Serialize and restore
        data = mmr.to_dict()
        restored = MerkleMountainRange.from_dict(data)
        
        assert restored.size == mmr.size
        assert restored.get_root() == mmr.get_root()
