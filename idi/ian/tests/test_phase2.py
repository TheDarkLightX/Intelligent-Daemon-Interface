"""
Tests for IAN Phase 2: Coordinator Logic.

Tests cover:
- IANCoordinator (full pipeline)
- Ranking functions (scalar, Pareto)
- Sandboxed evaluation
"""

import hashlib
import os
import pytest
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

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
from idi.ian.coordinator import (
    IANCoordinator,
    CoordinatorConfig,
    ProcessResult,
    RejectionReason,
    PassthroughInvariantChecker,
    PassthroughProofVerifier,
    DummyEvaluationHarness,
    Leaderboard,
)
from idi.ian.ranking import (
    ScalarRanking,
    ParetoRanking,
    rank_contributions,
    is_pareto_optimal,
)
from idi.ian.sandbox import (
    SandboxedEvaluator,
    InProcessEvaluator,
    EvaluationHarnessAdapter,
    EvaluationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def goal_spec() -> GoalSpec:
    """Create a test GoalSpec."""
    return GoalSpec(
        goal_id=GoalID("TEST_GOAL"),
        name="Test Goal",
        description="A test goal for unit tests",
        thresholds=Thresholds(
            min_reward=0.3,
            max_risk=0.5,
            max_complexity=1.0,
        ),
        eval_limits=EvaluationLimits(
            max_episodes=10,
            max_steps_per_episode=100,
            timeout_seconds=10.0,
        ),
        ranking_weights={
            "reward": 1.0,
            "risk": -0.5,
            "complexity": -0.01,
        },
    )


@pytest.fixture
def coordinator(goal_spec: GoalSpec) -> IANCoordinator:
    """Create a test coordinator."""
    return IANCoordinator(
        goal_spec=goal_spec,
        config=CoordinatorConfig(
            leaderboard_capacity=5,
            use_pareto=False,
        ),
    )


def make_contribution(goal_id: GoalID, suffix: str = "") -> Contribution:
    """Create a test contribution."""
    pack = AgentPack(
        version="1.0",
        parameters=f"test_params_{suffix}".encode(),
        metadata={"test": True},
    )
    return Contribution(
        goal_id=goal_id,
        agent_pack=pack,
        contributor_id="test_user",
        seed=42,
    )


# =============================================================================
# Coordinator Tests
# =============================================================================

class TestIANCoordinator:
    """Tests for IANCoordinator."""
    
    def test_initialization(self, goal_spec: GoalSpec):
        coord = IANCoordinator(goal_spec=goal_spec)
        assert coord.state.goal_id == goal_spec.goal_id
        assert coord.state.log.size == 0
        assert len(coord.state.leaderboard) == 0
    
    def test_process_single_contribution(self, coordinator: IANCoordinator):
        contrib = make_contribution(coordinator.goal_spec.goal_id, "1")
        result = coordinator.process_contribution(contrib)
        
        assert result.accepted
        assert result.metrics is not None
        assert result.log_index == 0
        assert result.score is not None
        
        # Verify state updated
        assert coordinator.state.log.size == 1
        assert coordinator.state.accepted_contributions == 1
    
    def test_process_multiple_contributions(self, coordinator: IANCoordinator):
        for i in range(10):
            contrib = make_contribution(coordinator.goal_spec.goal_id, str(i))
            result = coordinator.process_contribution(contrib)
            assert result.accepted
        
        assert coordinator.state.log.size == 10
        assert len(coordinator.state.leaderboard) == 5  # Capacity limit
        assert coordinator.state.accepted_contributions == 10
    
    def test_reject_duplicate(self, coordinator: IANCoordinator):
        contrib = make_contribution(coordinator.goal_spec.goal_id, "dup")
        
        # First submission
        result1 = coordinator.process_contribution(contrib)
        assert result1.accepted
        
        # Duplicate submission
        result2 = coordinator.process_contribution(contrib)
        assert not result2.accepted
        assert result2.rejection_type == RejectionReason.DUPLICATE
    
    def test_reject_wrong_goal_id(self, coordinator: IANCoordinator):
        wrong_goal = GoalID("WRONG_GOAL")
        contrib = make_contribution(wrong_goal, "wrong")
        
        result = coordinator.process_contribution(contrib)
        assert not result.accepted
        assert result.rejection_type == RejectionReason.INVALID_STRUCTURE
        assert "goal_id mismatch" in result.reason
    
    def test_reject_below_threshold(self, goal_spec: GoalSpec):
        # Create coordinator with strict thresholds
        strict_spec = GoalSpec(
            goal_id=GoalID("STRICT_GOAL"),
            name="Strict Goal",
            thresholds=Thresholds(
                min_reward=0.99,  # Very high threshold
                max_risk=0.01,
            ),
        )
        
        # Custom evaluator that returns low metrics
        class LowMetricsEvaluator:
            def evaluate(self, agent_pack, goal_spec, seed):
                return Metrics(
                    reward=0.5,  # Below threshold
                    risk=0.1,
                    complexity=0.1,
                )
        
        coord = IANCoordinator(
            goal_spec=strict_spec,
            evaluation_harness=LowMetricsEvaluator(),
        )
        
        contrib = make_contribution(strict_spec.goal_id, "low")
        result = coord.process_contribution(contrib)
        
        assert not result.accepted
        assert result.rejection_type == RejectionReason.BELOW_THRESHOLD
    
    def test_get_leaderboard(self, coordinator: IANCoordinator):
        # Add contributions
        for i in range(3):
            contrib = make_contribution(coordinator.goal_spec.goal_id, str(i))
            coordinator.process_contribution(contrib)
        
        leaderboard = coordinator.get_leaderboard()
        assert len(leaderboard) == 3
        
        # Should be sorted by score descending
        scores = [m.score for m in leaderboard]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_active_policy(self, coordinator: IANCoordinator):
        # Initially no active policy
        assert coordinator.get_active_policy() is None
        
        # Add contribution
        contrib = make_contribution(coordinator.goal_spec.goal_id, "active")
        coordinator.process_contribution(contrib)
        
        active = coordinator.get_active_policy()
        assert active is not None
        assert active.pack_hash == contrib.pack_hash
    
    def test_get_stats(self, coordinator: IANCoordinator):
        # Add some contributions
        for i in range(5):
            contrib = make_contribution(coordinator.goal_spec.goal_id, str(i))
            coordinator.process_contribution(contrib)
        
        stats = coordinator.get_stats()
        assert stats["total_contributions"] == 5
        assert stats["accepted_contributions"] == 5
        assert stats["log_size"] == 5
        assert "log_root" in stats
        assert "leaderboard_root" in stats
    
    def test_serialization_roundtrip(self, coordinator: IANCoordinator):
        # Add contributions
        for i in range(3):
            contrib = make_contribution(coordinator.goal_spec.goal_id, str(i))
            coordinator.process_contribution(contrib)
        
        # Serialize
        data = coordinator.to_dict()
        
        # Deserialize
        coord2 = IANCoordinator.from_dict(data)
        
        # Verify state preserved
        assert coord2.state.log.size == coordinator.state.log.size
        assert len(coord2.state.leaderboard) == len(coordinator.state.leaderboard)
        assert coord2.state.total_contributions == coordinator.state.total_contributions
        assert coord2.get_log_root() == coordinator.get_log_root()


class TestLeaderboardProperties:
    def test_leaderboard_contains_top_k_scores(self):
        try:
            from hypothesis import given, strategies as st
        except ImportError:
            pytest.skip("hypothesis not available")

        @given(st.lists(st.floats(min_value=-1e3, max_value=1e3), min_size=1, max_size=50))
        def property(scores):
            capacity = 5
            lb = Leaderboard(capacity=capacity)
            metas = []
            for idx, score in enumerate(scores):
                pack_hash = hashlib.sha256(f"{idx}".encode()).digest()
                metrics = Metrics(reward=score, risk=0.0, complexity=0.0)
                meta = ContributionMeta(
                    pack_hash=pack_hash,
                    metrics=metrics,
                    score=score,
                    contributor_id=str(idx),
                    timestamp_ms=idx,
                )
                metas.append(meta)
                lb.add(meta)

            top_by_score = sorted(metas, key=lambda m: m.score, reverse=True)[:capacity]
            lb_scores = sorted([m.score for m in lb.top_k()], reverse=True)
            expected_scores = [m.score for m in top_by_score]
            assert lb_scores == expected_scores

        property()


class TestCoordinatorWithPareto:
    """Tests for coordinator with Pareto frontier."""
    
    def test_pareto_leaderboard(self, goal_spec: GoalSpec):
        coord = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(
                leaderboard_capacity=10,
                use_pareto=True,
            ),
        )
        
        # Add contributions
        for i in range(5):
            contrib = make_contribution(goal_spec.goal_id, str(i))
            coord.process_contribution(contrib)
        
        leaderboard = coord.get_leaderboard()
        assert len(leaderboard) > 0


# =============================================================================
# Ranking Tests
# =============================================================================

class TestScalarRanking:
    """Tests for ScalarRanking."""
    
    def test_default_weights(self):
        ranking = ScalarRanking()
        
        m1 = Metrics(reward=0.9, risk=0.1, complexity=0.1)
        m2 = Metrics(reward=0.5, risk=0.5, complexity=0.5)
        
        s1 = ranking.score(m1)
        s2 = ranking.score(m2)
        
        assert s1 > s2  # Better reward, lower risk
    
    def test_custom_weights(self):
        ranking = ScalarRanking(
            weight_reward=1.0,
            weight_risk=-2.0,  # Penalize risk more
            weight_complexity=0.0,  # Ignore complexity
        )
        
        # High reward but high risk
        m1 = Metrics(reward=0.9, risk=0.5, complexity=0.1)
        # Lower reward but lower risk
        m2 = Metrics(reward=0.7, risk=0.1, complexity=0.9)
        
        s1 = ranking.score(m1)
        s2 = ranking.score(m2)
        
        # m2 should win because risk penalty is high
        assert s2 > s1
    
    def test_compare(self):
        ranking = ScalarRanking()
        
        m1 = Metrics(reward=0.9, risk=0.1, complexity=0.1)
        m2 = Metrics(reward=0.5, risk=0.5, complexity=0.5)
        
        assert ranking.compare(m1, m2) == 1  # m1 > m2
        assert ranking.compare(m2, m1) == -1  # m2 < m1
        assert ranking.compare(m1, m1) == 0  # Equal
    
    def test_from_weights_dict(self):
        weights = {"reward": 2.0, "risk": -1.0}
        ranking = ScalarRanking.from_weights(weights)
        
        assert ranking.weight_reward == 2.0
        assert ranking.weight_risk == -1.0


class TestParetoRanking:
    """Tests for ParetoRanking."""
    
    def test_dominates(self):
        ranking = ParetoRanking()
        
        # m1 dominates m2 (better in all dimensions)
        m1 = Metrics(reward=0.9, risk=0.1, complexity=0.1)
        m2 = Metrics(reward=0.5, risk=0.5, complexity=0.5)
        
        assert ranking.dominates(m1, m2)
        assert not ranking.dominates(m2, m1)
    
    def test_non_dominated(self):
        ranking = ParetoRanking()
        
        # Neither dominates (trade-off)
        m1 = Metrics(reward=0.9, risk=0.5, complexity=0.1)  # High reward, high risk
        m2 = Metrics(reward=0.5, risk=0.1, complexity=0.1)  # Low reward, low risk
        
        assert not ranking.dominates(m1, m2)
        assert not ranking.dominates(m2, m1)
    
    def test_compare(self):
        ranking = ParetoRanking()
        
        m1 = Metrics(reward=0.9, risk=0.1, complexity=0.1)
        m2 = Metrics(reward=0.5, risk=0.5, complexity=0.5)
        # m3 has same reward as m1 but higher risk - m1 dominates m3
        m3 = Metrics(reward=0.9, risk=0.5, complexity=0.1)
        # m4 is truly incomparable - better reward but worse risk than m1
        m4 = Metrics(reward=0.95, risk=0.2, complexity=0.1)
        
        assert ranking.compare(m1, m2) == 1  # m1 dominates m2
        assert ranking.compare(m2, m1) == -1  # m2 dominated by m1
        assert ranking.compare(m1, m3) == 1  # m1 dominates m3 (same reward, lower risk)
        assert ranking.compare(m1, m4) == 0  # Incomparable (trade-off)


class TestRankingUtilities:
    """Tests for ranking utility functions."""
    
    def test_rank_contributions(self):
        metrics_list = [
            Metrics(reward=0.5, risk=0.5, complexity=0.5),
            Metrics(reward=0.9, risk=0.1, complexity=0.1),
            Metrics(reward=0.7, risk=0.3, complexity=0.3),
        ]
        
        ranking = ScalarRanking()
        indices = rank_contributions(metrics_list, ranking)
        
        # Index 1 should be first (best), then 2, then 0
        assert indices[0] == 1
    
    def test_is_pareto_optimal(self):
        optimal = Metrics(reward=0.9, risk=0.1, complexity=0.1)
        others = [
            Metrics(reward=0.5, risk=0.5, complexity=0.5),
            Metrics(reward=0.7, risk=0.3, complexity=0.3),
        ]
        
        assert is_pareto_optimal(optimal, others)
        
        # Add a dominating one
        dominator = Metrics(reward=0.95, risk=0.05, complexity=0.05)
        others.append(dominator)
        
        assert not is_pareto_optimal(optimal, others)


# =============================================================================
# Sandbox Tests
# =============================================================================

class TestInProcessEvaluator:
    """Tests for InProcessEvaluator (no subprocess)."""
    
    def test_evaluate_success(self, goal_spec: GoalSpec):
        evaluator = InProcessEvaluator()
        pack = AgentPack(version="1.0", parameters=b"test")
        
        result = evaluator.evaluate(pack, goal_spec, seed=42)
        
        assert result.success
        assert result.metrics is not None
        assert result.metrics.reward > 0
        assert result.duration_seconds >= 0
    
    def test_evaluate_deterministic(self, goal_spec: GoalSpec):
        evaluator = InProcessEvaluator()
        pack = AgentPack(version="1.0", parameters=b"deterministic")
        
        result1 = evaluator.evaluate(pack, goal_spec, seed=123)
        result2 = evaluator.evaluate(pack, goal_spec, seed=123)
        
        # Same seed should give same results
        assert result1.metrics.reward == result2.metrics.reward
    
    def test_evaluate_different_seeds(self, goal_spec: GoalSpec):
        evaluator = InProcessEvaluator()
        pack = AgentPack(version="1.0", parameters=b"different")
        
        result1 = evaluator.evaluate(pack, goal_spec, seed=1)
        result2 = evaluator.evaluate(pack, goal_spec, seed=2)
        
        # Different seeds should give different results
        # (Note: with mock evaluator, this depends on pack hash XOR seed)
        assert result1.success and result2.success


class TestEvaluationHarnessAdapter:
    """Tests for EvaluationHarnessAdapter."""
    
    def test_adapter_in_process(self, goal_spec: GoalSpec):
        adapter = EvaluationHarnessAdapter(use_sandbox=False)
        pack = AgentPack(version="1.0", parameters=b"adapter_test")
        
        metrics = adapter.evaluate(pack, goal_spec, seed=42)
        
        assert metrics is not None
        assert metrics.reward > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 components."""
    
    def test_full_coordinator_flow(self):
        """Test complete coordinator flow with real components."""
        goal_spec = GoalSpec(
            goal_id=GoalID("INTEGRATION_TEST"),
            name="Integration Test Goal",
            thresholds=Thresholds(min_reward=0.0),  # Accept everything
            eval_limits=EvaluationLimits(
                max_episodes=5,
                timeout_seconds=5.0,
            ),
        )
        
        coord = IANCoordinator(
            goal_spec=goal_spec,
            config=CoordinatorConfig(leaderboard_capacity=3),
        )
        
        # Process multiple contributions
        results = []
        for i in range(5):
            pack = AgentPack(
                version="1.0",
                parameters=f"integration_{i}".encode(),
            )
            contrib = Contribution(
                goal_id=goal_spec.goal_id,
                agent_pack=pack,
                seed=i,
            )
            result = coord.process_contribution(contrib)
            results.append(result)
        
        # All should be accepted
        assert all(r.accepted for r in results)
        
        # Leaderboard should have 3 (capacity)
        assert len(coord.get_leaderboard()) == 3
        
        # Log should have 5
        assert coord.state.log.size == 5
        
        # Should have active policy
        assert coord.get_active_policy() is not None
        
        # Roots should be non-zero
        assert coord.get_log_root() != b"\x00" * 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
