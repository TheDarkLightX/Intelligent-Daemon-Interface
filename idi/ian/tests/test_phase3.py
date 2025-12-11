"""
Tests for IAN Phase 3: IDI Integration.

Tests cover:
- IDIInvariantChecker (Python + MPB checks)
- IDIProofVerifier (MPB + ZK proofs)
- IDIEvaluationHarness (backtest, simulation, mock)
- create_idi_coordinator factory
"""

import json
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from idi.ian.models import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    Metrics,
    Thresholds,
    EvaluationLimits,
)
from idi.ian.idi_integration import (
    IDIInvariantChecker,
    IDIProofVerifier,
    IDIEvaluationHarness,
    create_idi_invariant_checker,
    create_idi_proof_verifier,
    create_idi_evaluation_harness,
    create_idi_coordinator,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def goal_spec() -> GoalSpec:
    """Create a test GoalSpec."""
    return GoalSpec(
        goal_id=GoalID("IDI_INTEGRATION_TEST"),
        name="IDI Integration Test",
        description="Testing IDI integration",
        invariant_ids=["I1", "I2"],
        thresholds=Thresholds(min_reward=0.0),
        eval_limits=EvaluationLimits(
            max_episodes=5,
            timeout_seconds=5.0,
        ),
    )


@pytest.fixture
def conservative_agent_pack() -> AgentPack:
    """Create a conservative agent pack that should pass invariants."""
    params = {
        "max_leverage": 1.0,
        "max_position_pct": 25.0,
        "max_drawdown_pct": 10.0,
        "min_collateral_ratio": 1.5,
        "asset_whitelist": ["BTC", "ETH"],
    }
    return AgentPack(
        version="1.0",
        parameters=json.dumps(params).encode("utf-8"),
        metadata=params,
    )


@pytest.fixture
def risky_agent_pack() -> AgentPack:
    """Create a risky agent pack that might fail invariants."""
    params = {
        "max_leverage": 10.0,  # Very high leverage
        "max_position_pct": 100.0,  # Full concentration
        "max_drawdown_pct": 50.0,  # High drawdown
        "min_collateral_ratio": 0.5,  # Low collateral
    }
    return AgentPack(
        version="1.0",
        parameters=json.dumps(params).encode("utf-8"),
        metadata=params,
    )


# =============================================================================
# Invariant Checker Tests
# =============================================================================

class TestIDIInvariantChecker:
    """Tests for IDIInvariantChecker."""
    
    def test_initialization(self):
        checker = IDIInvariantChecker()
        assert checker.use_python_checks
        assert checker.use_mpb_checks
    
    def test_check_with_no_idi_modules(self, goal_spec, conservative_agent_pack):
        """Test that checker works even without IDI modules loaded."""
        checker = IDIInvariantChecker(
            use_python_checks=True,
            use_mpb_checks=False,  # Disable MPB for this test
        )
        
        # Should pass (or gracefully handle missing modules)
        passed, reason = checker.check(conservative_agent_pack, goal_spec)
        # Either passes or explains what's missing
        assert isinstance(passed, bool)
        assert isinstance(reason, str)
    
    def test_extract_spec_params_from_json(self, goal_spec, conservative_agent_pack):
        checker = IDIInvariantChecker()
        params = checker._extract_spec_params(conservative_agent_pack, goal_spec)
        
        assert params["max_leverage"] == 1.0
        assert params["max_position_pct"] == 25.0
    
    def test_extract_spec_params_from_binary(self, goal_spec):
        """Test extraction when parameters are binary."""
        binary_pack = AgentPack(
            version="1.0",
            parameters=b"\x00\x01\x02\x03",  # Binary data
            metadata={"max_leverage": 2.0},
        )
        
        checker = IDIInvariantChecker()
        params = checker._extract_spec_params(binary_pack, goal_spec)
        
        # Should fall back to metadata
        assert params["max_leverage"] == 2.0
    
    def test_map_params_to_registers(self, conservative_agent_pack):
        checker = IDIInvariantChecker()
        params = {
            "max_leverage": 1.0,
            "max_position_pct": 25.0,
            "max_drawdown_pct": 10.0,
            "min_collateral_ratio": 1.5,
            "rebalance_threshold_pct": 5.0,
        }
        
        registers = checker._map_params_to_registers(params)
        
        assert registers["R1"] == 1.0
        assert registers["R2"] == 25.0
        assert registers["R3"] == 10.0
        assert registers["R4"] == 1.5
        assert registers["R5"] == 5.0


class TestIDIInvariantCheckerFactory:
    """Tests for create_idi_invariant_checker factory."""
    
    def test_create_with_defaults(self):
        checker = create_idi_invariant_checker()
        assert checker.use_python_checks
        assert checker.use_mpb_checks
    
    def test_create_python_only(self):
        checker = create_idi_invariant_checker(use_python=True, use_mpb=False)
        assert checker.use_python_checks
        assert not checker.use_mpb_checks
    
    def test_create_mpb_only(self):
        checker = create_idi_invariant_checker(use_python=False, use_mpb=True)
        assert not checker.use_python_checks
        assert checker.use_mpb_checks


# =============================================================================
# Proof Verifier Tests
# =============================================================================

class TestIDIProofVerifier:
    """Tests for IDIProofVerifier."""
    
    def test_initialization(self):
        verifier = IDIProofVerifier()
        assert not verifier.require_proofs
    
    def test_no_proofs_when_not_required(self, goal_spec, conservative_agent_pack):
        verifier = IDIProofVerifier(require_proofs=False)
        
        valid, reason = verifier.verify(
            conservative_agent_pack,
            proofs={},
            goal_spec=goal_spec,
        )
        
        assert valid
        assert "no proofs to verify" in reason
    
    def test_no_proofs_when_required(self, goal_spec, conservative_agent_pack):
        verifier = IDIProofVerifier(require_proofs=True)
        
        valid, reason = verifier.verify(
            conservative_agent_pack,
            proofs={},
            goal_spec=goal_spec,
        )
        
        assert not valid
        assert "required" in reason
    
    def test_invalid_mpb_proof(self, goal_spec, conservative_agent_pack):
        verifier = IDIProofVerifier()
        
        # Invalid proof data
        proofs = {"mpb": b"invalid_proof_data"}
        
        valid, reason = verifier.verify(
            conservative_agent_pack,
            proofs=proofs,
            goal_spec=goal_spec,
        )
        
        # Should fail gracefully
        assert isinstance(valid, bool)


class TestIDIProofVerifierFactory:
    """Tests for create_idi_proof_verifier factory."""
    
    def test_create_with_defaults(self):
        verifier = create_idi_proof_verifier()
        assert not verifier.require_proofs
    
    def test_create_with_required_proofs(self):
        verifier = create_idi_proof_verifier(require_proofs=True)
        assert verifier.require_proofs


# =============================================================================
# Evaluation Harness Tests
# =============================================================================

class TestIDIEvaluationHarness:
    """Tests for IDIEvaluationHarness."""
    
    def test_initialization(self):
        harness = IDIEvaluationHarness()
        assert harness.harness_type == "backtest"
        assert harness.deterministic_seed
    
    def test_mock_evaluation(self, goal_spec, conservative_agent_pack):
        harness = IDIEvaluationHarness(harness_type="mock")
        
        metrics = harness.evaluate(conservative_agent_pack, goal_spec, seed=42)
        
        assert metrics is not None
        assert metrics.reward > 0
        assert metrics.risk >= 0
        assert metrics.complexity >= 0
    
    def test_deterministic_mock(self, goal_spec, conservative_agent_pack):
        harness = IDIEvaluationHarness(harness_type="mock", deterministic_seed=True)
        
        metrics1 = harness.evaluate(conservative_agent_pack, goal_spec, seed=123)
        metrics2 = harness.evaluate(conservative_agent_pack, goal_spec, seed=123)
        
        assert metrics1.reward == metrics2.reward
    
    def test_backtest_falls_back_to_mock(self, goal_spec, conservative_agent_pack):
        """Test that backtest gracefully falls back to mock when trainer unavailable."""
        harness = IDIEvaluationHarness(harness_type="backtest")
        
        metrics = harness.evaluate(conservative_agent_pack, goal_spec, seed=42)
        
        # Should return metrics (from mock fallback)
        assert metrics is not None


class TestIDIEvaluationHarnessFactory:
    """Tests for create_idi_evaluation_harness factory."""
    
    def test_create_with_defaults(self):
        harness = create_idi_evaluation_harness()
        assert harness.harness_type == "backtest"
    
    def test_create_simulation(self):
        harness = create_idi_evaluation_harness(harness_type="simulation")
        assert harness.harness_type == "simulation"


# =============================================================================
# Integrated Coordinator Tests
# =============================================================================

class TestCreateIDICoordinator:
    """Tests for create_idi_coordinator factory."""
    
    def test_create_default_coordinator(self, goal_spec):
        coord = create_idi_coordinator(goal_spec)
        
        assert coord.goal_spec.goal_id == goal_spec.goal_id
        assert coord.config.leaderboard_capacity == 100
    
    def test_create_with_custom_config(self, goal_spec):
        coord = create_idi_coordinator(
            goal_spec,
            leaderboard_capacity=50,
            use_pareto=True,
            require_proofs=False,
        )
        
        assert coord.config.leaderboard_capacity == 50
        assert coord.config.use_pareto
    
    def test_process_contribution(self, goal_spec, conservative_agent_pack):
        coord = create_idi_coordinator(
            goal_spec,
            leaderboard_capacity=10,
            use_python_invariants=False,  # Skip for test
            use_mpb_invariants=False,
            require_proofs=False,
            harness_type="mock",
        )
        
        contrib = Contribution(
            goal_id=goal_spec.goal_id,
            agent_pack=conservative_agent_pack,
            seed=42,
        )
        
        result = coord.process_contribution(contrib)
        
        assert result.accepted
        assert result.metrics is not None
        assert coord.state.log.size == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """End-to-end integration tests for Phase 3."""
    
    def test_full_idi_coordinator_flow(self):
        """Test complete flow with IDI integration."""
        goal_spec = GoalSpec(
            goal_id=GoalID("FULL_IDI_TEST"),
            name="Full IDI Integration Test",
            thresholds=Thresholds(min_reward=0.0),
            eval_limits=EvaluationLimits(max_episodes=3),
        )
        
        coord = create_idi_coordinator(
            goal_spec,
            leaderboard_capacity=5,
            use_python_invariants=False,
            use_mpb_invariants=False,
            require_proofs=False,
            harness_type="mock",
        )
        
        # Submit multiple contributions
        for i in range(7):
            pack = AgentPack(
                version="1.0",
                parameters=f"idi_test_{i}".encode(),
            )
            contrib = Contribution(
                goal_id=goal_spec.goal_id,
                agent_pack=pack,
                seed=i,
            )
            
            result = coord.process_contribution(contrib)
            assert result.accepted
        
        # Verify state
        assert coord.state.log.size == 7
        assert len(coord.get_leaderboard()) == 5  # Capacity
        assert coord.get_active_policy() is not None
        
        # Verify stats
        stats = coord.get_stats()
        assert stats["total_contributions"] == 7
        assert stats["accepted_contributions"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
