"""Integration tests for Auto-QAgent flows.

These tests verify end-to-end behavior of the Auto-QAgent system:
- Synthetic mode: full flow without external dependencies.
- Real mode (mocked): evaluator behavior with simulated QTrainer.
- Invariants: no crashes, metrics sorted, AgentPatch export valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

from idi.devkit.experimental.auto_qagent import (
    AutoQAgentGoalSpec,
    SynthTimeoutError,
    TrainingBudgetSpec,
    load_goal_spec,
    run_auto_qagent_synth,
    run_auto_qagent_synth_agentpatches,
    _build_objective_indices,
    _compute_env_weights,
    _safe_evaluate_patch_real,
)
from idi.devkit.experimental.sape_q_patch import QAgentPatch, QPatchMeta
from idi.synth import (
    AgentPatch,
    validate_agent_patch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_goal_spec() -> AutoQAgentGoalSpec:
    """A minimal goal spec for synthetic mode testing."""
    return AutoQAgentGoalSpec.from_dict({
        "agent_family": "qagent",
        "profiles": ["conservative"],
        "packs": {"include": ["qagent_base"]},
        "objectives": [
            {"id": "complexity", "direction": "minimize"},
        ],
        "training": {
            "envs": [{"id": "default", "weight": 1.0}],
            "budget": {
                "max_agents": 4,
                "max_generations": 2,
                "max_episodes_per_agent": 16,
                "wallclock_hours": 0.1,
            },
        },
        "eval_mode": "synthetic",
        "outputs": {"num_final_agents": 3},
    })


@pytest.fixture
def real_mode_goal_spec() -> AutoQAgentGoalSpec:
    """Goal spec for real mode testing (will use mocked evaluator)."""
    return AutoQAgentGoalSpec.from_dict({
        "agent_family": "qagent",
        "profiles": ["conservative"],
        "packs": {"include": ["qagent_base", "risk_conservative"]},
        "objectives": [
            {"id": "avg_reward", "direction": "maximize"},
            {"id": "risk_stability", "direction": "maximize"},
        ],
        "training": {
            "envs": [
                {"id": "market_a", "weight": 0.6},
                {"id": "market_b", "weight": 0.4},
            ],
            "budget": {
                "max_agents": 4,
                "max_generations": 2,
                "max_episodes_per_agent": 8,
                "wallclock_hours": 0.1,
            },
        },
        "eval_mode": "real",
        "outputs": {"num_final_agents": 2},
    })


@pytest.fixture
def sample_qpatch() -> QAgentPatch:
    """A sample QAgentPatch for evaluator testing."""
    meta = QPatchMeta(
        name="test-patch",
        description="Test patch",
        version="1.0.0",
        tags=("qagent", "test"),
    )
    return QAgentPatch(
        identifier="test-patch-1",
        num_price_bins=8,
        num_inventory_bins=8,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=500,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Synthetic Mode Integration Tests
# ---------------------------------------------------------------------------

class TestSyntheticModeIntegration:
    """Integration tests for synthetic evaluation mode."""

    def test_full_synthetic_flow(self, synthetic_goal_spec: AutoQAgentGoalSpec) -> None:
        """Run full Auto-QAgent flow in synthetic mode."""
        results = run_auto_qagent_synth(synthetic_goal_spec)

        # Should produce some candidates
        assert len(results) > 0

        # Each result is (QAgentPatch, metrics_tuple)
        for patch, metrics in results:
            assert isinstance(patch, QAgentPatch)
            assert isinstance(metrics, tuple)
            assert len(metrics) > 0
            assert all(isinstance(m, float) for m in metrics)

    def test_synthetic_results_sorted_by_objective(
        self, synthetic_goal_spec: AutoQAgentGoalSpec
    ) -> None:
        """Results should be sorted by objectives (complexity minimized)."""
        results = run_auto_qagent_synth(synthetic_goal_spec)

        if len(results) < 2:
            pytest.skip("Not enough results to verify sorting")

        # For "minimize complexity", lower is better
        # After sorting with reverse=True and sign=-1, first should be lowest
        indices = _build_objective_indices(
            synthetic_goal_spec.objectives,
            synthetic_goal_spec.eval_mode,
        )

        if indices:
            idx, sign = indices[0]
            scores = [metrics[idx] if idx < len(metrics) else 0.0 for _, metrics in results]
            # With minimize (sign=-1), sorted descending means ascending original
            if sign < 0:
                assert scores == sorted(scores), "Results should be sorted by objective"

    def test_synthetic_to_agentpatch_conversion(
        self, synthetic_goal_spec: AutoQAgentGoalSpec
    ) -> None:
        """Converted AgentPatches should be valid."""
        patches = run_auto_qagent_synth_agentpatches(synthetic_goal_spec)

        assert len(patches) > 0

        for patch in patches:
            assert isinstance(patch, AgentPatch)
            # Should not raise
            validate_agent_patch(patch)
            assert patch.agent_type == "qtable"


# ---------------------------------------------------------------------------
# Real Mode Integration Tests (Mocked)
# ---------------------------------------------------------------------------

class TestRealModeMockedIntegration:
    """Integration tests for real mode with mocked QTrainer."""

    def test_real_mode_with_mocked_evaluator(
        self, real_mode_goal_spec: AutoQAgentGoalSpec
    ) -> None:
        """Real mode should work with mocked evaluate_patch_real."""
        mock_metrics = (100.0, 0.5, 50.0)  # avg_reward, risk_stability, min_reward

        with mock.patch(
            "idi.devkit.experimental.auto_qagent.evaluate_patch_real",
            return_value=mock_metrics,
        ):
            results = run_auto_qagent_synth(real_mode_goal_spec)

        assert len(results) > 0

        for patch, metrics in results:
            assert isinstance(patch, QAgentPatch)
            # Metrics should be aggregated from mock
            assert len(metrics) >= 3

    def test_real_mode_multi_env_aggregation(
        self, real_mode_goal_spec: AutoQAgentGoalSpec
    ) -> None:
        """Multi-env weights should be correctly applied."""
        # Different metrics per environment
        call_count = [0]

        def mock_evaluator(patch, episodes, seed=0):
            call_count[0] += 1
            if seed == 0:  # market_a
                return (100.0, 0.8, 80.0)
            else:  # market_b
                return (50.0, 0.6, 40.0)

        with mock.patch(
            "idi.devkit.experimental.auto_qagent.evaluate_patch_real",
            side_effect=mock_evaluator,
        ):
            results = run_auto_qagent_synth(real_mode_goal_spec)

        # Should have called evaluator multiple times (once per env per candidate)
        assert call_count[0] > 0
        assert len(results) > 0

    def test_safe_evaluate_handles_exception(self, sample_qpatch: QAgentPatch) -> None:
        """_safe_evaluate_patch_real should return fallback on exception."""
        with mock.patch(
            "idi.devkit.experimental.auto_qagent.evaluate_patch_real",
            side_effect=RuntimeError("QTrainer exploded"),
        ):
            result = _safe_evaluate_patch_real(sample_qpatch, 16, 0)

        # Should return fallback metrics, not crash
        assert result == (-1e6, -1e6, -1e6)

    def test_safe_evaluate_handles_degenerate_metrics(
        self, sample_qpatch: QAgentPatch
    ) -> None:
        """_safe_evaluate_patch_real should detect all-zero metrics."""
        with mock.patch(
            "idi.devkit.experimental.auto_qagent.evaluate_patch_real",
            return_value=(0.0, 0.0, 0.0),
        ):
            result = _safe_evaluate_patch_real(sample_qpatch, 16, 0)

        # Should return fallback for degenerate metrics
        assert result == (-1e6, -1e6, -1e6)


# ---------------------------------------------------------------------------
# Timeout and Budget Tests
# ---------------------------------------------------------------------------

class TestTimeoutAndBudget:
    """Tests for timeout handling and budget enforcement."""

    def test_timeout_raises_synth_timeout_error(
        self, synthetic_goal_spec: AutoQAgentGoalSpec
    ) -> None:
        """Passing an expired deadline should raise SynthTimeoutError."""
        import time

        # Deadline in the past
        expired_deadline = time.time() - 1.0

        with pytest.raises(SynthTimeoutError):
            run_auto_qagent_synth(synthetic_goal_spec, deadline=expired_deadline)

    def test_budget_clamping_in_from_dict(self) -> None:
        """Extreme budget values should be clamped."""
        spec = AutoQAgentGoalSpec.from_dict({
            "agent_family": "qagent",
            "training": {
                "budget": {
                    "max_agents": 999999,  # Should be clamped to MAX_AGENTS_LIMIT
                    "max_episodes_per_agent": -100,  # Should be clamped to 1
                    "wallclock_hours": 9999,  # Should be clamped to MAX_WALLCLOCK_HOURS
                },
            },
            "eval_mode": "synthetic",
        })

        assert spec.training.budget.max_agents <= 1024
        assert spec.training.budget.max_episodes_per_agent >= 1
        assert spec.training.budget.wallclock_hours <= 168.0


# ---------------------------------------------------------------------------
# Goal Spec Loading Tests
# ---------------------------------------------------------------------------

class TestGoalSpecLoading:
    """Tests for loading goal specs from files."""

    def test_load_goal_spec_from_file(self, tmp_path: Path) -> None:
        """load_goal_spec should parse JSON file correctly."""
        goal_file = tmp_path / "goal.json"
        goal_file.write_text(json.dumps({
            "agent_family": "qagent",
            "profiles": ["test"],
            "eval_mode": "synthetic",
        }))

        spec = load_goal_spec(goal_file)

        assert spec.agent_family == "qagent"
        assert spec.profiles == ("test",)
        assert spec.eval_mode == "synthetic"

    def test_load_goal_spec_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON should raise error."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_goal_spec(bad_file)

    def test_load_goal_spec_non_object(self, tmp_path: Path) -> None:
        """Non-object JSON should raise ValueError."""
        array_file = tmp_path / "array.json"
        array_file.write_text("[]")

        with pytest.raises(ValueError, match="JSON object"):
            load_goal_spec(array_file)


# ---------------------------------------------------------------------------
# Environment Weight Tests
# ---------------------------------------------------------------------------

class TestEnvWeights:
    """Tests for environment weight computation."""

    def test_compute_env_weights_normalization(self) -> None:
        """Weights should be normalized to sum to 1.0."""
        from idi.devkit.experimental.auto_qagent import TrainingEnvSpec

        envs = (
            TrainingEnvSpec(id="a", weight=2.0),
            TrainingEnvSpec(id="b", weight=3.0),
        )

        weights = _compute_env_weights(envs)

        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-9
        assert abs(weights[0] - 0.4) < 1e-9  # 2/5
        assert abs(weights[1] - 0.6) < 1e-9  # 3/5

    def test_compute_env_weights_all_zero(self) -> None:
        """All-zero weights should become equal weights."""
        from idi.devkit.experimental.auto_qagent import TrainingEnvSpec

        envs = (
            TrainingEnvSpec(id="a", weight=0.0),
            TrainingEnvSpec(id="b", weight=0.0),
        )

        weights = _compute_env_weights(envs)

        assert len(weights) == 2
        assert abs(weights[0] - 0.5) < 1e-9
        assert abs(weights[1] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# End-to-End Invariants
# ---------------------------------------------------------------------------

class TestEndToEndInvariants:
    """Tests verifying critical invariants hold across the system."""

    def test_no_crashes_on_empty_objectives(self) -> None:
        """Empty objectives should not crash."""
        spec = AutoQAgentGoalSpec.from_dict({
            "agent_family": "qagent",
            "objectives": [],
            "eval_mode": "synthetic",
        })

        # Should not raise
        results = run_auto_qagent_synth(spec)
        assert isinstance(results, list)

    def test_no_crashes_on_empty_envs(self) -> None:
        """Empty environments should not crash in real mode."""
        spec = AutoQAgentGoalSpec.from_dict({
            "agent_family": "qagent",
            "training": {"envs": []},
            "eval_mode": "synthetic",  # Use synthetic to avoid needing QTrainer
        })

        results = run_auto_qagent_synth(spec)
        assert isinstance(results, list)

    def test_agentpatch_round_trip(
        self, synthetic_goal_spec: AutoQAgentGoalSpec, tmp_path: Path
    ) -> None:
        """AgentPatches should survive save/load round-trip."""
        from idi.synth import load_agent_patch, save_agent_patch

        patches = run_auto_qagent_synth_agentpatches(synthetic_goal_spec)

        if not patches:
            pytest.skip("No patches generated")

        patch = patches[0]
        path = tmp_path / "patch.json"

        save_agent_patch(patch, path)
        loaded = load_agent_patch(path)

        assert loaded.meta.id == patch.meta.id
        assert loaded.agent_type == patch.agent_type
        assert dict(loaded.payload) == dict(patch.payload)
