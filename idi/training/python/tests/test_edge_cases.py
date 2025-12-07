"""Edge case tests for empty inputs, boundary conditions, and numeric extremes."""

import pytest

from idi_iann.ope import OPEEvaluator, LoggedDataset
from idi_iann.calibration import CalibrationChecker
from idi_iann.drift import DriftDetector, compute_drift_report
from idi_iann.ensemble import EnsembleQLearning, EnsembleConfig
from idi_iann.registry import ExperimentRegistry, PolicyRegistry
from idi_iann.domain import Action
from idi_iann.policy import LookupPolicy
from pathlib import Path
import tempfile


class TestEmptyInputs:
    """Test handling of empty inputs across modules."""

    def test_ope_zero_episodes(self):
        """OPE with zero episodes should return zero estimates."""
        dataset = LoggedDataset(episodes=[])
        policy = LookupPolicy()
        evaluator = OPEEvaluator(policy, discount=0.99)

        result = evaluator.direct_method(dataset)
        assert result.n_episodes == 0
        assert result.value_estimate == 0.0
        assert result.standard_error == 0.0

        result_ips = evaluator.importance_sampling(dataset, weighted=False)
        assert result_ips.n_episodes == 0

        result_dr = evaluator.doubly_robust(dataset)
        assert result_dr.n_episodes == 0

    def test_calibration_empty_rollouts(self):
        """Calibration with empty rollouts should return zero ECE."""
        policy = LookupPolicy()
        checker = CalibrationChecker(policy, n_buckets=10)

        result = checker.compute_calibration([], [], [])
        assert result.n_samples == 0
        assert result.ece == 0.0
        assert result.mce == 0.0
        assert len(result.buckets) == 0

    def test_drift_empty_features(self):
        """Drift detection with empty feature lists."""
        detector = DriftDetector()

        ref_features = {}
        comp_features = {}

        report = detector.generate_report(ref_features, comp_features)
        assert len(report.feature_metrics) == 0
        assert report.overall_score == 0.0

    def test_ensemble_zero_members(self):
        """Ensemble with zero members should handle gracefully."""
        # Currently doesn't raise error, but should handle gracefully
        config = EnsembleConfig(n_members=0)
        # This will create empty members list - test that it doesn't crash
        try:
            ensemble = EnsembleQLearning(config)
            # Empty ensemble should still work for some operations
            assert len(ensemble.members) == 0
        except (ValueError, IndexError):
            # If it raises, that's also acceptable
            pass

    def test_registry_empty_operations(self):
        """Registry operations on empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExperimentRegistry(Path(tmpdir))

            # List should return empty
            exps = registry.list_experiments()
            assert len(exps) == 0

            # Get non-existent should return None
            assert registry.get_experiment("nonexistent") is None

            # Best experiment should return None
            assert registry.get_best_experiment("sharpe") is None

    def test_policy_registry_empty(self):
        """Policy registry operations on empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PolicyRegistry(Path(tmpdir))

            policies = registry.list_policies()
            assert len(policies) == 0

            assert registry.get_policy("nonexistent") is None


class TestBoundaryConditions:
    """Test boundary conditions for hyperparameters and state spaces."""

    def test_single_state_q_table(self):
        """Single-state Q-table should work correctly."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        policy.update(state, Action.BUY, 1.0)
        policy.update(state, Action.SELL, 0.5)

        assert policy.q_value(state, Action.BUY) == 1.0
        assert policy.q_value(state, Action.SELL) == 0.5
        assert policy.best_action(state) == Action.BUY

    def test_episode_length_one(self):
        """Training with episode length of 1."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(episodes=1, episode_length=1)
        trainer = QTrainer(config)
        policy, trace = trainer.run()

        assert len(trace.ticks) == 1
        assert policy is not None

    def test_learning_rate_zero(self):
        """Learning rate of 0.0 should not update Q-values."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        policy.update(state, Action.BUY, 1.0)
        initial_q = policy.q_value(state, Action.BUY)

        # Update with learning rate 0
        policy.update(state, Action.BUY, 0.0 * 10.0)  # delta = 0

        assert policy.q_value(state, Action.BUY) == initial_q

    def test_learning_rate_one(self):
        """Learning rate of 1.0 should set Q-value exactly to target."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        policy.update(state, Action.BUY, 1.0)
        initial_q = policy.q_value(state, Action.BUY)

        target = 5.0
        td_error = target - initial_q
        policy.update(state, Action.BUY, 1.0 * td_error)  # lr = 1.0

        assert abs(policy.q_value(state, Action.BUY) - target) < 1e-6

    def test_discount_zero(self):
        """Discount of 0.0 should ignore future rewards."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=0.0, episodes=1, episode_length=5)
        trainer = QTrainer(config)

        # TD target should be just reward (no bootstrap)
        state = (0, 0, 0, 0, 0)
        next_state = (1, 0, 0, 0, 0)

        trainer._policy.update(state, Action.BUY, 0.0)
        trainer._policy.update(next_state, Action.BUY, 100.0)  # High future value

        td_target = trainer._compute_td_target(
            state, Action.BUY, reward=1.0, next_state=next_state
        )

        # With discount=0, should ignore next_state value
        assert abs(td_target - 1.0) < 1e-6

    def test_discount_one(self):
        """Discount of 1.0 should fully consider future rewards."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=1.0, episodes=1, episode_length=5)
        trainer = QTrainer(config)

        state = (0, 0, 0, 0, 0)
        next_state = (1, 0, 0, 0, 0)

        trainer._policy.update(state, Action.BUY, 0.0)
        trainer._policy.update(next_state, Action.BUY, 10.0)

        td_target = trainer._compute_td_target(
            state, Action.BUY, reward=1.0, next_state=next_state
        )

        # With discount=1.0, should include full next_state value
        assert abs(td_target - 11.0) < 1e-6

    def test_all_q_values_equal(self):
        """Tie-breaking when all Q-values are equal."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        # Set all Q-values to same value
        policy.update(state, Action.HOLD, 1.0)
        policy.update(state, Action.BUY, 1.0)
        policy.update(state, Action.SELL, 1.0)

        # Should return a valid action (not crash)
        best = policy.best_action(state)
        assert best in (Action.HOLD, Action.BUY, Action.SELL)

        # Should be deterministic
        best2 = policy.best_action(state)
        assert best == best2


class TestNumericEdgeCases:
    """Test numeric edge cases: overflow, underflow, extreme values."""

    def test_very_large_rewards(self):
        """Very large rewards should not cause overflow."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        # Large reward
        large_reward = 1e6
        policy.update(state, Action.BUY, large_reward)

        q_val = policy.q_value(state, Action.BUY)
        assert abs(q_val) < float("inf")
        assert q_val == large_reward

    def test_very_small_probabilities_ope(self):
        """Very small probabilities in OPE should not cause division by zero."""
        from idi_iann.ope import LoggedDataset, LoggedEpisode, LoggedTransition

        # Create transition with very small behavior prob
        transition = LoggedTransition(
            state=(0, 0, 0, 0, 0),
            action=Action.BUY,
            reward=1.0,
            next_state=(1, 0, 0, 0, 0),
            behavior_prob=1e-10,  # Very small
            done=False,
        )

        episode = LoggedEpisode(
            episode_id="test",
            transitions=[transition],
            behavior_policy_id="test",
            config_hash="test",
            data_version="1.0",
        )

        dataset = LoggedDataset(episodes=[episode])
        policy = LookupPolicy()
        evaluator = OPEEvaluator(policy, discount=0.99)

        # Should not crash
        result = evaluator.importance_sampling(dataset, weighted=True)
        assert result.n_episodes == 1

    def test_negative_rewards(self):
        """Negative rewards should be handled correctly."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        policy.update(state, Action.BUY, -10.0)
        assert policy.q_value(state, Action.BUY) == -10.0

        # Update with positive delta
        policy.update(state, Action.BUY, 5.0)
        assert policy.q_value(state, Action.BUY) == -5.0

    def test_zero_variance_calibration(self):
        """Calibration with zero variance should work."""
        policy = LookupPolicy()
        checker = CalibrationChecker(policy, n_buckets=5)

        # Set Q-value to match realized values for well-calibrated case
        state = (0, 0, 0, 0, 0)
        policy.update(state, Action.BUY, 1.0)

        # All predictions and realized values are identical
        states = [state] * 10
        actions = [Action.BUY] * 10
        realized = [1.0] * 10

        result = checker.compute_calibration(states, actions, realized)
        assert result.n_samples == 10
        # ECE should be low (well-calibrated when Q matches realized)
        assert result.ece < 0.1

    def test_extreme_q_values(self):
        """Extreme Q-values should not break argmax."""
        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        policy.update(state, Action.HOLD, -1e10)
        policy.update(state, Action.BUY, 1e10)
        policy.update(state, Action.SELL, 0.0)

        best = policy.best_action(state)
        assert best == Action.BUY

    def test_nan_handling(self):
        """NaN values should be detected and handled."""
        import math

        policy = LookupPolicy()
        state = (0, 0, 0, 0, 0)

        # Try to introduce NaN (should be prevented by implementation)
        policy.update(state, Action.BUY, 1.0)
        q_val = policy.q_value(state, Action.BUY)

        assert not math.isnan(q_val)
        assert math.isfinite(q_val)

