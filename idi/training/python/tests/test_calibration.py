"""Tests for calibration module."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.calibration import CalibrationChecker, CalibrationResult, check_calibration
from idi_iann.domain import Action
from idi_iann.policy import LookupPolicy


def test_calibration_basic():
    """Test basic calibration computation."""
    policy = LookupPolicy()

    # Set up some Q-values
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)
    policy.update((1, 0, 0, 0, 0), Action.HOLD, 0.5)
    policy.update((2, 0, 0, 0, 0), Action.SELL, 0.2)

    checker = CalibrationChecker(policy, n_buckets=5)

    states = [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (2, 0, 0, 0, 0),
    ]
    actions = [Action.BUY, Action.HOLD, Action.SELL]
    realized = [1.2, 0.4, 0.3]  # Actual returns

    result = checker.compute_calibration(states, actions, realized)

    assert result.n_samples == 3
    assert result.ece >= 0.0
    assert result.mce >= 0.0
    assert len(result.buckets) > 0


def test_calibration_empty():
    """Test calibration with empty data."""
    policy = LookupPolicy()
    checker = CalibrationChecker(policy)

    result = checker.compute_calibration([], [], [])

    assert result.n_samples == 0
    assert result.ece == 0.0
    assert result.mce == 0.0


def test_calibration_from_rollouts():
    """Test calibration from rollout data format."""
    policy = LookupPolicy()
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)

    checker = CalibrationChecker(policy)

    rollout_data = [
        {
            "state": [0, 0, 0, 0, 0],
            "action": "buy",
            "rewards": [1.0, 0.5, 0.2],
        },
        {
            "state": [0, 0, 0, 0, 0],
            "action": "buy",
            "rewards": [0.8, 0.3],
        },
    ]

    result = checker.compute_from_rollouts(rollout_data, discount=0.99)

    assert result.n_samples == 2
    assert result.ece >= 0.0


def test_calibration_to_dict():
    """Test CalibrationResult serialization."""
    policy = LookupPolicy()
    checker = CalibrationChecker(policy)

    states = [(0, 0, 0, 0, 0)]
    actions = [Action.BUY]
    realized = [0.5]

    result = checker.compute_calibration(states, actions, realized)
    d = result.to_dict()

    assert "ece" in d
    assert "mce" in d
    assert "n_samples" in d
    assert "buckets" in d


def test_check_calibration_file():
    """Test check_calibration with file I/O."""
    policy = LookupPolicy()
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        rollout_path = Path(tmpdir) / "rollouts.json"
        output_path = Path(tmpdir) / "calibration.json"

        rollout_data = {
            "rollouts": [
                {"state": [0, 0, 0, 0, 0], "action": "buy", "rewards": [1.0, 0.5]},
                {"state": [0, 0, 0, 0, 0], "action": "buy", "rewards": [0.8]},
            ]
        }
        rollout_path.write_text(json.dumps(rollout_data))

        result = check_calibration(
            policy, rollout_path, n_buckets=5, output_path=output_path
        )

        assert output_path.exists()
        assert result.n_samples == 2

        # Verify output format
        output = json.loads(output_path.read_text())
        assert "ece" in output
        assert "mce" in output


def test_calibration_well_calibrated():
    """Test that well-calibrated predictions have low ECE."""
    policy = LookupPolicy()

    # Set Q-values that match realized returns
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)
    policy.update((1, 0, 0, 0, 0), Action.BUY, 0.5)

    checker = CalibrationChecker(policy, n_buckets=2)

    states = [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]
    actions = [Action.BUY, Action.BUY]
    realized = [1.0, 0.5]  # Match Q-values exactly

    result = checker.compute_calibration(states, actions, realized)

    # ECE should be low for well-calibrated predictions
    assert result.ece < 0.1


def test_calibration_poorly_calibrated():
    """Test that poorly-calibrated predictions have high ECE."""
    policy = LookupPolicy()

    # Set Q-values that don't match realized returns
    policy.update((0, 0, 0, 0, 0), Action.BUY, 10.0)  # Overestimate
    policy.update((1, 0, 0, 0, 0), Action.BUY, 10.0)  # Overestimate

    checker = CalibrationChecker(policy, n_buckets=2)

    states = [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]
    actions = [Action.BUY, Action.BUY]
    realized = [0.1, 0.1]  # Much lower than predicted

    result = checker.compute_calibration(states, actions, realized)

    # ECE should be high for poorly-calibrated predictions
    assert result.ece > 0.5 or result.mce > 5.0

