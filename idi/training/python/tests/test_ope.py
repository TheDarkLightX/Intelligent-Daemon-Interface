"""Tests for Off-Policy Evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.config import TrainingConfig
from idi_iann.domain import Action
from idi_iann.ope import (
    LoggedDataset,
    LoggedEpisode,
    LoggedTransition,
    OPEEvaluator,
    OPEResult,
    run_ope,
)
from idi_iann.policy import LookupPolicy
from idi_iann.trainer import QTrainer


def create_test_dataset() -> LoggedDataset:
    """Create a simple test dataset."""
    transitions = [
        LoggedTransition(
            state=(0, 0, 0, 0, 0),
            action=Action.BUY,
            reward=1.0,
            next_state=(1, 0, 0, 0, 0),
            behavior_prob=0.5,
        ),
        LoggedTransition(
            state=(1, 0, 0, 0, 0),
            action=Action.HOLD,
            reward=0.5,
            next_state=(1, 1, 0, 0, 0),
            behavior_prob=0.3,
        ),
        LoggedTransition(
            state=(1, 1, 0, 0, 0),
            action=Action.SELL,
            reward=-0.2,
            next_state=(0, 1, 0, 0, 0),
            behavior_prob=0.2,
            done=True,
        ),
    ]
    episode = LoggedEpisode(
        episode_id="ep1",
        transitions=transitions,
        behavior_policy_id="test_policy",
        config_hash="abc123",
        data_version="1.0",
    )
    return LoggedDataset(episodes=[episode, episode])  # Two episodes


def test_logged_dataset_roundtrip():
    """Test dataset JSON serialization roundtrip."""
    dataset = create_test_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.json"
        dataset.to_json(path)
        loaded = LoggedDataset.from_json(path)

        assert len(loaded.episodes) == len(dataset.episodes)
        assert loaded.episodes[0].episode_id == dataset.episodes[0].episode_id


def test_ope_direct_method():
    """Test Direct Method estimator."""
    dataset = create_test_dataset()
    policy = LookupPolicy()

    # Set some Q-values
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)
    policy.update((1, 0, 0, 0, 0), Action.HOLD, 0.5)

    evaluator = OPEEvaluator(policy, discount=0.99)
    result = evaluator.direct_method(dataset)

    assert result.estimator == "DM"
    assert result.n_episodes == 2
    assert result.value_estimate >= 0  # Q-values are non-negative


def test_ope_importance_sampling():
    """Test Importance Sampling estimators."""
    dataset = create_test_dataset()
    policy = LookupPolicy()

    evaluator = OPEEvaluator(policy, discount=0.99)

    # IPS
    ips_result = evaluator.importance_sampling(dataset, weighted=False)
    assert ips_result.estimator == "IPS"
    assert ips_result.n_episodes == 2

    # WIS
    wis_result = evaluator.importance_sampling(dataset, weighted=True)
    assert wis_result.estimator == "WIS"
    assert wis_result.n_episodes == 2


def test_ope_doubly_robust():
    """Test Doubly Robust estimator."""
    dataset = create_test_dataset()
    policy = LookupPolicy()

    evaluator = OPEEvaluator(policy, discount=0.99)
    result = evaluator.doubly_robust(dataset)

    assert result.estimator == "DR"
    assert result.n_episodes == 2
    assert "variance" in result.details


def test_ope_evaluate_all():
    """Test running all estimators."""
    dataset = create_test_dataset()
    policy = LookupPolicy()

    evaluator = OPEEvaluator(policy, discount=0.99)
    results = evaluator.evaluate_all(dataset)

    assert "DM" in results
    assert "IPS" in results
    assert "WIS" in results
    assert "DR" in results


def test_ope_empty_dataset():
    """Test OPE handles empty dataset gracefully."""
    dataset = LoggedDataset(episodes=[])
    policy = LookupPolicy()

    evaluator = OPEEvaluator(policy, discount=0.99)
    results = evaluator.evaluate_all(dataset)

    for result in results.values():
        assert result.n_episodes == 0
        assert result.value_estimate == 0.0


def test_run_ope_cli():
    """Test run_ope function with file I/O."""
    dataset = create_test_dataset()
    policy = LookupPolicy()

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "dataset.json"
        output_path = Path(tmpdir) / "results.json"

        dataset.to_json(dataset_path)
        results = run_ope(policy, dataset_path, discount=0.99, output_path=output_path)

        assert output_path.exists()
        assert "DM" in results
        assert "DR" in results

        # Verify output format
        output = json.loads(output_path.read_text())
        assert "DM" in output
        assert "value_estimate" in output["DM"]

