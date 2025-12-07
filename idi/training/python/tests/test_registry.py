"""Tests for experiment and policy registry."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    PolicyBundle,
    PolicyRegistry,
)


def test_experiment_record_serialization():
    """Test ExperimentRecord to/from dict."""
    record = ExperimentRecord(
        experiment_id="exp_123",
        config_hash="abc123",
        git_commit="def456",
        data_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
        config={"episodes": 100},
        metrics={"sharpe": 0.5},
        artifacts={"policy": "/path/to/policy.json"},
        tags=["test"],
        status="completed",
    )

    d = record.to_dict()
    loaded = ExperimentRecord.from_dict(d)

    assert loaded.experiment_id == record.experiment_id
    assert loaded.config_hash == record.config_hash
    assert loaded.metrics == record.metrics


def test_experiment_registry_create():
    """Test creating experiments in registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        config = {"episodes": 100, "learning_rate": 0.1}
        record = registry.create_experiment(config, tags=["test"])

        assert record.experiment_id.startswith("exp_")
        assert record.status == "running"
        assert record.config == config


def test_experiment_registry_update_metrics():
    """Test updating experiment metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        record = registry.create_experiment({"episodes": 50})
        registry.update_metrics(record.experiment_id, {"sharpe": 0.8, "mean_reward": 10.0})

        loaded = registry.get_experiment(record.experiment_id)
        assert loaded.metrics["sharpe"] == 0.8
        assert loaded.metrics["mean_reward"] == 10.0


def test_experiment_registry_complete():
    """Test marking experiment as complete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        record = registry.create_experiment({"episodes": 50})
        registry.complete_experiment(record.experiment_id, status="completed")

        loaded = registry.get_experiment(record.experiment_id)
        assert loaded.status == "completed"


def test_experiment_registry_list():
    """Test listing experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        # Create multiple experiments
        registry.create_experiment({"episodes": 50}, tags=["quick"])
        registry.create_experiment({"episodes": 100}, tags=["slow"])
        registry.create_experiment({"episodes": 200}, tags=["slow"])

        all_exps = registry.list_experiments()
        assert len(all_exps) == 3

        slow_exps = registry.list_experiments(tags=["slow"])
        assert len(slow_exps) == 2


def test_experiment_registry_find_by_config():
    """Test finding experiments by config hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        config = {"episodes": 100}
        record1 = registry.create_experiment(config)
        record2 = registry.create_experiment(config)  # Same config

        matches = registry.find_by_config_hash(record1.config_hash[:8])
        assert len(matches) == 2


def test_experiment_registry_get_best():
    """Test getting best experiment by metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        exp1 = registry.create_experiment({"lr": 0.1})
        registry.update_metrics(exp1.experiment_id, {"sharpe": 0.5})
        registry.complete_experiment(exp1.experiment_id)

        exp2 = registry.create_experiment({"lr": 0.2})
        registry.update_metrics(exp2.experiment_id, {"sharpe": 0.8})
        registry.complete_experiment(exp2.experiment_id)

        best = registry.get_best_experiment("sharpe", maximize=True)
        assert best.metrics["sharpe"] == 0.8


def test_experiment_registry_add_artifact():
    """Test adding artifact to experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ExperimentRegistry(Path(tmpdir))

        record = registry.create_experiment({"episodes": 50})
        registry.add_artifact(record.experiment_id, "policy", "/path/to/policy.json")

        loaded = registry.get_experiment(record.experiment_id)
        assert loaded.artifacts["policy"] == "/path/to/policy.json"


def test_policy_registry_register():
    """Test registering a policy bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = PolicyRegistry(Path(tmpdir))

        bundle = registry.register_policy(
            experiment_id="exp_123",
            policy_path="/path/to/policy.json",
            config_hash="abc123",
            trace_path="/path/to/traces",
            metrics={"sharpe": 0.8},
            hashes={"policy": "sha256_abc"},
        )

        assert bundle.bundle_id.startswith("policy_")
        assert bundle.experiment_id == "exp_123"


def test_policy_registry_get():
    """Test getting policy bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = PolicyRegistry(Path(tmpdir))

        bundle = registry.register_policy(
            experiment_id="exp_123",
            policy_path="/path/to/policy.json",
            config_hash="abc123",
        )

        loaded = registry.get_policy(bundle.bundle_id)
        assert loaded.experiment_id == "exp_123"
        assert loaded.policy_path == "/path/to/policy.json"


def test_policy_registry_list():
    """Test listing policy bundles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = PolicyRegistry(Path(tmpdir))

        registry.register_policy("exp_1", "/p1.json", "hash1")
        registry.register_policy("exp_2", "/p2.json", "hash2")

        policies = registry.list_policies()
        assert len(policies) == 2


def test_policy_bundle_hashes():
    """Test policy bundle hash storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = PolicyRegistry(Path(tmpdir))

        bundle = registry.register_policy(
            experiment_id="exp_123",
            policy_path="/policy.json",
            config_hash="config_hash",
            hashes={
                "q_table": "sha256_qtable",
                "traces": "sha256_traces",
            },
        )

        loaded = registry.get_policy(bundle.bundle_id)
        assert loaded.hashes["q_table"] == "sha256_qtable"
        assert loaded.hashes["traces"] == "sha256_traces"


def test_experiment_registry_persistence():
    """Test that registry persists across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_dir = Path(tmpdir)

        # Create experiment with first instance
        registry1 = ExperimentRegistry(registry_dir)
        record = registry1.create_experiment({"episodes": 100})
        exp_id = record.experiment_id

        # Load with new instance
        registry2 = ExperimentRegistry(registry_dir)
        loaded = registry2.get_experiment(exp_id)

        assert loaded is not None
        assert loaded.experiment_id == exp_id

