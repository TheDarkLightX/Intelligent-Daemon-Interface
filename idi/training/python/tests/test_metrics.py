"""Tests for metrics abstraction module."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.metrics import (
    CompositeBackend,
    InMemoryBackend,
    JSONFileBackend,
    MetricPoint,
    MetricsRecorder,
    StdoutBackend,
    get_global_recorder,
    set_global_recorder,
)


def test_in_memory_backend():
    """Test in-memory metrics backend."""
    backend = InMemoryBackend()

    backend.record("test.metric", 1.5, {"env": "test"})
    backend.record("test.metric", 2.5)

    assert len(backend.points) == 2
    assert backend.get_values("test.metric") == [1.5, 2.5]
    assert backend.get_latest("test.metric") == 2.5


def test_in_memory_get_latest_empty():
    """Test get_latest returns None for missing metric."""
    backend = InMemoryBackend()
    assert backend.get_latest("nonexistent") is None


def test_json_file_backend():
    """Test JSON file metrics backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.json"
        backend = JSONFileBackend(path, buffer_size=2)

        backend.record("metric1", 1.0)
        # Buffer not full, file shouldn't exist yet
        assert not path.exists()

        backend.record("metric2", 2.0)
        # Buffer full, should flush
        assert path.exists()

        data = json.loads(path.read_text())
        assert len(data) == 2


def test_json_file_backend_explicit_flush():
    """Test explicit flush of JSON backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.json"
        backend = JSONFileBackend(path, buffer_size=100)

        backend.record("metric1", 1.0)
        backend.flush()

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1


def test_stdout_backend(capsys):
    """Test stdout metrics backend."""
    backend = StdoutBackend(prefix="[TEST]")

    backend.record("test.metric", 1.2345, {"env": "dev"})
    captured = capsys.readouterr()

    assert "[TEST]" in captured.out
    assert "test.metric" in captured.out
    assert "1.2345" in captured.out
    assert "env=dev" in captured.out


def test_composite_backend():
    """Test composite backend forwards to all backends."""
    backend1 = InMemoryBackend()
    backend2 = InMemoryBackend()
    composite = CompositeBackend([backend1, backend2])

    composite.record("metric", 1.0)

    assert len(backend1.points) == 1
    assert len(backend2.points) == 1


def test_metrics_recorder_training():
    """Test MetricsRecorder training metrics."""
    backend = InMemoryBackend()
    recorder = MetricsRecorder(backend, default_tags={"run": "test"})

    recorder.record_td_error(0.5, episode=1)
    recorder.record_q_value("buy", 1.2, episode=1)
    recorder.record_episode_reward(10.0, episode=1)
    recorder.record_exploration(0.9, episode=1)

    assert len(backend.points) == 4
    assert backend.get_latest("training.td_error") == 0.5
    assert backend.get_latest("training.q_value") == 1.2
    assert backend.get_latest("training.episode_reward") == 10.0
    assert backend.get_latest("training.exploration") == 0.9


def test_metrics_recorder_environment():
    """Test MetricsRecorder environment metrics."""
    backend = InMemoryBackend()
    recorder = MetricsRecorder(backend)

    recorder.record_env_price(100.0, step=5)
    recorder.record_env_regime("bull", step=5)
    recorder.record_env_pnl(50.0, step=5)

    assert backend.get_latest("env.price") == 100.0
    assert backend.get_latest("env.pnl") == 50.0


def test_metrics_recorder_ope():
    """Test MetricsRecorder OPE metrics."""
    backend = InMemoryBackend()
    recorder = MetricsRecorder(backend)

    recorder.record_ope_estimate("DM", 1.5, policy_id="policy_v1")
    recorder.record_ope_estimate("WIS", 1.2, policy_id="policy_v1")

    assert len(backend.points) == 2


def test_metrics_recorder_drift():
    """Test MetricsRecorder drift metrics."""
    backend = InMemoryBackend()
    recorder = MetricsRecorder(backend)

    recorder.record_drift_score("price", 0.05, version="v1")

    assert backend.get_latest("drift.score") == 0.05


def test_metrics_recorder_default_tags():
    """Test that default tags are applied to all metrics."""
    backend = InMemoryBackend()
    recorder = MetricsRecorder(backend, default_tags={"env": "prod", "version": "1.0"})

    recorder.record_td_error(0.1)

    point = backend.points[0]
    assert point.tags["env"] == "prod"
    assert point.tags["version"] == "1.0"


def test_global_recorder():
    """Test global recorder singleton."""
    # Get default recorder
    recorder1 = get_global_recorder()
    recorder2 = get_global_recorder()
    assert recorder1 is recorder2

    # Set custom recorder
    custom_backend = InMemoryBackend()
    custom_recorder = MetricsRecorder(custom_backend)
    set_global_recorder(custom_recorder)

    recorder3 = get_global_recorder()
    assert recorder3 is custom_recorder


def test_metric_point_dataclass():
    """Test MetricPoint dataclass."""
    point = MetricPoint(
        name="test",
        value=1.0,
        timestamp=1234567890.0,
        tags={"key": "value"},
    )

    assert point.name == "test"
    assert point.value == 1.0
    assert point.timestamp == 1234567890.0
    assert point.tags["key"] == "value"


def test_flush_multiple_backends():
    """Test flush propagates to all backends in composite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "metrics1.json"
        path2 = Path(tmpdir) / "metrics2.json"

        backend1 = JSONFileBackend(path1, buffer_size=100)
        backend2 = JSONFileBackend(path2, buffer_size=100)
        composite = CompositeBackend([backend1, backend2])

        composite.record("metric", 1.0)
        composite.flush()

        assert path1.exists()
        assert path2.exists()

