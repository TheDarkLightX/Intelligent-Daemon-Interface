"""Metrics abstraction for training observability.

Provides a pluggable metrics backend for logging training, environment,
and evaluation metrics during RL training.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricPoint:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for filtering
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics."""
        pass


class InMemoryBackend(MetricsBackend):
    """In-memory metrics backend for testing and development."""

    def __init__(self):
        self.points: List[MetricPoint] = []

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        self.points.append(
            MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
            )
        )

    def flush(self) -> None:
        pass  # No-op for in-memory

    def get_values(self, name: str) -> List[float]:
        """Get all values for a metric name."""
        return [p.value for p in self.points if p.name == name]

    def get_latest(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric name."""
        matching = [p for p in self.points if p.name == name]
        return matching[-1].value if matching else None


class JSONFileBackend(MetricsBackend):
    """JSON file metrics backend for persistence."""

    def __init__(self, path: Path, buffer_size: int = 100):
        self.path = path
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        self.buffer.append({
            "name": name,
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {},
        })
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if self.path.exists():
            existing = json.loads(self.path.read_text())
        existing.extend(self.buffer)
        self.path.write_text(json.dumps(existing, indent=2))
        self.buffer.clear()


class StdoutBackend(MetricsBackend):
    """Stdout metrics backend for debugging."""

    def __init__(self, prefix: str = "[METRIC]"):
        self.prefix = prefix

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        tag_str = " ".join(f"{k}={v}" for k, v in (tags or {}).items())
        print(f"{self.prefix} {name}={value:.4f} {tag_str}".strip())

    def flush(self) -> None:
        pass


class CompositeBackend(MetricsBackend):
    """Composite backend that forwards to multiple backends."""

    def __init__(self, backends: List[MetricsBackend]):
        self.backends = backends

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        for backend in self.backends:
            backend.record(name, value, tags)

    def flush(self) -> None:
        for backend in self.backends:
            backend.flush()


class MetricsRecorder:
    """High-level metrics recorder with common metrics definitions."""

    def __init__(self, backend: Optional[MetricsBackend] = None, default_tags: Optional[Dict[str, str]] = None):
        """Initialize recorder.

        Args:
            backend: Metrics backend (default: InMemoryBackend)
            default_tags: Tags to add to all metrics
        """
        self.backend = backend or InMemoryBackend()
        self.default_tags = default_tags or {}

    def _tags(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        tags = dict(self.default_tags)
        if extra:
            tags.update(extra)
        return tags

    # Training metrics
    def record_td_error(self, value: float, episode: Optional[int] = None) -> None:
        """Record TD error."""
        tags = {}
        if episode is not None:
            tags["episode"] = str(episode)
        self.backend.record("training.td_error", value, self._tags(tags))

    def record_q_value(self, action: str, value: float, episode: Optional[int] = None) -> None:
        """Record Q-value for an action."""
        tags = {"action": action}
        if episode is not None:
            tags["episode"] = str(episode)
        self.backend.record("training.q_value", value, self._tags(tags))

    def record_episode_reward(self, value: float, episode: int) -> None:
        """Record total episode reward."""
        self.backend.record("training.episode_reward", value, self._tags({"episode": str(episode)}))

    def record_exploration(self, value: float, episode: int) -> None:
        """Record exploration rate."""
        self.backend.record("training.exploration", value, self._tags({"episode": str(episode)}))

    # Environment metrics
    def record_env_price(self, value: float, step: int) -> None:
        """Record environment price."""
        self.backend.record("env.price", value, self._tags({"step": str(step)}))

    def record_env_regime(self, regime: str, step: int) -> None:
        """Record environment regime."""
        self.backend.record("env.regime", hash(regime) % 1000 / 1000, self._tags({"step": str(step), "regime": regime}))

    def record_env_pnl(self, value: float, step: int) -> None:
        """Record environment PnL."""
        self.backend.record("env.pnl", value, self._tags({"step": str(step)}))

    # OPE metrics
    def record_ope_estimate(self, estimator: str, value: float, policy_id: str) -> None:
        """Record OPE estimate."""
        self.backend.record("ope.estimate", value, self._tags({"estimator": estimator, "policy_id": policy_id}))

    # Drift metrics
    def record_drift_score(self, feature: str, value: float, version: str) -> None:
        """Record drift score."""
        self.backend.record("drift.score", value, self._tags({"feature": feature, "version": version}))

    # Action distribution
    def record_action_count(self, action: str, count: int, episode: int) -> None:
        """Record action count."""
        self.backend.record("training.action_count", float(count), self._tags({"action": action, "episode": str(episode)}))

    def flush(self) -> None:
        """Flush the backend."""
        self.backend.flush()


# Global singleton for convenience
_global_recorder: Optional[MetricsRecorder] = None


def get_global_recorder() -> MetricsRecorder:
    """Get or create the global metrics recorder."""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = MetricsRecorder()
    return _global_recorder


def set_global_recorder(recorder: MetricsRecorder) -> None:
    """Set the global metrics recorder."""
    global _global_recorder
    _global_recorder = recorder

