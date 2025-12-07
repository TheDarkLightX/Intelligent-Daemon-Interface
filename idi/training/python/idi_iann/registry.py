"""Experiment and policy registry for tracking training runs.

Provides lightweight experiment tracking with JSON-based storage.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """A single experiment record."""

    experiment_id: str
    config_hash: str
    git_commit: Optional[str]
    data_version: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: str = "running"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "data_version": self.data_version,
            "timestamp": self.timestamp,
            "config": self.config,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        """Create from dictionary."""
        return cls(
            experiment_id=d["experiment_id"],
            config_hash=d["config_hash"],
            git_commit=d.get("git_commit"),
            data_version=d.get("data_version", "1.0"),
            timestamp=d["timestamp"],
            config=d.get("config", {}),
            metrics=d.get("metrics", {}),
            artifacts=d.get("artifacts", {}),
            tags=d.get("tags", []),
            status=d.get("status", "completed"),
        )


class ExperimentRegistry:
    """Registry for tracking experiments and policies."""

    def __init__(self, registry_dir: Path):
        """Initialize registry.

        Args:
            registry_dir: Directory to store experiment records
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = registry_dir / "index.json"
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load experiment index."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}

    def _save_index(self) -> None:
        """Save experiment index."""
        self.index_file.write_text(json.dumps(self._index, indent=2))

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _generate_id(self, config_hash: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"exp_{timestamp}_{config_hash[:8]}"

    def create_experiment(
        self,
        config: Dict[str, Any],
        data_version: str = "1.0",
        tags: Optional[List[str]] = None,
    ) -> ExperimentRecord:
        """Create a new experiment record.

        Args:
            config: Experiment configuration
            data_version: Version of training data
            tags: Optional tags for filtering

        Returns:
            ExperimentRecord for the new experiment
        """
        config_hash = self._hash_config(config)
        experiment_id = self._generate_id(config_hash)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            config_hash=config_hash,
            git_commit=self._get_git_commit(),
            data_version=data_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=config,
            tags=tags or [],
            status="running",
        )

        self._save_record(record)
        return record

    def _save_record(self, record: ExperimentRecord) -> None:
        """Save experiment record to file."""
        record_file = self.registry_dir / f"{record.experiment_id}.json"
        record_file.write_text(json.dumps(record.to_dict(), indent=2))
        self._index[record.experiment_id] = record_file.name
        self._save_index()

    def update_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            metrics: Metrics to add/update
        """
        record = self.get_experiment(experiment_id)
        if record:
            record.metrics.update(metrics)
            self._save_record(record)

    def add_artifact(self, experiment_id: str, name: str, path: str) -> None:
        """Add artifact reference to experiment.

        Args:
            experiment_id: Experiment ID
            name: Artifact name
            path: Path to artifact
        """
        record = self.get_experiment(experiment_id)
        if record:
            record.artifacts[name] = path
            self._save_record(record)

    def complete_experiment(self, experiment_id: str, status: str = "completed") -> None:
        """Mark experiment as completed.

        Args:
            experiment_id: Experiment ID
            status: Final status (completed, failed, cancelled)
        """
        record = self.get_experiment(experiment_id)
        if record:
            record.status = status
            self._save_record(record)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment record by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentRecord or None if not found
        """
        if experiment_id not in self._index:
            return None
        record_file = self.registry_dir / self._index[experiment_id]
        if not record_file.exists():
            return None
        return ExperimentRecord.from_dict(json.loads(record_file.read_text()))

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[ExperimentRecord]:
        """List experiments with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (any match)
            limit: Maximum number of results

        Returns:
            List of matching ExperimentRecords
        """
        results = []
        for exp_id in list(self._index.keys())[-limit:]:
            record = self.get_experiment(exp_id)
            if record is None:
                continue
            if status and record.status != status:
                continue
            if tags and not any(t in record.tags for t in tags):
                continue
            results.append(record)
        return results

    def find_by_config_hash(self, config_hash: str) -> List[ExperimentRecord]:
        """Find experiments with matching config hash.

        Args:
            config_hash: Configuration hash to match

        Returns:
            List of matching ExperimentRecords
        """
        return [r for r in self.list_experiments() if r.config_hash.startswith(config_hash)]

    def get_best_experiment(self, metric: str, maximize: bool = True) -> Optional[ExperimentRecord]:
        """Get the experiment with best value for a metric.

        Args:
            metric: Metric name to optimize
            maximize: If True, find max; if False, find min

        Returns:
            Best ExperimentRecord or None
        """
        completed = self.list_experiments(status="completed")
        with_metric = [r for r in completed if metric in r.metrics]
        if not with_metric:
            return None
        return max(with_metric, key=lambda r: r.metrics[metric] * (1 if maximize else -1))


@dataclass
class PolicyBundle:
    """A trained policy bundle with metadata."""

    bundle_id: str
    experiment_id: str
    policy_path: str
    trace_path: Optional[str]
    config_hash: str
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bundle_id": self.bundle_id,
            "experiment_id": self.experiment_id,
            "policy_path": self.policy_path,
            "trace_path": self.trace_path,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "hashes": self.hashes,
        }


class PolicyRegistry:
    """Registry for trained policy bundles."""

    def __init__(self, registry_dir: Path):
        """Initialize policy registry.

        Args:
            registry_dir: Directory to store policy records
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = registry_dir / "policies.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load policy index."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}

    def _save_index(self) -> None:
        """Save policy index."""
        self.index_file.write_text(json.dumps(self._index, indent=2))

    def register_policy(
        self,
        experiment_id: str,
        policy_path: str,
        config_hash: str,
        trace_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hashes: Optional[Dict[str, str]] = None,
    ) -> PolicyBundle:
        """Register a trained policy bundle.

        Args:
            experiment_id: Source experiment ID
            policy_path: Path to serialized policy
            config_hash: Configuration hash
            trace_path: Optional path to generated traces
            metrics: Optional evaluation metrics
            hashes: Optional artifact hashes

        Returns:
            PolicyBundle record
        """
        timestamp = datetime.now(timezone.utc)
        bundle_id = f"policy_{timestamp.strftime('%Y%m%d_%H%M%S')}_{config_hash[:8]}"

        bundle = PolicyBundle(
            bundle_id=bundle_id,
            experiment_id=experiment_id,
            policy_path=policy_path,
            trace_path=trace_path,
            config_hash=config_hash,
            timestamp=timestamp.isoformat(),
            metrics=metrics or {},
            hashes=hashes or {},
        )

        self._index[bundle_id] = bundle.to_dict()
        self._save_index()
        return bundle

    def get_policy(self, bundle_id: str) -> Optional[PolicyBundle]:
        """Get policy bundle by ID."""
        if bundle_id not in self._index:
            return None
        d = self._index[bundle_id]
        return PolicyBundle(
            bundle_id=d["bundle_id"],
            experiment_id=d["experiment_id"],
            policy_path=d["policy_path"],
            trace_path=d.get("trace_path"),
            config_hash=d["config_hash"],
            timestamp=d["timestamp"],
            metrics=d.get("metrics", {}),
            hashes=d.get("hashes", {}),
        )

    def list_policies(self, limit: int = 50) -> List[PolicyBundle]:
        """List registered policies."""
        return [self.get_policy(bid) for bid in list(self._index.keys())[-limit:] if bid in self._index]

