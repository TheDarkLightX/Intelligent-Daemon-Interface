"""Structured logging utilities for synth and Auto-QAgent.

This module provides JSON-serializable log records for observability
and diagnostics, per Section 4 of the production readiness checklist.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Configure module-level logger
logger = logging.getLogger("idi.synth")


@dataclass
class SynthRunConfig:
    """Captures configuration at synth run start."""

    beam_width: int
    max_depth: int
    profiles: Tuple[str, ...]
    eval_mode: str
    envs_count: int = 0
    budget_episodes: int = 0


@dataclass
class SynthRunStats:
    """Captures statistics at synth run end."""

    visited: int = 0
    pruned_krr: int = 0
    accepted: int = 0
    frontier_max: int = 0
    depth_levels: int = 0
    duration_seconds: float = 0.0


@dataclass
class SynthCandidate:
    """Represents a single candidate in structured logs."""

    id: str
    score: Tuple[float, ...]
    agent_type: str = "unknown"


@dataclass
class SynthRunLog:
    """Complete structured log for a synth run."""

    run_id: str
    timestamp_iso: str
    config: SynthRunConfig
    stats: SynthRunStats
    candidates: List[SynthCandidate] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp_iso,
            "config": {
                "beam_width": self.config.beam_width,
                "max_depth": self.config.max_depth,
                "profiles": list(self.config.profiles),
                "eval_mode": self.config.eval_mode,
                "envs_count": self.config.envs_count,
                "budget_episodes": self.config.budget_episodes,
            },
            "stats": {
                "visited": self.stats.visited,
                "pruned_krr": self.stats.pruned_krr,
                "accepted": self.stats.accepted,
                "frontier_max": self.stats.frontier_max,
                "depth_levels": self.stats.depth_levels,
                "duration_seconds": self.stats.duration_seconds,
            },
            "candidates": [
                {"id": c.id, "score": list(c.score), "agent_type": c.agent_type}
                for c in self.candidates
            ],
            "error": self.error,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class SynthLogger:
    """Context manager for structured synth logging."""

    def __init__(
        self,
        run_id: str,
        config: SynthRunConfig,
        emit_json: bool = False,
    ):
        self.run_id = run_id
        self.config = config
        self.emit_json = emit_json
        self._start_time: float = 0.0
        self._stats = SynthRunStats()
        self._candidates: List[SynthCandidate] = []
        self._error: Optional[str] = None

    def __enter__(self) -> "SynthLogger":
        self._start_time = time.perf_counter()
        logger.info(
            "synth_run_start",
            extra={"run_id": self.run_id, "config": self.config.__dict__},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._stats.duration_seconds = time.perf_counter() - self._start_time
        if exc_val is not None:
            self._error = str(exc_val)

        log = SynthRunLog(
            run_id=self.run_id,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config=self.config,
            stats=self._stats,
            candidates=self._candidates,
            error=self._error,
        )

        if self.emit_json:
            # FIX: Use logger instead of print to avoid stdout interference
            logger.info("synth_run_json: %s", log.to_json(indent=None))
        else:
            logger.info(
                "synth_run_end",
                extra={"run_id": self.run_id, "stats": self._stats.__dict__},
            )

        return False  # Don't suppress exceptions

    def record_stats(self, stats: Dict[str, float]) -> None:
        """Record search statistics."""
        self._stats.visited = int(stats.get("visited", 0))
        self._stats.pruned_krr = int(stats.get("pruned_krr", 0))
        self._stats.accepted = int(stats.get("accepted", 0))
        self._stats.frontier_max = int(stats.get("frontier_max", 0))
        self._stats.depth_levels = int(stats.get("depth_levels", 0))

    def record_candidate(
        self,
        candidate_id: str,
        score: Tuple[float, ...],
        agent_type: str = "unknown",
    ) -> None:
        """Record a single candidate."""
        self._candidates.append(
            SynthCandidate(id=candidate_id, score=score, agent_type=agent_type)
        )


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp and random suffix.

    FIX: Added random suffix to prevent collisions in high-throughput scenarios.
    """
    return f"synth-{int(time.time() * 1000)}-{random.randint(0, 9999):04d}"
