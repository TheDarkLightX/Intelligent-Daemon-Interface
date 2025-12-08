"""Performance monitoring and profiling utilities for Tau factory.

Tracks generation times, memory usage, and provides optimization insights.
"""

from __future__ import annotations

import time
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""

    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_start_mb: float
    memory_end_mb: float
    memory_delta_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_peak_mb(self) -> float:
        """Estimate peak memory usage (simplified)."""
        return max(self.memory_start_mb, self.memory_end_mb)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    operation: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = 0.0
    total_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    def update(self, metrics: PerformanceMetrics) -> None:
        """Update stats with new metrics."""
        self.call_count += 1
        self.total_time_ms += metrics.duration_ms
        self.total_memory_mb += metrics.memory_delta_mb

        if self.call_count == 1:
            self.max_time_ms = self.min_time_ms = metrics.duration_ms
        else:
            self.max_time_ms = max(self.max_time_ms, metrics.duration_ms)
            self.min_time_ms = min(self.min_time_ms, metrics.duration_ms)

        self.avg_time_ms = self.total_time_ms / self.call_count
        self.avg_memory_mb = self.total_memory_mb / self.call_count


class PerformanceMonitor:
    """Performance monitoring and profiling for Tau factory operations."""

    def __init__(self) -> None:
        self.metrics: List[PerformanceMetrics] = []
        self.stats: Dict[str, PerformanceStats] = {}
        self.enabled = os.environ.get("TAU_PERF_MONITOR", "0") == "1"

    def set_enabled(self, enabled: bool) -> None:
        """Manually set monitoring enabled state."""
        self.enabled = enabled

    @contextmanager
    def measure(self, operation: str, **metadata: Any):
        """Context manager to measure performance of an operation."""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        memory_start = 0.0

        if HAS_PSUTIL:
            process = psutil.Process()
            memory_start = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.perf_counter()
            memory_end = 0.0

            if HAS_PSUTIL:
                memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            metrics = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                memory_start_mb=memory_start,
                memory_end_mb=memory_end,
                memory_delta_mb=memory_end - memory_start,
                metadata=metadata
            )

            self.metrics.append(metrics)
            self._update_stats(metrics)

    def _update_stats(self, metrics: PerformanceMetrics) -> None:
        """Update aggregated statistics."""
        if metrics.operation not in self.stats:
            self.stats[metrics.operation] = PerformanceStats(metrics.operation)

        self.stats[metrics.operation].update(metrics)

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, PerformanceStats]:
        """Get performance statistics."""
        if operation:
            return {operation: self.stats.get(operation, PerformanceStats(operation))}
        return self.stats.copy()

    def report(self) -> str:
        """Generate a performance report."""
        if not self.enabled:
            return "Performance monitoring disabled (set TAU_PERF_MONITOR=1 to enable)"

        lines = ["=== Tau Factory Performance Report ==="]

        for op, stats in sorted(self.stats.items()):
            lines.append(
                f"\n{op}: calls={stats.call_count}, "
                f"avg={stats.avg_time_ms:.2f}ms, "
                f"min={stats.min_time_ms:.2f}ms, "
                f"max={stats.max_time_ms:.2f}ms, "
                f"avg_mem={stats.avg_memory_mb:.2f}MB"
            )

        lines.append(f"\nTotal operations tracked: {len(self.metrics)}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics and statistics."""
        self.metrics.clear()
        self.stats.clear()


# Global monitor instance
monitor = PerformanceMonitor()
