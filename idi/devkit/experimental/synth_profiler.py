"""Performance profiling utilities for synth and Auto-QAgent.

This module provides tools for measuring and analyzing performance of
synth operations, per Section 3 of the production readiness checklist.

Key measurements:
- Wall-clock time for synth operations
- Per-candidate evaluation time
- KRR evaluation overhead
- Memory usage estimates

Usage:
    from idi.devkit.experimental.synth_profiler import SynthProfiler

    with SynthProfiler() as profiler:
        results = run_auto_qagent_synth(goal)

    print(profiler.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Profiling Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """Record of a single timed operation."""

    name: str
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class ProfileStats:
    """Aggregated statistics for a profiled operation type."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Average duration in milliseconds."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def record(self, duration_ms: float) -> None:
        """Record a new timing."""
        self.count += 1
        self.total_ms += duration_ms
        if duration_ms < self.min_ms:
            self.min_ms = duration_ms
        if duration_ms > self.max_ms:
            self.max_ms = duration_ms


@dataclass
class ProfileReport:
    """Complete profiling report for a synth run."""

    total_duration_s: float
    operation_stats: Dict[str, ProfileStats]
    timeline: List[TimingRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "SYNTH PERFORMANCE PROFILE",
            "=" * 60,
            f"Total Duration: {self.total_duration_s:.3f}s ({self.total_duration_s * 1000:.1f}ms)",
            "",
            "Operation Breakdown:",
            "-" * 60,
        ]

        # Sort by total time
        sorted_stats = sorted(
            self.operation_stats.values(),
            key=lambda s: s.total_ms,
            reverse=True,
        )

        for stats in sorted_stats:
            pct = (stats.total_ms / (self.total_duration_s * 1000)) * 100 if self.total_duration_s > 0 else 0
            lines.append(
                f"  {stats.name:30s} "
                f"count={stats.count:4d}  "
                f"total={stats.total_ms:8.1f}ms ({pct:5.1f}%)  "
                f"avg={stats.avg_ms:6.2f}ms  "
                f"min={stats.min_ms:6.2f}ms  "
                f"max={stats.max_ms:6.2f}ms"
            )

        lines.append("-" * 60)

        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_duration_s": self.total_duration_s,
            "operations": {
                name: {
                    "count": stats.count,
                    "total_ms": stats.total_ms,
                    "avg_ms": stats.avg_ms,
                    "min_ms": stats.min_ms if stats.min_ms != float("inf") else 0.0,
                    "max_ms": stats.max_ms,
                }
                for name, stats in self.operation_stats.items()
            },
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Profiler Implementation
# ---------------------------------------------------------------------------

class SynthProfiler:
    """Context manager for profiling synth operations.

    Example:
        with SynthProfiler() as profiler:
            # Profile specific operations
            with profiler.measure("krr_eval"):
                evaluate_with_krr(...)

            with profiler.measure("patch_eval"):
                evaluate_patch_real(...)

        report = profiler.get_report()
        print(report.summary())
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._timeline: List[TimingRecord] = []
        self._stats: Dict[str, ProfileStats] = {}
        self._metadata: Dict[str, Any] = {}
        self._active: bool = False

    def __enter__(self) -> "SynthProfiler":
        self._start_time = time.perf_counter()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._end_time = time.perf_counter()
        self._active = False
        return False

    @contextmanager
    def measure(self, operation: str, **metadata):
        """Context manager to measure a single operation.

        Args:
            operation: Name of the operation being measured.
            **metadata: Additional metadata to record.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            record = TimingRecord(
                name=operation,
                start_time=start,
                end_time=end,
                metadata=metadata,
            )
            self._timeline.append(record)

            if operation not in self._stats:
                self._stats[operation] = ProfileStats(name=operation)
            self._stats[operation].record(record.duration_ms)

    def measure_func(self, operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to measure function execution time.

        Args:
            operation: Name for the operation.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs) -> T:
                with self.measure(operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record_metadata(self, key: str, value: Any) -> None:
        """Record additional metadata."""
        self._metadata[key] = value

    def get_report(self) -> ProfileReport:
        """Generate the profiling report."""
        total_duration = self._end_time - self._start_time if self._end_time > 0 else 0.0
        return ProfileReport(
            total_duration_s=total_duration,
            operation_stats=dict(self._stats),
            timeline=list(self._timeline),
            metadata=dict(self._metadata),
        )

    def summary(self) -> str:
        """Generate a summary string."""
        return self.get_report().summary()


# ---------------------------------------------------------------------------
# Profiled Wrappers for Common Operations
# ---------------------------------------------------------------------------

def profile_synth_run(
    synth_fn: Callable[..., T],
    *args,
    **kwargs,
) -> tuple[T, ProfileReport]:
    """Profile a synth function call.

    Args:
        synth_fn: The synth function to call.
        *args: Arguments to pass to synth_fn.
        **kwargs: Keyword arguments to pass to synth_fn.

    Returns:
        Tuple of (result, ProfileReport).
    """
    with SynthProfiler() as profiler:
        result = synth_fn(*args, **kwargs)

    return result, profiler.get_report()


def benchmark_evaluator(
    evaluator: Callable,
    patches: List[Any],
    iterations: int = 3,
) -> ProfileReport:
    """Benchmark an evaluator function across multiple patches.

    Args:
        evaluator: The evaluator function to benchmark.
        patches: List of patches to evaluate.
        iterations: Number of iterations per patch.

    Returns:
        ProfileReport with benchmark results.
    """
    with SynthProfiler() as profiler:
        profiler.record_metadata("num_patches", len(patches))
        profiler.record_metadata("iterations", iterations)

        for i, patch in enumerate(patches):
            for j in range(iterations):
                with profiler.measure("evaluator_call", patch_idx=i, iteration=j):
                    try:
                        evaluator(patch)
                    except Exception:
                        pass  # Record timing even on failure

    return profiler.get_report()


# ---------------------------------------------------------------------------
# Quick Profiling Functions
# ---------------------------------------------------------------------------

def quick_profile(label: str = "operation"):
    """Simple decorator for quick profiling.

    Prints timing to stderr after each call.

    Example:
        @quick_profile("my_func")
        def my_func():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                import sys
                print(f"[PROFILE] {label}: {duration_ms:.2f}ms", file=sys.stderr)
        return wrapper
    return decorator


def time_it(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """Time a single function call.

    Args:
        func: Function to call.
        *args: Arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tuple of (result, duration_ms).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms
