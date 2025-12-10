"""Tests for synth profiler utilities."""

from __future__ import annotations

import time

import pytest

from idi.devkit.experimental.synth_profiler import (
    ProfileReport,
    ProfileStats,
    SynthProfiler,
    TimingRecord,
    benchmark_evaluator,
    profile_synth_run,
    quick_profile,
    time_it,
)


# ---------------------------------------------------------------------------
# TimingRecord Tests
# ---------------------------------------------------------------------------

class TestTimingRecord:
    """Tests for TimingRecord dataclass."""

    def test_duration_calculation(self) -> None:
        """Duration should be calculated from start/end times."""
        record = TimingRecord(
            name="test",
            start_time=1.0,
            end_time=1.5,
        )

        assert record.duration_s == pytest.approx(0.5)
        assert record.duration_ms == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# ProfileStats Tests
# ---------------------------------------------------------------------------

class TestProfileStats:
    """Tests for ProfileStats dataclass."""

    def test_record_single(self) -> None:
        """Should record a single timing."""
        stats = ProfileStats(name="test")
        stats.record(100.0)

        assert stats.count == 1
        assert stats.total_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.avg_ms == 100.0

    def test_record_multiple(self) -> None:
        """Should aggregate multiple timings."""
        stats = ProfileStats(name="test")
        stats.record(100.0)
        stats.record(200.0)
        stats.record(150.0)

        assert stats.count == 3
        assert stats.total_ms == 450.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 200.0
        assert stats.avg_ms == 150.0

    def test_avg_with_no_records(self) -> None:
        """Average should be 0 with no records."""
        stats = ProfileStats(name="test")

        assert stats.avg_ms == 0.0


# ---------------------------------------------------------------------------
# SynthProfiler Tests
# ---------------------------------------------------------------------------

class TestSynthProfiler:
    """Tests for SynthProfiler context manager."""

    def test_context_manager_basic(self) -> None:
        """Should measure total duration."""
        with SynthProfiler() as profiler:
            time.sleep(0.01)

        report = profiler.get_report()
        assert report.total_duration_s >= 0.01

    def test_measure_operation(self) -> None:
        """Should measure individual operations."""
        with SynthProfiler() as profiler:
            with profiler.measure("op1"):
                time.sleep(0.01)
            with profiler.measure("op2"):
                time.sleep(0.02)

        report = profiler.get_report()

        assert "op1" in report.operation_stats
        assert "op2" in report.operation_stats
        assert report.operation_stats["op1"].count == 1
        assert report.operation_stats["op2"].count == 1
        assert report.operation_stats["op1"].total_ms >= 10
        assert report.operation_stats["op2"].total_ms >= 20

    def test_measure_same_operation_multiple_times(self) -> None:
        """Should aggregate same operation name."""
        with SynthProfiler() as profiler:
            for _ in range(3):
                with profiler.measure("repeated"):
                    time.sleep(0.005)

        report = profiler.get_report()

        assert report.operation_stats["repeated"].count == 3

    def test_record_metadata(self) -> None:
        """Should record custom metadata."""
        with SynthProfiler() as profiler:
            profiler.record_metadata("beam_width", 4)
            profiler.record_metadata("max_depth", 3)

        report = profiler.get_report()

        assert report.metadata["beam_width"] == 4
        assert report.metadata["max_depth"] == 3

    def test_timeline_preserved(self) -> None:
        """Should preserve operation order in timeline."""
        with SynthProfiler() as profiler:
            with profiler.measure("first"):
                pass
            with profiler.measure("second"):
                pass
            with profiler.measure("third"):
                pass

        report = profiler.get_report()

        assert len(report.timeline) == 3
        assert report.timeline[0].name == "first"
        assert report.timeline[1].name == "second"
        assert report.timeline[2].name == "third"


# ---------------------------------------------------------------------------
# ProfileReport Tests
# ---------------------------------------------------------------------------

class TestProfileReport:
    """Tests for ProfileReport."""

    def test_summary_generation(self) -> None:
        """Should generate readable summary."""
        with SynthProfiler() as profiler:
            with profiler.measure("test_op"):
                time.sleep(0.01)

        report = profiler.get_report()
        summary = report.summary()

        assert "SYNTH PERFORMANCE PROFILE" in summary
        assert "test_op" in summary
        assert "Total Duration" in summary

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        with SynthProfiler() as profiler:
            with profiler.measure("test_op"):
                pass

        report = profiler.get_report()
        d = report.to_dict()

        assert "total_duration_s" in d
        assert "operations" in d
        assert "test_op" in d["operations"]


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for helper profiling functions."""

    def test_profile_synth_run(self) -> None:
        """Should profile a function call."""
        def mock_synth(x):
            time.sleep(0.01)
            return x * 2

        result, report = profile_synth_run(mock_synth, 5)

        assert result == 10
        assert report.total_duration_s >= 0.01

    def test_time_it(self) -> None:
        """Should time a single call."""
        def slow_func():
            time.sleep(0.01)
            return 42

        result, duration_ms = time_it(slow_func)

        assert result == 42
        assert duration_ms >= 10

    def test_benchmark_evaluator(self) -> None:
        """Should benchmark across patches."""
        patches = ["a", "b", "c"]
        call_count = [0]

        def mock_eval(patch):
            call_count[0] += 1
            return len(patch)

        report = benchmark_evaluator(mock_eval, patches, iterations=2)

        # 3 patches × 2 iterations = 6 calls
        assert call_count[0] == 6
        assert report.operation_stats["evaluator_call"].count == 6
        assert report.metadata["num_patches"] == 3
        assert report.metadata["iterations"] == 2

    def test_quick_profile_decorator(self, capsys) -> None:
        """Quick profile should print timing."""
        @quick_profile("test_func")
        def test_func():
            return 123

        result = test_func()

        assert result == 123
        captured = capsys.readouterr()
        assert "[PROFILE] test_func" in captured.err
