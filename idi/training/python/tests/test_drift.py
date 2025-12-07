"""Tests for distribution drift detection."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.drift import (
    DriftDetector,
    DriftMetrics,
    FeatureStats,
    ShiftReport,
    compute_drift_report,
    extract_state_features,
)


def test_feature_stats_basic():
    """Test basic feature statistics computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = FeatureStats.from_values("test", values, n_bins=5)

    assert stats.name == "test"
    assert stats.mean == 3.0
    assert stats.min_val == 1.0
    assert stats.max_val == 5.0
    assert stats.n_samples == 5
    assert len(stats.histogram) == 5


def test_feature_stats_empty():
    """Test feature stats with empty values."""
    stats = FeatureStats.from_values("empty", [], n_bins=5)

    assert stats.n_samples == 0
    assert stats.mean == 0.0


def test_drift_detector_no_drift():
    """Test drift detection with identical distributions."""
    detector = DriftDetector(n_bins=5)

    ref_stats = FeatureStats.from_values("feature", [1.0, 2.0, 3.0, 4.0, 5.0])
    comp_stats = FeatureStats.from_values("feature", [1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = detector.compare_features(ref_stats, comp_stats)

    assert metrics.ks_statistic < 0.1
    assert not metrics.is_significant


def test_drift_detector_with_drift():
    """Test drift detection with different distributions."""
    detector = DriftDetector(n_bins=5)

    ref_stats = FeatureStats.from_values("feature", [1.0, 2.0, 3.0, 4.0, 5.0])
    comp_stats = FeatureStats.from_values("feature", [10.0, 20.0, 30.0, 40.0, 50.0])

    metrics = detector.compare_features(ref_stats, comp_stats)

    # Significantly different distributions
    assert metrics.mean_diff > 5.0
    assert metrics.wasserstein_approx > 1.0


def test_generate_shift_report():
    """Test full shift report generation."""
    detector = DriftDetector()

    reference = {
        "price": [1.0, 2.0, 3.0, 4.0, 5.0],
        "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
    }
    comparison = {
        "price": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
    }

    report = detector.generate_report(reference, comparison)

    assert len(report.feature_metrics) == 2
    assert report.reference_version == "training"
    assert report.comparison_version == "new"


def test_shift_report_serialization():
    """Test shift report to_dict method."""
    detector = DriftDetector()
    report = detector.generate_report(
        {"feature": [1.0, 2.0, 3.0]},
        {"feature": [2.0, 3.0, 4.0]},
    )

    d = report.to_dict()

    assert "reference_version" in d
    assert "overall_score" in d
    assert "feature_metrics" in d
    assert "timestamp" in d


def test_compute_drift_report_file():
    """Test drift report with file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.json"
        comp_path = Path(tmpdir) / "comparison.json"
        output_path = Path(tmpdir) / "report.json"

        ref_data = {"features": {"price": [1.0, 2.0, 3.0, 4.0, 5.0]}}
        comp_data = {"features": {"price": [2.0, 3.0, 4.0, 5.0, 6.0]}}

        ref_path.write_text(json.dumps(ref_data))
        comp_path.write_text(json.dumps(comp_data))

        report = compute_drift_report(ref_path, comp_path, output_path)

        assert output_path.exists()
        assert len(report.feature_metrics) == 1

        output = json.loads(output_path.read_text())
        assert "overall_score" in output


def test_extract_state_features():
    """Test state feature extraction."""
    states = [
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (2, 3, 4, 5, 6),
    ]

    features = extract_state_features(states)

    assert "price" in features
    assert "volume" in features
    assert len(features["price"]) == 3
    assert features["price"] == [0.0, 1.0, 2.0]


def test_significant_drift_detection():
    """Test that significant drift is correctly flagged."""
    metrics = DriftMetrics(
        feature_name="test",
        ks_statistic=0.5,
        psi=0.3,
        wasserstein_approx=10.0,
        mean_diff=5.0,
        std_diff=2.0,
    )

    assert metrics.is_significant

    metrics_low = DriftMetrics(
        feature_name="test",
        ks_statistic=0.05,
        psi=0.05,
        wasserstein_approx=0.1,
        mean_diff=0.1,
        std_diff=0.05,
    )

    assert not metrics_low.is_significant

