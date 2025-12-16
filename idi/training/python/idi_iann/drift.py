"""Distribution shift detection for RL training data.

Computes drift metrics (KS, Wasserstein, PSI) between training data,
simulator outputs, and new logs to detect distribution shifts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class FeatureStats:
    """Statistics for a single feature."""

    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    n_samples: int
    histogram: List[int] = field(default_factory=list)
    bin_edges: List[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, name: str, values: Sequence[float], n_bins: int = 10) -> "FeatureStats":
        """Compute stats from raw values."""
        if not values:
            return cls(name=name, mean=0.0, std=0.0, min_val=0.0, max_val=0.0, n_samples=0)

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
        std = math.sqrt(variance)
        min_val = min(values)
        max_val = max(values)

        # Compute histogram
        safe_bins = max(1, int(n_bins))
        if max_val > min_val:
            bin_width = (max_val - min_val) / safe_bins
            if not math.isfinite(bin_width) or bin_width <= 0.0:
                # Degenerate range (e.g., subnormal underflow) -> single bin.
                bin_edges = [min_val, max_val + 1e-9]
                histogram = [n]
            else:
                bin_edges = [min_val + i * bin_width for i in range(safe_bins + 1)]
                histogram = [0] * safe_bins
                for v in values:
                    idx = int((v - min_val) / bin_width)
                    if idx < 0:
                        idx = 0
                    if idx >= safe_bins:
                        idx = safe_bins - 1
                    histogram[idx] += 1
        else:
            bin_edges = [min_val, max_val + 1e-9]
            histogram = [n]

        return cls(
            name=name,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            n_samples=n,
            histogram=histogram,
            bin_edges=bin_edges,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "n_samples": self.n_samples,
            "histogram": self.histogram,
            "bin_edges": self.bin_edges,
        }


@dataclass
class DriftMetrics:
    """Drift metrics between two distributions."""

    feature_name: str
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    psi: float  # Population Stability Index
    wasserstein_approx: float  # Approximate Wasserstein distance
    mean_diff: float
    std_diff: float

    @property
    def is_significant(self) -> bool:
        """Check if drift is significant (PSI > 0.1 or KS > 0.1)."""
        return self.psi > 0.1 or self.ks_statistic > 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feature": self.feature_name,
            "ks_statistic": self.ks_statistic,
            "psi": self.psi,
            "wasserstein_approx": self.wasserstein_approx,
            "mean_diff": self.mean_diff,
            "std_diff": self.std_diff,
            "is_significant": self.is_significant,
        }


@dataclass
class ShiftReport:
    """Complete distribution shift report."""

    reference_version: str
    comparison_version: str
    feature_metrics: List[DriftMetrics]
    overall_score: float  # Aggregate drift score
    timestamp: str = ""

    @property
    def has_significant_drift(self) -> bool:
        """Check if any feature has significant drift."""
        return any(m.is_significant for m in self.feature_metrics)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "reference_version": self.reference_version,
            "comparison_version": self.comparison_version,
            "overall_score": self.overall_score,
            "has_significant_drift": self.has_significant_drift,
            "feature_metrics": [m.to_dict() for m in self.feature_metrics],
            "timestamp": self.timestamp,
        }


class DriftDetector:
    """Detect distribution shift between datasets."""

    def __init__(self, n_bins: int = 10):
        """Initialize detector.

        Args:
            n_bins: Number of bins for histogram-based metrics
        """
        self.n_bins = n_bins

    def _compute_ks_statistic(
        self, ref_hist: List[int], comp_hist: List[int]
    ) -> float:
        """Compute KS statistic from histograms."""
        ref_total = sum(ref_hist) or 1
        comp_total = sum(comp_hist) or 1

        ref_cdf = []
        comp_cdf = []
        ref_cum = 0
        comp_cum = 0

        for r, c in zip(ref_hist, comp_hist):
            ref_cum += r / ref_total
            comp_cum += c / comp_total
            ref_cdf.append(ref_cum)
            comp_cdf.append(comp_cum)

        if not ref_cdf:
            return 0.0

        return max(abs(r - c) for r, c in zip(ref_cdf, comp_cdf))

    def _compute_psi(self, ref_hist: List[int], comp_hist: List[int]) -> float:
        """Compute Population Stability Index."""
        ref_total = sum(ref_hist) or 1
        comp_total = sum(comp_hist) or 1

        psi = 0.0
        for r, c in zip(ref_hist, comp_hist):
            ref_pct = (r + 0.5) / (ref_total + 0.5 * len(ref_hist))
            comp_pct = (c + 0.5) / (comp_total + 0.5 * len(comp_hist))
            psi += (comp_pct - ref_pct) * math.log(comp_pct / ref_pct + 1e-9)

        return psi

    def _compute_wasserstein_approx(
        self, ref_stats: FeatureStats, comp_stats: FeatureStats
    ) -> float:
        """Approximate Wasserstein distance using mean and std."""
        mean_diff = abs(ref_stats.mean - comp_stats.mean)
        std_diff = abs(ref_stats.std - comp_stats.std)
        return mean_diff + std_diff

    def compare_features(
        self, ref_stats: FeatureStats, comp_stats: FeatureStats
    ) -> DriftMetrics:
        """Compare two feature distributions."""
        # Align histograms to same bin edges
        if len(ref_stats.histogram) != len(comp_stats.histogram):
            # Use reference histogram length
            comp_hist = comp_stats.histogram[: len(ref_stats.histogram)]
            comp_hist.extend([0] * (len(ref_stats.histogram) - len(comp_hist)))
        else:
            comp_hist = comp_stats.histogram

        ks = self._compute_ks_statistic(ref_stats.histogram, comp_hist)
        psi = self._compute_psi(ref_stats.histogram, comp_hist)
        wasserstein = self._compute_wasserstein_approx(ref_stats, comp_stats)

        return DriftMetrics(
            feature_name=ref_stats.name,
            ks_statistic=ks,
            psi=psi,
            wasserstein_approx=wasserstein,
            mean_diff=comp_stats.mean - ref_stats.mean,
            std_diff=comp_stats.std - ref_stats.std,
        )

    def generate_report(
        self,
        reference_data: Dict[str, Sequence[float]],
        comparison_data: Dict[str, Sequence[float]],
        reference_version: str = "training",
        comparison_version: str = "new",
    ) -> ShiftReport:
        """Generate a drift report comparing two datasets.

        Args:
            reference_data: Dict mapping feature name to values (reference/training)
            comparison_data: Dict mapping feature name to values (comparison/new)
            reference_version: Version label for reference data
            comparison_version: Version label for comparison data

        Returns:
            ShiftReport with drift metrics for each feature
        """
        feature_metrics = []
        total_score = 0.0

        for name in reference_data.keys():
            if name not in comparison_data:
                continue

            ref_stats = FeatureStats.from_values(
                name, reference_data[name], self.n_bins
            )
            comp_stats = FeatureStats.from_values(
                name, comparison_data[name], self.n_bins
            )

            metrics = self.compare_features(ref_stats, comp_stats)
            feature_metrics.append(metrics)
            total_score += metrics.psi

        overall_score = total_score / max(1, len(feature_metrics))

        import datetime
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return ShiftReport(
            reference_version=reference_version,
            comparison_version=comparison_version,
            feature_metrics=feature_metrics,
            overall_score=overall_score,
            timestamp=timestamp,
        )


def compute_drift_report(
    reference_path: Path,
    comparison_path: Path,
    output_path: Optional[Path] = None,
    feature_names: Optional[List[str]] = None,
) -> ShiftReport:
    """Compute drift report from data files.

    Args:
        reference_path: Path to reference data JSON
        comparison_path: Path to comparison data JSON
        output_path: Optional path to save report
        feature_names: Optional list of features to compare (default: all)

    Returns:
        ShiftReport with drift metrics
    """
    ref_data = json.loads(reference_path.read_text())
    comp_data = json.loads(comparison_path.read_text())

    # Extract feature values
    ref_features = ref_data.get("features", ref_data)
    comp_features = comp_data.get("features", comp_data)

    if feature_names:
        ref_features = {k: v for k, v in ref_features.items() if k in feature_names}
        comp_features = {k: v for k, v in comp_features.items() if k in feature_names}

    detector = DriftDetector()
    report = detector.generate_report(
        ref_features,
        comp_features,
        reference_version=ref_data.get("version", str(reference_path)),
        comparison_version=comp_data.get("version", str(comparison_path)),
    )

    if output_path:
        output_path.write_text(json.dumps(report.to_dict(), indent=2))

    return report


def extract_state_features(
    states: List[Tuple[int, ...]], feature_names: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """Extract feature values from state tuples.

    Args:
        states: List of state tuples
        feature_names: Optional feature names (default: price, volume, trend, scarcity, mood)

    Returns:
        Dict mapping feature name to list of values
    """
    if feature_names is None:
        feature_names = ["price", "volume", "trend", "scarcity", "mood"]

    features: Dict[str, List[float]] = {name: [] for name in feature_names}

    for state in states:
        for i, name in enumerate(feature_names):
            if i < len(state):
                features[name].append(float(state[i]))

    return features

