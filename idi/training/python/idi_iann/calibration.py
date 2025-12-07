"""Calibration checks for Q-table policies.

Computes calibration curves and Expected Calibration Error (ECE) to verify
that predicted Q-values are well-calibrated with respect to realized returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .domain import Action, StateKey
from .policy import LookupPolicy


@dataclass
class CalibrationBucket:
    """A single bucket for calibration analysis."""

    predicted_mean: float
    realized_mean: float
    count: int
    predicted_values: List[float] = field(default_factory=list)
    realized_values: List[float] = field(default_factory=list)

    @property
    def calibration_error(self) -> float:
        """Absolute difference between predicted and realized."""
        return abs(self.predicted_mean - self.realized_mean)


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    buckets: List[CalibrationBucket]
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "n_samples": self.n_samples,
            "buckets": [
                {
                    "predicted_mean": b.predicted_mean,
                    "realized_mean": b.realized_mean,
                    "count": b.count,
                    "calibration_error": b.calibration_error,
                }
                for b in self.buckets
            ],
        }


class CalibrationChecker:
    """Compute calibration metrics for Q-table policies."""

    def __init__(self, policy: LookupPolicy, n_buckets: int = 10):
        """Initialize checker.

        Args:
            policy: Policy to evaluate
            n_buckets: Number of buckets for calibration curve
        """
        self.policy = policy
        self.n_buckets = n_buckets

    def compute_calibration(
        self,
        states: List[StateKey],
        actions: List[Action],
        realized_returns: List[float],
    ) -> CalibrationResult:
        """Compute calibration metrics.

        Args:
            states: List of states where actions were taken
            actions: Actions taken at each state
            realized_returns: Actual returns observed from each (state, action)

        Returns:
            CalibrationResult with ECE, MCE, and per-bucket stats
        """
        if not states:
            return CalibrationResult(ece=0.0, mce=0.0, buckets=[], n_samples=0)

        # Get predicted Q-values
        predictions = []
        for state, action in zip(states, actions):
            q = self.policy.q_value(state, action)
            predictions.append(q)

        # Find min/max for bucketing
        all_values = predictions + realized_returns
        min_val = min(all_values) if all_values else 0.0
        max_val = max(all_values) if all_values else 1.0
        bucket_width = (max_val - min_val) / self.n_buckets if max_val > min_val else 1.0

        # Initialize buckets
        buckets_data: List[Tuple[List[float], List[float]]] = [
            ([], []) for _ in range(self.n_buckets)
        ]

        # Assign to buckets based on predicted value
        for pred, real in zip(predictions, realized_returns):
            bucket_idx = int((pred - min_val) / bucket_width)
            bucket_idx = min(bucket_idx, self.n_buckets - 1)
            bucket_idx = max(bucket_idx, 0)
            buckets_data[bucket_idx][0].append(pred)
            buckets_data[bucket_idx][1].append(real)

        # Compute bucket statistics
        buckets = []
        total_samples = len(predictions)
        weighted_error_sum = 0.0
        max_error = 0.0

        for preds, reals in buckets_data:
            if preds:
                pred_mean = sum(preds) / len(preds)
                real_mean = sum(reals) / len(reals)
                bucket = CalibrationBucket(
                    predicted_mean=pred_mean,
                    realized_mean=real_mean,
                    count=len(preds),
                    predicted_values=preds,
                    realized_values=reals,
                )
                buckets.append(bucket)

                error = bucket.calibration_error
                weighted_error_sum += (len(preds) / total_samples) * error
                max_error = max(max_error, error)

        return CalibrationResult(
            ece=weighted_error_sum,
            mce=max_error,
            buckets=buckets,
            n_samples=total_samples,
        )

    def compute_from_rollouts(
        self,
        rollout_data: List[Dict],
        discount: float = 0.99,
    ) -> CalibrationResult:
        """Compute calibration from rollout data.

        Args:
            rollout_data: List of dicts with 'state', 'action', 'rewards' (list of future rewards)
            discount: Discount factor for computing returns

        Returns:
            CalibrationResult
        """
        states = []
        actions = []
        realized_returns = []

        for entry in rollout_data:
            state = tuple(entry["state"])
            action = (
                Action.from_str(entry["action"])
                if isinstance(entry["action"], str)
                else entry["action"]
            )
            rewards = entry.get("rewards", [])

            # Compute discounted return
            ret = 0.0
            factor = 1.0
            for r in rewards:
                ret += factor * r
                factor *= discount

            states.append(state)
            actions.append(action)
            realized_returns.append(ret)

        return self.compute_calibration(states, actions, realized_returns)


def check_calibration(
    policy: LookupPolicy,
    rollout_path: Path,
    n_buckets: int = 10,
    discount: float = 0.99,
    output_path: Optional[Path] = None,
) -> CalibrationResult:
    """Check calibration from rollout data file.

    Args:
        policy: Policy to evaluate
        rollout_path: Path to JSON file with rollout data
        n_buckets: Number of calibration buckets
        discount: Discount factor
        output_path: Optional path to save results

    Returns:
        CalibrationResult
    """
    data = json.loads(rollout_path.read_text())
    rollouts = data.get("rollouts", data)  # Support both wrapped and direct format

    checker = CalibrationChecker(policy, n_buckets=n_buckets)
    result = checker.compute_from_rollouts(rollouts, discount=discount)

    if output_path:
        output_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result

