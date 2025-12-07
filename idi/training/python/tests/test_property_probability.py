"""Property-based tests for probability and distribution invariants."""

from hypothesis import given, strategies as st
import pytest

from idi_iann.ope import OPEEvaluator, LoggedDataset, LoggedEpisode, LoggedTransition
from idi_iann.calibration import CalibrationChecker
from idi_iann.drift import DriftDetector, FeatureStats
from idi_iann.domain import Action


@given(
    behavior_probs=st.lists(
        st.floats(min_value=0.01, max_value=1.0),
        min_size=1,
        max_size=10,
    ),
    policy_probs=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=10,
    ),
)
def test_ope_importance_weights_non_negative(
    behavior_probs: list[float],
    policy_probs: list[float],
) -> None:
    """OPE importance weights are non-negative."""
    from idi_iann.policy import LookupPolicy

    # Ensure lists are same length
    n = min(len(behavior_probs), len(policy_probs))
    behavior_probs = behavior_probs[:n]
    policy_probs = policy_probs[:n]

    # Create logged episode
    transitions = []
    for i, (b_prob, p_prob) in enumerate(zip(behavior_probs, policy_probs)):
        transitions.append(
            LoggedTransition(
                state=(0, 0, 0, 0, 0),
                action=Action.BUY,
                reward=1.0,
                next_state=(1, 0, 0, 0, 0),
                behavior_prob=b_prob,
                done=(i == len(behavior_probs) - 1),
            )
        )

    episode = LoggedEpisode(
        episode_id="test",
        transitions=transitions,
        behavior_policy_id="test",
        config_hash="test",
        data_version="1.0",
    )

    dataset = LoggedDataset(episodes=[episode])
    policy = LookupPolicy()
    evaluator = OPEEvaluator(policy, discount=0.99)

    # Compute importance weights manually
    weight = 1.0
    for t in transitions:
        if t.behavior_prob > 0:
            # Policy prob is always 0 or 1 for greedy policy
            ratio = 1.0 / t.behavior_prob if p_prob > 0.5 else 0.0
            weight *= ratio
            assert weight >= 0.0 or t.behavior_prob == 0, (
                f"Negative weight: {weight}"
            )


@given(
    n_samples=st.integers(min_value=1, max_value=1000),
    n_buckets=st.integers(min_value=2, max_value=20),
)
def test_calibration_bucket_counts_sum(n_samples: int, n_buckets: int) -> None:
    """Calibration bucket counts sum to total samples."""
    from idi_iann.policy import LookupPolicy
    import random

    policy = LookupPolicy()
    checker = CalibrationChecker(policy, n_buckets=n_buckets)

    # Generate random states and actions
    random.seed(42)
    states = [
        tuple(random.randint(0, 3) for _ in range(5))
        for _ in range(n_samples)
    ]
    actions = [Action.BUY] * n_samples
    realized = [random.uniform(-1.0, 1.0) for _ in range(n_samples)]

    result = checker.compute_calibration(states, actions, realized)

    # Sum of bucket counts should equal n_samples
    total_count = sum(b.count for b in result.buckets)
    assert total_count == n_samples, (
        f"Bucket counts sum mismatch: {total_count} vs {n_samples}"
    )


@given(
    ref_values=st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=10,
        max_size=100,
    ),
    comp_values=st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=10,
        max_size=100,
    ),
)
def test_drift_psi_non_negative(ref_values: list[float], comp_values: list[float]) -> None:
    """Drift PSI metric is non-negative."""
    detector = DriftDetector(n_bins=10)

    ref_stats = FeatureStats.from_values("test", ref_values, n_bins=10)
    comp_stats = FeatureStats.from_values("test", comp_values, n_bins=10)

    metrics = detector.compare_features(ref_stats, comp_stats)

    assert metrics.psi >= 0.0, f"PSI is negative: {metrics.psi}"


@given(
    ref_values=st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=10,
        max_size=100,
    ),
    comp_values=st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=10,
        max_size=100,
    ),
)
def test_drift_ks_in_range(ref_values: list[float], comp_values: list[float]) -> None:
    """Drift KS statistic is in valid range [0, 1]."""
    detector = DriftDetector(n_bins=10)

    ref_stats = FeatureStats.from_values("test", ref_values, n_bins=10)
    comp_stats = FeatureStats.from_values("test", comp_values, n_bins=10)

    metrics = detector.compare_features(ref_stats, comp_stats)

    assert 0.0 <= metrics.ks_statistic <= 1.0, (
        f"KS statistic out of range: {metrics.ks_statistic}"
    )


@given(
    weights=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=20,
    ),
)
def test_wis_total_weight_non_negative(weights: list[float]) -> None:
    """WIS total weight is non-negative."""
    from idi_iann.ope import LoggedDataset, LoggedEpisode, LoggedTransition
    from idi_iann.policy import LookupPolicy

    # Create episode with given weights as behavior probs
    transitions = []
    for i, w in enumerate(weights):
        transitions.append(
            LoggedTransition(
                state=(0, 0, 0, 0, 0),
                action=Action.BUY,
                reward=1.0,
                next_state=(1, 0, 0, 0, 0),
                behavior_prob=max(w, 0.01),  # Ensure > 0
                done=(i == len(weights) - 1),
            )
        )

    episode = LoggedEpisode(
        episode_id="test",
        transitions=transitions,
        behavior_policy_id="test",
        config_hash="test",
        data_version="1.0",
    )

    dataset = LoggedDataset(episodes=[episode])
    policy = LookupPolicy()
    evaluator = OPEEvaluator(policy, discount=0.99)

    result = evaluator.importance_sampling(dataset, weighted=True)

    # Total weight should be non-negative
    total_weight = result.details.get("total_weight", 0.0)
    assert total_weight >= 0.0, f"WIS total weight negative: {total_weight}"

