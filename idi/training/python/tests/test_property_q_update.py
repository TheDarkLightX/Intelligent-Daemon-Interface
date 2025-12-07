"""Property-based tests for Q-learning invariants."""

from hypothesis import given, strategies as st
import pytest

from idi_iann.domain import Action
from idi_iann.policy import LookupPolicy


@given(
    current_q=st.floats(min_value=-1000.0, max_value=1000.0),
    target_q=st.floats(min_value=-1000.0, max_value=1000.0),
    learning_rate=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
def test_q_update_convex_hull(current_q: float, target_q: float, learning_rate: float) -> None:
    """Q-update produces values in convex hull of current Q and target.

    Q_new = Q_old + alpha * (target - Q_old)
    This is a convex combination: Q_new = (1-alpha) * Q_old + alpha * target
    """
    policy = LookupPolicy()
    state = (0, 0, 0, 0, 0)

    # Set initial Q-value
    policy.update(state, Action.BUY, current_q)

    # Compute update
    td_error = target_q - current_q
    policy.update(state, Action.BUY, learning_rate * td_error)

    new_q = policy.q_value(state, Action.BUY)

    # Q_new should be between Q_old and target (convex hull)
    min_val = min(current_q, target_q)
    max_val = max(current_q, target_q)

    assert min_val - 1e-6 <= new_q <= max_val + 1e-6, (
        f"Q-update violated convex hull: {current_q} -> {new_q} -> {target_q}"
    )


@given(
    q_values=st.lists(
        st.floats(min_value=-100.0, max_value=100.0),
        min_size=3,
        max_size=3,
    ),
)
def test_greedy_action_is_argmax(q_values: list[float]) -> None:
    """Greedy action is always argmax of Q-values."""
    policy = LookupPolicy()
    state = (0, 0, 0, 0, 0)

    # Set Q-values for all actions
    actions = [Action.HOLD, Action.BUY, Action.SELL]
    for action, q_val in zip(actions, q_values):
        policy.update(state, action, q_val)

    # Get best action
    best_action = policy.best_action(state)

    # Verify it's the argmax
    best_q = policy.q_value(state, best_action)
    all_qs = [policy.q_value(state, a) for a in actions]

    assert best_q == max(all_qs), (
        f"best_action returned {best_action} with Q={best_q}, "
        f"but max Q is {max(all_qs)}"
    )


@given(
    state=st.tuples(
        st.integers(min_value=0, max_value=3),
        st.integers(min_value=0, max_value=3),
        st.integers(min_value=0, max_value=3),
        st.integers(min_value=0, max_value=7),
        st.integers(min_value=0, max_value=3),
    ),
)
def test_best_action_returns_valid_enum(state: tuple[int, ...]) -> None:
    """Policy best_action returns valid Action enum member."""
    policy = LookupPolicy()

    # Set some Q-values
    policy.update(state, Action.BUY, 1.0)
    policy.update(state, Action.SELL, 0.5)

    best = policy.best_action(state)

    # Should be a valid Action enum member
    assert isinstance(best, Action)
    assert best in (Action.HOLD, Action.BUY, Action.SELL)


@given(
    q_values=st.lists(
        st.floats(min_value=-100.0, max_value=100.0),
        min_size=1,
        max_size=10,
    ),
)
def test_ensemble_mean_std_computation(q_values: list[float]) -> None:
    """Ensemble mean/std computations are mathematically correct."""
    from idi_iann.ensemble import EnsembleQLearning, EnsembleConfig

    config = EnsembleConfig(n_members=len(q_values))
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Set Q-values for each member
    for i, q_val in enumerate(q_values):
        ensemble.members[i].update(state, Action.BUY, q_val)

    mean_q = ensemble.mean_q(state, Action.BUY)
    std_q = ensemble.std_q(state, Action.BUY)

    # Verify mean
    expected_mean = sum(q_values) / len(q_values)
    assert abs(mean_q - expected_mean) < 1e-6, (
        f"Mean mismatch: {mean_q} vs {expected_mean}"
    )

    # Verify std (if more than 1 value)
    if len(q_values) > 1:
        variance = sum((q - expected_mean) ** 2 for q in q_values) / (len(q_values) - 1)
        expected_std = variance ** 0.5
        assert abs(std_q - expected_std) < 1e-6, (
            f"Std mismatch: {std_q} vs {expected_std}"
        )
    else:
        assert std_q == 0.0


@given(
    learning_rate=st.floats(min_value=0.0, max_value=1.0),
    discount=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
    reward=st.floats(min_value=-100.0, max_value=100.0),
    current_q=st.floats(min_value=-1000.0, max_value=1000.0),
    next_q=st.floats(min_value=-1000.0, max_value=1000.0),
)
def test_td_target_bounds(
    learning_rate: float,
    discount: float,
    reward: float,
    current_q: float,
    next_q: float,
) -> None:
    """TD target is bounded by reward and discounted next-state value."""
    td_target = reward + discount * next_q

    # TD target should be finite
    assert abs(td_target) < float("inf")

    # If discount < 1, target is bounded by reward + next_q
    if discount < 1.0:
        assert td_target >= reward + discount * min(0, next_q)
        assert td_target <= reward + discount * max(0, next_q)


@given(
    q1=st.floats(min_value=-100.0, max_value=100.0),
    q2=st.floats(min_value=-100.0, max_value=100.0),
    q3=st.floats(min_value=-100.0, max_value=100.0),
)
def test_tie_breaking_consistency(q1: float, q2: float, q3: float) -> None:
    """When Q-values are equal, best_action breaks ties consistently."""
    policy = LookupPolicy()
    state = (0, 0, 0, 0, 0)

    # Set all Q-values to the same value
    policy.update(state, Action.HOLD, q1)
    policy.update(state, Action.BUY, q1)
    policy.update(state, Action.SELL, q1)

    # Should return a valid action (not crash)
    best = policy.best_action(state)
    assert best in (Action.HOLD, Action.BUY, Action.SELL)

    # Multiple calls should be consistent (deterministic)
    best2 = policy.best_action(state)
    assert best == best2


@given(
    initial_q=st.floats(min_value=-100.0, max_value=100.0),
    delta=st.floats(min_value=-100.0, max_value=100.0),
)
def test_q_update_commutative(initial_q: float, delta: float) -> None:
    """Q-update is additive: update(a) + update(b) = update(a+b)."""
    policy1 = LookupPolicy()
    policy2 = LookupPolicy()
    state = (0, 0, 0, 0, 0)

    # Policy 1: single update
    policy1.update(state, Action.BUY, initial_q)
    policy1.update(state, Action.BUY, delta)

    # Policy 2: combined update
    policy2.update(state, Action.BUY, initial_q + delta)

    q1 = policy1.q_value(state, Action.BUY)
    q2 = policy2.q_value(state, Action.BUY)

    assert abs(q1 - q2) < 1e-6, (
        f"Update not commutative: {q1} vs {q2}"
    )

