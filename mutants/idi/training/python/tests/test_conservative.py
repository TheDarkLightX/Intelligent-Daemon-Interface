"""Tests for conservative Q-learning module."""

import pytest

from idi_iann.conservative import (
    BehaviorPolicy,
    ConservativeConfig,
    ConservativeQLearning,
    VisitationCounter,
    create_conservative_learner,
)
from idi_iann.domain import Action
from idi_iann.policy import LookupPolicy


def test_visitation_counter():
    """Test visitation counting."""
    counter = VisitationCounter()

    state = (0, 0, 0, 0, 0)
    counter.record(state, Action.BUY)
    counter.record(state, Action.BUY)
    counter.record(state, Action.SELL)

    assert counter.get_count(state, Action.BUY) == 2
    assert counter.get_count(state, Action.SELL) == 1
    assert counter.get_count(state, Action.HOLD) == 0
    assert counter.total_visits == 3


def test_visitation_frequency():
    """Test visitation frequency computation."""
    counter = VisitationCounter()
    state = (0, 0, 0, 0, 0)

    counter.record(state, Action.BUY)
    counter.record(state, Action.BUY)
    counter.record(state, Action.SELL)
    counter.record(state, Action.HOLD)

    assert counter.get_frequency(state, Action.BUY) == 0.5
    assert counter.get_frequency(state, Action.SELL) == 0.25
    assert counter.get_frequency(state, Action.HOLD) == 0.25


def test_behavior_policy():
    """Test behavior policy estimation."""
    behavior = BehaviorPolicy()
    state = (0, 0, 0, 0, 0)

    behavior.record_action(state, Action.BUY)
    behavior.record_action(state, Action.BUY)
    behavior.record_action(state, Action.SELL)

    behavior.normalize()

    assert behavior.get_prob(state, Action.BUY) == pytest.approx(2 / 3)
    assert behavior.get_prob(state, Action.SELL) == pytest.approx(1 / 3)


def test_behavior_policy_unknown_state():
    """Test behavior policy returns uniform for unknown states."""
    behavior = BehaviorPolicy()

    unknown_state = (99, 99, 99, 99, 99)
    prob = behavior.get_prob(unknown_state, Action.BUY)

    assert prob == pytest.approx(1 / 3)


def test_cql_penalty():
    """Test CQL penalty computation."""
    policy = LookupPolicy()
    config = ConservativeConfig(use_cql=True, cql_alpha=1.0)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)
    policy.update(state, Action.BUY, 1.0)
    policy.update(state, Action.SELL, 0.5)
    policy.update(state, Action.HOLD, 0.2)

    penalty = learner.compute_cql_penalty(state, Action.BUY)

    # Penalty should be positive (logsumexp > Q(s,a) for non-max actions)
    assert penalty >= 0.0


def test_support_penalty():
    """Test low-support penalty."""
    policy = LookupPolicy()
    config = ConservativeConfig(min_visits_threshold=5, low_support_penalty=0.1)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)

    # No visits - should have penalty
    penalty_no_visits = learner.compute_support_penalty(state, Action.BUY)
    assert penalty_no_visits == pytest.approx(0.1)

    # Record some visits
    for _ in range(3):
        learner.visitation.record(state, Action.BUY)

    penalty_some_visits = learner.compute_support_penalty(state, Action.BUY)
    assert penalty_some_visits < penalty_no_visits

    # Full support
    for _ in range(2):
        learner.visitation.record(state, Action.BUY)

    penalty_full_support = learner.compute_support_penalty(state, Action.BUY)
    assert penalty_full_support == 0.0


def test_conservative_update():
    """Test conservative Q-update."""
    policy = LookupPolicy()
    config = ConservativeConfig(use_cql=True, cql_alpha=0.5)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0)

    # Record some behavior
    learner.record_behavior(state, Action.BUY)

    # Compute update
    delta = learner.compute_conservative_update(
        state, Action.BUY, reward=1.0, next_state=next_state,
        learning_rate=0.1, discount=0.99
    )

    # Delta should be finite
    assert abs(delta) < 100.0


def test_conservative_update_applies():
    """Test that conservative update modifies policy."""
    policy = LookupPolicy()
    config = ConservativeConfig(use_cql=True, cql_alpha=0.5)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0)

    initial_q = policy.q_value(state, Action.BUY)

    learner.update(
        state, Action.BUY, reward=1.0, next_state=next_state,
        learning_rate=0.1, discount=0.99
    )

    updated_q = policy.q_value(state, Action.BUY)
    assert updated_q != initial_q


def test_factory_function():
    """Test create_conservative_learner factory."""
    policy = LookupPolicy()

    learner = create_conservative_learner(
        policy,
        use_cql=True,
        cql_alpha=2.0,
        use_behavior_reg=True,
        behavior_alpha=0.3,
    )

    assert learner.config.use_cql
    assert learner.config.cql_alpha == 2.0
    assert learner.config.use_behavior_reg
    assert learner.config.behavior_alpha == 0.3


def test_behavior_regularization_penalty():
    """Test behavior regularization penalty."""
    policy = LookupPolicy()
    config = ConservativeConfig(use_behavior_reg=True, behavior_alpha=0.5, use_cql=False)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)

    # Record behavior heavily favoring BUY
    for _ in range(10):
        learner.record_behavior(state, Action.BUY)
    learner.record_behavior(state, Action.SELL)
    learner.finalize_behavior()

    # Penalty for BUY should be lower than SELL
    penalty_buy = learner.compute_behavior_penalty(state, Action.BUY)
    penalty_sell = learner.compute_behavior_penalty(state, Action.SELL)

    assert penalty_buy < penalty_sell


def test_disabled_penalties():
    """Test that disabled penalties return zero."""
    policy = LookupPolicy()
    config = ConservativeConfig(use_cql=False, use_behavior_reg=False)
    learner = ConservativeQLearning(policy, config)

    state = (0, 0, 0, 0, 0)

    assert learner.compute_cql_penalty(state, Action.BUY) == 0.0
    assert learner.compute_behavior_penalty(state, Action.BUY) == 0.0

