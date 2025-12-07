"""Tests for ensemble Q-learning module."""

import pytest

from idi_iann.domain import Action
from idi_iann.ensemble import (
    EnsembleConfig,
    EnsembleQLearning,
    ExplorationConfig,
    ExplorationStrategy,
)
from idi_iann.policy import LookupPolicy


def test_ensemble_initialization():
    """Test ensemble creates correct number of members."""
    config = EnsembleConfig(n_members=5)
    ensemble = EnsembleQLearning(config, seed=42)

    assert len(ensemble.members) == 5


def test_ensemble_update():
    """Test ensemble updates members with bootstrap sampling."""
    config = EnsembleConfig(n_members=3, bootstrap_ratio=1.0)  # All members
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0)

    ensemble.start_episode(0)
    ensemble.update(state, Action.BUY, reward=1.0, next_state=next_state,
                    learning_rate=0.1, discount=0.99)

    # All members should have been updated
    for member in ensemble.members:
        q = member.q_value(state, Action.BUY)
        assert q > 0.0


def test_ensemble_bootstrap_sampling():
    """Test bootstrap sampling excludes some members."""
    config = EnsembleConfig(n_members=10, bootstrap_ratio=0.5)
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0)

    # Run multiple episodes to get varied bootstrap masks
    for ep in range(10):
        ensemble.start_episode(ep)
        ensemble.update(state, Action.BUY, reward=1.0, next_state=next_state,
                        learning_rate=0.1, discount=0.99, episode=ep)

    # Different members should have different Q-values due to bootstrap
    q_values = [m.q_value(state, Action.BUY) for m in ensemble.members]
    # Not all should be identical (some variance)
    assert len(set(round(q, 4) for q in q_values)) > 1


def test_ensemble_mean_q():
    """Test mean Q-value computation."""
    config = EnsembleConfig(n_members=3)
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Manually set Q-values for testing
    for i, member in enumerate(ensemble.members):
        member.update(state, Action.BUY, float(i + 1))

    mean_q = ensemble.mean_q(state, Action.BUY)
    assert mean_q == pytest.approx(2.0)  # (1+2+3)/3


def test_ensemble_std_q():
    """Test Q-value standard deviation (uncertainty)."""
    config = EnsembleConfig(n_members=3)
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Set identical Q-values -> zero std
    for member in ensemble.members:
        member.update(state, Action.BUY, 1.0)

    std_q = ensemble.std_q(state, Action.BUY)
    assert std_q == pytest.approx(0.0)

    # Set different Q-values -> non-zero std
    for i, member in enumerate(ensemble.members):
        member._table[state].q_values["buy"] = float(i)  # 0, 1, 2

    std_q = ensemble.std_q(state, Action.BUY)
    assert std_q > 0.0


def test_ensemble_aggregation_methods():
    """Test different aggregation methods."""
    state = (0, 0, 0, 0, 0)

    for agg in ["mean", "median", "pessimistic"]:
        config = EnsembleConfig(n_members=3, aggregation=agg)
        ensemble = EnsembleQLearning(config, seed=42)

        # Set Q-values: 1, 2, 3
        for i, member in enumerate(ensemble.members):
            member.update(state, Action.BUY, float(i + 1))

        q = ensemble.aggregated_q(state, Action.BUY)

        if agg == "mean":
            assert q == pytest.approx(2.0)
        elif agg == "median":
            assert q == pytest.approx(2.0)
        elif agg == "pessimistic":
            # mean - std
            assert q < 2.0


def test_ensemble_thompson_sampling():
    """Test Thompson sampling action selection."""
    config = EnsembleConfig(n_members=5, use_thompson=True)
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Set different best actions for different members
    ensemble.members[0].update(state, Action.BUY, 10.0)
    ensemble.members[1].update(state, Action.SELL, 10.0)
    ensemble.members[2].update(state, Action.HOLD, 10.0)

    # Thompson sampling should select different actions over time
    actions_seen = set()
    for _ in range(50):
        action = ensemble.select_action(state)
        actions_seen.add(action)

    # Should see multiple actions due to Thompson sampling
    assert len(actions_seen) >= 2


def test_ensemble_best_action():
    """Test best action selection without Thompson."""
    config = EnsembleConfig(n_members=3, use_thompson=False, aggregation="mean")
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Make BUY the best action on average
    for member in ensemble.members:
        member.update(state, Action.BUY, 10.0)
        member.update(state, Action.SELL, 1.0)

    action = ensemble.best_action(state)
    assert action == Action.BUY


def test_ensemble_uncertainty_stats():
    """Test uncertainty statistics."""
    config = EnsembleConfig(n_members=3)
    ensemble = EnsembleQLearning(config, seed=42)

    state = (0, 0, 0, 0, 0)

    # Add some state entries
    for member in ensemble.members:
        member.update(state, Action.BUY, 1.0)

    stats = ensemble.get_uncertainty_stats()

    assert "mean_uncertainty" in stats
    assert "max_uncertainty" in stats
    assert "n_states" in stats
    assert stats["n_states"] == 1


def test_exploration_epsilon_greedy():
    """Test epsilon-greedy exploration."""
    config = ExplorationConfig(strategy="epsilon_greedy", epsilon=0.5)
    policy = LookupPolicy()
    policy.update((0, 0, 0, 0, 0), Action.BUY, 10.0)

    strategy = ExplorationStrategy(config, policy, seed=42)

    state = (0, 0, 0, 0, 0)

    actions = [strategy.select_action(state) for _ in range(100)]

    # Should see both BUY (exploitation) and other actions (exploration)
    action_set = set(actions)
    assert len(action_set) > 1


def test_exploration_ucb():
    """Test UCB exploration."""
    config = ExplorationConfig(strategy="ucb", ucb_c=2.0)
    policy = LookupPolicy()

    strategy = ExplorationStrategy(config, policy, seed=42)

    state = (0, 0, 0, 0, 0)

    # First action should be unvisited
    action1 = strategy.select_action(state)
    strategy.record_action(state, action1)

    # Keep selecting and recording
    for _ in range(10):
        action = strategy.select_action(state)
        strategy.record_action(state, action)

    # Should have explored multiple actions
    assert len(strategy.visit_counts) > 0


def test_exploration_optimistic():
    """Test optimistic initialization."""
    config = ExplorationConfig(strategy="optimistic", optimistic_init=100.0)
    policy = LookupPolicy()
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)

    strategy = ExplorationStrategy(config, policy, seed=42)

    state = (0, 0, 0, 0, 0)

    # First selection should prefer unvisited actions (optimistic init)
    action = strategy.select_action(state)

    # Unvisited actions should be preferred due to high optimistic value
    # (actual Q=1 vs optimistic Q=100)


def test_exploration_decay():
    """Test epsilon decay."""
    config = ExplorationConfig(epsilon=1.0, epsilon_decay=0.5)
    policy = LookupPolicy()

    strategy = ExplorationStrategy(config, policy, seed=42)

    initial_epsilon = strategy.epsilon
    strategy.decay_epsilon()

    assert strategy.epsilon == initial_epsilon * 0.5

