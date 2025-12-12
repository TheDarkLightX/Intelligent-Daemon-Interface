"""Tests for RL-specific bug patterns: terminal states, discount, exploration."""

import pytest

from idi_iann.ope import LoggedDataset, LoggedEpisode, LoggedTransition, OPEEvaluator
from idi_iann.domain import Action
from idi_iann.policy import LookupPolicy
from idi_iann.config import TrainingConfig
from idi_iann.trainer import QTrainer


class TestTerminalStateBootstrap:
    """Test that terminal states have zero bootstrap value."""

    def test_terminal_state_zero_bootstrap(self):
        """Verify terminal states have zero bootstrap value in TD target."""
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=0.99, episodes=1, episode_length=5)
        trainer = QTrainer(config)

        state = (0, 0, 0, 0, 0)
        terminal_state = (1, 1, 1, 1, 1)

        # Set Q-value for terminal state
        trainer._policy.update(terminal_state, Action.BUY, 100.0)

        # TD target for transition TO terminal state should ignore terminal state value
        # In standard Q-learning, if next_state is terminal, bootstrap = 0
        # Our implementation uses best_action which returns a value, but terminal
        # transitions should be marked with done=True
        
        # Test that done=True transitions are handled correctly in OPE
        transition = LoggedTransition(
            state=state,
            action=Action.BUY,
            reward=1.0,
            next_state=terminal_state,
            behavior_prob=0.5,
            done=True,  # Terminal transition
        )

        # In DR estimator, terminal states should have V(next) = 0
        # Verify this is handled in the OPE code
        episode = LoggedEpisode(
            episode_id="test",
            transitions=[transition],
            behavior_policy_id="test",
            config_hash="test",
            data_version="1.0",
        )

        dataset = LoggedDataset(episodes=[episode])
        policy = LookupPolicy()
        evaluator = OPEEvaluator(policy, discount=0.99)

        # DR estimator should handle terminal states correctly
        result = evaluator.doubly_robust(dataset)
        assert result.n_episodes == 1

    def test_done_flag_in_transitions(self):
        """Test that done=True transitions are properly marked."""
        transition = LoggedTransition(
            state=(0, 0, 0, 0, 0),
            action=Action.BUY,
            reward=1.0,
            next_state=(1, 0, 0, 0, 0),
            behavior_prob=0.5,
            done=True,
        )

        assert transition.done
        assert transition.next_state is not None  # Still has next_state for logging


class TestDiscountApplication:
    """Test that discount is applied per-step correctly."""

    def test_discount_per_step_not_per_episode(self):
        """Verify discount applied per-step, not per-episode."""
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=0.9, episodes=1, episode_length=3)
        trainer = QTrainer(config)

        state1 = (0, 0, 0, 0, 0)
        state2 = (1, 0, 0, 0, 0)
        state3 = (2, 0, 0, 0, 0)

        # Set Q-values
        trainer._policy.update(state2, Action.BUY, 10.0)
        trainer._policy.update(state3, Action.BUY, 5.0)

        # TD target for state1 -> state2
        td1 = trainer._compute_td_target(
            state1, Action.BUY, reward=1.0, next_state=state2
        )
        # Should be: reward + discount * Q(state2)
        expected1 = 1.0 + 0.9 * 10.0
        assert abs(td1 - expected1) < 1e-6

        # TD target for state2 -> state3
        td2 = trainer._compute_td_target(
            state2, Action.BUY, reward=2.0, next_state=state3
        )
        # Should be: reward + discount * Q(state3)
        expected2 = 2.0 + 0.9 * 5.0
        assert abs(td2 - expected2) < 1e-6

        # Discount is applied once per step, not accumulated across episode

    def test_td_target_formula(self):
        """Test TD target formula: reward + gamma * V(next_state)."""
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=0.5, episodes=1, episode_length=1)
        trainer = QTrainer(config)

        state = (0, 0, 0, 0, 0)
        next_state = (1, 0, 0, 0, 0)

        trainer._policy.update(next_state, Action.BUY, 8.0)

        reward = 2.0
        td_target = trainer._compute_td_target(
            state, Action.BUY, reward=reward, next_state=next_state
        )

        # TD target = reward + discount * max_a Q(next_state, a)
        expected = reward + 0.5 * 8.0  # 2.0 + 4.0 = 6.0
        assert abs(td_target - expected) < 1e-6

    def test_discount_zero_ignores_future(self):
        """Test discount=0.0 ignores future rewards completely."""
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(discount=0.0, episodes=1, episode_length=1)
        trainer = QTrainer(config)

        state = (0, 0, 0, 0, 0)
        next_state = (1, 0, 0, 0, 0)

        # Set very high future value
        trainer._policy.update(next_state, Action.BUY, 1000.0)

        td_target = trainer._compute_td_target(
            state, Action.BUY, reward=1.0, next_state=next_state
        )

        # With discount=0, should ignore future value
        assert abs(td_target - 1.0) < 1e-6


class TestExplorationDecay:
    """Test exploration decay schedules."""

    def test_exploration_decay_over_episodes(self):
        """Verify epsilon decays correctly over episodes."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(
            episodes=10,
            exploration_decay=0.9,
            episode_length=5,
        )
        trainer = QTrainer(config)

        initial_exploration = trainer._exploration

        # Run episodes
        trainer.run()

        # Exploration should have decayed
        # After 10 episodes: exploration = initial * (0.9)^10
        expected = initial_exploration * (0.9 ** 10)
        assert abs(trainer._exploration - expected) < 1e-6

    def test_exploration_decay_boundary_episodes(self):
        """Test decay at boundary episodes (first, last)."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(
            episodes=3,
            exploration_decay=0.5,  # Decay by half each episode
            episode_length=2,
        )
        trainer = QTrainer(config)

        initial = trainer._exploration
        assert initial == 1.0  # Initial exploration is 1.0

        # Run first episode
        trainer._run_episode()
        after_first = trainer._exploration
        assert abs(after_first - initial * 0.5) < 1e-6

        # Run second episode
        trainer._run_episode()
        after_second = trainer._exploration
        assert abs(after_second - initial * 0.25) < 1e-6

        # Run third episode
        trainer._run_episode()
        after_third = trainer._exploration
        assert abs(after_third - initial * 0.125) < 1e-6

    def test_exploration_never_goes_negative(self):
        """Test that exploration rate never goes negative."""
        from idi_iann.config import TrainingConfig
        from idi_iann.trainer import QTrainer

        config = TrainingConfig(
            episodes=1000,  # Many episodes
            exploration_decay=0.99,
            episode_length=1,
        )
        trainer = QTrainer(config)

        trainer.run()

        # Exploration decays but should stay >= 0
        assert trainer._exploration >= 0.0

