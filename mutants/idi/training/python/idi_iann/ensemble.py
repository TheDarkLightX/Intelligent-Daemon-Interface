"""Ensemble Q-learning for uncertainty estimation.

Maintains K Q-tables with different seeds for epistemic uncertainty.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .domain import Action, StateKey
from .policy import LookupPolicy


@dataclass
class EnsembleConfig:
    """Configuration for ensemble Q-learning."""

    n_members: int = 5
    bootstrap_ratio: float = 0.8  # Fraction of data for each member
    use_thompson: bool = True  # Thompson sampling for exploration
    aggregation: str = "mean"  # mean, median, pessimistic


class EnsembleQLearning:
    """Ensemble of Q-tables for uncertainty-aware decision making."""

    def __init__(
        self,
        config: EnsembleConfig,
        actions: Sequence[Action] = (Action.HOLD, Action.BUY, Action.SELL),
        seed: Optional[int] = None,
    ):
        """Initialize ensemble.

        Args:
            config: Ensemble configuration
            actions: Available actions
            seed: Random seed for reproducibility
        """
        self.config = config
        self.actions = actions
        self.rng = random.Random(seed)
        self.members: List[LookupPolicy] = [LookupPolicy() for _ in range(config.n_members)]
        self._bootstrap_masks: Dict[int, List[bool]] = {}  # episode -> mask per member
        self._current_episode = 0

    def start_episode(self, episode: int) -> None:
        """Start a new episode with bootstrap mask.

        Args:
            episode: Episode number
        """
        self._current_episode = episode
        # Generate bootstrap mask for this episode
        mask = [self.rng.random() < self.config.bootstrap_ratio for _ in range(self.config.n_members)]
        # Ensure at least one member is included
        if not any(mask):
            mask[self.rng.randint(0, self.config.n_members - 1)] = True
        self._bootstrap_masks[episode] = mask

    def update(
        self,
        state: StateKey,
        action: Action,
        reward: float,
        next_state: StateKey,
        learning_rate: float,
        discount: float,
        episode: Optional[int] = None,
    ) -> None:
        """Update ensemble members (with bootstrap sampling).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            learning_rate: Learning rate
            discount: Discount factor
            episode: Episode number for bootstrap mask
        """
        ep = episode if episode is not None else self._current_episode
        mask = self._bootstrap_masks.get(ep, [True] * self.config.n_members)

        for i, (member, include) in enumerate(zip(self.members, mask)):
            if not include:
                continue

            # Standard Q-update for this member
            best_next = self._member_best_action(member, next_state)
            td_target = reward + discount * member.q_value(next_state, best_next)
            td_error = td_target - member.q_value(state, action)
            member.update(state, action, learning_rate * td_error)

    def _member_best_action(self, member: LookupPolicy, state: StateKey) -> Action:
        """Get best action for a single member."""
        return member.best_action(state)

    def q_values(self, state: StateKey, action: Action) -> List[float]:
        """Get Q-values from all members for state-action pair."""
        return [m.q_value(state, action) for m in self.members]

    def mean_q(self, state: StateKey, action: Action) -> float:
        """Get mean Q-value across ensemble."""
        values = self.q_values(state, action)
        return sum(values) / len(values)

    def std_q(self, state: StateKey, action: Action) -> float:
        """Get std of Q-values across ensemble (epistemic uncertainty)."""
        values = self.q_values(state, action)
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def uncertainty(self, state: StateKey) -> float:
        """Get aggregate uncertainty for a state (max std across actions)."""
        return max(self.std_q(state, a) for a in self.actions)

    def aggregated_q(self, state: StateKey, action: Action) -> float:
        """Get aggregated Q-value using configured aggregation method."""
        values = self.q_values(state, action)
        if self.config.aggregation == "mean":
            return sum(values) / len(values)
        elif self.config.aggregation == "median":
            sorted_vals = sorted(values)
            mid = len(sorted_vals) // 2
            if len(sorted_vals) % 2 == 0:
                return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            return sorted_vals[mid]
        elif self.config.aggregation == "pessimistic":
            # Lower confidence bound: mean - std
            mean = sum(values) / len(values)
            std = self.std_q(state, action)
            return mean - std
        else:
            return sum(values) / len(values)

    def best_action(self, state: StateKey) -> Action:
        """Get best action using aggregated Q-values."""
        best_a = self.actions[0]
        best_q = float("-inf")
        for a in self.actions:
            q = self.aggregated_q(state, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def thompson_action(self, state: StateKey) -> Action:
        """Get action using Thompson sampling (sample one member)."""
        member_idx = self.rng.randint(0, self.config.n_members - 1)
        return self._member_best_action(self.members[member_idx], state)

    def select_action(self, state: StateKey) -> Action:
        """Select action using configured exploration strategy."""
        if self.config.use_thompson:
            return self.thompson_action(state)
        return self.best_action(state)

    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get summary statistics about ensemble uncertainty."""
        all_states = set()
        for member in self.members:
            all_states.update(member._table.keys())

        if not all_states:
            return {"mean_uncertainty": 0.0, "max_uncertainty": 0.0, "n_states": 0}

        uncertainties = [self.uncertainty(s) for s in all_states]
        return {
            "mean_uncertainty": sum(uncertainties) / len(uncertainties),
            "max_uncertainty": max(uncertainties),
            "n_states": len(all_states),
        }


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies."""

    strategy: str = "epsilon_greedy"  # epsilon_greedy, ucb, optimistic, thompson
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    ucb_c: float = 2.0
    optimistic_init: float = 1.0


class ExplorationStrategy:
    """Configurable exploration strategy."""

    def __init__(
        self,
        config: ExplorationConfig,
        policy: LookupPolicy,
        actions: Sequence[Action] = (Action.HOLD, Action.BUY, Action.SELL),
        seed: Optional[int] = None,
    ):
        """Initialize exploration strategy.

        Args:
            config: Exploration configuration
            policy: Policy to use for exploitation
            actions: Available actions
            seed: Random seed
        """
        self.config = config
        self.policy = policy
        self.actions = actions
        self.rng = random.Random(seed)
        self.epsilon = config.epsilon
        self.visit_counts: Dict[Tuple[StateKey, str], int] = {}
        self.total_steps = 0

    def select_action(self, state: StateKey) -> Action:
        """Select action using configured strategy."""
        self.total_steps += 1

        if self.config.strategy == "epsilon_greedy":
            return self._epsilon_greedy(state)
        elif self.config.strategy == "ucb":
            return self._ucb(state)
        elif self.config.strategy == "optimistic":
            return self._optimistic(state)
        else:
            return self._epsilon_greedy(state)

    def _epsilon_greedy(self, state: StateKey) -> Action:
        """Standard epsilon-greedy exploration."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.policy.best_action(state)

    def _ucb(self, state: StateKey) -> Action:
        """Upper Confidence Bound exploration."""
        best_a = self.actions[0]
        best_score = float("-inf")

        for a in self.actions:
            action_str = a.value if isinstance(a, Action) else a
            count = self.visit_counts.get((state, action_str), 0)

            if count == 0:
                return a  # Explore unvisited action

            q = self.policy.q_value(state, a)
            bonus = self.config.ucb_c * math.sqrt(math.log(self.total_steps + 1) / count)
            score = q + bonus

            if score > best_score:
                best_score = score
                best_a = a

        return best_a

    def _optimistic(self, state: StateKey) -> Action:
        """Optimistic initialization exploration."""
        best_a = self.actions[0]
        best_q = float("-inf")

        for a in self.actions:
            action_str = a.value if isinstance(a, Action) else a
            count = self.visit_counts.get((state, action_str), 0)

            if count == 0:
                q = self.config.optimistic_init
            else:
                q = self.policy.q_value(state, a)

            if q > best_q:
                best_q = q
                best_a = a

        return best_a

    def record_action(self, state: StateKey, action: Action) -> None:
        """Record action taken for UCB calculations."""
        action_str = action.value if isinstance(action, Action) else action
        key = (state, action_str)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy."""
        self.epsilon *= self.config.epsilon_decay

