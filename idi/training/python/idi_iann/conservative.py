"""Conservative Q-learning variants for robust offline RL.

Implements:
- Conservative Q-Learning (CQL): Penalizes Q-values for OOD actions
- Behavior-Regularized Q-Learning: Regularizes toward behavior policy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from .domain import Action, StateKey
from .policy import LookupPolicy


@dataclass
class VisitationCounter:
    """Tracks state-action visitation counts for conservative penalties."""

    counts: Dict[Tuple[StateKey, str], int] = field(default_factory=dict)
    total_visits: int = 0

    def record(self, state: StateKey, action: Action) -> None:
        """Record a visit to state-action pair."""
        action_str = action.value if isinstance(action, Action) else action
        key = (state, action_str)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.total_visits += 1

    def get_count(self, state: StateKey, action: Action) -> int:
        """Get visit count for state-action pair."""
        action_str = action.value if isinstance(action, Action) else action
        return self.counts.get((state, action_str), 0)

    def get_frequency(self, state: StateKey, action: Action) -> float:
        """Get visit frequency (probability) for state-action pair."""
        if self.total_visits == 0:
            return 0.0
        return self.get_count(state, action) / self.total_visits


@dataclass
class BehaviorPolicy:
    """Estimated behavior policy from logged data."""

    action_probs: Dict[StateKey, Dict[str, float]] = field(default_factory=dict)

    def record_action(self, state: StateKey, action: Action) -> None:
        """Record an action taken in a state."""
        action_str = action.value if isinstance(action, Action) else action
        if state not in self.action_probs:
            self.action_probs[state] = {}
        self.action_probs[state][action_str] = self.action_probs[state].get(action_str, 0) + 1

    def normalize(self) -> None:
        """Convert counts to probabilities."""
        for state, actions in self.action_probs.items():
            total = sum(actions.values())
            if total > 0:
                for action in actions:
                    actions[action] /= total

    def get_prob(self, state: StateKey, action: Action) -> float:
        """Get probability of action under behavior policy."""
        action_str = action.value if isinstance(action, Action) else action
        if state not in self.action_probs:
            return 1.0 / 3.0  # Uniform prior over 3 actions
        return self.action_probs[state].get(action_str, 0.01)


@dataclass
class ConservativeConfig:
    """Configuration for conservative Q-learning."""

    # CQL parameters
    cql_alpha: float = 1.0  # CQL regularization strength
    use_cql: bool = True

    # Behavior regularization parameters
    behavior_alpha: float = 0.5  # Behavior regularization strength
    use_behavior_reg: bool = False

    # Visitation-based penalty
    min_visits_threshold: int = 5  # Minimum visits before trusting Q-value
    low_support_penalty: float = 0.1  # Penalty for low-support state-actions


class ConservativeQLearning:
    """Conservative Q-learning with CQL and behavior regularization."""

    def __init__(
        self,
        policy: LookupPolicy,
        config: ConservativeConfig,
        actions: Sequence[Action] = (Action.HOLD, Action.BUY, Action.SELL),
    ):
        """Initialize conservative Q-learner.

        Args:
            policy: Policy to update
            config: Conservative learning configuration
            actions: Available actions
        """
        self.policy = policy
        self.config = config
        self.actions = actions
        self.visitation = VisitationCounter()
        self.behavior = BehaviorPolicy()

    def record_behavior(self, state: StateKey, action: Action) -> None:
        """Record behavior policy action for regularization."""
        self.visitation.record(state, action)
        self.behavior.record_action(state, action)

    def finalize_behavior(self) -> None:
        """Finalize behavior policy estimation."""
        self.behavior.normalize()

    def compute_cql_penalty(self, state: StateKey, action: Action) -> float:
        """Compute CQL penalty for state-action pair.

        CQL penalizes Q-values for actions that are unlikely under the
        data distribution, encouraging conservative value estimates.
        """
        if not self.config.use_cql:
            return 0.0

        # Log-sum-exp over all actions
        q_values = [self.policy.q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        logsumexp = max_q + math.log(sum(math.exp(q - max_q) for q in q_values))

        # Q-value for the action taken in data
        q_action = self.policy.q_value(state, action)

        # CQL penalty: logsumexp(Q) - Q(s, a_data)
        penalty = self.config.cql_alpha * (logsumexp - q_action)
        return penalty

    def compute_behavior_penalty(self, state: StateKey, action: Action) -> float:
        """Compute behavior regularization penalty.

        Penalizes deviation from the estimated behavior policy.
        """
        if not self.config.use_behavior_reg:
            return 0.0

        behavior_prob = self.behavior.get_prob(state, action)
        # KL-like penalty: -log(behavior_prob)
        penalty = -self.config.behavior_alpha * math.log(behavior_prob + 1e-9)
        return penalty

    def compute_support_penalty(self, state: StateKey, action: Action) -> float:
        """Compute penalty for low-support state-action pairs."""
        visit_count = self.visitation.get_count(state, action)
        if visit_count >= self.config.min_visits_threshold:
            return 0.0

        # Linear penalty for low support
        support_ratio = visit_count / self.config.min_visits_threshold
        penalty = self.config.low_support_penalty * (1.0 - support_ratio)
        return penalty

    def compute_conservative_update(
        self,
        state: StateKey,
        action: Action,
        reward: float,
        next_state: StateKey,
        learning_rate: float,
        discount: float,
    ) -> float:
        """Compute conservative Q-update with penalties.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            learning_rate: Learning rate
            discount: Discount factor

        Returns:
            Q-value delta to apply
        """
        # Standard TD target
        best_next = self.policy.best_action(next_state)
        td_target = reward + discount * self.policy.q_value(next_state, best_next)

        # Current Q-value
        current_q = self.policy.q_value(state, action)

        # Standard TD error
        td_error = td_target - current_q

        # Conservative penalties
        cql_penalty = self.compute_cql_penalty(state, action)
        behavior_penalty = self.compute_behavior_penalty(state, action)
        support_penalty = self.compute_support_penalty(state, action)

        total_penalty = cql_penalty + behavior_penalty + support_penalty

        # Adjusted update
        adjusted_error = td_error - total_penalty

        return learning_rate * adjusted_error

    def update(
        self,
        state: StateKey,
        action: Action,
        reward: float,
        next_state: StateKey,
        learning_rate: float,
        discount: float,
    ) -> None:
        """Update Q-value with conservative penalties."""
        delta = self.compute_conservative_update(
            state, action, reward, next_state, learning_rate, discount
        )
        self.policy.update(state, action, delta)


def create_conservative_learner(
    policy: LookupPolicy,
    use_cql: bool = True,
    cql_alpha: float = 1.0,
    use_behavior_reg: bool = False,
    behavior_alpha: float = 0.5,
) -> ConservativeQLearning:
    """Factory function to create a conservative Q-learner.

    Args:
        policy: Policy to update
        use_cql: Whether to use CQL penalty
        cql_alpha: CQL regularization strength
        use_behavior_reg: Whether to use behavior regularization
        behavior_alpha: Behavior regularization strength

    Returns:
        Configured ConservativeQLearning instance
    """
    config = ConservativeConfig(
        cql_alpha=cql_alpha,
        use_cql=use_cql,
        behavior_alpha=behavior_alpha,
        use_behavior_reg=use_behavior_reg,
    )
    return ConservativeQLearning(policy, config)

