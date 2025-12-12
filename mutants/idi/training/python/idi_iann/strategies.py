"""Action selection strategies used by ``QTrainer``."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from .policy import LookupPolicy, StateKey


class ActionStrategy(Protocol):
    """Strategy interface for selecting the next action."""

    def select(
        self,
        *,
        state: StateKey,
        policy: LookupPolicy,
        actions: Sequence[str],
        exploration_rate: float,
        prng: random.Random,
    ) -> str:
        """Return the action to execute."""


@dataclass
class EpsilonGreedyStrategy(ActionStrategy):
    """Simple epsilon-greedy selector."""

    minimum_exploration: float = 0.05

    def select(
        self,
        *,
        state: StateKey,
        policy: LookupPolicy,
        actions: Sequence[str],
        exploration_rate: float,
        prng: random.Random,
    ) -> str:
        epsilon = max(self.minimum_exploration, exploration_rate)
        if prng.random() < epsilon:
            return prng.choice(list(actions))
        return policy.best_action(state)

