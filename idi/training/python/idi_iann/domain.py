"""Shared domain model types for IDI/IAN."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

# StateKey is a tuple of 5 quantized features
StateKey = Tuple[int, int, int, int, int]


class Action(Enum):
    """Trading action enum."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, s: str) -> Action:
        """Parse action from string."""
        for action in cls:
            if action.value == s:
                return action
        raise ValueError(f"Invalid action: {s}")


class Regime(Enum):
    """Market regime enum."""

    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    PANIC = "panic"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, s: str) -> Regime:
        """Parse regime from string."""
        for regime in cls:
            if regime.value == s:
                return regime
        raise ValueError(f"Invalid regime: {s}")


@dataclass(frozen=True)
class Transition:
    """State-action-reward-next_state transition."""

    state: StateKey
    action: Action
    reward: float
    next_state: StateKey
    done: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "state": list(self.state),
            "action": self.action.value,
            "reward": self.reward,
            "next_state": list(self.next_state),
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transition:
        """Deserialize from dictionary."""
        return cls(
            state=tuple(d["state"]),
            action=Action.from_str(d["action"]),
            reward=d["reward"],
            next_state=tuple(d["next_state"]),
            done=d.get("done", False),
        )


@dataclass(frozen=True)
class Observation:
    """Environment observation."""

    price: int
    volume: int
    trend: int
    scarcity: int
    mood: int

    def as_state(self) -> StateKey:
        """Convert to state key tuple."""
        return (self.price, self.volume, self.trend, self.scarcity, self.mood)

