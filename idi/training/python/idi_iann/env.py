"""Synthetic environment harness for Q-table training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import random

from .config import QuantizerConfig, RewardWeights


@dataclass(frozen=True)
class EnvObservation:
    """Quantized snapshot returned by the simulator."""

    price_bucket: int
    volume_bucket: int
    trend_bucket: int
    scarcity_bucket: int
    mood_bucket: int

    def as_state(self) -> Tuple[int, int, int, int, int]:
        return (
            self.price_bucket,
            self.volume_bucket,
            self.trend_bucket,
            self.scarcity_bucket,
            self.mood_bucket,
        )


class EmoteChannel:
    """Tracks high-level mood for emoji/text generation."""

    def __init__(self, max_bucket: int):
        self._max = max_bucket
        self._value = 0

    def update(self, reward_signal: float) -> int:
        if reward_signal > 0.5:
            self._value = min(self._max - 1, self._value + 1)
        elif reward_signal < -0.2:
            self._value = max(0, self._value - 1)
        return self._value


@dataclass
class SyntheticMarketEnv:
    """Very small stochastic market emulator tailored for lookup tables."""

    ACTIONS = ("hold", "buy", "sell")

    quantizer: QuantizerConfig
    rewards: RewardWeights
    seed: int = 0
    _random: random.Random = field(init=False, repr=False)
    _emote: EmoteChannel = field(init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _position: int = field(default=0, init=False, repr=False)  # -1, 0, 1

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)
        self._emote = EmoteChannel(self.quantizer.mood_buckets)

    def reset(self) -> EnvObservation:
        self._tick = 0
        self._position = 0
        self._emote = EmoteChannel(self.quantizer.mood_buckets)
        return self._observe()

    def step(self, action: str) -> Tuple[EnvObservation, float]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}")
        self._apply_action(action)
        reward = self._compute_reward()
        self._tick += 1
        mood_bucket = self._emote.update(reward)
        obs = self._observe(overrides={"mood_bucket": mood_bucket})
        reward += self.rewards.communication_clarity * mood_bucket
        return obs, reward

    def _observe(self, overrides: dict | None = None) -> EnvObservation:
        price = self._random.randint(0, self.quantizer.price_buckets - 1)
        volume = self._random.randint(0, self.quantizer.volume_buckets - 1)
        trend = self._random.randint(0, self.quantizer.trend_buckets - 1)
        scarcity = self._random.randint(0, self.quantizer.scarcity_buckets - 1)
        mood = 0
        if overrides:
            mood = overrides.get("mood_bucket", mood)
        return EnvObservation(price, volume, trend, scarcity, mood)

    def _apply_action(self, action: str) -> None:
        delta = {"buy": 1, "sell": -1}.get(action, 0)
        self._position = max(-1, min(1, self._position + delta))

    def _compute_reward(self) -> float:
        price_drift = self._random.uniform(-1.0, 1.0)
        scarcity_change = self._random.uniform(-0.5, 0.5)
        pnl_reward = self._position * price_drift * self.rewards.pnl
        scarcity_bonus = scarcity_change * self.rewards.scarcity_alignment
        ethics_bonus = self.rewards.ethics_bonus if self._position >= 0 else -self.rewards.ethics_bonus
        return pnl_reward + scarcity_bonus + ethics_bonus

