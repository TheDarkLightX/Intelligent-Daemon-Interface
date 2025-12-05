"""Configuration dataclasses for the IDI/IAN training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class QuantizerConfig:
    """Quantization knobs for market + emotional signals."""

    price_buckets: int = 4
    volume_buckets: int = 4
    trend_buckets: int = 4
    scarcity_buckets: int = 8
    mood_buckets: int = 4

    def validate(self) -> None:
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0 (got {value})")


@dataclass(frozen=True)
class RewardWeights:
    """Multi-objective weights for Q updates."""

    pnl: float = 1.0
    scarcity_alignment: float = 0.5
    ethics_bonus: float = 0.75
    communication_clarity: float = 0.2

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (
            self.pnl,
            self.scarcity_alignment,
            self.ethics_bonus,
            self.communication_clarity,
        )


@dataclass(frozen=True)
class EmoteConfig:
    """Maps mood buckets to textual/emoji payloads."""

    palette: Dict[int, str] = field(
        default_factory=lambda: {
            0: "🙂 steady",
            1: "🚀 optimistic",
            2: "😐 cautious",
            3: "⚠️ alert",
        }
    )
    linger_ticks: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    """High-level training knobs."""

    episodes: int = 128
    episode_length: int = 64
    discount: float = 0.92
    learning_rate: float = 0.2
    exploration_decay: float = 0.995
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    emote: EmoteConfig = field(default_factory=EmoteConfig)

    def validate(self) -> None:
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0.0 < self.exploration_decay <= 1.0):
            raise ValueError("exploration_decay must be in (0, 1]")
        self.quantizer.validate()

