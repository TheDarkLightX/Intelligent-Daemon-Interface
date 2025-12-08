"""Configuration dataclasses for the IDI/IAN training pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


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
class TileCoderConfig:
    """Tile-coding abstraction parameters for compact Q-tables."""

    num_tilings: int = 3
    tile_sizes: Tuple[int, ...] = (4, 4, 4, 4, 4)
    offsets: Tuple[int, ...] = (0, 1, 2, 3, 4)

    def validate(self) -> None:
        if len(self.tile_sizes) != len(self.offsets):
            raise ValueError("tile_sizes and offsets must have the same length")
        if self.num_tilings <= 0:
            raise ValueError("num_tilings must be > 0")


@dataclass(frozen=True)
class CommunicationConfig:
    """Controls the auxiliary Q-table for expressive communication."""

    actions: Sequence[str] = ("silent", "positive", "alert", "persist")

    def validate(self) -> None:
        if not self.actions:
            raise ValueError("communication actions cannot be empty")


@dataclass(frozen=True)
class LayerConfig:
    """Controls layer-weight stream generation for layered Tau specs."""

    emit_weight_streams: bool = True
    momentum_threshold: float = 0.6  # threshold on normalized trend bucket
    contrarian_threshold: float = 0.3
    trend_favors_even: bool = True

    def validate(self) -> None:
        if not (0.0 <= self.contrarian_threshold <= 1.0):
            raise ValueError("contrarian_threshold must be in [0, 1]")
        if not (0.0 <= self.momentum_threshold <= 1.0):
            raise ValueError("momentum_threshold must be in [0, 1]")
        if self.contrarian_threshold > self.momentum_threshold:
            raise ValueError("contrarian_threshold cannot exceed momentum_threshold")


@dataclass(frozen=True)
class FractalLevelConfig:
    """Configuration for a single fractal abstraction level."""
    features: Tuple[Tuple[str, int], ...]  # [(feature_name, num_buckets), ...]
    scale_factor: float  # Scaling factor for this level
    visit_threshold: int = 5  # Minimum visits before using this level


@dataclass(frozen=True)
class FractalConfig:
    """Configuration for fractal state abstraction."""
    levels: Tuple[FractalLevelConfig, ...]
    backoff_enabled: bool = True
    hierarchical_updates: bool = True

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("fractal config must have at least one level")
        for level in self.levels:
            if level.scale_factor <= 0:
                raise ValueError("scale_factor must be > 0")
            if level.visit_threshold < 0:
                raise ValueError("visit_threshold must be >= 0")


@dataclass(frozen=True)
class MultiLayerConfig:
    """Configuration for multi-layer Q-learning."""
    layers: Tuple[str, ...] = ("momentum", "mean_reversion", "regime_aware")
    coordination: str = "weighted_voting"  # "weighted_voting", "ensemble", "hierarchical"
    communication_enabled: bool = True
    layer_specific_rewards: Dict[str, RewardWeights] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("multi_layer config must have at least one layer")
        valid_coordination = {"weighted_voting", "ensemble", "hierarchical"}
        if self.coordination not in valid_coordination:
            raise ValueError(f"coordination must be one of {valid_coordination}")


@dataclass(frozen=True)
class EpisodicConfig:
    """Optional episodic memory settings."""

    enabled: bool = False
    capacity: int = 1024
    k: int = 8
    decay: float = 0.99
    blend_alpha: float = 0.5  # weight for episodic vs q-table

    def validate(self) -> None:
        if self.capacity <= 0:
            raise ValueError("episodic.capacity must be > 0")
        if self.k <= 0:
            raise ValueError("episodic.k must be > 0")
        if not (0.0 <= self.decay <= 1.0):
            raise ValueError("episodic.decay must be in [0,1]")
        if not (0.0 <= self.blend_alpha <= 1.0):
            raise ValueError("episodic.blend_alpha must be in [0,1]")


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
    layers: LayerConfig = field(default_factory=LayerConfig)
    tile_coder: Optional[TileCoderConfig] = None
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    fractal: Optional[FractalConfig] = None
    multi_layer: Optional[MultiLayerConfig] = None
    episodic: Optional[EpisodicConfig] = None

    def validate(self) -> None:
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0.0 < self.exploration_decay <= 1.0):
            raise ValueError("exploration_decay must be in (0, 1]")
        self.quantizer.validate()
        self.layers.validate()
        self.communication.validate()
        if self.tile_coder:
            self.tile_coder.validate()
        if self.fractal:
            self.fractal.validate()
        if self.multi_layer:
            self.multi_layer.validate()
        if self.episodic:
            self.episodic.validate()

    @classmethod
    def from_json(cls, path: Path) -> TrainingConfig:
        """Load and validate a TrainingConfig from JSON."""
        data = json.loads(path.read_text())
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_json(self, path: Path) -> None:
        """Persist the config to disk in canonical form."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), sort_keys=True, indent=2))

    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint for binding receipts/specs."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
