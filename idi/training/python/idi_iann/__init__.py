"""IDI/IAN training toolkit (Python edition).

The package exposes typed building blocks so offline training stays simple,
testable, and deterministic.  Each module carries a single responsibility:

- ``config``: dataclasses describing quantizers and reward weights.
- ``env``: synthetic market + communicative environment simulator.
- ``policy``: lookup-table container with serialization helpers.
- ``trainer``: high-level Q-learning orchestrator and trace export.
- ``emote``: emoji/text mood generation utilities.
- ``benchmarks``: deterministic smoke tests to guard regressions.
"""

from .config import QuantizerConfig, RewardWeights, TrainingConfig, EmoteConfig
from .env import SyntheticMarketEnv, EmoteChannel
from .policy import LookupPolicy, StateKey
from .strategies import ActionStrategy, EpsilonGreedyStrategy
from .trainer import QTrainer, TraceBatch
from .emote import EmotionEngine, EmotionState

__all__ = [
    "QuantizerConfig",
    "RewardWeights",
    "TrainingConfig",
    "EmoteConfig",
    "SyntheticMarketEnv",
    "EmoteChannel",
    "LookupPolicy",
    "StateKey",
    "ActionStrategy",
    "EpsilonGreedyStrategy",
    "QTrainer",
    "TraceBatch",
    "EmotionEngine",
    "EmotionState",
]

