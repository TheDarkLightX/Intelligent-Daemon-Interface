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

from .config import (
    QuantizerConfig,
    RewardWeights,
    TrainingConfig,
    EmoteConfig,
    LayerConfig,
    TileCoderConfig,
    CommunicationConfig,
)
from .abstraction import TileCoder
from .env import SyntheticMarketEnv, EmoteChannel
from .crypto_env import CryptoMarket, MarketParams, MarketState
from .policy import LookupPolicy, StateKey
from .strategies import ActionStrategy, EpsilonGreedyStrategy
from .trainer import QTrainer, TraceBatch
from .ope import OPEEvaluator, LoggedDataset, LoggedEpisode, LoggedTransition, OPEResult, run_ope
from .calibration import CalibrationChecker, CalibrationResult, check_calibration
from .drift import DriftDetector, DriftMetrics, FeatureStats, ShiftReport, compute_drift_report, extract_state_features
from .conservative import ConservativeQLearning, ConservativeConfig, BehaviorPolicy, VisitationCounter, create_conservative_learner
from .metrics import MetricsRecorder, MetricsBackend, InMemoryBackend, JSONFileBackend, get_global_recorder
from .emote import EmotionEngine, EmotionState
from .communication import CommunicationPolicy
from .rewards import CommunicationRewardShaper
from .factories import create_environment, create_policy, create_trainer
from .domain import Action, Regime, Transition, Observation

__all__ = [
    # OPE
    "OPEEvaluator",
    "LoggedDataset",
    "LoggedEpisode",
    "LoggedTransition",
    "OPEResult",
    "run_ope",
    # Calibration
    "CalibrationChecker",
    "CalibrationResult",
    "check_calibration",
    # Drift detection
    "DriftDetector",
    "DriftMetrics",
    "FeatureStats",
    "ShiftReport",
    "compute_drift_report",
    "extract_state_features",
    # Conservative Q-learning
    "ConservativeQLearning",
    "ConservativeConfig",
    "BehaviorPolicy",
    "VisitationCounter",
    "create_conservative_learner",
    # Metrics
    "MetricsRecorder",
    "MetricsBackend",
    "InMemoryBackend",
    "JSONFileBackend",
    "get_global_recorder",
    # Existing exports
    "QuantizerConfig",
    "RewardWeights",
    "TrainingConfig",
    "LayerConfig",
    "TileCoderConfig",
    "CommunicationConfig",
    "EmoteConfig",
    "SyntheticMarketEnv",
    "CryptoMarket",
    "MarketParams",
    "MarketState",
    "EmoteChannel",
    "LookupPolicy",
    "StateKey",
    "ActionStrategy",
    "EpsilonGreedyStrategy",
    "QTrainer",
    "TraceBatch",
    "EmotionEngine",
    "EmotionState",
    "TileCoder",
    "CommunicationPolicy",
    "CommunicationRewardShaper",
    "create_environment",
    "create_policy",
    "create_trainer",
    "Action",
    "Regime",
    "Transition",
    "Observation",
]

