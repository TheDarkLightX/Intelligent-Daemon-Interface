"""
Intelligent Augmentation Network (IAN) - Coordination layer for IDI agents.

IAN provides:
- Append-only experiment logs (MMR-based) per goal
- Top-K leaderboards for candidate agents
- Deterministic contribution processing
- Integration with Tau Net for consensus and upgrades
"""

from .models import (
    GoalID,
    GoalSpec,
    AgentPack,
    Contribution,
    ContributionMeta,
    Metrics,
    GoalState,
    EvaluationLimits,
    Thresholds,
)
from .mmr import MerkleMountainRange
from .leaderboard import Leaderboard, ParetoFrontier
from .dedup import BloomFilter, DedupIndex, DedupService
from .coordinator import (
    IANCoordinator,
    CoordinatorConfig,
    ProcessResult,
    RejectionReason,
    InvariantChecker,
    ProofVerifier,
    EvaluationHarness,
)
from .ranking import (
    RankingFunction,
    ScalarRanking,
    ParetoRanking,
    rank_contributions,
    is_pareto_optimal,
)
from .sandbox import (
    SandboxedEvaluator,
    InProcessEvaluator,
    EvaluationHarnessAdapter,
    EvaluationResult,
)
from .idi_integration import (
    IDIInvariantChecker,
    IDIProofVerifier,
    IDIEvaluationHarness,
    create_idi_invariant_checker,
    create_idi_proof_verifier,
    create_idi_evaluation_harness,
    create_idi_coordinator,
)
from .tau_bridge import (
    IanTxType,
    IanGoalRegisterTx,
    IanLogCommitTx,
    IanUpgradeTx,
    IanTauState,
    TauBridge,
    TauBridgeConfig,
    TauIntegratedCoordinator,
    parse_ian_tx,
    create_tau_integrated_coordinator,
)
from .security import (
    SecurityLimits,
    InputValidator,
    ValidationResult,
    RateLimiter,
    TokenBucket,
    ProofOfWork,
    SybilResistance,
    SecureCoordinator,
    constant_time_compare,
)
from .config import (
    IANConfig,
    CoordinatorConfig as ConfigCoordinatorConfig,
    SecurityConfig,
    TauConfig,
    EvaluationConfig,
    LoggingConfig,
    MetricsConfig,
    StorageConfig,
    get_config,
    set_config,
)
from .observability import (
    setup_logging,
    get_logger,
    metrics,
    tracer,
    timed,
    traced,
)

__all__ = [
    # Models
    "GoalID",
    "GoalSpec",
    "AgentPack",
    "Contribution",
    "ContributionMeta",
    "Metrics",
    "GoalState",
    "EvaluationLimits",
    "Thresholds",
    # Data structures
    "MerkleMountainRange",
    "Leaderboard",
    "ParetoFrontier",
    "BloomFilter",
    "DedupIndex",
    "DedupService",
    # Coordinator
    "IANCoordinator",
    "CoordinatorConfig",
    "ProcessResult",
    "RejectionReason",
    "InvariantChecker",
    "ProofVerifier",
    "EvaluationHarness",
    # Ranking
    "RankingFunction",
    "ScalarRanking",
    "ParetoRanking",
    "rank_contributions",
    "is_pareto_optimal",
    # Sandbox
    "SandboxedEvaluator",
    "InProcessEvaluator",
    "EvaluationHarnessAdapter",
    "EvaluationResult",
    # IDI Integration
    "IDIInvariantChecker",
    "IDIProofVerifier",
    "IDIEvaluationHarness",
    "create_idi_invariant_checker",
    "create_idi_proof_verifier",
    "create_idi_evaluation_harness",
    "create_idi_coordinator",
    # Tau Bridge
    "IanTxType",
    "IanGoalRegisterTx",
    "IanLogCommitTx",
    "IanUpgradeTx",
    "IanTauState",
    "TauBridge",
    "TauBridgeConfig",
    "TauIntegratedCoordinator",
    "parse_ian_tx",
    "create_tau_integrated_coordinator",
    # Security
    "SecurityLimits",
    "InputValidator",
    "ValidationResult",
    "RateLimiter",
    "TokenBucket",
    "ProofOfWork",
    "SybilResistance",
    "SecureCoordinator",
    "constant_time_compare",
    # Config
    "IANConfig",
    "SecurityConfig",
    "TauConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "MetricsConfig",
    "StorageConfig",
    "get_config",
    "set_config",
    # Observability
    "setup_logging",
    "get_logger",
    "metrics",
    "tracer",
    "timed",
    "traced",
]

__version__ = "0.1.0"
