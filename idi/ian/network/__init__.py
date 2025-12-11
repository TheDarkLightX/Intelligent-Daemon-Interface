"""
IAN Network Module - P2P networking, API, and decentralized L2.

Provides:
1. Node identity and key management
2. Peer discovery (seed-node based)
3. P2P message protocol (gossip + request/response)
4. REST API server
5. State synchronization
6. **Decentralized L2 components:**
   - Consensus coordinator (multi-node)
   - Contribution ordering (deterministic)
   - Fraud proofs (detection & verification)
   - Economic security (bonding & slashing)
   - Evaluation quorum (distributed)
"""

from .node import NodeIdentity, NodeInfo, NodeCapabilities
from .discovery import PeerDiscovery, SeedNodeDiscovery
from .protocol import (
    Message,
    MessageType,
    ContributionAnnounce,
    ContributionRequest,
    ContributionResponse,
    StateRequest,
    StateResponse,
    PeerExchange,
)
from .transport import Transport, TCPTransport
from .api import IANApiServer, create_api_app

# Decentralized L2 components
from .ordering import (
    OrderingKey,
    ContributionMempool,
    OrderingProof,
    ContributionGossip,
)
from .consensus import (
    ConsensusCoordinator,
    ConsensusConfig,
    ConsensusState,
    PeerStateSnapshot,
)
from .fraud import (
    FraudType,
    FraudProof,
    InvalidLogRootProof,
    InvalidLeaderboardProof,
    SkippedContributionProof,
    WrongOrderingProof,
    WrongEvaluationProof,
    FraudProofGenerator,
    FraudProofVerifier,
    ChallengeManager,
)
from .economics import (
    EconomicConfig,
    EconomicManager,
    CommitterBond,
    ChallengeBond,
    SlashEvent,
    BondStatus,
)
from .evaluation import (
    EvaluationQuorumConfig,
    EvaluationQuorumManager,
    EvaluatorInfo,
    EvaluatorStatus,
    EvaluationRequest,
    EvaluationResponse,
    QuorumResult,
)
from .decentralized_node import (
    DecentralizedNode,
    DecentralizedNodeConfig,
    create_decentralized_node,
    run_decentralized_node,
)
from .p2p_manager import (
    P2PManager,
    P2PConfig,
    PeerSession,
    PeerState,
)
from .sync import (
    StateSynchronizer,
    SyncConfig,
    SyncState,
    SyncProgress,
    ContributionStorage,
    LogSyncRequest,
    LogSyncResponse,
)

# Production utilities
try:
    from .production import (
        TaskSupervisor,
        SupervisedTask,
        TaskState,
        NodeMetrics,
        HealthServer,
        HealthCheck,
        HealthStatus,
        PeerScore,
        PeerScoreManager,
        GracefulShutdown,
        NodeState,
        BackoffStrategy,
        backoff_with_jitter,
    )
except ImportError:
    pass

# Resilience utilities
try:
    from .resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerError,
        CircuitState,
        StructuredLogger,
        StructuredFormatter,
        Bulkhead,
        AsyncRateLimiter,
        HealthAggregator,
        DependencyHealth,
        get_circuit_breaker,
        circuit_breaker,
        retry_with_circuit_breaker,
        get_correlation_id,
        set_correlation_id,
        new_correlation_id,
        setup_structured_logging,
        with_timeout,
    )
except ImportError:
    pass

# TLS support
try:
    from .tls import (
        TLSConfig,
        TLSTransport,
        CertificateAuthority,
        create_self_signed_tls_config,
        verify_peer_certificate,
        HAS_CRYPTO,
    )
except ImportError:
    HAS_CRYPTO = False

# Optional imports (may not have dependencies)
try:
    from .websocket_transport import (
        WebSocketServer,
        WebSocketClient,
        WebSocketConfig,
    )
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

__all__ = [
    # Node
    "NodeIdentity",
    "NodeInfo",
    "NodeCapabilities",
    # Discovery
    "PeerDiscovery",
    "SeedNodeDiscovery",
    # Protocol
    "Message",
    "MessageType",
    "ContributionAnnounce",
    "ContributionRequest",
    "ContributionResponse",
    "StateRequest",
    "StateResponse",
    "PeerExchange",
    # Transport
    "Transport",
    "TCPTransport",
    # API
    "IANApiServer",
    "create_api_app",
    # Ordering
    "OrderingKey",
    "ContributionMempool",
    "OrderingProof",
    "ContributionGossip",
    # Consensus
    "ConsensusCoordinator",
    "ConsensusConfig",
    "ConsensusState",
    "PeerStateSnapshot",
    # Fraud
    "FraudType",
    "FraudProof",
    "InvalidLogRootProof",
    "InvalidLeaderboardProof",
    "SkippedContributionProof",
    "WrongOrderingProof",
    "WrongEvaluationProof",
    "FraudProofGenerator",
    "FraudProofVerifier",
    "ChallengeManager",
    # Economics
    "EconomicConfig",
    "EconomicManager",
    "CommitterBond",
    "ChallengeBond",
    "SlashEvent",
    "BondStatus",
    # Evaluation Quorum
    "EvaluationQuorumConfig",
    "EvaluationQuorumManager",
    "EvaluatorInfo",
    "EvaluatorStatus",
    "EvaluationRequest",
    "EvaluationResponse",
    "QuorumResult",
    # Decentralized Node
    "DecentralizedNode",
    "DecentralizedNodeConfig",
    "create_decentralized_node",
    "run_decentralized_node",
    # P2P Manager
    "P2PManager",
    "P2PConfig",
    "PeerSession",
    "PeerState",
    # State Sync
    "StateSynchronizer",
    "SyncConfig",
    "SyncState",
    "SyncProgress",
    "ContributionStorage",
    # Production utilities
    "TaskSupervisor",
    "NodeMetrics",
    "HealthServer",
    "HealthCheck",
    "HealthStatus",
    "PeerScore",
    "PeerScoreManager",
    "GracefulShutdown",
    "NodeState",
    "backoff_with_jitter",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerError",
    "StructuredLogger",
    "Bulkhead",
    "AsyncRateLimiter",
    "HealthAggregator",
    "get_circuit_breaker",
    "circuit_breaker",
    "retry_with_circuit_breaker",
    "get_correlation_id",
    "setup_structured_logging",
    # TLS
    "TLSConfig",
    "TLSTransport",
    "CertificateAuthority",
    "create_self_signed_tls_config",
    "HAS_CRYPTO",
    # WebSocket (optional)
    "HAS_WEBSOCKET",
]

# Conditionally export WebSocket classes
if HAS_WEBSOCKET:
    __all__.extend([
        "WebSocketServer",
        "WebSocketClient",
        "WebSocketConfig",
    ])
