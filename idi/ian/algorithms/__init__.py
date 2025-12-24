"""
Novel Algorithms for IAN (Intelligent Augmentation Network).

This module contains A-grade algorithms designed through systematic research
and iterative Codex collaboration (December 2025).

Algorithms:
- DIV: Deterministic Invariant Verification
- VEP: Verifiable Evaluation Protocol  
- TVRF: Threshold VRF for Evaluator Selection
- IFP: Interactive Fraud Proofs
- DAG: DAG-based Consensus (future)
- Nova: Incremental State Proofs (future)
"""

from .div import (
    DIVInvariantChecker,
    Invariant,
    InvariantFormula,
    InvariantAtom,
    AtomOperator,
    FormulaOperator,
    InvariantCertificate,
)

from .vep import (
    VEPEvaluationHarness,
    EvaluationTrace,
    TraceCommitment,
    AuditRequest,
    AuditResult,
)

from .tvrf import (
    TVRFCoordinator,
    EvaluatorInfo,
    EvaluatorSet,
    VRFSeed,
    VRFOutput,
    select_evaluators_from_vrf,
    create_evaluator_set,
)

from .ifp import (
    IFPDisputeManager,
    IFPState,
    Dispute,
    DisputeStatus,
    DisputeAssertion,
    DisputeChallenge,
    BisectionRound,
    FinalStepProof,
    create_state_trace,
)

from .crypto import (
    SparseMerkleTree,
    SMTProof,
    SMTError,
    BLSOperations,
    BLSError,
    BLS_PUBKEY_LEN,
    BLS_SIGNATURE_LEN,
    BLS_PRIVKEY_LEN,
    MAX_MESSAGE_LEN,
    MAX_SMT_VALUE_LEN,
    MAX_SMT_ENTRIES,
    MAX_POP_REGISTRY_SIZE,
)

__all__ = [
    # DIV
    "DIVInvariantChecker",
    "Invariant",
    "InvariantFormula",
    "InvariantAtom",
    "AtomOperator",
    "FormulaOperator",
    "InvariantCertificate",
    # VEP
    "VEPEvaluationHarness",
    "EvaluationTrace",
    "TraceCommitment",
    "AuditRequest",
    "AuditResult",
    # TVRF
    "TVRFCoordinator",
    "EvaluatorInfo",
    "EvaluatorSet",
    "VRFSeed",
    "VRFOutput",
    "select_evaluators_from_vrf",
    "create_evaluator_set",
    # IFP
    "IFPDisputeManager",
    "IFPState",
    "Dispute",
    "DisputeStatus",
    "DisputeAssertion",
    "DisputeChallenge",
    "BisectionRound",
    "FinalStepProof",
    "create_state_trace",
    # Crypto
    "SparseMerkleTree",
    "SMTProof",
    "SMTError",
    "BLSOperations",
    "BLSError",
    "BLS_PUBKEY_LEN",
    "BLS_SIGNATURE_LEN",
    "BLS_PRIVKEY_LEN",
    "MAX_MESSAGE_LEN",
    "MAX_SMT_VALUE_LEN",
    "MAX_SMT_ENTRIES",
    "MAX_POP_REGISTRY_SIZE",
]
