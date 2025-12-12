"""Tau Testnet ZK Proof Integration Bridge.

This module provides a clean integration layer between IDI ZK proof infrastructure
and the Tau Testnet blockchain, following SOLID principles and TDD methodology.
"""

from idi.taunet_bridge.protocols import (
    LocalZkProofBundle,
    NetworkZkProofBundle,
    ZkProofBundle,
    ZkWitness,
    ZkValidationResult,
    ZkVerifier,
    ZkProver,
    InvalidZkProofError,
    deserialize_proof_bundle,
)
from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.validation import ZkValidationStep, ValidationContext
from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.block_extension import BlockZkExtension, compute_zk_merkle_root
from idi.taunet_bridge.gossip import ZkGossipProtocol, TAU_PROTOCOL_ZK_PROOFS
from idi.taunet_bridge.state_integration import (
    apply_verified_transition,
    get_zk_verifier,
    set_zk_verifier,
)

__all__ = [
    "LocalZkProofBundle",
    "NetworkZkProofBundle",
    "ZkProofBundle",
    "ZkWitness",
    "ZkValidationResult",
    "ZkVerifier",
    "ZkProver",
    "InvalidZkProofError",
    "deserialize_proof_bundle",
    "TauNetZkAdapter",
    "ZkValidationStep",
    "ValidationContext",
    "ZkConfig",
    "BlockZkExtension",
    "compute_zk_merkle_root",
    "ZkGossipProtocol",
    "TAU_PROTOCOL_ZK_PROOFS",
    "apply_verified_transition",
    "get_zk_verifier",
    "set_zk_verifier",
]
