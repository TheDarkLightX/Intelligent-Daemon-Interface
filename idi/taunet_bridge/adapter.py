"""Adapter bridging IDI ZK infrastructure to Tau Testnet.

This module implements the Bridge pattern, adapting IDI ZK proof infrastructure
for use with Tau Testnet while maintaining clean separation of concerns.
"""

from __future__ import annotations

from pathlib import Path

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.zk.proof_manager import verify_proof as idi_verify_proof
# WitnessGenerator will be used for proof generation (future enhancement)

from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.protocols import ZkProofBundle, ZkVerifier


class TauNetZkAdapter(ZkVerifier):
    """Adapts IDI ZK infrastructure for Tau Testnet.

    This adapter implements the ZkVerifier protocol by delegating to IDI's
    proof_manager module, following the Adapter pattern and Dependency
    Inversion Principle (DIP).
    """

    def __init__(self, config: ZkConfig):
        """Initialize adapter with configuration.

        Args:
            config: ZK configuration settings
        """
        self._config = config
        self._merkle_builder = MerkleTreeBuilder()
        # Witness generator will be initialized lazily if needed for proof generation

    def verify(self, proof: ZkProofBundle) -> bool:
        """Verify a ZK proof bundle.

        Delegates to IDI's proof_manager.verify_proof function, converting
        the ZkProofBundle to IDI's ProofBundle format.

        Args:
            proof: The ZK proof bundle to verify

        Returns:
            True if proof is valid, False otherwise
        """
        if not self._config.enabled:
            return True  # ZK verification disabled, always pass

        # Convert to IDI ProofBundle format
        idi_bundle = proof.to_idi_bundle()

        # Delegate to IDI proof manager
        return idi_verify_proof(idi_bundle)

