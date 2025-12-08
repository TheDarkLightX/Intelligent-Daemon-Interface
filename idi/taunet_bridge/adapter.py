"""Adapter bridging IDI ZK infrastructure to Tau Testnet.

This module implements the Bridge pattern, adapting IDI ZK proof infrastructure
for use with Tau Testnet while maintaining clean separation of concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.zk import proof_manager
from idi.zk.proof_manager import verify_proof as idi_verify_proof
# WitnessGenerator will be used for proof generation (future enhancement)

from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.protocols import (
    LocalZkProofBundle,
    NetworkZkProofBundle,
    ZkProofBundle,
    ZkVerifier,
)


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

        Uses Risc0 receipt verification when proof_system="risc0",
        otherwise falls back to digest-based verification.

        Args:
            proof: The ZK proof bundle to verify

        Returns:
            True if proof is valid, False otherwise
        """
        if not self._config.enabled:
            return True  # ZK verification disabled, always pass

        # Require proofs to be explicitly bound to a tx hash when configured
        if self._config.require_proofs and not proof.tx_hash:
            return False

        # Branch on proof system
        if self._config.proof_system == "risc0":
            return self._verify_risc0(proof)
        else:
            return self._verify_stub(proof)

    def _materialize_local(self, proof: ZkProofBundle):
        """Return a LocalZkProofBundle plus optional temp dir for cleanup."""
        import tempfile

        if isinstance(proof, LocalZkProofBundle):
            local = proof
            temp_dir = None
        elif isinstance(proof, NetworkZkProofBundle):
            temp_dir = tempfile.TemporaryDirectory()
            local = proof.to_local(Path(temp_dir.name))
        else:
            raise ValueError("Unsupported proof bundle type")

        if local.stream_dir is None:
            default_streams = local.manifest_path.parent / "streams"
            if default_streams.exists():
                local.stream_dir = default_streams

        return local, temp_dir

    def _verify_risc0(self, proof: ZkProofBundle) -> bool:
        """Verify using Risc0 receipt verification."""
        local_bundle, temp_dir = self._materialize_local(proof)
        try:
            if not all(
                p.exists() for p in (local_bundle.proof_path, local_bundle.receipt_path, local_bundle.manifest_path)
            ):
                return False
            if local_bundle.proof_path.stat().st_size > self._config.max_proof_bytes:
                return False
            if local_bundle.receipt_path.stat().st_size > self._config.max_receipt_bytes:
                return False

            # Enforce tx_hash binding if present
            try:
                receipt = json.loads(local_bundle.receipt_path.read_bytes())
                if proof.tx_hash and receipt.get("tx_hash") and receipt.get("tx_hash") != proof.tx_hash:
                    return False
            except Exception:
                return False

            idi_bundle = local_bundle.to_idi_bundle()
            return bool(idi_verify_proof(idi_bundle, use_risc0=True))
        except OSError:
            return False
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _verify_stub(self, proof: ZkProofBundle) -> bool:
        """Verify using stub (digest-based) verification."""
        local_bundle, temp_dir = self._materialize_local(proof)
        try:
            # Basic file existence and size checks
            if not all(
                p.exists() for p in (local_bundle.proof_path, local_bundle.receipt_path, local_bundle.manifest_path)
            ):
                return False
            if local_bundle.proof_path.stat().st_size > self._config.max_proof_bytes:
                return False
            if local_bundle.receipt_path.stat().st_size > self._config.max_receipt_bytes:
                return False

            # Bind receipt to manifest + streams
            try:
                receipt = json.loads(local_bundle.receipt_path.read_bytes())
                if proof.tx_hash and receipt.get("tx_hash") and receipt.get("tx_hash") != proof.tx_hash:
                    return False
                receipt_streams = local_bundle.stream_dir or Path(
                    receipt.get("streams", local_bundle.manifest_path.parent / "streams")
                )

                if local_bundle.manifest_path.exists() and receipt_streams.exists():
                    recomputed = proof_manager.compute_manifest_streams_digest(
                        local_bundle.manifest_path, receipt_streams
                    )
                    if recomputed != receipt.get("digest") and recomputed != receipt.get("digest_hex"):
                        return False
            except Exception:
                return False

            idi_bundle = local_bundle.to_idi_bundle()
            try:
                return idi_verify_proof(idi_bundle)
            except Exception:
                return False

            return False
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        return False
