"""Adapter bridging IDI ZK infrastructure to Tau Testnet.

This module implements the Bridge pattern, adapting IDI ZK proof infrastructure
for use with Tau Testnet while maintaining clean separation of concerns.
"""

from __future__ import annotations

import json
from pathlib import Path

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.zk import proof_manager
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

    def _verify_risc0(self, proof: ZkProofBundle) -> bool:
        """Verify using Risc0 receipt verification."""
        import subprocess
        import shlex
        from pathlib import Path
        
        # Get proof binary path or bytes
        proof_path: Optional[Path] = None
        if proof.proof_bytes:
            # Save bytes to temp file for verification
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp.write(proof.proof_bytes)
                proof_path = Path(tmp.name)
        elif proof.proof_path:
            proof_path = proof.proof_path
        else:
            return False
        
        if not proof_path or not proof_path.exists():
            return False
        
        # Check proof size
        try:
            if proof_path.stat().st_size > self._config.max_proof_bytes:
                return False
        except OSError:
            return False
        
        # Call Risc0 verifier
        try:
            # Find idi_risc0_host binary
            current_file = Path(__file__)
            risc0_host = current_file.parent.parent / "zk" / "risc0" / "host" / "target" / "release" / "idi_risc0_host"
            if not risc0_host.exists():
                # Try cargo run as fallback
                risc0_workspace = current_file.parent.parent / "zk" / "risc0"
                cmd = shlex.split(
                    f"cargo run --release -p idi_risc0_host -- verify --proof {proof_path}"
                )
                result = subprocess.run(
                    cmd,
                    cwd=str(risc0_workspace),
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            else:
                result = subprocess.run(
                    [str(risc0_host), "verify", "--proof", str(proof_path)],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            
            if result.returncode == 0:
                return True
            else:
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        finally:
            # Clean up temp file if we created one
            if proof.proof_bytes and proof_path:
                try:
                    proof_path.unlink()
                except OSError:
                    pass

    def _verify_stub(self, proof: ZkProofBundle) -> bool:
        """Verify using stub (digest-based) verification."""
        # Load bytes from paths if needed
        if not proof.proof_bytes and proof.proof_path:
            proof.load_from_paths()
        
        # Basic file existence and size checks
        if proof.proof_path and proof.receipt_path and proof.manifest_path:
            try:
                if not all(p.exists() for p in (proof.proof_path, proof.receipt_path, proof.manifest_path)):
                    return False
                if proof.proof_path.stat().st_size > self._config.max_proof_bytes:
                    return False
                if proof.receipt_path.stat().st_size > self._config.max_receipt_bytes:
                    return False
            except OSError:
                return False

        # Bind receipt to manifest + streams
        try:
            if proof.receipt_json:
                receipt = proof.receipt_json
            elif proof.receipt_path:
                receipt = json.loads(proof.receipt_path.read_text())
            else:
                return False
                
            receipt_manifest = Path(receipt.get("manifest", ""))
            receipt_streams = Path(receipt.get("streams", receipt_manifest.parent / "streams" if receipt_manifest else Path(".")))
            
            if proof.manifest_path and receipt_manifest.resolve() != proof.manifest_path.resolve():
                return False
                
            # Verify digest
            if proof.manifest_path and receipt_streams.exists():
                recomputed = proof_manager._combined_hash(receipt_manifest, receipt_streams)  # type: ignore[attr-defined]
                if recomputed != receipt.get("digest") and recomputed != receipt.get("digest_hex"):
                    return False
        except Exception:
            return False

        # Convert to IDI ProofBundle format and delegate
        if proof.proof_path and proof.receipt_path and proof.manifest_path:
            idi_bundle = proof.to_idi_bundle()
            try:
                return idi_verify_proof(idi_bundle)
            except Exception:
                return False
        
        return False
