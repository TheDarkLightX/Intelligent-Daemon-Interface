"""ZK validation step for transaction validation pipeline.

This module implements the Chain of Responsibility pattern for ZK proof
validation, allowing it to be integrated into Tau Testnet's transaction
validation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from idi.taunet_bridge.protocols import ZkProofBundle, ZkVerifier, InvalidZkProofError


@dataclass
class ValidationContext:
    """Context for transaction validation.

    Contains transaction data and validation state, following the
    Context pattern for passing data through validation pipeline.
    """

    tx_hash: str
    payload: Dict[str, Any]
    zk_proof: ZkProofBundle | None = None

    def __post_init__(self):
        """Extract ZK proof from payload if present."""
        if "zk_proof" in self.payload:
            proof_data = self.payload["zk_proof"]
            if isinstance(proof_data, ZkProofBundle):
                self.zk_proof = proof_data
            elif isinstance(proof_data, dict):
                # Deserialize from dict if needed
                self.zk_proof = ZkProofBundle.deserialize(
                    proof_data.get("serialized", b"")
                )

    @property
    def has_zk_proof(self) -> bool:
        """Check if context contains a ZK proof."""
        return self.zk_proof is not None


class ZkValidationStep:
    """Chain-of-responsibility step for ZK proof validation.

    This class implements a validation step that can be inserted into
    Tau Testnet's transaction validation pipeline. It follows the
    Chain of Responsibility pattern and Dependency Inversion Principle (DIP)
    by depending on the ZkVerifier interface rather than concrete implementations.
    """

    def __init__(self, verifier: ZkVerifier, required: bool = False):
        """Initialize validation step.

        Args:
            verifier: ZK proof verifier (following DIP)
            required: If True, reject transactions without proofs
        """
        self._verifier = verifier
        self._required = required

    def run(self, ctx: ValidationContext) -> None:
        """Run ZK validation step.

        Args:
            ctx: Validation context containing transaction data

        Raises:
            InvalidZkProofError: If proof is invalid or missing (when required)
        """
        if not ctx.has_zk_proof:
            if self._required:
                raise InvalidZkProofError(
                    ctx.tx_hash, reason="ZK proof required but missing"
                )
            return  # Optional ZK verification, skip if no proof

        if not self._verifier.verify(ctx.zk_proof):
            raise InvalidZkProofError(ctx.tx_hash, reason="Proof verification failed")

