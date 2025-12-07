"""Configuration for Tau Testnet ZK integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ZkConfig:
    """ZK proof configuration for Tau Testnet integration.

    This configuration class follows the Single Responsibility Principle (SRP)
    by containing only configuration-related data and validation logic.
    """

    enabled: bool = False  # Opt-in ZK verification
    proof_system: Literal["risc0", "stub"] = "stub"  # Proof system to use
    require_proofs: bool = False  # True = reject txs without proofs
    merkle_threshold: int = 100  # Use Merkle for tables > 100 entries
    max_proof_bytes: int = 256 * 1024  # Upper bound on proof blob read
    max_receipt_bytes: int = 64 * 1024  # Upper bound on receipt file read

    def __post_init__(self):
        """Validate configuration."""
        if self.proof_system not in ("risc0", "stub"):
            raise ValueError(f"Invalid proof_system: {self.proof_system}")
        if self.merkle_threshold < 0:
            raise ValueError(f"merkle_threshold must be >= 0, got {self.merkle_threshold}")
        if self.max_proof_bytes <= 0:
            raise ValueError("max_proof_bytes must be positive")
        if self.max_receipt_bytes <= 0:
            raise ValueError("max_receipt_bytes must be positive")
