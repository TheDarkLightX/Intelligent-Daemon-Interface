"""Block extension for ZK proof integration.

This module extends Tau Testnet's Block structure with ZK proof support,
following the Open/Closed Principle (OCP) by extending rather than modifying
the core Block class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from idi.taunet_bridge.protocols import ZkProofBundle


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hash and return hex string."""
    return hashlib.sha256(data).hexdigest()


def compute_zk_merkle_root(proofs: List[ZkProofBundle]) -> str:
    """Compute Merkle root from list of ZK proof bundles.

    This function follows the same Merkle tree construction as Tau Testnet's
    block.py compute_merkle_root function, ensuring consistency.

    Args:
        proofs: List of ZK proof bundles

    Returns:
        Hex-encoded SHA256 Merkle root
    """
    if not proofs:
        return sha256_hex(b"")

    # Compute hash for each proof (using receipt digest if available)
    proof_hashes = []
    for proof in proofs:
        # Use receipt path as identifier, hash the receipt content if available
        proof_id = str(proof.receipt_path)
        if proof.receipt_path.exists():
            receipt_data = proof.receipt_path.read_bytes()
        else:
            receipt_data = proof_id.encode()
        proof_hash = sha256_hex(receipt_data)
        proof_hashes.append(proof_hash)

    # Build Merkle tree (same algorithm as block.py)
    level = [bytes.fromhex(h) for h in proof_hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # Duplicate odd node
        next_level = []
        for i in range(0, len(level), 2):
            combined = level[i] + level[i + 1]
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level

    return level[0].hex()


@dataclass
class BlockZkExtension:
    """Extension data for Block with ZK proofs.

    This dataclass can be attached to Block instances without modifying
    the core Block class, following the Open/Closed Principle.
    """

    zk_commitment: Optional[str] = None  # Merkle root of ZK proofs
    zk_proofs: List[ZkProofBundle] = field(default_factory=list)

    def __post_init__(self):
        """Compute commitment if proofs are provided."""
        if self.zk_proofs and not self.zk_commitment:
            self.zk_commitment = compute_zk_merkle_root(self.zk_proofs)

    def add_proof(self, proof: ZkProofBundle) -> None:
        """Add a proof and recompute commitment."""
        self.zk_proofs.append(proof)
        self.zk_commitment = compute_zk_merkle_root(self.zk_proofs)

    def serialize(self) -> bytes:
        """Serialize extension to bytes."""
        data = {
            "zk_commitment": self.zk_commitment,
            "zk_proofs": [p.serialize().hex() for p in self.zk_proofs],
        }
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> BlockZkExtension:
        """Deserialize extension from bytes."""
        obj = json.loads(data.decode())
        proofs = [
            ZkProofBundle.deserialize(bytes.fromhex(hex_str))
            for hex_str in obj.get("zk_proofs", [])
        ]
        return cls(
            zk_commitment=obj.get("zk_commitment"),
            zk_proofs=proofs,
        )

