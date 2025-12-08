"""ZK proof protocols and data models following SOLID principles.

This module defines the core interfaces and data structures for ZK proof
integration, following Interface Segregation Principle (ISP) and ensuring
type safety throughout the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

from idi.zk.proof_manager import ProofBundle as IdiProofBundle


class InvalidZkProofError(Exception):
    """Raised when ZK proof verification fails."""

    def __init__(self, tx_hash: str, reason: Optional[str] = None):
        self.tx_hash = tx_hash
        self.reason = reason
        super().__init__(f"Invalid ZK proof for transaction {tx_hash}: {reason or 'verification failed'}")


@dataclass
class ZkProofBundle:
    """ZK proof bundle containing proof binary, receipt, and manifest.

    This dataclass provides a typed interface for ZK proofs, ensuring
    type safety and enabling serialization for network transmission.
    
    For network portability, this class can serialize either:
    - File paths (for local development)
    - Actual proof bytes (for network transmission)
    """

    proof_path: Optional[Path] = None
    receipt_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    # Network-portable fields (serialize actual bytes)
    proof_bytes: Optional[bytes] = None
    receipt_json: Optional[dict] = None
    manifest_bytes: Optional[bytes] = None
    tx_hash: Optional[str] = None
    timestamp: Optional[int] = None

    def serialize(self) -> bytes:
        """Serialize proof bundle to bytes for network transmission.
        
        Serializes actual proof/receipt/manifest bytes, not file paths,
        for network portability across machines.
        """
        # Prefer bytes over paths for network transmission
        if self.proof_bytes and self.receipt_json and self.manifest_bytes:
            import base64
            data = {
                "proof_bytes": base64.b64encode(self.proof_bytes).decode(),
                "receipt": self.receipt_json,
                "manifest_bytes": base64.b64encode(self.manifest_bytes).decode(),
                "tx_hash": self.tx_hash,
                "timestamp": self.timestamp,
            }
        else:
            # Fallback to paths for local development
            if not all([self.proof_path, self.receipt_path, self.manifest_path]):
                raise ValueError("Either bytes or paths must be provided")
            data = {
                "proof_path": str(self.proof_path),
                "receipt_path": str(self.receipt_path),
                "manifest_path": str(self.manifest_path),
                "tx_hash": self.tx_hash,
                "timestamp": self.timestamp,
            }
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> ZkProofBundle:
        """Deserialize proof bundle from bytes."""
        obj = json.loads(data.decode())
        
        # Check if bytes are present (network format)
        if "proof_bytes" in obj and "receipt" in obj and "manifest_bytes" in obj:
            import base64
            return cls(
                proof_bytes=base64.b64decode(obj["proof_bytes"]),
                receipt_json=obj["receipt"],
                manifest_bytes=base64.b64decode(obj["manifest_bytes"]),
                tx_hash=obj.get("tx_hash"),
                timestamp=obj.get("timestamp"),
            )
        else:
            # Fallback to paths (local format)
            return cls(
                proof_path=Path(obj["proof_path"]),
                receipt_path=Path(obj["receipt_path"]),
                manifest_path=Path(obj["manifest_path"]),
                tx_hash=obj.get("tx_hash"),
                timestamp=obj.get("timestamp"),
            )
    
    def load_from_paths(self) -> None:
        """Load bytes from file paths (for conversion from path-based to bytes-based)."""
        if self.proof_path and self.receipt_path and self.manifest_path:
            self.proof_bytes = self.proof_path.read_bytes()
            self.receipt_json = json.loads(self.receipt_path.read_text())
            self.manifest_bytes = self.manifest_path.read_bytes()
    
    def save_to_paths(self, base_dir: Path) -> None:
        """Save bytes to file paths (for local development/testing)."""
        if self.proof_bytes and self.receipt_json and self.manifest_bytes:
            base_dir.mkdir(parents=True, exist_ok=True)
            self.proof_path = base_dir / "proof.bin"
            self.receipt_path = base_dir / "receipt.json"
            self.manifest_path = base_dir / "manifest.json"
            self.proof_path.write_bytes(self.proof_bytes)
            self.receipt_path.write_text(json.dumps(self.receipt_json, indent=2))
            self.manifest_path.write_bytes(self.manifest_bytes)

    def to_idi_bundle(self) -> IdiProofBundle:
        """Convert to IDI ProofBundle for integration with existing infrastructure."""
        if not all([self.proof_path, self.receipt_path, self.manifest_path]):
            raise ValueError("Cannot convert to IdiProofBundle: paths not available")
        return IdiProofBundle(
            manifest_path=self.manifest_path,
            proof_path=self.proof_path,
            receipt_path=self.receipt_path,
        )


@dataclass
class ZkWitness:
    """ZK witness data for Q-table proof generation.

    Contains the Q-table entry, Merkle proof (if applicable), and state key
    needed to generate a ZK proof.
    """

    state_key: str
    q_table_root: bytes  # 32-byte SHA256 hash
    selected_action: int  # 0=hold, 1=buy, 2=sell
    merkle_proof: Optional[List[tuple[bytes, bool]]] = None  # List of (sibling_hash, is_right) tuples
    q_entry: Optional[dict[str, float]] = None  # Q-table entry values

    def serialize(self) -> bytes:
        """Serialize witness to bytes."""
        data = {
            "state_key": self.state_key,
            "q_table_root": self.q_table_root.hex(),
            "selected_action": self.selected_action,
            "q_entry": self.q_entry,
        }
        if self.merkle_proof:
            data["merkle_proof"] = [
                {"sibling_hash": h.hex(), "is_right": is_right}
                for h, is_right in self.merkle_proof
            ]
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> ZkWitness:
        """Deserialize witness from bytes."""
        obj = json.loads(data.decode())
        merkle_proof = None
        if "merkle_proof" in obj:
            merkle_proof = [
                (bytes.fromhex(item["sibling_hash"]), item["is_right"])
                for item in obj["merkle_proof"]
            ]
        return cls(
            state_key=obj["state_key"],
            q_table_root=bytes.fromhex(obj["q_table_root"]),
            selected_action=obj["selected_action"],
            merkle_proof=merkle_proof,
            q_entry=obj.get("q_entry"),
        )


@dataclass
class ZkValidationResult:
    """Result of ZK proof validation."""

    success: bool
    error: Optional[InvalidZkProofError] = None

    def __post_init__(self):
        """Validate result consistency."""
        if self.success and self.error is not None:
            raise ValueError("Successful validation cannot have an error")
        if not self.success and self.error is None:
            raise ValueError("Failed validation must have an error")


class ZkVerifier(Protocol):
    """Protocol for ZK proof verification.

    Following Interface Segregation Principle (ISP), this protocol defines
    only the verification interface, separate from proof generation.
    """

    def verify(self, proof: ZkProofBundle) -> bool:
        """Verify a ZK proof bundle.

        Args:
            proof: The ZK proof bundle to verify

        Returns:
            True if proof is valid, False otherwise

        Raises:
            InvalidZkProofError: If proof format is invalid (not just verification failure)
        """
        ...


class ZkProver(Protocol):
    """Protocol for ZK proof generation.

    Following Interface Segregation Principle (ISP), this protocol defines
    only the proof generation interface, separate from verification.
    """

    def prove(self, witness: ZkWitness) -> ZkProofBundle:
        """Generate a ZK proof from a witness.

        Args:
            witness: The witness data for proof generation

        Returns:
            A ZK proof bundle

        Raises:
            ValueError: If witness data is invalid
        """
        ...

