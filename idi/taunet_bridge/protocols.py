"""ZK proof protocols and data models following SOLID principles.

This module defines the core interfaces and data structures for ZK proof
integration, following Interface Segregation Principle (ISP) and ensuring
type safety throughout the system.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Union, runtime_checkable

from idi.zk.proof_manager import ProofBundle as IdiProofBundle


class InvalidZkProofError(Exception):
    """Raised when ZK proof verification fails."""

    def __init__(self, tx_hash: str, reason: Optional[str] = None):
        self.tx_hash = tx_hash
        self.reason = reason
        super().__init__(f"Invalid ZK proof for transaction {tx_hash}: {reason or 'verification failed'}")


@runtime_checkable
class ProofBundleProtocol(Protocol):
    """Common interface for local and network proof bundles."""

    tx_hash: Optional[str]
    timestamp: Optional[int]

    def serialize(self) -> bytes:
        """Serialize to network-portable bytes."""
        ...

    def to_network(self) -> "NetworkZkProofBundle":
        """Return a network-safe representation."""
        ...


@dataclass
class NetworkZkProofBundle(ProofBundleProtocol):
    """Network-portable proof bundle containing proof, receipt, and manifest bytes."""

    proof_bytes: bytes
    receipt_bytes: bytes
    manifest_bytes: bytes
    tx_hash: Optional[str] = None
    timestamp: Optional[int] = None
    stream_dir: Optional[Path] = None

    def serialize(self) -> bytes:
        """Serialize to JSON bytes with base64-encoded blobs."""
        data = {
            "proof_b64": base64.b64encode(self.proof_bytes).decode(),
            "receipt_b64": base64.b64encode(self.receipt_bytes).decode(),
            "manifest_b64": base64.b64encode(self.manifest_bytes).decode(),
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> "NetworkZkProofBundle":
        obj = json.loads(data.decode())

        def _decode(key: str) -> bytes:
            if key in obj:
                return base64.b64decode(obj[key])
            if key.replace("_b64", "_bytes") in obj:
                return base64.b64decode(obj[key.replace("_b64", "_bytes")])
            if key == "receipt_b64" and "receipt" in obj:
                # Backwards compatibility with receipt stored as JSON object
                return json.dumps(obj["receipt"]).encode()
            raise ValueError(f"Missing required field {key}")

        return cls(
            proof_bytes=_decode("proof_b64"),
            receipt_bytes=_decode("receipt_b64"),
            manifest_bytes=_decode("manifest_b64"),
            tx_hash=obj.get("tx_hash"),
            timestamp=obj.get("timestamp"),
            stream_dir=Path(obj["stream_dir"]) if obj.get("stream_dir") else None,
        )

    def to_local(self, base_dir: Path) -> "LocalZkProofBundle":
        """Persist bundle to disk for local verification."""
        base_dir.mkdir(parents=True, exist_ok=True)
        proof_path = base_dir / "proof.bin"
        receipt_path = base_dir / "receipt.json"
        manifest_path = base_dir / "manifest.json"
        proof_path.write_bytes(self.proof_bytes)
        receipt_path.write_bytes(self.receipt_bytes)
        manifest_path.write_bytes(self.manifest_bytes)
        return LocalZkProofBundle(
            proof_path=proof_path,
            receipt_path=receipt_path,
            manifest_path=manifest_path,
            tx_hash=self.tx_hash,
            timestamp=self.timestamp,
            stream_dir=self.stream_dir,
        )

    def to_network(self) -> "NetworkZkProofBundle":
        return self


@dataclass
class LocalZkProofBundle(ProofBundleProtocol):
    """Local proof bundle backed by filesystem paths."""

    proof_path: Path
    receipt_path: Path
    manifest_path: Path
    tx_hash: Optional[str] = None
    timestamp: Optional[int] = None
    stream_dir: Optional[Path] = None

    def serialize(self) -> bytes:
        """Serialize by converting to a network bundle (portable, hash-stable)."""
        return self.to_network().serialize()

    def to_network(self) -> NetworkZkProofBundle:
        return NetworkZkProofBundle(
            proof_bytes=self.proof_path.read_bytes(),
            receipt_bytes=self.receipt_path.read_bytes(),
            manifest_bytes=self.manifest_path.read_bytes(),
            tx_hash=self.tx_hash,
            timestamp=self.timestamp,
            stream_dir=self.stream_dir,
        )

    def to_idi_bundle(self) -> IdiProofBundle:
        """Convert to IDI ProofBundle for integration with existing infrastructure."""
        return IdiProofBundle(
            manifest_path=self.manifest_path,
            proof_path=self.proof_path,
            receipt_path=self.receipt_path,
            stream_dir=self.stream_dir,
        )


ZkProofBundle = Union[LocalZkProofBundle, NetworkZkProofBundle]


def deserialize_proof_bundle(data: bytes | str) -> ZkProofBundle:
    """Deserialize bytes (or UTF-8 string) into the appropriate bundle type."""
    raw = data if isinstance(data, (bytes, bytearray)) else str(data).encode()
    obj = json.loads(raw.decode())

    if any(key in obj for key in ("proof_b64", "proof_bytes")):
        return NetworkZkProofBundle.deserialize(raw)

    if all(key in obj for key in ("proof_path", "receipt_path", "manifest_path")):
        return LocalZkProofBundle(
            proof_path=Path(obj["proof_path"]),
            receipt_path=Path(obj["receipt_path"]),
            manifest_path=Path(obj["manifest_path"]),
            tx_hash=obj.get("tx_hash"),
            timestamp=obj.get("timestamp"),
            stream_dir=Path(obj["stream_dir"]) if obj.get("stream_dir") else None,
        )

    raise ValueError("Unrecognized proof bundle format")


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
