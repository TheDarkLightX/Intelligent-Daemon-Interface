"""Wire Bundle V1 - Network-portable ZK proof bundles.

This module provides self-contained proof bundles that can be transmitted
over the network and verified without filesystem access to original files.

Wire bundles include:
- ZK receipt binary (base64)
- Attestation JSON (base64)
- Manifest JSON (base64)
- Streams pack (tar.gz, base64)
- Streams SHA-256 hash for integrity

Schema Version: 1.0
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from idi.zk.verification import VerificationErrorCode, VerificationReport


# Size limits (enforced before decoding)
MAX_WIRE_BUNDLE_BYTES = 10 * 1024 * 1024   # 10MB total
MAX_PROOF_BYTES = 5 * 1024 * 1024          # 5MB
MAX_ATTESTATION_BYTES = 512 * 1024         # 512KB
MAX_MANIFEST_BYTES = 1 * 1024 * 1024       # 1MB
MAX_STREAMS_BYTES = 8 * 1024 * 1024        # 8MB

# Decompression size limits (zip bomb protection)
MAX_UNCOMPRESSED_MEMBER_BYTES = 10 * 1024 * 1024   # 10MB per member
MAX_UNCOMPRESSED_TOTAL_BYTES = 50 * 1024 * 1024    # 50MB total


@dataclass
class ZkProofBundleLocal:
    """Local filesystem-based proof bundle.
    
    Represents a proof bundle with paths to files on disk.
    Can be converted to a wire bundle for network transmission.
    """
    
    proof_path: Path              # ZK receipt binary (e.g., Risc0 receipt)
    attestation_path: Path        # JSON metadata (formerly receipt.json)
    manifest_path: Path           # Manifest JSON
    stream_dir: Path              # Directory containing stream files
    tx_hash: Optional[str] = None
    timestamp: Optional[int] = None
    
    def to_wire(self, include_streams: bool = True) -> "ZkProofBundleWireV1":
        """Convert to wire format for network transmission.
        
        Args:
            include_streams: Whether to pack streams into the bundle
            
        Returns:
            Network-portable wire bundle
        """
        # Read and encode files
        proof_bytes = self.proof_path.read_bytes()
        attestation_bytes = self.attestation_path.read_bytes()
        manifest_bytes = self.manifest_path.read_bytes()
        
        # Determine proof system from attestation
        try:
            attestation = json.loads(attestation_bytes)
            proof_system = attestation.get("prover", "stub")
        except json.JSONDecodeError:
            proof_system = "stub"
        
        # Pack streams if requested
        streams_pack_b64: Optional[str] = None
        streams_sha256: Optional[str] = None
        
        if include_streams and self.stream_dir.exists():
            streams_pack = _pack_streams(self.stream_dir)
            streams_pack_b64 = base64.b64encode(streams_pack).decode("ascii")
            streams_sha256 = hashlib.sha256(streams_pack).hexdigest()
        
        return ZkProofBundleWireV1(
            schema_version="1.0",
            proof_system=proof_system if proof_system in ("risc0", "stub") else "stub",
            zk_receipt_bin_b64=base64.b64encode(proof_bytes).decode("ascii"),
            attestation_json_b64=base64.b64encode(attestation_bytes).decode("ascii"),
            manifest_json_b64=base64.b64encode(manifest_bytes).decode("ascii"),
            streams_pack_b64=streams_pack_b64,
            streams_sha256=streams_sha256,
            tx_hash=self.tx_hash,
            timestamp=self.timestamp,
        )


@dataclass
class ZkProofBundleWireV1:
    """Network-portable proof bundle with streams included.
    
    This bundle is self-contained and can be verified without access
    to the original filesystem.
    
    Attributes:
        schema_version: Wire format version (always "1.0")
        proof_system: Either "risc0" or "stub"
        zk_receipt_bin_b64: Base64-encoded ZK receipt binary
        attestation_json_b64: Base64-encoded attestation JSON
        manifest_json_b64: Base64-encoded manifest JSON
        streams_pack_b64: Base64-encoded tar.gz of streams (optional)
        streams_sha256: SHA-256 of streams_pack for integrity
        tx_hash: Optional transaction hash binding
        timestamp: Optional timestamp
    """
    
    schema_version: Literal["1.0"] = "1.0"
    proof_system: Literal["risc0", "stub"] = "stub"
    zk_receipt_bin_b64: str = ""
    attestation_json_b64: str = ""
    manifest_json_b64: str = ""
    streams_pack_b64: Optional[str] = None
    streams_sha256: Optional[str] = None
    tx_hash: Optional[str] = None
    timestamp: Optional[int] = None
    
    def serialize(self) -> bytes:
        """Serialize to JSON bytes with canonical ordering.
        
        Returns:
            JSON bytes suitable for network transmission
        """
        data = {
            "schema_version": self.schema_version,
            "proof_system": self.proof_system,
            "zk_receipt_bin_b64": self.zk_receipt_bin_b64,
            "attestation_json_b64": self.attestation_json_b64,
            "manifest_json_b64": self.manifest_json_b64,
            "streams_pack_b64": self.streams_pack_b64,
            "streams_sha256": self.streams_sha256,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    
    @classmethod
    def deserialize(cls, data: bytes) -> "ZkProofBundleWireV1":
        """Deserialize with validation.
        
        Args:
            data: JSON bytes
            
        Returns:
            ZkProofBundleWireV1 instance
            
        Raises:
            ValueError: On validation failure (size, hash mismatch, etc.)
        """
        # Check total size
        if len(data) > MAX_WIRE_BUNDLE_BYTES:
            raise ValueError(f"Wire bundle exceeds size limit ({MAX_WIRE_BUNDLE_BYTES})")
        
        obj = json.loads(data)
        
        # Validate schema version
        if obj.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported schema version: {obj.get('schema_version')}")
        
        # Validate proof system
        proof_system = obj.get("proof_system", "stub")
        if proof_system not in ("risc0", "stub"):
            raise ValueError(f"Invalid proof system: {proof_system}")
        
        # Estimate and check sizes before decoding
        for field_name, limit in [
            ("zk_receipt_bin_b64", MAX_PROOF_BYTES),
            ("attestation_json_b64", MAX_ATTESTATION_BYTES),
            ("manifest_json_b64", MAX_MANIFEST_BYTES),
            ("streams_pack_b64", MAX_STREAMS_BYTES),
        ]:
            if field_name in obj and obj[field_name]:
                # Estimate decoded size (base64 is ~4/3 of original)
                estimated = len(obj[field_name]) * 3 // 4
                if estimated > limit:
                    raise ValueError(f"{field_name} exceeds size limit ({limit})")
        
        # Verify streams digest if present
        if obj.get("streams_pack_b64") and obj.get("streams_sha256"):
            try:
                pack_bytes = base64.b64decode(obj["streams_pack_b64"], validate=True)
            except Exception as e:
                raise ValueError(f"Invalid streams_pack_b64: {e}")
            
            computed = hashlib.sha256(pack_bytes).hexdigest()
            if computed != obj["streams_sha256"]:
                raise ValueError(
                    f"streams_sha256 mismatch: expected {obj['streams_sha256']}, "
                    f"got {computed}"
                )
        
        return cls(
            schema_version=obj["schema_version"],
            proof_system=proof_system,
            zk_receipt_bin_b64=obj.get("zk_receipt_bin_b64", ""),
            attestation_json_b64=obj.get("attestation_json_b64", ""),
            manifest_json_b64=obj.get("manifest_json_b64", ""),
            streams_pack_b64=obj.get("streams_pack_b64"),
            streams_sha256=obj.get("streams_sha256"),
            tx_hash=obj.get("tx_hash"),
            timestamp=obj.get("timestamp"),
        )
    
    def to_local(self, base_dir: Path) -> ZkProofBundleLocal:
        """Materialize to local filesystem.
        
        Args:
            base_dir: Directory to write files to
            
        Returns:
            Local proof bundle with file paths
        """
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Write proof
        proof_path = base_dir / "proof.bin"
        proof_path.write_bytes(base64.b64decode(self.zk_receipt_bin_b64))
        
        # Write attestation
        attestation_path = base_dir / "attestation.json"
        attestation_path.write_bytes(base64.b64decode(self.attestation_json_b64))
        
        # Write manifest
        manifest_path = base_dir / "manifest.json"
        manifest_path.write_bytes(base64.b64decode(self.manifest_json_b64))
        
        # Unpack streams
        stream_dir = base_dir / "streams"
        stream_dir.mkdir(exist_ok=True)
        
        if self.streams_pack_b64:
            pack_bytes = base64.b64decode(self.streams_pack_b64)
            _unpack_streams(pack_bytes, stream_dir)
        
        return ZkProofBundleLocal(
            proof_path=proof_path,
            attestation_path=attestation_path,
            manifest_path=manifest_path,
            stream_dir=stream_dir,
            tx_hash=self.tx_hash,
            timestamp=self.timestamp,
        )
    
    def get_manifest_bytes(self) -> bytes:
        """Get decoded manifest bytes."""
        return base64.b64decode(self.manifest_json_b64)
    
    def get_attestation(self) -> Dict[str, Any]:
        """Get parsed attestation JSON."""
        return json.loads(base64.b64decode(self.attestation_json_b64))
    
    def get_streams(self) -> Dict[str, bytes]:
        """Get decoded streams as dict.
        
        Returns:
            Dictionary of stream name -> content
        """
        if not self.streams_pack_b64:
            return {}
        
        pack_bytes = base64.b64decode(self.streams_pack_b64)
        return _unpack_streams_to_dict(pack_bytes)
    
    def compute_commitment(self) -> str:
        """Compute commitment digest from bundle contents.
        
        Uses the CommitmentSpecV1 encoding for network-portable verification.
        
        Returns:
            Hex-encoded SHA-256 commitment digest
        """
        from idi.zk.commitment import compute_commitment_bytes
        
        manifest_bytes = self.get_manifest_bytes()
        streams = self.get_streams()
        
        return compute_commitment_bytes(manifest_bytes, streams)
    
    def verify_integrity(self) -> VerificationReport:
        """Verify bundle integrity (size limits, hash checks).
        
        Returns:
            VerificationReport indicating success or specific failure
        """
        # Check streams hash
        if self.streams_pack_b64 and self.streams_sha256:
            try:
                pack_bytes = base64.b64decode(self.streams_pack_b64)
                computed = hashlib.sha256(pack_bytes).hexdigest()
                if computed != self.streams_sha256:
                    return VerificationReport.fail(
                        VerificationErrorCode.STREAMS_DIGEST_MISMATCH,
                        "Streams hash mismatch",
                        expected=self.streams_sha256,
                        actual=computed,
                    )
            except Exception as e:
                return VerificationReport.fail(
                    VerificationErrorCode.RECEIPT_PARSE_ERROR,
                    f"Failed to decode streams: {e}",
                )
        
        return VerificationReport.ok()


def _pack_streams(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def _unpack_streams(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
        
    Raises:
        ValueError: If archive contains unsafe paths or exceeds size limits.
    """
    buf = io.BytesIO(pack_bytes)
    total_extracted = 0
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                
                # Security: Check individual member size (zip bomb protection)
                if member.size > MAX_UNCOMPRESSED_MEMBER_BYTES:
                    raise ValueError(
                        f"Member '{member.name}' exceeds size limit "
                        f"({member.size} > {MAX_UNCOMPRESSED_MEMBER_BYTES})"
                    )
                
                # Security: Check cumulative size
                total_extracted += member.size
                if total_extracted > MAX_UNCOMPRESSED_TOTAL_BYTES:
                    raise ValueError(
                        f"Total extracted size exceeds limit "
                        f"({total_extracted} > {MAX_UNCOMPRESSED_TOTAL_BYTES})"
                    )
                
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def _unpack_streams_to_dict(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
        
    Raises:
        ValueError: If archive contains unsafe paths or exceeds size limits.
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    total_extracted = 0
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                
                # Security: Check individual member size (zip bomb protection)
                if member.size > MAX_UNCOMPRESSED_MEMBER_BYTES:
                    raise ValueError(
                        f"Member '{member.name}' exceeds size limit "
                        f"({member.size} > {MAX_UNCOMPRESSED_MEMBER_BYTES})"
                    )
                
                # Security: Check cumulative size
                total_extracted += member.size
                if total_extracted > MAX_UNCOMPRESSED_TOTAL_BYTES:
                    raise ValueError(
                        f"Total extracted size exceeds limit "
                        f"({total_extracted} > {MAX_UNCOMPRESSED_TOTAL_BYTES})"
                    )
                
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result
