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
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg is not None:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


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


def x__pack_streams__mutmut_orig(stream_dir: Path) -> bytes:
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


def x__pack_streams__mutmut_1(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = None
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_2(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=None, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_3(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode=None, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_4(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=None) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_5(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_6(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_7(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", ) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_8(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="XXwbXX", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_9(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="WB", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_10(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=1) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_11(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=None, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_12(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode=None) as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_13(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(mode="w") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_14(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, ) as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_15(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="XXwXX") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_16(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="W") as tar:
            for f in sorted(stream_dir.glob("*.in")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_17(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(None):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_18(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob(None)):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_19(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("XX*.inXX")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_20(stream_dir: Path) -> bytes:
    """Pack stream files into tar.gz bytes.
    
    Args:
        stream_dir: Directory containing stream files
        
    Returns:
        Compressed tar archive bytes
    """
    buf = io.BytesIO()
    
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for f in sorted(stream_dir.glob("*.IN")):
                # Use relative name in archive
                tar.add(str(f), arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_21(stream_dir: Path) -> bytes:
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
                tar.add(None, arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_22(stream_dir: Path) -> bytes:
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
                tar.add(str(f), arcname=None)
    
    return buf.getvalue()


def x__pack_streams__mutmut_23(stream_dir: Path) -> bytes:
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
                tar.add(arcname=f.name)
    
    return buf.getvalue()


def x__pack_streams__mutmut_24(stream_dir: Path) -> bytes:
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
                tar.add(str(f), )
    
    return buf.getvalue()


def x__pack_streams__mutmut_25(stream_dir: Path) -> bytes:
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
                tar.add(str(None), arcname=f.name)
    
    return buf.getvalue()

x__pack_streams__mutmut_mutants : ClassVar[MutantDict] = {
'x__pack_streams__mutmut_1': x__pack_streams__mutmut_1, 
    'x__pack_streams__mutmut_2': x__pack_streams__mutmut_2, 
    'x__pack_streams__mutmut_3': x__pack_streams__mutmut_3, 
    'x__pack_streams__mutmut_4': x__pack_streams__mutmut_4, 
    'x__pack_streams__mutmut_5': x__pack_streams__mutmut_5, 
    'x__pack_streams__mutmut_6': x__pack_streams__mutmut_6, 
    'x__pack_streams__mutmut_7': x__pack_streams__mutmut_7, 
    'x__pack_streams__mutmut_8': x__pack_streams__mutmut_8, 
    'x__pack_streams__mutmut_9': x__pack_streams__mutmut_9, 
    'x__pack_streams__mutmut_10': x__pack_streams__mutmut_10, 
    'x__pack_streams__mutmut_11': x__pack_streams__mutmut_11, 
    'x__pack_streams__mutmut_12': x__pack_streams__mutmut_12, 
    'x__pack_streams__mutmut_13': x__pack_streams__mutmut_13, 
    'x__pack_streams__mutmut_14': x__pack_streams__mutmut_14, 
    'x__pack_streams__mutmut_15': x__pack_streams__mutmut_15, 
    'x__pack_streams__mutmut_16': x__pack_streams__mutmut_16, 
    'x__pack_streams__mutmut_17': x__pack_streams__mutmut_17, 
    'x__pack_streams__mutmut_18': x__pack_streams__mutmut_18, 
    'x__pack_streams__mutmut_19': x__pack_streams__mutmut_19, 
    'x__pack_streams__mutmut_20': x__pack_streams__mutmut_20, 
    'x__pack_streams__mutmut_21': x__pack_streams__mutmut_21, 
    'x__pack_streams__mutmut_22': x__pack_streams__mutmut_22, 
    'x__pack_streams__mutmut_23': x__pack_streams__mutmut_23, 
    'x__pack_streams__mutmut_24': x__pack_streams__mutmut_24, 
    'x__pack_streams__mutmut_25': x__pack_streams__mutmut_25
}

def _pack_streams(*args, **kwargs):
    result = _mutmut_trampoline(x__pack_streams__mutmut_orig, x__pack_streams__mutmut_mutants, args, kwargs)
    return result 

_pack_streams.__signature__ = _mutmut_signature(x__pack_streams__mutmut_orig)
x__pack_streams__mutmut_orig.__name__ = 'x__pack_streams'


def x__unpack_streams__mutmut_orig(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_1(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = None
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_2(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(None)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_3(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=None, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_4(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode=None) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_5(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_6(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, ) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_7(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="XXrbXX") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_8(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="RB") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_9(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=None, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_10(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode=None) as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_11(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_12(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, ) as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_13(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="XXrXX") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_14(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="R") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_15(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") and ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_16(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith(None) or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_17(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("XX/XX") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_18(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or "XX..XX" in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_19(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." not in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_20(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(None)
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_21(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_22(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(None):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_23(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith("XX.inXX"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_24(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".IN"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_25(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    break  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_26(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = None
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_27(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir * member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_28(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(None) as f:
                        if f:
                            member_path.write_bytes(f.read())


def x__unpack_streams__mutmut_29(pack_bytes: bytes, dest_dir: Path) -> None:
    """Unpack stream tar.gz to directory.
    
    Args:
        pack_bytes: Compressed tar archive
        dest_dir: Destination directory
    """
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.name.endswith(".in"):
                    continue  # Skip non-.in files
                
                # Extract safely
                member_path = dest_dir / member.name
                if member.isfile():
                    with tar.extractfile(member) as f:
                        if f:
                            member_path.write_bytes(None)

x__unpack_streams__mutmut_mutants : ClassVar[MutantDict] = {
'x__unpack_streams__mutmut_1': x__unpack_streams__mutmut_1, 
    'x__unpack_streams__mutmut_2': x__unpack_streams__mutmut_2, 
    'x__unpack_streams__mutmut_3': x__unpack_streams__mutmut_3, 
    'x__unpack_streams__mutmut_4': x__unpack_streams__mutmut_4, 
    'x__unpack_streams__mutmut_5': x__unpack_streams__mutmut_5, 
    'x__unpack_streams__mutmut_6': x__unpack_streams__mutmut_6, 
    'x__unpack_streams__mutmut_7': x__unpack_streams__mutmut_7, 
    'x__unpack_streams__mutmut_8': x__unpack_streams__mutmut_8, 
    'x__unpack_streams__mutmut_9': x__unpack_streams__mutmut_9, 
    'x__unpack_streams__mutmut_10': x__unpack_streams__mutmut_10, 
    'x__unpack_streams__mutmut_11': x__unpack_streams__mutmut_11, 
    'x__unpack_streams__mutmut_12': x__unpack_streams__mutmut_12, 
    'x__unpack_streams__mutmut_13': x__unpack_streams__mutmut_13, 
    'x__unpack_streams__mutmut_14': x__unpack_streams__mutmut_14, 
    'x__unpack_streams__mutmut_15': x__unpack_streams__mutmut_15, 
    'x__unpack_streams__mutmut_16': x__unpack_streams__mutmut_16, 
    'x__unpack_streams__mutmut_17': x__unpack_streams__mutmut_17, 
    'x__unpack_streams__mutmut_18': x__unpack_streams__mutmut_18, 
    'x__unpack_streams__mutmut_19': x__unpack_streams__mutmut_19, 
    'x__unpack_streams__mutmut_20': x__unpack_streams__mutmut_20, 
    'x__unpack_streams__mutmut_21': x__unpack_streams__mutmut_21, 
    'x__unpack_streams__mutmut_22': x__unpack_streams__mutmut_22, 
    'x__unpack_streams__mutmut_23': x__unpack_streams__mutmut_23, 
    'x__unpack_streams__mutmut_24': x__unpack_streams__mutmut_24, 
    'x__unpack_streams__mutmut_25': x__unpack_streams__mutmut_25, 
    'x__unpack_streams__mutmut_26': x__unpack_streams__mutmut_26, 
    'x__unpack_streams__mutmut_27': x__unpack_streams__mutmut_27, 
    'x__unpack_streams__mutmut_28': x__unpack_streams__mutmut_28, 
    'x__unpack_streams__mutmut_29': x__unpack_streams__mutmut_29
}

def _unpack_streams(*args, **kwargs):
    result = _mutmut_trampoline(x__unpack_streams__mutmut_orig, x__unpack_streams__mutmut_mutants, args, kwargs)
    return result 

_unpack_streams.__signature__ = _mutmut_signature(x__unpack_streams__mutmut_orig)
x__unpack_streams__mutmut_orig.__name__ = 'x__unpack_streams'


def x__unpack_streams_to_dict__mutmut_orig(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_1(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = None
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_2(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = None
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_3(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(None)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_4(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=None, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_5(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode=None) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_6(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_7(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, ) as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_8(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="XXrbXX") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_9(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="RB") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_10(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=None, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_11(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode=None) as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_12(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_13(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, ) as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_14(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="XXrXX") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_15(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="R") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_16(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") and ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_17(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith(None) or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_18(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("XX/XX") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_19(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or "XX..XX" in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_20(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." not in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_21(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(None)
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_22(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_23(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    break
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_24(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(None) as f:
                    if f:
                        result[member.name] = f.read()
    
    return result


def x__unpack_streams_to_dict__mutmut_25(pack_bytes: bytes) -> Dict[str, bytes]:
    """Unpack stream tar.gz to memory dict.
    
    Args:
        pack_bytes: Compressed tar archive
        
    Returns:
        Dictionary of filename -> content
    """
    result: Dict[str, bytes] = {}
    buf = io.BytesIO(pack_bytes)
    
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        with tarfile.open(fileobj=gz, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
                if not member.isfile():
                    continue
                
                with tar.extractfile(member) as f:
                    if f:
                        result[member.name] = None
    
    return result

x__unpack_streams_to_dict__mutmut_mutants : ClassVar[MutantDict] = {
'x__unpack_streams_to_dict__mutmut_1': x__unpack_streams_to_dict__mutmut_1, 
    'x__unpack_streams_to_dict__mutmut_2': x__unpack_streams_to_dict__mutmut_2, 
    'x__unpack_streams_to_dict__mutmut_3': x__unpack_streams_to_dict__mutmut_3, 
    'x__unpack_streams_to_dict__mutmut_4': x__unpack_streams_to_dict__mutmut_4, 
    'x__unpack_streams_to_dict__mutmut_5': x__unpack_streams_to_dict__mutmut_5, 
    'x__unpack_streams_to_dict__mutmut_6': x__unpack_streams_to_dict__mutmut_6, 
    'x__unpack_streams_to_dict__mutmut_7': x__unpack_streams_to_dict__mutmut_7, 
    'x__unpack_streams_to_dict__mutmut_8': x__unpack_streams_to_dict__mutmut_8, 
    'x__unpack_streams_to_dict__mutmut_9': x__unpack_streams_to_dict__mutmut_9, 
    'x__unpack_streams_to_dict__mutmut_10': x__unpack_streams_to_dict__mutmut_10, 
    'x__unpack_streams_to_dict__mutmut_11': x__unpack_streams_to_dict__mutmut_11, 
    'x__unpack_streams_to_dict__mutmut_12': x__unpack_streams_to_dict__mutmut_12, 
    'x__unpack_streams_to_dict__mutmut_13': x__unpack_streams_to_dict__mutmut_13, 
    'x__unpack_streams_to_dict__mutmut_14': x__unpack_streams_to_dict__mutmut_14, 
    'x__unpack_streams_to_dict__mutmut_15': x__unpack_streams_to_dict__mutmut_15, 
    'x__unpack_streams_to_dict__mutmut_16': x__unpack_streams_to_dict__mutmut_16, 
    'x__unpack_streams_to_dict__mutmut_17': x__unpack_streams_to_dict__mutmut_17, 
    'x__unpack_streams_to_dict__mutmut_18': x__unpack_streams_to_dict__mutmut_18, 
    'x__unpack_streams_to_dict__mutmut_19': x__unpack_streams_to_dict__mutmut_19, 
    'x__unpack_streams_to_dict__mutmut_20': x__unpack_streams_to_dict__mutmut_20, 
    'x__unpack_streams_to_dict__mutmut_21': x__unpack_streams_to_dict__mutmut_21, 
    'x__unpack_streams_to_dict__mutmut_22': x__unpack_streams_to_dict__mutmut_22, 
    'x__unpack_streams_to_dict__mutmut_23': x__unpack_streams_to_dict__mutmut_23, 
    'x__unpack_streams_to_dict__mutmut_24': x__unpack_streams_to_dict__mutmut_24, 
    'x__unpack_streams_to_dict__mutmut_25': x__unpack_streams_to_dict__mutmut_25
}

def _unpack_streams_to_dict(*args, **kwargs):
    result = _mutmut_trampoline(x__unpack_streams_to_dict__mutmut_orig, x__unpack_streams_to_dict__mutmut_mutants, args, kwargs)
    return result 

_unpack_streams_to_dict.__signature__ = _mutmut_signature(x__unpack_streams_to_dict__mutmut_orig)
x__unpack_streams_to_dict__mutmut_orig.__name__ = 'x__unpack_streams_to_dict'
