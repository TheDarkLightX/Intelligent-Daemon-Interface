"""Commitment Spec V1 - Deterministic artifact commitment.

⚠️ NOTE: This V1 encoding format is NOT currently used by Risc0 proofs!
The Risc0 guest (idi-manifest) uses a simpler format without the version prefix.
For verification of Risc0 proofs, use `compute_artifact_digest` from proof_manager.py.

This module provides bytes-first commitment computation that is
independent of filesystem representation. This enables:
- Network-portable commitment verification (from wire bundles)
- Deterministic cross-platform hashing
- Consistent encoding regardless of file paths

FUTURE: When Rust guest is updated to use V1 format, this will become primary. (V1):
    preimage = b"IDI_COMMITMENT_V1\\0" ||
               concat over sorted(names, case-insensitive)(
                   u32_le(name_len) || name_bytes || u64_le(data_len) || data_bytes
               )
    commitment = SHA-256(preimage)

This encoding is unambiguous and versioned for forward compatibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional


# Commitment version prefix (null-terminated for C compatibility)
_COMMITMENT_V1_PREFIX = b"IDI_COMMITMENT_V1\x00"


@dataclass(frozen=True)
class CommitmentSpecV1:
    """Versioned commitment specification.
    
    Attributes:
        version: Specification version
        hash_algorithm: Hash algorithm used
        encoding: Encoding scheme for preimage
    """
    version: Literal["1.0"] = "1.0"
    hash_algorithm: Literal["sha256"] = "sha256"
    encoding: Literal["length_prefixed"] = "length_prefixed"


def compute_commitment_bytes(
    manifest_bytes: bytes,
    streams: Dict[str, bytes],
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> str:
    """Compute commitment from raw bytes.
    
    This is the canonical, network-portable commitment computation.
    Produces identical results regardless of filesystem or platform.
    
    Args:
        manifest_bytes: Raw manifest content (JSON bytes)
        streams: Dictionary of stream name -> content
        extra_bindings: Optional extra data to bind (e.g., policy_root)
        
    Returns:
        Hex-encoded SHA-256 digest
        
    Encoding:
        For each (name, data) pair:
            - name_len: u32 LE (4 bytes)
            - name: UTF-8 bytes
            - data_len: u64 LE (8 bytes)
            - data: raw bytes
            
        Order:
            1. "manifest" + manifest_bytes
            2. "streams/<name>" + stream_bytes (sorted case-insensitively)
            3. "extra/<key>" + extra_bytes (sorted case-insensitively)
    """
    hasher = hashlib.sha256()
    
    # Version prefix
    hasher.update(_COMMITMENT_V1_PREFIX)
    
    def _update(name: str, data: bytes) -> None:
        """Update hasher with length-prefixed name and data."""
        name_bytes = name.encode("utf-8")
        hasher.update(len(name_bytes).to_bytes(4, "little"))
        hasher.update(name_bytes)
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    
    # Manifest always first
    _update("manifest", manifest_bytes)
    
    # Streams in case-insensitive sorted order
    for name in sorted(streams.keys(), key=str.lower):
        _update(f"streams/{name}", streams[name])
    
    # Extra bindings in case-insensitive sorted order
    if extra_bindings:
        for key in sorted(extra_bindings.keys(), key=str.lower):
            _update(f"extra/{key}", extra_bindings[key])
    
    return hasher.hexdigest()


def compute_commitment_fs(
    manifest_path: Path,
    stream_dir: Path,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> str:
    """Compute commitment from filesystem paths.
    
    Thin wrapper around compute_commitment_bytes for convenience.
    
    Args:
        manifest_path: Path to manifest JSON file
        stream_dir: Directory containing stream files (*.in)
        extra_bindings: Optional extra data to bind
        
    Returns:
        Hex-encoded SHA-256 digest
        
    Raises:
        FileNotFoundError: If manifest or stream files don't exist
    """
    manifest_bytes = manifest_path.read_bytes()
    
    streams: Dict[str, bytes] = {}
    if stream_dir.exists():
        # Collect all .in files
        for f in sorted(stream_dir.glob("*.in"), key=lambda p: p.name.lower()):
            streams[f.name] = f.read_bytes()
    
    return compute_commitment_bytes(manifest_bytes, streams, extra_bindings)


def compute_commitment_preimage(
    manifest_bytes: bytes,
    streams: Dict[str, bytes],
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> bytes:
    """Compute the raw preimage bytes (for debugging/testing).
    
    Same logic as compute_commitment_bytes but returns the preimage
    instead of the hash.
    
    Args:
        manifest_bytes: Raw manifest content
        streams: Stream name -> content mapping
        extra_bindings: Optional extra data
        
    Returns:
        Raw preimage bytes
    """
    parts: list[bytes] = [_COMMITMENT_V1_PREFIX]
    
    def _encode(name: str, data: bytes) -> bytes:
        """Encode a single name/data pair."""
        name_bytes = name.encode("utf-8")
        return (
            len(name_bytes).to_bytes(4, "little") +
            name_bytes +
            len(data).to_bytes(8, "little") +
            data
        )
    
    # Manifest
    parts.append(_encode("manifest", manifest_bytes))
    
    # Streams
    for name in sorted(streams.keys(), key=str.lower):
        parts.append(_encode(f"streams/{name}", streams[name]))
    
    # Extra bindings
    if extra_bindings:
        for key in sorted(extra_bindings.keys(), key=str.lower):
            parts.append(_encode(f"extra/{key}", extra_bindings[key]))
    
    return b"".join(parts)


# Alias for backwards compatibility with existing code
def compute_artifact_digest(
    manifest_path: Path,
    stream_dir: Path,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> str:
    """Compute artifact digest (alias for compute_commitment_fs).
    
    Deprecated: Use compute_commitment_fs instead.
    
    This alias maintains backwards compatibility with existing code.
    """
    return compute_commitment_fs(manifest_path, stream_dir, extra_bindings)
