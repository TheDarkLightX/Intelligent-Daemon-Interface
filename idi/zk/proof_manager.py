"""Proof bundle helpers for IDI zk workflows.

Manages proof generation and verification for manifest-based ZK proofs.
Supports both stub (SHA-256 only) and zkVM provers (Risc0).

Security Properties:
- Integrity: Proof digest binds manifest and stream data
- Determinism: Same inputs always produce same digest
- Verifiability: Anyone can verify proofs without proving key

Dependencies: hashlib, json, subprocess
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .secure_errors import SecureError, handle_file_operation_error, handle_json_parse_error

# Security constants
MAX_RECEIPT_SIZE_BYTES = 512 * 1024  # 512KB limit for receipt files


def _validate_path_safety(path: Path, base_dir: Optional[Path] = None) -> None:
    """Validate that a path is safe to access (no traversal, reasonable length)."""
    try:
        # Resolve to absolute path to check for traversal
        resolved = path.resolve()

        # Check for path traversal (should not go above base_dir if provided)
        if base_dir and not resolved.is_relative_to(base_dir):
            raise ValueError(f"Path traversal detected: {path}")

        # Check path length (prevent extremely long paths)
        if len(str(resolved)) > 4096:  # 4KB limit for path length
            raise ValueError(f"Path too long: {path}")

        # Check for null bytes in path (potential security issue)
        if '\x00' in str(path):
            raise ValueError(f"Invalid null byte in path: {path}")

    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {path} - {e}")

MAX_RECEIPT_SIZE_BYTES = 512 * 1024  # safety cap

# Default Risc0 command template
_DEFAULT_RISC0_CMD = (
    "cargo run --release -p idi_risc0_host -- "
    "--manifest {manifest} --streams {streams} --proof {proof} --receipt {receipt}"
)


def _detect_risc0_available() -> bool:
    """Check if Risc0 prover is available and built.
    
    Returns True if:
    - cargo is available
    - Risc0 workspace exists
    - idi_risc0_host package can be found
    """
    try:
        # Check if cargo is available
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        
        # Check if Risc0 workspace exists
        # Try to find the workspace from common locations
        import os
        current_file = Path(__file__)
        risc0_workspace = current_file.parent / "risc0" / "Cargo.toml"
        
        if not risc0_workspace.exists():
            # Try relative to project root
            project_root = current_file.parent.parent.parent
            risc0_workspace = project_root / "idi" / "zk" / "risc0" / "Cargo.toml"
            if not risc0_workspace.exists():
                return False
        
        # Check if the package exists in the workspace
        # This is a lightweight check - actual build verification would be expensive
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _get_default_prover_command() -> Optional[str]:
    """Get default prover command, auto-detecting Risc0 if available.
    
    Returns:
        Risc0 command template if available, None otherwise (will use stub).
    """
    if _detect_risc0_available():
        return _DEFAULT_RISC0_CMD
    return None


def compute_artifact_digest(
    manifest_path: Path, stream_dir: Path, extra: Optional[Dict[str, bytes]] = None
) -> str:
    hasher = hashlib.sha256()

    def _update(name: str, payload: bytes) -> None:
        hasher.update(name.encode("utf-8"))
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)

    if manifest_path.exists():
        _update("manifest", manifest_path.read_bytes())
    for stream_file in sorted(stream_dir.glob("*.in")):
        rel_name = f"streams/{stream_file.name}"
        _update(rel_name, stream_file.read_bytes())

    if extra:
        for key in sorted(extra.keys()):
            _update(f"extra/{key}", extra[key])
    return hasher.hexdigest()


def compute_manifest_streams_digest(manifest_path: Path, stream_dir: Path) -> str:
    """Public helper to compute manifest+streams digest (no extra bindings)."""
    return compute_artifact_digest(manifest_path, stream_dir)


@dataclass(frozen=True)
class ProofBundle:
    """Proof bundle containing manifest, proof binary, and receipt.
    
    Immutable dataclass representing a complete ZK proof bundle.
    All paths must exist and be valid for the bundle to be usable.
    
    Security: Frozen dataclass prevents accidental modification.
    """
    
    manifest_path: Path
    proof_path: Path
    receipt_path: Path
    stream_dir: Optional[Path] = None


def generate_proof(
    *,
    manifest_path: Path,
    stream_dir: Path,
    out_dir: Path,
    prover_command: Optional[str] = None,
    auto_detect_risc0: bool = True,
    tx_hash: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    spec_hash: Optional[str] = None,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> ProofBundle:
    """Generate a proof bundle via stub or external prover command.
    
    Args:
        manifest_path: Path to artifact manifest JSON
        stream_dir: Directory containing input stream files
        out_dir: Output directory for proof bundle
        prover_command: Optional explicit prover command template.
            If None and auto_detect_risc0=True, will auto-detect Risc0.
            If None and auto_detect_risc0=False, will use stub (SHA-256 only).
        auto_detect_risc0: If True and prover_command is None, automatically
            detect and use Risc0 if available. Default True.
    
    Returns:
        ProofBundle with paths to generated proof artifacts.
    """

    # Path safety validation
    _validate_path_safety(manifest_path)
    _validate_path_safety(stream_dir)
    _validate_path_safety(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    proof_path = out_dir / "proof.bin"
    receipt_path = out_dir / "receipt.json"

    digest = compute_artifact_digest(manifest_path, stream_dir, extra=extra_bindings)

    # Auto-detect Risc0 if no explicit command provided
    if prover_command is None and auto_detect_risc0:
        prover_command = _get_default_prover_command()
        # If Risc0 not available, prover_command will be None
        # and we'll fall back to stub proof generation

    if prover_command:
        # Security: Use shlex.split to prevent command injection
        # Format the command string first, then split safely
        cmd_str = prover_command.format(
            manifest=str(manifest_path),
            streams=str(stream_dir),
            proof=str(proof_path),
            receipt=str(receipt_path),
        )
        # Split command safely (handles quoted arguments, escapes, etc.)
        cmd_parts = shlex.split(cmd_str)
        # Security: Never use shell=True with user-controlled input
        subprocess.run(cmd_parts, check=True)
    else:
        proof_path.write_text(digest, encoding="utf-8")

    external_receipt = {}
    if prover_command and receipt_path.exists():
        external_receipt = json.loads(receipt_path.read_text())

    receipt = {
        "timestamp": time.time(),
        "manifest": str(manifest_path),
        "streams": str(stream_dir),
        "proof": str(proof_path),
        "digest": digest,
        "prover": "external" if prover_command else "stub",
    }
    if tx_hash is not None:
        receipt["tx_hash"] = tx_hash
    if config_fingerprint is not None:
        receipt["config_fingerprint"] = config_fingerprint
    if spec_hash is not None:
        receipt["spec_hash"] = spec_hash
    if external_receipt:
        receipt["prover"] = external_receipt.get("prover", receipt["prover"])
        receipt["method_id"] = external_receipt.get("method_id")
        receipt["prover_digest"] = external_receipt.get("digest_hex")
        receipt["prover_meta"] = external_receipt
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return ProofBundle(
        manifest_path=manifest_path,
        proof_path=proof_path,
        receipt_path=receipt_path,
        stream_dir=stream_dir,
    )


def verify_proof(
    bundle: ProofBundle,
    use_risc0: bool = False,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> bool:
    """Verify that the proof digest matches the manifest and stream directory.
    
    Args:
        bundle: Proof bundle to verify
        use_risc0: If True, perform Risc0 receipt verification instead of just digest matching
    
    Returns:
        True if proof is valid, False otherwise
    """
    try:
        # Security: Validate receipt size before parsing
        receipt_bytes = bundle.receipt_path.read_bytes()
        if len(receipt_bytes) > MAX_RECEIPT_SIZE_BYTES:
            return False

        receipt = json.loads(receipt_bytes.decode())

        manifest_path = bundle.manifest_path
        # Security: Validate manifest path safety
        _validate_path_safety(manifest_path)

        stream_dir = bundle.stream_dir or Path(receipt.get("streams", manifest_path.parent / "streams"))
        # Security: Validate stream directory path safety
        _validate_path_safety(stream_dir)

        if not stream_dir.exists():
            stream_dir = manifest_path.parent / "streams"
            # Validate the fallback path too
            _validate_path_safety(stream_dir)

        if not manifest_path.exists() or not stream_dir.exists():
            return False

        # If Risc0 proof, verify receipt and digest binding
        if use_risc0 or receipt.get("prover") == "risc0":
            return _verify_risc0_receipt(bundle.proof_path, manifest_path, stream_dir)

    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        # Invalid data, paths, or missing required fields
        return False

    digest = compute_artifact_digest(manifest_path, stream_dir, extra=extra_bindings)

    # Stub proofs include digest text; ensure the proof artifact matches
    if bundle.proof_path.exists():
        try:
            proof_bytes = bundle.proof_path.read_bytes().strip()
            if proof_bytes and proof_bytes.decode(errors="ignore") != digest:
                return False
        except OSError:
            return False

    return digest == receipt.get("digest") or digest == receipt.get("digest_hex")


def _verify_risc0_receipt(proof_path: Path, manifest_path: Path, stream_dir: Path) -> bool:
    """Verify a Risc0 receipt by calling the verifier binary with inputs bound."""
    import subprocess
    import shlex
    from pathlib import Path
    
    if not proof_path.exists() or not manifest_path.exists() or not stream_dir.exists():
        return False
    
    try:
        current_file = Path(__file__)
        risc0_host = current_file.parent / "risc0" / "host" / "target" / "release" / "idi_risc0_host"
        if not risc0_host.exists():
            # Try cargo run as fallback
            risc0_workspace = current_file.parent / "risc0"
            cmd = shlex.split(
                f"cargo run --release -p idi_risc0_host -- verify --proof {proof_path} --manifest {manifest_path} --streams {stream_dir}"
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
                [
                    str(risc0_host),
                    "verify",
                    "--proof",
                    str(proof_path),
                    "--manifest",
                    str(manifest_path),
                    "--streams",
                    str(stream_dir),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
