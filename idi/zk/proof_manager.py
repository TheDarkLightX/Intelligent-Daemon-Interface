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

# Security constants
MAX_RECEIPT_SIZE_BYTES = 512 * 1024  # 512KB limit for receipt files
PROVER_TIMEOUT_SECONDS = 300  # 5 minutes timeout for external prover


def _validate_path_safety(path: Path, base_dir: Optional[Path] = None) -> None:
    """Validate that a path is safe to access (no traversal, reasonable length).

    If base_dir is provided, the resolved path must remain within base_dir.
    """
    try:
        # Resolve to absolute path to check for traversal
        resolved = path.resolve()

        # Check for path traversal (should not go above base_dir if provided)
        if base_dir is not None:
            base_dir_resolved = base_dir.resolve()
            if not resolved.is_relative_to(base_dir_resolved):
                raise ValueError(f"Path traversal detected: {path} (base_dir={base_dir_resolved})")

        # Check path length (prevent extremely long paths)
        if len(str(resolved)) > 4096:  # 4KB limit for path length
            raise ValueError(f"Path too long: {path}")

        # Check for null bytes in path (potential security issue)
        if "\x00" in str(path):
            raise ValueError(f"Invalid null byte in path: {path}")

    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {path} - {e}")

# Default Risc0 command template (invokes prebuilt host binary directly).
# The actual binary path is prefixed in `_get_default_prover_command`.
_DEFAULT_RISC0_CMD = "prove --manifest {manifest} --streams {streams} --proof {proof} --receipt {receipt}"


def _get_default_prover_command() -> Optional[str]:
    """Get default prover command, auto-detecting Risc0 if available.
    
    Returns:
        Risc0 command template if available, None otherwise (will use stub).
    """
    # Require a prebuilt idi_risc0_host binary; do not invoke cargo directly.
    current_file = Path(__file__)
    risc0_host = current_file.parent / "risc0" / "target" / "release" / "idi_risc0_host"
    if not risc0_host.exists():
        return None
    return f"{risc0_host} {_DEFAULT_RISC0_CMD}"


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
    policy_root: Optional[bytes] = None,
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
    # Use caller-provided out_dir as the bundle directory; we do not
    # currently constrain manifest/streams to be inside it to preserve
    # existing workflows (tests expect absolute paths in the receipt).
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_path_safety(out_dir)

    proof_path = out_dir / "proof.bin"
    receipt_path = out_dir / "receipt.json"

    combined_extras = dict(extra_bindings or {})
    if policy_root:
        combined_extras.setdefault("policy_root", policy_root)
    if config_fingerprint:
        combined_extras.setdefault("config_fingerprint", config_fingerprint.encode())
    if spec_hash:
        combined_extras.setdefault("spec_hash", spec_hash.encode())

    digest = compute_artifact_digest(manifest_path, stream_dir, extra=combined_extras or None)

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
        # Security: Enforce timeout to prevent hangs from malicious/broken provers
        subprocess.run(cmd_parts, check=True, timeout=PROVER_TIMEOUT_SECONDS)
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
    if policy_root is not None:
        receipt["policy_root"] = policy_root.hex()
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


def verify_commitment(
    bundle: ProofBundle,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> bool:
    """Verify that the proof digest matches the manifest and stream directory.

    This only checks the commitment binding (manifest + streams + extra bindings)
    and, for stub proofs, that the proof artifact contains the digest. It does
    not perform zkVM receipt verification.

    Returns:
        True if commitment is valid and consistent with the receipt, False otherwise.
    """
    try:
        # Security: Validate receipt size before parsing
        receipt_bytes = bundle.receipt_path.read_bytes()
        if len(receipt_bytes) > MAX_RECEIPT_SIZE_BYTES:
            return False

        receipt = json.loads(receipt_bytes.decode())
        prover = str(receipt.get("prover", "stub"))

        manifest_path = bundle.manifest_path.resolve()
        _validate_path_safety(manifest_path)

        stream_dir = (bundle.stream_dir or Path(
            receipt.get("streams", manifest_path.parent / "streams")
        )).resolve()
        _validate_path_safety(stream_dir)

        if not manifest_path.exists() or not stream_dir.exists():
            return False

    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        # Invalid data, paths, or missing required fields
        return False

    derived_extras = dict(extra_bindings or {})

    # Pull bindings from receipt if present
    if receipt.get("policy_root") and "policy_root" not in derived_extras:
        try:
            derived_extras["policy_root"] = bytes.fromhex(receipt["policy_root"])
        except ValueError:
            return False
    if receipt.get("config_fingerprint") and "config_fingerprint" not in derived_extras:
        derived_extras["config_fingerprint"] = str(receipt["config_fingerprint"]).encode()
    if receipt.get("spec_hash") and "spec_hash" not in derived_extras:
        derived_extras["spec_hash"] = str(receipt["spec_hash"]).encode()

    digest = compute_artifact_digest(manifest_path, stream_dir, extra=derived_extras or None)

    # Stub proofs include digest text; ensure the proof artifact matches
    if prover == "stub" and bundle.proof_path.exists():
        try:
            proof_bytes = bundle.proof_path.read_bytes().strip()
            if proof_bytes and proof_bytes.decode(errors="ignore") != digest:
                return False
        except OSError:
            return False

    return digest == receipt.get("digest") or digest == receipt.get("digest_hex")


def verify_proof(
    bundle: ProofBundle,
    use_risc0: bool = False,
    extra_bindings: Optional[Dict[str, bytes]] = None,
) -> bool:
    """High-level verifier that combines commitment and optional zk verification.

    When use_risc0 is True or the receipt prover is "risc0", this will first
    verify the Risc0 receipt and then verify the manifest/streams commitment.
    For stub proofs, it only verifies the commitment.
    """
    try:
        receipt_bytes = bundle.receipt_path.read_bytes()
        if len(receipt_bytes) > MAX_RECEIPT_SIZE_BYTES:
            return False
        receipt = json.loads(receipt_bytes.decode())
    except (OSError, json.JSONDecodeError):
        return False

    # For Risc0 proofs, require both zk verification and commitment verification.
    if use_risc0 or receipt.get("prover") == "risc0":
        manifest_path = bundle.manifest_path
        stream_dir = bundle.stream_dir or Path(
            receipt.get("streams", manifest_path.parent / "streams")
        )
        if not _verify_risc0_receipt(bundle.proof_path, manifest_path, stream_dir):
            return False

    return verify_commitment(bundle, extra_bindings=extra_bindings)


def _verify_risc0_receipt(proof_path: Path, manifest_path: Path, stream_dir: Path) -> bool:
    """Verify a Risc0 receipt by calling the verifier binary with inputs bound.
    
    Security: This function requires a prebuilt verifier binary. It will NOT
    fall back to 'cargo run' to avoid triggering builds during verification,
    which could be slow, unpredictable, or a security risk if source is modified.
    
    To build the verifier, run:
        cd idi/zk/risc0 && cargo build --release -p idi_risc0_host
    """
    import subprocess
    import logging
    from pathlib import Path
    
    if not proof_path.exists() or not manifest_path.exists() or not stream_dir.exists():
        return False
    
    try:
        current_file = Path(__file__)
        risc0_host = current_file.parent / "risc0" / "target" / "release" / "idi_risc0_host"
        
        if not risc0_host.exists():
            # Do NOT fall back to cargo run - require prebuilt binary
            logging.warning(
                "Risc0 verifier binary not found at %s. "
                "Build it with: cd idi/zk/risc0 && cargo build --release -p idi_risc0_host",
                risc0_host
            )
            return False
        
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
        
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            logging.error("Risc0 verifier failed (code %s): %s", result.returncode, stderr)
            return False
        
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logging.error("Error invoking Risc0 verifier: %s", e)
        return False


def verify_zk_receipt(
    proof_path: Path,
    manifest_path: Path,
    stream_dir: Path,
    expected_method_id: str | None = None,
    expected_journal_digest: str | None = None,
    timeout_s: int = 30,
) -> "VerificationReport":
    """Verify a ZK receipt with method ID and journal digest enforcement.
    
    This is the secure verification function that enforces cryptographic
    binding of the proof to specific program logic (method ID) and data
    (journal digest).
    
    Args:
        proof_path: Path to proof binary file
        manifest_path: Path to manifest JSON file
        stream_dir: Directory containing stream files
        expected_method_id: Required method ID (hex) for Risc0 proofs
        expected_journal_digest: Optional expected journal digest (hex)
        timeout_s: Verification timeout in seconds
        
    Returns:
        VerificationReport with success/failure and error details
        
    Security:
        - Requires prebuilt verifier binary (no cargo run fallback)
        - Enforces method ID if provided (prevents program substitution)
        - Verifies journal digest matches commitment
    """
    import subprocess
    import logging
    from pathlib import Path
    from idi.zk.verification import VerificationReport, VerificationErrorCode
    
    # Check file existence
    if not proof_path.exists():
        return VerificationReport.fail(
            VerificationErrorCode.RECEIPT_MISSING,
            f"Proof file not found: {proof_path}",
        )
    if not manifest_path.exists():
        return VerificationReport.fail(
            VerificationErrorCode.MANIFEST_MISSING,
            f"Manifest file not found: {manifest_path}",
        )
    if not stream_dir.exists():
        return VerificationReport.fail(
            VerificationErrorCode.STREAMS_MISSING,
            f"Streams directory not found: {stream_dir}",
        )
    
    # Locate verifier binary
    current_file = Path(__file__)
    risc0_host = current_file.parent / "risc0" / "target" / "release" / "idi_risc0_host"
    
    if not risc0_host.exists():
        return VerificationReport.fail(
            VerificationErrorCode.VERIFIER_UNAVAILABLE,
            f"Risc0 verifier not found. Build with: cd idi/zk/risc0 && cargo build --release",
            binary_path=str(risc0_host),
        )
    
    # Build command
    cmd = [
        str(risc0_host),
        "verify",
        "--proof", str(proof_path),
        "--manifest", str(manifest_path),
        "--streams", str(stream_dir),
    ]
    
    if expected_method_id:
        cmd.extend(["--method-id", expected_method_id])
    
    if expected_journal_digest:
        cmd.extend(["--journal-digest", expected_journal_digest])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        
        if result.returncode == 0:
            return VerificationReport.ok(
                "ZK proof verified successfully",
                method_id=expected_method_id,
            )
        
        # Parse error from stderr
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        
        if "Method ID mismatch" in stderr:
            return VerificationReport.fail(
                VerificationErrorCode.METHOD_ID_MISMATCH,
                stderr,
                expected=expected_method_id,
            )
        elif "Journal digest mismatch" in stderr:
            return VerificationReport.fail(
                VerificationErrorCode.JOURNAL_DIGEST_MISMATCH,
                stderr,
                expected=expected_journal_digest,
            )
        elif "guest digest" in stderr and "does not match" in stderr:
            return VerificationReport.fail(
                VerificationErrorCode.COMMITMENT_MISMATCH,
                stderr,
            )
        else:
            return VerificationReport.fail(
                VerificationErrorCode.ZK_RECEIPT_INVALID,
                f"Verification failed: {stderr}",
                returncode=result.returncode,
            )
            
    except subprocess.TimeoutExpired:
        return VerificationReport.fail(
            VerificationErrorCode.VERIFIER_TIMEOUT,
            f"Verification timed out after {timeout_s}s",
        )
    except FileNotFoundError:
        return VerificationReport.fail(
            VerificationErrorCode.VERIFIER_UNAVAILABLE,
            f"Verifier binary not found: {risc0_host}",
        )
    except OSError as e:
        return VerificationReport.fail(
            VerificationErrorCode.INTERNAL_ERROR,
            f"OS error during verification: {e}",
        )


# =============================================================================
# Concurrent Verification Support
# =============================================================================

def verify_proof_single(bundle: ProofBundle, use_risc0: bool = False) -> bool:
    """Wrapper for single proof verification (for use with thread pools).
    
    Args:
        bundle: ProofBundle to verify
        use_risc0: Whether to use Risc0 ZK verification
        
    Returns:
        True if proof is valid, False otherwise
    """
    try:
        return verify_proof(bundle, use_risc0=use_risc0)
    except Exception:
        return False


def verify_proofs_parallel(
    bundles: list[ProofBundle],
    *,
    use_risc0: bool = False,
    max_workers: int | None = None,
) -> list[bool]:
    """Verify multiple proof bundles in parallel using a thread pool.
    
    This function distributes proof verification across multiple threads,
    allowing independent proofs to be verified concurrently. This is
    particularly useful for batch verification of transaction proofs.
    
    Args:
        bundles: List of ProofBundle instances to verify
        use_risc0: Whether to use Risc0 ZK verification for all proofs
        max_workers: Maximum number of worker threads (default: min(len(bundles), 4))
        
    Returns:
        List of boolean results in the same order as input bundles
        
    Example:
        bundles = [bundle1, bundle2, bundle3]
        results = verify_proofs_parallel(bundles, use_risc0=True)
        # results[i] corresponds to bundles[i]
        
    Performance:
        - Uses ThreadPoolExecutor for I/O-bound operations (file reads, subprocess)
        - Subprocess calls (Risc0 verifier) release the GIL, enabling true parallelism
        - For CPU-bound-only verification, consider ProcessPoolExecutor instead
        
    Thread Safety:
        - Each verification operates on independent files
        - No shared mutable state between verifications
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not bundles:
        return []
    
    # Default to 4 workers or number of bundles, whichever is smaller
    workers = max_workers if max_workers is not None else min(len(bundles), 4)
    
    # Pre-allocate results list to maintain order
    results: list[bool | None] = [None] * len(bundles)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all verification tasks with their index
        future_to_idx = {
            executor.submit(verify_proof_single, bundle, use_risc0): idx
            for idx, bundle in enumerate(bundles)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    
    # Convert None to False (shouldn't happen, but defensive)
    return [r if r is not None else False for r in results]


async def verify_proofs_async(
    bundles: list[ProofBundle],
    *,
    use_risc0: bool = False,
    max_workers: int | None = None,
) -> list[bool]:
    """Async version of parallel proof verification.
    
    Wraps the thread pool execution in an async interface for use
    in async/await codebases (e.g., FastAPI, aiohttp servers).
    
    Args:
        bundles: List of ProofBundle instances to verify
        use_risc0: Whether to use Risc0 ZK verification
        max_workers: Maximum number of worker threads
        
    Returns:
        List of boolean results in the same order as input bundles
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: verify_proofs_parallel(bundles, use_risc0=use_risc0, max_workers=max_workers)
    )
