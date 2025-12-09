"""High-level bundle verification functions.

This module provides unified verification for both local and wire bundles,
combining commitment verification, method ID enforcement, and ZK proof
validation into a single interface.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from idi.zk.commitment import compute_commitment_bytes
from idi.zk.verification import VerificationErrorCode, VerificationReport, validate_path_safety

if TYPE_CHECKING:
    from idi.zk.wire import ZkProofBundleLocal, ZkProofBundleWireV1


def verify_proof_bundle_local(
    bundle: "ZkProofBundleLocal",
    expected_method_id: str | None = None,
    require_zk: bool = True,
    max_proof_bytes: int = 5 * 1024 * 1024,
    max_attestation_bytes: int = 512 * 1024,
) -> VerificationReport:
    """Verify a local proof bundle completely.
    
    Performs:
    1. File existence and size checks
    2. Path safety validation
    3. Commitment verification (digest matches manifest + streams)
    4. ZK receipt verification (if require_zk=True)
    5. Method ID enforcement (if provided)
    
    Args:
        bundle: Local proof bundle with file paths
        expected_method_id: Required method ID for Risc0 proofs
        require_zk: If True, requires actual ZK verification; if False, skip
        max_proof_bytes: Maximum allowed proof size
        max_attestation_bytes: Maximum allowed attestation size
        
    Returns:
        VerificationReport with success/failure and detailed error info
    """
    # Check file existence
    if not bundle.proof_path.exists():
        return VerificationReport.fail(
            VerificationErrorCode.RECEIPT_MISSING,
            f"Proof file not found: {bundle.proof_path}",
        )
    
    if not bundle.attestation_path.exists():
        return VerificationReport.fail(
            VerificationErrorCode.RECEIPT_MISSING,
            f"Attestation file not found: {bundle.attestation_path}",
        )
    
    if not bundle.manifest_path.exists():
        return VerificationReport.fail(
            VerificationErrorCode.MANIFEST_MISSING,
            f"Manifest file not found: {bundle.manifest_path}",
        )
    
    # Size checks
    if bundle.proof_path.stat().st_size > max_proof_bytes:
        return VerificationReport.fail(
            VerificationErrorCode.SIZE_LIMIT_EXCEEDED,
            f"Proof exceeds size limit: {bundle.proof_path.stat().st_size} > {max_proof_bytes}",
        )
    
    if bundle.attestation_path.stat().st_size > max_attestation_bytes:
        return VerificationReport.fail(
            VerificationErrorCode.SIZE_LIMIT_EXCEEDED,
            f"Attestation exceeds size limit",
        )
    
    # Parse attestation
    try:
        attestation = json.loads(bundle.attestation_path.read_bytes())
    except json.JSONDecodeError as e:
        return VerificationReport.fail(
            VerificationErrorCode.RECEIPT_PARSE_ERROR,
            f"Failed to parse attestation JSON: {e}",
        )
    
    # Determine proof system
    proof_system = attestation.get("prover", "stub")
    
    # Compute expected commitment using the same algorithm as proof generation
    # (uses compute_artifact_digest which has a different format than commitment.py)
    from idi.zk.proof_manager import compute_artifact_digest
    
    stream_dir = bundle.stream_dir
    if stream_dir and stream_dir.exists():
        computed_digest = compute_artifact_digest(bundle.manifest_path, stream_dir)
    else:
        computed_digest = compute_artifact_digest(bundle.manifest_path, bundle.manifest_path.parent / "streams")
    
    # Check commitment in attestation
    attestation_digest = attestation.get("digest_hex") or attestation.get("digest")
    if attestation_digest and attestation_digest != computed_digest:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment digest mismatch",
            expected=computed_digest,
            actual=attestation_digest,
        )
    
    # Check tx_hash binding if present
    if bundle.tx_hash:
        receipt_tx_hash = attestation.get("tx_hash")
        if receipt_tx_hash and receipt_tx_hash != bundle.tx_hash:
            return VerificationReport.fail(
                VerificationErrorCode.TX_HASH_MISMATCH,
                "Transaction hash mismatch",
                expected=bundle.tx_hash,
                actual=receipt_tx_hash,
            )
    
    # ZK verification (if required)
    if require_zk and proof_system == "risc0":
        from idi.zk.proof_manager import verify_zk_receipt
        
        stream_dir_path = stream_dir if stream_dir else bundle.manifest_path.parent / "streams"
        
        zk_report = verify_zk_receipt(
            proof_path=bundle.proof_path,
            manifest_path=bundle.manifest_path,
            stream_dir=stream_dir_path,
            expected_method_id=expected_method_id,
            expected_journal_digest=computed_digest,
        )
        
        if not zk_report.success:
            return zk_report
    
    return VerificationReport.ok(
        "Bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def verify_proof_bundle_wire(
    wire: "ZkProofBundleWireV1",
    expected_method_id: str | None = None,
    require_zk: bool = True,
) -> VerificationReport:
    """Verify a wire bundle without persistent filesystem access.
    
    Materializes the wire bundle to a temporary directory, then
    delegates to verify_proof_bundle_local.
    
    Args:
        wire: Network-portable wire bundle
        expected_method_id: Required method ID for Risc0 proofs
        require_zk: If True, requires actual ZK verification
        
    Returns:
        VerificationReport with success/failure
    """
    # First check wire bundle integrity (hash check)
    integrity_report = wire.verify_integrity()
    if not integrity_report.success:
        return integrity_report
    
    # Parse attestation
    try:
        attestation = wire.get_attestation()
    except (json.JSONDecodeError, Exception) as e:
        return VerificationReport.fail(
            VerificationErrorCode.RECEIPT_PARSE_ERROR,
            f"Failed to parse attestation: {e}",
        )
    
    # Materialize to temp to compute commitment with same algorithm as proof generation
    with tempfile.TemporaryDirectory() as tmpdir:
        local = wire.to_local(Path(tmpdir))
        
        from idi.zk.proof_manager import compute_artifact_digest
        
        computed_digest = compute_artifact_digest(local.manifest_path, local.stream_dir)
        
        # Check commitment against attestation
        attestation_digest = attestation.get("digest_hex") or attestation.get("digest")
        if attestation_digest and attestation_digest != computed_digest:
            return VerificationReport.fail(
                VerificationErrorCode.COMMITMENT_MISMATCH,
                "Wire bundle commitment mismatch",
                expected=computed_digest,
                actual=attestation_digest,
            )
        
        # For ZK verification
        if require_zk and wire.proof_system == "risc0":
            return verify_proof_bundle_local(
                local,
                expected_method_id=expected_method_id,
                require_zk=True,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def verify_commitment_only(
    manifest_bytes: bytes,
    streams: dict[str, bytes],
    expected_digest: str,
) -> VerificationReport:
    """Verify commitment matches without full proof verification.
    
    Useful for quick validation before expensive ZK verification.
    
    Args:
        manifest_bytes: Raw manifest content
        streams: Stream name -> content mapping
        expected_digest: Expected commitment digest (hex)
        
    Returns:
        VerificationReport
    """
    computed = compute_commitment_bytes(manifest_bytes, streams)
    
    if computed == expected_digest:
        return VerificationReport.ok(
            "Commitment verified",
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )
