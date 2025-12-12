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


def x_verify_proof_bundle_local__mutmut_orig(
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


def x_verify_proof_bundle_local__mutmut_1(
    bundle: "ZkProofBundleLocal",
    expected_method_id: str | None = None,
    require_zk: bool = False,
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


def x_verify_proof_bundle_local__mutmut_2(
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
    if bundle.proof_path.exists():
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


def x_verify_proof_bundle_local__mutmut_3(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_4(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_5(
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


def x_verify_proof_bundle_local__mutmut_6(
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


def x_verify_proof_bundle_local__mutmut_7(
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
    
    if bundle.attestation_path.exists():
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


def x_verify_proof_bundle_local__mutmut_8(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_9(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_10(
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


def x_verify_proof_bundle_local__mutmut_11(
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


def x_verify_proof_bundle_local__mutmut_12(
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
    
    if bundle.manifest_path.exists():
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


def x_verify_proof_bundle_local__mutmut_13(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_14(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_15(
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


def x_verify_proof_bundle_local__mutmut_16(
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


def x_verify_proof_bundle_local__mutmut_17(
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
    if bundle.proof_path.stat().st_size >= max_proof_bytes:
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


def x_verify_proof_bundle_local__mutmut_18(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_19(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_20(
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


def x_verify_proof_bundle_local__mutmut_21(
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


def x_verify_proof_bundle_local__mutmut_22(
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
    
    if bundle.attestation_path.stat().st_size >= max_attestation_bytes:
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


def x_verify_proof_bundle_local__mutmut_23(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_24(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_25(
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


def x_verify_proof_bundle_local__mutmut_26(
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


def x_verify_proof_bundle_local__mutmut_27(
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
        attestation = None
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


def x_verify_proof_bundle_local__mutmut_28(
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
        attestation = json.loads(None)
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


def x_verify_proof_bundle_local__mutmut_29(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_30(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_31(
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


def x_verify_proof_bundle_local__mutmut_32(
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


def x_verify_proof_bundle_local__mutmut_33(
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
    proof_system = None
    
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


def x_verify_proof_bundle_local__mutmut_34(
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
    proof_system = attestation.get(None, "stub")
    
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


def x_verify_proof_bundle_local__mutmut_35(
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
    proof_system = attestation.get("prover", None)
    
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


def x_verify_proof_bundle_local__mutmut_36(
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
    proof_system = attestation.get("stub")
    
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


def x_verify_proof_bundle_local__mutmut_37(
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
    proof_system = attestation.get("prover", )
    
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


def x_verify_proof_bundle_local__mutmut_38(
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
    proof_system = attestation.get("XXproverXX", "stub")
    
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


def x_verify_proof_bundle_local__mutmut_39(
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
    proof_system = attestation.get("PROVER", "stub")
    
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


def x_verify_proof_bundle_local__mutmut_40(
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
    proof_system = attestation.get("prover", "XXstubXX")
    
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


def x_verify_proof_bundle_local__mutmut_41(
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
    proof_system = attestation.get("prover", "STUB")
    
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


def x_verify_proof_bundle_local__mutmut_42(
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
    
    stream_dir = None
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


def x_verify_proof_bundle_local__mutmut_43(
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
    if stream_dir or stream_dir.exists():
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


def x_verify_proof_bundle_local__mutmut_44(
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
        computed_digest = None
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


def x_verify_proof_bundle_local__mutmut_45(
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
        computed_digest = compute_artifact_digest(None, stream_dir)
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


def x_verify_proof_bundle_local__mutmut_46(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, None)
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


def x_verify_proof_bundle_local__mutmut_47(
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
        computed_digest = compute_artifact_digest(stream_dir)
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


def x_verify_proof_bundle_local__mutmut_48(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, )
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


def x_verify_proof_bundle_local__mutmut_49(
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
        computed_digest = None
    
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


def x_verify_proof_bundle_local__mutmut_50(
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
        computed_digest = compute_artifact_digest(None, bundle.manifest_path.parent / "streams")
    
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


def x_verify_proof_bundle_local__mutmut_51(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, None)
    
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


def x_verify_proof_bundle_local__mutmut_52(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path.parent / "streams")
    
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


def x_verify_proof_bundle_local__mutmut_53(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, )
    
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


def x_verify_proof_bundle_local__mutmut_54(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, bundle.manifest_path.parent * "streams")
    
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


def x_verify_proof_bundle_local__mutmut_55(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, bundle.manifest_path.parent / "XXstreamsXX")
    
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


def x_verify_proof_bundle_local__mutmut_56(
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
        computed_digest = compute_artifact_digest(bundle.manifest_path, bundle.manifest_path.parent / "STREAMS")
    
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


def x_verify_proof_bundle_local__mutmut_57(
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
    attestation_digest = None
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


def x_verify_proof_bundle_local__mutmut_58(
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
    attestation_digest = attestation.get("digest_hex") and attestation.get("digest")
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


def x_verify_proof_bundle_local__mutmut_59(
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
    attestation_digest = attestation.get(None) or attestation.get("digest")
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


def x_verify_proof_bundle_local__mutmut_60(
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
    attestation_digest = attestation.get("XXdigest_hexXX") or attestation.get("digest")
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


def x_verify_proof_bundle_local__mutmut_61(
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
    attestation_digest = attestation.get("DIGEST_HEX") or attestation.get("digest")
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


def x_verify_proof_bundle_local__mutmut_62(
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
    attestation_digest = attestation.get("digest_hex") or attestation.get(None)
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


def x_verify_proof_bundle_local__mutmut_63(
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
    attestation_digest = attestation.get("digest_hex") or attestation.get("XXdigestXX")
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


def x_verify_proof_bundle_local__mutmut_64(
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
    attestation_digest = attestation.get("digest_hex") or attestation.get("DIGEST")
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


def x_verify_proof_bundle_local__mutmut_65(
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
    if attestation_digest or attestation_digest != computed_digest:
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


def x_verify_proof_bundle_local__mutmut_66(
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
    if attestation_digest and attestation_digest == computed_digest:
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


def x_verify_proof_bundle_local__mutmut_67(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_68(
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
            None,
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


def x_verify_proof_bundle_local__mutmut_69(
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
            expected=None,
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


def x_verify_proof_bundle_local__mutmut_70(
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
            actual=None,
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


def x_verify_proof_bundle_local__mutmut_71(
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


def x_verify_proof_bundle_local__mutmut_72(
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


def x_verify_proof_bundle_local__mutmut_73(
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


def x_verify_proof_bundle_local__mutmut_74(
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


def x_verify_proof_bundle_local__mutmut_75(
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
            "XXCommitment digest mismatchXX",
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


def x_verify_proof_bundle_local__mutmut_76(
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
            "commitment digest mismatch",
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


def x_verify_proof_bundle_local__mutmut_77(
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
            "COMMITMENT DIGEST MISMATCH",
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


def x_verify_proof_bundle_local__mutmut_78(
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
        receipt_tx_hash = None
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


def x_verify_proof_bundle_local__mutmut_79(
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
        receipt_tx_hash = attestation.get(None)
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


def x_verify_proof_bundle_local__mutmut_80(
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
        receipt_tx_hash = attestation.get("XXtx_hashXX")
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


def x_verify_proof_bundle_local__mutmut_81(
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
        receipt_tx_hash = attestation.get("TX_HASH")
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


def x_verify_proof_bundle_local__mutmut_82(
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
        if receipt_tx_hash or receipt_tx_hash != bundle.tx_hash:
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


def x_verify_proof_bundle_local__mutmut_83(
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
        if receipt_tx_hash and receipt_tx_hash == bundle.tx_hash:
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


def x_verify_proof_bundle_local__mutmut_84(
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
                None,
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


def x_verify_proof_bundle_local__mutmut_85(
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
                None,
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


def x_verify_proof_bundle_local__mutmut_86(
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
                expected=None,
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


def x_verify_proof_bundle_local__mutmut_87(
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
                actual=None,
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


def x_verify_proof_bundle_local__mutmut_88(
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


def x_verify_proof_bundle_local__mutmut_89(
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


def x_verify_proof_bundle_local__mutmut_90(
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


def x_verify_proof_bundle_local__mutmut_91(
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


def x_verify_proof_bundle_local__mutmut_92(
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
                "XXTransaction hash mismatchXX",
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


def x_verify_proof_bundle_local__mutmut_93(
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
                "transaction hash mismatch",
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


def x_verify_proof_bundle_local__mutmut_94(
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
                "TRANSACTION HASH MISMATCH",
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


def x_verify_proof_bundle_local__mutmut_95(
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
    if require_zk or proof_system == "risc0":
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


def x_verify_proof_bundle_local__mutmut_96(
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
    if require_zk and proof_system != "risc0":
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


def x_verify_proof_bundle_local__mutmut_97(
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
    if require_zk and proof_system == "XXrisc0XX":
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


def x_verify_proof_bundle_local__mutmut_98(
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
    if require_zk and proof_system == "RISC0":
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


def x_verify_proof_bundle_local__mutmut_99(
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
        
        stream_dir_path = None
        
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


def x_verify_proof_bundle_local__mutmut_100(
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
        
        stream_dir_path = stream_dir if stream_dir else bundle.manifest_path.parent * "streams"
        
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


def x_verify_proof_bundle_local__mutmut_101(
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
        
        stream_dir_path = stream_dir if stream_dir else bundle.manifest_path.parent / "XXstreamsXX"
        
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


def x_verify_proof_bundle_local__mutmut_102(
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
        
        stream_dir_path = stream_dir if stream_dir else bundle.manifest_path.parent / "STREAMS"
        
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


def x_verify_proof_bundle_local__mutmut_103(
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
        
        zk_report = None
        
        if not zk_report.success:
            return zk_report
    
    return VerificationReport.ok(
        "Bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_104(
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
            proof_path=None,
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


def x_verify_proof_bundle_local__mutmut_105(
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
            manifest_path=None,
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


def x_verify_proof_bundle_local__mutmut_106(
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
            stream_dir=None,
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


def x_verify_proof_bundle_local__mutmut_107(
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
            expected_method_id=None,
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


def x_verify_proof_bundle_local__mutmut_108(
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
            expected_journal_digest=None,
        )
        
        if not zk_report.success:
            return zk_report
    
    return VerificationReport.ok(
        "Bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_109(
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


def x_verify_proof_bundle_local__mutmut_110(
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


def x_verify_proof_bundle_local__mutmut_111(
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


def x_verify_proof_bundle_local__mutmut_112(
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


def x_verify_proof_bundle_local__mutmut_113(
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
            )
        
        if not zk_report.success:
            return zk_report
    
    return VerificationReport.ok(
        "Bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_114(
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
        
        if zk_report.success:
            return zk_report
    
    return VerificationReport.ok(
        "Bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_115(
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
        None,
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_116(
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
        proof_system=None,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_117(
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
        digest=None,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_118(
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
        method_id=None,
    )


def x_verify_proof_bundle_local__mutmut_119(
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
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_120(
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
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_121(
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
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_122(
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
        )


def x_verify_proof_bundle_local__mutmut_123(
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
        "XXBundle verified successfullyXX",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_124(
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
        "bundle verified successfully",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )


def x_verify_proof_bundle_local__mutmut_125(
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
        "BUNDLE VERIFIED SUCCESSFULLY",
        proof_system=proof_system,
        digest=computed_digest,
        method_id=expected_method_id,
    )

x_verify_proof_bundle_local__mutmut_mutants : ClassVar[MutantDict] = {
'x_verify_proof_bundle_local__mutmut_1': x_verify_proof_bundle_local__mutmut_1, 
    'x_verify_proof_bundle_local__mutmut_2': x_verify_proof_bundle_local__mutmut_2, 
    'x_verify_proof_bundle_local__mutmut_3': x_verify_proof_bundle_local__mutmut_3, 
    'x_verify_proof_bundle_local__mutmut_4': x_verify_proof_bundle_local__mutmut_4, 
    'x_verify_proof_bundle_local__mutmut_5': x_verify_proof_bundle_local__mutmut_5, 
    'x_verify_proof_bundle_local__mutmut_6': x_verify_proof_bundle_local__mutmut_6, 
    'x_verify_proof_bundle_local__mutmut_7': x_verify_proof_bundle_local__mutmut_7, 
    'x_verify_proof_bundle_local__mutmut_8': x_verify_proof_bundle_local__mutmut_8, 
    'x_verify_proof_bundle_local__mutmut_9': x_verify_proof_bundle_local__mutmut_9, 
    'x_verify_proof_bundle_local__mutmut_10': x_verify_proof_bundle_local__mutmut_10, 
    'x_verify_proof_bundle_local__mutmut_11': x_verify_proof_bundle_local__mutmut_11, 
    'x_verify_proof_bundle_local__mutmut_12': x_verify_proof_bundle_local__mutmut_12, 
    'x_verify_proof_bundle_local__mutmut_13': x_verify_proof_bundle_local__mutmut_13, 
    'x_verify_proof_bundle_local__mutmut_14': x_verify_proof_bundle_local__mutmut_14, 
    'x_verify_proof_bundle_local__mutmut_15': x_verify_proof_bundle_local__mutmut_15, 
    'x_verify_proof_bundle_local__mutmut_16': x_verify_proof_bundle_local__mutmut_16, 
    'x_verify_proof_bundle_local__mutmut_17': x_verify_proof_bundle_local__mutmut_17, 
    'x_verify_proof_bundle_local__mutmut_18': x_verify_proof_bundle_local__mutmut_18, 
    'x_verify_proof_bundle_local__mutmut_19': x_verify_proof_bundle_local__mutmut_19, 
    'x_verify_proof_bundle_local__mutmut_20': x_verify_proof_bundle_local__mutmut_20, 
    'x_verify_proof_bundle_local__mutmut_21': x_verify_proof_bundle_local__mutmut_21, 
    'x_verify_proof_bundle_local__mutmut_22': x_verify_proof_bundle_local__mutmut_22, 
    'x_verify_proof_bundle_local__mutmut_23': x_verify_proof_bundle_local__mutmut_23, 
    'x_verify_proof_bundle_local__mutmut_24': x_verify_proof_bundle_local__mutmut_24, 
    'x_verify_proof_bundle_local__mutmut_25': x_verify_proof_bundle_local__mutmut_25, 
    'x_verify_proof_bundle_local__mutmut_26': x_verify_proof_bundle_local__mutmut_26, 
    'x_verify_proof_bundle_local__mutmut_27': x_verify_proof_bundle_local__mutmut_27, 
    'x_verify_proof_bundle_local__mutmut_28': x_verify_proof_bundle_local__mutmut_28, 
    'x_verify_proof_bundle_local__mutmut_29': x_verify_proof_bundle_local__mutmut_29, 
    'x_verify_proof_bundle_local__mutmut_30': x_verify_proof_bundle_local__mutmut_30, 
    'x_verify_proof_bundle_local__mutmut_31': x_verify_proof_bundle_local__mutmut_31, 
    'x_verify_proof_bundle_local__mutmut_32': x_verify_proof_bundle_local__mutmut_32, 
    'x_verify_proof_bundle_local__mutmut_33': x_verify_proof_bundle_local__mutmut_33, 
    'x_verify_proof_bundle_local__mutmut_34': x_verify_proof_bundle_local__mutmut_34, 
    'x_verify_proof_bundle_local__mutmut_35': x_verify_proof_bundle_local__mutmut_35, 
    'x_verify_proof_bundle_local__mutmut_36': x_verify_proof_bundle_local__mutmut_36, 
    'x_verify_proof_bundle_local__mutmut_37': x_verify_proof_bundle_local__mutmut_37, 
    'x_verify_proof_bundle_local__mutmut_38': x_verify_proof_bundle_local__mutmut_38, 
    'x_verify_proof_bundle_local__mutmut_39': x_verify_proof_bundle_local__mutmut_39, 
    'x_verify_proof_bundle_local__mutmut_40': x_verify_proof_bundle_local__mutmut_40, 
    'x_verify_proof_bundle_local__mutmut_41': x_verify_proof_bundle_local__mutmut_41, 
    'x_verify_proof_bundle_local__mutmut_42': x_verify_proof_bundle_local__mutmut_42, 
    'x_verify_proof_bundle_local__mutmut_43': x_verify_proof_bundle_local__mutmut_43, 
    'x_verify_proof_bundle_local__mutmut_44': x_verify_proof_bundle_local__mutmut_44, 
    'x_verify_proof_bundle_local__mutmut_45': x_verify_proof_bundle_local__mutmut_45, 
    'x_verify_proof_bundle_local__mutmut_46': x_verify_proof_bundle_local__mutmut_46, 
    'x_verify_proof_bundle_local__mutmut_47': x_verify_proof_bundle_local__mutmut_47, 
    'x_verify_proof_bundle_local__mutmut_48': x_verify_proof_bundle_local__mutmut_48, 
    'x_verify_proof_bundle_local__mutmut_49': x_verify_proof_bundle_local__mutmut_49, 
    'x_verify_proof_bundle_local__mutmut_50': x_verify_proof_bundle_local__mutmut_50, 
    'x_verify_proof_bundle_local__mutmut_51': x_verify_proof_bundle_local__mutmut_51, 
    'x_verify_proof_bundle_local__mutmut_52': x_verify_proof_bundle_local__mutmut_52, 
    'x_verify_proof_bundle_local__mutmut_53': x_verify_proof_bundle_local__mutmut_53, 
    'x_verify_proof_bundle_local__mutmut_54': x_verify_proof_bundle_local__mutmut_54, 
    'x_verify_proof_bundle_local__mutmut_55': x_verify_proof_bundle_local__mutmut_55, 
    'x_verify_proof_bundle_local__mutmut_56': x_verify_proof_bundle_local__mutmut_56, 
    'x_verify_proof_bundle_local__mutmut_57': x_verify_proof_bundle_local__mutmut_57, 
    'x_verify_proof_bundle_local__mutmut_58': x_verify_proof_bundle_local__mutmut_58, 
    'x_verify_proof_bundle_local__mutmut_59': x_verify_proof_bundle_local__mutmut_59, 
    'x_verify_proof_bundle_local__mutmut_60': x_verify_proof_bundle_local__mutmut_60, 
    'x_verify_proof_bundle_local__mutmut_61': x_verify_proof_bundle_local__mutmut_61, 
    'x_verify_proof_bundle_local__mutmut_62': x_verify_proof_bundle_local__mutmut_62, 
    'x_verify_proof_bundle_local__mutmut_63': x_verify_proof_bundle_local__mutmut_63, 
    'x_verify_proof_bundle_local__mutmut_64': x_verify_proof_bundle_local__mutmut_64, 
    'x_verify_proof_bundle_local__mutmut_65': x_verify_proof_bundle_local__mutmut_65, 
    'x_verify_proof_bundle_local__mutmut_66': x_verify_proof_bundle_local__mutmut_66, 
    'x_verify_proof_bundle_local__mutmut_67': x_verify_proof_bundle_local__mutmut_67, 
    'x_verify_proof_bundle_local__mutmut_68': x_verify_proof_bundle_local__mutmut_68, 
    'x_verify_proof_bundle_local__mutmut_69': x_verify_proof_bundle_local__mutmut_69, 
    'x_verify_proof_bundle_local__mutmut_70': x_verify_proof_bundle_local__mutmut_70, 
    'x_verify_proof_bundle_local__mutmut_71': x_verify_proof_bundle_local__mutmut_71, 
    'x_verify_proof_bundle_local__mutmut_72': x_verify_proof_bundle_local__mutmut_72, 
    'x_verify_proof_bundle_local__mutmut_73': x_verify_proof_bundle_local__mutmut_73, 
    'x_verify_proof_bundle_local__mutmut_74': x_verify_proof_bundle_local__mutmut_74, 
    'x_verify_proof_bundle_local__mutmut_75': x_verify_proof_bundle_local__mutmut_75, 
    'x_verify_proof_bundle_local__mutmut_76': x_verify_proof_bundle_local__mutmut_76, 
    'x_verify_proof_bundle_local__mutmut_77': x_verify_proof_bundle_local__mutmut_77, 
    'x_verify_proof_bundle_local__mutmut_78': x_verify_proof_bundle_local__mutmut_78, 
    'x_verify_proof_bundle_local__mutmut_79': x_verify_proof_bundle_local__mutmut_79, 
    'x_verify_proof_bundle_local__mutmut_80': x_verify_proof_bundle_local__mutmut_80, 
    'x_verify_proof_bundle_local__mutmut_81': x_verify_proof_bundle_local__mutmut_81, 
    'x_verify_proof_bundle_local__mutmut_82': x_verify_proof_bundle_local__mutmut_82, 
    'x_verify_proof_bundle_local__mutmut_83': x_verify_proof_bundle_local__mutmut_83, 
    'x_verify_proof_bundle_local__mutmut_84': x_verify_proof_bundle_local__mutmut_84, 
    'x_verify_proof_bundle_local__mutmut_85': x_verify_proof_bundle_local__mutmut_85, 
    'x_verify_proof_bundle_local__mutmut_86': x_verify_proof_bundle_local__mutmut_86, 
    'x_verify_proof_bundle_local__mutmut_87': x_verify_proof_bundle_local__mutmut_87, 
    'x_verify_proof_bundle_local__mutmut_88': x_verify_proof_bundle_local__mutmut_88, 
    'x_verify_proof_bundle_local__mutmut_89': x_verify_proof_bundle_local__mutmut_89, 
    'x_verify_proof_bundle_local__mutmut_90': x_verify_proof_bundle_local__mutmut_90, 
    'x_verify_proof_bundle_local__mutmut_91': x_verify_proof_bundle_local__mutmut_91, 
    'x_verify_proof_bundle_local__mutmut_92': x_verify_proof_bundle_local__mutmut_92, 
    'x_verify_proof_bundle_local__mutmut_93': x_verify_proof_bundle_local__mutmut_93, 
    'x_verify_proof_bundle_local__mutmut_94': x_verify_proof_bundle_local__mutmut_94, 
    'x_verify_proof_bundle_local__mutmut_95': x_verify_proof_bundle_local__mutmut_95, 
    'x_verify_proof_bundle_local__mutmut_96': x_verify_proof_bundle_local__mutmut_96, 
    'x_verify_proof_bundle_local__mutmut_97': x_verify_proof_bundle_local__mutmut_97, 
    'x_verify_proof_bundle_local__mutmut_98': x_verify_proof_bundle_local__mutmut_98, 
    'x_verify_proof_bundle_local__mutmut_99': x_verify_proof_bundle_local__mutmut_99, 
    'x_verify_proof_bundle_local__mutmut_100': x_verify_proof_bundle_local__mutmut_100, 
    'x_verify_proof_bundle_local__mutmut_101': x_verify_proof_bundle_local__mutmut_101, 
    'x_verify_proof_bundle_local__mutmut_102': x_verify_proof_bundle_local__mutmut_102, 
    'x_verify_proof_bundle_local__mutmut_103': x_verify_proof_bundle_local__mutmut_103, 
    'x_verify_proof_bundle_local__mutmut_104': x_verify_proof_bundle_local__mutmut_104, 
    'x_verify_proof_bundle_local__mutmut_105': x_verify_proof_bundle_local__mutmut_105, 
    'x_verify_proof_bundle_local__mutmut_106': x_verify_proof_bundle_local__mutmut_106, 
    'x_verify_proof_bundle_local__mutmut_107': x_verify_proof_bundle_local__mutmut_107, 
    'x_verify_proof_bundle_local__mutmut_108': x_verify_proof_bundle_local__mutmut_108, 
    'x_verify_proof_bundle_local__mutmut_109': x_verify_proof_bundle_local__mutmut_109, 
    'x_verify_proof_bundle_local__mutmut_110': x_verify_proof_bundle_local__mutmut_110, 
    'x_verify_proof_bundle_local__mutmut_111': x_verify_proof_bundle_local__mutmut_111, 
    'x_verify_proof_bundle_local__mutmut_112': x_verify_proof_bundle_local__mutmut_112, 
    'x_verify_proof_bundle_local__mutmut_113': x_verify_proof_bundle_local__mutmut_113, 
    'x_verify_proof_bundle_local__mutmut_114': x_verify_proof_bundle_local__mutmut_114, 
    'x_verify_proof_bundle_local__mutmut_115': x_verify_proof_bundle_local__mutmut_115, 
    'x_verify_proof_bundle_local__mutmut_116': x_verify_proof_bundle_local__mutmut_116, 
    'x_verify_proof_bundle_local__mutmut_117': x_verify_proof_bundle_local__mutmut_117, 
    'x_verify_proof_bundle_local__mutmut_118': x_verify_proof_bundle_local__mutmut_118, 
    'x_verify_proof_bundle_local__mutmut_119': x_verify_proof_bundle_local__mutmut_119, 
    'x_verify_proof_bundle_local__mutmut_120': x_verify_proof_bundle_local__mutmut_120, 
    'x_verify_proof_bundle_local__mutmut_121': x_verify_proof_bundle_local__mutmut_121, 
    'x_verify_proof_bundle_local__mutmut_122': x_verify_proof_bundle_local__mutmut_122, 
    'x_verify_proof_bundle_local__mutmut_123': x_verify_proof_bundle_local__mutmut_123, 
    'x_verify_proof_bundle_local__mutmut_124': x_verify_proof_bundle_local__mutmut_124, 
    'x_verify_proof_bundle_local__mutmut_125': x_verify_proof_bundle_local__mutmut_125
}

def verify_proof_bundle_local(*args, **kwargs):
    result = _mutmut_trampoline(x_verify_proof_bundle_local__mutmut_orig, x_verify_proof_bundle_local__mutmut_mutants, args, kwargs)
    return result 

verify_proof_bundle_local.__signature__ = _mutmut_signature(x_verify_proof_bundle_local__mutmut_orig)
x_verify_proof_bundle_local__mutmut_orig.__name__ = 'x_verify_proof_bundle_local'


def x_verify_proof_bundle_wire__mutmut_orig(
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


def x_verify_proof_bundle_wire__mutmut_1(
    wire: "ZkProofBundleWireV1",
    expected_method_id: str | None = None,
    require_zk: bool = False,
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


def x_verify_proof_bundle_wire__mutmut_2(
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
    integrity_report = None
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


def x_verify_proof_bundle_wire__mutmut_3(
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
    if integrity_report.success:
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


def x_verify_proof_bundle_wire__mutmut_4(
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
        attestation = None
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


def x_verify_proof_bundle_wire__mutmut_5(
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
            None,
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


def x_verify_proof_bundle_wire__mutmut_6(
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
            None,
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


def x_verify_proof_bundle_wire__mutmut_7(
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


def x_verify_proof_bundle_wire__mutmut_8(
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


def x_verify_proof_bundle_wire__mutmut_9(
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
        local = None
        
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


def x_verify_proof_bundle_wire__mutmut_10(
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
        local = wire.to_local(None)
        
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


def x_verify_proof_bundle_wire__mutmut_11(
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
        local = wire.to_local(Path(None))
        
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


def x_verify_proof_bundle_wire__mutmut_12(
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
        
        computed_digest = None
        
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


def x_verify_proof_bundle_wire__mutmut_13(
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
        
        computed_digest = compute_artifact_digest(None, local.stream_dir)
        
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


def x_verify_proof_bundle_wire__mutmut_14(
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
        
        computed_digest = compute_artifact_digest(local.manifest_path, None)
        
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


def x_verify_proof_bundle_wire__mutmut_15(
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
        
        computed_digest = compute_artifact_digest(local.stream_dir)
        
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


def x_verify_proof_bundle_wire__mutmut_16(
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
        
        computed_digest = compute_artifact_digest(local.manifest_path, )
        
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


def x_verify_proof_bundle_wire__mutmut_17(
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
        attestation_digest = None
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


def x_verify_proof_bundle_wire__mutmut_18(
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
        attestation_digest = attestation.get("digest_hex") and attestation.get("digest")
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


def x_verify_proof_bundle_wire__mutmut_19(
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
        attestation_digest = attestation.get(None) or attestation.get("digest")
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


def x_verify_proof_bundle_wire__mutmut_20(
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
        attestation_digest = attestation.get("XXdigest_hexXX") or attestation.get("digest")
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


def x_verify_proof_bundle_wire__mutmut_21(
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
        attestation_digest = attestation.get("DIGEST_HEX") or attestation.get("digest")
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


def x_verify_proof_bundle_wire__mutmut_22(
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
        attestation_digest = attestation.get("digest_hex") or attestation.get(None)
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


def x_verify_proof_bundle_wire__mutmut_23(
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
        attestation_digest = attestation.get("digest_hex") or attestation.get("XXdigestXX")
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


def x_verify_proof_bundle_wire__mutmut_24(
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
        attestation_digest = attestation.get("digest_hex") or attestation.get("DIGEST")
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


def x_verify_proof_bundle_wire__mutmut_25(
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
        if attestation_digest or attestation_digest != computed_digest:
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


def x_verify_proof_bundle_wire__mutmut_26(
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
        if attestation_digest and attestation_digest == computed_digest:
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


def x_verify_proof_bundle_wire__mutmut_27(
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
                None,
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


def x_verify_proof_bundle_wire__mutmut_28(
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
                None,
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


def x_verify_proof_bundle_wire__mutmut_29(
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
                expected=None,
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


def x_verify_proof_bundle_wire__mutmut_30(
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
                actual=None,
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


def x_verify_proof_bundle_wire__mutmut_31(
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


def x_verify_proof_bundle_wire__mutmut_32(
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


def x_verify_proof_bundle_wire__mutmut_33(
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


def x_verify_proof_bundle_wire__mutmut_34(
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


def x_verify_proof_bundle_wire__mutmut_35(
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
                "XXWire bundle commitment mismatchXX",
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


def x_verify_proof_bundle_wire__mutmut_36(
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
                "wire bundle commitment mismatch",
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


def x_verify_proof_bundle_wire__mutmut_37(
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
                "WIRE BUNDLE COMMITMENT MISMATCH",
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


def x_verify_proof_bundle_wire__mutmut_38(
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
        if require_zk or wire.proof_system == "risc0":
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


def x_verify_proof_bundle_wire__mutmut_39(
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
        if require_zk and wire.proof_system != "risc0":
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


def x_verify_proof_bundle_wire__mutmut_40(
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
        if require_zk and wire.proof_system == "XXrisc0XX":
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


def x_verify_proof_bundle_wire__mutmut_41(
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
        if require_zk and wire.proof_system == "RISC0":
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


def x_verify_proof_bundle_wire__mutmut_42(
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
                None,
                expected_method_id=expected_method_id,
                require_zk=True,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_43(
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
                expected_method_id=None,
                require_zk=True,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_44(
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
                require_zk=None,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_45(
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
                expected_method_id=expected_method_id,
                require_zk=True,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_46(
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
                require_zk=True,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_47(
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
                )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_48(
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
                require_zk=False,
            )
        
        return VerificationReport.ok(
            "Wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_49(
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
            None,
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_50(
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
            proof_system=None,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_51(
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
            digest=None,
        )


def x_verify_proof_bundle_wire__mutmut_52(
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
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_53(
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
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_54(
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
            )


def x_verify_proof_bundle_wire__mutmut_55(
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
            "XXWire bundle verified successfullyXX",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_56(
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
            "wire bundle verified successfully",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )


def x_verify_proof_bundle_wire__mutmut_57(
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
            "WIRE BUNDLE VERIFIED SUCCESSFULLY",
            proof_system=wire.proof_system,
            digest=computed_digest,
        )

x_verify_proof_bundle_wire__mutmut_mutants : ClassVar[MutantDict] = {
'x_verify_proof_bundle_wire__mutmut_1': x_verify_proof_bundle_wire__mutmut_1, 
    'x_verify_proof_bundle_wire__mutmut_2': x_verify_proof_bundle_wire__mutmut_2, 
    'x_verify_proof_bundle_wire__mutmut_3': x_verify_proof_bundle_wire__mutmut_3, 
    'x_verify_proof_bundle_wire__mutmut_4': x_verify_proof_bundle_wire__mutmut_4, 
    'x_verify_proof_bundle_wire__mutmut_5': x_verify_proof_bundle_wire__mutmut_5, 
    'x_verify_proof_bundle_wire__mutmut_6': x_verify_proof_bundle_wire__mutmut_6, 
    'x_verify_proof_bundle_wire__mutmut_7': x_verify_proof_bundle_wire__mutmut_7, 
    'x_verify_proof_bundle_wire__mutmut_8': x_verify_proof_bundle_wire__mutmut_8, 
    'x_verify_proof_bundle_wire__mutmut_9': x_verify_proof_bundle_wire__mutmut_9, 
    'x_verify_proof_bundle_wire__mutmut_10': x_verify_proof_bundle_wire__mutmut_10, 
    'x_verify_proof_bundle_wire__mutmut_11': x_verify_proof_bundle_wire__mutmut_11, 
    'x_verify_proof_bundle_wire__mutmut_12': x_verify_proof_bundle_wire__mutmut_12, 
    'x_verify_proof_bundle_wire__mutmut_13': x_verify_proof_bundle_wire__mutmut_13, 
    'x_verify_proof_bundle_wire__mutmut_14': x_verify_proof_bundle_wire__mutmut_14, 
    'x_verify_proof_bundle_wire__mutmut_15': x_verify_proof_bundle_wire__mutmut_15, 
    'x_verify_proof_bundle_wire__mutmut_16': x_verify_proof_bundle_wire__mutmut_16, 
    'x_verify_proof_bundle_wire__mutmut_17': x_verify_proof_bundle_wire__mutmut_17, 
    'x_verify_proof_bundle_wire__mutmut_18': x_verify_proof_bundle_wire__mutmut_18, 
    'x_verify_proof_bundle_wire__mutmut_19': x_verify_proof_bundle_wire__mutmut_19, 
    'x_verify_proof_bundle_wire__mutmut_20': x_verify_proof_bundle_wire__mutmut_20, 
    'x_verify_proof_bundle_wire__mutmut_21': x_verify_proof_bundle_wire__mutmut_21, 
    'x_verify_proof_bundle_wire__mutmut_22': x_verify_proof_bundle_wire__mutmut_22, 
    'x_verify_proof_bundle_wire__mutmut_23': x_verify_proof_bundle_wire__mutmut_23, 
    'x_verify_proof_bundle_wire__mutmut_24': x_verify_proof_bundle_wire__mutmut_24, 
    'x_verify_proof_bundle_wire__mutmut_25': x_verify_proof_bundle_wire__mutmut_25, 
    'x_verify_proof_bundle_wire__mutmut_26': x_verify_proof_bundle_wire__mutmut_26, 
    'x_verify_proof_bundle_wire__mutmut_27': x_verify_proof_bundle_wire__mutmut_27, 
    'x_verify_proof_bundle_wire__mutmut_28': x_verify_proof_bundle_wire__mutmut_28, 
    'x_verify_proof_bundle_wire__mutmut_29': x_verify_proof_bundle_wire__mutmut_29, 
    'x_verify_proof_bundle_wire__mutmut_30': x_verify_proof_bundle_wire__mutmut_30, 
    'x_verify_proof_bundle_wire__mutmut_31': x_verify_proof_bundle_wire__mutmut_31, 
    'x_verify_proof_bundle_wire__mutmut_32': x_verify_proof_bundle_wire__mutmut_32, 
    'x_verify_proof_bundle_wire__mutmut_33': x_verify_proof_bundle_wire__mutmut_33, 
    'x_verify_proof_bundle_wire__mutmut_34': x_verify_proof_bundle_wire__mutmut_34, 
    'x_verify_proof_bundle_wire__mutmut_35': x_verify_proof_bundle_wire__mutmut_35, 
    'x_verify_proof_bundle_wire__mutmut_36': x_verify_proof_bundle_wire__mutmut_36, 
    'x_verify_proof_bundle_wire__mutmut_37': x_verify_proof_bundle_wire__mutmut_37, 
    'x_verify_proof_bundle_wire__mutmut_38': x_verify_proof_bundle_wire__mutmut_38, 
    'x_verify_proof_bundle_wire__mutmut_39': x_verify_proof_bundle_wire__mutmut_39, 
    'x_verify_proof_bundle_wire__mutmut_40': x_verify_proof_bundle_wire__mutmut_40, 
    'x_verify_proof_bundle_wire__mutmut_41': x_verify_proof_bundle_wire__mutmut_41, 
    'x_verify_proof_bundle_wire__mutmut_42': x_verify_proof_bundle_wire__mutmut_42, 
    'x_verify_proof_bundle_wire__mutmut_43': x_verify_proof_bundle_wire__mutmut_43, 
    'x_verify_proof_bundle_wire__mutmut_44': x_verify_proof_bundle_wire__mutmut_44, 
    'x_verify_proof_bundle_wire__mutmut_45': x_verify_proof_bundle_wire__mutmut_45, 
    'x_verify_proof_bundle_wire__mutmut_46': x_verify_proof_bundle_wire__mutmut_46, 
    'x_verify_proof_bundle_wire__mutmut_47': x_verify_proof_bundle_wire__mutmut_47, 
    'x_verify_proof_bundle_wire__mutmut_48': x_verify_proof_bundle_wire__mutmut_48, 
    'x_verify_proof_bundle_wire__mutmut_49': x_verify_proof_bundle_wire__mutmut_49, 
    'x_verify_proof_bundle_wire__mutmut_50': x_verify_proof_bundle_wire__mutmut_50, 
    'x_verify_proof_bundle_wire__mutmut_51': x_verify_proof_bundle_wire__mutmut_51, 
    'x_verify_proof_bundle_wire__mutmut_52': x_verify_proof_bundle_wire__mutmut_52, 
    'x_verify_proof_bundle_wire__mutmut_53': x_verify_proof_bundle_wire__mutmut_53, 
    'x_verify_proof_bundle_wire__mutmut_54': x_verify_proof_bundle_wire__mutmut_54, 
    'x_verify_proof_bundle_wire__mutmut_55': x_verify_proof_bundle_wire__mutmut_55, 
    'x_verify_proof_bundle_wire__mutmut_56': x_verify_proof_bundle_wire__mutmut_56, 
    'x_verify_proof_bundle_wire__mutmut_57': x_verify_proof_bundle_wire__mutmut_57
}

def verify_proof_bundle_wire(*args, **kwargs):
    result = _mutmut_trampoline(x_verify_proof_bundle_wire__mutmut_orig, x_verify_proof_bundle_wire__mutmut_mutants, args, kwargs)
    return result 

verify_proof_bundle_wire.__signature__ = _mutmut_signature(x_verify_proof_bundle_wire__mutmut_orig)
x_verify_proof_bundle_wire__mutmut_orig.__name__ = 'x_verify_proof_bundle_wire'


def x_verify_commitment_only__mutmut_orig(
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


def x_verify_commitment_only__mutmut_1(
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
    computed = None
    
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


def x_verify_commitment_only__mutmut_2(
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
    computed = compute_commitment_bytes(None, streams)
    
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


def x_verify_commitment_only__mutmut_3(
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
    computed = compute_commitment_bytes(manifest_bytes, None)
    
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


def x_verify_commitment_only__mutmut_4(
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
    computed = compute_commitment_bytes(streams)
    
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


def x_verify_commitment_only__mutmut_5(
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
    computed = compute_commitment_bytes(manifest_bytes, )
    
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


def x_verify_commitment_only__mutmut_6(
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
    
    if computed != expected_digest:
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


def x_verify_commitment_only__mutmut_7(
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
            None,
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_8(
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
            digest=None,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_9(
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
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_10(
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
            )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_11(
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
            "XXCommitment verifiedXX",
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_12(
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
            "commitment verified",
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_13(
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
            "COMMITMENT VERIFIED",
            digest=computed,
        )
    else:
        return VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_14(
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
            None,
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_15(
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
            None,
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_16(
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
            expected=None,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_17(
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
            computed=None,
        )


def x_verify_commitment_only__mutmut_18(
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
            "Commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_19(
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
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_20(
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
            computed=computed,
        )


def x_verify_commitment_only__mutmut_21(
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
            )


def x_verify_commitment_only__mutmut_22(
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
            "XXCommitment mismatchXX",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_23(
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
            "commitment mismatch",
            expected=expected_digest,
            computed=computed,
        )


def x_verify_commitment_only__mutmut_24(
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
            "COMMITMENT MISMATCH",
            expected=expected_digest,
            computed=computed,
        )

x_verify_commitment_only__mutmut_mutants : ClassVar[MutantDict] = {
'x_verify_commitment_only__mutmut_1': x_verify_commitment_only__mutmut_1, 
    'x_verify_commitment_only__mutmut_2': x_verify_commitment_only__mutmut_2, 
    'x_verify_commitment_only__mutmut_3': x_verify_commitment_only__mutmut_3, 
    'x_verify_commitment_only__mutmut_4': x_verify_commitment_only__mutmut_4, 
    'x_verify_commitment_only__mutmut_5': x_verify_commitment_only__mutmut_5, 
    'x_verify_commitment_only__mutmut_6': x_verify_commitment_only__mutmut_6, 
    'x_verify_commitment_only__mutmut_7': x_verify_commitment_only__mutmut_7, 
    'x_verify_commitment_only__mutmut_8': x_verify_commitment_only__mutmut_8, 
    'x_verify_commitment_only__mutmut_9': x_verify_commitment_only__mutmut_9, 
    'x_verify_commitment_only__mutmut_10': x_verify_commitment_only__mutmut_10, 
    'x_verify_commitment_only__mutmut_11': x_verify_commitment_only__mutmut_11, 
    'x_verify_commitment_only__mutmut_12': x_verify_commitment_only__mutmut_12, 
    'x_verify_commitment_only__mutmut_13': x_verify_commitment_only__mutmut_13, 
    'x_verify_commitment_only__mutmut_14': x_verify_commitment_only__mutmut_14, 
    'x_verify_commitment_only__mutmut_15': x_verify_commitment_only__mutmut_15, 
    'x_verify_commitment_only__mutmut_16': x_verify_commitment_only__mutmut_16, 
    'x_verify_commitment_only__mutmut_17': x_verify_commitment_only__mutmut_17, 
    'x_verify_commitment_only__mutmut_18': x_verify_commitment_only__mutmut_18, 
    'x_verify_commitment_only__mutmut_19': x_verify_commitment_only__mutmut_19, 
    'x_verify_commitment_only__mutmut_20': x_verify_commitment_only__mutmut_20, 
    'x_verify_commitment_only__mutmut_21': x_verify_commitment_only__mutmut_21, 
    'x_verify_commitment_only__mutmut_22': x_verify_commitment_only__mutmut_22, 
    'x_verify_commitment_only__mutmut_23': x_verify_commitment_only__mutmut_23, 
    'x_verify_commitment_only__mutmut_24': x_verify_commitment_only__mutmut_24
}

def verify_commitment_only(*args, **kwargs):
    result = _mutmut_trampoline(x_verify_commitment_only__mutmut_orig, x_verify_commitment_only__mutmut_mutants, args, kwargs)
    return result 

verify_commitment_only.__signature__ = _mutmut_signature(x_verify_commitment_only__mutmut_orig)
x_verify_commitment_only__mutmut_orig.__name__ = 'x_verify_commitment_only'
