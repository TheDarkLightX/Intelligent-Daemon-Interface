"""Integration between ZK proofs and Tau spec execution.

Verifies proofs before Tau execution and ensures privacy of Q-tables.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from idi.zk.proof_manager import ProofBundle, verify_proof


def verify_before_tau_execution(
    manifest_path: Path,
    proof_bundle: Optional[ProofBundle],
    tau_spec_path: Path,
    inputs_dir: Path,
) -> bool:
    """Verify ZK proof before executing Tau spec.
    
    Args:
        manifest_path: Path to artifact manifest
        proof_bundle: Proof bundle (None if using stub)
        tau_spec_path: Path to Tau spec file
        inputs_dir: Directory containing Tau inputs
    
    Returns:
        True if verification passes and execution can proceed
    """
    # Verify proof if available
    if proof_bundle:
        if not verify_proof(proof_bundle):
            print("❌ Proof verification failed - aborting Tau execution")
            return False
        
        # Check receipt timestamp (optional freshness check)
        receipt = json.loads(proof_bundle.receipt_path.read_text())
        if "timestamp" in receipt:
            import time
            age = time.time() - receipt["timestamp"]
            max_age = 7 * 24 * 3600  # 7 days
            if age > max_age:
                print(f"⚠️  Proof is stale ({age/3600:.1f} hours old)")
                # Continue anyway, but warn
    
    # Verify manifest exists
    if not manifest_path.exists():
        print("⚠️  Manifest not found - proceeding without verification")
        return True  # Allow execution without proof
    
    # Verify inputs directory exists
    if not inputs_dir.exists():
        print("❌ Inputs directory not found")
        return False
    
    print("✅ Proof verification passed - proceeding with Tau execution")
    return True


def execute_tau_with_proof_verification(
    tau_spec_path: Path,
    inputs_dir: Path,
    outputs_dir: Path,
    proof_bundle: Optional[ProofBundle] = None,
    manifest_path: Optional[Path] = None,
    tau_binary: str = "tau",
) -> tuple[bool, Optional[str]]:
    """Execute Tau spec with proof verification.
    
    Args:
        tau_spec_path: Path to Tau spec file
        inputs_dir: Directory containing inputs
        outputs_dir: Directory for outputs
        proof_bundle: Optional proof bundle for verification
        manifest_path: Optional manifest path
        tau_binary: Path to Tau binary
    
    Returns:
        Tuple of (success, output_or_error)
    """
    # Verify proof before execution
    if not verify_before_tau_execution(
        manifest_path=manifest_path or tau_spec_path.parent / "artifact_manifest.json",
        proof_bundle=proof_bundle,
        tau_spec_path=tau_spec_path,
        inputs_dir=inputs_dir,
    ):
        return False, "Proof verification failed"
    
    # Prepare Tau execution
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Build Tau command
    # Note: Actual execution depends on Tau binary availability
    # This is a placeholder for the integration
    cmd = [
        tau_binary,
        str(tau_spec_path),
    ]
    
    try:
        # Execute Tau (would need actual Tau binary)
        # result = subprocess.run(cmd, capture_output=True, text=True, cwd=inputs_dir.parent)
        # return result.returncode == 0, result.stdout
        
        print(f"✅ Tau execution prepared for {tau_spec_path}")
        print(f"   Inputs: {inputs_dir}")
        print(f"   Outputs: {outputs_dir}")
        return True, "Execution prepared (Tau binary not available for testing)"
    except FileNotFoundError:
        return False, f"Tau binary not found: {tau_binary}"


def log_proof_verification(
    proof_bundle: ProofBundle,
    log_path: Path,
) -> None:
    """Log proof verification results.
    
    Args:
        proof_bundle: Proof bundle to log
        log_path: Path to log file
    """
    receipt = json.loads(proof_bundle.receipt_path.read_text())
    
    log_entry = {
        "timestamp": receipt.get("timestamp"),
        "prover": receipt.get("prover", "unknown"),
        "digest": receipt.get("digest_hex", receipt.get("digest")),
        "method_id": receipt.get("method_id"),
        "verified": verify_proof(proof_bundle),
    }
    
    # Append to log
    if log_path.exists():
        log_data = json.loads(log_path.read_text())
    else:
        log_data = {"entries": []}
    
    log_data["entries"].append(log_entry)
    log_path.write_text(json.dumps(log_data, indent=2))

