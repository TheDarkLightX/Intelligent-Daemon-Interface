"""Proof bundle helpers for IDI zk workflows.

Manages proof generation and verification for manifest-based ZK proofs.
Supports both stub (SHA-256 only) and real zkVM provers (Risc0).

Security Properties:
- Integrity: Proof digest binds manifest and stream data
- Determinism: Same inputs always produce same digest
- Verifiability: Anyone can verify proofs without proving key

Dependencies: hashlib, json, subprocess
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_RECEIPT_SIZE_BYTES = 512 * 1024  # safety cap

def _combined_hash(manifest_path: Path, stream_dir: Path) -> str:
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
    return hasher.hexdigest()


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


def generate_proof(
    *,
    manifest_path: Path,
    stream_dir: Path,
    out_dir: Path,
    prover_command: Optional[str] = None,
) -> ProofBundle:
    """Generate a proof bundle via stub or external prover command."""

    out_dir.mkdir(parents=True, exist_ok=True)
    proof_path = out_dir / "proof.bin"
    receipt_path = out_dir / "receipt.json"

    digest = _combined_hash(manifest_path, stream_dir)

    if prover_command:
        cmd = prover_command.format(
            manifest=str(manifest_path),
            streams=str(stream_dir),
            proof=str(proof_path),
            receipt=str(receipt_path),
        )
        subprocess.run(cmd, shell=True, check=True)
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
    if external_receipt:
        receipt["prover"] = external_receipt.get("prover", receipt["prover"])
        receipt["method_id"] = external_receipt.get("method_id")
        receipt["prover_digest"] = external_receipt.get("digest_hex")
        receipt["prover_meta"] = external_receipt
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return ProofBundle(manifest_path=manifest_path, proof_path=proof_path, receipt_path=receipt_path)


def verify_proof(bundle: ProofBundle) -> bool:
    """Verify that the proof digest matches the manifest and stream directory."""

    receipt_bytes = bundle.receipt_path.read_bytes()
    if len(receipt_bytes) > MAX_RECEIPT_SIZE_BYTES:
        return False
    receipt = json.loads(receipt_bytes.decode())
    manifest_path = Path(receipt["manifest"])
    stream_dir = Path(receipt.get("streams", manifest_path.parent / "streams"))
    if not stream_dir.exists():
        stream_dir = manifest_path.parent / "streams"
    digest = _combined_hash(manifest_path, stream_dir)
    return digest == receipt["digest"]

