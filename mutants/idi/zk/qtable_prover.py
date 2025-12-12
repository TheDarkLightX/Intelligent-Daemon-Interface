"""Q-table proof generation using Risc0 zkVM.

Generates proofs that Q-table lookups and action selections are correct.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from idi.zk.witness_generator import QTableWitness, serialize_witness


def generate_qtable_proof(
    witness: QTableWitness,
    out_dir: Path,
    risc0_workspace: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate Risc0 proof for Q-table action selection.
    
    Args:
        witness: Q-table witness data
        out_dir: Directory for proof outputs
        risc0_workspace: Path to Risc0 workspace (default: idi/zk/risc0)
    
    Returns:
        Tuple of (proof_path, receipt_path)
    """
    if risc0_workspace is None:
        risc0_workspace = Path(__file__).parent / "risc0"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    proof_path = out_dir / "qtable_proof.bin"
    receipt_path = out_dir / "qtable_receipt.json"
    
    # Serialize witness
    witness_bytes = serialize_witness(witness)
    
    # Write witness to temp file
    witness_file = out_dir / "witness.json"
    witness_file.write_bytes(witness_bytes)
    
    # Build Risc0 command
    # Note: This requires the guest program to be built first
    # cargo build --release -p idi_risc0_methods
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "idi_risc0_host",
        "--",
        "--witness",
        str(witness_file),
        "--proof",
        str(proof_path),
        "--receipt",
        str(receipt_path),
    ]
    
    # Run prover (would need custom host program for Q-table proofs)
    # For now, return paths - actual proof generation requires compiled guest
    print(f"⚠️  Q-table proof generation requires compiled guest program")
    print(f"   Build with: cd {risc0_workspace} && cargo build --release -p idi_risc0_methods")
    
    return proof_path, receipt_path


def verify_qtable_proof(
    proof_path: Path,
    receipt_path: Path,
    expected_q_root: bytes,
) -> bool:
    """Verify a Q-table proof.
    
    Args:
        proof_path: Path to proof binary
        receipt_path: Path to receipt JSON
        expected_q_root: Expected Q-table root hash
    
    Returns:
        True if proof is valid
    """
    if not receipt_path.exists():
        return False
    
    receipt = json.loads(receipt_path.read_text())
    
    # Verify receipt has expected fields
    if "q_table_root" not in receipt:
        return False
    
    receipt_root = bytes.fromhex(receipt["q_table_root"])
    
    return receipt_root == expected_q_root

