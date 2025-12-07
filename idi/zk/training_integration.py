"""Integration between training pipeline and ZK proof generation.

Connects Q-learning training output to witness generation and proof creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from idi.zk.witness_generator import (
    QTableWitness,
    generate_witness_from_q_table,
    serialize_witness,
)
from idi.zk.proof_manager import ProofBundle, generate_proof, verify_proof
from idi.zk.qtable_prover import generate_qtable_proof, verify_qtable_proof


def generate_proofs_from_training_output(
    q_table_path: Path,
    manifest_path: Path,
    stream_dir: Path,
    out_dir: Path,
    use_merkle: bool = True,
) -> Dict[str, ProofBundle]:
    """Generate ZK proofs from training output.
    
    Args:
        q_table_path: Path to trained Q-table JSON file
        manifest_path: Path to artifact manifest
        stream_dir: Directory containing Tau input streams
        out_dir: Directory for proof outputs
        use_merkle: Whether to use Merkle trees for large Q-tables
    
    Returns:
        Dictionary mapping state keys to proof bundles
    """
    # Load Q-table
    q_table_data = json.loads(q_table_path.read_text())
    q_table: Dict[str, Dict[str, float]] = q_table_data.get("q_table", q_table_data)
    
    # Generate manifest proof (existing)
    manifest_proof_dir = out_dir / "manifest_proof"
    manifest_bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=stream_dir,
        out_dir=manifest_proof_dir,
    )
    
    # Generate Q-table proofs for sample states
    qtable_proofs = {}
    sample_states = list(q_table.keys())[:10]  # Sample first 10 states
    
    for state_key in sample_states:
        # Generate witness
        witness = generate_witness_from_q_table(
            q_table=q_table,
            state_key=state_key,
            use_merkle=use_merkle,
        )
        
        # Generate proof
        qtable_proof_dir = out_dir / "qtable_proofs" / state_key
        proof_path, receipt_path = generate_qtable_proof(
            witness=witness,
            out_dir=qtable_proof_dir,
        )
        
        qtable_proofs[state_key] = ProofBundle(
            manifest_path=manifest_path,
            proof_path=proof_path,
            receipt_path=receipt_path,
        )
    
    return {
        "manifest": manifest_bundle,
        **qtable_proofs,
    }


def verify_training_proofs(
    proof_bundles: Dict[str, ProofBundle],
    q_table_path: Path,
) -> Dict[str, bool]:
    """Verify all proofs from training output.
    
    Args:
        proof_bundles: Dictionary of proof bundles
        q_table_path: Path to Q-table for verification
    
    Returns:
        Dictionary mapping proof names to verification results
    """
    results = {}
    
    # Load Q-table root
    q_table_data = json.loads(q_table_path.read_text())
    q_table = q_table_data.get("q_table", q_table_data)
    
    # Verify manifest proof
    if "manifest" in proof_bundles:
        results["manifest"] = verify_proof(proof_bundles["manifest"])
    
    # Verify Q-table proofs
    for state_key, bundle in proof_bundles.items():
        if state_key == "manifest":
            continue
        
        # Get expected Q-table root
        witness = generate_witness_from_q_table(q_table, state_key, use_merkle=True)
        
        results[state_key] = verify_qtable_proof(
            proof_path=bundle.proof_path,
            receipt_path=bundle.receipt_path,
            expected_q_root=witness.q_table_root,
        )
    
    return results

