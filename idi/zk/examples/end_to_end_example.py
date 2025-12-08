"""End-to-end example: Training → Witness → Proof → Tau Execution

Demonstrates the complete workflow from Q-table training to verified Tau execution.

Run from project root:
    python -m idi.zk.examples.end_to_end_example
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from idi.zk.witness_generator import generate_witness_from_q_table
from idi.zk.proof_manager import generate_proof, verify_proof
from idi.zk.training_integration import (
    generate_proofs_from_training_output,
    verify_training_proofs,
)
from idi.zk.tau_integration import (
    verify_before_tau_execution,
    execute_tau_with_proof_verification,
)


def example_complete_workflow():
    """Complete workflow example."""
    print("=" * 80)
    print("IDI ZK Proof End-to-End Example")
    print("=" * 80)
    print()
    
    # Step 1: Simulate trained Q-table
    print("Step 1: Load Trained Q-Table")
    print("-" * 80)
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
        "state_1": {"hold": 0.0, "buy": 0.0, "sell": 0.5},
        "state_2": {"hold": 0.3, "buy": 0.2, "sell": 0.1},
    }
    print(f"✅ Loaded Q-table with {len(q_table)} states")
    print()
    
    # Step 2: Generate witness
    print("Step 2: Generate Witness")
    print("-" * 80)
    witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=False)
    print(f"✅ Generated witness for state_0")
    print(f"   Selected action: {witness.selected_action} (0=hold, 1=buy, 2=sell)")
    print(f"   Q-table root: {witness.q_table_root.hex()[:16]}...")
    print()
    
    # Step 3: Create manifest and streams
    print("Step 3: Create Manifest and Streams")
    print("-" * 80)
    artifact_dir = Path("/tmp/idi_example")
    artifact_dir.mkdir(exist_ok=True)
    
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "example_agent",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {"episodes": 100},
        "policy_summary": {"states": len(q_table), "actions": ["hold", "buy", "sell"]},
        "trace_summary": {"length": 64, "stream_hashes": {}},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Created manifest: {manifest_path}")
    
    streams_dir = artifact_dir / "streams"
    streams_dir.mkdir(exist_ok=True)
    (streams_dir / "q_buy.in").write_text("1\n0\n1\n")
    (streams_dir / "q_sell.in").write_text("0\n1\n0\n")
    print(f"✅ Created streams in: {streams_dir}")
    print()
    
    # Step 4: Generate proof
    print("Step 4: Generate Proof")
    print("-" * 80)
    proof_dir = artifact_dir / "proof_stub"
    bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams_dir,
        out_dir=proof_dir,
    )
    print(f"✅ Generated proof bundle")
    print(f"   Proof: {bundle.proof_path}")
    print(f"   Receipt: {bundle.receipt_path}")
    print()
    
    # Step 5: Verify proof
    print("Step 5: Verify Proof")
    print("-" * 80)
    verified = verify_proof(bundle)
    print(f"{'✅' if verified else '❌'} Proof verification: {verified}")
    print()
    
    # Step 6: Tau execution preparation
    print("Step 6: Prepare Tau Execution")
    print("-" * 80)
    tau_spec_path = artifact_dir / "agent.tau"
    tau_spec_path.write_text("# Example Tau spec\n")
    
    can_execute = verify_before_tau_execution(
        manifest_path=manifest_path,
        proof_bundle=bundle,
        tau_spec_path=tau_spec_path,
        inputs_dir=streams_dir,
    )
    print(f"{'✅' if can_execute else '❌'} Tau execution ready: {can_execute}")
    print()
    
    print("=" * 80)
    print("✅ End-to-End Workflow Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Compile Risc0 guest programs: cd idi/zk/risc0 && cargo build")
    print("  2. Generate proofs: Use Risc0 prover instead of stub")
    print("  3. Execute Tau spec: Run with verified inputs")
    print("  4. Deploy: Use proofs for on-chain verification")


if __name__ == "__main__":
    example_complete_workflow()

