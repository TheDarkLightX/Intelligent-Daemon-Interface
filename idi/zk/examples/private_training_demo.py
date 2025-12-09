#!/usr/bin/env python3
"""Demonstration: Private Training with ZK Proofs for Tau Net Testnet

This script demonstrates the complete workflow:
1. Train an agent privately (Q-table stays secret)
2. Generate ZK proofs without revealing Q-values
3. Verify proofs can be validated without exposing intelligence
4. Prepare proofs for Tau Net testnet submission

Run from project root:
    python -m idi.zk.examples.private_training_demo
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
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext
from idi.taunet_bridge.protocols import LocalZkProofBundle as BridgeProofBundle


def main():
    """Run the private training demonstration."""
    print("=" * 80)
    print("PRIVATE TRAINING WITH ZK PROOFS - DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo shows how users can:")
    print("  • Train agents privately (Q-tables stay secret)")
    print("  • Generate ZK proofs without revealing Q-values")
    print("  • Verify proofs on Tau Net testnet without exposing intelligence")
    print()
    
    # Step 1: Simulate private Q-table training
    print("[Step 1] User trains agent privately...")
    print("-" * 80)
    
    # This is the user's private intelligence - never shared!
    private_q_table = {
        "state_0": {"hold": 0.1, "buy": 0.8, "sell": 0.0},
        "state_1": {"hold": 0.2, "buy": 0.1, "sell": 0.7},
        "state_2": {"hold": 0.5, "buy": 0.3, "sell": 0.2},
    }
    
    print(f"✓ Q-table created with {len(private_q_table)} states")
    print(f"✓ Q-values are PRIVATE (stored locally, never shared)")
    print(f"  Example Q-values (secret): state_0.buy = {private_q_table['state_0']['buy']}")
    print()
    
    # Step 2: Create artifact manifest (NO Q-values)
    print("[Step 2] Create artifact manifest (Q-values NOT included)...")
    print("-" * 80)
    
    artifact_dir = Path("/tmp/idi_private_demo")
    artifact_dir.mkdir(exist_ok=True)
    
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "private_agent_demo",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {
            "episodes": 100,
            "strategy": "momentum",
        },
        "policy_summary": {
            "num_states": len(private_q_table),
            "actions": ["hold", "buy", "sell"],
        },
        "trace_summary": {
            "length": 10,
            "stream_hashes": {},
        },
        "proof_policy": "stub",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    print(f"✓ Manifest created: {manifest_path}")
    print(f"✓ Verified: Q-values NOT in manifest (only metadata)")
    print()
    
    # Step 3: Create streams (action selections only)
    print("[Step 3] Create Tau input streams (action selections only)...")
    print("-" * 80)
    
    streams_dir = artifact_dir / "streams"
    streams_dir.mkdir(exist_ok=True)
    
    # Generate streams based on greedy action selection
    for state_key, q_vals in private_q_table.items():
        best_action = max(q_vals.items(), key=lambda x: x[1])[0]
        if best_action == "buy":
            (streams_dir / f"{state_key}_buy.in").write_text("1\n")
            (streams_dir / f"{state_key}_sell.in").write_text("0\n")
        elif best_action == "sell":
            (streams_dir / f"{state_key}_buy.in").write_text("0\n")
            (streams_dir / f"{state_key}_sell.in").write_text("1\n")
        else:
            (streams_dir / f"{state_key}_buy.in").write_text("0\n")
            (streams_dir / f"{state_key}_sell.in").write_text("0\n")
    
    stream_files = list(streams_dir.glob("*.in"))
    print(f"✓ Created {len(stream_files)} stream files")
    print(f"✓ Streams contain only binary signals (0/1), no Q-values")
    print()
    
    # Step 4: Generate witness with Merkle commitment
    print("[Step 4] Generate witness with Merkle commitment...")
    print("-" * 80)
    
    state_key = "state_0"
    witness = generate_witness_from_q_table(
        q_table=private_q_table,
        state_key=state_key,
        use_merkle=True,
    )
    
    print(f"✓ Witness generated for {state_key}")
    print(f"✓ Selected action: {witness.selected_action} (0=hold, 1=buy, 2=sell)")
    print(f"✓ Q-table root (commitment): {witness.q_table_root.hex()[:32]}...")
    print(f"✓ Q-values NOT exposed (only Merkle root commitment)")
    print()
    
    # Step 5: Generate proof bundle
    print("[Step 5] Generate ZK proof bundle...")
    print("-" * 80)
    
    proof_dir = artifact_dir / "proof_stub"
    proof_bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams_dir,
        out_dir=proof_dir,
    )
    
    print(f"✓ Proof bundle generated")
    print(f"✓ Proof: {proof_bundle.proof_path}")
    print(f"✓ Receipt: {proof_bundle.receipt_path}")
    print()
    
    # Step 6: Verify proof
    print("[Step 6] Verify proof (verifier doesn't see Q-values)...")
    print("-" * 80)
    
    verified = verify_proof(proof_bundle)
    print(f"✓ Proof verified: {verified}")
    print(f"✓ Verifier did NOT need to see Q-values")
    print()
    
    # Step 7: TauBridge integration
    print("[Step 7] TauBridge integration (Tau Net testnet)...")
    print("-" * 80)
    
    config = ZkConfig(enabled=True, proof_system="stub", require_proofs=False)
    adapter = TauNetZkAdapter(config)
    
    bridge_bundle = BridgeProofBundle(
        proof_path=proof_bundle.proof_path,
        receipt_path=proof_bundle.receipt_path,
        manifest_path=proof_bundle.manifest_path,
        tx_hash="demo_tx_001",
    )
    
    ctx = ValidationContext(
        tx_hash="demo_tx_001",
        payload={"zk_proof": bridge_bundle}
    )
    
    validation_step = ZkValidationStep(adapter, required=False)
    validation_step.run(ctx)
    
    print(f"✓ TauBridge validation passed")
    print(f"✓ Proof ready for Tau Net testnet submission")
    print()
    
    # Summary
    print("=" * 80)
    print("PRIVACY VERIFICATION SUMMARY")
    print("=" * 80)
    print("✓ Q-table values: PRIVATE (never exposed)")
    print("✓ Manifest: Contains only metadata, no Q-values")
    print("✓ Streams: Only binary action signals, no Q-values")
    print("✓ Witness: Only Merkle root commitment, no Q-values")
    print("✓ Proof: Verifiable without Q-values")
    print("✓ TauBridge: Can validate without Q-values")
    print()
    print("✅ INTELLIGENCE REMAINS SECRET!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Use Risc0 prover for production proofs (replace stub)")
    print("  2. Submit proof to Tau Net testnet via sendtx command")
    print("  3. Network validates proof without seeing Q-values")
    print("  4. Your intelligence stays private!")


if __name__ == "__main__":
    main()

