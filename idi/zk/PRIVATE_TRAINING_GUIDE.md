# Private Training with ZK Proofs - User Guide

## Overview

This guide demonstrates how users can train intelligent agents privately, keep their Q-tables secret, and use zero-knowledge proofs on the Tau Net testnet blockchain without exposing their intelligence.

## Key Privacy Guarantees

✅ **Q-table privacy**: Only Merkle root commitments are revealed, never the actual Q-values  
✅ **Training privacy**: Training techniques and hyperparameters remain private  
✅ **Action verification**: Proofs verify action selection without revealing Q-values  
✅ **On-chain attestation**: Proofs enable trustless verification without exposing secrets  

## Complete Workflow

### Step 1: Train Agent Privately

Train your agent locally. The Q-table stays on your machine and is never shared.

```python
from idi.training.python.idi_iann.trainer import QTrainer
from idi.training.python.idi_iann.config import TrainingConfig

# Train agent (Q-table stays private)
config = TrainingConfig(episodes=100, strategy="momentum")
trainer = QTrainer(config, seed=42)
policy, trace = trainer.run()

# Save Q-table locally (NEVER share this!)
q_table = policy._table  # This is your private intelligence
```

### Step 2: Create Artifact Manifest

Create a manifest that describes your agent WITHOUT including Q-values.

```python
from pathlib import Path
import json

manifest = {
    "schema_version": "1.0.0",
    "artifact_id": "my_private_agent",
    "training_config": {
        "episodes": 100,
        "strategy": "momentum",
        # NOTE: Q-values are NOT included!
    },
    "policy_summary": {
        "num_states": len(q_table),
        "actions": ["hold", "buy", "sell"],
        # Only summary stats, not actual Q-values
    },
}

manifest_path.write_text(json.dumps(manifest, indent=2))
```

### Step 3: Generate Witness with Merkle Commitment

Generate a witness that commits to your Q-table using a Merkle root, without revealing Q-values.

```python
from idi.zk.witness_generator import generate_witness_from_q_table

# Generate witness (only Merkle root revealed, not Q-values)
witness = generate_witness_from_q_table(
    q_table=q_table,
    state_key="state_0",
    use_merkle=True,  # Use Merkle for privacy
)

# witness.q_table_root is the commitment (safe to share)
# witness.selected_action is the action (safe to share)
# Q-values are NOT in the witness!
```

### Step 4: Generate ZK Proof Bundle

Generate a proof bundle that can be verified without exposing Q-values.

```python
from idi.zk.proof_manager import generate_proof, verify_proof

# Generate proof (Q-values stay private)
proof_bundle = generate_proof(
    manifest_path=manifest_path,
    stream_dir=streams_dir,
    out_dir=proof_dir,
)

# Verify proof (verifier doesn't see Q-values)
verified = verify_proof(proof_bundle)
assert verified is True
```

### Step 5: Submit to Tau Net Testnet

Use TauBridge to submit your proof to Tau Net testnet.

```python
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext
from idi.taunet_bridge.protocols import ZkProofBundle as BridgeProofBundle

# Configure TauBridge
config = ZkConfig(enabled=True, proof_system="stub")  # Use "risc0" for production
adapter = TauNetZkAdapter(config)

# Convert to bridge format
bridge_bundle = BridgeProofBundle(
    proof_path=proof_bundle.proof_path,
    receipt_path=proof_bundle.receipt_path,
    manifest_path=proof_bundle.manifest_path,
    tx_hash="your_tx_hash",
)

# Validate (ready for testnet submission)
ctx = ValidationContext(
    tx_hash="your_tx_hash",
    payload={"zk_proof": bridge_bundle}
)

validation_step = ZkValidationStep(adapter, required=False)
validation_step.run(ctx)  # Ready for testnet!
```

## Running the Demo

See the complete workflow in action:

```bash
# Run the demonstration
python -m idi.zk.examples.private_training_demo

# Run the comprehensive test suite
pytest idi/zk/tests/test_private_training_e2e.py -v
```

## Privacy Verification

The test suite verifies that:

1. ✅ Q-values are NOT in manifest files
2. ✅ Q-values are NOT in stream files (only binary signals)
3. ✅ Q-values are NOT in witness (only Merkle root)
4. ✅ Q-values are NOT in proof receipts
5. ✅ Proofs can be verified without Q-values
6. ✅ TauBridge can validate without Q-values

## Security Properties

- **Integrity**: Proof digest binds manifest and stream data
- **Privacy**: Q-values never exposed in any artifact
- **Verifiability**: Anyone can verify proofs without proving key
- **Tamper Detection**: Any modification to Q-table invalidates proof

## Production Usage

For production, replace stub proofs with Risc0:

```bash
# Build Risc0 guest programs
cd idi/zk/risc0
cargo build --release -p idi_risc0_methods

# Generate Risc0 proof
cargo run --release -p idi_risc0_host -- \
    --manifest artifacts/my_agent/artifact_manifest.json \
    --streams artifacts/my_agent/streams \
    --proof artifacts/my_agent/proof_risc0/proof.bin \
    --receipt artifacts/my_agent/proof_risc0/receipt.json
```

## File Structure

```
artifacts/my_agent/
├── q_table.json              # PRIVATE - Never share!
├── artifact_manifest.json    # Public - No Q-values
├── streams/                   # Public - Only binary signals
│   ├── q_buy.in
│   └── q_sell.in
└── proof_risc0/              # Public - Verifiable without Q-values
    ├── proof.bin
    └── receipt.json
```

## Next Steps

1. ✅ **Training**: Train your agent privately
2. ✅ **Proof Generation**: Generate ZK proofs (stub or Risc0)
3. ✅ **Verification**: Verify proofs locally
4. ✅ **TauBridge**: Prepare for Tau Net testnet submission
5. ✅ **Deployment**: Submit proofs to testnet (intelligence stays secret!)

## Summary

**Your intelligence stays private!** The ZK proof system allows you to:
- Train agents privately (Q-tables never leave your machine)
- Generate verifiable proofs without exposing Q-values
- Submit proofs to Tau Net testnet for validation
- Keep your training techniques and Q-values completely secret

The network can verify that your agent makes correct decisions without ever seeing how it was trained or what Q-values it uses.

