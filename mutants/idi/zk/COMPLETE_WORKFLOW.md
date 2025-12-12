# Complete ZK Proof Workflow

## Overview

This document describes the complete workflow from Q-table training to verified Tau execution using zero-knowledge proofs.

## Workflow Steps

### 1. Train Q-Table

```bash
cd idi/training/python
python -m run_idi_trainer \
    --config config.json \
    --out artifacts/my_agent/streams
```

This generates:
- Q-table (JSON format)
- Tau input streams (`q_buy.in`, `q_sell.in`, etc.)
- Training metadata

### 2. Create Artifact Manifest

```bash
python -m idi.devkit.builder \
    --config config.json \
    --out artifacts/my_agent \
    --install-inputs specs/V38_Minimal_Core/inputs
```

This creates:
- `artifact_manifest.json` - Stream hashes and metadata
- `streams/` - Tau input files

### 3. Generate Witness

```python
from idi.zk.witness_generator import generate_witness_from_q_table
import json

# Load Q-table
with open("artifacts/my_agent/q_table.json") as f:
    q_table = json.load(f)

# Generate witness for a state
witness = generate_witness_from_q_table(
    q_table=q_table,
    state_key="state_0",
    use_merkle=True,  # Use Merkle for large tables
)
```

### 4. Generate Proofs

#### Option A: Stub Proof (for testing)

```python
from idi.zk.proof_manager import generate_proof, verify_proof
from pathlib import Path

bundle = generate_proof(
    manifest_path=Path("artifacts/my_agent/artifact_manifest.json"),
    stream_dir=Path("artifacts/my_agent/streams"),
    out_dir=Path("artifacts/my_agent/proof_stub"),
)

assert verify_proof(bundle)
```

#### Option B: Risc0 Proof (production)

```bash
cd idi/zk/risc0
cargo build --release -p idi_risc0_methods
cargo run --release -p idi_risc0_host -- \
    --manifest artifacts/my_agent/artifact_manifest.json \
    --streams artifacts/my_agent/streams \
    --proof artifacts/my_agent/proof_risc0/proof.bin \
    --receipt artifacts/my_agent/proof_risc0/receipt.json
```

### 5. Verify Before Tau Execution

```python
from idi.zk.tau_integration import verify_before_tau_execution
from idi.zk.proof_manager import ProofBundle
from pathlib import Path

# Load proof bundle
bundle = ProofBundle(
    manifest_path=Path("artifacts/my_agent/artifact_manifest.json"),
    proof_path=Path("artifacts/my_agent/proof_risc0/proof.bin"),
    receipt_path=Path("artifacts/my_agent/proof_risc0/receipt.json"),
)

# Verify before execution
if verify_before_tau_execution(
    manifest_path=bundle.manifest_path,
    proof_bundle=bundle,
    tau_spec_path=Path("specs/my_agent.tau"),
    inputs_dir=Path("artifacts/my_agent/streams"),
):
    # Execute Tau spec
    execute_tau_spec(...)
else:
    print("Proof verification failed - aborting")
```

### 6. Execute Tau Spec

```bash
cd specs/my_agent
tau < agent.tau
```

The Tau daemon should verify proofs before accepting Q-table inputs.

## Complete Example

See `idi/zk/examples/end_to_end_example.py` for a complete working example.

## File Structure

```
artifacts/my_agent/
├── artifact_manifest.json      # Stream hashes + metadata
├── q_table.json                # Trained Q-table
├── streams/                     # Tau input streams
│   ├── q_buy.in
│   ├── q_sell.in
│   └── ...
├── proof_stub/                  # Stub proof (testing)
│   ├── proof.bin
│   └── receipt.json
└── proof_risc0/                # Risc0 proof (production)
    ├── proof.bin
    └── receipt.json
```

## Privacy Guarantees

- **Q-table privacy**: Only Merkle root commitments are revealed
- **Training privacy**: Training techniques remain private
- **Action verification**: Proofs verify action selection without revealing Q-values
- **On-chain attestation**: Proofs enable trustless verification

## Performance Considerations

- **Small Q-tables** (<100 entries): Use in-memory hashing
- **Large Q-tables** (>100 entries): Use Merkle trees
- **Proof generation**: Risc0 proofs take ~seconds for small manifests
- **Verification**: Fast (milliseconds)

## Next Steps

1. Compile Risc0 guest programs
2. Test with Q-tables from training
3. Integrate with Tau daemon
4. Deploy on-chain verification

