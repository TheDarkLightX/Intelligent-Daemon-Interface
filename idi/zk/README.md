# IDI zk Workflow

This module glues lookup-table manifests into zero-knowledge proof flows.

## Stub prover
```
python - <<'PY'
from pathlib import Path
from idi.zk.proof_manager import generate_proof, verify_proof

bundle = generate_proof(
    manifest_path=Path("build/artifacts/sample/artifact_manifest.json"),
    stream_dir=Path("build/artifacts/sample/streams"),
    out_dir=Path("build/proofs/sample"),
)
assert verify_proof(bundle)
print("Proof digest:", bundle.receipt_path.read_text())
PY
```
The stub uses SHA-256 commitments to emulate a proof so downstream automation (Tau daemon, ledger, etc.) can proceed even before the zkVM integration is wired up.

Batch all manifests under `idi/artifacts/` with:
```
python -m idi.zk.run_stub_proofs --artifacts-root idi/artifacts
```
This creates `proof_stub/` folders next to each artifact (regime layers, emotive layers, etc.) and verifies the receipts in one go.

## Native Risc0 prover ✅ Production Ready

- **Status**: Fully functional end-to-end implementation with comprehensive testing
- Workspace lives in `idi/zk/risc0/` (`cargo` workspace with `host/`, `methods/`, and generated guest code)
- **Guest programs**:
  - `idi/zk/risc0/methods/idi-manifest/src/main.rs` - Verifies manifest and stream hashes
  - `idi/zk/risc0/methods/idi-qtable/src/main.rs` - Verifies Q-table action selection with Merkle proofs
- **Host program**: `idi/zk/risc0/host/src/main.rs` - Generates STARK proofs via Risc0 zkVM
- **Build**:
  ```bash
  cd idi/zk/risc0
  cargo build --release -p idi_risc0_methods  # Build guest programs
  cargo build --release -p idi_risc0_host      # Build host program
  ```
- **Usage**:
  ```bash
  cargo run --release -p idi_risc0_host -- \
      --manifest artifacts/my_agent/artifact_manifest.json \
      --streams artifacts/my_agent/streams \
      --proof artifacts/my_agent/proof_risc0/proof.bin \
      --receipt artifacts/my_agent/proof_risc0/receipt.json
  ```
- **Verification**: Receipt contains `method_id` (image ID) and `digest_hex` for cryptographic verification
- **Testing**: Complete end-to-end test suite in `idi/zk/tests/test_private_training_e2e.py`
  - Verifies Q-values remain private throughout workflow
  - Tests Risc0 proof generation and verification
  - Validates TauBridge integration

## Witness Generation ✅ Implemented

- **witness_generator.py**: Converts Q-tables to witness data for zk proofs
- **merkle_tree.py**: Builds Merkle trees for large Q-table commitments
- Supports both small (in-memory) and large (Merkle tree) Q-tables
- Fixed-point arithmetic (Q16.16) for zk-friendly Q-values
- Example:
  ```python
  from idi.zk.witness_generator import generate_witness_from_q_table
  
  q_table = {"state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0}}
  witness = generate_witness_from_q_table(q_table, "state_0", use_merkle=True)
  ```

## Integrating Risc0 (example)
```
export IDI_RISC0_CMD='risc0 prove --elf idi/risc0/idi_prover.elf --manifest {manifest} --streams {streams} --output {output}'
python - <<'PY'
from pathlib import Path
from idi.zk.proof_manager import generate_proof
import os

generate_proof(
    manifest_path=Path("build/artifacts/sample/artifact_manifest.json"),
    stream_dir=Path("build/artifacts/sample/streams"),
    out_dir=Path("build/proofs/sample_risc0"),
    prover_command=os.environ["IDI_RISC0_CMD"],
)
PY
```

## Verifier hand-off

### Integration with TauBridge

The `idi/taunet_bridge/` module provides integration with [Tau Testnet](https://github.com/IDNI/tau-testnet):

```python
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext

# Configure ZK verification
config = ZkConfig(enabled=True, proof_system="risc0")
adapter = TauNetZkAdapter(config)

# Create validation step
zk_step = ZkValidationStep(adapter, required=False)

# Validate transaction with ZK proof
ctx = ValidationContext(tx_hash="...", payload={"zk_proof": proof_bundle})
zk_step.run(ctx)  # Raises InvalidZkProofError if verification fails
```

### End-to-End Workflow

1. **Train agent privately** - Q-table stays on user's machine
2. **Generate artifact manifest** - No Q-values included
3. **Create Tau input streams** - Only binary action signals
4. **Generate witness** - Merkle root commitment (Q-values private)
5. **Generate Risc0 proof** - Cryptographically verifiable proof
6. **Submit to Tau Net** - Via TauBridge integration

See [`PRIVATE_TRAINING_GUIDE.md`](PRIVATE_TRAINING_GUIDE.md) for complete workflow.

### Privacy Guarantees

✅ Q-values never exposed in any artifact  
✅ Only Merkle root commitments revealed  
✅ Proofs verifiable without Q-values  
✅ Tamper detection (any modification invalidates proof)  
✅ Complete test coverage verifying privacy

