# ZK Proof Integration Guide

## Quick Start

### 1. Install Risc0 Toolchain

```bash
# Install Risc0 toolchain
curl -L https://risc0.github.io/gh-release/dev/risc0-rust-toolchain.toml -o risc0-rust-toolchain.toml
rustup toolchain install --path risc0-rust-toolchain.toml
```

### 2. Build Guest Programs

```bash
cd idi/zk/risc0
cargo build --release -p idi_risc0_methods
```

This compiles both guest programs:
- `idi-manifest`: Manifest verification
- `idi-qtable`: Q-table action selection verification

### 3. Generate Proofs

```bash
# Using Python wrapper
python -m idi.zk.run_risc0_proofs \
    --artifacts-root idi/artifacts

# Or directly with cargo
cd idi/zk/risc0
cargo run --release -p idi_risc0_host -- \
    --manifest idi/artifacts/my_agent/artifact_manifest.json \
    --streams idi/artifacts/my_agent/streams \
    --proof idi/artifacts/my_agent/proof_risc0/proof.bin \
    --receipt idi/artifacts/my_agent/proof_risc0/receipt.json
```

## Integration Points

### Training Pipeline Integration

```python
from idi.zk.training_integration import generate_proofs_from_training_output

# After training completes
proofs = generate_proofs_from_training_output(
    q_table_path=Path("artifacts/my_agent/q_table.json"),
    manifest_path=Path("artifacts/my_agent/artifact_manifest.json"),
    stream_dir=Path("artifacts/my_agent/streams"),
    out_dir=Path("artifacts/my_agent/proofs"),
    use_merkle=True,
)
```

### Tau Execution Integration

```python
from idi.zk.tau_integration import verify_before_tau_execution, execute_tau_with_proof_verification

# Before executing Tau spec
if verify_before_tau_execution(
    manifest_path=manifest_path,
    proof_bundle=proof_bundle,
    tau_spec_path=tau_spec_path,
    inputs_dir=inputs_dir,
):
    # Execute Tau
    success, output = execute_tau_with_proof_verification(
        tau_spec_path=tau_spec_path,
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        proof_bundle=proof_bundle,
    )
```

## Witness Generation

### Small Q-Tables (<100 entries)

```python
from idi.zk.witness_generator import generate_witness_from_q_table

witness = generate_witness_from_q_table(
    q_table=q_table,
    state_key="state_0",
    use_merkle=False,  # In-memory hashing
)
```

### Large Q-Tables (>100 entries)

```python
witness = generate_witness_from_q_table(
    q_table=q_table,
    state_key="state_0",
    use_merkle=True,  # Merkle tree commitments
)
```

## Proof Verification

### Verify Manifest Proof

```python
from idi.zk.proof_manager import verify_proof

verified = verify_proof(bundle)
assert verified, "Proof verification failed"
```

### Verify Q-Table Proof

```python
from idi.zk.qtable_prover import verify_qtable_proof

verified = verify_qtable_proof(
    proof_path=proof_path,
    receipt_path=receipt_path,
    expected_q_root=q_table_root,
)
```

## Privacy Features

1. **Q-Table Privacy**: Only Merkle root commitments are public
2. **Training Privacy**: Training techniques remain private
3. **Action Verification**: Proofs verify correctness without revealing Q-values
4. **Selective Disclosure**: Can prove specific properties without full disclosure

## Performance

- **Small tables**: ~100ms proof generation
- **Large tables**: ~1-5s proof generation (depends on table size)
- **Verification**: <10ms (very fast)
- **Merkle proofs**: O(log n) size for n entries

## Troubleshooting

### Guest Program Compilation Errors

- Ensure Risc0 toolchain is installed: `rzup install`
- Check Rust version: `rustc --version` (should be from Risc0 toolchain)
- Clean build: `cargo clean && cargo build`

### Proof Verification Failures

- Check manifest and stream directory match
- Verify receipt timestamp is recent
- Ensure Q-table root matches expected value

### Integration Issues

- Verify all paths are absolute or relative to correct directory
- Check file permissions
- Ensure Tau binary is available for execution

