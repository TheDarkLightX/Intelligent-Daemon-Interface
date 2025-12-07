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
The stub uses SHA-256 commitments to emulate a proof so downstream automation (Tau daemon, ledger, etc.) can proceed even before the real zkVM integration is wired up.

Batch all manifests under `idi/artifacts/` with:
```
python -m idi.zk.run_stub_proofs --artifacts-root idi/artifacts
```
This creates `proof_stub/` folders next to each artifact (regime layers, emotive layers, etc.) and verifies the receipts in one go.

## Native Risc0 prover (alpha)

- Workspace lives in `idi/zk/risc0/` (`cargo` workspace with `host/`, `methods/`, and generated guest code).
- Build: `cargo run --release -p idi_risc0_host -- --help` (requires `rzup install` + the `riscv32im-risc0-zkvm-elf` toolchain).
- Usage example:
  ```
  cargo run --release -p idi_risc0_host -- \
      --manifest idi/artifacts/regime_macro/artifact_manifest.json \
      --streams idi/artifacts/regime_macro/streams \
      --proof idi/artifacts/regime_macro/proof_risc0/proof.bin \
      --receipt idi/artifacts/regime_macro/proof_risc0/receipt.json
  ```
- The host consumes the manifest + streams, embeds them into the guest, proves via Risc0, checks the guest digest against a deterministic host hash, and writes both the binary proof and a JSON receipt (including the method ID and digest).

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
- Store `artifact_manifest.json`, proof binary, and `receipt.json` in the Tau ledger.
- Tau daemon should only accept `q_*` streams when the receipt digest matches the expected manifest hash.
- On-chain deployments can re-run `verify_proof` (or the actual Risc0 verifier) before actioning trades.

