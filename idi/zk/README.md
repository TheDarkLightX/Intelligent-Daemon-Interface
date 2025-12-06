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

