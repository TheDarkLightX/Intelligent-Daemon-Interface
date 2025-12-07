"""Standardized workflow for training → manifest → proof → Tau spec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

# Add training Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "python"))

from .proof_manager import ProofBundle, generate_proof, verify_proof
from .spec_generator import TauSpecGenerator, generate_spec_from_config


@dataclass
class WorkflowResult:
    """Result of end-to-end workflow execution."""

    artifact_dir: Path
    manifest_path: Path
    proof_bundle: Optional[ProofBundle]
    spec_path: Optional[Path]
    success: bool
    errors: list[str]


def run_training_to_proof_workflow(
    *,
    config_path: Path,
    artifact_dir: Path,
    prover_command: Optional[str] = None,
    generate_spec: bool = True,
    spec_type: str = "v38",
) -> WorkflowResult:
    """Run complete workflow: training → manifest → proof → spec.

    Args:
        config_path: Path to training config JSON
        artifact_dir: Directory for artifacts (manifests, streams, proofs)
        prover_command: Optional command template for external prover
        generate_spec: Whether to generate Tau spec
        spec_type: Type of spec to generate ("v38" or "layered")

    Returns:
        WorkflowResult with paths and status
    """
    errors: list[str] = []
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Training (assumes already done, streams exist)
    streams_dir = artifact_dir / "streams"
    manifest_path = artifact_dir / "artifact_manifest.json"

    if not manifest_path.exists():
        errors.append(f"Manifest not found: {manifest_path}")
        return WorkflowResult(
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            proof_bundle=None,
            spec_path=None,
            success=False,
            errors=errors,
        )

    if not streams_dir.exists():
        errors.append(f"Streams directory not found: {streams_dir}")
        return WorkflowResult(
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            proof_bundle=None,
            spec_path=None,
            success=False,
            errors=errors,
        )

    # Step 2: Generate proof
    proof_dir = artifact_dir / "proof_risc0"
    proof_bundle: Optional[ProofBundle] = None
    try:
        proof_bundle = generate_proof(
            manifest_path=manifest_path,
            stream_dir=streams_dir,
            out_dir=proof_dir,
            prover_command=prover_command,
        )
    except Exception as e:
        errors.append(f"Proof generation failed: {e}")

    # Step 3: Verify proof
    if proof_bundle:
        try:
            if not verify_proof(proof_bundle):
                errors.append("Proof verification failed")
        except Exception as e:
            errors.append(f"Proof verification error: {e}")

    # Step 4: Generate Tau spec
    spec_path: Optional[Path] = None
    if generate_spec:
        try:
            spec_path = artifact_dir / f"agent_{spec_type}.tau"
            generate_spec_from_config(config_path, spec_path, spec_type=spec_type)
        except Exception as e:
            errors.append(f"Spec generation failed: {e}")

    success = len(errors) == 0
    return WorkflowResult(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        proof_bundle=proof_bundle,
        spec_path=spec_path,
        success=success,
        errors=errors,
    )


def document_workflow(output_path: Path) -> None:
    """Write workflow documentation to file.

    Args:
        output_path: Path to write documentation
    """
    doc = """# IDI Training → Proof → Tau Spec Workflow

## Overview

Standardized workflow for generating verifiable intelligent agents:

1. **Training**: Generate Q-tables and traces using Python/Rust trainers
2. **Manifest**: Create artifact manifest with stream hashes
3. **Proof**: Generate zk proof bundle (Risc0 or stub)
4. **Spec**: Generate Tau-language spec from config
5. **Verification**: Verify proof and spec consistency

## Workflow Steps

### Step 1: Training
```bash
python -m idi.training.python.run_idi_trainer \\
    --config config.json \\
    --out artifacts/my_agent/streams
```

### Step 2: Manifest Generation
```bash
python -m idi.devkit.builder \\
    --config config.json \\
    --out artifacts/my_agent \\
    --install-inputs specs/V38_Minimal_Core/inputs
```

### Step 3: Proof Generation
```bash
python -m idi.zk.run_risc0_proofs \\
    --manifest artifacts/my_agent/artifact_manifest.json \\
    --streams artifacts/my_agent/streams \\
    --proof artifacts/my_agent/proof_risc0/proof.bin \\
    --receipt artifacts/my_agent/proof_risc0/receipt.json
```

### Step 4: Spec Generation
```bash
python -c "from idi.zk.spec_generator import generate_spec_from_config; \\
    generate_spec_from_config('config.json', 'specs/my_agent.tau', 'v38')"
```

### Step 5: Verification
```bash
python -c "from idi.zk.workflow import run_training_to_proof_workflow; \\
    result = run_training_to_proof_workflow( \\
        config_path='config.json', \\
        artifact_dir='artifacts/my_agent' \\
    ); \\
    print('Success:', result.success)"
```

## File Structure

```
artifacts/my_agent/
├── artifact_manifest.json      # Stream hashes + metadata
├── streams/                    # Tau input streams
│   ├── q_buy.in
│   ├── q_sell.in
│   └── ...
├── proof_risc0/               # Proof bundle
│   ├── proof.bin
│   └── receipt.json
└── agent_v38.tau              # Generated spec
```

## Consistency Checks

- Manifest hash matches stream directory contents
- Proof receipt digest matches host-computed hash
- Spec inputs match stream names
- All streams have corresponding Tau spec inputs/outputs
"""
    output_path.write_text(doc, encoding="utf-8")

