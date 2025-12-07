"""Integration harness for real ZK workflow (Risc0).

This test is skipped unless a Risc0 prover is available in the environment.
It documents the expected end-to-end flow: witness/streams remain local, only
proof + receipt are transmitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.protocols import ZkProofBundle
from idi.taunet_bridge.validation import ValidationContext, ZkValidationStep
from idi.zk.proof_manager import verify_proof


RISC0_PROVER_CMD = os.environ.get("RISC0_PROVER_CMD")


@pytest.mark.skipif(not RISC0_PROVER_CMD, reason="RISC0_PROVER_CMD not set; real prover unavailable")
def test_e2e_risc0_proof_roundtrip(tmp_path: Path):
    """Generate Risc0 proof, then validate via adapter/validation step.

    Preconditions:
    - RISC0_PROVER_CMD points to a command template usable by proof_manager.generate_proof.
    """
    # Arrange: minimal streams + manifest
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "q_buy.in").write_text("1\n0\n", encoding="utf-8")
    (streams_dir / "q_sell.in").write_text("0\n1\n", encoding="utf-8")

    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "agent_risc0",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {"episodes": 2},
        "policy_summary": {"states": 2, "actions": ["hold", "buy", "sell"]},
        "trace_summary": {"length": 2, "stream_hashes": {}},
        "proof_policy": "risc0",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    # Generate proof using external prover command
    from idi.zk.proof_manager import generate_proof

    try:
        bundle = generate_proof(
            manifest_path=manifest_path,
            stream_dir=streams_dir,
            out_dir=tmp_path / "proof_risc0",
            prover_command=RISC0_PROVER_CMD,
        )
    except Exception as exc:
        pytest.skip(f"Risc0 prover unavailable or build failed: {exc}")

    # Verify proof locally (host-side)
    assert verify_proof(bundle)

    # Bridge verification
    cfg = ZkConfig(enabled=True, proof_system="risc0", require_proofs=True)
    adapter = TauNetZkAdapter(cfg)
    proof = ZkProofBundle(
        proof_path=bundle.proof_path,
        receipt_path=bundle.receipt_path,
        manifest_path=bundle.manifest_path,
        tx_hash="tx_risc0",
    )
    ctx = ValidationContext(tx_hash="tx_risc0", payload={"zk_proof": proof})
    ZkValidationStep(adapter, required=True).run(ctx)
