"""End-to-end tests covering Q-table witness -> proof -> verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from idi.zk.proof_manager import generate_proof, verify_proof
from idi.zk.witness_generator import generate_witness_from_q_table


def _rehash_path(leaf_hash: bytes, proof_path, root_hash: bytes) -> bool:
    """Replay Merkle path to check membership."""
    current = leaf_hash
    for sibling, is_right in proof_path:
        current = hashlib.sha256(
            (current + sibling) if is_right else (sibling + current)
        ).digest()
    return current == root_hash


def test_stub_proof_roundtrip_detects_tamper(tmp_path: Path) -> None:
    """Generate stub proof then detect tampering of streams/manifest."""
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n0\n")
    (streams / "q_sell.in").write_text("0\n1\n")

    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "agent_stub",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {"episodes": 2},
        "policy_summary": {"states": 2, "actions": ["hold", "buy", "sell"]},
        "trace_summary": {"length": 2, "stream_hashes": {}},
        "proof_policy": "stub",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    proof_dir = tmp_path / "proof"
    bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams,
        out_dir=proof_dir,
        prover_command=None,
    )

    assert verify_proof(bundle) is True

    # Tamper with stream content -> verification must fail
    (streams / "q_buy.in").write_text("9\n9\n")
    assert verify_proof(bundle) is False

    # Restore stream and tamper manifest -> verification must fail
    (streams / "q_buy.in").write_text("1\n0\n")
    manifest["artifact_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert verify_proof(bundle) is False


def test_merkle_witness_matches_root(tmp_path: Path) -> None:
    """Large Q-table witness includes Merkle proof that matches root."""
    q_table = {
        f"state_{i}": {"hold": 0.1, "buy": 0.2 + (i % 3) * 0.1, "sell": 0.0}
        for i in range(128)
    }
    state_key = "state_42"
    witness = generate_witness_from_q_table(q_table, state_key, use_merkle=True)

    assert witness.merkle_proof is not None
    proof = witness.merkle_proof

    # Leaf hash must be 32 bytes and path must resolve to recorded root
    assert len(proof.leaf_hash) == 32
    assert _rehash_path(proof.leaf_hash, proof.path, proof.root_hash)
    # Witness root must equal Merkle proof root
    assert witness.q_table_root == proof.root_hash

    # Action selection is deterministic (greedy) for the chosen state
    assert witness.selected_action in (0, 1, 2)
