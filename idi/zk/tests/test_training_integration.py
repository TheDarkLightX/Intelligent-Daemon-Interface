"""Tests for training-ZK integration."""

import json
from pathlib import Path

import pytest

from idi.zk.training_integration import (
    generate_proofs_from_training_output,
    verify_training_proofs,
)


def test_generate_proofs_from_training(tmp_path: Path):
    """Test proof generation from training output."""
    # Create mock Q-table
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
        "state_1": {"hold": 0.0, "buy": 0.0, "sell": 0.5},
    }
    q_table_path = tmp_path / "q_table.json"
    q_table_path.write_text(json.dumps({"q_table": q_table}))
    
    # Create mock manifest
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"artifact_id": "test"}))
    
    # Create mock streams
    stream_dir = tmp_path / "streams"
    stream_dir.mkdir()
    (stream_dir / "q_buy.in").write_text("1\n")
    
    # Generate proofs
    out_dir = tmp_path / "proofs"
    bundles = generate_proofs_from_training_output(
        q_table_path=q_table_path,
        manifest_path=manifest_path,
        stream_dir=stream_dir,
        out_dir=out_dir,
        use_merkle=False,
    )
    
    assert "manifest" in bundles
    assert len(bundles) > 1  # Should have Q-table proofs


def test_verify_training_proofs(tmp_path: Path):
    """Test verification of training proofs."""
    # Create mock Q-table
    q_table = {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": 0.0},
    }
    q_table_path = tmp_path / "q_table.json"
    q_table_path.write_text(json.dumps({"q_table": q_table}))
    
    # Create mock proof bundles (simplified)
    from idi.zk.proof_manager import ProofBundle
    
    bundles = {
        "manifest": ProofBundle(
            manifest_path=tmp_path / "manifest.json",
            proof_path=tmp_path / "proof.bin",
            receipt_path=tmp_path / "receipt.json",
        ),
    }
    
    # Create mock receipt (must match proof_manager format)
    (tmp_path / "receipt.json").write_text(
        json.dumps({
            "digest": "test",
            "timestamp": 0,
            "manifest": str(tmp_path / "manifest.json"),
            "streams": str(tmp_path / "streams"),
            "proof": str(tmp_path / "proof.bin"),
        })
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"test": "data"}))
    (tmp_path / "streams").mkdir()
    (tmp_path / "streams" / "test.in").write_text("1\n")
    (tmp_path / "proof.bin").write_bytes(b"test")
    
    # Verify (will use stub verification)
    results = verify_training_proofs(bundles, q_table_path)
    
    assert "manifest" in results

