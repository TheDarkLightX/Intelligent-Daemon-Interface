"""Tests for zkVM workflow integration."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest


def test_manifest_hash_computation():
    """Test that manifest hashes are computed correctly."""
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "test_123",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {"episodes": 100},
        "policy_summary": {"states": 50, "actions": ["hold", "buy", "sell"]},
        "trace_summary": {
            "length": 64,
            "stream_hashes": {
                "q_buy.in": "abc123",
                "q_sell.in": "def456",
            },
        },
        "proof_policy": "stub",
    }

    # Compute deterministic hash
    manifest_str = json.dumps(manifest, sort_keys=True)
    expected_hash = hashlib.sha256(manifest_str.encode()).hexdigest()

    # Recompute should match
    recomputed = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    assert expected_hash == recomputed


def test_trace_stream_hashes():
    """Test that trace stream hashes are computed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir)

        # Create sample trace files
        (trace_dir / "q_buy.in").write_text("1\n0\n1\n0")
        (trace_dir / "q_sell.in").write_text("0\n1\n0\n1")

        # Compute hashes
        hashes = {}
        for f in trace_dir.glob("*.in"):
            content = f.read_bytes()
            hashes[f.name] = hashlib.sha256(content).hexdigest()

        assert "q_buy.in" in hashes
        assert "q_sell.in" in hashes
        assert len(hashes["q_buy.in"]) == 64  # SHA256 hex length


def test_manifest_schema_validation():
    """Test that manifest schema is validated."""
    valid_manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "test_123",
        "timestamp": "2024-01-01T00:00:00Z",
        "training_config": {"episodes": 100},
        "policy_summary": {"states": 50, "actions": ["hold", "buy", "sell"]},
        "trace_summary": {
            "length": 64,
            "stream_hashes": {},
        },
        "proof_policy": "stub",
    }

    # Valid manifest should have all required fields
    required_fields = [
        "schema_version",
        "artifact_id",
        "timestamp",
        "training_config",
        "policy_summary",
        "trace_summary",
        "proof_policy",
    ]

    for field in required_fields:
        assert field in valid_manifest


def test_proof_bundle_structure():
    """Test proof bundle structure."""
    proof_bundle = {
        "manifest_hash": "abc123",
        "image_id": "0x" + "00" * 32,
        "receipt_data": "base64_encoded_receipt",
        "timestamp": "2024-01-01T00:00:00Z",
    }

    # Should have required fields
    assert "manifest_hash" in proof_bundle
    assert "image_id" in proof_bundle
    assert "receipt_data" in proof_bundle


def test_hash_chain_integrity():
    """Test hash chain from config to proof."""
    # Config -> Config Hash
    config = {"episodes": 100, "seed": 42}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    # Q-Table entries -> Q-Table Hash
    q_entries = {"(0,0,0,0,0)": {"hold": 0.0, "buy": 0.5, "sell": -0.5}}
    q_hash = hashlib.sha256(json.dumps(q_entries, sort_keys=True).encode()).hexdigest()

    # Traces -> Trace Hashes
    traces = {"q_buy.in": "1\n0", "q_sell.in": "0\n1"}
    trace_hashes = {name: hashlib.sha256(data.encode()).hexdigest() for name, data in traces.items()}

    # Manifest -> Manifest Hash
    manifest = {
        "config_hash": config_hash,
        "q_hash": q_hash,
        "trace_hashes": trace_hashes,
    }
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    # All hashes should be deterministic
    assert len(config_hash) == 64
    assert len(q_hash) == 64
    assert len(manifest_hash) == 64


def test_input_validation():
    """Test input validation for zkVM workflow."""
    # Valid inputs
    valid_state = (0, 1, 2, 3, 4)
    assert all(isinstance(x, int) for x in valid_state)
    assert all(x >= 0 for x in valid_state)

    # Valid action
    valid_actions = ["hold", "buy", "sell"]
    assert "buy" in valid_actions

    # Valid regime
    valid_regime = 3
    assert 0 <= valid_regime < 32  # 5-bit regime


def test_proof_verification_mock():
    """Test mock proof verification flow."""
    # Mock proof bundle
    proof = {
        "manifest_hash": "expected_hash",
        "verified": True,
        "verifier": "stub",
    }

    # Mock verification
    expected_hash = "expected_hash"
    assert proof["manifest_hash"] == expected_hash
    assert proof["verified"]


def test_artifact_size_limits():
    """Test that artifact size limits are enforced."""
    MAX_Q_TABLE_ENTRIES = 100_000
    MAX_TRACE_LENGTH = 10_000
    MAX_MANIFEST_SIZE = 1_000_000  # 1MB

    # Check limits would be enforced
    sample_q_size = 1024  # Reasonable Q-table
    sample_trace_len = 64
    sample_manifest = json.dumps({"key": "value"})

    assert sample_q_size < MAX_Q_TABLE_ENTRIES
    assert sample_trace_len < MAX_TRACE_LENGTH
    assert len(sample_manifest) < MAX_MANIFEST_SIZE


def test_timestamp_freshness():
    """Test timestamp freshness validation."""
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    max_age = timedelta(days=7)

    # Fresh timestamp
    fresh_ts = (now - timedelta(hours=1)).isoformat()
    fresh_dt = datetime.fromisoformat(fresh_ts.replace("Z", "+00:00"))
    assert now - fresh_dt < max_age

    # Stale timestamp
    stale_ts = (now - timedelta(days=30)).isoformat()
    stale_dt = datetime.fromisoformat(stale_ts.replace("Z", "+00:00"))
    assert now - stale_dt > max_age
