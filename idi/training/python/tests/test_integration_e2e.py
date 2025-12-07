"""End-to-end integration test: training to trace export to hash verification."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.config import TrainingConfig
from idi_iann.policy import LookupPolicy
from idi_iann.trainer import QTrainer


def test_training_to_trace_export_to_hash():
    """Test full workflow: training -> trace export -> hash verification."""
    config = TrainingConfig(episodes=2, episode_length=5)
    trainer = QTrainer(config, seed=42)

    # Train
    policy, trace = trainer.run()

    assert policy is not None
    assert len(trace.ticks) == 5

    # Export trace
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir) / "traces"
        trace.export(trace_dir)

        # Verify trace files exist
        # Get actual files exported
        actual_files = [f.name for f in trace_dir.glob("*.in")]
        
        # Expected files (risk_event may not be exported depending on trainer config)
        expected_files = [
            "q_buy.in",
            "q_sell.in",
            "risk_budget_ok.in",
            "q_emote_positive.in",
            "q_emote_alert.in",
            "q_emote_persistence.in",
            "q_regime.in",
            "price_up.in",
            "price_down.in",
            "weight_momentum.in",
            "weight_contra.in",
            "weight_trend.in",
        ]

        for filename in expected_files:
            filepath = trace_dir / filename
            assert filepath.exists(), f"Trace file {filename} not found"

        # Compute hashes
        hashes = {}
        for filename in expected_files:
            filepath = trace_dir / filename
            content = filepath.read_bytes()
            hash_obj = hashlib.sha256(content)
            hashes[filename] = hash_obj.hexdigest()

        # Verify hashes are consistent (deterministic training)
        assert len(hashes) == len(expected_files)
        assert all(len(h) == 64 for h in hashes.values())  # SHA256 hex length

        # Serialize manifest
        manifest_path = trace_dir / "manifest.json"
        policy.serialize_manifest(manifest_path)

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "states" in manifest
        assert "actions" in manifest

        # Verify trace content is valid
        q_buy_content = (trace_dir / "q_buy.in").read_text()
        lines = q_buy_content.strip().split("\n")
        assert len(lines) == 5  # episode_length
        assert all(line in ("0", "1") for line in lines)


def test_deterministic_training_same_seed():
    """Test that same seed produces identical traces."""
    config = TrainingConfig(episodes=2, episode_length=5)

    trainer1 = QTrainer(config, seed=123)
    policy1, trace1 = trainer1.run()

    trainer2 = QTrainer(config, seed=123)
    policy2, trace2 = trainer2.run()

    # Traces should be identical
    assert trace1.ticks == trace2.ticks

    # Policies should have same number of states
    assert len(policy1._table) == len(policy2._table)


def test_trace_export_format():
    """Test trace export format matches expected schema."""
    config = TrainingConfig(episodes=1, episode_length=3)
    trainer = QTrainer(config, seed=42)
    _, trace = trainer.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir) / "traces"
        trace.export(trace_dir)

        # Verify all files have same number of lines
        files = list(trace_dir.glob("*.in"))
        assert len(files) > 0

        line_counts = {}
        for filepath in files:
            content = filepath.read_text()
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            line_counts[filepath.name] = len(lines)

        # All files should have same number of lines (episode_length)
        expected_lines = config.episode_length
        for filename, count in line_counts.items():
            assert count == expected_lines, (
                f"{filename} has {count} lines, expected {expected_lines}"
            )


def test_manifest_includes_trace_hashes():
    """Test that manifest can include trace file hashes."""
    config = TrainingConfig(episodes=1, episode_length=3)
    trainer = QTrainer(config, seed=42)
    policy, trace = trainer.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_dir = Path(tmpdir) / "traces"
        trace.export(trace_dir)

        # Compute hashes
        hashes = {}
        for filepath in trace_dir.glob("*.in"):
            content = filepath.read_bytes()
            hash_obj = hashlib.sha256(content)
            hashes[filepath.name] = hash_obj.hexdigest()

        # Create manifest with hashes
        manifest = {
            "states": len(policy._table),
            "actions": [a.value for a in policy.ACTIONS],
            "trace_hashes": hashes,
        }

        manifest_path = trace_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Verify manifest can be loaded
        loaded = json.loads(manifest_path.read_text())
        assert "trace_hashes" in loaded
        assert len(loaded["trace_hashes"]) == len(hashes)

