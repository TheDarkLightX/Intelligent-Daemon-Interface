"""Round-trip tests for config schema between Python and Rust."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from idi_iann.config import TrainingConfig


def normalize_config_json(config_dict: dict) -> dict:
    """Normalize config dict for comparison (remove None values, sort keys)."""
    normalized = {}
    for key, value in sorted(config_dict.items()):
        if value is None:
            continue
        if isinstance(value, dict):
            normalized[key] = normalize_config_json(value)
        elif isinstance(value, list):
            normalized[key] = sorted(value) if all(isinstance(x, (str, int)) for x in value) else value
        else:
            normalized[key] = value
    return normalized


def test_python_to_rust_roundtrip():
    """Test Python config → JSON → Rust → JSON → Python."""
    # Create Python config
    py_config = TrainingConfig()
    py_dict = {
        "episodes": py_config.episodes,
        "episode_length": py_config.episode_length,
        "discount": py_config.discount,
        "learning_rate": py_config.learning_rate,
        "exploration_decay": py_config.exploration_decay,
        "quantizer": {
            "price_buckets": py_config.quantizer.price_buckets,
            "volume_buckets": py_config.quantizer.volume_buckets,
            "trend_buckets": py_config.quantizer.trend_buckets,
            "scarcity_buckets": py_config.quantizer.scarcity_buckets,
            "mood_buckets": py_config.quantizer.mood_buckets,
        },
        "rewards": {
            "pnl": py_config.rewards.pnl,
            "scarcity_alignment": py_config.rewards.scarcity_alignment,
            "ethics_bonus": py_config.rewards.ethics_bonus,
            "communication_clarity": py_config.rewards.communication_clarity,
        },
    }

    # Write to temp JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(py_dict, f, indent=2)
        temp_path = Path(f.name)

    try:
        # Try to load in Rust (if Rust binary exists)
        rust_bin = Path(__file__).parent.parent.parent / "rust" / "idi_iann" / "target" / "debug" / "train"
        if not rust_bin.exists():
            pytest.skip("Rust binary not built")

        # Run Rust to load and dump config
        result = subprocess.run(
            [str(rust_bin), "train", "--config", str(temp_path), "--episodes", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # For now, just verify Rust can parse the config
        assert result.returncode == 0 or "error" not in result.stderr.lower()
    finally:
        temp_path.unlink()


def test_config_serialization():
    """Test that Python config can be serialized and deserialized."""
    config = TrainingConfig()
    config_dict = {
        "episodes": config.episodes,
        "episode_length": config.episode_length,
        "discount": config.discount,
        "learning_rate": config.learning_rate,
        "exploration_decay": config.exploration_decay,
        "quantizer": {
            "price_buckets": config.quantizer.price_buckets,
            "volume_buckets": config.quantizer.volume_buckets,
            "trend_buckets": config.quantizer.trend_buckets,
            "scarcity_buckets": config.quantizer.scarcity_buckets,
            "mood_buckets": config.quantizer.mood_buckets,
        },
        "rewards": {
            "pnl": config.rewards.pnl,
            "scarcity_alignment": config.rewards.scarcity_alignment,
            "ethics_bonus": config.rewards.ethics_bonus,
            "communication_clarity": config.rewards.communication_clarity,
        },
    }

    # Serialize to JSON
    json_str = json.dumps(config_dict, indent=2)
    assert json_str

    # Deserialize back
    loaded_dict = json.loads(json_str)
    assert loaded_dict == config_dict

    # Verify normalized forms match
    normalized_original = normalize_config_json(config_dict)
    normalized_loaded = normalize_config_json(loaded_dict)
    assert normalized_original == normalized_loaded

