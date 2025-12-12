"""Golden/snapshot tests for deterministic RL behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idi_iann import TrainingConfig, create_trainer


@pytest.fixture
def golden_dir() -> Path:
    """Directory for golden test files."""
    return Path(__file__).parent / "golden"


def test_golden_training_run(golden_dir: Path):
    """Test that training produces consistent outputs with fixed seed."""
    config = TrainingConfig(episodes=4, episode_length=8)
    trainer = create_trainer(config, seed=42)
    policy, trace = trainer.run()

    # Load or create golden file
    golden_file = golden_dir / "training_run_seed42.json"
    golden_dir.mkdir(exist_ok=True)

    # Extract key metrics
    stats = trainer.stats()
    trace_summary = {
        "num_ticks": len(trace.ticks),
        "first_tick": trace.ticks[0] if trace.ticks else {},
        "last_tick": trace.ticks[-1] if trace.ticks else {},
        "mean_reward": stats["mean_reward"],
    }

    if golden_file.exists():
        # Compare against golden
        with golden_file.open() as f:
            golden = json.load(f)
        assert trace_summary["num_ticks"] == golden["num_ticks"]
        assert abs(trace_summary["mean_reward"] - golden["mean_reward"]) < 0.01
    else:
        # Create golden file (first run)
        with golden_file.open("w") as f:
            json.dump(trace_summary, f, indent=2)
        pytest.skip("Golden file created; re-run to verify")


def test_deterministic_with_seed():
    """Test that same seed produces same results."""
    config = TrainingConfig(episodes=2, episode_length=4)
    trainer1 = create_trainer(config, seed=123)
    policy1, trace1 = trainer1.run()

    trainer2 = create_trainer(config, seed=123)
    policy2, trace2 = trainer2.run()

    # Traces should be identical
    assert len(trace1.ticks) == len(trace2.ticks)
    assert trace1.ticks == trace2.ticks

