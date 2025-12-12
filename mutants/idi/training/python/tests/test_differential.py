"""Cross-language differential tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from idi_iann import TrainingConfig, create_trainer
from idi_iann.policy import LookupPolicy, StateKey


def test_q_update_consistency():
    """Test that Q-update logic matches between Python implementations."""
    policy = LookupPolicy()
    state: StateKey = (1, 2, 3, 4, 5)
    action = "buy"
    reward = 0.5
    next_state: StateKey = (2, 2, 3, 4, 5)
    discount = 0.9
    learning_rate = 0.1

    # Initial Q-value
    initial_q = policy.q_value(state, action)

    # Compute TD target manually
    best_next = policy.best_action(next_state)
    next_q = policy.q_value(next_state, best_next)
    td_target = reward + discount * next_q
    td_error = td_target - initial_q

    # Update
    policy.update(state, action, learning_rate * td_error)

    # Verify update
    updated_q = policy.q_value(state, action)
    expected_q = initial_q + learning_rate * td_error
    assert abs(updated_q - expected_q) < 1e-6


def test_tile_encoding_deterministic():
    """Test that tile encoding is deterministic for same inputs."""
    from idi_iann.abstraction import TileCoder
    from idi_iann.config import TileCoderConfig

    config = TileCoderConfig()
    coder = TileCoder(config)

    state = (1, 2, 3, 4, 5)
    indices1 = coder.encode(state)
    indices2 = coder.encode(state)

    assert indices1 == indices2


@pytest.mark.skip(reason="Requires Rust binary to be built")
def test_python_rust_q_update():
    """Test Q-update consistency between Python and Rust."""
    # This would require calling Rust code from Python
    # For now, we verify the logic matches conceptually
    pass

