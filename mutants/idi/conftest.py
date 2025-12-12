"""Shared pytest fixtures for IDI tests.

This module provides common fixtures used across all IDI test modules,
reducing duplication and ensuring consistent test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def sample_qtable() -> Dict[str, Dict[str, float]]:
    """Minimal Q-table for testing.
    
    Returns a Q-table with:
    - state_0: clear winner (buy)
    - state_1: tie case (all equal)
    - state_2: negative values
    """
    return {
        "state_0": {"hold": 0.0, "buy": 0.5, "sell": -0.25},
        "state_1": {"hold": 0.1, "buy": 0.1, "sell": 0.1},  # Tie: buy wins
        "state_2": {"hold": -0.5, "buy": -0.3, "sell": -0.1},  # Sell wins (least negative)
    }


@pytest.fixture
def sample_manifest(tmp_path: Path) -> Path:
    """Valid manifest file for testing."""
    manifest = {
        "version": "1.0",
        "type": "test",
        "artifact_id": "test-artifact-001",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def sample_streams(tmp_path: Path) -> Path:
    """Stream directory with test files."""
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "action.in").write_text("1\n0\n2\n", encoding="utf-8")
    (streams / "reward.in").write_text("0.5\n-0.1\n0.3\n", encoding="utf-8")
    return streams


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Sample training config for testing."""
    config = {
        "episodes": 10,
        "learning_rate": 0.01,
        "gamma": 0.99,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def zk_test_dir(tmp_path: Path, sample_manifest: Path, sample_streams: Path) -> Path:
    """Complete test directory structure for ZK tests.
    
    Creates:
        tmp_path/
        ├── manifest.json
        ├── streams/
        │   ├── action.in
        │   └── reward.in
        └── proofs/
    """
    # Manifest and streams already created by fixtures
    proofs_dir = tmp_path / "proofs"
    proofs_dir.mkdir()
    return tmp_path
