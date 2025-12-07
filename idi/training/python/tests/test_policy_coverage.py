"""Tests for policy.py methods with missing coverage."""

import json
import tempfile
from pathlib import Path

import pytest

from idi_iann.policy import LookupPolicy
from idi_iann.domain import Action


def test_export_trace_multiple_episodes():
    """Test export_trace with multiple episodes."""
    policy = LookupPolicy()

    episodes = [
        [
            {
                "q_buy": 1,
                "q_sell": 0,
                "risk_budget_ok": 1,
                "q_emote_positive": 1,
                "q_emote_alert": 0,
                "q_emote_persistence": 0,
                "q_regime": 2,
            },
            {
                "q_buy": 0,
                "q_sell": 1,
                "risk_budget_ok": 1,
                "q_emote_positive": 0,
                "q_emote_alert": 1,
                "q_emote_persistence": 1,
                "q_regime": 3,
            },
        ],
        [
            {
                "q_buy": 0,
                "q_sell": 0,
                "risk_budget_ok": 1,
                "q_emote_positive": 0,
                "q_emote_alert": 0,
                "q_emote_persistence": 0,
                "q_regime": 1,
            },
        ],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir) / "traces"
        policy.export_trace(episodes, target_dir)

        # Check all stream files exist
        assert (target_dir / "q_buy.in").exists()
        assert (target_dir / "q_sell.in").exists()
        assert (target_dir / "risk_budget_ok.in").exists()
        assert (target_dir / "q_emote_positive.in").exists()
        assert (target_dir / "q_emote_alert.in").exists()
        assert (target_dir / "q_emote_persistence.in").exists()
        assert (target_dir / "q_regime.in").exists()

        # Check content
        q_buy_content = (target_dir / "q_buy.in").read_text()
        assert q_buy_content == "1\n0\n0"

        q_regime_content = (target_dir / "q_regime.in").read_text()
        assert q_regime_content == "2\n3\n1"


def test_export_trace_empty_episodes():
    """Test export_trace with empty episodes."""
    policy = LookupPolicy()

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir) / "traces"
        policy.export_trace([], target_dir)

        # Files should exist but be empty
        assert (target_dir / "q_buy.in").exists()
        assert (target_dir / "q_buy.in").read_text() == ""


def test_export_trace_missing_keys():
    """Test export_trace handles missing keys with defaults."""
    policy = LookupPolicy()

    episodes = [
        [
            {
                "q_buy": 1,
                # Missing other keys
            },
        ],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir) / "traces"
        policy.export_trace(episodes, target_dir)

        # Should use defaults for missing keys
        q_sell_content = (target_dir / "q_sell.in").read_text()
        assert q_sell_content == "0"  # Default

        risk_budget_ok_content = (target_dir / "risk_budget_ok.in").read_text()
        assert risk_budget_ok_content == "1"  # Default


def test_serialize_manifest():
    """Test serialize_manifest output format."""
    policy = LookupPolicy()

    # Add some states
    policy.update((0, 0, 0, 0, 0), Action.BUY, 1.0)
    policy.update((1, 0, 0, 0, 0), Action.SELL, 0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        policy.serialize_manifest(manifest_path)

        assert manifest_path.exists()

        # Parse and verify structure
        manifest = json.loads(manifest_path.read_text())
        assert "states" in manifest
        assert "actions" in manifest
        assert manifest["states"] == 2
        assert set(manifest["actions"]) == {"hold", "buy", "sell"}


def test_serialize_manifest_empty_policy():
    """Test serialize_manifest with empty policy."""
    policy = LookupPolicy()

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        policy.serialize_manifest(manifest_path)

        manifest = json.loads(manifest_path.read_text())
        assert manifest["states"] == 0
        assert len(manifest["actions"]) == 3


def test_action_strings_property():
    """Test action_strings property."""
    policy = LookupPolicy()

    action_strings = policy.action_strings
    assert isinstance(action_strings, tuple)
    assert len(action_strings) == 3
    assert "hold" in action_strings
    assert "buy" in action_strings
    assert "sell" in action_strings

    # Should be consistent
    assert policy.action_strings == policy.action_strings

