"""Tests for Hex stake pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestHexStakePattern:
    """Test Hex stake pattern implementation."""

    def test_hex_stake_basic(self):
        """Test basic Hex stake pattern (lock_active only)."""
        schema = AgentSchema(
            name="hex_stake_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="stake_amount", stream_type="bv", width=32),
                StreamConfig(name="stake_duration", stream_type="bv", width=16),
                StreamConfig(name="current_time", stream_type="bv", width=32),
                StreamConfig(name="action_stake", stream_type="sbf"),
                StreamConfig(name="action_end", stream_type="sbf"),
                StreamConfig(name="lock_active", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="hex_stake",
                    inputs=("stake_amount", "stake_duration", "current_time", "action_stake", "action_end"),
                    output="lock_active",
                    params={"max_duration": 3650}
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain lock_active logic
        assert "lock_active" in spec
        assert "[t]" in spec
        assert "[t-1]" in spec

    def test_hex_stake_with_time_tracking(self):
        """Test Hex stake with lock_start and remaining_days."""
        schema = AgentSchema(
            name="hex_stake_time",
            strategy="custom",
            streams=(
                StreamConfig(name="stake_amount", stream_type="bv", width=32),
                StreamConfig(name="stake_duration", stream_type="bv", width=16),
                StreamConfig(name="current_time", stream_type="bv", width=32),
                StreamConfig(name="action_stake", stream_type="sbf"),
                StreamConfig(name="action_end", stream_type="sbf"),
                StreamConfig(name="lock_active", stream_type="sbf", is_input=False),
                StreamConfig(name="lock_start", stream_type="bv", width=32, is_input=False),
                StreamConfig(name="remaining_days", stream_type="bv", width=16, is_input=False),
                StreamConfig(name="is_matured", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="hex_stake",
                    inputs=("stake_amount", "stake_duration", "current_time", "action_stake", "action_end"),
                    output="lock_active",
                    params={
                        "max_duration": 3650,
                        "lock_start_output": "lock_start",
                        "remaining_days_output": "remaining_days",
                        "is_matured_output": "is_matured",
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain time tracking
        assert "lock_start" in spec
        assert "remaining_days" in spec
        assert "is_matured" in spec
        assert "+" in spec  # Addition for time calculations
        assert "-" in spec  # Subtraction for remaining_days
        assert ">=" in spec  # Comparison for is_matured

