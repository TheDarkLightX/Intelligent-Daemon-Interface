"""Tests for weighted vote and time lock patterns using bitvectors."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestWeightedVotePattern:
    """Test weighted voting pattern with bitvector arithmetic."""

    def test_weighted_vote_boolean_output(self):
        """Test weighted vote with boolean output (comparison result)."""
        schema = AgentSchema(
            name="weighted_vote_bool",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1", stream_type="sbf"),
                StreamConfig(name="agent2", stream_type="sbf"),
                StreamConfig(name="agent3", stream_type="sbf"),
                StreamConfig(name="decision", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="weighted_vote",
                    inputs=("agent1", "agent2", "agent3"),
                    output="decision",
                    params={
                        "weights": [3, 2, 1],
                        "threshold": 4
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain weighted sum computation
        assert "decision" in spec
        assert ">=" in spec or ">=" in spec  # Comparison operator
        assert "[t]" in spec
        
        # Should contain bitvector arithmetic
        assert "bv[" in spec or ":bv[" in spec

    def test_weighted_vote_bitvector_output(self):
        """Test weighted vote with bitvector output (weighted sum)."""
        schema = AgentSchema(
            name="weighted_vote_bv",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1", stream_type="sbf"),
                StreamConfig(name="agent2", stream_type="sbf"),
                StreamConfig(name="weighted_sum", stream_type="bv", width=8, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="weighted_vote",
                    inputs=("agent1", "agent2"),
                    output="weighted_sum",
                    params={
                        "weights": [3, 2],
                        "threshold": 3
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain weighted sum computation
        assert "weighted_sum" in spec
        assert "+" in spec  # Addition for weighted sum
        assert "[t]" in spec


class TestTimeLockPattern:
    """Test time-lock pattern with bitvector arithmetic."""

    def test_time_lock_boolean_output(self):
        """Test time lock with boolean output (lock_active)."""
        schema = AgentSchema(
            name="time_lock_bool",
            strategy="custom",
            streams=(
                StreamConfig(name="lock_start", stream_type="bv", width=16),
                StreamConfig(name="lock_duration", stream_type="bv", width=16),
                StreamConfig(name="current_time", stream_type="bv", width=16),
                StreamConfig(name="lock_active", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="time_lock",
                    inputs=("lock_start", "lock_duration", "current_time"),
                    output="lock_active",
                    params={
                        "lock_start": "lock_start",
                        "lock_duration": "lock_duration",
                        "current_time": "current_time"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain time arithmetic
        assert "lock_active" in spec
        assert "+" in spec  # Addition: lock_start + lock_duration
        assert "<" in spec  # Safe comparison: current_time < end_time
        
        # Should contain bitvector references
        assert "bv[" in spec or ":bv[" in spec

    def test_time_lock_bitvector_output(self):
        """Test time lock with bitvector output (remaining_time)."""
        schema = AgentSchema(
            name="time_lock_bv",
            strategy="custom",
            streams=(
                StreamConfig(name="lock_start", stream_type="bv", width=16),
                StreamConfig(name="lock_duration", stream_type="bv", width=16),
                StreamConfig(name="current_time", stream_type="bv", width=16),
                StreamConfig(name="remaining_time", stream_type="bv", width=16, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="time_lock",
                    inputs=("lock_start", "lock_duration", "current_time"),
                    output="remaining_time",
                    params={
                        "lock_start": "lock_start",
                        "lock_duration": "lock_duration",
                        "current_time": "current_time"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain time arithmetic
        assert "remaining_time" in spec
        assert "+" in spec  # Addition
        assert "-" in spec  # Subtraction

