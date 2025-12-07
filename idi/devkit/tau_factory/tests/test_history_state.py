"""Tests for history state pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestHistoryStatePattern:
    """Test history state pattern."""

    def test_history_state_basic(self):
        """Test basic history state."""
        schema = AgentSchema(
            name="history_state_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="substate", stream_type="bv", width=2),
                StreamConfig(name="superstate_entry", stream_type="sbf"),
                StreamConfig(name="restored_state", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="history_state",
                    inputs=("substate", "superstate_entry"),
                    output="restored_state",
                    params={
                        "substate_input": "substate",
                        "superstate_entry": "superstate_entry",
                        "initial_substate": 0
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "restored_state" in spec
        assert "substate" in spec
        assert "superstate_entry" in spec

    def test_history_state_with_exit(self):
        """Test history state with explicit exit signal."""
        schema = AgentSchema(
            name="history_state_exit",
            strategy="custom",
            streams=(
                StreamConfig(name="substate", stream_type="bv", width=2),
                StreamConfig(name="superstate_entry", stream_type="sbf"),
                StreamConfig(name="superstate_exit", stream_type="sbf"),
                StreamConfig(name="restored_state", stream_type="bv", width=2, is_input=False),
                StreamConfig(name="history", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="history_state",
                    inputs=("substate", "superstate_entry"),
                    output="restored_state",
                    params={
                        "substate_input": "substate",
                        "superstate_entry": "superstate_entry",
                        "superstate_exit": "superstate_exit",
                        "history_output": "history",
                        "initial_substate": 0
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "restored_state" in spec
        assert "history" in spec
        assert "superstate_exit" in spec

