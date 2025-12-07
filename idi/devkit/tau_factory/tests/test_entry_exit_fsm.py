"""Tests for entry-exit FSM pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestEntryExitFSMPattern:
    """Test entry-exit FSM pattern."""

    def test_entry_exit_fsm_basic(self):
        """Test basic entry-exit FSM."""
        schema = AgentSchema(
            name="entry_exit_fsm_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="entry_signal", stream_type="sbf"),
                StreamConfig(name="exit_signal", stream_type="sbf"),
                StreamConfig(name="phase", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="entry_exit_fsm",
                    inputs=("entry_signal", "exit_signal"),
                    output="phase",
                    params={
                        "phases": ["PRE_TRADE", "IN_TRADE", "POST_TRADE"],
                        "entry_signal": "entry_signal",
                        "exit_signal": "exit_signal"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "phase" in spec
        assert "entry_signal" in spec
        assert "exit_signal" in spec

    def test_entry_exit_fsm_with_stop_loss_take_profit(self):
        """Test entry-exit FSM with stop loss and take profit."""
        schema = AgentSchema(
            name="entry_exit_fsm_advanced",
            strategy="custom",
            streams=(
                StreamConfig(name="entry_signal", stream_type="sbf"),
                StreamConfig(name="exit_signal", stream_type="sbf"),
                StreamConfig(name="stop_loss", stream_type="sbf"),
                StreamConfig(name="take_profit", stream_type="sbf"),
                StreamConfig(name="phase", stream_type="bv", width=2, is_input=False),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="entry_exit_fsm",
                    inputs=("entry_signal", "exit_signal", "stop_loss", "take_profit"),
                    output="phase",
                    params={
                        "phases": ["PRE_TRADE", "IN_TRADE", "POST_TRADE"],
                        "entry_signal": "entry_signal",
                        "exit_signal": "exit_signal",
                        "stop_loss": "stop_loss",
                        "take_profit": "take_profit",
                        "position_output": "position"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "phase" in spec
        assert "position" in spec
        assert "stop_loss" in spec
        assert "take_profit" in spec

