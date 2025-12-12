"""Tests for decomposed FSM pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestDecomposedFSMPattern:
    """Test decomposed FSM pattern."""

    def test_decomposed_fsm_basic(self):
        """Test basic decomposed FSM."""
        schema = AgentSchema(
            name="decomposed_fsm_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="buy", stream_type="sbf"),
                StreamConfig(name="sell", stream_type="sbf"),
                StreamConfig(name="idle_low", stream_type="sbf", is_input=False),
                StreamConfig(name="idle_high", stream_type="sbf", is_input=False),
                StreamConfig(name="pos_low", stream_type="sbf", is_input=False),
                StreamConfig(name="pos_high", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="decomposed_fsm",
                    inputs=("buy", "sell"),
                    output="idle_low",
                    params={
                        "hierarchy": {
                            "IDLE": {
                                "substates": ["idle_low", "idle_high"],
                                "initial": "idle_low"
                            },
                            "POSITION": {
                                "substates": ["pos_low", "pos_high"],
                                "initial": "pos_low"
                            }
                        },
                        "transitions": [
                            {"from": "idle_low", "to": "pos_low", "condition": "buy[t]"},
                            {"from": "pos_low", "to": "idle_low", "condition": "sell[t]"}
                        ],
                        "substate_outputs": ["idle_low", "idle_high", "pos_low", "pos_high"]
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "idle_low" in spec
        assert "buy" in spec
        assert "sell" in spec

