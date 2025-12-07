"""Tests for state aggregation pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestStateAggregationPattern:
    """Test state aggregation pattern."""

    def test_state_aggregation_majority(self):
        """Test state aggregation with majority method."""
        schema = AgentSchema(
            name="state_aggregation_majority",
            strategy="custom",
            streams=(
                StreamConfig(name="fsm1_state", stream_type="sbf"),
                StreamConfig(name="fsm2_state", stream_type="sbf"),
                StreamConfig(name="fsm3_state", stream_type="sbf"),
                StreamConfig(name="aggregate_state", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="state_aggregation",
                    inputs=("fsm1_state", "fsm2_state", "fsm3_state"),
                    output="aggregate_state",
                    params={
                        "method": "majority",
                        "threshold": 2,
                        "total": 3
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "aggregate_state" in spec
        assert "fsm1_state" in spec
        assert "fsm2_state" in spec
        assert "fsm3_state" in spec

    def test_state_aggregation_unanimous(self):
        """Test state aggregation with unanimous method."""
        schema = AgentSchema(
            name="state_aggregation_unanimous",
            strategy="custom",
            streams=(
                StreamConfig(name="fsm1_state", stream_type="sbf"),
                StreamConfig(name="fsm2_state", stream_type="sbf"),
                StreamConfig(name="aggregate_state", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="state_aggregation",
                    inputs=("fsm1_state", "fsm2_state"),
                    output="aggregate_state",
                    params={
                        "method": "unanimous"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "aggregate_state" in spec
        assert "fsm1_state" in spec
        assert "fsm2_state" in spec

    def test_state_aggregation_custom(self):
        """Test state aggregation with custom expression."""
        schema = AgentSchema(
            name="state_aggregation_custom",
            strategy="custom",
            streams=(
                StreamConfig(name="fsm1_state", stream_type="sbf"),
                StreamConfig(name="fsm2_state", stream_type="sbf"),
                StreamConfig(name="aggregate_state", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="state_aggregation",
                    inputs=("fsm1_state", "fsm2_state"),
                    output="aggregate_state",
                    params={
                        "method": "custom",
                        "expression": "(fsm1_state[t] & fsm2_state[t]) | (fsm1_state[t]' & fsm2_state[t]')"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "aggregate_state" in spec
        assert "fsm1_state" in spec
        assert "fsm2_state" in spec

