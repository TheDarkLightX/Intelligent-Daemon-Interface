"""Tests for orthogonal regions pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestOrthogonalRegionsPattern:
    """Test orthogonal regions pattern."""

    def test_orthogonal_regions_basic(self):
        """Test basic orthogonal regions with 2 regions."""
        schema = AgentSchema(
            name="orthogonal_regions_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="execution_signal", stream_type="sbf"),
                StreamConfig(name="risk_signal", stream_type="sbf"),
                StreamConfig(name="execution_state", stream_type="bv", width=2, is_input=False),
                StreamConfig(name="risk_state", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="orthogonal_regions",
                    inputs=("execution_signal", "risk_signal"),
                    output="execution_state",  # First output
                    params={
                        "regions": [
                            {
                                "name": "execution",
                                "inputs": ["execution_signal"],
                                "states": ["FLAT", "LONG"],
                                "initial_state": 0
                            },
                            {
                                "name": "risk",
                                "inputs": ["risk_signal"],
                                "states": ["NORMAL", "WARNING"],
                                "initial_state": 0
                            }
                        ],
                        "region_outputs": ["execution_state", "risk_state"]
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "execution_state" in spec
        assert "risk_state" in spec
        assert "execution_signal" in spec
        assert "risk_signal" in spec

    def test_orthogonal_regions_fsm(self):
        """Test orthogonal regions with FSM (buy/sell) inputs."""
        schema = AgentSchema(
            name="orthogonal_regions_fsm",
            strategy="custom",
            streams=(
                StreamConfig(name="execution_buy", stream_type="sbf"),
                StreamConfig(name="execution_sell", stream_type="sbf"),
                StreamConfig(name="risk_buy", stream_type="sbf"),
                StreamConfig(name="risk_sell", stream_type="sbf"),
                StreamConfig(name="execution_state", stream_type="bv", width=2, is_input=False),
                StreamConfig(name="risk_state", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="orthogonal_regions",
                    inputs=("execution_buy", "execution_sell", "risk_buy", "risk_sell"),
                    output="execution_state",
                    params={
                        "regions": [
                            {
                                "name": "execution",
                                "inputs": ["execution_buy", "execution_sell"],
                                "states": ["FLAT", "LONG"],
                                "initial_state": 0
                            },
                            {
                                "name": "risk",
                                "inputs": ["risk_buy", "risk_sell"],
                                "states": ["NORMAL", "WARNING", "CRITICAL"],
                                "initial_state": 0
                            }
                        ],
                        "region_outputs": ["execution_state", "risk_state"]
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "execution_state" in spec
        assert "risk_state" in spec
        assert "execution_buy" in spec
        assert "execution_sell" in spec

