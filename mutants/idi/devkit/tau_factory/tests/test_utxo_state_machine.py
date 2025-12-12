"""Tests for UTXO state machine pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestUTXOStateMachinePattern:
    """Test UTXO state machine pattern."""

    def test_utxo_state_machine_basic(self):
        """Test basic UTXO state machine."""
        schema = AgentSchema(
            name="utxo_state_machine_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="tx_inputs", stream_type="bv", width=32),
                StreamConfig(name="tx_outputs", stream_type="bv", width=32),
                StreamConfig(name="utxo_set", stream_type="bv", width=32, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="utxo_state_machine",
                    inputs=("tx_inputs", "tx_outputs"),
                    output="utxo_set",
                    params={
                        "tx_inputs": "tx_inputs",
                        "tx_outputs": "tx_outputs",
                        "initial_utxo_set": 0
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "utxo_set" in spec
        assert "tx_inputs" in spec
        assert "tx_outputs" in spec

    def test_utxo_state_machine_with_validation(self):
        """Test UTXO state machine with transaction validation."""
        schema = AgentSchema(
            name="utxo_state_machine_validation",
            strategy="custom",
            streams=(
                StreamConfig(name="tx_inputs", stream_type="bv", width=32),
                StreamConfig(name="tx_outputs", stream_type="bv", width=32),
                StreamConfig(name="tx_valid", stream_type="sbf"),
                StreamConfig(name="utxo_set", stream_type="bv", width=32, is_input=False),
                StreamConfig(name="tx_valid_output", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="utxo_state_machine",
                    inputs=("tx_inputs", "tx_outputs"),
                    output="utxo_set",
                    params={
                        "tx_inputs": "tx_inputs",
                        "tx_outputs": "tx_outputs",
                        "tx_valid": "tx_valid",
                        "tx_valid_output": "tx_valid_output"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "utxo_set" in spec
        assert "tx_valid" in spec
        assert "tx_valid_output" in spec

