"""Tests for script execution pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestScriptExecutionPattern:
    """Test script execution pattern."""

    def test_script_execution_basic(self):
        """Test basic script execution."""
        schema = AgentSchema(
            name="script_execution_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="script", stream_type="bv", width=8),
                StreamConfig(name="stack", stream_type="bv", width=32),
                StreamConfig(name="execution_result", stream_type="bv", width=32, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="script_execution",
                    inputs=("script", "stack"),
                    output="execution_result",
                    params={
                        "script_input": "script",
                        "stack_input": "stack",
                        "opcodes": {
                            "OP_DUP": "duplicate_top",
                            "OP_HASH160": "external_hash"
                        }
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "execution_result" in spec
        assert "script" in spec
        assert "stack" in spec

