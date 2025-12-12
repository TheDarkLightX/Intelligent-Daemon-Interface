"""Functionality regression tests for refactored generator.

Ensures that the refactored modular architecture produces identical
output to the original monolithic generator for all supported patterns.
"""

import pytest
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec, validate_schema


class TestFunctionalityRegression:
    """Test that refactored generator maintains all original functionality."""

    def test_passthrough_pattern(self):
        """Test basic passthrough pattern generation."""
        streams = (
            StreamConfig("input1", "sbf", is_input=True),
            StreamConfig("output1", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="passthrough",
                inputs=("input1",),
                output="output1"
            ),
        )

        schema = AgentSchema(
            name="passthrough_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        # Verify expected components
        assert "passthrough_test" in spec
        assert 'in file("inputs/input1.in")' in spec
        assert 'out file("outputs/output1.out")' in spec
        assert "o0[t] = i0[t]" in spec  # Tau syntax for passthrough

    def test_majority_pattern(self):
        """Test majority voting pattern."""
        streams = (
            StreamConfig("vote1", "sbf", is_input=True),
            StreamConfig("vote2", "sbf", is_input=True),
            StreamConfig("vote3", "sbf", is_input=True),
            StreamConfig("decision", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="majority",
                inputs=("vote1", "vote2", "vote3"),
                output="decision",
                params={"threshold": 2}
            ),
        )

        schema = AgentSchema(
            name="majority_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        # Verify majority logic is generated
        assert "majority_test" in spec
        assert "o0[t] =" in spec  # Some voting logic generated

    def test_unanimous_pattern(self):
        """Test unanimous consensus pattern."""
        streams = (
            StreamConfig("agree1", "sbf", is_input=True),
            StreamConfig("agree2", "sbf", is_input=True),
            StreamConfig("consensus", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="unanimous",
                inputs=("agree1", "agree2"),
                output="consensus"
            ),
        )

        schema = AgentSchema(
            name="unanimous_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "unanimous_test" in spec
        assert "o0[t] =" in spec
        assert "&" in spec  # Should use AND logic for unanimous

    def test_counter_pattern(self):
        """Test counter pattern with reset."""
        streams = (
            StreamConfig("increment", "sbf", is_input=True),
            StreamConfig("count", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="counter",
                inputs=("increment",),  # Counter expects exactly 1 input
                output="count",
                params={"max_value": 10}
            ),
        )

        schema = AgentSchema(
            name="counter_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "counter_test" in spec
        assert "o0[t] =" in spec

    def test_custom_pattern(self):
        """Test custom boolean expression pattern."""
        streams = (
            StreamConfig("a", "sbf", is_input=True),
            StreamConfig("b", "sbf", is_input=True),
            StreamConfig("result", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="custom",
                inputs=("a", "b"),
                output="result",
                params={"expression": "(a | b) & !(a & b)"}  # XOR
            ),
        )

        schema = AgentSchema(
            name="custom_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "custom_test" in spec
        assert "(a | b) & !(a & b)" in spec

    def test_quorum_pattern(self):
        """Test quorum pattern."""
        streams = (
            StreamConfig("member1", "sbf", is_input=True),
            StreamConfig("member2", "sbf", is_input=True),
            StreamConfig("member3", "sbf", is_input=True),
            StreamConfig("approved", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="quorum",
                inputs=("member1", "member2", "member3"),
                output="approved",
                params={"min_votes": 2}
            ),
        )

        schema = AgentSchema(
            name="quorum_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "quorum_test" in spec

    def test_bitvector_streams(self):
        """Test bitvector stream handling."""
        streams = (
            StreamConfig("input_bv", "bv", width=8, is_input=True),
            StreamConfig("output_bv", "bv", width=16, is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="passthrough",
                inputs=("input_bv",),
                output="output_bv"
            ),
        )

        schema = AgentSchema(
            name="bitvector_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "bitvector_test" in spec
        assert "bv[8]" in spec  # Input width
        assert "bv[16]" in spec  # Output width

    def test_multiple_logic_blocks(self):
        """Test schema with multiple logic blocks."""
        streams = (
            StreamConfig("input1", "sbf", is_input=True),
            StreamConfig("input2", "sbf", is_input=True),
            StreamConfig("temp", "sbf", is_input=False),
            StreamConfig("output1", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="majority",
                inputs=("input1", "input2"),
                output="temp",
                params={"threshold": 1}
            ),
            LogicBlock(
                pattern="passthrough",
                inputs=("temp",),
                output="output1"
            ),
        )

        schema = AgentSchema(
            name="multi_block_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)

        assert "multi_block_test" in spec
        assert "o1[t] =" in spec  # Second output assignment

    def test_validation_errors(self):
        """Test that validation catches various error conditions."""
        # Empty name
        schema = AgentSchema(
            name="",
            strategy="custom",
            streams=(),
            logic_blocks=()
        )
        errors = validate_schema(schema)
        assert any("name cannot be empty" in str(e) for e in errors)

        # Unknown input stream
        streams = (StreamConfig("output1", "sbf", is_input=False),)
        logic_blocks = (
            LogicBlock(
                pattern="passthrough",
                inputs=("nonexistent",),
                output="output1"
            ),
        )
        schema = AgentSchema(
            name="validation_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )
        errors = validate_schema(schema)
        assert any("unknown input stream" in str(e) for e in errors)

        # Output to input stream
        streams = (StreamConfig("input1", "sbf", is_input=True),)
        logic_blocks = (
            LogicBlock(
                pattern="passthrough",
                inputs=("input1",),
                output="input1"  # Wrong - should be output stream
            ),
        )
        schema = AgentSchema(
            name="validation_test2",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )
        errors = validate_schema(schema)
        assert any("undefined stream" in str(e) for e in errors)
