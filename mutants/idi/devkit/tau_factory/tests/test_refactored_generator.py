"""Tests for refactored Tau generator components (TDD)."""

import pytest
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec, validate_schema, create_minimal_schema
from idi.devkit.tau_factory.dsl_parser import ValidationError


class TestRefactoredGenerator:
    """Test the new clean generator interface."""

    def test_generate_minimal_schema(self):
        """Test that minimal schema generation works."""
        schema = create_minimal_schema("test_agent")
        assert schema.name == "test_agent"
        assert schema.strategy == "custom"
        assert len(schema.streams) == 2
        assert len(schema.logic_blocks) == 1

    def test_validate_valid_schema(self):
        """Test validation of a valid schema."""
        schema = create_minimal_schema("test_agent")
        errors = validate_schema(schema)
        assert len(errors) == 0

    def test_validate_invalid_schema(self):
        """Test validation catches invalid schemas."""
        # Create invalid schema (missing required fields)
        invalid_schema = AgentSchema(
            name="",  # Empty name should fail
            strategy="custom",  # Valid strategy
            streams=(),
            logic_blocks=()
        )
        errors = validate_schema(invalid_schema)
        assert len(errors) > 0
        assert any("name cannot be empty" in str(error) for error in errors)

    def test_generate_tau_spec(self):
        """Test end-to-end Tau spec generation."""
        schema = create_minimal_schema("test_agent")
        spec = generate_tau_spec(schema)

        # Should contain header
        assert "test_agent - Generated Tau Agent" in spec
        assert "Auto-generated from AgentSchema" in spec

        # Should contain I/O declarations
        assert 'in file("inputs/' in spec
        assert 'out file("outputs/' in spec

        # Should contain logic (basic passthrough)
        assert "output1 :- input1" in spec

    def test_generate_complex_schema(self):
        """Test generation with more complex patterns."""
        streams = (
            StreamConfig("input1", "sbf", is_input=True),
            StreamConfig("input2", "sbf", is_input=True),
            StreamConfig("output1", "sbf", is_input=False),
        )

        logic_blocks = (
            LogicBlock(
                pattern="majority",
                inputs=("input1", "input2"),
                output="output1",
                params={"threshold": 2}
            ),
        )

        schema = AgentSchema(
            name="majority_agent",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        spec = generate_tau_spec(schema)
        assert "majority_agent" in spec
        assert "Majority voting implementation" in spec

    def test_validation_error_details(self):
        """Test that validation errors provide helpful information."""
        # Create schema with invalid stream reference
        streams = (StreamConfig("input1", "sbf", is_input=True),)
        logic_blocks = (
            LogicBlock(
                pattern="passthrough",
                inputs=("nonexistent_input",),  # References non-existent stream
                output="input1"  # This should be an output stream, not input
            ),
        )

        schema = AgentSchema(
            name="invalid_agent",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        errors = validate_schema(schema)
        assert len(errors) > 0
        # Should catch both unknown input and invalid output
        error_messages = [str(error) for error in errors]
        assert any("unknown input stream" in msg for msg in error_messages)
        assert any("undefined stream" in msg for msg in error_messages)
