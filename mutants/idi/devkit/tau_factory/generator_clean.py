"""Tau spec generator - compiles AgentSchema to valid Tau Language spec.

This module provides a clean interface to the refactored code generation system.
The complex logic has been moved to specialized modules for better maintainability.
"""

from __future__ import annotations

from idi.devkit.tau_factory.schema import AgentSchema
from .code_generator import TauCodeGenerator
from .dsl_parser import DSLParser, ValidationError


def generate_tau_spec(schema: AgentSchema) -> str:
    """Generate a complete Tau specification from an AgentSchema.

    This is the main entry point for Tau code generation. It uses the
    refactored modular architecture for better maintainability and testing.

    Args:
        schema: Validated AgentSchema instance

    Returns:
        Complete Tau language specification as a string

    Raises:
        ValidationError: If the schema contains validation errors
    """
    generator = TauCodeGenerator()
    return generator.generate(schema)


def validate_schema(schema: AgentSchema) -> list[ValidationError]:
    """Validate an AgentSchema without generating code.

    Args:
        schema: AgentSchema to validate

    Returns:
        List of validation errors (empty if valid)
    """
    parser = DSLParser()
    return parser.validate_schema(schema)


def create_minimal_schema(name: str, strategy: str = "custom") -> AgentSchema:
    """Create a minimal valid AgentSchema for testing.

    Args:
        name: Name for the agent schema
        strategy: Trading strategy type

    Returns:
        Minimal AgentSchema instance
    """
    from idi.devkit.tau_factory.schema import StreamConfig, LogicBlock

    # Create basic I/O streams
    streams = (
        StreamConfig("input1", "sbf", is_input=True),
        StreamConfig("output1", "sbf", is_input=False),
    )

    # Create basic logic block
    logic_blocks = (
        LogicBlock(
            pattern="passthrough",
            inputs=("input1",),
            output="output1"
        ),
    )

    return AgentSchema(
        name=name,
        strategy=strategy,
        streams=streams,
        logic_blocks=logic_blocks
    )
