"""Validation tests for DSLParser pattern-specific rules."""

import pytest

from idi.devkit.tau_factory.dsl_parser import DSLParser, ValidationError
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock


def _minimal_schema(pattern: str, inputs, output: str, params=None) -> AgentSchema:
    streams = (
        StreamConfig(inputs[0], "sbf", is_input=True),
        StreamConfig(output, "sbf", is_input=False),
    )
    logic_blocks = (
        LogicBlock(pattern=pattern, inputs=tuple(inputs), output=output, params=params or {}),
    )
    return AgentSchema(
        name="test_agent",
        strategy="custom",
        streams=streams,
        logic_blocks=logic_blocks,
    )


def test_hex_stake_requires_input_count():
    """Hex stake pattern must enforce required input arity."""
    parser = DSLParser()
    schema = _minimal_schema("hex_stake", ["a", "b"], "out")
    with pytest.raises(ValidationError) as exc:
        parser.parse(schema)
    assert "hex stake pattern requires" in str(exc.value).lower()


def test_unknown_pattern_rejected():
    """Unknown patterns are rejected with clear validation error."""
    with pytest.raises(ValueError):
        _minimal_schema("not_a_pattern", ["a"], "out")
