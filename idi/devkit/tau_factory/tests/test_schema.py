"""Tests for AgentSchema validation (TDD)."""

import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

# Import will be available after implementation
try:
    from idi.devkit.tau_factory.schema import (
        StreamConfig,
        LogicBlock,
        AgentSchema,
        validate_schema,
    )
except ImportError:
    pytest.skip("Schema module not yet implemented", allow_module_level=True)


class TestStreamConfig:
    """Test StreamConfig dataclass."""

    def test_create_sbf_stream(self):
        """SBF stream requires no width."""
        stream = StreamConfig(name="price_up", stream_type="sbf")
        assert stream.name == "price_up"
        assert stream.stream_type == "sbf"
        assert stream.width == 8  # Default
        assert stream.is_input is True

    def test_create_bv_stream(self):
        """BV stream requires width."""
        stream = StreamConfig(name="price", stream_type="bv", width=16)
        assert stream.name == "price"
        assert stream.stream_type == "bv"
        assert stream.width == 16

    def test_stream_is_frozen(self):
        """StreamConfig should be immutable."""
        stream = StreamConfig(name="test", stream_type="sbf")
        with pytest.raises(FrozenInstanceError):
            stream.name = "changed"

    def test_invalid_stream_type(self):
        """Invalid stream type should raise error."""
        with pytest.raises(ValueError):
            StreamConfig(name="test", stream_type="invalid")

    def test_bv_requires_width(self):
        """BV streams must specify width."""
        with pytest.raises(ValueError):
            StreamConfig(name="test", stream_type="bv", width=None)


class TestLogicBlock:
    """Test LogicBlock dataclass."""

    def test_create_fsm_block(self):
        """FSM pattern block."""
        block = LogicBlock(
            pattern="fsm",
            inputs=("buy", "sell"),
            output="position",
            params={"initial_state": "idle"},
        )
        assert block.pattern == "fsm"
        assert block.inputs == ("buy", "sell")
        assert block.output == "position"

    def test_create_counter_block(self):
        """Counter pattern block."""
        block = LogicBlock(
            pattern="counter",
            inputs=("event",),
            output="count",
            params={"max_value": 255},
        )
        assert block.pattern == "counter"

    def test_invalid_pattern(self):
        """Invalid pattern should raise error."""
        with pytest.raises(ValueError):
            LogicBlock(pattern="invalid", inputs=(), output="out")


class TestAgentSchema:
    """Test AgentSchema validation."""

    def test_create_momentum_schema(self):
        """Create momentum strategy schema."""
        streams = (
            StreamConfig(name="q_buy", stream_type="sbf"),
            StreamConfig(name="q_sell", stream_type="sbf"),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
        )
        logic = (
            LogicBlock(
                pattern="fsm",
                inputs=("q_buy", "q_sell"),
                output="position",
            ),
        )
        schema = AgentSchema(
            name="test_momentum",
            strategy="momentum",
            streams=streams,
            logic_blocks=logic,
        )
        assert schema.name == "test_momentum"
        assert schema.strategy == "momentum"
        assert len(schema.streams) == 3

    def test_schema_requires_name(self):
        """Schema must have a name."""
        schema = AgentSchema(
            name="",
            strategy="momentum",
            streams=(),
            logic_blocks=(),
        )
        with pytest.raises(ValueError):
            validate_schema(schema)

    def test_schema_validates_stream_references(self):
        """Logic blocks must reference valid streams."""
        streams = (StreamConfig(name="input1", stream_type="sbf"),)
        logic = (
            LogicBlock(
                pattern="fsm",
                inputs=("nonexistent",),
                output="output1",
            ),
        )
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=streams,
            logic_blocks=logic,
        )
        with pytest.raises(ValueError, match="nonexistent"):
            validate_schema(schema)

    def test_schema_validates_output_streams(self):
        """Logic block outputs must be defined as output streams."""
        streams = (
            StreamConfig(name="input1", stream_type="sbf", is_input=True),
            StreamConfig(name="output1", stream_type="sbf", is_input=False),
        )
        logic = (
            LogicBlock(
                pattern="fsm",
                inputs=("input1",),
                output="nonexistent_output",
            ),
        )
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=streams,
            logic_blocks=logic,
        )
        with pytest.raises(ValueError, match="nonexistent_output"):
            validate_schema(schema)

    def test_validate_schema_function(self):
        """Test validate_schema helper function."""
        valid_schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="out", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="fsm", inputs=("in",), output="out"),
            ),
        )
        # Should not raise
        validate_schema(valid_schema)

        invalid_schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(StreamConfig(name="in", stream_type="sbf"),),
            logic_blocks=(
                LogicBlock(pattern="fsm", inputs=("in",), output="missing"),
            ),
        )
        with pytest.raises(ValueError, match="missing"):
            validate_schema(invalid_schema)

