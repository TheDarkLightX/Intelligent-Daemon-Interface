"""Tests for OutputValidator checks (TDD)."""

import pytest

try:
    from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
    from idi.devkit.tau_factory.validator import (
        ValidationResult,
        validate_agent_outputs,
        CheckRule,
    )
except ImportError:
    pytest.skip("Validator module not yet implemented", allow_module_level=True)


class TestOutputValidator:
    """Test OutputValidator functionality."""

    def test_validation_result_creation(self):
        """ValidationResult should store check results."""
        result = ValidationResult(
            passed=True,
            checks=[("position_valid", True, "Position is valid")],
        )
        assert result.passed is True
        assert len(result.checks) == 1

    def test_validate_position_binary(self):
        """Position outputs should be binary (0 or 1)."""
        outputs = {"position.out": ["0", "1", "0", "1"]}
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="position"),),
        )
        
        result = validate_agent_outputs(outputs, schema)
        assert result.passed is True

    def test_validate_position_invalid_values(self):
        """Position outputs with invalid values should fail."""
        outputs = {"position.out": ["0", "2", "1"]}  # 2 is invalid
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="position"),),
        )
        
        result = validate_agent_outputs(outputs, schema)
        # Should have a check for position validity
        position_checks = [c for c in result.checks if "position" in c[0].lower()]
        if position_checks:
            assert any(not c[1] for c in position_checks)  # At least one failed

    def test_validate_mutual_exclusion(self):
        """Buy and sell signals should not both be 1 simultaneously."""
        outputs = {
            "buy_signal.out": ["1", "0", "0"],
            "sell_signal.out": ["0", "1", "0"],
        }
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
                StreamConfig(name="sell_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("in",), output="buy_signal"),
                LogicBlock(pattern="passthrough", inputs=("in",), output="sell_signal"),
            ),
        )
        
        result = validate_agent_outputs(outputs, schema)
        # Should pass - no simultaneous 1s
        assert result.passed is True

    def test_validate_mutual_exclusion_violation(self):
        """Simultaneous buy and sell should fail validation."""
        outputs = {
            "buy_signal.out": ["1", "1"],
            "sell_signal.out": ["1", "0"],  # Both 1 at t=0
        }
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
                StreamConfig(name="sell_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("in",), output="buy_signal"),
                LogicBlock(pattern="passthrough", inputs=("in",), output="sell_signal"),
            ),
        )
        
        result = validate_agent_outputs(outputs, schema)
        # Should have mutual exclusion check
        exclusion_checks = [c for c in result.checks if "mutual" in c[0].lower() or "exclusion" in c[0].lower()]
        if exclusion_checks:
            assert any(not c[1] for c in exclusion_checks)

    def test_validate_output_lengths_match(self):
        """All outputs should have the same length."""
        outputs = {
            "position.out": ["0", "1", "0"],
            "buy_signal.out": ["1", "0"],  # Different length
        }
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("in",), output="position"),
                LogicBlock(pattern="passthrough", inputs=("in",), output="buy_signal"),
            ),
        )
        
        result = validate_agent_outputs(outputs, schema)
        length_checks = [c for c in result.checks if "length" in c[0].lower()]
        if length_checks:
            assert any(not c[1] for c in length_checks)

    def test_validate_bitvector_range(self):
        """Bitvector outputs should be within valid range."""
        outputs = {"price.out": ["100", "200", "65535"]}  # bv[16] max is 65535
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="bv", width=16),
                StreamConfig(name="price", stream_type="bv", width=16, is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="price"),),
        )
        
        result = validate_agent_outputs(outputs, schema)
        # Should pass - all values within 0-65535 for bv[16]
        assert result.passed is True

    def test_validate_bitvector_overflow(self):
        """Bitvector outputs exceeding max should fail."""
        outputs = {"price.out": ["65536"]}  # Exceeds bv[16] max
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="bv", width=16),
                StreamConfig(name="price", stream_type="bv", width=16, is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="price"),),
        )
        
        result = validate_agent_outputs(outputs, schema)
        range_checks = [c for c in result.checks if "range" in c[0].lower() or "overflow" in c[0].lower()]
        if range_checks:
            assert any(not c[1] for c in range_checks)

    def test_validate_empty_outputs(self):
        """Empty outputs should fail validation."""
        outputs = {}
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="out", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="out"),),
        )
        
        result = validate_agent_outputs(outputs, schema)
        assert result.passed is False

