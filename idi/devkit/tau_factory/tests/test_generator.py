"""Tests for TauSpecGenerator output format (TDD)."""

import pytest
from pathlib import Path

try:
    from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
    from idi.devkit.tau_factory.generator import generate_tau_spec
except ImportError:
    pytest.skip("Generator module not yet implemented", allow_module_level=True)


def create_minimal_schema() -> AgentSchema:
    """Create a minimal valid schema for testing."""
    return AgentSchema(
        name="test_agent",
        strategy="momentum",
        streams=(
            StreamConfig(name="q_buy", stream_type="sbf"),
            StreamConfig(name="q_sell", stream_type="sbf"),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="fsm", inputs=("q_buy", "q_sell"), output="position"),
        ),
        num_steps=5,
    )


class TestTauSpecGenerator:
    """Test TauSpecGenerator output format."""

    def test_generates_header_comment(self):
        """Generated spec should include header comment."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert f"# {schema.name} Agent" in spec
        assert "Auto-generated" in spec

    def test_generates_input_declarations(self):
        """Generated spec should include input stream declarations."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert 'i0:sbf = in file("inputs/q_buy.in").' in spec
        assert 'i1:sbf = in file("inputs/q_sell.in").' in spec

    def test_generates_output_declarations(self):
        """Generated spec should include output stream declarations."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert 'o0:sbf = out file("outputs/position.out").' in spec

    def test_generates_input_mirrors(self):
        """Generated spec should include input mirrors when enabled."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert 'i0:sbf = out file("outputs/i0_mirror.out").' in spec
        assert 'i1:sbf = out file("outputs/i1_mirror.out").' in spec

    def test_skips_mirrors_when_disabled(self):
        """Generated spec should skip mirrors when include_mirrors=False."""
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="in", stream_type="sbf"),
                StreamConfig(name="out", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(LogicBlock(pattern="passthrough", inputs=("in",), output="out"),),
            include_mirrors=False,
        )
        spec = generate_tau_spec(schema)
        assert "mirror" not in spec.lower()

    def test_generates_defs_keyword(self):
        """Generated spec should include 'defs' keyword before recurrence."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert "defs" in spec
        # Should appear before 'r'
        defs_pos = spec.find("defs")
        r_pos = spec.find("r (")
        assert defs_pos < r_pos

    def test_generates_recurrence_block(self):
        """Generated spec should include recurrence block."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        assert "r (" in spec
        assert ")" in spec  # Closing paren

    def test_generates_execution_commands(self):
        """Generated spec should include 'n' commands and 'q'."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Should have num_steps 'n' commands
        assert spec.count("\nn\n") == schema.num_steps - 1 or spec.count("n\n") == schema.num_steps
        assert spec.strip().endswith("q")

    def test_generates_bitvector_streams(self):
        """Generated spec should handle bv[N] streams correctly."""
        schema = AgentSchema(
            name="test",
            strategy="momentum",
            streams=(
                StreamConfig(name="price", stream_type="bv", width=16),
                StreamConfig(name="ema", stream_type="bv", width=16, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("price",), output="ema"),
            ),
        )
        spec = generate_tau_spec(schema)
        assert 'i0:bv[16] = in file("inputs/price.in").' in spec
        assert 'o0:bv[16] = out file("outputs/ema.out").' in spec

    def test_generates_fsm_pattern(self):
        """Generated spec should include FSM logic pattern."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # FSM pattern should create position state machine
        assert "position" in spec.lower()
        assert "q_buy" in spec.lower() or "i0" in spec
        assert "q_sell" in spec.lower() or "i1" in spec

    def test_stream_naming_consistency(self):
        """Stream names should map consistently to i0, i1, o0, etc."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Inputs should be i0, i1, ...
        assert "i0" in spec
        assert "i1" in spec
        # Outputs should be o0, o1, ...
        assert "o0" in spec

    def test_spec_is_valid_syntax(self):
        """Generated spec should be syntactically valid Tau."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Basic syntax checks
        assert spec.count("(") == spec.count(")")  # Balanced parens
        assert "defs" in spec
        assert "r (" in spec
        assert spec.strip().endswith("q")

