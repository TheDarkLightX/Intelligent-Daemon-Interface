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
        # Note: no trailing dots in new Tau CLI format
        assert 'i0:sbf = in file("inputs/q_buy.in")' in spec
        assert 'i1:sbf = in file("inputs/q_sell.in")' in spec

    def test_generates_output_declarations(self):
        """Generated spec should include output stream declarations."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Note: no trailing dots in new Tau CLI format
        assert 'o0:sbf = out file("outputs/position.out")' in spec

    def test_generates_input_mirrors(self):
        """Generated spec should include input mirrors when enabled."""
        schema = AgentSchema(
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
            include_mirrors=True,  # Explicitly enable mirrors
        )
        spec = generate_tau_spec(schema)
        # Mirrors use 'mi' prefix to avoid name conflict with input streams
        assert 'mi0:sbf = out file("outputs/i0_mirror.out")' in spec
        assert 'mi1:sbf = out file("outputs/i1_mirror.out")' in spec

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

    def test_generates_run_command(self):
        """Generated spec should include 'r <wff>' run command.
        
        In Tau CLI mode, we use 'r <wff>' directly (not 'defs' + 'r (' block).
        """
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Should have 'r ' followed by the WFF
        assert "r " in spec
        # The run command should contain stream references
        r_line = next((ln for ln in spec.splitlines() if ln.strip().startswith('r ')), None)
        assert r_line is not None, "No 'r' run command found"
        assert "o0[t]" in r_line or "o0" in r_line

    def test_generates_recurrence_logic(self):
        """Generated spec should include recurrence logic in run command."""
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        # Run command should contain the WFF with stream references
        assert "r " in spec
        # Should have balanced parentheses
        assert spec.count("(") == spec.count(")")

    def test_generates_execution_commands(self):
        """Generated spec should include empty lines for stepping and 'q'.
        
        In Tau REPL, empty lines advance execution (not 'n' commands).
        """
        schema = create_minimal_schema()
        spec = generate_tau_spec(schema)
        lines = spec.splitlines()
        
        # Find r command and q to count empty lines between them
        r_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith('r ')), -1)
        q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == 'q'), -1)
        
        assert r_idx >= 0, "No 'r' run command"
        assert q_idx > r_idx, "No 'q' after 'r'"
        
        # Count empty lines (execution steps)
        empty_count = sum(1 for ln in lines[r_idx+1:q_idx] if ln.strip() == '')
        assert empty_count == schema.num_steps
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
        # Note: no trailing dots in new Tau CLI format
        assert 'i0:bv[16] = in file("inputs/price.in")' in spec
        assert 'o0:bv[16] = out file("outputs/ema.out")' in spec

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
        assert "r " in spec  # Run command
        assert spec.strip().endswith("q")  # Quit command

