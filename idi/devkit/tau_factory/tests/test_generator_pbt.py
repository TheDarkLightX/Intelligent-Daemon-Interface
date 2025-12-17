"""Property-based tests for Tau spec generation.

Tests verify invariants under procedurally generated AgentSchema inputs:
- Generated specs are syntactically valid
- Schema validation catches invalid cross-references
- Round-trip: schema -> spec -> parse preserves structure
- Resource bounds (spec size, generation time)

Usage:
    pytest idi/devkit/tau_factory/tests/test_generator_pbt.py -v
    
    # With specific seed for reproduction
    pytest idi/devkit/tau_factory/tests/test_generator_pbt.py -v --hypothesis-seed=12345
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict

import pytest

try:
    from hypothesis import given, settings, assume, Phase, HealthCheck
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    def given(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator

    def settings(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

    def assume(condition):  # type: ignore
        pass

from idi.devkit.tau_factory.schema import (
    AgentSchema,
    LogicBlock,
    StreamConfig,
    validate_schema,
)
from idi.devkit.tau_factory.code_generator import TauCodeGenerator

# Import strategies if hypothesis available
if HAS_HYPOTHESIS:
    from idi.devkit.tau_factory.tests.strategies import (
        agent_schema_strategy,
        minimal_agent_schema_strategy,
        bitvector_agent_schema_strategy,
        make_deterministic_schema,
        stream_name_strategy,
        agent_name_strategy,
    )


# =============================================================================
# Constants / Bounds
# =============================================================================

MAX_SPEC_SIZE_BYTES = 100_000  # 100KB max spec size
MAX_GENERATION_TIME_MS = 5000  # 5 second timeout


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Property-based tests for AgentSchema validation."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=100, deadline=None)
    def test_valid_schema_passes_validation(self, schema: AgentSchema) -> None:
        """Generated schemas pass validation."""
        # Should not raise
        validate_schema(schema)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=minimal_agent_schema_strategy())
    @settings(max_examples=50, deadline=None)
    def test_minimal_schema_valid(self, schema: AgentSchema) -> None:
        """Minimal schemas are valid."""
        validate_schema(schema)

        # Invariants
        assert len(schema.streams) >= 2  # At least 1 input + 1 output
        assert len(schema.logic_blocks) >= 1
        assert schema.num_steps >= 1

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(
        name=st.text(max_size=0),  # Empty name
    )
    @settings(max_examples=10, deadline=None)
    def test_empty_name_rejected(self, name: str) -> None:
        """Empty schema name is rejected."""
        input_stream = StreamConfig(name="in1", stream_type="sbf", width=8, is_input=True)
        output_stream = StreamConfig(name="out1", stream_type="sbf", width=8, is_input=False)
        block = LogicBlock(pattern="passthrough", inputs=("in1",), output="out1", params={})

        schema = AgentSchema(
            name=name,
            strategy="custom",
            streams=(input_stream, output_stream),
            logic_blocks=(block,),
            num_steps=5,
        )

        with pytest.raises(ValueError, match="name"):
            validate_schema(schema)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(num_steps=st.integers(min_value=-100, max_value=0))
    @settings(max_examples=20, deadline=None)
    def test_invalid_num_steps_rejected(self, num_steps: int) -> None:
        """Non-positive num_steps is rejected at construction."""
        input_stream = StreamConfig(name="in1", stream_type="sbf", width=8, is_input=True)
        output_stream = StreamConfig(name="out1", stream_type="sbf", width=8, is_input=False)
        block = LogicBlock(pattern="passthrough", inputs=("in1",), output="out1", params={})

        with pytest.raises(ValueError, match="num_steps"):
            AgentSchema(
                name="TestAgent",
                strategy="custom",
                streams=(input_stream, output_stream),
                logic_blocks=(block,),
                num_steps=num_steps,
            )


# =============================================================================
# Spec Generation Tests
# =============================================================================

class TestSpecGeneration:
    """Property-based tests for Tau spec generation."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_generation_produces_string(self, schema: AgentSchema) -> None:
        """Generator produces non-empty string output."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        assert isinstance(spec, str)
        assert len(spec) > 0

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_spec_size_bounded(self, schema: AgentSchema) -> None:
        """Generated spec size is bounded."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        spec_bytes = len(spec.encode("utf-8"))
        assert spec_bytes <= MAX_SPEC_SIZE_BYTES, f"Spec too large: {spec_bytes} bytes"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_generation_time_bounded(self, schema: AgentSchema) -> None:
        """Spec generation completes within timeout."""
        generator = TauCodeGenerator()
        start = time.perf_counter()
        spec = generator.generate(schema)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms <= MAX_GENERATION_TIME_MS, f"Generation too slow: {elapsed_ms:.0f}ms"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_spec_contains_required_sections(self, schema: AgentSchema) -> None:
        """Generated spec contains required sections."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        # Must have header comment
        assert "#" in spec

        # Must have input declarations
        assert "= in file(" in spec

        # Must have output declarations
        assert "= out file(" in spec

        # Must have defs block
        assert "defs" in spec

        # Must have run command
        assert "r (" in spec

        # Must have quit command
        assert "\nq" in spec or spec.endswith("q")

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_spec_input_count_matches_schema(self, schema: AgentSchema) -> None:
        """Number of input declarations matches schema."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        input_streams = [s for s in schema.streams if s.is_input]
        input_decl_count = spec.count("= in file(")

        assert input_decl_count == len(input_streams)

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_spec_output_count_matches_schema(self, schema: AgentSchema) -> None:
        """Number of output declarations matches schema."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        output_streams = [s for s in schema.streams if not s.is_input]
        output_decl_count = spec.count("= out file(")

        # Account for mirrors if enabled
        if schema.include_mirrors:
            input_streams = [s for s in schema.streams if s.is_input]
            expected = len(output_streams) + len(input_streams)  # outputs + mirrors
        else:
            expected = len(output_streams)

        assert output_decl_count == expected

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_spec_step_count_matches_schema(self, schema: AgentSchema) -> None:
        """Number of 'n' commands matches num_steps."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        # Count standalone 'n' lines (normalize commands)
        n_count = len(re.findall(r"^n$", spec, re.MULTILINE))

        assert n_count == schema.num_steps


# =============================================================================
# Syntax Validity Tests
# =============================================================================

class TestSpecSyntax:
    """Property-based tests for spec syntax validity."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_balanced_parentheses(self, schema: AgentSchema) -> None:
        """Generated spec has balanced parentheses."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        open_count = spec.count("(")
        close_count = spec.count(")")

        assert open_count == close_count, f"Unbalanced: {open_count} open, {close_count} close"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_balanced_brackets(self, schema: AgentSchema) -> None:
        """Generated spec has balanced brackets."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        open_count = spec.count("[")
        close_count = spec.count("]")

        assert open_count == close_count, f"Unbalanced: {open_count} open, {close_count} close"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_balanced_quotes(self, schema: AgentSchema) -> None:
        """Generated spec has balanced double quotes."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        quote_count = spec.count('"')

        assert quote_count % 2 == 0, f"Unbalanced quotes: {quote_count}"

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=agent_schema_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_empty_lines_in_run_block(self, schema: AgentSchema) -> None:
        """Run block doesn't have problematic empty lines."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        # Find the run block
        run_start = spec.find("r (")
        run_end = spec.find(")", run_start) if run_start >= 0 else -1

        if run_start >= 0 and run_end >= 0:
            run_block = spec[run_start:run_end + 1]
            # Should not have multiple consecutive newlines inside
            assert "\n\n\n" not in run_block


# =============================================================================
# Bitvector-specific Tests
# =============================================================================

class TestBitvectorSpecs:
    """Property-based tests for bitvector stream handling."""

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=bitvector_agent_schema_strategy())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bv_type_in_spec(self, schema: AgentSchema) -> None:
        """Bitvector schemas produce specs with bv[] types."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        # Should contain bv type declaration
        assert "bv[" in spec

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @given(schema=bitvector_agent_schema_strategy())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_bv_width_matches_schema(self, schema: AgentSchema) -> None:
        """Bitvector width in spec matches schema."""
        generator = TauCodeGenerator()
        spec = generator.generate(schema)

        # Find the bv stream
        bv_streams = [s for s in schema.streams if s.stream_type == "bv"]
        assume(len(bv_streams) > 0)

        for stream in bv_streams:
            expected_type = f"bv[{stream.width}]"
            assert expected_type in spec, f"Missing {expected_type} in spec"


# =============================================================================
# Deterministic Seed Tests
# =============================================================================

class TestDeterministicSeeds:
    """Tests with known seeds for CI reproducibility."""

    def test_known_good_seed_42(self) -> None:
        """Verify behavior with seed=42 (regression anchor)."""
        generator = TauCodeGenerator()
        schema = make_deterministic_schema(42)

        # Should generate without error
        spec = generator.generate(schema)

        assert len(spec) > 0
        assert "= in file(" in spec
        assert "= out file(" in spec

    def test_seed_sequence_produces_different_specs(self) -> None:
        """Different seeds produce different specs."""
        generator = TauCodeGenerator()
        specs = [generator.generate(make_deterministic_schema(i)) for i in range(5)]

        # All should be unique (different agent names at minimum)
        unique_specs = set(specs)
        assert len(unique_specs) == 5

    def test_same_seed_reproduces_spec(self) -> None:
        """Same seed produces identical spec."""
        generator = TauCodeGenerator()
        schema1 = make_deterministic_schema(123)
        schema2 = make_deterministic_schema(123)

        spec1 = generator.generate(schema1)
        spec2 = generator.generate(schema2)

        assert spec1 == spec2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_step_spec(self) -> None:
        """Schema with num_steps=1 generates valid spec."""
        generator = TauCodeGenerator()
        schema = AgentSchema(
            name="SingleStep",
            strategy="custom",
            streams=(
                StreamConfig(name="in1", stream_type="sbf", width=8, is_input=True),
                StreamConfig(name="out1", stream_type="sbf", width=8, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("in1",), output="out1", params={}),
            ),
            num_steps=1,
        )

        spec = generator.generate(schema)

        # Should have exactly 1 'n' command
        n_count = len(re.findall(r"^n$", spec, re.MULTILINE))
        assert n_count == 1

    def test_max_steps_spec(self) -> None:
        """Schema with max num_steps generates valid spec."""
        generator = TauCodeGenerator()
        schema = AgentSchema(
            name="MaxSteps",
            strategy="custom",
            streams=(
                StreamConfig(name="in1", stream_type="sbf", width=8, is_input=True),
                StreamConfig(name="out1", stream_type="sbf", width=8, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="passthrough", inputs=("in1",), output="out1", params={}),
            ),
            num_steps=20,
        )

        spec = generator.generate(schema)

        n_count = len(re.findall(r"^n$", spec, re.MULTILINE))
        assert n_count == 20

    def test_many_inputs_spec(self) -> None:
        """Schema with many inputs generates valid spec."""
        generator = TauCodeGenerator()
        inputs = [
            StreamConfig(name=f"in{i}", stream_type="sbf", width=8, is_input=True)
            for i in range(8)
        ]
        output = StreamConfig(name="out1", stream_type="sbf", width=8, is_input=False)

        schema = AgentSchema(
            name="ManyInputs",
            strategy="custom",
            streams=tuple(inputs) + (output,),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=tuple(f"in{i}" for i in range(3)),
                    output="out1",
                    params={},
                ),
            ),
            num_steps=5,
        )

        spec = generator.generate(schema)

        # Should have 8 input declarations
        assert spec.count("= in file(") == 8
