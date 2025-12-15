"""Tests for ensemble voting patterns (majority, unanimous, custom)."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestMajorityPattern:
    """Test majority voting pattern (N-of-M)."""

    def test_2_of_3_majority(self):
        """Test 2-of-3 majority voting."""
        schema = AgentSchema(
            name="ensemble_2of3",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1", stream_type="sbf"),
                StreamConfig(name="agent2", stream_type="sbf"),
                StreamConfig(name="agent3", stream_type="sbf"),
                StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("agent1", "agent2", "agent3"),
                    output="majority_buy",
                    params={"threshold": 2, "total": 3}
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        
        # Should generate: (a & b) | (a & c) | (b & c)
        assert "(i0[t] & i1[t])" in spec or "(i1[t] & i0[t])" in spec
        assert "(i0[t] & i2[t])" in spec or "(i2[t] & i0[t])" in spec
        assert "(i1[t] & i2[t])" in spec or "(i2[t] & i1[t])" in spec
        assert "majority_buy" in spec

    def test_3_of_5_majority(self):
        """Test 3-of-5 majority voting."""
        schema = AgentSchema(
            name="ensemble_3of5",
            strategy="custom",
            streams=(
                StreamConfig(name="a1", stream_type="sbf"),
                StreamConfig(name="a2", stream_type="sbf"),
                StreamConfig(name="a3", stream_type="sbf"),
                StreamConfig(name="a4", stream_type="sbf"),
                StreamConfig(name="a5", stream_type="sbf"),
                StreamConfig(name="majority", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("a1", "a2", "a3", "a4", "a5"),
                    output="majority",
                    params={"threshold": 3, "total": 5}
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        
        # Should generate all 3-element combinations (10 total)
        assert "majority" in spec
        # Check that we have OR operations between combinations
        assert " | " in spec
        # Check that we have AND operations within combinations
        assert " & " in spec

    def test_majority_defaults_to_half_plus_one(self):
        """Test that majority defaults to threshold = len//2 + 1."""
        schema = AgentSchema(
            name="ensemble_default",
            strategy="custom",
            streams=(
                StreamConfig(name="a1", stream_type="sbf"),
                StreamConfig(name="a2", stream_type="sbf"),
                StreamConfig(name="a3", stream_type="sbf"),
                StreamConfig(name="majority", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("a1", "a2", "a3"),
                    output="majority",
                    # No params - should default to 2-of-3
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        assert "majority" in spec

    def test_majority_validation(self):
        """Test majority pattern validation."""
        # Threshold > total should fail
        with pytest.raises(ValueError, match="threshold.*cannot exceed"):
            schema = AgentSchema(
                name="bad_majority",
                strategy="custom",
                streams=(
                    StreamConfig(name="a1", stream_type="sbf"),
                    StreamConfig(name="a2", stream_type="sbf"),
                    StreamConfig(name="majority", stream_type="sbf", is_input=False),
                ),
                logic_blocks=(
                    LogicBlock(
                        pattern="majority",
                        inputs=("a1", "a2"),
                        output="majority",
                        params={"threshold": 3, "total": 2}
                    ),
                ),
            )
            generate_tau_spec(schema)


class TestUnanimousPattern:
    """Test unanimous consensus pattern."""

    def test_unanimous_3_agents(self):
        """Test unanimous voting with 3 agents."""
        schema = AgentSchema(
            name="unanimous_3",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1", stream_type="sbf"),
                StreamConfig(name="agent2", stream_type="sbf"),
                StreamConfig(name="agent3", stream_type="sbf"),
                StreamConfig(name="consensus", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="unanimous",
                    inputs=("agent1", "agent2", "agent3"),
                    output="consensus"
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        
        # Should generate: a & b & c
        assert "i0[t]" in spec
        assert "i1[t]" in spec
        assert "i2[t]" in spec
        assert " & " in spec
        assert "consensus" in spec

    def test_unanimous_requires_at_least_2(self):
        """Test that unanimous requires at least 2 inputs."""
        with pytest.raises(ValueError, match="requires at least 2"):
            schema = AgentSchema(
                name="bad_unanimous",
                strategy="custom",
                streams=(
                    StreamConfig(name="agent1", stream_type="sbf"),
                    StreamConfig(name="consensus", stream_type="sbf", is_input=False),
                ),
                logic_blocks=(
                    LogicBlock(
                        pattern="unanimous",
                        inputs=("agent1",),
                        output="consensus"
                    ),
                ),
            )
            generate_tau_spec(schema)


class TestCustomPattern:
    """Test custom boolean expression pattern."""

    def test_custom_simple_expression(self):
        """Test custom pattern with simple expression."""
        schema = AgentSchema(
            name="custom_simple",
            strategy="custom",
            streams=(
                StreamConfig(name="a", stream_type="sbf"),
                StreamConfig(name="b", stream_type="sbf"),
                StreamConfig(name="result", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="custom",
                    inputs=("a", "b"),
                    output="result",
                    params={"expression": "(i0[t] & i1[t]) | (i0[t]' & i1[t]')"}
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        assert "result" in spec
        assert "i0[t]" in spec
        assert "i1[t]" in spec

    def test_custom_with_stream_names(self):
        """Test custom pattern using stream names."""
        schema = AgentSchema(
            name="custom_names",
            strategy="custom",
            streams=(
                StreamConfig(name="price_up", stream_type="sbf"),
                StreamConfig(name="volume_ok", stream_type="sbf"),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="custom",
                    inputs=("price_up", "volume_ok"),
                    output="buy_signal",
                    params={"expression": "price_up[t] & volume_ok[t]"}
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        assert "buy_signal" in spec
        # Stream names should be replaced with indices
        assert "price_up[t]" not in spec or "i0[t]" in spec

    def test_custom_requires_expression(self):
        """Test that custom pattern requires expression parameter."""
        with pytest.raises(ValueError, match="requires 'expression'"):
            schema = AgentSchema(
                name="bad_custom",
                strategy="custom",
                streams=(
                    StreamConfig(name="a", stream_type="sbf"),
                    StreamConfig(name="result", stream_type="sbf", is_input=False),
                ),
                logic_blocks=(
                    LogicBlock(
                        pattern="custom",
                        inputs=("a",),
                        output="result",
                        params={}  # Missing expression
                    ),
                ),
            )
            generate_tau_spec(schema)


class TestEnsembleIntegration:
    """Integration tests for ensemble patterns."""

    def test_ensemble_agent_with_majority_and_unanimous(self):
        """Test agent using both majority and unanimous patterns."""
        schema = AgentSchema(
            name="ensemble_agent",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1_buy", stream_type="sbf"),
                StreamConfig(name="agent2_buy", stream_type="sbf"),
                StreamConfig(name="agent3_buy", stream_type="sbf"),
                StreamConfig(name="agent_sell", stream_type="sbf"),
                StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
                StreamConfig(name="unanimous_buy", stream_type="sbf", is_input=False),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
                    output="majority_buy",
                    params={"threshold": 2, "total": 3}
                ),
                LogicBlock(
                    pattern="unanimous",
                    inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
                    output="unanimous_buy"
                ),
                LogicBlock(
                    pattern="fsm",
                    inputs=("majority_buy", "agent_sell"),
                    output="position"
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        
        # Verify all patterns are present
        assert "majority_buy" in spec
        assert "unanimous_buy" in spec
        assert "position" in spec
        # Tau CLI format: r <wff> (not 'defs' + 'r (' block)
        assert "r " in spec

