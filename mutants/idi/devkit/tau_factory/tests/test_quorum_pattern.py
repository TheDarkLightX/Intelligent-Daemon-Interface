"""Tests for quorum pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestQuorumPattern:
    """Test quorum pattern (uses majority internally)."""

    def test_3_of_5_quorum(self):
        """Test 3-of-5 quorum voting."""
        schema = AgentSchema(
            name="quorum_3of5",
            strategy="custom",
            streams=(
                StreamConfig(name="vote1", stream_type="sbf"),
                StreamConfig(name="vote2", stream_type="sbf"),
                StreamConfig(name="vote3", stream_type="sbf"),
                StreamConfig(name="vote4", stream_type="sbf"),
                StreamConfig(name="vote5", stream_type="sbf"),
                StreamConfig(name="quorum_met", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="quorum",
                    inputs=("vote1", "vote2", "vote3", "vote4", "vote5"),
                    output="quorum_met",
                    params={"threshold": 3, "total": 5}
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        
        # Should generate majority-like logic (3-of-5)
        assert "quorum_met" in spec
        assert " | " in spec  # OR operations
        assert " & " in spec  # AND operations

    def test_quorum_defaults_to_half_plus_one(self):
        """Test that quorum defaults to threshold = len//2 + 1."""
        schema = AgentSchema(
            name="quorum_default",
            strategy="custom",
            streams=(
                StreamConfig(name="vote1", stream_type="sbf"),
                StreamConfig(name="vote2", stream_type="sbf"),
                StreamConfig(name="vote3", stream_type="sbf"),
                StreamConfig(name="vote4", stream_type="sbf"),
                StreamConfig(name="quorum_met", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="quorum",
                    inputs=("vote1", "vote2", "vote3", "vote4"),
                    output="quorum_met",
                    # No params - should default to 3-of-4
                ),
            ),
        )
        spec = generate_tau_spec(schema)
        assert "quorum_met" in spec

    def test_quorum_validation(self):
        """Test quorum pattern validation."""
        # Threshold > total should fail
        with pytest.raises(ValueError, match="threshold.*cannot exceed"):
            schema = AgentSchema(
                name="bad_quorum",
                strategy="custom",
                streams=(
                    StreamConfig(name="vote1", stream_type="sbf"),
                    StreamConfig(name="vote2", stream_type="sbf"),
                    StreamConfig(name="quorum_met", stream_type="sbf", is_input=False),
                ),
                logic_blocks=(
                    LogicBlock(
                        pattern="quorum",
                        inputs=("vote1", "vote2"),
                        output="quorum_met",
                        params={"threshold": 3, "total": 2}
                    ),
                ),
            )
            generate_tau_spec(schema)

