"""Tests for TCP connection FSM pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestTCPConnectionFSMPattern:
    """Test TCP connection FSM pattern."""

    def test_tcp_connection_fsm_basic(self):
        """Test basic TCP connection FSM."""
        schema = AgentSchema(
            name="tcp_connection_fsm_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="syn_flag", stream_type="sbf"),
                StreamConfig(name="ack_flag", stream_type="sbf"),
                StreamConfig(name="fin_flag", stream_type="sbf"),
                StreamConfig(name="rst_flag", stream_type="sbf"),
                StreamConfig(name="tcp_state", stream_type="bv", width=4, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="tcp_connection_fsm",
                    inputs=("syn_flag", "ack_flag", "fin_flag", "rst_flag"),
                    output="tcp_state",
                    params={
                        "syn_flag": "syn_flag",
                        "ack_flag": "ack_flag",
                        "fin_flag": "fin_flag",
                        "rst_flag": "rst_flag"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "tcp_state" in spec
        assert "syn_flag" in spec
        assert "ack_flag" in spec
        assert "fin_flag" in spec
        assert "rst_flag" in spec

    def test_tcp_connection_fsm_with_timeout(self):
        """Test TCP connection FSM with timeout signal."""
        schema = AgentSchema(
            name="tcp_connection_fsm_timeout",
            strategy="custom",
            streams=(
                StreamConfig(name="syn_flag", stream_type="sbf"),
                StreamConfig(name="ack_flag", stream_type="sbf"),
                StreamConfig(name="fin_flag", stream_type="sbf"),
                StreamConfig(name="rst_flag", stream_type="sbf"),
                StreamConfig(name="timeout_signal", stream_type="sbf"),
                StreamConfig(name="tcp_state", stream_type="bv", width=4, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="tcp_connection_fsm",
                    inputs=("syn_flag", "ack_flag", "fin_flag", "rst_flag"),
                    output="tcp_state",
                    params={
                        "syn_flag": "syn_flag",
                        "ack_flag": "ack_flag",
                        "fin_flag": "fin_flag",
                        "rst_flag": "rst_flag",
                        "timeout_signal": "timeout_signal"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "tcp_state" in spec
        assert "timeout_signal" in spec

