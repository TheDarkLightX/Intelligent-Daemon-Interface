"""Tests for high-priority patterns: multi-bit counter, streak counter, mode switch, proposal FSM, risk FSM."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestMultiBitCounterPattern:
    """Test multi-bit counter pattern."""

    def test_multi_bit_counter_basic(self):
        """Test basic multi-bit counter."""
        schema = AgentSchema(
            name="multi_bit_counter_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="increment", stream_type="sbf"),
                StreamConfig(name="counter", stream_type="bv", width=3, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="multi_bit_counter",
                    inputs=("increment",),
                    output="counter",
                    params={"width": 3}
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "counter" in spec
        assert "[t]" in spec
        assert "[t-1]" in spec
        assert "+" in spec  # Addition for increment
        assert "bv[3]" in spec

    def test_multi_bit_counter_with_reset(self):
        """Test multi-bit counter with reset."""
        schema = AgentSchema(
            name="multi_bit_counter_reset",
            strategy="custom",
            streams=(
                StreamConfig(name="increment", stream_type="sbf"),
                StreamConfig(name="reset", stream_type="sbf"),
                StreamConfig(name="counter", stream_type="bv", width=4, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="multi_bit_counter",
                    inputs=("increment", "reset"),
                    output="counter",
                    params={"width": 4, "initial_value": 0}
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "counter" in spec
        assert "reset" in spec
        assert "increment" in spec


class TestStreakCounterPattern:
    """Test streak counter pattern."""

    def test_streak_counter_basic(self):
        """Test basic streak counter."""
        schema = AgentSchema(
            name="streak_counter_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="win", stream_type="sbf"),
                StreamConfig(name="loss", stream_type="sbf"),
                StreamConfig(name="streak", stream_type="bv", width=4, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="streak_counter",
                    inputs=("win", "loss"),
                    output="streak",
                    params={"width": 4, "opposite_event": "loss"}
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "streak" in spec
        assert "win" in spec
        assert "loss" in spec
        assert "+" in spec  # Addition for increment


class TestModeSwitchPattern:
    """Test mode switch pattern."""

    def test_mode_switch_basic(self):
        """Test basic mode switch."""
        schema = AgentSchema(
            name="mode_switch_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="aggressive_signal", stream_type="sbf"),
                StreamConfig(name="defensive_signal", stream_type="sbf"),
                StreamConfig(name="mode", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="mode_switch",
                    inputs=("aggressive_signal", "defensive_signal"),
                    output="mode",
                    params={
                        "modes": ["DEFENSIVE", "AGGRESSIVE"],
                        "transitions": {
                            "DEFENSIVE": {"to": "AGGRESSIVE", "on": "aggressive_signal"},
                            "AGGRESSIVE": {"to": "DEFENSIVE", "on": "defensive_signal"}
                        },
                        "initial_mode": 0
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "mode" in spec
        assert "aggressive_signal" in spec
        assert "defensive_signal" in spec


class TestProposalFSMPattern:
    """Test proposal FSM pattern."""

    def test_proposal_fsm_basic(self):
        """Test basic proposal FSM."""
        schema = AgentSchema(
            name="proposal_fsm_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="create", stream_type="sbf"),
                StreamConfig(name="vote", stream_type="sbf"),
                StreamConfig(name="execute", stream_type="sbf"),
                StreamConfig(name="cancel", stream_type="sbf"),
                StreamConfig(name="quorum_met", stream_type="sbf"),
                StreamConfig(name="proposal_state", stream_type="bv", width=3, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="proposal_fsm",
                    inputs=("create", "vote", "execute", "cancel"),
                    output="proposal_state",
                    params={
                        "create_input": "create",
                        "vote_input": "vote",
                        "execute_input": "execute",
                        "cancel_input": "cancel",
                        "quorum_met": "quorum_met"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "proposal_state" in spec
        assert "create" in spec
        assert "vote" in spec
        assert "execute" in spec
        assert "cancel" in spec


class TestRiskFSMPattern:
    """Test risk FSM pattern."""

    def test_risk_fsm_basic(self):
        """Test basic risk FSM."""
        schema = AgentSchema(
            name="risk_fsm_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="warning_signal", stream_type="sbf"),
                StreamConfig(name="critical_signal", stream_type="sbf"),
                StreamConfig(name="normal_signal", stream_type="sbf"),
                StreamConfig(name="risk_state", stream_type="bv", width=2, is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="risk_fsm",
                    inputs=("warning_signal", "critical_signal", "normal_signal"),
                    output="risk_state",
                    params={
                        "warning_signal": "warning_signal",
                        "critical_signal": "critical_signal",
                        "normal_signal": "normal_signal"
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        assert "risk_state" in spec
        assert "warning_signal" in spec
        assert "critical_signal" in spec
        assert "normal_signal" in spec

