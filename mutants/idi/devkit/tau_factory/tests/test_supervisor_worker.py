"""Tests for supervisor-worker hierarchical FSM pattern."""

import pytest

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


class TestSupervisorWorkerPattern:
    """Test supervisor-worker hierarchical FSM pattern."""

    def test_supervisor_worker_basic(self):
        """Test basic supervisor-worker pattern."""
        schema = AgentSchema(
            name="supervisor_worker_basic",
            strategy="custom",
            streams=(
                StreamConfig(name="global_mode", stream_type="sbf"),
                StreamConfig(name="worker1_signal", stream_type="sbf"),
                StreamConfig(name="supervisor_state", stream_type="sbf", is_input=False),
                StreamConfig(name="worker1_enable", stream_type="sbf", is_input=False),
                StreamConfig(name="worker1_state", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="supervisor_worker",
                    inputs=("global_mode", "worker1_signal"),
                    output="supervisor_state",
                    params={
                        "supervisor_inputs": ["global_mode"],
                        "worker_inputs": ["worker1_signal"],
                        "worker_enable_outputs": ["worker1_enable"],
                        "worker_outputs": ["worker1_state"],
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain supervisor FSM
        assert "supervisor_state" in spec
        assert "[t]" in spec
        assert "[t-1]" in spec
        
        # Should contain worker enable
        assert "worker1_enable" in spec
        
        # Should contain worker FSM
        assert "worker1_state" in spec

    def test_supervisor_worker_multiple_workers(self):
        """Test supervisor-worker with multiple workers."""
        schema = AgentSchema(
            name="supervisor_multi_worker",
            strategy="custom",
            streams=(
                StreamConfig(name="global_mode", stream_type="sbf"),
                StreamConfig(name="worker1_signal", stream_type="sbf"),
                StreamConfig(name="worker2_signal", stream_type="sbf"),
                StreamConfig(name="supervisor_state", stream_type="sbf", is_input=False),
                StreamConfig(name="worker1_enable", stream_type="sbf", is_input=False),
                StreamConfig(name="worker2_enable", stream_type="sbf", is_input=False),
                StreamConfig(name="worker1_state", stream_type="sbf", is_input=False),
                StreamConfig(name="worker2_state", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="supervisor_worker",
                    inputs=("global_mode", "worker1_signal", "worker2_signal"),
                    output="supervisor_state",
                    params={
                        "supervisor_inputs": ["global_mode"],
                        "worker_inputs": ["worker1_signal", "worker2_signal"],
                        "worker_enable_outputs": ["worker1_enable", "worker2_enable"],
                        "worker_outputs": ["worker1_state", "worker2_state"],
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        
        # Should contain both workers
        assert "worker1_enable" in spec
        assert "worker2_enable" in spec
        assert "worker1_state" in spec
        assert "worker2_state" in spec

    def test_supervisor_worker_requires_outputs(self):
        """Test that worker outputs must be declared."""
        # This should work - outputs declared
        schema = AgentSchema(
            name="test",
            strategy="custom",
            streams=(
                StreamConfig(name="mode", stream_type="sbf"),
                StreamConfig(name="signal", stream_type="sbf"),
                StreamConfig(name="supervisor", stream_type="sbf", is_input=False),
                StreamConfig(name="enable", stream_type="sbf", is_input=False),
                StreamConfig(name="worker", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="supervisor_worker",
                    inputs=("mode", "signal"),
                    output="supervisor",
                    params={
                        "supervisor_inputs": ["mode"],
                        "worker_inputs": ["signal"],
                        "worker_enable_outputs": ["enable"],
                        "worker_outputs": ["worker"],
                    }
                ),
            ),
        )
        
        spec = generate_tau_spec(schema)
        assert "supervisor" in spec

