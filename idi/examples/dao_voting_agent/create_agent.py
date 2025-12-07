#!/usr/bin/env python3
"""Create DAO voting agent example."""

from pathlib import Path
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

# Create agent schema
schema = AgentSchema(
    name="dao_voting_agent",
    strategy="custom",
    streams=(
        # Inputs: 5 voters
        StreamConfig(name="voter1", stream_type="sbf"),
        StreamConfig(name="voter2", stream_type="sbf"),
        StreamConfig(name="voter3", stream_type="sbf"),
        StreamConfig(name="voter4", stream_type="sbf"),
        StreamConfig(name="voter5", stream_type="sbf"),
        StreamConfig(name="reset", stream_type="sbf"),
        
        # Outputs
        StreamConfig(name="consensus", stream_type="sbf", is_input=False),
        StreamConfig(name="proposal_active", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        # Unanimous consensus: all must agree
        LogicBlock(
            pattern="unanimous",
            inputs=("voter1", "voter2", "voter3", "voter4", "voter5"),
            output="consensus"
        ),
        
        # Proposal active FSM: set on consensus, reset on signal
        LogicBlock(
            pattern="fsm",
            inputs=("consensus", "reset"),
            output="proposal_active"
        ),
    ),
    num_steps=10,
)

# Generate spec
spec = generate_tau_spec(schema)

# Save to file
output_dir = Path(__file__).parent
spec_path = output_dir / "dao_voting_agent.tau"
spec_path.write_text(spec)

print(f"✅ Generated DAO voting agent: {spec_path}")
print(f"\nSpec preview:\n{spec[:500]}...")

