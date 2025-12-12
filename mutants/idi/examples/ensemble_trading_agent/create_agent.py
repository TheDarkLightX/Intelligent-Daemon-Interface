#!/usr/bin/env python3
"""Create ensemble trading agent example."""

from pathlib import Path
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

# Create agent schema
schema = AgentSchema(
    name="ensemble_trading_agent",
    strategy="custom",
    streams=(
        # Inputs: Three independent agents
        StreamConfig(name="agent1_buy", stream_type="sbf"),
        StreamConfig(name="agent2_buy", stream_type="sbf"),
        StreamConfig(name="agent3_buy", stream_type="sbf"),
        StreamConfig(name="agent_sell", stream_type="sbf"),
        
        # Outputs
        StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        # 2-of-3 majority voting
        LogicBlock(
            pattern="majority",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="majority_buy",
            params={"threshold": 2, "total": 3}
        ),
        
        # Position FSM: buy on majority, sell on signal
        LogicBlock(
            pattern="fsm",
            inputs=("majority_buy", "agent_sell"),
            output="position"
        ),
    ),
    num_steps=10,
)

# Generate spec
spec = generate_tau_spec(schema)

# Save to file
output_dir = Path(__file__).parent
spec_path = output_dir / "ensemble_trading_agent.tau"
spec_path.write_text(spec)

print(f"✅ Generated ensemble trading agent: {spec_path}")
print(f"\nSpec preview:\n{spec[:500]}...")

