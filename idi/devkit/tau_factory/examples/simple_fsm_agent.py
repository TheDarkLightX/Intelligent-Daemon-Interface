"""Simple FSM Agent Example

Demonstrates a basic FSM agent that tracks position state based on buy/sell signals.
"""

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from pathlib import Path


def create_simple_fsm_agent():
    """Create a simple FSM agent."""
    schema = AgentSchema(
        name="simple_fsm_agent",
        strategy="custom",
        streams=(
            StreamConfig(name="buy", stream_type="sbf"),
            StreamConfig(name="sell", stream_type="sbf"),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(
                pattern="fsm",
                inputs=("buy", "sell"),
                output="position",
            ),
        ),
    )
    
    return schema


def generate_example():
    """Generate spec and example inputs."""
    schema = create_simple_fsm_agent()
    spec = generate_tau_spec(schema)
    
    # Create example directory
    example_dir = Path(__file__).parent / "simple_fsm_agent"
    example_dir.mkdir(exist_ok=True)
    (example_dir / "inputs").mkdir(exist_ok=True)
    (example_dir / "outputs").mkdir(exist_ok=True)
    
    # Write spec
    spec_file = example_dir / "agent.tau"
    spec_file.write_text(spec)
    print(f"✅ Generated spec: {spec_file}")
    
    # Generate example inputs
    inputs_file = example_dir / "inputs" / "buy.in"
    inputs_file.write_text("0\n1\n0\n0\n1\n0\n")
    
    inputs_file = example_dir / "inputs" / "sell.in"
    inputs_file.write_text("0\n0\n1\n0\n0\n1\n")
    
    print(f"✅ Generated example inputs in {example_dir / 'inputs'}")
    print()
    print("To run:")
    print(f"  cd {example_dir}")
    print("  echo -e 'r (agent.tau)\\nn\\nn\\nn\\nn\\nn\\nn\\nq' | tau")
    print()
    print("Expected outputs:")
    print("  position.out: 0, 1, 1, 1, 1, 0")


if __name__ == "__main__":
    generate_example()

