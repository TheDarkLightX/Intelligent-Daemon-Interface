"""Trading Agent Example

Demonstrates Entry-Exit FSM pattern for multi-phase trade lifecycle.
"""

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from pathlib import Path


def create_trading_agent():
    """Create a trading agent with Entry-Exit FSM."""
    schema = AgentSchema(
        name="trading_agent",
        strategy="custom",
        streams=(
            StreamConfig(name="entry_signal", stream_type="sbf"),
            StreamConfig(name="exit_signal", stream_type="sbf"),
            StreamConfig(name="phase", stream_type="bv", width=2, is_input=False),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(
                pattern="entry_exit_fsm",
                inputs=("entry_signal", "exit_signal"),
                output="phase",
                params={
                    "phases": ["PRE_TRADE", "IN_TRADE", "POST_TRADE"],
                    "phase_output": "phase",
                    "position_output": "position"
                }
            ),
        ),
    )
    
    return schema


def generate_example():
    """Generate spec and example inputs."""
    schema = create_trading_agent()
    spec = generate_tau_spec(schema)
    
    # Create example directory
    example_dir = Path(__file__).parent / "trading_agent"
    example_dir.mkdir(exist_ok=True)
    (example_dir / "inputs").mkdir(exist_ok=True)
    (example_dir / "outputs").mkdir(exist_ok=True)
    
    # Write spec
    spec_file = example_dir / "agent.tau"
    spec_file.write_text(spec)
    print(f"✅ Generated spec: {spec_file}")
    
    # Generate example inputs
    inputs_file = example_dir / "inputs" / "entry_signal.in"
    inputs_file.write_text("0\n1\n0\n0\n0\n")
    
    inputs_file = example_dir / "inputs" / "exit_signal.in"
    inputs_file.write_text("0\n0\n0\n1\n0\n")
    
    print(f"✅ Generated example inputs in {example_dir / 'inputs'}")
    print()
    print("Expected outputs:")
    print("  phase.out: 0 (PRE_TRADE), 1 (IN_TRADE), 1, 2 (POST_TRADE), 0")
    print("  position.out: 0, 1, 1, 0, 0")


if __name__ == "__main__":
    generate_example()

