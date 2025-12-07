"""Majority Vote Agent Example

Demonstrates an ensemble agent using majority voting pattern.
"""

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from pathlib import Path


def create_majority_vote_agent():
    """Create a majority vote ensemble agent."""
    schema = AgentSchema(
        name="majority_vote_agent",
        strategy="ensemble",
        streams=(
            StreamConfig(name="agent1_vote", stream_type="sbf"),
            StreamConfig(name="agent2_vote", stream_type="sbf"),
            StreamConfig(name="agent3_vote", stream_type="sbf"),
            StreamConfig(name="ensemble_decision", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(
                pattern="majority",
                inputs=("agent1_vote", "agent2_vote", "agent3_vote"),
                output="ensemble_decision",
                params={
                    "threshold": 2,
                    "total": 3
                }
            ),
        ),
    )
    
    return schema


def generate_example():
    """Generate spec and example inputs."""
    schema = create_majority_vote_agent()
    spec = generate_tau_spec(schema)
    
    # Create example directory
    example_dir = Path(__file__).parent / "majority_vote_agent"
    example_dir.mkdir(exist_ok=True)
    (example_dir / "inputs").mkdir(exist_ok=True)
    (example_dir / "outputs").mkdir(exist_ok=True)
    
    # Write spec
    spec_file = example_dir / "agent.tau"
    spec_file.write_text(spec)
    print(f"✅ Generated spec: {spec_file}")
    
    # Generate example inputs (2-of-3 majority)
    inputs_file = example_dir / "inputs" / "agent1_vote.in"
    inputs_file.write_text("1\n1\n0\n0\n1\n")
    
    inputs_file = example_dir / "inputs" / "agent2_vote.in"
    inputs_file.write_text("1\n0\n1\n0\n0\n")
    
    inputs_file = example_dir / "inputs" / "agent3_vote.in"
    inputs_file.write_text("0\n1\n1\n0\n1\n")
    
    print(f"✅ Generated example inputs in {example_dir / 'inputs'}")
    print()
    print("Expected outputs (2-of-3 majority):")
    print("  Step 0: 1,1,0 -> 1 (2 agree)")
    print("  Step 1: 1,0,1 -> 1 (2 agree)")
    print("  Step 2: 0,1,1 -> 1 (2 agree)")
    print("  Step 3: 0,0,0 -> 0 (0 agree)")
    print("  Step 4: 1,0,1 -> 1 (2 agree)")


if __name__ == "__main__":
    generate_example()

