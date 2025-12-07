"""End-to-end test harness for Tau agent generation and execution."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from idi.devkit.tau_factory.schema import AgentSchema
from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.runner import run_tau_spec, TauResult
from idi.devkit.tau_factory.validator import validate_agent_outputs, ValidationResult


@dataclass
class TestReport:
    """Complete test report for an agent."""
    schema: AgentSchema
    spec: str
    result: TauResult
    validation: ValidationResult
    spec_path: Optional[Path] = None


def generate_sample_inputs(schema: AgentSchema, num_ticks: int = 10) -> dict[str, list[str]]:
    """Generate sample input files for testing.
    
    Args:
        schema: Agent schema
        num_ticks: Number of timesteps to generate
        
    Returns:
        Dictionary mapping input names to lists of values
    """
    inputs = {}
    input_streams = [s for s in schema.streams if s.is_input]
    
    for stream in input_streams:
        if stream.stream_type == "sbf":
            # Generate alternating 0/1 pattern
            inputs[stream.name] = [str(i % 2) for i in range(num_ticks)]
        elif stream.stream_type == "bv":
            # Generate incrementing values within range
            max_val = min((1 << stream.width) - 1, 100)
            inputs[stream.name] = [str(i % max_val) for i in range(num_ticks)]
    
    return inputs


def write_inputs(inputs_dir: Path, inputs: dict[str, list[str]]) -> None:
    """Write input files to directory.
    
    Args:
        inputs_dir: Directory to write input files
        inputs: Dictionary mapping input names to value lists
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    for name, values in inputs.items():
        input_file = inputs_dir / f"{name}.in"
        input_file.write_text("\n".join(values) + "\n")


def run_end_to_end_test(
    schema: AgentSchema,
    tau_bin: Path,
    num_ticks: int = 10,
    work_dir: Optional[Path] = None,
) -> TestReport:
    """Run complete end-to-end test: generate spec, create inputs, run Tau, validate.
    
    Args:
        schema: Agent schema to test
        tau_bin: Path to Tau binary
        num_ticks: Number of timesteps to test
        work_dir: Optional working directory (creates temp if None)
        
    Returns:
        TestReport with all results
    """
    # Create working directory
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="tau_test_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate spec
    spec = generate_tau_spec(schema)
    spec_path = work_dir / f"{schema.name}.tau"
    spec_path.write_text(spec)
    
    # 2. Generate sample inputs
    inputs = generate_sample_inputs(schema, num_ticks)
    inputs_dir = work_dir / "inputs"
    write_inputs(inputs_dir, inputs)
    
    # 3. Create outputs directory
    outputs_dir = work_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # 4. Run Tau
    result = run_tau_spec(spec_path, tau_bin)
    
    # 5. Validate outputs
    validation = validate_agent_outputs(result.outputs, schema)
    
    return TestReport(
        schema=schema,
        spec=spec,
        result=result,
        validation=validation,
        spec_path=spec_path,
    )

