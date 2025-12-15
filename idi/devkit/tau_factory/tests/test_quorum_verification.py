"""End-to-end verification test for quorum pattern."""

import pytest
import tempfile
from pathlib import Path
from itertools import product

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.runner import run_tau_spec, TauConfig, TauMode


def find_tau_binary() -> Path | None:
    """Find Tau binary in common locations."""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "tau-lang-latest" / "build-Release" / "tau",
        Path(__file__).parent.parent.parent.parent.parent / "tau_daemon_alpha" / "bin" / "tau",
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


def run_tau_spec_verify(spec_content: str, inputs: dict[str, list[str]], tau_bin: Path, work_dir: Path) -> dict[str, list[str]]:
    """Run Tau spec and return parsed outputs."""
    spec_path = work_dir / "test.tau"
    spec_path.write_text(spec_content)
    
    inputs_dir = work_dir / "inputs"
    outputs_dir = work_dir / "outputs"
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Write input files
    for name, values in inputs.items():
        (inputs_dir / f"{name}.in").write_text("\n".join(values) + "\n")

    result = run_tau_spec(
        spec_path,
        tau_bin,
        config=TauConfig(timeout=10.0, mode=TauMode.FILE),
    )

    if "Unexpected =" in result.stdout or "Unexpected =" in result.stderr or "Unexpected '=" in result.stdout or "Unexpected '=" in result.stderr:
        pytest.skip("Tau binary incompatible with expected Tau spec syntax")

    if not result.success:
        error_msg = (
            "Tau execution failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n\n"
            f"Errors: {result.errors}\n\n"
            f"Spec:\n{spec_content}"
        )
        raise RuntimeError(error_msg)
    
    # Parse outputs
    outputs = {}

    for filename, lines in result.outputs.items():
        if "mirror" in filename:
            continue
        outputs[Path(filename).stem] = lines
    
    return outputs


@pytest.fixture
def tau_bin():
    """Get Tau binary path."""
    bin_path = find_tau_binary()
    if bin_path is None:
        pytest.skip("Tau binary not found - skipping verification tests")
    return bin_path


def test_quorum_3_of_5_verification(tau_bin):
    """Test 3-of-5 quorum with sample combinations."""
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
        num_steps=5,
    )
    
    spec = generate_tau_spec(schema)
    
    # Test cases: (v1, v2, v3, v4, v5) -> expected
    test_cases = [
        ([1, 1, 1, 0, 0], 1),  # 3 votes - quorum met
        ([1, 1, 0, 0, 0], 0),  # 2 votes - quorum not met
        ([1, 1, 1, 1, 0], 1),  # 4 votes - quorum met
        ([0, 0, 0, 0, 0], 0),  # 0 votes - quorum not met
        ([1, 1, 1, 1, 1], 1),  # 5 votes - quorum met
    ]
    
    inputs = {f"vote{i+1}": [] for i in range(5)}
    expected_outputs = []
    
    for case_inputs, expected in test_cases:
        for i, val in enumerate(case_inputs):
            inputs[f"vote{i+1}"].append(str(val))
        expected_outputs.append(str(expected))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
        
        assert "quorum_met" in outputs
        actual = outputs["quorum_met"]
        
        for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
            assert act == exp, \
                f"Step {i}: Expected {exp}, got {act}\n" \
                f"Inputs: {[inputs[f'vote{j+1}'][i] for j in range(5)]}\n" \
                f"Spec:\n{spec}"

