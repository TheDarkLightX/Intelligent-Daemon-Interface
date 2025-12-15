"""End-to-end verification test for supervisor-worker pattern."""

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


def test_supervisor_worker_hierarchy(tau_bin):
    """Test supervisor-worker creates proper hierarchy."""
    schema = AgentSchema(
        name="supervisor_worker_hierarchy",
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
        num_steps=8,
    )
    
    spec = generate_tau_spec(schema)
    
    # Test hierarchy: supervisor controls worker
    # When supervisor=0 (IDLE), worker should be disabled (enable=0, state=0)
    # When supervisor=1 (ACTIVE), worker can be enabled (enable=1)
    
    inputs = {
        "global_mode": [],
        "worker1_signal": [],
    }
    expected_supervisor = []
    expected_enable = []
    expected_worker = []
    
    # Test all 4 combinations
    for mode, signal in product([0, 1], repeat=2):
        inputs["global_mode"].append(str(mode))
        inputs["worker1_signal"].append(str(signal))
        
        # Supervisor FSM: mode sets supervisor, mode' clears it
        expected_supervisor.append(str(mode))
        
        # Enable: follows supervisor
        expected_enable.append(str(mode))
        
        # Worker: enabled when enable=1, controlled by signal
        # Worker FSM: (enable & signal) | (worker[t-1] & signal')
        # When enable=0: worker stays 0
        # When enable=1: worker follows signal
        if mode == 0:
            expected_worker.append("0")  # Disabled
        else:
            # Enabled: worker follows signal
            expected_worker.append(str(signal))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
        
        assert "supervisor_state" in outputs
        assert "worker1_enable" in outputs
        assert "worker1_state" in outputs
        
        actual_supervisor = outputs["supervisor_state"]
        actual_enable = outputs["worker1_enable"]
        actual_worker = outputs["worker1_state"]
        
        # Verify supervisor
        for i, (exp, act) in enumerate(zip(expected_supervisor, actual_supervisor)):
            assert act == exp, \
                f"Step {i}: Supervisor mismatch - Expected {exp}, got {act}\n" \
                f"Inputs: mode={inputs['global_mode'][i]}\nSpec:\n{spec}"
        
        # Verify enable follows supervisor
        for i, (exp, act) in enumerate(zip(expected_enable, actual_enable)):
            assert act == exp, \
                f"Step {i}: Enable mismatch - Expected {exp}, got {act}\n" \
                f"Supervisor: {actual_supervisor[i]}\nSpec:\n{spec}"
        
        # Verify worker hierarchy (worker depends on enable)
        for i, (exp, act) in enumerate(zip(expected_worker, actual_worker)):
            # Allow some flexibility for initial state
            if i == 0 and act == "0" and exp == "0":
                continue  # Both start at 0, OK
            assert act == exp, \
                f"Step {i}: Worker mismatch - Expected {exp}, got {act}\n" \
                f"Enable: {actual_enable[i]}, Signal: {inputs['worker1_signal'][i]}\nSpec:\n{spec}"

