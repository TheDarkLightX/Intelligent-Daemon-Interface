"""End-to-end verification tests for ensemble patterns - actually run Tau specs and verify outputs."""

import pytest
import tempfile
import subprocess
from pathlib import Path
from itertools import product

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec


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
    
    # Run Tau
    proc = subprocess.run(
        [str(tau_bin)],
        stdin=spec_path.open("rb"),
        capture_output=True,
        timeout=10,
        cwd=str(work_dir),
    )
    
    # Check for errors
    stderr = proc.stderr.decode("utf-8", errors="ignore")
    stdout = proc.stdout.decode("utf-8", errors="ignore")
    
    if proc.returncode != 0 or "Error" in stderr or "Error" in stdout or "unsat" in stderr:
        error_msg = f"Tau execution failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n\nSpec:\n{spec_content}"
        raise RuntimeError(error_msg)
    
    # Parse outputs
    outputs = {}
    for out_file in sorted(outputs_dir.glob("*.out")):
        if "mirror" in out_file.name:
            continue  # Skip mirror outputs
        content = out_file.read_text().strip()
        outputs[out_file.stem] = [line.strip() for line in content.split("\n") if line.strip()]
    
    return outputs


@pytest.fixture
def tau_bin():
    """Get Tau binary path."""
    bin_path = find_tau_binary()
    if bin_path is None:
        pytest.skip("Tau binary not found - skipping verification tests")
    return bin_path


class TestMajorityVerification:
    """Verify majority pattern with truth table testing."""
    
    def test_2_of_3_majority_truth_table(self, tau_bin):
        """Test 2-of-3 majority with all 8 input combinations."""
        schema = AgentSchema(
            name="majority_2of3",
            strategy="custom",
            streams=(
                StreamConfig(name="a", stream_type="sbf"),
                StreamConfig(name="b", stream_type="sbf"),
                StreamConfig(name="c", stream_type="sbf"),
                StreamConfig(name="majority", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("a", "b", "c"),
                    output="majority",
                    params={"threshold": 2, "total": 3}
                ),
            ),
            num_steps=8,
        )
        
        spec = generate_tau_spec(schema)
        
        # Generate all 8 combinations (2^3)
        inputs = {
            "a": [],
            "b": [],
            "c": [],
        }
        expected_outputs = []
        
        for i, (a_val, b_val, c_val) in enumerate(product([0, 1], repeat=3)):
            inputs["a"].append(str(a_val))
            inputs["b"].append(str(b_val))
            inputs["c"].append(str(c_val))
            
            # 2-of-3 majority: at least 2 must be 1
            count = a_val + b_val + c_val
            expected = 1 if count >= 2 else 0
            expected_outputs.append(str(expected))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "majority" in outputs, f"Output 'majority' not found. Outputs: {list(outputs.keys())}"
            actual = outputs["majority"]
            
            assert len(actual) == len(expected_outputs), \
                f"Output length mismatch: expected {len(expected_outputs)}, got {len(actual)}"
            
            for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
                assert act == exp, \
                    f"Step {i}: Inputs a={inputs['a'][i]}, b={inputs['b'][i]}, c={inputs['c'][i]} - " \
                    f"Expected majority={exp}, got {act}\nSpec:\n{spec}"
    
    def test_3_of_5_majority_verification(self, tau_bin):
        """Test 3-of-5 majority with sample combinations."""
        schema = AgentSchema(
            name="majority_3of5",
            strategy="custom",
            streams=(
                StreamConfig(name="a1", stream_type="sbf"),
                StreamConfig(name="a2", stream_type="sbf"),
                StreamConfig(name="a3", stream_type="sbf"),
                StreamConfig(name="a4", stream_type="sbf"),
                StreamConfig(name="a5", stream_type="sbf"),
                StreamConfig(name="majority", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("a1", "a2", "a3", "a4", "a5"),
                    output="majority",
                    params={"threshold": 3, "total": 5}
                ),
            ),
            num_steps=5,
        )
        
        spec = generate_tau_spec(schema)
        
        # Test cases: (a1, a2, a3, a4, a5) -> expected
        test_cases = [
            ([1, 1, 1, 0, 0], 1),  # 3 votes - majority
            ([1, 1, 0, 0, 0], 0),  # 2 votes - no majority
            ([1, 1, 1, 1, 0], 1),  # 4 votes - majority
            ([0, 0, 0, 0, 0], 0),  # 0 votes - no majority
            ([1, 1, 1, 1, 1], 1),  # 5 votes - majority
        ]
        
        inputs = {f"a{i+1}": [] for i in range(5)}
        expected_outputs = []
        
        for case_inputs, expected in test_cases:
            for i, val in enumerate(case_inputs):
                inputs[f"a{i+1}"].append(str(val))
            expected_outputs.append(str(expected))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "majority" in outputs
            actual = outputs["majority"]
            
            for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
                assert act == exp, \
                    f"Step {i}: Expected {exp}, got {act}\n" \
                    f"Inputs: {[inputs[f'a{j+1}'][i] for j in range(5)]}\n" \
                    f"Spec:\n{spec}"


class TestUnanimousVerification:
    """Verify unanimous pattern with truth table testing."""
    
    def test_unanimous_3_agents_truth_table(self, tau_bin):
        """Test unanimous with all 8 input combinations."""
        schema = AgentSchema(
            name="unanimous_3",
            strategy="custom",
            streams=(
                StreamConfig(name="a", stream_type="sbf"),
                StreamConfig(name="b", stream_type="sbf"),
                StreamConfig(name="c", stream_type="sbf"),
                StreamConfig(name="unanimous", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="unanimous",
                    inputs=("a", "b", "c"),
                    output="unanimous"
                ),
            ),
            num_steps=8,
        )
        
        spec = generate_tau_spec(schema)
        
        # Generate all 8 combinations
        inputs = {
            "a": [],
            "b": [],
            "c": [],
        }
        expected_outputs = []
        
        for a_val, b_val, c_val in product([0, 1], repeat=3):
            inputs["a"].append(str(a_val))
            inputs["b"].append(str(b_val))
            inputs["c"].append(str(c_val))
            
            # Unanimous: all must be 1
            expected = 1 if (a_val == 1 and b_val == 1 and c_val == 1) else 0
            expected_outputs.append(str(expected))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "unanimous" in outputs
            actual = outputs["unanimous"]
            
            assert len(actual) == len(expected_outputs)
            
            for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
                assert act == exp, \
                    f"Step {i}: Inputs a={inputs['a'][i]}, b={inputs['b'][i]}, c={inputs['c'][i]} - " \
                    f"Expected unanimous={exp}, got {act}\nSpec:\n{spec}"


class TestCustomExpressionVerification:
    """Verify custom boolean expressions."""
    
    def test_custom_and_expression(self, tau_bin):
        """Test custom AND expression."""
        schema = AgentSchema(
            name="custom_and",
            strategy="custom",
            streams=(
                StreamConfig(name="a", stream_type="sbf"),
                StreamConfig(name="b", stream_type="sbf"),
                StreamConfig(name="result", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="custom",
                    inputs=("a", "b"),
                    output="result",
                    params={"expression": "a[t] & b[t]"}
                ),
            ),
            num_steps=4,
        )
        
        spec = generate_tau_spec(schema)
        
        # Test all 4 combinations
        inputs = {
            "a": ["0", "0", "1", "1"],
            "b": ["0", "1", "0", "1"],
        }
        expected_outputs = ["0", "0", "0", "1"]  # AND: only (1,1) -> 1
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "result" in outputs
            actual = outputs["result"]
            
            for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
                assert act == exp, \
                    f"Step {i}: a={inputs['a'][i]}, b={inputs['b'][i]} - " \
                    f"Expected {exp}, got {act}\nSpec:\n{spec}"
    
    def test_custom_xor_expression(self, tau_bin):
        """Test custom XOR expression."""
        schema = AgentSchema(
            name="custom_xor",
            strategy="custom",
            streams=(
                StreamConfig(name="a", stream_type="sbf"),
                StreamConfig(name="b", stream_type="sbf"),
                StreamConfig(name="result", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="custom",
                    inputs=("a", "b"),
                    output="result",
                    params={"expression": "(a[t] & b[t]') | (a[t]' & b[t])"}
                ),
            ),
            num_steps=4,
        )
        
        spec = generate_tau_spec(schema)
        
        inputs = {
            "a": ["0", "0", "1", "1"],
            "b": ["0", "1", "0", "1"],
        }
        expected_outputs = ["0", "1", "1", "0"]  # XOR: different -> 1
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "result" in outputs
            actual = outputs["result"]
            
            for i, (exp, act) in enumerate(zip(expected_outputs, actual)):
                assert act == exp, \
                    f"Step {i}: a={inputs['a'][i]}, b={inputs['b'][i]} - " \
                    f"Expected XOR={exp}, got {act}\nSpec:\n{spec}"


class TestEnsembleIntegrationVerification:
    """Verify ensemble agents with multiple patterns."""
    
    def test_ensemble_agent_majority_and_unanimous(self, tau_bin):
        """Test ensemble agent with both majority and unanimous."""
        schema = AgentSchema(
            name="ensemble_integration",
            strategy="custom",
            streams=(
                StreamConfig(name="agent1", stream_type="sbf"),
                StreamConfig(name="agent2", stream_type="sbf"),
                StreamConfig(name="agent3", stream_type="sbf"),
                StreamConfig(name="majority", stream_type="sbf", is_input=False),
                StreamConfig(name="unanimous", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(
                    pattern="majority",
                    inputs=("agent1", "agent2", "agent3"),
                    output="majority",
                    params={"threshold": 2, "total": 3}
                ),
                LogicBlock(
                    pattern="unanimous",
                    inputs=("agent1", "agent2", "agent3"),
                    output="unanimous"
                ),
            ),
            num_steps=8,
        )
        
        spec = generate_tau_spec(schema)
        
        # Test all combinations
        inputs = {
            "agent1": [],
            "agent2": [],
            "agent3": [],
        }
        expected_majority = []
        expected_unanimous = []
        
        for a1, a2, a3 in product([0, 1], repeat=3):
            inputs["agent1"].append(str(a1))
            inputs["agent2"].append(str(a2))
            inputs["agent3"].append(str(a3))
            
            # Majority: 2 or more votes
            count = a1 + a2 + a3
            expected_majority.append("1" if count >= 2 else "0")
            
            # Unanimous: all must be 1
            expected_unanimous.append("1" if (a1 == 1 and a2 == 1 and a3 == 1) else "0")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_tau_spec_verify(spec, inputs, tau_bin, Path(tmpdir))
            
            assert "majority" in outputs
            assert "unanimous" in outputs
            
            actual_majority = outputs["majority"]
            actual_unanimous = outputs["unanimous"]
            
            for i, (exp_maj, act_maj, exp_una, act_una) in enumerate(
                zip(expected_majority, actual_majority, expected_unanimous, actual_unanimous)
            ):
                assert act_maj == exp_maj, \
                    f"Step {i}: Majority mismatch - Expected {exp_maj}, got {act_maj}\n" \
                    f"Inputs: {[inputs[f'agent{j+1}'][i] for j in range(3)]}\nSpec:\n{spec}"
                
                assert act_una == exp_una, \
                    f"Step {i}: Unanimous mismatch - Expected {exp_una}, got {act_una}\n" \
                    f"Inputs: {[inputs[f'agent{j+1}'][i] for j in range(3)]}\nSpec:\n{spec}"

