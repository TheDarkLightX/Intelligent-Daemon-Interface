"""Test that generated Tau specs actually work with Tau binary."""

import pytest
import tempfile
import subprocess
from pathlib import Path

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.test_harness import generate_sample_inputs, write_inputs


def find_tau_binary() -> Path | None:
    """Find Tau binary in common locations."""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "tau-lang-latest" / "build-Release" / "tau",
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
        Path.home() / "Downloads" / "tau-lang-latest" / "build-Release" / "tau",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


@pytest.fixture
def tau_bin():
    """Get Tau binary path."""
    bin_path = find_tau_binary()
    if bin_path is None:
        pytest.skip("Tau binary not found")
    return bin_path


def test_generated_spec_executes_successfully(tau_bin):
    """Test that a generated spec executes without errors."""
    schema = AgentSchema(
        name="test_execution",
        strategy="momentum",
        streams=(
            StreamConfig(name="q_buy", stream_type="sbf"),
            StreamConfig(name="q_sell", stream_type="sbf"),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
            StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
            StreamConfig(name="sell_signal", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="fsm", inputs=("q_buy", "q_sell"), output="position"),
            LogicBlock(pattern="passthrough", inputs=("q_buy",), output="buy_signal"),
            LogicBlock(pattern="passthrough", inputs=("q_sell",), output="sell_signal"),
        ),
        num_steps=5,
    )
    
    # Generate spec
    spec = generate_tau_spec(schema)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        spec_path = tmp_path / f"{schema.name}.tau"
        spec_path.write_text(spec)
        
        # Create inputs
        inputs_dir = tmp_path / "inputs"
        outputs_dir = tmp_path / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()
        
        inputs = generate_sample_inputs(schema, num_ticks=5)
        write_inputs(inputs_dir, inputs)
        
        # Run Tau
        try:
            proc = subprocess.run(
                [str(tau_bin)],
                stdin=spec_path.open("rb"),
                capture_output=True,
                timeout=10,
                cwd=str(tmp_path),
            )
            
            # Check for syntax errors
            stderr_text = proc.stderr.decode("utf-8", errors="ignore")
            stdout_text = proc.stdout.decode("utf-8", errors="ignore")
            
            # Check for common Tau errors
            error_indicators = [
                "Syntax Error",
                "Unexpected",
                "Error",
                "unsat",
            ]
            
            has_error = any(indicator in stderr_text or indicator in stdout_text 
                          for indicator in error_indicators)
            
            if has_error:
                pytest.fail(f"Tau execution failed:\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}\n\nGenerated spec:\n{spec}")
            
            # Check outputs were created
            output_files = list(outputs_dir.glob("*.out"))
            assert len(output_files) > 0, f"No output files created. Spec:\n{spec}"
            
        except subprocess.TimeoutExpired:
            pytest.fail(f"Tau execution timed out. Spec:\n{spec}")
        except Exception as e:
            pytest.fail(f"Failed to run Tau: {e}\n\nSpec:\n{spec}")


def test_fsm_with_initial_condition(tau_bin):
    """Test FSM pattern with proper initial condition."""
    schema = AgentSchema(
        name="test_fsm_init",
        strategy="momentum",
        streams=(
            StreamConfig(name="buy", stream_type="sbf"),
            StreamConfig(name="sell", stream_type="sbf"),
            StreamConfig(name="position", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="position"),
        ),
        num_steps=3,
    )
    
    spec = generate_tau_spec(schema)
    
    # Check spec has initial condition or proper FSM logic
    assert "position" in spec.lower()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        spec_path = tmp_path / "test.tau"
        spec_path.write_text(spec)
        
        inputs_dir = tmp_path / "inputs"
        outputs_dir = tmp_path / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()
        
        # Create simple inputs
        (inputs_dir / "buy.in").write_text("1\n0\n0\n")
        (inputs_dir / "sell.in").write_text("0\n0\n1\n")
        
        proc = subprocess.run(
            [str(tau_bin)],
            stdin=spec_path.open("rb"),
            capture_output=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        
        # Should not have syntax errors
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if "Syntax Error" in stderr or "Unexpected" in stderr:
            pytest.fail(f"Syntax error in generated spec:\n{stderr}\n\nSpec:\n{spec}")


def test_counter_pattern_execution(tau_bin):
    """Test counter pattern generates valid Tau."""
    schema = AgentSchema(
        name="test_counter",
        strategy="momentum",
        streams=(
            StreamConfig(name="event", stream_type="sbf"),
            StreamConfig(name="count", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="counter", inputs=("event",), output="count"),
        ),
        num_steps=3,
    )
    
    spec = generate_tau_spec(schema)
    
    # Counter should have initial condition (uses o0, not count directly)
    assert "[0]" in spec, "Counter should have initial condition"
    assert "[t-1]" in spec, "Counter should reference previous timestep"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        spec_path = tmp_path / "test.tau"
        spec_path.write_text(spec)
        
        inputs_dir = tmp_path / "inputs"
        outputs_dir = tmp_path / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()
        
        (inputs_dir / "event.in").write_text("1\n0\n1\n")
        
        proc = subprocess.run(
            [str(tau_bin)],
            stdin=spec_path.open("rb"),
            capture_output=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if "Syntax Error" in stderr:
            pytest.fail(f"Counter pattern syntax error:\n{stderr}\n\nSpec:\n{spec}")

