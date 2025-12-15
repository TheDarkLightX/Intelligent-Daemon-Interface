"""Test that generated Tau specs actually work with Tau binary."""

import pytest
import tempfile
from pathlib import Path

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.test_harness import generate_sample_inputs, write_inputs
from idi.devkit.tau_factory.runner import run_tau_spec, TauConfig, TauMode


def find_tau_binary() -> Path | None:
    """Find Tau binary in common locations."""
    # Start from repo root (idi/devkit/tau_factory/tests -> go up 4 levels to IDI)
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    
    possible_paths = [
        # Primary: external/tau-lang in the workspace
        repo_root / "external" / "tau-lang" / "build-Release" / "tau",
        # Also check parent directory structure
        repo_root.parent / "external" / "tau-lang" / "build-Release" / "tau",
        # Legacy paths
        repo_root / "tau-lang-latest" / "build-Release" / "tau",
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
        Path.home() / "Downloads" / "tau-lang-latest" / "build-Release" / "tau",
        Path.home() / "Downloads" / "IDI" / "external" / "tau-lang" / "build-Release" / "tau",
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
        result = run_tau_spec(spec_path, tau_bin, config=TauConfig(timeout=10.0))

        if not result.success:
            pytest.fail(
                "Tau execution failed:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"Errors: {result.errors}\n\n"
                f"Generated spec:\n{spec}"
            )

        assert len(result.outputs) > 0, f"No output files created. Spec:\n{spec}"


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

        result = run_tau_spec(spec_path, tau_bin, config=TauConfig(timeout=10.0))
        if not result.success:
            pytest.fail(f"Tau execution failed: {result.errors}\n\nSpec:\n{spec}")

        # FSM logic: o[t] = buy[t] | (o[t-1] & sell[t]'), initial o[0]=0
        # Tau outputs include initial condition at t=0, then formula results
        # Verify output exists and has expected structure
        actual = result.outputs.get("position.out")
        assert actual is not None, "FSM should produce position.out"
        assert len(actual) >= 3, f"FSM should produce at least 3 output values, got {len(actual)}"
        assert actual[0] == "0", f"FSM initial condition should be 0, got {actual[0]}"


@pytest.mark.skip(reason="Experimental (-x) mode uses interactive step loop that blocks with stdin=DEVNULL; requires tau-lang CLI --steps flag or non-interactive mode")
def test_experimental_file_mode_operator_support(tau_bin):
    spec = """\
i0:sbf = in file(\"inputs/c.in\").
i1:sbf = in file(\"inputs/a.in\").
i2:sbf = in file(\"inputs/b.in\").
i3:bv[8] = in file(\"inputs/x.in\").

i0:sbf = out file(\"outputs/c_mirror.out\").
i1:sbf = out file(\"outputs/a_mirror.out\").
i2:sbf = out file(\"outputs/b_mirror.out\").
i3:bv[8] = out file(\"outputs/x_mirror.out\").

o0:sbf = out file(\"outputs/ternary.out\").
o1:sbf = out file(\"outputs/gte.out\").

sel0() := (i0[0] = 1:sbf ? (i1[0] = 1:sbf) : (i2[0] = 1:sbf)).
sel1() := (i0[1] = 1:sbf ? (i1[1] = 1:sbf) : (i2[1] = 1:sbf)).
sel2() := (i0[2] = 1:sbf ? (i1[2] = 1:sbf) : (i2[2] = 1:sbf)).

((sel0() -> o0[0] = 1:sbf) && (!sel0() -> o0[0] = 0:sbf) && (i3[0] >= {10}:bv[8] -> o1[0] = 1:sbf) && (i3[0] !>= {10}:bv[8] -> o1[0] = 0:sbf) && (sel1() -> o0[1] = 1:sbf) && (!sel1() -> o0[1] = 0:sbf) && (i3[1] >= {10}:bv[8] -> o1[1] = 1:sbf) && (i3[1] !>= {10}:bv[8] -> o1[1] = 0:sbf) && (sel2() -> o0[2] = 1:sbf) && (!sel2() -> o0[2] = 0:sbf) && (i3[2] >= {10}:bv[8] -> o1[2] = 1:sbf) && (i3[2] !>= {10}:bv[8] -> o1[2] = 0:sbf)).
"""

    inputs = {
        "c": ["1", "0", "1"],
        "a": ["1", "1", "0"],
        "b": ["0", "1", "1"],
        "x": ["5", "10", "11"],
    }

    expected_o0 = ["1", "1", "0"]
    expected_o1 = ["0", "1", "1"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        spec_path = tmp_path / "operators_x.tau"
        spec_path.write_text(spec)

        inputs_dir = tmp_path / "inputs"
        outputs_dir = tmp_path / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()

        for name, values in inputs.items():
            (inputs_dir / f"{name}.in").write_text("\n".join(values) + "\n")

        result = run_tau_spec(
            spec_path,
            tau_bin,
            config=TauConfig(mode=TauMode.FILE, experimental=True, timeout=10.0),
        )

        if not result.success:
            pytest.fail(
                "Tau execution failed:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"Errors: {result.errors}\n\n"
                f"Spec:\n{spec}"
            )

        assert result.outputs.get("ternary.out") == expected_o0
        assert result.outputs.get("gte.out") == expected_o1


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

        result = run_tau_spec(spec_path, tau_bin, config=TauConfig(timeout=10.0))
        if not result.success:
            pytest.fail(f"Tau execution failed: {result.errors}\n\nSpec:\n{spec}")

        # Counter (toggle) logic: o[t] = (event[t] & o[t-1]') | (event[t]' & o[t-1])
        # Tau outputs include initial condition at t=0, then formula results
        # Verify output exists and has expected structure
        actual = result.outputs.get("count.out")
        assert actual is not None, "Counter should produce count.out"
        assert len(actual) >= 3, f"Counter should produce at least 3 output values, got {len(actual)}"
        assert actual[0] == "0", f"Counter initial condition should be 0, got {actual[0]}"


def test_sbf_ternary_and_recurrence_operators(tau_bin):
    schema = AgentSchema(
        name="test_ternary_recurrence",
        strategy="momentum",
        streams=(
            StreamConfig(name="selector", stream_type="sbf"),
            StreamConfig(name="input_a", stream_type="sbf"),
            StreamConfig(name="input_b", stream_type="sbf"),
            StreamConfig(name="selected", stream_type="sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="custom", inputs=("selector", "input_a", "input_b"), output="selected",
                       params={"expression": "(i0[t] & i1[t]) | (i0[t]' & i2[t])"}),
        ),
        num_steps=4,
    )

    spec = generate_tau_spec(schema)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        spec_path = tmp_path / "test_ternary.tau"
        spec_path.write_text(spec)

        inputs_dir = tmp_path / "inputs"
        outputs_dir = tmp_path / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()

        (inputs_dir / "selector.in").write_text("1\n0\n1\n0\n")
        (inputs_dir / "input_a.in").write_text("1\n1\n0\n0\n")
        (inputs_dir / "input_b.in").write_text("0\n1\n1\n0\n")

        result = run_tau_spec(spec_path, tau_bin, config=TauConfig(timeout=10.0))
        if not result.success:
            pytest.fail(f"Tau execution failed: {result.errors}\n\nSpec:\n{spec}")

        actual = result.outputs.get("selected.out")
        assert actual is not None, "Should produce selected.out"
        assert len(actual) >= 4, f"Should produce at least 4 output values, got {len(actual)}"

