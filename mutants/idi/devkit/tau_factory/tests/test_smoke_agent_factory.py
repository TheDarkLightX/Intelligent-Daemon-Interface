"""Smoke tests that exercise the Agent Factory end-to-end.

These tests generate Tau specs from parameterized AgentSchema instances,
materialize input files, run the Tau binary, and assert the produced outputs
match the expected behavior for simple patterns. They serve as a user-facing
sanity check that the factory can generate runnable intelligent agents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.runner import run_tau_spec
from idi.devkit.tau_factory.schema import AgentSchema, LogicBlock, StreamConfig


def _find_tau_binary() -> Path | None:
    """Locate a Tau binary in common workspace locations."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "tau-lang-latest"
        / "build-Release"
        / "tau",
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
        Path.home() / "Downloads" / "tau-lang-latest" / "build-Release" / "tau",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


@pytest.fixture(scope="session")
def tau_bin():
    """Provide Tau binary path or skip if unavailable."""
    bin_path = _find_tau_binary()
    if bin_path is None:
        pytest.skip("Tau binary not found for smoke tests")
    return bin_path


def _write_inputs(root: Path, inputs: dict[str, list[str]]) -> None:
    """Create inputs directory and populate .in files."""
    inputs_dir = root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in inputs.items():
        (inputs_dir / f"{name}.in").write_text("\n".join(lines) + "\n")


def _simulate_passthrough(inputs: list[str]) -> list[str]:
    return inputs


def _simulate_majority(vectors: list[list[str]], threshold: int) -> list[str]:
    out = []
    for cols in zip(*vectors):
        votes = sum(1 for c in cols if c == "1")
        out.append("1" if votes >= threshold else "0")
    return out


def _simulate_vote_or(vectors: list[list[str]]) -> list[str]:
    return ["1" if any(c == "1" for c in cols) else "0" for cols in zip(*vectors)]


def _simulate_unanimous_and(vectors: list[list[str]]) -> list[str]:
    return ["1" if all(c == "1" for c in cols) else "0" for cols in zip(*vectors)]


def _simulate_counter(events: list[str], initial: str = "0") -> list[str]:
    state = int(initial)
    out = [initial]
    for e in events:
        bit = int(e)
        state = state ^ bit  # toggle on 1
        out.append(str(state))
    return out


def _simulate_accumulator(vals: list[str], initial: int = 0) -> list[str]:
    acc = initial
    out = [str(acc)]
    for v in vals:
        acc += int(v)
        out.append(str(acc))
    return out


def _simulate_fsm(buys: list[str], sells: list[str], initial: str = "0") -> list[str]:
    state = int(initial)
    out = [initial]
    for b, s in zip(buys, sells):
        b_i = int(b)
        s_i = int(s)
        state = b_i or (state and (not s_i))
        out.append(str(int(state)))
    return out


def _run_and_assert(schema: AgentSchema, inputs: dict[str, list[str]], expected_outputs: dict[str, list[str]], tau_bin: Path):
    """Generate spec, write inputs, run Tau, and check outputs."""
    spec = generate_tau_spec(schema)
    with tempfile.TemporaryDirectory(prefix="tau_smoke_") as tmpdir:
        root = Path(tmpdir)
        spec_path = root / f"{schema.name}.tau"
        spec_path.write_text(spec)

        _write_inputs(root, inputs)
        (root / "outputs").mkdir(exist_ok=True)

        result = run_tau_spec(spec_path, tau_bin)
        assert result.success, f"Tau execution failed: {result.errors}\nSpec:\n{spec}"

        # Check expected outputs exactly match
        for out_name, out_values in expected_outputs.items():
            key = f"{out_name}.out"
            assert key in result.outputs, f"Missing output {key}; got {list(result.outputs.keys())}"
            assert result.outputs[key] == out_values, f"Output {key} mismatch. Expected {out_values}, got {result.outputs[key]}"


def test_smoke_passthrough_echo(tau_bin):
    """Passthrough should echo input to output end-to-end."""
    schema = AgentSchema(
        name="smoke_echo",
        strategy="custom",
        streams=(
            StreamConfig("input_signal", "sbf", is_input=True),
            StreamConfig("echo", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="passthrough", inputs=("input_signal",), output="echo"),
        ),
        num_steps=3,
        include_mirrors=False,
    )

    inputs = {"input_signal": ["1", "0", "1"]}
    expected_outputs = {"echo": _simulate_passthrough(inputs["input_signal"])}

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)


def test_smoke_majority_threshold(tau_bin):
    """Majority (2-of-3) should emit 1 only when at least two votes are 1."""
    schema = AgentSchema(
        name="smoke_majority",
        strategy="custom",
        streams=(
            StreamConfig("v1", "sbf", is_input=True),
            StreamConfig("v2", "sbf", is_input=True),
            StreamConfig("v3", "sbf", is_input=True),
            StreamConfig("decision", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(
                pattern="majority",
                inputs=("v1", "v2", "v3"),
                output="decision",
                params={"threshold": 2},
            ),
        ),
        num_steps=4,
        include_mirrors=False,
    )

    inputs = {
        "v1": ["1", "0", "1", "0"],
        "v2": ["0", "1", "1", "0"],
        "v3": ["1", "1", "0", "0"],
    }
    expected_outputs = {
        "decision": _simulate_majority(
            [inputs["v1"], inputs["v2"], inputs["v3"]],
            threshold=2,
        ),
    }

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)


def test_smoke_counter_toggle(tau_bin):
    """Counter pattern should toggle on event pulses."""
    schema = AgentSchema(
        name="smoke_counter",
        strategy="custom",
        streams=(
            StreamConfig("event", "sbf", is_input=True),
            StreamConfig("count", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="counter", inputs=("event",), output="count"),
        ),
        num_steps=5,
        include_mirrors=False,
    )

    inputs = {"event": ["1", "0", "1", "1", "0"]}
    expected_outputs = {"count": _simulate_counter(inputs["event"], initial="0")}

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)


def test_smoke_accumulator_bv(tau_bin):
    """Accumulator (bv) should sum sbf inputs over time."""
    schema = AgentSchema(
        name="smoke_accumulator",
        strategy="custom",
        streams=(
            StreamConfig("tick", "bv", width=8, is_input=True),
            StreamConfig("sum", "bv", width=8, is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="accumulator", inputs=("tick",), output="sum"),
        ),
        num_steps=4,
        include_mirrors=False,
    )

    inputs = {"tick": ["1", "0", "1", "1"]}
    expected_outputs = {"sum": _simulate_accumulator(inputs["tick"], initial=0)}

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)


def test_smoke_fsm_buy_sell(tau_bin):
    """FSM should set position on buy and clear on sell."""
    schema = AgentSchema(
        name="smoke_fsm",
        strategy="custom",
        streams=(
            StreamConfig("buy", "sbf", is_input=True),
            StreamConfig("sell", "sbf", is_input=True),
            StreamConfig("position", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="position"),
        ),
        num_steps=5,
        include_mirrors=False,
    )

    inputs = {
        "buy": ["1", "0", "0", "0", "0"],
        "sell": ["0", "0", "1", "0", "0"],
    }
    expected_outputs = {
        "position": _simulate_fsm(inputs["buy"], inputs["sell"], initial="0")
    }


def test_smoke_vote_or(tau_bin):
    """Vote pattern should OR all inputs."""
    schema = AgentSchema(
        name="smoke_vote",
        strategy="custom",
        streams=(
            StreamConfig("a", "sbf", is_input=True),
            StreamConfig("b", "sbf", is_input=True),
            StreamConfig("decision", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="vote", inputs=("a", "b"), output="decision"),
        ),
        num_steps=4,
        include_mirrors=False,
    )

    inputs = {
        "a": ["0", "1", "0", "0"],
        "b": ["0", "0", "1", "0"],
    }
    expected_outputs = {
        "decision": _simulate_vote_or([inputs["a"], inputs["b"]]),
    }

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)


def test_smoke_unanimous_and(tau_bin):
    """Unanimous pattern should AND all inputs."""
    schema = AgentSchema(
        name="smoke_unanimous",
        strategy="custom",
        streams=(
            StreamConfig("x", "sbf", is_input=True),
            StreamConfig("y", "sbf", is_input=True),
            StreamConfig("z", "sbf", is_input=True),
            StreamConfig("agree", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="unanimous", inputs=("x", "y", "z"), output="agree"),
        ),
        num_steps=4,
        include_mirrors=False,
    )

    inputs = {
        "x": ["1", "1", "0", "1"],
        "y": ["1", "0", "1", "1"],
        "z": ["1", "1", "1", "0"],
    }
    expected_outputs = {
        "agree": _simulate_unanimous_and([inputs["x"], inputs["y"], inputs["z"]]),
    }

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)

    _run_and_assert(schema, inputs, expected_outputs, tau_bin)

