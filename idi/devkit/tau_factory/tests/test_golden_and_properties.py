"""Golden spec regression tests and light property checks (no Tau required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from idi.devkit.tau_factory.generator import generate_tau_spec
from idi.devkit.tau_factory.schema import AgentSchema, LogicBlock, StreamConfig


FIXTURES = Path(__file__).parent / "fixtures" / "golden_smoke"


def _normalize(text: str) -> list[str]:
    """Normalize spec text for stable comparison."""
    return [line.rstrip() for line in text.strip().splitlines()]


def _load(path: str) -> str:
    return (FIXTURES / path).read_text()


def _schema_echo(num_steps: int = 3) -> AgentSchema:
    return AgentSchema(
        name="smoke_echo",
        strategy="custom",
        streams=(
            StreamConfig("input_signal", "sbf", is_input=True),
            StreamConfig("echo", "sbf", is_input=False),
        ),
        logic_blocks=(
            LogicBlock(pattern="passthrough", inputs=("input_signal",), output="echo"),
        ),
        num_steps=num_steps,
        include_mirrors=False,
    )


def _schema_majority() -> AgentSchema:
    return AgentSchema(
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


def _schema_counter() -> AgentSchema:
    return AgentSchema(
        name="smoke_counter",
        strategy="custom",
        streams=(
            StreamConfig("event", "sbf", is_input=True),
            StreamConfig("count", "sbf", is_input=False),
        ),
        logic_blocks=(LogicBlock(pattern="counter", inputs=("event",), output="count"),),
        num_steps=5,
        include_mirrors=False,
    )


def _schema_accumulator() -> AgentSchema:
    return AgentSchema(
        name="smoke_accumulator",
        strategy="custom",
        streams=(
            StreamConfig("tick", "bv", width=8, is_input=True),
            StreamConfig("sum", "bv", width=8, is_input=False),
        ),
        logic_blocks=(LogicBlock(pattern="accumulator", inputs=("tick",), output="sum"),),
        num_steps=4,
        include_mirrors=False,
    )


def _schema_fsm() -> AgentSchema:
    return AgentSchema(
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


def test_golden_echo():
    spec = generate_tau_spec(_schema_echo())
    assert _normalize(spec) == _normalize(_load("smoke_echo.tau"))


def test_golden_majority():
    spec = generate_tau_spec(_schema_majority())
    assert _normalize(spec) == _normalize(_load("smoke_majority.tau"))


def test_golden_counter():
    spec = generate_tau_spec(_schema_counter())
    assert _normalize(spec) == _normalize(_load("smoke_counter.tau"))


def test_golden_accumulator():
    spec = generate_tau_spec(_schema_accumulator())
    assert _normalize(spec) == _normalize(_load("smoke_accumulator.tau"))


def test_golden_fsm():
    spec = generate_tau_spec(_schema_fsm())
    assert _normalize(spec) == _normalize(_load("smoke_fsm.tau"))


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 6, 7, 8])
def test_property_num_steps_commands(num_steps: int):
    """Spec should emit exactly num_steps 'n' commands plus a trailing 'q'."""
    schema = _schema_echo(num_steps=num_steps)
    spec = generate_tau_spec(schema)
    lines = [ln.strip() for ln in spec.splitlines() if ln.strip()]
    n_count = sum(1 for ln in lines if ln == "n")
    assert n_count == num_steps
    assert lines[-1] == "q"


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 6, 7, 8])
def test_property_output_decl_count(num_steps: int):
    """Every output stream should have an out file declaration."""
    schema = _schema_echo(num_steps=num_steps)
    spec = generate_tau_spec(schema)
    outputs = [s for s in schema.streams if not s.is_input]
    for idx, stream in enumerate(outputs):
        hex_idx = hex(idx)[2:].upper()
        marker = f"o{hex_idx}:{'sbf' if stream.stream_type=='sbf' else f'bv[{stream.width}]'} = out file(\"outputs/{stream.name}.out\")."
        assert marker in spec

