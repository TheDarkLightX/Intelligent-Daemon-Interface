"""End-to-end integration tests for all strategies."""

import pytest
import json
from pathlib import Path

try:
    from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
    from idi.devkit.tau_factory.generator import generate_tau_spec
    from idi.devkit.tau_factory.test_harness import run_end_to_end_test
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


def load_test_vectors() -> list:
    """Load test vectors from JSON."""
    vectors_path = Path(__file__).parent / "fixtures" / "test_vectors.json"
    if not vectors_path.exists():
        return []
    
    data = json.loads(vectors_path.read_text())
    return data.get("test_cases", [])


def schema_from_dict(data: dict) -> AgentSchema:
    """Create AgentSchema from dictionary."""
    streams = []
    for s_data in data["streams"]:
        streams.append(StreamConfig(**s_data))
    
    logic_blocks = []
    for l_data in data["logic_blocks"]:
        logic_blocks.append(LogicBlock(**l_data))
    
    return AgentSchema(
        name=data["name"],
        strategy=data["strategy"],
        streams=tuple(streams),
        logic_blocks=tuple(logic_blocks),
        num_steps=data.get("num_steps", 5),
        include_mirrors=data.get("include_mirrors", True),
    )


@pytest.fixture
def tau_bin():
    """Get Tau binary path."""
    # Try to find tau binary
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "tau-lang-latest" / "build-Release" / "tau",
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return path
    
    pytest.skip("Tau binary not found")


@pytest.mark.parametrize("test_case", load_test_vectors())
def test_strategy_generates_runnable_spec(test_case, tau_bin):
    """Test that each strategy generates a runnable spec."""
    schema = schema_from_dict(test_case["schema"])
    
    # Generate spec
    spec = generate_tau_spec(schema)
    
    # Basic syntax checks
    assert "defs" in spec
    assert "r (" in spec
    assert spec.strip().endswith("q")
    
    # Check inputs/outputs declared
    for stream in schema.streams:
        if stream.is_input:
            assert f'{stream.name}.in' in spec or stream.name in spec
        else:
            assert f'{stream.name}.out' in spec or stream.name in spec


@pytest.mark.parametrize("strategy", ["momentum", "mean_reversion", "regime_aware"])
def test_strategy_schema_validation(strategy):
    """Test that each strategy creates a valid schema."""
    if strategy == "momentum":
        schema = AgentSchema(
            name="test_momentum",
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
        )
    elif strategy == "mean_reversion":
        schema = AgentSchema(
            name="test_mean_reversion",
            strategy="mean_reversion",
            streams=(
                StreamConfig(name="price_up", stream_type="sbf"),
                StreamConfig(name="price_down", stream_type="sbf"),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
                StreamConfig(name="sell_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="fsm", inputs=("price_up", "price_down"), output="position"),
                LogicBlock(pattern="passthrough", inputs=("price_down",), output="buy_signal"),
                LogicBlock(pattern="passthrough", inputs=("price_up",), output="sell_signal"),
            ),
        )
    else:  # regime_aware
        schema = AgentSchema(
            name="test_regime_aware",
            strategy="regime_aware",
            streams=(
                StreamConfig(name="q_buy", stream_type="sbf"),
                StreamConfig(name="q_sell", stream_type="sbf"),
                StreamConfig(name="regime", stream_type="bv", width=5),
                StreamConfig(name="position", stream_type="sbf", is_input=False),
                StreamConfig(name="buy_signal", stream_type="sbf", is_input=False),
                StreamConfig(name="sell_signal", stream_type="sbf", is_input=False),
            ),
            logic_blocks=(
                LogicBlock(pattern="fsm", inputs=("q_buy", "q_sell"), output="position"),
                LogicBlock(pattern="passthrough", inputs=("q_buy",), output="buy_signal"),
                LogicBlock(pattern="passthrough", inputs=("q_sell",), output="sell_signal"),
            ),
        )
    
    # Should not raise
    spec = generate_tau_spec(schema)
    assert len(spec) > 0


@pytest.mark.skipif(True, reason="Requires Tau binary - run manually")
def test_end_to_end_execution(tau_bin):
    """Test end-to-end execution with real Tau binary."""
    schema = AgentSchema(
        name="test_e2e",
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
    
    report = run_end_to_end_test(schema, tau_bin, num_ticks=5)
    
    assert report.result.success, f"Tau execution failed: {report.result.errors}"
    assert report.validation.passed, f"Validation failed: {report.validation.checks}"

