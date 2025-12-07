# Tau Agent Factory

A child-friendly, parameterized Tau agent generation system with wizard-style GUI (Python/Rust), schema-driven spec generation, and end-to-end testing pipeline.

## Features

- **Parameterized Generation**: Create Tau specs without writing Tau code
- **Wizard Interface**: Step-by-step GUI for both Python (tkinter) and Rust (egui)
- **Schema-Driven**: Define agents using simple configuration schemas
- **End-to-End Testing**: Automated test harness with Tau binary execution
- **Validation**: Output validation against expected patterns
- **TDD Approach**: Test-driven development with comprehensive test coverage

## Architecture

The system follows a **Passive View / MVP pattern**:

1. **Core** (pure functions): Schema validation, Tau spec generation, test vector creation
2. **Runner** (I/O): Tau binary execution, output capture, validation
3. **GUI** (thin shell): Wizard steps, form rendering, no business logic

## Quick Start

### Python GUI

```bash
cd idi/devkit/tau_factory
python wizard_gui.py
```

### Rust GUI

```bash
cd idi/devkit/rust
cargo run --bin wizard-gui
```

### Programmatic Usage

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

# Define schema
schema = AgentSchema(
    name="my_agent",
    strategy="momentum",
    streams=(
        StreamConfig(name="q_buy", stream_type="sbf"),
        StreamConfig(name="q_sell", stream_type="sbf"),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(pattern="fsm", inputs=("q_buy", "q_sell"), output="position"),
    ),
)

# Generate spec
spec = generate_tau_spec(schema)
print(spec)
```

## Testing

Run all tests:

```bash
pytest idi/devkit/tau_factory/tests/ -v
```

Run specific test suites:

```bash
# Schema tests
pytest idi/devkit/tau_factory/tests/test_schema.py -v

# Generator tests
pytest idi/devkit/tau_factory/tests/test_generator.py -v

# Integration tests
pytest idi/devkit/tau_factory/tests/test_e2e.py -v
```

## File Structure

```
tau_factory/
  __init__.py
  schema.py              # AgentSchema, StreamConfig, LogicBlock
  generator.py           # generate_tau_spec()
  runner.py              # run_tau_spec(), TauResult
  validator.py           # validate_agent_outputs()
  wizard_controller.py   # WizardController (logic only)
  wizard_gui.py          # WizardGUI (tkinter thin shell)
  test_harness.py        # run_end_to_end_test()
  tests/
    test_schema.py
    test_generator.py
    test_runner.py
    test_validator.py
    test_e2e.py
    fixtures/
      test_vectors.json
  templates/
    patterns.json        # Logic block patterns
```

## Wizard Steps

1. **Strategy**: Pick trading approach (momentum, mean-reversion, regime-aware)
2. **Inputs**: Select market signals to watch
3. **Layers**: Configure Q-learning layers
4. **Safety**: Add risk management options
5. **Review**: Preview and save generated spec

## Supported Patterns

- **FSM**: Finite State Machine for position tracking
- **Counter**: Toggle counter on events
- **Accumulator**: Accumulate values over time
- **Vote**: Weighted voting from multiple signals
- **Passthrough**: Pass input directly to output

## Design Principles

- **Child-Friendly**: Large buttons, clear icons, simple workflows
- **Low Complexity**: Thin GUI shells, pure logic functions
- **TDD**: Tests written first, implementation follows
- **Dual Language**: Python for rapid prototyping, Rust for production

## Future Enhancements

1. Visual preview with syntax highlighting
2. Strategy marketplace (community-shared schemas)
3. Live testing (run Tau in background while configuring)
4. Undo/redo navigation
5. Export to Docker
6. Multi-agent coordination
7. Performance benchmarking

