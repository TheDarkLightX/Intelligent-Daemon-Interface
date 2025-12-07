# IDI DevKit Usage Guide

## Quick Start

### Python CLI

```bash
# List available templates
python3 create_agent.py --list-templates

# Create a new agent
python3 create_agent.py --name my_agent --strategy momentum --out ../practice/

# Create with custom template
python3 create_agent.py --name custom_agent --strategy momentum --out ../practice/ --custom-template my_template.json
```

### Python GUI

```bash
python3 create_agent_gui.py
```

Interactive GUI for creating agents with:
- Dropdown strategy selection
- Visual feedback
- Status logging

### Rust CLI

```bash
# Build first
cd rust
cargo build --release

# List templates
cargo run --bin create-agent -- list

# Create agent
cargo run --bin create-agent -- new --name my_agent --strategy momentum --out ../practice/
```

### Rust GUI

```bash
cd rust
cargo run --bin create-agent-gui
```

Native GUI with egui for:
- Fast performance
- Modern UI
- Cross-platform support

## Generated Structure

Each agent gets:

```
agent_name/
├── agent_name.tau          # Tau specification
├── train_agent.py          # Q-table training script
├── run_agent.sh            # Execution script
├── README.md               # Documentation
├── inputs/                 # Trace inputs (after training)
├── outputs/                # Tau execution outputs
└── tests/                  # Test cases
```

## Templates

### Momentum
- **Description**: Follow price momentum
- **Best for**: Trending markets
- **Config**: High trend granularity, momentum-focused rewards

### Mean Reversion
- **Description**: Buy dips, sell spikes
- **Best for**: Range-bound markets
- **Config**: Balanced quantizer, mean-reversion logic

### Regime Aware
- **Description**: Adapt to market regimes
- **Best for**: Volatile markets
- **Config**: Extended episodes, regime tracking

## Language Comparison

| Task | Python | Rust |
|------|--------|------|
| **Quick prototyping** | ✅ Excellent | ⚠️ Slower compile |
| **Production tools** | ⚠️ Good | ✅ Excellent |
| **GUI development** | ✅ Simple (tkinter) | ✅ Modern (egui) |
| **Performance** | ⚠️ Fast enough | ✅ Fastest |
| **Type safety** | ⚠️ Runtime checks | ✅ Compile-time |

## Best Practices

1. **Start with Python** for rapid iteration
2. **Use Rust** for production tooling
3. **Test both** for cross-validation
4. **Use templates** as starting points
5. **Customize** training configs per strategy

## Troubleshooting

### Python Import Errors
```bash
# Ensure training directory is in PYTHONPATH
export PYTHONPATH=/path/to/idi/training/python:$PYTHONPATH
```

### Rust Compilation Errors
```bash
# Update dependencies
cd rust
cargo update
```

### GUI Not Showing
- **Python**: Ensure tkinter is installed (`sudo apt install python3-tk`)
- **Rust**: Ensure OpenGL is available

