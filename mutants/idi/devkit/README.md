# IDI DevKit

Development toolkit for creating and managing Tau Language intelligent agents.

## Tools

### Agent Development CLI

**Python Version** (`create_agent.py`):
- Rapid prototyping
- Rich ecosystem integration
- Easy customization

**Rust Version** (`rust/src/bin/create_agent.rs`):
- High performance
- Type safety
- Production-ready

**Usage:**
```bash
# Python
python3 create_agent.py --name my_agent --strategy momentum --out ../practice/

# Rust
cargo run --bin create-agent -- --name my_agent --strategy momentum --out ../practice/
```

### Agent Development GUI

**Python Version** (`create_agent_gui.py`):
- tkinter-based GUI
- Simple and lightweight
- Cross-platform

**Rust Version** (`rust/src/bin/create_agent_gui.rs`):
- egui-based GUI
- Native performance
- Modern UI

**Usage:**
```bash
# Python
python3 create_agent_gui.py

# Rust
cargo run --bin create-agent-gui
```

## Features

- **Template System**: Pre-built templates for common strategies
- **Auto-scaffolding**: Creates complete agent structure
- **Training Integration**: Generates training scripts
- **Test Framework**: Includes test scaffolding

## Templates

- `momentum`: Momentum following strategy
- `mean_reversion`: Mean reversion strategy
- `regime_aware`: Regime-aware adaptive strategy

## Language Comparison

| Feature | Python | Rust |
|---------|--------|------|
| **Speed** | Fast enough for prototyping | Fastest |
| **Type Safety** | Dynamic typing | Static typing |
| **Ecosystem** | Rich libraries | Growing ecosystem |
| **GUI** | tkinter (simple) | egui (modern) |
| **Best For** | Rapid development | Production tools |

## Future Enhancements

- [ ] Visual agent builder
- [ ] Real-time testing interface
- [ ] Performance profiling
- [ ] Template marketplace
