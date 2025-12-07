# Intelligent Daemon Interface (IDI)

> **Development toolkit for creating, training, and deploying intelligent Tau Language agents**

The Intelligent Daemon Interface (IDI) provides a complete ecosystem for building intelligent agents using Tau Language specifications, Q-learning training, and zero-knowledge proof integration.

## 🚀 Quick Start

### Install Dependencies

```bash
# Python dependencies
cd idi/training/python
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Rust dependencies
cd idi/devkit/rust
cargo build
```

### Create Your First Agent

**Using the GUI (Python):**
```bash
cd idi/devkit/tau_factory
python wizard_gui.py
```

**Using the GUI (Rust):**
```bash
cd idi/devkit/rust
cargo run --bin wizard-gui
```

**Using the CLI:**
```bash
cd idi/devkit
python create_agent.py --name my_agent --strategy momentum --out ../practice/
```

### Train a Q-Table

```bash
cd idi/practice/my_agent
python train_agent.py
```

### Run Agent with Tau

```bash
cd idi/practice/my_agent
./run_agent.sh
```

## 📁 Project Structure

```
idi/                          # Main IDI project
├── devkit/                   # Agent development toolkit
│   ├── tau_factory/         # Parameterized agent generator
│   │   ├── wizard_gui.py    # Python GUI (tkinter)
│   │   ├── generator.py     # Tau spec generator
│   │   └── tests/           # Comprehensive test suite
│   ├── rust/                # Rust devkit (CLI + GUI)
│   └── templates/           # Agent templates
├── training/                 # Q-learning training system
│   ├── python/              # Python training (idi_iann)
│   │   ├── fractal_abstraction.py
│   │   ├── multi_layer_trainer.py
│   │   └── config.py
│   └── rust/                # Rust training (future)
├── zk/                       # Zero-knowledge proof integration
│   └── fractal_prover/      # Risc0 proof generation
├── examples/                 # Example agents
├── practice/                # Practice/development agents
├── specs/                   # Agent specifications
└── docs/                    # IDI documentation

tau_daemon_alpha/             # Rust daemon for Tau execution
specification/                # Tau agent specs (V35-V54, libraries)
tau_q_agents/                 # Legacy Q-learning implementations
verification/                 # Verification tools
scripts/                      # Build and test scripts
```

## 🎯 Key Features

### Agent Factory
- **Wizard GUI** - Child-friendly step-by-step agent creation
- **Schema-Driven** - Define agents without writing Tau code
- **Pattern Library** - FSM, counter, accumulator, voting patterns
- **End-to-End Testing** - Automated validation with Tau binary

### Q-Learning Training
- **Multi-Layer Training** - Momentum, mean-reversion, regime-aware layers
- **Fractal Abstraction** - Hierarchical state representation
- **Emotional Expression** - Trainable communication layer
- **Benchmarking** - Performance metrics and evaluation

### Zero-Knowledge Integration
- **Risc0 Proofs** - Verifiable Q-table inference
- **Privacy-Preserving** - Private lookup tables
- **On-Chain Attestations** - Trustless agent verification

## 📚 Documentation

- [IDI Architecture](docs/IDI_IAN_ARCHITECTURE.md) - System architecture
- [Tau Agent Factory](idi/devkit/tau_factory/README.md) - Agent generation guide
- [Complexity Analysis](idi/devkit/tau_factory/COMPLEXITY_ANALYSIS.md) - Current capabilities
- [Ensemble & DAO Support](idi/devkit/tau_factory/ENSEMBLE_DAO_ANALYSIS.md) - Advanced patterns

## 🧪 Testing

```bash
# Run all tests
pytest idi/devkit/tau_factory/tests/ -v
pytest idi/training/python/tests/ -v

# Run specific test suites
pytest idi/devkit/tau_factory/tests/test_real_tau_execution.py -v
```

## 🔧 Development

### Code Quality
- **Python**: `ruff check`, `pytest`
- **Rust**: `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`

### Building Artifacts
```bash
# Build Q-table artifacts
python -m idi.devkit.builder --config configs/sample.json --out artifacts/my_agent

# Build with installation
python -m idi.devkit.builder \
    --config configs/sample.json \
    --out artifacts/my_agent \
    --install-inputs specs/V38_Minimal_Core/inputs
```

## 📦 External Dependencies

- **Tau Language** - Build locally from `tau-lang-latest/` (see LICENSE notes)
- **Python 3.12+** - For training and devkit
- **Rust 1.70+** - For daemon and Rust devkit

## 📄 License

See [LICENSE](LICENSE) for details.

**Note:** Tau Language from IDNI - build locally for internal testing only; do not distribute built artifacts. See `tau-lang-latest/LICENSE.txt`.

## 🤝 Contributing

1. Follow code quality standards (ruff, clippy)
2. Write tests for new features
3. Update documentation
4. Run full test suite before committing

## 📖 Related Projects

- **Tau Daemon Alpha** - Rust daemon for executing Tau specs (`tau_daemon_alpha/`)
- **Agent Specifications** - Tau Language agent specs (`specification/`)
- **Legacy Q-Agents** - Previous Q-learning implementations (`tau_q_agents/`)

## 🗄️ Archived Content

Unrelated content has been moved to `archive/` directory:
- Alignment Theorem project
- Lean proof system files

See [archive/README.md](archive/README.md) for details.
