# Intelligent Daemon Interface (IDI)

> **Development toolkit for creating, training, and deploying intelligent Tau Language agents**

The Intelligent Daemon Interface (IDI) provides a complete ecosystem for building intelligent agents using Tau Language specifications, Q-learning training, and zero-knowledge proof integration.

## 🚀 Quick Start

### Tau Testnet Integration

The IDI ZK proof infrastructure integrates with [Tau Testnet](https://github.com/IDNI/tau-testnet) through the `idi/taunet_bridge/` module:

```python
from idi.taunet_bridge import TauNetZkAdapter, ZkConfig, ZkValidationStep
from idi.taunet_bridge.validation import ValidationContext

# Configure ZK verification
config = ZkConfig(enabled=True, proof_system="stub")
adapter = TauNetZkAdapter(config)

# Create validation step for transaction pipeline
zk_step = ZkValidationStep(adapter, required=False)

# Validate transaction with ZK proof
ctx = ValidationContext(tx_hash="...", payload={"zk_proof": proof_bundle})
zk_step.run(ctx)  # Raises InvalidZkProofError if verification fails
```

See [`idi/taunet_bridge/README.md`](idi/taunet_bridge/README.md) for detailed integration guide.

## 🚀 Quick Start (Original)

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

### Q-Agent Persistence Format (Safe)

Q-agent training scripts that persist learned state use a **safe, data-only** format:

```text
<base>/
├── metadata.json
└── arrays.npz
```

Notes:
- **Suffix stripping (UX):** if you pass a path like `model.tar.gz`, artifacts are written to `model/` (no misleading extensions).
- **Legacy compatibility:** loaders still accept the older `<base>.meta.json` + `<base>.arrays.npz` layout.

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
│   │   ├── wizard_controller.py  # Wizard logic
│   │   ├── generator.py     # Tau spec generator
│   │   ├── schema.py        # Agent schema definitions
│   │   ├── runner.py        # Tau execution runner
│   │   ├── validator.py     # Output validation
│   │   ├── templates/       # Pattern templates
│   │   └── tests/          # Comprehensive test suite
│   └── rust/               # Rust devkit (CLI + GUI, egui)
├── training/                # Q-learning training system (IAN)
│   ├── python/              # Python training (idi_iann)
│   │   ├── fractal_abstraction.py  # Hierarchical state encoding
│   │   ├── multi_layer_trainer.py  # Multi-layer Q-learning
│   │   ├── config.py        # Training configuration
│   │   └── tests/          # Training tests
│   └── rust/               # Rust training (future)
├── zk/                      # Zero-knowledge proof integration
│   ├── risc0/              # Risc0 zkVM workspace (host + guest)
│   ├── witness_generator.py  # Q-table witness generation
│   ├── merkle_tree.py      # Merkle tree for large Q-tables
│   ├── proof_manager.py    # Proof generation and verification
│   └── workflow.py         # End-to-end proof workflow
├── examples/                # Example agents
│   ├── ensemble_trading_agent/  # Multi-agent voting example
│   └── dao_voting_agent/    # Governance example
├── practice/               # Practice/development agents (git-ignored)
├── specs/                  # Agent specifications
└── docs/                   # IDI documentation

tau_daemon_alpha/            # Rust daemon for Tau execution
specification/               # Tau agent specs (V35-V54, libraries)
tau_q_agents/               # Legacy Q-learning implementations
verification/                # Verification tools
scripts/                     # Build and test scripts
archive/                     # Archived content (Alignment Theorem, Lean proofs)
```

## 🎯 Key Features

### Agent Factory (Tau Agent Factory)
- **Wizard GUI** - Child-friendly step-by-step agent creation (Python tkinter + Rust egui)
- **Schema-Driven** - Define agents without writing Tau code
- **26 Pattern Library** - FSM, counter, accumulator, voting, majority, quorum, supervisor-worker, weighted vote, time-lock, hex stake, multi-bit counter, streak counter, mode switch, proposal FSM, risk FSM, entry-exit FSM, orthogonal regions, state aggregation, TCP connection FSM, UTXO state machine, history state, decomposed FSM, script execution, and more
- **End-to-End Testing** - Automated validation with Tau binary execution
- **100% Pattern Coverage** - 26/26 patterns implemented ✅

### Q-Learning Training (IAN - Intelligence Augmentation Network)
- **Multi-Layer Training** - Momentum, mean-reversion, regime-aware layers
- **Fractal Abstraction** - Hierarchical state representation with tile coding
- **Emotional Expression** - Trainable communication layer (emojis, text, ASCII art)
- **Benchmarking** - Performance metrics and evaluation
- **Realistic Simulators** - Crypto market simulation with GARCH volatility, regime switching

### IAN Networking (P2P) — FrontierSync + Authenticated IBLT
- **Bandwidth-efficient sync** - FrontierSync uses an IBLT-based exchange to synchronize logs with O(Δ) bandwidth (falls back safely when decode fails).
- **Tamper resistance** - IBLT payloads can be HMAC-authenticated to prevent untrusted peers from injecting or mutating reconciliation data.
- **No global shared secret** - A per-peer ephemeral session key (X25519 + HKDF) is derived during handshake and exposed to higher layers via the transport.
- **Replay/DoS controls** - Per-message replay detection, bounded caches, message size limits, and rate limiting.

### Zero-Knowledge Integration ✅ Production Ready
- **Risc0 zkVM Proofs** - Fully functional end-to-end ZK proof system
  - Guest programs: Manifest verification and Q-table action selection
  - Host program: Generates STARK proofs with method_id verification
  - Receipt verification: Cryptographically secure proof validation
  - **Auto-Detection** - Automatically uses Risc0 when available (default behavior)
- **Witness Generation** - Convert Q-tables to zk-friendly format with Merkle commitments
- **Merkle Trees** - Efficient commitments for large Q-tables (>100 entries)
- **Privacy-Preserving** - Q-values never exposed; only commitments revealed
- **End-to-End Testing** - Complete test suite verifies privacy guarantees
- **TauBridge Integration** - Ready for Tau Net testnet deployment


## 📚 Documentation

For a structured, wiki-like overview of all documentation, see the index in [`docs/README.md`](docs/README.md).

### Core Documentation
- [IDI Architecture](docs/IDI_IAN_ARCHITECTURE.md) - System architecture and design (includes IAN overview)
- [IAN L2 Node & Network](idi/ian/README.md) - Intelligent Augmentation Network (L2) overview and quick start
- [IAN L2 Detailed Docs](idi/ian/docs/README.md) - Architecture guide, API reference, and operator runbook for IAN
- [GUI Walkthrough](docs/gui_walkthrough/GUI_WALKTHROUGH.md) - Visual guide to the web-based GUI interface
- [Tau Agent Factory](idi/devkit/tau_factory/README.md) - Agent generation guide
- [Pattern Landscape](idi/devkit/tau_factory/PATTERN_LANDSCAPE.md) - Complete pattern taxonomy
- [Implementation Status](idi/devkit/tau_factory/IMPLEMENTATION_STATUS.md) - Current progress (26/26 patterns, 100% complete ✅)
- [Modular Synth & Auto-QAgent](docs/MODULAR_SYNTH_AND_AUTO_QAGENT.md) - Synth/Auto-QAgent overview
- [idi.synth API](docs/IDI_SYNTH_API.md) - Stable synth API surface
- [Modular Synth Quality & Testing](docs/MODULAR_SYNTH_QUALITY_AND_TESTING.md) - Invariants and test strategy
- [Parameterization & Synth Visual Guide](docs/PARAMETERIZATION_AND_SYNTH_VISUALS.md) - Visual-first explanation of parameters and search

### Pattern Documentation
- [Ensemble & DAO Patterns](idi/devkit/tau_factory/ENSEMBLE_PATTERNS.md) - Voting and consensus patterns
- [Hierarchical FSMs](idi/devkit/tau_factory/HIERARCHICAL_FSM_DESIGN.md) - Supervisor-worker and composition
- [Bitvector Patterns](idi/devkit/tau_factory/BITVECTOR_PATTERNS.md) - Weighted voting and time-locks
- [Hex Pattern](idi/devkit/tau_factory/HEX_SUMMARY.md) - Time-lock staking implementation
- [High Priority Patterns](idi/devkit/tau_factory/HIGH_PRIORITY_PATTERNS_COMPLETE.md) - Multi-bit counter, streak counter, mode switch, proposal FSM, risk FSM

### Pattern Limitations
- [Limitations & Why](idi/devkit/tau_factory/LIMITATIONS_AND_WHY.md) - What can't be done and why

## 🧪 Testing

```bash
# Run all Tau Factory tests
pytest idi/devkit/tau_factory/tests/ -v

# Run specific test suites
pytest idi/devkit/tau_factory/tests/test_high_priority_patterns.py -v  # New patterns
pytest idi/devkit/tau_factory/tests/test_real_tau_execution.py -v      # End-to-end
pytest idi/devkit/tau_factory/tests/test_ensemble_patterns.py -v       # Voting patterns
pytest idi/devkit/tau_factory/tests/test_supervisor_worker.py -v        # Hierarchical FSMs

# Run training tests
pytest idi/training/python/tests/ -v

# Run focused IAN security tests (FrontierSync authenticated IBLT)
pytest -q idi/ian/tests/test_frontiersync.py::TestFrontierSyncIBLTAuthentication -q

# Syntax/import sanity for the handshake and network modules
python3 -m compileall -q idi/ian/network/p2p_manager.py idi/ian/network/protocol.py idi/ian/network/frontiersync.py
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

## 📊 Current Status

### Pattern Implementation: 26/26 (100%) ✅
- ✅ **Basic Patterns** (5): FSM, Counter, Accumulator, Passthrough, Vote
- ✅ **Composite Patterns** (4): Majority, Unanimous, Custom, Quorum
- ✅ **Hierarchical Patterns** (4): Supervisor-Worker, Orthogonal Regions, State Aggregation, Decomposed FSM
- ✅ **Bitvector Patterns** (2): Weighted Vote, Time Lock
- ✅ **Domain Patterns** (5): Hex Stake, Entry-Exit FSM, Proposal FSM, Risk FSM, TCP Connection FSM, UTXO State Machine
- ✅ **High Priority Patterns** (5): Multi-Bit Counter, Streak Counter, Mode Switch, Proposal FSM, Risk FSM
- ✅ **Low Priority Patterns** (3): History State, Decomposed FSM, Script Execution

**All patterns complete!** See [IMPLEMENTATION_STATUS.md](idi/devkit/tau_factory/IMPLEMENTATION_STATUS.md) for details.

## 📖 Related Projects

- **Tau Daemon Alpha** - Rust daemon for executing Tau specs (`tau_daemon_alpha/`)
- **Agent Specifications** - Tau Language agent specs V35-V54 (`specification/`)
- **Legacy Q-Agents** - Previous Q-learning implementations (`tau_q_agents/`)

## 🗄️ Archived Content

Unrelated content has been moved to `archive/` directory:
- **Alignment Theorem** - Economic alignment research project
- **Lean Proofs** - Formal verification proofs (Lean 4)

See [archive/README.md](archive/README.md) and [ARCHIVE.md](ARCHIVE.md) for details.

## 🚧 Development Roadmap

### Completed ✅
- [x] Basic pattern library (FSM, Counter, Accumulator, etc.)
- [x] Ensemble patterns (Majority, Unanimous, Quorum)
- [x] Hierarchical FSMs (Supervisor-Worker)
- [x] Bitvector patterns (Weighted Vote, Time Lock)
- [x] High-priority patterns (Multi-Bit Counter, Streak Counter, Mode Switch, Proposal FSM, Risk FSM)
- [x] Medium-priority patterns (Entry-Exit FSM, Orthogonal Regions, State Aggregation, TCP Connection FSM, UTXO State Machine)
- [x] Low-priority patterns (Decomposed FSM, History State, Script Execution)
- [x] Wizard GUI (Python + Rust)
- [x] End-to-end testing pipeline
- [x] Example agents using new patterns
- [x] Tau Testnet ZK integration infrastructure

### Completed ✅ (Latest)
- [x] Risc0 ZK proof integration - **Fully functional end-to-end** ✅
  - Guest programs compile and execute correctly
  - Host program generates proofs with proper verification
  - End-to-end tests pass with Risc0 proofs
  - Privacy guarantees verified (Q-values never exposed)
- [x] TauBridge integration - Ready for Tau Net testnet
- [x] Private training workflow - Complete with ZK proofs

### In Progress 🚧
- [ ] Rust training implementation
- [ ] Performance optimization
- [ ] Enhanced Tau Testnet integration (see recommendations below)
