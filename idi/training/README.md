# IDI/IANN Lookup Libraries

This folder houses the parallel Python/Rust training stacks used to precompute Intelligent Daemon Interface lookup tables.

## Python (`idi/training/python/`)
- `idi_iann/`: typed package containing quantizer config, tile-coding abstractions, synthetic environment, emotion/communication layers, lookup policy, and Q-trainer.
- `tests/`: pytest suite exercising the trainer end-to-end plus schema parity, golden tests, differential tests, and reward/crypto checks.
- `run_idi_trainer.py`: CLI entrypoint for training runs with market parameter controls.
- `backtest.py`: CLI for backtesting policies against historical data.

### Quick Start

**Training:**
```bash
cd idi/training/python
python -m run_idi_trainer --config ../config_schema.json --out artifacts/my_agent
```

**Using factories:**
```python
from idi_iann import create_trainer, TrainingConfig

config = TrainingConfig(episodes=128, episode_length=64)
trainer = create_trainer(config, seed=42)
policy, trace = trainer.run()
```

**Testing:**
```bash
cd idi/training/python
python -m pytest                    # All tests
python -m pytest tests/test_golden.py  # Golden tests
python -m ruff check idi_iann       # Linting
```

### Highlights
- **State abstraction**: optional tile coding (`TileCoderConfig`) compresses high-cardinality state spaces into a handful of tiles so the lookup tables stay tractable; hashing-friendly encodings keep memory bounded.
- **Trainable emotional layer**: the `EmotionEngine` now cooperates with a dedicated communication Q-table, learning when to emit positive/alert/persistent cues instead of relying on heuristics.
- **Communication Q-table**: `CommunicationPolicy` mirrors the trading Q-table, producing expressive outputs (`q_emote_*`) that respond to rewards and regime shifts; reward shaping hooks bias alerts toward risky regimes.

## Rust (`idi/training/rust/idi_iann/`)
- Cargo crate mirroring the Python abstractions for deterministic cross-checks.
- `src/bin/train.rs`: CLI binary with `train`, `backtest`, and `export` subcommands.

### Quick Start

**Training:**
```bash
cd idi/training/rust/idi_iann
cargo run --bin train train --config config.json --out outputs/traces
```

**Testing:**
```bash
cd idi/training/rust/idi_iann
cargo fmt -- --check      # Format check
cargo clippy -- -D warnings  # Linting
cargo test                 # All tests
cargo test --test integration  # Integration tests
```

Both stacks serialize the same stream schema:

| Stream | Description |
|--------|-------------|
| `q_buy.in` | Requested buy events (sbf). |
| `q_sell.in` | Requested sell events (sbf). |
| `risk_budget_ok.in` | Budget gate (sbf). |
| `q_emote_positive.in` / `q_emote_alert.in` | Emotive cues for the art pipeline. |
| `q_emote_persistence.in` | Helper bit for linger logic. |
| `q_regime.in` | Regime identifier (bv[5]). |
| `price_up.in` / `price_down.in` | Derived price-direction bits for layered Tau specs. |
| `weight_momentum.in` / `weight_contra.in` / `weight_trend.in` | Layer-selection weights mirrored from the Q-table committee. |

Output directories can be copied directly into a Tau spec folder (e.g., `idi/specs/V38_Minimal_Core/inputs/` or `idi/specs/Q_Layered_Strategy/inputs/`) before running the Tau binary— or use `idi/devkit/build_layers.py` to train and install multiple layers in one go.

## Cross-language parity
- Shared default schema lives at `idi/training/config_schema.json`; both Python and Rust tests assert the defaults match.
- For fixed-seed parity runs, regenerate traces with the same config/seed in Python and Rust and compare line counts in the emitted `*.in` files before installing into Tau specs.

## Backtesting & Controls
- `run_idi_trainer.py` exposes market knobs (`--drift-bull`, `--vol-base`, `--shock-prob`, `--fee-bps`, `--seed`) and `--use-crypto-env`.
- `backtest.py` loads CSV/Parquet, quantizes to the current state space, replays a policy, and emits KPIs (mean reward, Sharpe-like ratio, max drawdown, win rate) plus action precision/recall.
- `idi_iann/gui_sim.py` adds sliders for drift/vol/shock/seed to visually inspect price paths (optional dependency on matplotlib widgets).

## Workflow: Training → Proof → Tau Spec

See `idi/zk/workflow.py` for the standardized end-to-end workflow:

1. **Train**: Generate Q-tables and traces
2. **Manifest**: Create artifact manifest with stream hashes
3. **Proof**: Generate zk proof bundle (Risc0 or stub)
4. **Spec**: Generate Tau-language spec from config
5. **Verify**: Verify proof and spec consistency

Example:
```bash
python -c "from idi.zk.workflow import run_training_to_proof_workflow; \\
    result = run_training_to_proof_workflow( \\
        config_path='config.json', \\
        artifact_dir='artifacts/my_agent' \\
    )"
```

## Architecture & Domain Model

- See `idi/docs/ARCHITECTURE.md` for component inventory and data flows
- See `idi/docs/DOMAIN_MODEL.md` for shared domain types (Action, Regime, StateKey, Transition, etc.)
- See `idi/specs/schemas/` for JSON schemas (config, trace, manifest)

