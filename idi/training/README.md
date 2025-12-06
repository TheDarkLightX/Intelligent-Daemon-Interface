# IDI/IANN Lookup Libraries

This folder houses the parallel Python/Rust training stacks used to precompute Intelligent Daemon Interface lookup tables.

## Python (`idi/training/python/`)
- `idi_iann/`: typed package containing quantizer config, synthetic environment, emotion engine, lookup policy, and Q-trainer.
- `tests/`: pytest suite exercising the trainer end-to-end.
- `run_idi_trainer.py`: CLI entrypoint (`python run_idi_trainer.py --out build/idi_traces`) that emits Tau-ready `*.in` files plus a manifest JSON. The configs under `idi/devkit/configs/` (`sample_config.json`, `regime_macro.json`, `regime_micro.json`, etc.) are ready-to-use templates.

## Rust (`idi/training/rust/idi_iann/`)
- Cargo crate mirroring the Python abstractions for deterministic cross-checks.
- `cargo test` runs the smoke test in `tests/smoke.rs`.

Both stacks serialize the same stream schema:

| Stream | Description |
|--------|-------------|
| `q_buy.in` | Requested buy events (sbf). |
| `q_sell.in` | Requested sell events (sbf). |
| `risk_budget_ok.in` | Budget gate (sbf). |
| `q_emote_positive.in` / `q_emote_alert.in` | Emotive cues for the art pipeline. |
| `q_emote_persistence.in` | Helper bit for linger logic. |
| `q_regime.in` | Regime identifier (bv[5]). |

Output directories can be copied directly into a Tau spec folder (e.g., `idi/specs/V38_Minimal_Core/inputs/`) before running the Tau binary— or use `idi/devkit/build_layers.py` to train and install multiple layers in one go.

