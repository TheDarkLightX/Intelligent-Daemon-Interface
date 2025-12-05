# IDI-Augmented Intelligent Agent Demo

This demo replays the Intelligent Daemon Interface (IDI) signals generated offline by the lookup-table trainers and feeds them into the `agent4_testnet_v38` Tau specification. It showcases:

- Trading actions determined by private Q-tables (`q_buy`, `q_sell`).
- Risk budget gating (`risk_budget_ok`).
- Emotive outputs (`q_emote_positive`, `q_emote_alert`, `q_emote_persistence`) that drive Alignment Theorem art/communication layers.

## Run
```
cd idi/demos/idi_demo
./run_demo.sh
```

The script copies the Tau spec into an isolated temp directory, injects the prepared inputs under `inputs/`, runs the Tau binary, and exports the resulting `outputs/*.out` files back into this folder for inspection.

## Verification
- `verification.md` contains the AoT/truth-table checklist showing how `q_buy`/`q_sell` interact with timers and risk budget.
- Mirror streams (`o19..o1F`) echo every IDI input so the Tau daemon, AoT checks, and humans can diff expected vs. actual signals.

## Updating inputs
1. Generate fresh traces via `idi/devkit` (recommended: `python -m idi.devkit.builder --config idi/devkit/configs/sample_config.json --out build/idi_traces --install-inputs idi/specs/V38_Minimal_Core/inputs`) or directly with `idi/training/python/run_idi_trainer.py --out build/idi_traces`.
2. Copy the resulting `.in` files into this demo’s `inputs/` directory.
3. Re-run `./run_demo.sh` to validate the new intelligence snapshot.

