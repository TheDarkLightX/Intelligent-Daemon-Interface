# IDI Development Log

## 2025-12-06

- Regenerated all macro/micro/emote/layered lookup artifacts via `python -m idi.devkit.build_layers`, ensuring the new layer-weight streams land in `idi/specs/Q_Layered_Strategy/inputs/`.
- Produced a real Risc0 proof bundle for the layered demo (`idi/artifacts/layered_demo/proof_risc0/`) using the upgraded 3.0.4 toolchain and `idi_risc0_host`.
- Pulled the upstream Tau parser grammar (`parser/tau.tgf`) to confirm the canonical `name : type = in/out ...` syntax—handy when the daemon binary disagrees.
- Rebased the layered Tau spec onto the canonical `i*/o*` naming so the parser matches the grammar; next step is resolving the “Failed to find output stream for stream ‘i0’” runtime expectation by tracing the interpreter’s stream bookkeeping.
- Added tile-coded state abstraction plus a dedicated communication Q-table + trainable emotional expression layer to `idi_iann`, so `q_emote_*` streams are now learned policies instead of heuristics.
- Smoke-tested the upgraded trainer via `python run_idi_trainer.py --out idi/artifacts/dev_test`, verifying the new streams (`price_*`, `weight_*`, `q_emote_*`, etc.) are emitted alongside manifests.
- Ran the conversational config (`idi/devkit/configs/conversational_macro.json`) to emit tile-coded + communication-shaped traces at `idi/artifacts/conversational_macro/` (ready for zk proving and Tau replay).

