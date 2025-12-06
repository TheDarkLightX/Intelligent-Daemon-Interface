# Intelligent Daemon Interface (IDI)

This folder is the canonical root for the Intelligent Daemon Interface & Intelligence Augmentation Network workstream.

- **Primary repository:** https://github.com/TheDarkLightX/Intelligent-Daemon-Interface  
  (push the contents of this directory to that remote when publishing updates.)

## Layout
| Path | Description |
|------|-------------|
| `docs/` | Architecture + zk-privacy reference material. |
| `specs/` | Tau specifications and the `idi_core` stream contract. |
| `training/` | Python & Rust lookup-table toolchains (linted & tested). |
| `devkit/` | Builder CLI (`builder.py`), batch builder (`build_layers.py`), manifests, sample configs, and unit tests. |
| `zk/` | Proof manager stubs + integration helpers for real zkVMs. |
| `demos/` | Replay harness with AoT verification and curated inputs. |

## Development checklist
1. Edit/train inside `idi/training/...`, then run `ruff check`, `pytest`, `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test`.
2. Use `python -m idi.devkit.builder --config ... --out ... --install-inputs specs/V38_Minimal_Core/inputs` (or `python -m idi.devkit.build_layers`) to produce streams + manifests.
3. Optionally call `python -m idi.zk.proof_manager` (or integrate Risc0) to emit `proof.bin` + `receipt.json`; store them alongside the manifest.
4. Run `demos/idi_demo/run_demo.sh` to validate end-to-end behavior.
5. Commit updated manifests/proofs/docs; push to the GitHub repo above.

