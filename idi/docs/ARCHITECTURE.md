# Intelligent Daemon Interface & IAN Blueprint

## 1. Guardrails & requirements
- Tau specs remain 32-bit-bounded FSMs with symmetric file I/O; heavy learning stays outside the spec and is fed back in as deterministic streams (see `specification/TAU_LANGUAGE_CAPABILITIES.md` for canonical patterns).
- Alignment Theorem economics force every agent layer to respect scarcity, pressure, and burn invariants; Tau Net guard adapters already enforce these constraints via daemon hooks (`tau_daemon_alpha/README.md`).
- Goal: let offline intelligence choose actions privately while Tau specs verify only the proof-carrying action indices and safety guards; both trading and “emotive” channels (emoji/text cues) must share the same deterministic interface.

### Reusable FSM motifs
- **V35/V38 kernels** (see `specification/agent4_testnet_v35.tau` and `idi/specs/V38_Minimal_Core/agent4_testnet_v38.tau`): provide the canonical 2-bit timer, nonce gating, and burn tracker used by all downstream specs.
- **Demo suite** (`idi/demos/idi_demo`): showcases 4-state FSM entry/exit logic with mirrored inputs; we reuse its smart-entry timer as the base for `idi_core`.
- **lookup_q_table_brief.md**: documents layered Q-table decomposition (regime selector, per-regime actions, shield mask) that this architecture formalizes for both trading and communicative outputs.

## 2. Proof stack decision matrix
| Stack | License | Strengths for lookup proofs | Drawbacks / fit |
|-------|---------|-----------------------------|-----------------|
| Risc0 zkVM | Apache-2.0 + MIT dual (`LICENSE-APACHE`,`LICENSE-MIT`) – [link](https://github.com/risc0/risc0/blob/main/LICENSE-APACHE) | Programmability in Rust/C, recursion-ready STARK proofs, host-side receipt verification; good for private ROM tables + general compute. | Proof sizes larger than PLONK, GPU prover still heavy for <50 ms latency. |
| ezkl (Halo2-based zkML) | Apache-2.0 – [link](https://github.com/Spectral-Finance/ezkl/blob/main/LICENSE) | Native ONNX import, plookup-heavy layers, straightforward private weights; ideal for snapshot Q-table inference. | Circuit flows best for feed-forward shapes; custom ROM logic requires tinkering with halo2 gadgets. |
| zkSync era-boojum | Apache-2.0 variant – [link](https://github.com/matter-labs/era-boojum/blob/main/LICENSE-APACHE) | Highly optimized STARK prover with RAM/ROM tables and batching; good candidate when aligning with zkSync ecosystem. | Built for zkSync rollup state transitions; tailoring to Tau-only workflows needs adapter work. |
| Polygon zkEVM prover | AGPLv3 – [link](https://github.com/0xPolygon/zkevm-prover/blob/main/LICENSE) | Production-grade PLONKish prover + lookup arguments; integrates easily with EVM proof markets. | Copyleft – cannot embed directly into closed-source daemons; requires isolating as a service. |
| Nil Foundation zkLLVM | Repo exposes no license metadata (`license: null` via [API](https://api.github.com/repos/NilFoundation/zkLLVM)), so reuse requires explicit legal clearance. | Powerful compiler from Rust/C++ into Placeholder-style arithmetizations, ideal for bespoke low-level ROM tables. | Licensing ambiguity blocks redistribution until Nil provides terms. |
| Modulus Aura zkVM | No public repo/license advertised as of Dec 2025 (press coverage only). Custom licensing with TFH takeover means we treat it as proprietary and non-portable. | Research-grade zk coprocessor tuned for AI workloads. | Not open; only viable via partnership. |

**Implications:** we can start with off-the-shelf, MIT/Apache-friendly stacks (Risc0 + ezkl). If proofs need Tau-specific primitives, we can wrap our own lookup circuits while still committing ROM/Q tables via polynomial commitments; nothing prevents us from building a bespoke prover later, but it is not required for confidentiality.

## 3. Keeping lookup tables private
1. **Offline pipeline** (Python/Rust libs below) quantizes state → `(regime, action, emote, risk)` per tick and stores them in append-only binary traces.
2. **zk coprocessor** (Risc0 or ezkl) loads the encrypted/committed Q artifacts, evaluates the same logic the daemon would, and emits:
   - `q_trade_action` (2-bit, buy/sell/hold),
   - `q_emote_action` (2-bit, emoji palette or text macro),
   - `risk_budget` (bv[8], e.g., max position / cooldown ticks),
   - Proof+receipt referencing the commitment.
3. **Daemon adapter** verifies the receipt, mirrors actions into `idi/specs/V38_Minimal_Core/agent4_testnet_v38.tau` inputs, and logs proofs in the ledger. Tau never sees the table, only the index + proof flag; we keep optional `q_*_echo` outputs to audit what was consumed.

## 4. Layered lookup technology
```
┌──────────────┐  states, rewards, mood   ┌───────────────┐   proofs, echoes   ┌───────────────┐
│  Training    │ ───────────────────────▶ │ zk Coprocessor│ ─────────────────▶ │ Tau Specs     │
│  Libraries   │  (Python/Rust)           │ (Risc0/ezkl)  │   q_* streams      │ (IDI-ready)   │
└──────────────┘                          └───────────────┘                    └───────────────┘
```
- **Regime selector:** multi-dimensional quantizer (volatility, scarcity, EETF, daemon health) → `q_regime : bv[5]`.
- **Trading table:** per-regime action LUT returning 2-bit action + risk budgets.
- **Emotion table:** maps regime + “mood” counters to emoji/text macros for the art output pipeline (feeds future demos).
- **Mirrors/guards:** `idi/specs/libraries/idi_core/idi_core.tau` standardizes stream names, mirror outputs, and de-bounce timers so each spec can read the same schema with minimal cyclomatic complexity.

## 5. Training libraries & devkit
- Python package under `idi/training/python/idi_iann/` exposes:
  - `StateQuantizer`, `MarketEnvelope`, and `EmotionPalette` (SRP).
  - `QTrainer` implementing strategy & observer patterns (pluggable reward models, benchmark hooks).
  - CLI entrypoint for generating `.in` traces + manifest (checksums, version, proof policy).
- Rust crate under `idi/training/rust/idi_iann/` mirrors the same abstractions (traits for env, policy, serializer) and ships unit/property tests so both stacks stay aligned.
- Devkit (`idi/devkit/`):
  - `builder.py` CLI wraps the trainer, writes `streams/`, installs them into Tau inputs, and emits `artifact_manifest.json`.
  - `build_layers.py` reads `configs/layer_plan.json` and runs multiple configs (macro/micro/emote) for multi-layer deployments.
  - `manifest.py` computes SHA-256 fingerprints for every stream so zk proof verifiers know exactly what was trained.
- Shared YAML/JSON manifest ensures reproducibility across languages and fuels regression tests.

## 6. Tau Net & daemon alignment
- `tau_daemon_alpha` gains an “IDI bus” step: read verified actions, write `inputs/q_regime.in`, `inputs/q_trade_action.in`, `inputs/q_emote_action.in`, and `inputs/risk_budget.in` before kernel execution.
- Ledger entries store proof receipts + Q-table artifact digests so auditors can confirm which intelligence snapshot ran.
- Alignment Theorem metrics (scarcity, pressure, EETF) remain guard predicates that Tau specs must still satisfy before honoring any action, ensuring private intelligence cannot violate ethical economics.

## 7. When to build bespoke proof infra
- **Reuse** MIT/Apache stacks whenever action tables fit as ROM segments + lookup arguments and licensing is compatible with our daemon.
- **Build custom** when:
  - Tau spec needs cross-table constraints specific to timers/nonce gating;
  - Latency budget <10 ms and existing zkVMs cannot keep up;
  - We must integrate the prover directly inside the daemon binary without copyleft contamination.
- A custom prover would still follow the same commitment + lookup plan outlined above, so today’s layered lookup work is forward-compatible.

## 8. Implementation layers
1. **Perception & preprocessing** (Python/Rust trainers)  
   - Quantize raw feeds (price, scarcity, daemon guards, sentiment).  
   - Emit deterministic `.in` traces plus manifests (hashes, schema version, training config).  
   - Strategy pattern handles action selection; Observer hooks surface benchmark metrics.

2. **zk coprocessor tier**  
   - Load manifests + lookup tables, run inference, prove outputs.  
   - Proof receipts include: artifact hash, action stream Merkle root, prover metadata.

3. **Intelligent Daemon Interface (IDI) bus**  
   - Daemon verifies proofs, writes `inputs/q_*.in`, mirrors them post-run, and logs `(artifact, proof)` to the ledger for audit.

4. **Tau kernel layer**  
   - Specs import `idi_core` or inline the same clauses, keeping FSM logic unchanged while gating on `q_buy/q_sell/risk_budget`.  
   - Mirrors (`o19..o1F`) guarantee traceability and satisfy Tau’s stream symmetry.

5. **Monitoring & demos**  
   - AoT checklists + truth tables per demo; run scripts collect outputs into GitHub Pages–ready folders.

## 9. Directory map (worktree)
| Path | Purpose |
|------|---------|
| `idi/training/python/idi_iann/` | Python training toolkit + CLI + tests. |
| `idi/training/rust/idi_iann/` | Rust mirror crate for deterministic cross-checks + Clippy gate. |
| `idi/specs/libraries/idi_core/` | Stream contract + helper predicates for specs. |
| `idi/devkit/` | Builder CLI, manifest utilities, sample configs, and dev-focused tests. |
| `idi/zk/` | Proof manager + integration stubs for zkVMs. |
| `idi/docs/ARCHITECTURE.md` | This blueprint (kept in sync as architecture evolves). |
| `idi/docs/ZK_PRIVACY.md` | Proof stack / licensing reference. |
| `idi/demos/idi_demo/` | End-to-end replay harness with curated inputs + AoT verification. |

## 10. Verification & linting stack
- **Python:** Ruff (`ruff check`), pytest (unit tests), optional mypy for type coverage.  
- **Rust:** `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`.  
- **Tau:** `run_all.sh` demos treat Tau binary failures as CI blockers; mirror streams used by AoT/truth-table scripts.  
- **AoT workflows:** Verification markdown per demo enumerates premises → reasoning → conclusion; traces re-run whenever inputs change.

## 11. Operational playbook
1. Generate fresh lookup artifacts: `idi/training/python/run_idi_trainer.py --out build/idi_traces`.
2. Feed artifacts into zk coprocessor; archive `(manifest, proof)` bundle.
3. Drop `.in` files + proof receipts into spec folder (`idi/specs/V38_Minimal_Core/inputs/`).
4. Run Tau via daemon or demo script; inspect mirrored outputs + AoT checklist.
5. Commit manifests/proof hashes in documentation (never commit the private lookup tables unless intentionally open-source).

## 13. Devkit + zk workflow
1. Author a JSON config under `idi/devkit/configs/` (or edit the provided `regime_macro.json`, `regime_micro.json`, `emote_balanced.json`).
2. Run `python -m idi.devkit.builder --config ... --out ... --install-inputs idi/specs/V38_Minimal_Core/inputs` or batch multiple via `python -m idi.devkit.build_layers`.
3. Feed the resulting `artifact_manifest.json` + `streams/` into `idi/zk/proof_manager.py` (stub or real prover) to produce `proof.bin` and `receipt.json`.
4. Only after proof verification succeeds should the Tau daemon consume the new `q_*` streams.

## 12. Future extensions
- **Regime-aware composability:** extend `idi_core` to accept `q_regime`-indexed safety overrides (e.g., disable buys in critical regimes without regenerating tables).
- **Multi-agent coordination:** feed multiple Q-table outputs into a voter spec that enforces mutual exclusion before reaching `agent4_testnet_v38`.
- **On-chain attestations:** push proof receipts to Tau Net governance so other participants can verify which intelligence snapshot was active.
- **Visualization hooks:** integrate `o1D`/`o1F` streams with the Alignment Theorem website to animate mood-based art when demos run under GitHub Pages.


