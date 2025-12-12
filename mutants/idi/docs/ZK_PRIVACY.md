# IDI zk-Privacy Strategy

## Goals
- Keep lookup-table intelligence (trading + emotive cues) private.
- Emit only proof-backed action indices (`q_buy`, `q_sell`, `risk_budget_ok`, `q_emote_*`) into Tau specs.
- Reuse permissively licensed zkML/zkVM stacks when possible; fall back to bespoke prover only if latency or language constraints demand it.

## Off-the-shelf stacks

| Stack | License | Notes |
|-------|---------|-------|
| Risc0 zkVM | Apache-2.0 + MIT dual-license (`LICENSE-APACHE`, `LICENSE-MIT`). [link](https://github.com/risc0/risc0/tree/main) | General-purpose zkVM for Rust/C code; supports ROM commitments and receipt verification suitable for our daemon. |
| ezkl | Apache-2.0 (`LICENSE`). [link](https://github.com/Spectral-Finance/ezkl/blob/main/LICENSE) | Halo2-based zkML with native plookup gadgets; ideal for frozen lookup inference. |
| zkSync era-boojum | Apache-2.0 (`LICENSE-APACHE`). [link](https://github.com/matter-labs/era-boojum/blob/main/LICENSE-APACHE) | Optimized STARK prover with RAM tables; reuse if we align with zkSync ecosystems. |
| Polygon zkEVM prover | AGPLv3 (`LICENSE`). [link](https://github.com/0xPolygon/zkevm-prover/blob/main/LICENSE) | Production-ready but copyleft; wrap as remote service to avoid contaminating the Tau daemon. |
| Nil Foundation zkLLVM | No explicit license in repo (`license: null` via GitHub API). [link](https://github.com/NilFoundation/zkLLVM) | Treat as “source available”; requires separate legal clearance before embedding. |
| Modulus Aura zkVM | No public repository/license (press-only). | Assume proprietary; collaborate only via TFH partnership if needed. |

## Integration blueprint
1. **Preprocessing:** Python/Rust trainers (see `idi/training/...`) generate traces + manifest via `idi/devkit/builder.py`.
2. **Proving:** 
   - Invoke `idi/zk/proof_manager.py` with a stub (default) or real prover command (Risc0, ezkl). For ROM-style lookups (static Q tables), wrap exported tables inside ezkl circuits; for mixed computation (timers, debounce), port the logic to Risc0 to attest to the entire inference function.
3. **Verification:** 
   - Tau daemon reads `receipt.json`, recomputes the digest, and only then copies `streams/` into spec inputs. Ledger entries should capture `(artifact hash, proof hash, verification result)` so Tau Net peers can audit.
4. **Tau interface:** Verified outputs are written to `inputs/q_*.in` before each Tau tick. Mirror outputs (`o19`..`o1F`) already log these for traceability.

## When to build custom zk infra
- Need <10 ms proofs embedded inside the daemon binary (current zkVMs exceed this).
- Desire Tau-specific primitives (e.g., native bitvector semantics, direct reference to Tau solver).
- Licensing conflicts (e.g., we need full MIT/Apache stack but only AGPL option exists for a given feature).

In those cases, we can reuse the same architecture: commit lookup tables via polynomial commitments, verify lookups with custom plookup argument, and expose receipts identical to the external stacks. This keeps Tau specs unchanged while letting us iterate on prover performance independently.

