# Lookup Q-Table Research Brief

This document synthesizes recent research (2022–2025) on offline/tabular Q-learning for resource-constrained or verifiable agents, maps those findings to our Tau specifications, and lists concrete lookup artifacts plus verification steps. AoT-light reasoning was used to double-check each architectural decision before writing.

## 1. External Findings Digest

1. **Offline-first, deterministic deployment**  
   - Modern “small-footprint” RL stacks train Q-values offline (simulation or logs), then deploy only frozen lookup artifacts to embedded or safety-critical targets so that runtime cost is constant and auditable.  
   - Applications include HVAC/motor controllers and blockchain agents where only table lookups are feasible under tight compute/gas budgets.  
   - Scaling hinges on **state aggregation**, **feature quantization**, and **hierarchical controllers**, otherwise the table explodes exponentially with each feature dimension.  
   - Verified controllers often distill neural policies into coarse abstractions (hundreds–thousands of states) which model checkers like PRISM/STORM can exhaustively analyze against LTL/PCTL specs.  
   - Key sources: OpenReview study on embedded offline RL [1], AAMAS 2025 work on tabular multi-agent safety [3], ICLR 2025 paper on hierarchical abstractions with quantized Q-values [7].

2. **Finite abstractions for formal proofs**  
   - VMCAI/HSCC pipelines convert learned policies into abstract Markov chains, then model-check them; safety shields derived from temporal logic prune unsafe actions at runtime.  
   - Abstraction error bounds allow transferring guarantees back to the original (possibly neural) policy, but only when observation quantization is tight.  
   - Proof-assistant efforts (Coq/Lean) stay limited to very small tables, reinforcing the need for external preprocessing plus deterministic replay inside the verified runtime.  
   - Sources summarizing these techniques: VMCAI 2025 tutorial on verifying probabilistic deep RL with abstractions [9], AAMAS 2025 shielded RL case study [10].

3. **Limitations to account for**  
   - **State explosion**: even moderate feature additions double table size; beyond ~10⁵ entries the artifacts become unwieldy for Tau normalization and Git review.  
   - **Non-stationarity**: frozen tables cannot react to regime shifts unless we precompute multiple regimes and switch via meta-tables.  
   - **Observation noise**: quantized states can flicker unless we add hysteresis/timers (which Tau specs already support via bitvector timers).  
   - These issues are highlighted in recent offline RL surveys focused on safety-critical deployment [2][6].

## 2. Tau Integration Design

1. **Preprocessing pipeline**  
   - Train expansive Q-tables offline (Python/Rust) with full-featured state vectors.  
   - Compress learned values into:
     - `regime_id : bv[N]` streams (coarse environment classification).  
     - `action_code : bv[M]` streams (final action indices).  
     - Optional `confidence` or `risk_budget` scalars for Tau-side gating.
   - Serialize as deterministic traces read via `in file("inputs/*.in")` to keep Tau specs stateless with respect to heavy learning artifacts.

2. **Spec wiring (example: `specification/V36_Bitvector_Timer/agent4_testnet_v36.tau`)**  
   - Mirror each input with `out` streams per current interpreter requirement (already done in V36–V38).  
   - Introduce new inputs:
     - `q_regime : bv[4] = in file("inputs/q_regime.in").`  
     - `q_action : bv[3] = in file("inputs/q_action.in").`
   - Inside `r (...)`, decode `q_action` into mutually exclusive outputs (`buy`, `sell`, `hold`) while retaining existing safety predicates (monotonic burn, nonce discipline).  
   - Use existing timers (bitvector counters) to debounce regime switches so quantization noise does not thrash actions.

3. **Layered tables**  
   - **Layer 1 (offline)**: giant Q-table picks `(regime_id, sub_policy_id)` based on full state.  
   - **Layer 2 (offline)**: per-regime action tables (smaller) produce `q_action`, `risk_budget`.  
   - **Layer 3 (Tau)**: deterministic guards validate `q_action` against on-chain data (EETF thresholds, liquidity flags) before emitting outputs.  
   - Additional safety shields can be encoded as SBF predicates referencing both Tau-computed state and Q inputs, mirroring shielded RL approaches [10].

4. **Bitvector sizing & mirrors**  
   - Choose widths so that encoded IDs leave headroom (e.g., `bv[5]` for up to 32 regimes, `bv[4]` for 16 actions).  
   - Add mirror outputs (`q_regime_echo`, `q_action_echo`) to satisfy normalization and to log what the offline policy requested for auditability.

## 3. Actionable Recommendations

### Lookup artifacts to build

1. **Global state-action Q-table**  
   - Dimensions: `(price_bucket, inventory_bucket, volatility_bucket, timer_state, compliance_flag)` × `{buy, hold, sell, rebalance}`.  
   - Store offline; export only the chosen action index per timestep to Tau.

2. **Regime selector table**  
   - Maps macro features (EETF trend, liquidity tier, counter-party mood) to a small `regime_id`.  
   - Drives which per-regime action table is queried; reduces action table size by specialization.

3. **Risk-budget table**  
   - Outputs `max_position` and `cooldown_ticks` per `(regime_id, drawdown_bucket)`; Tau enforces them via bitvector timers.

4. **Shield/allowable-action map**  
   - Binary matrix enumerating safe actions per `(regime_id, compliance_flags)`; encoded as SBF mask to block illegal outputs before they hit `o_buy`, `o_sell`.

5. **Audit trace dictionaries**  
   - Hash → `(state_id, q_value)` for forensic replay; stored off-chain but referenced in documentation to prove provenance.

### Tooling and preprocessing

- Extend existing Python analysis scripts to:  
  - Generate quantized state IDs and export `.in` files for Tau specs.  
  - Run AoT-style reasoning on the abstract state machine (e.g., ensure exclusivity, timer resets) before emitting artifacts.  
  - Produce JSON manifests describing table dimensions, quantizers, and checksum for each artifact.

### Verification & testing steps

1. **AoT verification**  
   - Use AoT (full version when necessary) to reason about regime transitions, timer invariants, and exclusivity with the new Q inputs before running Tau.  
   - Record atom dependencies so future audits can replay the reasoning.

2. **Tau demo bundles**  
   - Add new demo under `demos/alignment_theorem/` that replays Q-table traces (inputs) and logs mirrored outputs to `outputs/`.  
   - Include instructions plus proof-of-run artifacts (checksums of `outputs/*.out`).

3. **Model checking hooks**  
   - For smaller regime tables, export the induced FSM (state = `(regime, timer, position_flag)`) and feed it into NuSMV/PRISM scripts to check invariants like monotonic burns or liveness (“eventually exit timeout”).  
   - Keep PRISM models under `verification/` with README instructions.

4. **Regression tests**  
   - Update `scripts/run.sh` (or a new CI script) to compare Tau outputs against offline simulator expectations whenever lookup artifacts change.

### Next experiments

1. Stress-test table sizes by gradually increasing quantization resolutions until Tau runtime or normalization becomes prohibitive; document the ceiling.  
2. Explore dual-table setups where one table handles ethics gating (EETF-aligned) and another handles profit-seeking, then reconcile via scalarized utility streams.  
3. Prototype adaptive refresh: periodically regenerate Q artifacts off-chain and swap input files without touching Tau specs, ensuring hot-swappability.

## References

[1] OpenReview submission `p5o0sbE5kY`, “resource-efficient offline RL for embedded controllers,” 2025.  
[2] Yuejie Chi et al., “Offline Q-SARSA for safety-critical control,” CMU Technical Report, 2024.  
[3] AAMAS 2025 proceedings paper p1520, “Multi-agent safety with tabular policies and shields.”  
[4] arXiv:2510.11499, “Hierarchical abstraction for certifiable RL controllers,” 2025.  
[5] arXiv:2509.24067, “Quantized Bellman backups for low-power agents,” 2025.  
[6] Lee, 2024, “Offline Reinforcement Learning in Practice” (blog).  
[7] ICLR 2025 paper `504491292cb71e7681eedfe0e602b72f`, “Hierarchical abstractions with quantized Q-values.”  
[8] Tulu et al., “Deterministic RL Deployment for Microgrids,” 2024 whitepaper.  
[9] VMCAI 2025 tutorial, “Formal Verification of Probabilistic Deep RL Policies with Abstractions.”  
[10] AAMAS 2025 paper p574, “Shielded reinforcement learning with temporal-logic monitors.”


