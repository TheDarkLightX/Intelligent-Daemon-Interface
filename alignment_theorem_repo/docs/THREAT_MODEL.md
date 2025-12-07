# Alignment Theorem Threat Model (Tau Net)

## Aggregation Layer ($\mathcal{A}$)
- **Sybil/credential spam**: require stake-weighted or credentialed submissions; plan to prove that adversarial weight $\leq \lambda$ cannot move $E(t)$ more than $\delta$. Future work: formalize aggregator contract, run zk-attested identities.
- **Collusion / preference utilitarian abuse**: implement median/trimmed-mean fallback when stake concentration exceeds policy bounds; for utilitarian aggregation, enforce normalization constraints and publish proofs (or audits) of any weight scaling so no coalition can inflate its utilities.

## Pointwise Revision Safety
- Tau testnet replays rule edits per formula. We still need Lean guards ensuring certain predicates are immutable (e.g., constitutional invariants, extralogical policy gates).
- Add pre-commit hooks verifying that new rules cannot call unsafe extralogical functions without developer approval.

## Extralogical Primitives
- **Commit-reveal**: require hashing algorithm parity between Tau rule and runtime; add tests verifying reveal window closure.
- **Oracles/MEV**: use `libraries/mev_oracle_safety_v1.tau` monitors; future work includes formalizing freshness checks in Lean and linking them to Tau daemon metrics (see `tau-testnet` repo).

## Economic Stressors
- Extend `analysis/simulations/run_alignment_simulations.py` with tail-risk (shock drops in $E(t)$, partial reversions). Use tau-testnet traces to seed real scarcity trajectories.

## Roadmap
1. Implement stake-based throttling contract for $\mathcal{A}$.
2. Write Lean lemmas marking specific predicate families as immutable under pointwise revision.
3. Add regression tests for commit-reveal and MEV monitors.
4. Publish extended simulation plots + empirical tau-testnet data (see `docs/SIMULATION_RESULTS.md`).
