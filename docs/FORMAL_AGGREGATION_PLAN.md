# Formal Aggregation Proof Plan

## Goal
Prove bounds for the Tau preference aggregator $\mathcal{A}$ (utilitarian/median/trimmed operators) showing that an adversary with stake weight $\lambda$ cannot move $E(t)$ by more than $\delta$.

## Steps
1. **Define operator semantics** in Tau logic (utility normalization, weight constraints, commit-reveal).  
2. **Express adversarial model**: number of Sybil identities, total stake share, credential proofs.  
3. **Add Lean lemmas** showing the aggregation function is 1-Lipschitz in individual utility contributions under normalization.  
4. **Encode constraints** in `formal_verification.py` to check weighted sums vs. bounds via Z3.  
5. **Model-check scenarios** via `analysis/simulations/run_alignment_simulations.py` using adversarial weights to confirm empirical bounds.

## Deliverables
- Lean module `proofs/AggregationBounds.lean`
- Tau spec snippet enforcing normalization
- Z3 script verifying numeric bounds
