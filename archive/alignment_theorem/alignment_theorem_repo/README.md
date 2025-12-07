# Alignment Theorem on Tau Net

This directory is a drop-in payload for [`github.com/TheDarkLightX/AlignmentTheorem`](https://github.com/TheDarkLightX/AlignmentTheorem). It contains the paper, Lean proof, simulations, and verification scripts described throughout the Tau Alignment Theorem workstream.

How can we ensure that autonomous economic agents, human or non-human, act in ways that respect the values of the communities that host them? Traditional approaches to AI alignment try to encode values directly into models through hand-written rules, reinforcement learning from human feedback, or constitutional documents. These approaches are hard to keep consistent and easy to game. They also tend to be model- and vendor-specific. Tau Net offers a different approach. Users express their ethical views as logical theories in Tau Language. Tau aggregates these theories into a shared, on-chain ethical worldview. From that worldview it computes a live scalar signal, the Ethical–Eco Transaction Factor (EETF), and feeds this signal into economic agents such as the Virtuous Cycle Compounder (VCC). Because scarcity-driven rewards are tied directly to the EETF, rational agents find that aligning with the community’s worldview is the only profit-maximizing strategy.

## Contents

| Path | Purpose |
| --- | --- |
| `docs/Alignment_Theorem_Academic.tex` | LaTeX source for the journal-ready paper |
| `docs/Alignment_Theorem_Academic.pdf` | Pre-built PDF for quick review |
| `docs/SIMULATION_RESULTS.md` | Narrative summary of convergence experiments |
| `docs/THREAT_MODEL.md` | Aggregation/MEV/pointwise-revision threat catalog |
| `docs/VERIFICATION_SUMMARY.md` | Lean, simulator, FSM, and stress-test results |
| `analysis/simulations/` | Python scripts + CSV data for convergence sweeps |
| `verification/` | Tau exact simulator, completeness checker, SMT harness |
| `proofs/AlignmentTheorem.lean` | Lean 4 proof with no `sorry` placeholders |

## Build & Verification Instructions

### Paper

```bash
tectonic -o docs docs/Alignment_Theorem_Academic.tex
```

### Lean Proof

```bash
cd proofs
lake build
```

### Simulations

```bash
cd analysis/simulations
python3 run_alignment_simulations.py
python3 plot_alignment_results.py
```

### Tau Trace Harness

```bash
cd verification
python3 tau_exact_simulator.py  # runs reference traces
python3 tau_completeness_checker.py ../docs/Alignment_Theorem_Academic.tex
python3 verify_transitions_tau.py            # optional SMT spot checks
```

## Repository Usage

1. Initialize the target repo (or clone `git@github.com:TheDarkLightX/AlignmentTheorem.git`).
2. Copy the contents of this directory into that repository root.
3. Commit and push:

```bash
git add .
git commit -m "Import Alignment Theorem artifacts"
git push origin main
```

Feel free to add additional Tau specifications, dashboards, or CI workflows on top of this foundation.

