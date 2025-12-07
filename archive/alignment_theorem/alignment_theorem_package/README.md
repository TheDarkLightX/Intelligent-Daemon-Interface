# Alignment Theorem Package

This folder collects all artifacts needed to publish the Alignment Theorem + VCC work as a standalone repository.

## Layout

- `docs/`
  - `THE_ALIGNMENT_THEOREM.md`: narrative summary.
  - `Alignment_Theorem_Academic.tex` / `.fodt`: journal-ready manuscript with TikZ figures.
  - `Alignment_Theorem_VCC_JournalReady.docx`: Word export.
  - `SIMULATION_RESULTS.md`: table referencing `analysis/simulations/alignment_sim_results.csv`.
  - `THREAT_MODEL.md`: threat analysis and roadmap.
  - `figures/alignment_convergence.png`: convergence plot generated from the simulations.
- `analysis/simulations/`
  - `run_alignment_simulations.py`: replicator/best-response sweeps.
  - `plot_alignment_results.py`: generates the PNG plot.
  - `alignment_sim_results.csv`: raw data.
- `proofs/AlignmentTheorem.lean`: mechanized Lean 4 proof.
- `visuals/alignment_theorem_visualizer.html`: neon dashboard for stakeholders.

## Repro Instructions

```bash
python3 analysis/simulations/run_alignment_simulations.py
python3 analysis/simulations/plot_alignment_results.py
```

Lean proof:
```bash
lake build
```

Tau specs + traces live in the parent repo (`specification/`, `verification/`), and the runtime substrate is described in [tau-testnet](https://github.com/IDNI/tau-testnet).
