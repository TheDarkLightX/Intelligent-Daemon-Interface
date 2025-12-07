#!/usr/bin/env python3
"""Generate convergence plot from alignment_sim_results.csv."""
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

CSV_PATH = Path('analysis/simulations/alignment_sim_results.csv')
OUT_DIR = Path('docs/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load() -> List[dict]:
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        return list(reader)

def main() -> None:
    rows = load()
    scenarios = [row['scenario'] for row in rows]
    steps = [None if row['convergence_step'] == 'none' else int(row['convergence_step']) for row in rows]
    plt.figure(figsize=(8,4))
    plt.bar(scenarios, steps, color='#9d4edd')
    plt.ylabel('Convergence step (share ≥ 0.99)')
    plt.title('Alignment convergence across scenarios')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    for idx, step in enumerate(steps):
        if step is not None:
            plt.text(idx, step + 1, str(step), ha='center', va='bottom', color='white', fontweight='bold')
    plt.tight_layout()
    out_file = OUT_DIR / 'alignment_convergence.png'
    plt.savefig(out_file, dpi=200)
    print(f'Wrote {out_file}')

if __name__ == '__main__':
    main()
