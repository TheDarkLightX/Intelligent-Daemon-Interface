#!/usr/bin/env python3
"""Replicator-style simulations for the Alignment Theorem.
Generates CSV summary of convergence times under varying parameters.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

@dataclass
class Scenario:
    name: str
    growth: float  # scarcity growth per step
    epsilon: float  # EETF floor
    g_max: float    # bounded misaligned gain

@dataclass
class Result:
    scenario: Scenario
    convergence_step: int | None
    final_share: float

M0 = 5.0
K_R = 1.0
TIER_MIN = 1.0
K_C = 0.5
X = 1.0
MAX_STEPS = 200
TARGET_SHARE = 0.99

SCENARIOS: List[Scenario] = [
    Scenario("baseline_fast", growth=0.08, epsilon=0.8, g_max=5.0),
    Scenario("baseline_slow", growth=0.04, epsilon=0.8, g_max=5.0),
    Scenario("stochastic_eetf", growth=0.06, epsilon=0.6, g_max=5.0),
    Scenario("adversarial_injection", growth=0.06, epsilon=0.8, g_max=8.0),
]


def scarcity(step: int, growth: float) -> float:
    return M0 * ((1.0 + growth) ** step)


def eetf(step: int, epsilon: float) -> float:
    # oscillate mildly to mimic changing consensus
    return epsilon + 0.2 * math.sin(step / 12.0)


def payoff_eth(step: int, scen: Scenario) -> float:
    return K_R * TIER_MIN * scarcity(step, scen.growth)


def payoff_uneth(step: int, scen: Scenario) -> float:
    return scen.g_max - K_C * X * scarcity(step, scen.growth) * max(eetf(step, scen.epsilon), 0.0)


def simulate(scen: Scenario) -> Result:
    share = 0.2
    convergence = None
    for step in range(MAX_STEPS):
        u_eth = payoff_eth(step, scen)
        u_un = payoff_uneth(step, scen)
        delta = share * (1.0 - share) * (u_eth - u_un) * 0.01
        share = max(0.0, min(1.0, share + delta))
        if share >= TARGET_SHARE and convergence is None:
            convergence = step
    return Result(scenario=scen, convergence_step=convergence, final_share=share)


def run_all() -> Iterable[Result]:
    for scen in SCENARIOS:
        yield simulate(scen)


def write_csv(results: Iterable[Result], dest: Path) -> None:
    with dest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "growth", "epsilon", "g_max", "convergence_step", "final_share"])
        for res in results:
            writer.writerow([
                res.scenario.name,
                res.scenario.growth,
                res.scenario.epsilon,
                res.scenario.g_max,
                res.convergence_step if res.convergence_step is not None else "none",
                f"{res.final_share:.4f}",
            ])


def main() -> None:
    out_dir = Path("analysis/simulations")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = list(run_all())
    write_csv(results, out_dir / "alignment_sim_results.csv")
    for res in results:
        print(f"{res.scenario.name}: convergence={res.convergence_step}, final_share={res.final_share:.3f}")


if __name__ == "__main__":
    main()
