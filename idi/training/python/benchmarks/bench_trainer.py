#!/usr/bin/env python3
"""Benchmark runner for IDI QTrainer with crypto-style environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import math

from idi_iann.config import TrainingConfig
from idi_iann.trainer import QTrainer


def run_once(cfg: TrainingConfig, seed: int, use_crypto: bool) -> dict:
    trainer = QTrainer(cfg, seed=seed, use_crypto_env=use_crypto)
    _, trace = trainer.run()
    stats = trainer.stats()
    # Compute alert precision/recall vs risk_event if present in ticks
    alerts = 0
    risk_events = 0
    true_alerts = 0
    for tick in trace.ticks:
        alert = tick.get("q_emote_alert", 0)
        risk = tick.get("risk_event", 0)
        alerts += alert
        risk_events += risk
        if alert and risk:
            true_alerts += 1
    precision = true_alerts / alerts if alerts else 0.0
    recall = true_alerts / risk_events if risk_events else 0.0
    # Trading KPIs: cumulative reward as proxy PnL; simple Sharpe and drawdown over episode rewards
    rewards = stats.get("episode_rewards", [])
    mean_ep = sum(rewards) / len(rewards) if rewards else 0.0
    std_ep = math.sqrt(sum((r - mean_ep) ** 2 for r in rewards) / len(rewards)) if rewards else 0.0
    sharpe = mean_ep / (std_ep + 1e-6) if rewards else 0.0
    cum = [sum(rewards[: i + 1]) for i in range(len(rewards))] if rewards else []
    peak = -1e9
    mdd = 0.0
    for c in cum:
        peak = max(peak, c)
        mdd = min(mdd, c - peak)

    return {
        "seed": seed,
        "mean_reward": stats["mean_reward"],
        "comm_action_counts": stats["comm_action_counts"],
        "ticks": len(trace.ticks),
        "alert_precision": precision,
        "alert_recall": recall,
        "sharpe_like": sharpe,
        "max_drawdown": mdd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IDI QTrainer.")
    parser.add_argument("--config", type=Path, help="Training config JSON", required=False)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--crypto-env", action="store_true", help="Use crypto-style simulator")
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.config and args.config.exists():
        cfg = TrainingConfig(**json.loads(args.config.read_text()))

    results: List[dict] = []
    for seed in args.seeds:
        res = run_once(cfg, seed, args.crypto_env)
        results.append(res)
        print(f"seed={seed} mean_reward={res['mean_reward']:.4f} comm={res['comm_action_counts']}")

    summary: Dict[str, object] = {
        "runs": results,
        "mean_of_means": sum(r["mean_reward"] for r in results) / len(results),
        "mean_alert_precision": sum(r["alert_precision"] for r in results) / len(results),
        "mean_alert_recall": sum(r["alert_recall"] for r in results) / len(results),
        "mean_sharpe_like": sum(r["sharpe_like"] for r in results) / len(results),
        "mean_max_drawdown": sum(r["max_drawdown"] for r in results) / len(results),
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()

