#!/usr/bin/env python3
"""CLI for Off-Policy Evaluation of IDI/IANN policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idi_iann.config import TrainingConfig
from idi_iann.ope import LoggedDataset, OPEEvaluator, run_ope
from idi_iann.policy import LookupPolicy
from idi_iann.trainer import QTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run off-policy evaluation for IDI policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run OPE with all estimators
  python run_ope.py logged_data.json --config config.json --out ope_results.json

  # Run OPE with trained policy from file
  python run_ope.py logged_data.json --policy policy.json --out ope_results.json

  # Compare against baseline
  python run_ope.py logged_data.json --policy policy.json --baseline 0.5

Estimators:
  - DM:  Direct Method (Q-table based)
  - IPS: Importance Sampling
  - WIS: Weighted Importance Sampling
  - DR:  Doubly Robust
        """,
    )
    parser.add_argument("dataset", type=Path, help="Path to logged dataset JSON.")
    parser.add_argument("--config", type=Path, help="Training config JSON (for fresh policy).")
    parser.add_argument("--policy", type=Path, help="Serialized policy JSON.")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--baseline", type=float, default=None, help="Baseline value to compare against.")
    parser.add_argument("--out", type=Path, default=None, help="Output path for results JSON.")
    parser.add_argument("--estimators", nargs="+", default=["DM", "IPS", "WIS", "DR"],
                        help="Which estimators to run.")
    return parser.parse_args()


def load_policy(config_path: Path | None, policy_path: Path | None) -> LookupPolicy:
    """Load or train a policy."""
    if policy_path and policy_path.exists():
        raw = json.loads(policy_path.read_text())
        policy = LookupPolicy()
        for state_str, actions in raw.items():
            state = tuple(int(x) for x in state_str.split(","))
            for act, val in actions.items():
                policy.update(state, act, 0.0)
                policy._table[state].q_values[act] = float(val)
        return policy

    cfg = TrainingConfig()
    if config_path and config_path.exists():
        cfg = TrainingConfig(**json.loads(config_path.read_text()))

    trainer = QTrainer(cfg)
    policy, _ = trainer.run()
    return policy


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return

    policy = load_policy(args.config, args.policy)
    dataset = LoggedDataset.from_json(args.dataset)
    evaluator = OPEEvaluator(policy, discount=args.discount)

    print("=" * 60)
    print("Off-Policy Evaluation Results")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {len(dataset.episodes)}")
    print(f"Discount: {args.discount}")
    print("-" * 60)

    results = {}
    for estimator in args.estimators:
        if estimator == "DM":
            result = evaluator.direct_method(dataset)
        elif estimator == "IPS":
            result = evaluator.importance_sampling(dataset, weighted=False)
        elif estimator == "WIS":
            result = evaluator.importance_sampling(dataset, weighted=True)
        elif estimator == "DR":
            result = evaluator.doubly_robust(dataset)
        else:
            print(f"Unknown estimator: {estimator}")
            continue

        results[estimator] = result.to_dict()

        ci_low, ci_high = result.confidence_interval
        print(f"{estimator:4s}: {result.value_estimate:8.4f} ± {result.standard_error:.4f}")
        print(f"      95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        if args.baseline is not None:
            improvement = result.value_estimate - args.baseline
            pct = (improvement / abs(args.baseline)) * 100 if args.baseline != 0 else 0
            print(f"      vs baseline: {improvement:+.4f} ({pct:+.1f}%)")
        print()

    print("=" * 60)

    if args.out:
        output = {
            "dataset": str(args.dataset),
            "discount": args.discount,
            "baseline": args.baseline,
            "results": results,
        }
        args.out.write_text(json.dumps(output, indent=2))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()

