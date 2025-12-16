#!/usr/bin/env python3
"""Backtesting CLI for IDI/IANN policies on historical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from idi_iann.config import TrainingConfig
from idi_iann.policy import LookupPolicy
from idi_iann.trainer import QTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest IDI policies on historical data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python backtest.py data.csv --price-col close

  # With config and fees
  python backtest.py data.parquet --config config.json --fee-bps 10

  # Calibration only (compute drift/vol stats)
  python backtest.py data.csv --calibrate-only
        """,
    )
    parser.add_argument("data", type=Path, help="CSV/Parquet with columns: price, volume (optional).")
    parser.add_argument("--config", type=Path, help="Optional JSON config for quantization.")
    parser.add_argument("--policy", type=Path, help="Optional serialized policy JSON (else train fresh).")
    parser.add_argument("--out", type=Path, default=Path("outputs/backtest"), help="Output directory.")
    parser.add_argument("--price-col", type=str, default="price", help="Price column name.")
    parser.add_argument("--volume-col", type=str, default=None, help="Volume column name.")
    parser.add_argument("--risk-col", type=str, default=None, help="Risk event column name.")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Transaction fee in basis points.")
    parser.add_argument("--calibrate-only", action="store_true", help="Compute drift/vol stats and exit.")
    return parser.parse_args()


def _load_policy(cfg: TrainingConfig, path: Path | None) -> LookupPolicy:
    if path is None or not path.exists():
        trainer = QTrainer(cfg)
        policy, _ = trainer.run()
        return policy
    # simple manifest-free loader: assumes serialized q-values as json {state: {action: q}}
    raw = json.loads(path.read_text())
    policy = LookupPolicy()
    for state_str, actions in raw.items():
        state = tuple(int(x) for x in state_str.split(","))
        for act, val in actions.items():
            policy.update(state, act, 0.0)  # ensure entry exists
            # direct set
            policy._table[state].q_values[act] = float(val)  # type: ignore[attr-defined]
    return policy


def _quantize(series: pd.Series, buckets: int) -> pd.Series:
    # simple quantile-based bucketing
    return pd.qcut(series.rank(method="first"), buckets, labels=False, duplicates="drop").fillna(0).astype(int)


def _compute_kpis(
    returns: Iterable[float],
    actions: List[str] | None = None,
    risk_events: List[bool] | None = None,
) -> Dict[str, float]:
    """Compute comprehensive KPIs for trading and communication.

    Args:
        returns: Sequence of per-step returns
        actions: Optional list of actions taken (for action metrics)
        risk_events: Optional list of risk event flags (for alert metrics)

    Returns:
        Dictionary of KPI name to value
    """
    import numpy as np

    rets = list(returns)
    if not rets:
        return {
            "mean": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "cvar_5": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
        }

    arr = np.array(rets, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    sharpe = float(mean / (std + 1e-9))

    # Cumulative return and drawdown
    cum = np.cumsum(arr)
    total_return = float(cum[-1]) if len(cum) > 0 else 0.0
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / (np.abs(peak) + 1e-9)
    max_dd = float(np.max(dd))

    # CVaR (Conditional Value at Risk) at 5%
    sorted_rets = np.sort(arr)
    cutoff_idx = max(1, int(len(sorted_rets) * 0.05))
    cvar_5 = float(np.mean(sorted_rets[:cutoff_idx]))

    # Win rate
    wins = np.sum(arr > 0)
    win_rate = float(wins / len(arr)) if len(arr) > 0 else 0.0

    kpis = {
        "mean": mean,
        "std": std,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
        "cvar_5": cvar_5,
        "win_rate": win_rate,
        "total_return": total_return,
        "n_steps": len(arr),
    }

    # Action distribution metrics
    if actions:
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        total_actions = len(actions)
        kpis["action_buy_pct"] = action_counts.get("buy", 0) / total_actions
        kpis["action_sell_pct"] = action_counts.get("sell", 0) / total_actions
        kpis["action_hold_pct"] = action_counts.get("hold", 0) / total_actions

    # Communication / alert metrics
    if risk_events and actions:
        # Check if "alert" or similar communication happened during risk events
        # For now, use "sell" as a proxy for alert (defensive action)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for action, is_risk in zip(actions, risk_events):
            is_alert = action == "sell"  # Treat sell as defensive/alert
            if is_risk and is_alert:
                true_positives += 1
            elif not is_risk and is_alert:
                false_positives += 1
            elif is_risk and not is_alert:
                false_negatives += 1
            else:
                true_negatives += 1

        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        kpis["alert_precision"] = float(precision)
        kpis["alert_recall"] = float(recall)
        kpis["alert_f1"] = float(f1)

    return kpis


def _calibrate_stats(series: pd.Series) -> Dict[str, float]:
    rets = series.pct_change().dropna()
    if rets.empty:
        return {"drift": 0.0, "vol": 0.0}
    drift = float(rets.mean())
    vol = float(rets.std())
    return {"drift": drift, "vol": vol}


def backtest(
    cfg: TrainingConfig,
    policy: LookupPolicy,
    df: pd.DataFrame,
    price_col: str,
    volume_col: str | None,
    fee_bps: float = 0.0,
    risk_col: str | None = None,
):
    """Run backtest on historical data.

    Args:
        cfg: Training configuration
        policy: Policy to evaluate
        df: Historical data DataFrame
        price_col: Column name for price
        volume_col: Optional column name for volume
        fee_bps: Transaction fee in basis points
        risk_col: Optional column for risk event flags

    Returns:
        Tuple of (kpis dict, actions list)
    """
    # quantize inputs to existing state layout: (price, volume?, trend placeholder, scarcity, mood placeholder)
    df = df.copy()
    df["price_bucket"] = _quantize(df[price_col], cfg.quantizer.price_buckets)
    if volume_col:
        df["volume_bucket"] = _quantize(df[volume_col], cfg.quantizer.volume_buckets)
    else:
        df["volume_bucket"] = 0
    df["trend_bucket"] = 0
    df["scarcity_bucket"] = 0
    df["mood_bucket"] = 0

    position = 0
    last_price = float(df[price_col].iloc[0])
    returns: List[float] = []
    actions_taken: List[str] = []
    risk_events: List[bool] = []

    for _, row in df.iterrows():
        state = (
            int(row["price_bucket"]),
            int(row["volume_bucket"]),
            int(row["trend_bucket"]),
            int(row["scarcity_bucket"]),
            int(row["mood_bucket"]),
        )
        action = policy.best_action(state)
        # Convert Action enum to string if needed
        action_str = action.value if hasattr(action, "value") else str(action)

        price = float(row[price_col])
        pnl = (price - last_price) * position
        if position != 0 and fee_bps > 0.0:
            pnl -= price * (fee_bps / 1e4)
        returns.append(pnl)

        if action_str == "buy":
            position = 1
        elif action_str == "sell":
            position = -1

        actions_taken.append(action_str)

        # Track risk events if available
        if risk_col and risk_col in df.columns:
            risk_events.append(bool(row[risk_col]))
        else:
            # Compute simple risk proxy: large negative return
            ret = (price - last_price) / (last_price + 1e-9)
            risk_events.append(ret < -0.02)  # 2% drop as risk threshold

        last_price = price

    kpis = _compute_kpis(returns, actions_taken, risk_events)
    return kpis, actions_taken


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(**json.loads(args.config.read_text())) if args.config and args.config.exists() else TrainingConfig()
    policy = _load_policy(cfg, args.policy)

    if args.data.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    stats = _calibrate_stats(df[args.price_col])
    if args.calibrate_only:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "calibration.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("Calibration:", stats)
        return

    kpis, actions = backtest(
        cfg, policy, df, args.price_col, args.volume_col,
        fee_bps=args.fee_bps, risk_col=args.risk_col
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"kpis": kpis, "calibration": stats}
    (args.out / "kpis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "actions.txt").write_text("\n".join(actions), encoding="utf-8")

    # Print summary
    print("=" * 50)
    print("Backtest Results")
    print("=" * 50)
    print(f"Total Return: {kpis.get('total_return', 0):.4f}")
    print(f"Sharpe-like:  {kpis.get('sharpe_like', 0):.4f}")
    print(f"Max Drawdown: {kpis.get('max_drawdown', 0):.4f}")
    print(f"CVaR (5%):    {kpis.get('cvar_5', 0):.4f}")
    print(f"Win Rate:     {kpis.get('win_rate', 0):.2%}")
    if "alert_f1" in kpis:
        print("-" * 50)
        print(f"Alert Precision: {kpis.get('alert_precision', 0):.2%}")
        print(f"Alert Recall:    {kpis.get('alert_recall', 0):.2%}")
        print(f"Alert F1:        {kpis.get('alert_f1', 0):.2%}")
    print("=" * 50)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()

