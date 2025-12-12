#!/usr/bin/env python3
"""CLI helper to generate IDI lookup traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idi_iann.config import TrainingConfig
from idi_iann.trainer import QTrainer
from idi_iann.crypto_env import MarketParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lookup-table traces for Tau specs.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--out", type=Path, default=Path("outputs/q_traces"), help="Output directory.")
    parser.add_argument("--use-crypto-env", action="store_true", help="Use the crypto market simulator.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (set to -1 for nondeterministic).")
    parser.add_argument("--drift-bull", type=float, help="Bull drift override.")
    parser.add_argument("--drift-bear", type=float, help="Bear drift override.")
    parser.add_argument("--vol-base", type=float, help="Base volatility override.")
    parser.add_argument("--vol-panic", type=float, help="Panic volatility override.")
    parser.add_argument("--shock-prob", type=float, help="Shock probability override.")
    parser.add_argument("--shock-scale", type=float, help="Shock scale override.")
    parser.add_argument("--fee-bps", type=float, help="Trading fee in bps override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and args.config.exists():
        cfg = TrainingConfig(**json.loads(args.config.read_text()))
    else:
        cfg = TrainingConfig()
    seed = None if args.seed == -1 else args.seed
    market_params = MarketParams()
    if args.drift_bull is not None:
        market_params.drift_bull = args.drift_bull
    if args.drift_bear is not None:
        market_params.drift_bear = args.drift_bear
    if args.vol_base is not None:
        market_params.vol_base = args.vol_base
    if args.vol_panic is not None:
        market_params.vol_panic = args.vol_panic
    if args.shock_prob is not None:
        market_params.shock_prob = args.shock_prob
    if args.shock_scale is not None:
        market_params.shock_scale = args.shock_scale
    if args.fee_bps is not None:
        market_params.fee_bps = args.fee_bps

    def env_factory(cfg: TrainingConfig, use_crypto: bool, seed_val: int):
        if use_crypto:
            mp = market_params
            mp.seed = seed_val
            return QTrainer._default_env_factory_static(cfg, use_crypto=True, seed=seed_val, market_params=mp)
        return QTrainer._default_env_factory_static(cfg, use_crypto=False, seed=seed_val, market_params=None)

    trainer = QTrainer(cfg, use_crypto_env=args.use_crypto_env, seed=seed, env_factory=env_factory)
    policy, trace = trainer.run()
    trace.export(args.out)
    manifest_path = args.out / "manifest.json"
    policy.serialize_manifest(manifest_path)
    print(f"Traces written to {args.out}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

