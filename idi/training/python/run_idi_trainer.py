#!/usr/bin/env python3
"""CLI helper to generate IDI lookup traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idi_iann.config import TrainingConfig
from idi_iann.trainer import QTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lookup-table traces for Tau specs.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--out", type=Path, default=Path("outputs/q_traces"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and args.config.exists():
        cfg = TrainingConfig(**json.loads(args.config.read_text()))
    else:
        cfg = TrainingConfig()
    trainer = QTrainer(cfg)
    policy, trace = trainer.run()
    trace.export(args.out)
    manifest_path = args.out / "manifest.json"
    policy.serialize_manifest(manifest_path)
    print(f"Traces written to {args.out}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

