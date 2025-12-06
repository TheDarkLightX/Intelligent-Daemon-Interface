"""Batch builder for layered IDI lookup tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from idi.devkit.builder import build_artifact


def load_plan(plan_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(plan_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Layer plan must be a list of layer objects")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multiple IDI layers from a plan file.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("idi/devkit/configs/layer_plan.json"),
        help="JSON file describing the layers to build.",
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    summaries: List[str] = []
    for entry in plan:
        name = entry.get("name", "layer")
        config = Path(entry["config"])
        out_dir = Path(entry["out"])
        install_inputs = Path(entry["install_inputs"]) if entry.get("install_inputs") else None
        metadata = entry.get("metadata", [])
        print(f"▶ Building layer '{name}' from {config}")
        build_artifact(
            config_path=config,
            out_dir=out_dir,
            install_inputs=install_inputs,
            metadata_pairs=list(metadata),
        )
        summaries.append(f"{name}: streams -> {out_dir / 'streams'}")

    print("\n✅ Layer build summary")
    for line in summaries:
        print(f" - {line}")


if __name__ == "__main__":
    main()

