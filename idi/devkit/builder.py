"""CLI for building layered IDI lookup artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from idi.training.python.idi_iann.config import (
    TrainingConfig,
    QuantizerConfig,
    RewardWeights,
    EmoteConfig,
)
from idi.training.python.idi_iann.trainer import QTrainer
from idi.devkit.manifest import build_manifest, write_manifest


def load_training_config(config_path: Path) -> TrainingConfig:
    data = json.loads(config_path.read_text())
    quantizer = QuantizerConfig(**data.get("quantizer", {}))
    rewards = RewardWeights(**data.get("rewards", {}))
    emote = EmoteConfig(**data.get("emote", {}))
    return TrainingConfig(
        episodes=data.get("episodes", 128),
        episode_length=data.get("episode_length", 64),
        discount=data.get("discount", 0.92),
        learning_rate=data.get("learning_rate", 0.2),
        exploration_decay=data.get("exploration_decay", 0.995),
        quantizer=quantizer,
        rewards=rewards,
        emote=emote,
    )


def metadata_from_args(pairs: List[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Metadata entries must be KEY=VALUE (got {pair})")
        key, value = pair.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def copy_streams(stream_dir: Path, install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    for stream_file in stream_dir.glob("*.in"):
        shutil.copy2(stream_file, install_dir / stream_file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IDI lookup artifacts from config.")
    parser.add_argument("--config", type=Path, required=True, help="Training config JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--install-inputs",
        type=Path,
        help="Optional directory (e.g., idi/specs/V38_Minimal_Core/inputs) to receive streams.",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metadata entries recorded in the manifest (repeatable).",
    )
    args = parser.parse_args()

    training_config = load_training_config(args.config)
    builder = QTrainer(training_config)
    policy, trace = builder.run()

    streams_dir = args.out / "streams"
    policy_manifest_path = args.out / "policy_manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)
    if streams_dir.exists():
        shutil.rmtree(streams_dir)
    streams_dir.mkdir()
    trace.export(streams_dir)

    if hasattr(policy, "serialize_manifest"):
        policy.serialize_manifest(policy_manifest_path)  # type: ignore[attr-defined]

    try:
        metadata = metadata_from_args(args.meta)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = build_manifest(
        config_path=args.config,
        stream_dir=streams_dir,
        metadata=metadata,
    )
    manifest_path = args.out / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    if args.install_inputs:
        copy_streams(streams_dir, args.install_inputs)

    print("✅ IDI artifact generated")
    print(f" - Streams dir: {streams_dir}")
    print(f" - Manifest:    {manifest_path}")
    if args.install_inputs:
        print(f" - Installed to: {args.install_inputs}")


if __name__ == "__main__":
    main()

