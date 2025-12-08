"""CLI for building layered IDI lookup artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from idi.zk.policy_commitment import build_policy_commitment, save_policy_commitment


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


def metadata_from_pairs(pairs: List[str]) -> Dict[str, str]:
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


def build_artifact(
    *,
    config_path: Path,
    out_dir: Path,
    install_inputs: Path | None = None,
    metadata_pairs: List[str] | None = None,
    verbose: bool = False,
) -> Path:
    """Train, export, and optionally install a lookup-table artifact.

    Args:
        config_path: Path to training config JSON
        out_dir: Output directory for artifacts
        install_inputs: Optional directory to install streams (e.g., Tau spec inputs/)
        metadata_pairs: Optional metadata key=value pairs
        verbose: Whether to print progress messages

    Returns:
        Path to generated manifest file

    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if verbose:
        print(f"Loading config from {config_path}...")

    try:
        training_config = load_training_config(config_path)
        training_config.validate()
    except Exception as e:
        raise ValueError(f"Invalid config: {e}") from e

    if verbose:
        print(f"Training {training_config.episodes} episodes...")

    builder = QTrainer(training_config)
    policy, trace = builder.run()

    if verbose:
        stats = builder.stats()
        print(f"Training complete. Mean reward: {stats['mean_reward']:.2f}")

    streams_dir = out_dir / "streams"
    policy_dir = out_dir / "policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    if streams_dir.exists():
        shutil.rmtree(streams_dir)
    streams_dir.mkdir()

    if verbose:
        print(f"Exporting {len(trace.ticks)} trace ticks to {streams_dir}...")

    trace.export(streams_dir)

    # Optional: export policy commitment for downstream proofs
    policy_commitment = None
    policy_proofs = None
    if hasattr(policy, "table"):
        try:
            entries = {
                state: QTableEntry.from_float(
                    q_hold=vals.get("hold", 0.0),
                    q_buy=vals.get("buy", 0.0),
                    q_sell=vals.get("sell", 0.0),
                )
                for state, vals in policy.table.items()  # type: ignore[attr-defined]
            }
            policy_commitment, policy_proofs = build_policy_commitment(entries)
            save_policy_commitment(policy_dir, policy_commitment, policy_proofs)
        except Exception:
            # Non-fatal: continue without commitment if policy structure unexpected
            policy_commitment = None

    metadata_pairs = metadata_pairs or []
    try:
        metadata = metadata_from_pairs(metadata_pairs)
    except ValueError as e:
        raise ValueError(f"Invalid metadata: {e}") from e

    if verbose:
        print("Building manifest...")

    manifest = build_manifest(
        config_path=config_path,
        stream_dir=streams_dir,
        metadata={
            **metadata,
            **(
                {
                    "policy_root_hex": policy_commitment.root.hex(),
                    "policy_leaf_encoding": policy_commitment.leaf_encoding,
                    "policy_q_scale": str(policy_commitment.q_scale),
                }
                if policy_commitment
                else {}
            ),
        },
    )
    manifest_path = out_dir / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    if install_inputs:
        if verbose:
            print(f"Installing streams to {install_inputs}...")
        copy_streams(streams_dir, install_inputs)
        if verbose:
            print(f"Installed {len(list(streams_dir.glob('*.in')))} stream files")

    if verbose:
        print(f"Artifact built successfully: {manifest_path}")

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IDI lookup artifacts from config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic build
  python -m idi.devkit.builder --config configs/sample.json --out artifacts/my_agent

  # Build and install to Tau spec inputs
  python -m idi.devkit.builder \\
      --config configs/sample.json \\
      --out artifacts/my_agent \\
      --install-inputs specs/V38_Minimal_Core/inputs

  # Build with metadata
  python -m idi.devkit.builder \\
      --config configs/sample.json \\
      --out artifacts/my_agent \\
      --metadata version=1.0.0 author=alice
        """,
    )
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

    try:
        build_artifact(
            config_path=args.config,
            out_dir=args.out,
            install_inputs=args.install_inputs,
            metadata_pairs=args.meta,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print("✅ IDI artifact generated")
    print(f" - Streams dir: {args.out / 'streams'}")
    print(f" - Manifest:    {args.out / 'artifact_manifest.json'}")
    if args.install_inputs:
        print(f" - Installed to: {args.install_inputs}")


if __name__ == "__main__":
    main()
