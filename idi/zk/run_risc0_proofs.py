"""Generate real proofs using the integrated Risc0 prover."""

from __future__ import annotations

import argparse
from pathlib import Path

from idi.zk.proof_manager import generate_proof

RISC0_CMD = (
    "cargo run --release -p idi_risc0_host -- "
    "--manifest {manifest} --streams {streams} --proof {proof} --receipt {receipt}"
)


def find_artifacts(root: Path) -> list[Path]:
    return list(root.rglob("artifact_manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Risc0 proofs for IDI artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("idi/artifacts"),
        help="Root directory containing artifact manifests.",
    )
    args = parser.parse_args()

    manifests = find_artifacts(args.artifacts_root)
    if not manifests:
        print("No artifact manifests found.")
        return

    for manifest in manifests:
        stream_dir = manifest.parent / "streams"
        out_dir = manifest.parent / "proof_risc0"
        print(f"▶ Proving {manifest}")
        generate_proof(
            manifest_path=manifest,
            stream_dir=stream_dir,
            out_dir=out_dir,
            prover_command=RISC0_CMD,
        )
        print(f"  ✅ receipt: {out_dir / 'receipt.json'}")


if __name__ == "__main__":
    main()

