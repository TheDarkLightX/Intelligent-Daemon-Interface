"""Generate stub proof bundles for every artifact manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from idi.zk.proof_manager import generate_proof, verify_proof


def find_manifests(root: Path) -> List[Path]:
    return list(root.rglob("artifact_manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stub proofs for all manifests.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("idi/artifacts"),
        help="Root directory containing artifact outputs.",
    )
    args = parser.parse_args()

    manifests = find_manifests(args.artifacts_root)
    if not manifests:
        print("No artifact_manifest.json files found.")
        return

    for manifest in manifests:
        stream_dir = manifest.parent / "streams"
        proof_dir = manifest.parent / "proof_stub"
        print(f"▶ Generating stub proof for {manifest}")
        bundle = generate_proof(
            manifest_path=manifest,
            stream_dir=stream_dir,
            out_dir=proof_dir,
            auto_detect_risc0=False,  # Explicitly use stub for this script
        )
        verified = verify_proof(bundle)
        status = "✅" if verified else "❌"
        print(f"  {status} receipt: {bundle.receipt_path}")


if __name__ == "__main__":
    main()

