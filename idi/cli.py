"""Lightweight CLI for IDI workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from idi.pipeline.agent_pack import build_agent_pack
from idi.zk.proof_manager import ProofBundle, verify_proof


def _cmd_build(args: argparse.Namespace) -> int:
    build_agent_pack(
        config_path=Path(args.config),
        out_dir=Path(args.out),
        spec_kind=args.spec_kind,
        proof_enabled=not args.no_proof,
        prover_cmd=args.prover_cmd,
        tau_bin=None,
        tx_hash=args.tx_hash,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    pack_dir = Path(args.agentpack)
    proof_dir = pack_dir / "proof"
    manifest_path = pack_dir / "artifact_manifest.json"
    bundle = ProofBundle(
        manifest_path=manifest_path,
        proof_path=proof_dir / "proof.bin",
        receipt_path=proof_dir / "receipt.json",
        stream_dir=pack_dir / "streams",
    )
    ok = verify_proof(bundle, use_risc0=False)
    if not ok:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IDI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build an agent pack")
    p_build.add_argument("--config", required=True, help="Path to training config JSON")
    p_build.add_argument("--out", required=True, help="Output directory for agent pack")
    p_build.add_argument("--spec-kind", default="v38", help="Spec type (default: v38)")
    p_build.add_argument("--prover-cmd", help="Optional prover command for proofs")
    p_build.add_argument("--no-proof", action="store_true", help="Skip proof generation")
    p_build.add_argument("--tx-hash", help="Optional transaction hash binding")
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="Verify an existing agent pack")
    p_verify.add_argument("--agentpack", required=True, help="Path to built agent pack directory")
    p_verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
