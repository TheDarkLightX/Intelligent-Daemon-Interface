"""Friendly CLI for IDI workflows (build / verify agent packs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from idi.pipeline.agent_pack import build_agent_pack
from idi.zk.proof_manager import ProofBundle, verify_proof

SUCCESS = "✅"
STEP = "🚀"
WARN = "⚠️"
ERROR = "❌"


def _poke_yoke_path(path: Path, must_exist: bool = True) -> None:
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{ERROR} Path not found: {path}")


def _print_header(title: str) -> None:
    print(f"{STEP} {title}")


def _cmd_build(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    out_dir = Path(args.out)
    try:
        _poke_yoke_path(config_path, must_exist=True)
        _print_header("Building agent pack")
        build_agent_pack(
            config_path=config_path,
            out_dir=out_dir,
            spec_kind=args.spec_kind,
            proof_enabled=not args.no_proof,
            prover_cmd=args.prover_cmd,
            tau_bin=None,
            tx_hash=args.tx_hash,
        )
        print(f"{SUCCESS} Agent pack created at {out_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Build failed: {exc}")
        return 1


def _load_bundle(pack_dir: Path) -> ProofBundle:
    proof_dir = pack_dir / "proof"
    manifest_path = pack_dir / "artifact_manifest.json"
    return ProofBundle(
        manifest_path=manifest_path,
        proof_path=proof_dir / "proof.bin",
        receipt_path=proof_dir / "receipt.json",
        stream_dir=pack_dir / "streams",
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    pack_dir = Path(args.agentpack)
    try:
        _poke_yoke_path(pack_dir, must_exist=True)
        bundle = _load_bundle(pack_dir)
        for required in [bundle.manifest_path, bundle.proof_path, bundle.receipt_path, bundle.stream_dir]:
            _poke_yoke_path(required, must_exist=True)

        _print_header(f"Verifying agent pack at {pack_dir}")
        ok = verify_proof(bundle, use_risc0=args.risc0)
        if not ok:
            print(f"{ERROR} Verification failed")
            return 2
        print(f"{SUCCESS} Verification succeeded")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Verify failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IDI agent pack CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Train, export, prove, and bundle an agent pack")
    p_build.add_argument("--config", required=True, help="Path to training config JSON")
    p_build.add_argument("--out", required=True, help="Output directory for agent pack")
    p_build.add_argument("--spec-kind", default="v38", help="Spec type to generate")
    p_build.add_argument("--prover-cmd", help="Optional prover command for proofs")
    p_build.add_argument("--no-proof", action="store_true", help="Skip proof generation")
    p_build.add_argument("--tx-hash", help="Optional transaction hash binding")
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="Verify an existing agent pack (proof + receipt)")
    p_verify.add_argument("--agentpack", required=True, help="Path to built agent pack directory")
    p_verify.add_argument("--risc0", action="store_true", help="Use Risc0 verification path")
    p_verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
