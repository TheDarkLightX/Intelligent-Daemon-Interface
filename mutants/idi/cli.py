"""Friendly CLI for IDI workflows (build / verify agent packs, ZK bundles)."""

from __future__ import annotations

import argparse
import json
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


def _cmd_bundle_pack(args: argparse.Namespace) -> int:
    """Pack a local proof into a network-portable wire bundle."""
    from idi.zk.wire import ZkProofBundleLocal
    
    proof_dir = Path(args.proof_dir)
    out_path = Path(args.out)
    
    try:
        _poke_yoke_path(proof_dir, must_exist=True)
        
        # Locate required files
        proof_path = proof_dir / "proof.bin"
        attestation_path = proof_dir / "receipt.json"
        manifest_path = proof_dir.parent / "artifact_manifest.json"
        if not manifest_path.exists():
            manifest_path = proof_dir / "manifest.json"
        stream_dir = proof_dir.parent / "streams"
        if not stream_dir.exists():
            stream_dir = proof_dir / "streams"
        
        for p in [proof_path, attestation_path, manifest_path]:
            _poke_yoke_path(p, must_exist=True)
        
        _print_header(f"Packing wire bundle from {proof_dir}")
        
        local = ZkProofBundleLocal(
            proof_path=proof_path,
            attestation_path=attestation_path,
            manifest_path=manifest_path,
            stream_dir=stream_dir if stream_dir.exists() else proof_dir,
            tx_hash=args.tx_hash,
        )
        
        wire = local.to_wire(include_streams=not args.no_streams)
        
        # Write wire bundle
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wire.serialize())
        
        print(f"{SUCCESS} Wire bundle created: {out_path}")
        print(f"    Schema: {wire.schema_version}")
        print(f"    Proof system: {wire.proof_system}")
        print(f"    Streams included: {wire.streams_pack_b64 is not None}")
        return 0
        
    except Exception as exc:
        print(f"{ERROR} Pack failed: {exc}")
        return 1


def _cmd_bundle_verify(args: argparse.Namespace) -> int:
    """Verify a wire bundle."""
    from idi.zk.wire import ZkProofBundleWireV1
    from idi.zk.bundle_verify import verify_proof_bundle_wire
    
    bundle_path = Path(args.bundle)
    
    try:
        _poke_yoke_path(bundle_path, must_exist=True)
        
        _print_header(f"Verifying wire bundle: {bundle_path}")
        
        # Load and deserialize
        data = bundle_path.read_bytes()
        wire = ZkProofBundleWireV1.deserialize(data)
        
        print(f"    Schema: {wire.schema_version}")
        print(f"    Proof system: {wire.proof_system}")
        
        # Verify
        report = verify_proof_bundle_wire(
            wire,
            expected_method_id=args.method_id,
            require_zk=args.require_zk,
        )
        
        if report.success:
            print(f"{SUCCESS} Verification succeeded")
            if report.details.get("digest"):
                print(f"    Commitment: {report.details['digest'][:16]}...")
            return 0
        else:
            print(f"{ERROR} Verification failed: {report.error_code.value}")
            print(f"    {report.message}")
            return 2
            
    except ValueError as exc:
        print(f"{ERROR} Invalid bundle: {exc}")
        return 1
    except Exception as exc:
        print(f"{ERROR} Verify failed: {exc}")
        return 1


def _cmd_bundle_info(args: argparse.Namespace) -> int:
    """Display information about a wire bundle."""
    from idi.zk.wire import ZkProofBundleWireV1
    
    bundle_path = Path(args.bundle)
    
    try:
        _poke_yoke_path(bundle_path, must_exist=True)
        
        data = bundle_path.read_bytes()
        wire = ZkProofBundleWireV1.deserialize(data)
        
        print(f"Wire Bundle: {bundle_path}")
        print(f"  Schema version: {wire.schema_version}")
        print(f"  Proof system: {wire.proof_system}")
        print(f"  Has streams: {wire.streams_pack_b64 is not None}")
        if wire.streams_sha256:
            print(f"  Streams SHA256: {wire.streams_sha256[:16]}...")
        if wire.tx_hash:
            print(f"  TX hash: {wire.tx_hash}")
        
        # Compute commitment
        commitment = wire.compute_commitment()
        print(f"  Commitment: {commitment[:32]}...")
        
        # Parse attestation
        try:
            attestation = wire.get_attestation()
            print(f"  Attestation keys: {list(attestation.keys())}")
        except Exception:
            print(f"  {WARN} Could not parse attestation")
        
        return 0
        
    except Exception as exc:
        print(f"{ERROR} Info failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IDI agent pack CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Agent pack commands
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

    # ZK bundle commands
    p_bundle = sub.add_parser("bundle", help="ZK wire bundle operations")
    bundle_sub = p_bundle.add_subparsers(dest="bundle_command", required=True)
    
    p_pack = bundle_sub.add_parser("pack", help="Pack proof directory into wire bundle")
    p_pack.add_argument("--proof-dir", required=True, help="Directory containing proof.bin and receipt.json")
    p_pack.add_argument("--out", required=True, help="Output wire bundle JSON file")
    p_pack.add_argument("--tx-hash", help="Optional transaction hash binding")
    p_pack.add_argument("--no-streams", action="store_true", help="Exclude streams from bundle")
    p_pack.set_defaults(func=_cmd_bundle_pack)
    
    p_bverify = bundle_sub.add_parser("verify", help="Verify a wire bundle")
    p_bverify.add_argument("--bundle", required=True, help="Path to wire bundle JSON")
    p_bverify.add_argument("--method-id", help="Expected method ID (hex) for Risc0")
    p_bverify.add_argument("--require-zk", action="store_true", help="Require ZK verification")
    p_bverify.set_defaults(func=_cmd_bundle_verify)
    
    p_info = bundle_sub.add_parser("info", help="Display wire bundle information")
    p_info.add_argument("--bundle", required=True, help="Path to wire bundle JSON")
    p_info.set_defaults(func=_cmd_bundle_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
