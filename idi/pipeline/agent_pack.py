"""Agent pack builder: train -> export -> spec -> prove/verify -> bundle metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from idi.training.python.idi_iann import QTrainer
from idi.training.python.idi_iann.config import TrainingConfig
from idi.devkit.manifest import build_manifest, write_manifest
from idi.zk.spec_generator import TauSpecGenerator
from idi.zk import proof_manager


@dataclass
class BuildReport:
    config_fingerprint: str
    spec_hash: str
    tx_hash: Optional[str]
    streams_dir: Path
    manifest_path: Path
    spec_path: Path
    proof_dir: Optional[Path]
    proof_verified: bool


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_agent_pack(
    config_path: Path,
    out_dir: Path,
    *,
    spec_kind: str = "v38",
    proof_enabled: bool = True,
    prover_cmd: str | None = None,
    tau_bin: Path | None = None,  # reserved for future sanity-run hook
    tx_hash: str | None = None,
) -> BuildReport:
    """Orchestrate training, export, spec generation, and proof/verify."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config and derive fingerprint
    cfg = TrainingConfig.from_json(config_path)
    cfg_fp = cfg.fingerprint()

    # Persist canonical config copy inside pack
    config_copy = out_dir / "config.json"
    cfg.to_json(config_copy)

    # Train
    trainer = QTrainer(cfg)
    _policy, trace = trainer.run()

    # Export streams
    streams_dir = out_dir / "streams"
    trace.export(streams_dir, contract=spec_kind)

    # Build manifest
    manifest = build_manifest(
        config_path=config_copy,
        stream_dir=streams_dir,
        metadata={"config_fingerprint": cfg_fp, "spec_kind": spec_kind},
    )
    manifest_path = out_dir / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    # Generate spec
    spec_path = out_dir / "spec.tau"
    generator = TauSpecGenerator(cfg)
    if spec_kind == "v38":
        generator.generate_v38_spec(spec_path, contract_name=spec_kind)
    else:
        generator.generate_layered_spec(spec_path)
    spec_hash = _sha256_file(spec_path)

    proof_dir: Optional[Path] = None
    proof_verified = False

    extras = {}
    extras_bytes = {}
    if cfg_fp:
        extras["config_fingerprint"] = cfg_fp
    if spec_hash:
        extras["spec_hash"] = spec_hash
    if tx_hash:
        extras["tx_hash"] = tx_hash
    extras_bytes = {k: v.encode() for k, v in extras.items()}

    if proof_enabled:
        proof_dir = out_dir / "proof"
        bundle = proof_manager.generate_proof(
            manifest_path=manifest_path,
            stream_dir=streams_dir,
            out_dir=proof_dir,
            prover_command=prover_cmd,
            auto_detect_risc0=True,
            tx_hash=tx_hash,
            config_fingerprint=cfg_fp,
            spec_hash=spec_hash,
            extra_bindings=extras_bytes or None,
        )
        proof_verified = proof_manager.verify_proof(bundle, use_risc0=False, extra_bindings=extras_bytes or None)

    # Save build report
    report_path = out_dir / "build_report.json"
    report_payload = {
        "config_fingerprint": cfg_fp,
        "spec_hash": spec_hash,
        "tx_hash": tx_hash,
        "streams_dir": str(streams_dir),
        "manifest_path": str(manifest_path),
        "spec_path": str(spec_path),
        "proof_dir": str(proof_dir) if proof_dir else None,
        "proof_verified": proof_verified,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return BuildReport(
        config_fingerprint=cfg_fp,
        spec_hash=spec_hash,
        tx_hash=tx_hash,
        streams_dir=streams_dir,
        manifest_path=manifest_path,
        spec_path=spec_path,
        proof_dir=proof_dir,
        proof_verified=proof_verified,
    )
