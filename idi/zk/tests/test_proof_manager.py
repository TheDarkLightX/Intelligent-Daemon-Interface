from pathlib import Path
import json

from idi.devkit.manifest import build_manifest, write_manifest
from idi.zk.proof_manager import (
    generate_proof,
    verify_proof,
    compute_artifact_digest,
    compute_manifest_streams_digest,
)


def test_generate_and_verify_proof(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n", encoding="utf-8")

    manifest = build_manifest(config_path=cfg, stream_dir=streams, metadata={})
    manifest_path = tmp_path / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    proofs_dir = tmp_path / "proofs"
    bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams,
        out_dir=proofs_dir,
        prover_command=None,
        auto_detect_risc0=False,
    )

    assert bundle.proof_path.exists()
    assert verify_proof(bundle)


def test_compute_manifest_streams_digest_matches_artifact_digest(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n", encoding="utf-8")

    manifest = build_manifest(config_path=cfg, stream_dir=streams, metadata={})
    manifest_path = tmp_path / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    digest_a = compute_artifact_digest(manifest_path, streams)
    digest_b = compute_manifest_streams_digest(manifest_path, streams)

    assert digest_a == digest_b
