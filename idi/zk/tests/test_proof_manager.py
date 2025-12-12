from pathlib import Path
import json

from idi.devkit.manifest import build_manifest, write_manifest
from idi.zk.proof_manager import (
    generate_proof,
    verify_proof,
    compute_artifact_digest,
    compute_manifest_streams_digest,
)
from idi.zk.verification import VerificationErrorCode


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


def test_policy_root_binding_in_receipt(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n", encoding="utf-8")

    manifest = build_manifest(config_path=cfg, stream_dir=streams, metadata={})
    manifest_path = tmp_path / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    proofs_dir = tmp_path / "proofs"
    policy_root = b"\x01" * 32
    bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams,
        out_dir=proofs_dir,
        prover_command=None,
        auto_detect_risc0=False,
        policy_root=policy_root,
    )

    assert verify_proof(bundle, extra_bindings=None)

    # Tamper policy_root in receipt; verification should fail
    receipt = json.loads(bundle.receipt_path.read_text())
    receipt["policy_root"] = "00" * 32
    bundle.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert verify_proof(bundle, extra_bindings=None) is False


def test_config_spec_binding_in_receipt(tmp_path: Path) -> None:
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
        config_fingerprint="cfg123",
        spec_hash="spec123",
    )

    assert verify_proof(bundle, extra_bindings=None)

    # Tamper config_fingerprint
    receipt = json.loads(bundle.receipt_path.read_text())
    receipt["config_fingerprint"] = "other"
    bundle.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_proof(bundle, extra_bindings=None) is False


def _create_dummy_risc0_env(tmp_path: Path):
    import idi.zk.proof_manager as proof_manager

    current_file = Path(proof_manager.__file__)
    host_path = current_file.parent / "risc0" / "target" / "release" / "idi_risc0_host"
    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_bytes(b"binary")

    proof = tmp_path / "proof.bin"
    proof.write_bytes(b"proof")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "input.in").write_text("1\n", encoding="utf-8")

    return proof, manifest, streams


def test_verify_zk_receipt_method_id_mismatch(monkeypatch, tmp_path: Path) -> None:
    from idi.zk.proof_manager import verify_zk_receipt
    import subprocess

    proof, manifest, streams = _create_dummy_risc0_env(tmp_path)

    def fake_run(cmd, capture_output, timeout, check):
        class Result:
            def __init__(self) -> None:
                self.returncode = 1
                self.stderr = b"Method ID mismatch: expected abc"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = verify_zk_receipt(
        proof_path=proof,
        manifest_path=manifest,
        stream_dir=streams,
        expected_method_id="abc",
    )

    assert report.success is False
    assert report.error_code == VerificationErrorCode.METHOD_ID_MISMATCH
    assert report.details["expected"] == "abc"


def test_verify_zk_receipt_journal_digest_mismatch(monkeypatch, tmp_path: Path) -> None:
    from idi.zk.proof_manager import verify_zk_receipt
    import subprocess

    proof, manifest, streams = _create_dummy_risc0_env(tmp_path)

    def fake_run(cmd, capture_output, timeout, check):
        class Result:
            def __init__(self) -> None:
                self.returncode = 1
                self.stderr = b"Journal digest mismatch: expected digest"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = verify_zk_receipt(
        proof_path=proof,
        manifest_path=manifest,
        stream_dir=streams,
        expected_method_id="abc",
        expected_journal_digest="deadbeef",
    )

    assert report.success is False
    assert report.error_code == VerificationErrorCode.JOURNAL_DIGEST_MISMATCH
    assert report.details["expected"] == "deadbeef"


def test_verify_zk_receipt_commitment_mismatch(monkeypatch, tmp_path: Path) -> None:
    from idi.zk.proof_manager import verify_zk_receipt
    import subprocess

    proof, manifest, streams = _create_dummy_risc0_env(tmp_path)

    def fake_run(cmd, capture_output, timeout, check):
        class Result:
            def __init__(self) -> None:
                self.returncode = 1
                self.stderr = b"guest digest does not match commitment"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = verify_zk_receipt(
        proof_path=proof,
        manifest_path=manifest,
        stream_dir=streams,
        expected_method_id="abc",
    )

    assert report.success is False
    assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH


def test_verify_zk_receipt_generic_failure(monkeypatch, tmp_path: Path) -> None:
    from idi.zk.proof_manager import verify_zk_receipt
    import subprocess

    proof, manifest, streams = _create_dummy_risc0_env(tmp_path)

    def fake_run(cmd, capture_output, timeout, check):
        class Result:
            def __init__(self) -> None:
                self.returncode = 3
                self.stderr = b"other error"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = verify_zk_receipt(
        proof_path=proof,
        manifest_path=manifest,
        stream_dir=streams,
        expected_method_id="abc",
    )

    assert report.success is False
    assert report.error_code == VerificationErrorCode.ZK_RECEIPT_INVALID
    assert report.details["returncode"] == 3
