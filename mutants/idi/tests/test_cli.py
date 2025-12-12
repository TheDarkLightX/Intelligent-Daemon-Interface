import pytest

from idi import cli


def test_cli_build_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["build", "--help"])
    assert exc.value.code == 0


def test_cli_build_invokes_agent_pack(tmp_path, monkeypatch):
    called = {}

    def fake_build_agent_pack(config_path, out_dir, **kwargs):
        called["config_path"] = config_path
        called["out_dir"] = out_dir

    monkeypatch.setattr(cli, "build_agent_pack", fake_build_agent_pack)

    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")

    exit_code = cli.main(
        [
            "build",
            "--config",
            str(cfg),
            "--out",
            str(tmp_path / "out"),
            "--no-proof",
        ]
    )

    assert exit_code == 0
    assert called["config_path"] == cfg


def test_cli_verify_missing_pack(tmp_path):
    exit_code = cli.main(
        [
            "verify",
            "--agentpack",
            str(tmp_path / "missing"),
        ]
    )
    assert exit_code != 0


def test_cli_verify_happy_path(tmp_path, monkeypatch):
    # Set up minimal agentpack structure
    pack = tmp_path / "pack"
    proof_dir = pack / "proof"
    streams = pack / "streams"
    proof_dir.mkdir(parents=True)
    streams.mkdir(parents=True)
    (pack / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (proof_dir / "proof.bin").write_bytes(b"stub")
    (proof_dir / "receipt.json").write_text("{}", encoding="utf-8")

    def fake_verify_proof(bundle, use_risc0=False):
        return True

    monkeypatch.setattr(cli, "verify_proof", fake_verify_proof)

    exit_code = cli.main(
        [
            "verify",
            "--agentpack",
            str(pack),
        ]
    )
    assert exit_code == 0
