from pathlib import Path
import json

from idi.devkit.manifest import build_manifest


def test_build_manifest(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"episodes": 2}), encoding="utf-8")

    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n0\n", encoding="utf-8")
    (streams / "q_sell.in").write_text("0\n1\n", encoding="utf-8")

    manifest = build_manifest(config_path=cfg, stream_dir=streams, metadata={"owner": "test"})

    assert manifest.config_path == str(cfg.resolve())
    assert len(manifest.streams) == 2
    assert manifest.metadata["owner"] == "test"

