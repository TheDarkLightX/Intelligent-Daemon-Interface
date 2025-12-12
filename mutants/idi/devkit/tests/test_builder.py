from pathlib import Path
import json

from idi.devkit.builder import build_artifact


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "episodes": 2,
                "episode_length": 4,
                "quantizer": {"price_buckets": 2, "volume_buckets": 2, "trend_buckets": 2, "scarcity_buckets": 2, "mood_buckets": 2},
                "rewards": {"pnl": 0.5, "scarcity_alignment": 0.5, "ethics_bonus": 0.5, "communication_clarity": 0.1},
            }
        ),
        encoding="utf-8",
    )


def test_build_artifact_creates_manifest_and_streams(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    out_dir = tmp_path / "artifact"
    install_dir = tmp_path / "spec_inputs"

    manifest_path = build_artifact(
        config_path=config_path,
        out_dir=out_dir,
        install_inputs=install_dir,
        metadata_pairs=["layer=test"],
    )

    assert manifest_path.exists()
    streams_dir = out_dir / "streams"
    assert any(streams_dir.glob("*.in"))
    assert any(install_dir.glob("*.in"))

