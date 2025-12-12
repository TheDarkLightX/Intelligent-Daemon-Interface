"""Artifact manifest helpers for IDI lookup tables."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class StreamInfo:
    """Metadata for a single `.in` stream."""

    name: str
    sha256: str
    length: int


@dataclass
class ArtifactManifest:
    """Serializable description of a lookup-table artifact."""

    schema_version: str
    generated_at: float
    config_path: str
    config_sha256: str
    streams: List[StreamInfo]
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_manifest(
    *,
    config_path: Path,
    stream_dir: Path,
    metadata: Dict[str, str] | None = None,
    schema_version: str = "idi-manifest-v1",
) -> ArtifactManifest:
    """Create a manifest from the given config and stream directory."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing: {config_path}")
    if not stream_dir.exists():
        raise FileNotFoundError(f"Stream directory missing: {stream_dir}")

    config_sha = _sha256_file(config_path)
    stream_infos: List[StreamInfo] = []
    for stream_file in sorted(stream_dir.glob("*.in")):
        stream_infos.append(
            StreamInfo(
                name=stream_file.name,
                sha256=_sha256_file(stream_file),
                length=sum(1 for _ in stream_file.read_text().splitlines() if _),
            )
        )

    if not stream_infos:
        raise ValueError(f"No .in streams found under {stream_dir}")

    return ArtifactManifest(
        schema_version=schema_version,
        generated_at=time.time(),
        config_path=str(config_path.resolve()),
        config_sha256=config_sha,
        streams=stream_infos,
        metadata=metadata or {},
    )


def write_manifest(manifest: ArtifactManifest, target_path: Path) -> None:
    """Persist a manifest to disk."""

    target_path.write_text(manifest.to_json(), encoding="utf-8")

