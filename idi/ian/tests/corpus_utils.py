from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CorpusWriteResult:
    path: str
    bytes_written: int


def _require_relative_filename(filename: str) -> None:
    if not filename:
        raise ValueError("filename must be non-empty")
    if os.path.isabs(filename):
        raise ValueError("filename must be relative")
    if ".." in Path(filename).parts:
        raise ValueError("filename must not contain '..'")


def write_json_corpus_case(
    *,
    corpus_dir: str,
    filename: str,
    payload: Mapping[str, Any],
    max_bytes: int = 256_000,
) -> CorpusWriteResult:
    """Write a JSON corpus case.

    Preconditions:
    - corpus_dir exists
    - filename is relative and safe
    - payload is JSON-serializable

    Postconditions:
    - File is created/overwritten atomically
    - File size <= max_bytes
    """
    _require_relative_filename(filename)

    base = Path(corpus_dir)
    if not base.exists() or not base.is_dir():
        raise ValueError("corpus_dir must exist and be a directory")

    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError("payload exceeds max_bytes")

    target = base / filename
    tmp = target.with_suffix(target.suffix + ".tmp")

    tmp.write_bytes(data)
    tmp.replace(target)

    return CorpusWriteResult(path=str(target), bytes_written=len(data))


def read_json_corpus_case(path: str) -> dict[str, Any]:
    """Read a JSON corpus case."""
    data = Path(path).read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("corpus case must be a JSON object")
    return parsed
