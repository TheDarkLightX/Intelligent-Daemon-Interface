"""Generic AgentPatch engine (experimental).

.. deprecated::
    This module is in `idi.devkit.experimental` and may change without
    notice. For stable usage, import from `idi.synth` instead:

        from idi.synth import (
            AgentPatch,
            AgentPatchMeta,
            load_agent_patch,
            save_agent_patch,
            validate_agent_patch,
            diff_agent_patches,
        )

    The experimental module will be deprecated once idi.synth is stable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import json


# ---------------------------------------------------------------------------
# Validation bounds (DoS protection)
# ---------------------------------------------------------------------------

MAX_STRING_LENGTH = 1024
MAX_ID_LENGTH = 256
MAX_TAGS_COUNT = 32
MAX_TAG_LENGTH = 64
MAX_PAYLOAD_KEYS = 128
MAX_SPEC_PARAMS_KEYS = 64
MAX_ZK_PROFILE_KEYS = 32


@dataclass(frozen=True)
class AgentPatchMeta:
    """Generic metadata for agent patches.

    This mirrors the conceptual AgentPatch schema in the design docs and is
    intended to be stable across agent families.
    """

    id: str
    name: str
    description: str
    version: str
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentPatch:
    """Generic agent patch configuration.

    The `payload` field carries the family-specific configuration (e.g. a
    QAgent-specific structure or a module-graph document). The remaining
    fields capture spec / proof wiring that is common across families.
    """

    meta: AgentPatchMeta
    agent_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    spec_backend: str = "tau"
    spec_params: Mapping[str, Any] = field(default_factory=dict)
    zk_profile: Mapping[str, Any] = field(default_factory=dict)


def _require_non_empty(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must be non-empty")


def _require_bounded_length(value: str, max_len: int, label: str) -> None:
    if len(value) > max_len:
        raise ValueError(f"{label} exceeds max length {max_len}")


def _require_bounded_count(items: Tuple | Mapping, max_count: int, label: str) -> None:
    if len(items) > max_count:
        raise ValueError(f"{label} exceeds max count {max_count}")


def validate_agent_patch(patch: AgentPatch) -> None:
    """Validate basic invariants for AgentPatch.

    Preconditions:
        - `patch` is a well-formed AgentPatch instance.
    Postconditions:
        - Raises ValueError if required fields are empty or exceed bounds.
    """

    # Required non-empty
    _require_non_empty(patch.meta.id, "meta.id")
    _require_non_empty(patch.meta.name, "meta.name")
    _require_non_empty(patch.meta.version, "meta.version")
    _require_non_empty(patch.agent_type, "agent_type")

    # Length bounds
    _require_bounded_length(patch.meta.id, MAX_ID_LENGTH, "meta.id")
    _require_bounded_length(patch.meta.name, MAX_STRING_LENGTH, "meta.name")
    _require_bounded_length(patch.meta.description, MAX_STRING_LENGTH, "meta.description")
    _require_bounded_length(patch.meta.version, MAX_ID_LENGTH, "meta.version")
    _require_bounded_length(patch.agent_type, MAX_ID_LENGTH, "agent_type")

    # Collection bounds
    _require_bounded_count(patch.meta.tags, MAX_TAGS_COUNT, "meta.tags")
    for tag in patch.meta.tags:
        _require_bounded_length(tag, MAX_TAG_LENGTH, "meta.tags element")

    _require_bounded_count(patch.payload, MAX_PAYLOAD_KEYS, "payload")
    _require_bounded_count(patch.spec_params, MAX_SPEC_PARAMS_KEYS, "spec_params")
    _require_bounded_count(patch.zk_profile, MAX_ZK_PROFILE_KEYS, "zk_profile")


def agent_patch_to_dict(patch: AgentPatch) -> Dict[str, Any]:
    validate_agent_patch(patch)
    return {
        "meta": {
            "id": patch.meta.id,
            "name": patch.meta.name,
            "description": patch.meta.description,
            "version": patch.meta.version,
            "tags": list(patch.meta.tags),
        },
        "agent_type": patch.agent_type,
        "payload": dict(patch.payload),
        "spec_backend": patch.spec_backend,
        "spec_params": dict(patch.spec_params),
        "zk_profile": dict(patch.zk_profile),
    }


def agent_patch_from_dict(data: Mapping[str, Any]) -> AgentPatch:
    """Construct AgentPatch from a dictionary.

    Raises:
        ValueError: If data is not a mapping or meta is malformed.
    """
    if not isinstance(data, Mapping):
        raise ValueError("AgentPatch must be built from a mapping")

    meta_raw = data.get("meta", {})
    if not isinstance(meta_raw, Mapping):
        raise ValueError("AgentPatch.meta must be a mapping")
    meta_data = meta_raw or {}

    tags_raw = meta_data.get("tags", ())
    if isinstance(tags_raw, (list, tuple)):
        tags = tuple(str(t) for t in tags_raw)
    else:
        tags = (str(tags_raw),) if tags_raw else ()

    meta = AgentPatchMeta(
        id=str(meta_data.get("id", "")),
        name=str(meta_data.get("name", "")),
        description=str(meta_data.get("description", "")),
        version=str(meta_data.get("version", "")),
        tags=tags,
    )

    patch = AgentPatch(
        meta=meta,
        agent_type=str(data.get("agent_type", "")),
        payload=dict(data.get("payload", {}) or {}),
        spec_backend=str(data.get("spec_backend", "tau")),
        spec_params=dict(data.get("spec_params", {}) or {}),
        zk_profile=dict(data.get("zk_profile", {}) or {}),
    )

    validate_agent_patch(patch)
    return patch


def load_agent_patch(path: Path) -> AgentPatch:
    """Load an AgentPatch from a JSON file.

    Raises:
        ValueError: If file does not contain a JSON object.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"AgentPatch file must contain a JSON object, got {type(raw).__name__}"
        )
    return agent_patch_from_dict(raw)


def save_agent_patch(patch: AgentPatch, path: Path) -> None:
    payload = agent_patch_to_dict(patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def diff_agent_patches(a: AgentPatch, b: AgentPatch) -> Dict[str, Any]:
    """Return a shallow semantic diff between two patches.

    The result is a mapping from field name to a tuple `(old, new)` for
    fields that differ. This is intentionally small and JSON-friendly.
    """

    left = agent_patch_to_dict(a)
    right = agent_patch_to_dict(b)

    diff: Dict[str, Any] = {}
    for key in ("meta", "agent_type", "payload", "spec_backend", "spec_params", "zk_profile"):
        if left.get(key) != right.get(key):
            diff[key] = (left.get(key), right.get(key))
    return diff
