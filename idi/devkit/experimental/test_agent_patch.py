from __future__ import annotations

from pathlib import Path
from typing import Dict

from idi.devkit.experimental.agent_patch import (
    AgentPatch,
    AgentPatchMeta,
    agent_patch_from_dict,
    agent_patch_to_dict,
    diff_agent_patches,
    load_agent_patch,
    save_agent_patch,
)
from idi.devkit.experimental.qagent_agentpatch_adapter import (
    AGENT_TYPE_QTABLE,
    agent_patch_to_qagent_patch,
    qagent_patch_to_agent_patch,
)
from idi.devkit.experimental.sape_q_patch import QAgentPatch, QPatchMeta, validate_patch_fast


def _make_agent_patch() -> AgentPatch:
    meta = AgentPatchMeta(
        id="p1",
        name="patch-one",
        description="test patch",
        version="0.0.1",
        tags=("test",),
    )
    return AgentPatch(
        meta=meta,
        agent_type="q_table",
        payload={"alpha": 1, "beta": 2},
        spec_backend="tau",
        spec_params={"max_drawdown": 0.2},
        zk_profile={"mode": "stub"},
    )


def test_agent_patch_dict_roundtrip() -> None:
    patch = _make_agent_patch()
    as_dict = agent_patch_to_dict(patch)
    restored = agent_patch_from_dict(as_dict)
    assert restored == patch


def test_agent_patch_load_save_roundtrip(tmp_path: Path) -> None:
    patch = _make_agent_patch()
    path = tmp_path / "patch.json"
    save_agent_patch(patch, path)
    loaded = load_agent_patch(path)
    assert loaded == patch


def test_diff_agent_patches_reports_changes() -> None:
    a = _make_agent_patch()
    b = AgentPatch(
        meta=a.meta,
        agent_type=a.agent_type,
        payload={"alpha": 2, "beta": 2},
        spec_backend=a.spec_backend,
        spec_params=a.spec_params,
        zk_profile=a.zk_profile,
    )
    diff = diff_agent_patches(a, b)
    assert "payload" in diff
    old, new = diff["payload"]
    assert old != new


def _make_qagent_patch() -> QAgentPatch:
    meta = QPatchMeta(
        name="qpatch",
        description="qagent patch",
        version="0.0.1",
        tags=("qagent", "test"),
    )
    patch = QAgentPatch(
        identifier="q1",
        num_price_bins=4,
        num_inventory_bins=4,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        meta=meta,
    )
    assert validate_patch_fast(patch)
    return patch


def test_qagent_adapter_roundtrip() -> None:
    qpatch = _make_qagent_patch()
    agent_patch = qagent_patch_to_agent_patch(qpatch)
    assert agent_patch.agent_type == AGENT_TYPE_QTABLE
    restored = agent_patch_to_qagent_patch(agent_patch)
    assert restored == qpatch


def test_qagent_adapter_payload_keys_present() -> None:
    qpatch = _make_qagent_patch()
    agent_patch = qagent_patch_to_agent_patch(qpatch)
    payload: Dict[str, object] = dict(agent_patch.payload)
    for key in (
        "identifier",
        "num_price_bins",
        "num_inventory_bins",
        "learning_rate",
        "discount_factor",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
    ):
        assert key in payload
