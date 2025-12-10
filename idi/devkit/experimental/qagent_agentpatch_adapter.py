from __future__ import annotations

from typing import Any, Mapping

from idi.devkit.experimental.agent_patch import AgentPatch, AgentPatchMeta
from idi.devkit.experimental.sape_q_patch import QAgentPatch, QPatchMeta, validate_patch_fast


AGENT_TYPE_QTABLE = "q_table"


def qagent_patch_to_agent_patch(patch: QAgentPatch) -> AgentPatch:
    if not validate_patch_fast(patch):
        raise ValueError("QAgentPatch is structurally invalid")

    tags = patch.meta.tags
    if "qagent" not in tags:
        tags = tags + ("qagent",)

    meta = AgentPatchMeta(
        id=patch.identifier,
        name=patch.meta.name,
        description=patch.meta.description,
        version=patch.meta.version,
        tags=tags,
    )

    payload: dict[str, Any] = {
        "identifier": patch.identifier,
        "num_price_bins": patch.num_price_bins,
        "num_inventory_bins": patch.num_inventory_bins,
        "learning_rate": patch.learning_rate,
        "discount_factor": patch.discount_factor,
        "epsilon_start": patch.epsilon_start,
        "epsilon_end": patch.epsilon_end,
        "epsilon_decay_steps": patch.epsilon_decay_steps,
    }

    return AgentPatch(
        meta=meta,
        agent_type=AGENT_TYPE_QTABLE,
        payload=payload,
    )


def agent_patch_to_qagent_patch(patch: AgentPatch) -> QAgentPatch:
    if patch.agent_type != AGENT_TYPE_QTABLE:
        raise ValueError(f"Unsupported agent_type for QAgent adapter: {patch.agent_type}")

    payload: Mapping[str, Any] = patch.payload

    identifier = str(payload.get("identifier", patch.meta.id))

    qmeta = QPatchMeta(
        name=patch.meta.name,
        description=patch.meta.description,
        version=patch.meta.version,
        tags=patch.meta.tags,
    )

    qpatch = QAgentPatch(
        identifier=identifier,
        num_price_bins=int(payload["num_price_bins"]),
        num_inventory_bins=int(payload["num_inventory_bins"]),
        learning_rate=float(payload["learning_rate"]),
        discount_factor=float(payload["discount_factor"]),
        epsilon_start=float(payload["epsilon_start"]),
        epsilon_end=float(payload["epsilon_end"]),
        epsilon_decay_steps=int(payload["epsilon_decay_steps"]),
        meta=qmeta,
    )

    if not validate_patch_fast(qpatch):
        raise ValueError("AgentPatch payload does not describe a valid QAgentPatch")

    return qpatch
