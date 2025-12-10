"""Experimental helpers for integrating CommsBasePack with the
communication / emoji engine.

These utilities take a simple view of communication context and use the
STRIKE/IKL engine together with the `comms_base` knowledge pack to
validate or veto proposed communication actions.

They are intentionally lightweight and do not modify idi_iann training
code directly; instead they provide a reusable pattern that can be
called from trainers, daemons, or UI layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

from idi.devkit.experimental.strike_krr import (
    KnowledgePack,
    build_comms_base_pack,
    build_kb_from_packs,
    evaluate_with_krr,
)

SubjectId = str


@dataclass(frozen=True)
class CommContext:
    """Minimal context for communication KRR checks.

    subject_id: identifier for the agent or user context.
    risk_level: summarized risk state (e.g. "low", "medium", "high").
    sensitivity: user sensitivity (e.g. "low", "high").
    action: proposed communication action (e.g. "silent", "alert").
    """

    subject_id: SubjectId
    risk_level: str
    sensitivity: str
    action: str


def _comm_context_to_facts(ctx: CommContext) -> Dict[str, Set[Tuple[Any, ...]]]:
    """Translate CommContext into IKL facts for STRIKE.

    Predicates produced:
    - risk_state(Subject, Level)
    - user_sensitivity(Subject, Level)
    - comm_action(Subject, ActionId)
    """

    return {
        "risk_state": {(ctx.subject_id, ctx.risk_level)},
        "user_sensitivity": {(ctx.subject_id, ctx.sensitivity)},
        "comm_action": {(ctx.subject_id, ctx.action)},
    }


def evaluate_comm_action_with_krr(
    ctx: CommContext,
    packs: Iterable[KnowledgePack] | None = None,
) -> Tuple[bool, List[str]]:
    """Evaluate a proposed communication action against KRR packs.

    Returns (allowed, reasons). Reasons are human-readable strings
    derived from any `violates_constraint` facts produced by STRIKE.
    """

    if packs is None:
        packs = (build_comms_base_pack(),)

    kb = build_kb_from_packs(packs)
    facts = _comm_context_to_facts(ctx)
    allowed, reasons = evaluate_with_krr(
        kb,
        base_facts=facts,
        params={},
        active_profiles=set(),
    )
    return allowed, reasons
