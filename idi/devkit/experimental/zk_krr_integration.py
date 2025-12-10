"""Experimental helpers for integrating ZK/Tau invariants pack with
bundle verification results.

These utilities treat verification outcomes as high-level status flags
and use the STRIKE/IKL engine together with the `zk_tau_invariants`
knowledge pack to derive uniform `violates_constraint` facts for
bundles.

They are kept in the experimental namespace and do not modify the
existing zk modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

from idi.devkit.experimental.strike_krr import (
    KnowledgePack,
    build_kb_from_packs,
    build_zk_tau_invariants_pack,
    evaluate_with_krr,
)


BundleId = str


@dataclass(frozen=True)
class BundleStatusContext:
    """Summarized ZK/Tau status for a single bundle.

    Each *_ok flag should reflect the outcome of the corresponding
    invariant check in the zk pipeline. This context is intentionally
    small and can be constructed from richer verification objects.
    """

    bundle_id: BundleId
    commitment_ok: bool
    method_id_ok: bool
    journal_ok: bool
    tx_hash_ok: bool
    path_ok: bool


def _bundle_context_to_facts(ctx: BundleStatusContext) -> Dict[str, Set[Tuple[Any, ...]]]:
    """Translate BundleStatusContext into IKL facts for STRIKE.

    Predicates produced:
    - bundle(BundleId)
    - commitment_status(BundleId, "ok" | "fail")
    - method_id_status(BundleId, "ok" | "fail")
    - journal_status(BundleId, "ok" | "fail")
    - tx_hash_status(BundleId, "ok" | "fail")
    - path_status(BundleId, "ok" | "fail")
    """

    def status(ok: bool) -> str:
        return "ok" if ok else "fail"

    return {
        "bundle": {(ctx.bundle_id,)},
        "commitment_status": {(ctx.bundle_id, status(ctx.commitment_ok))},
        "method_id_status": {(ctx.bundle_id, status(ctx.method_id_ok))},
        "journal_status": {(ctx.bundle_id, status(ctx.journal_ok))},
        "tx_hash_status": {(ctx.bundle_id, status(ctx.tx_hash_ok))},
        "path_status": {(ctx.bundle_id, status(ctx.path_ok))},
    }


def evaluate_bundle_with_krr(
    ctx: BundleStatusContext,
    packs: Iterable[KnowledgePack] | None = None,
) -> Tuple[bool, List[str]]:
    """Evaluate a bundle's ZK/Tau status against KRR packs.

    Returns (allowed, reasons). Reasons are human-readable strings
    derived from any `violates_constraint` facts produced by STRIKE.
    """

    if packs is None:
        packs = (build_zk_tau_invariants_pack(),)

    kb = build_kb_from_packs(packs)
    facts = _bundle_context_to_facts(ctx)
    allowed, reasons = evaluate_with_krr(
        kb,
        base_facts=facts,
        params={},
        active_profiles=set(),
    )
    return allowed, reasons
