from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.sape_q_patch import QPatchMeta
from idi.devkit.experimental.strike_krr import (
    KnowledgeBase,
    KnowledgePack,
    build_comms_base_pack,
    build_kb_from_packs,
    build_qagent_base_pack,
    build_risk_conservative_pack,
    build_zk_tau_invariants_pack,
    evaluate_with_krr,
)


ProfileSet = Set[str]


@dataclass(frozen=True)
class PlanPack:
    """Serializable template for a planned agent configuration.

    This is an optional artifact: a materialized plan graph + profiles
    and knowledge-pack references that can be reused as an initial
    condition for planning or deployment. The actual IKL/STRIKE packs
    must be supplied separately at load time and are always re-checked.
    """

    name: str
    version: str
    description: str
    module_ids: Tuple[str, ...]
    active_profiles: Tuple[str, ...]
    knowledge_pack_names: Tuple[str, ...]


def planpack_to_dict(plan: PlanPack) -> Dict[str, Any]:
    """Convert PlanPack to a JSON-serializable dictionary."""

    return {
        "name": plan.name,
        "version": plan.version,
        "description": plan.description,
        "module_ids": list(plan.module_ids),
        "active_profiles": list(plan.active_profiles),
        "knowledge_pack_names": list(plan.knowledge_pack_names),
    }


def planpack_from_dict(data: Dict[str, Any]) -> PlanPack:
    """Reconstruct PlanPack from a dictionary.

    The caller is responsible for validating identifiers and wiring
    module_ids / knowledge_pack_names to concrete implementations.
    """

    return PlanPack(
        name=str(data.get("name", "")),
        version=str(data.get("version", "")),
        description=str(data.get("description", "")),
        module_ids=tuple(data.get("module_ids", ()) or ()),
        active_profiles=tuple(data.get("active_profiles", ()) or ()),
        knowledge_pack_names=tuple(data.get("knowledge_pack_names", ()) or ()),
    )


def _canonical_packs_for_qagent() -> List[KnowledgePack]:
    """Return the default pack set for QAgent-style planning.

    This keeps the canonical list in one place for the Synth planner.
    """

    packs: List[KnowledgePack] = []
    packs.append(build_qagent_base_pack())
    packs.append(build_risk_conservative_pack())
    packs.append(build_comms_base_pack())
    packs.append(build_zk_tau_invariants_pack())
    return packs


def select_packs_for_qpatch(
    meta: QPatchMeta,
    requested_profiles: ProfileSet,
    extra_packs: Iterable[KnowledgePack] | None = None,
) -> List[KnowledgePack]:
    """Select knowledge packs for a QPatch based on tags and profiles.

    Packs are selected if any of the following holds:
    - Their domain_tags intersect the patch meta tags.
    - Their profile_tags intersect the requested profiles.
    - They carry the generic "qagent" domain tag.

    This keeps selection simple and deterministic while allowing pack
    authors to steer applicability via tags.
    """

    packs = list(_canonical_packs_for_qagent())
    if extra_packs is not None:
        packs.extend(extra_packs)

    selected: List[KnowledgePack] = []
    meta_tags = set(meta.tags)

    for pack in packs:
        domain_tags = set(pack.domain_tags)
        profile_tags = set(pack.profile_tags)
        has_qagent_tag = "qagent" in domain_tags
        domain_overlap = bool(domain_tags.intersection(meta_tags))
        profile_overlap = bool(profile_tags.intersection(requested_profiles))
        if not domain_overlap and not profile_overlap and not has_qagent_tag:
            continue
        selected.append(pack)

    return selected


def build_kb_for_qpatch(
    meta: QPatchMeta,
    requested_profiles: ProfileSet | None = None,
    extra_packs: Iterable[KnowledgePack] | None = None,
) -> KnowledgeBase:
    """Build a KnowledgeBase for a QPatch using tagged packs.

    This is the main entry point for the Synth planning layer: given
    patch metadata and requested profiles, choose applicable packs and
    merge them into a single KnowledgeBase for STRIKE.
    """

    profiles: ProfileSet = set() if requested_profiles is None else set(requested_profiles)
    packs = select_packs_for_qpatch(meta, profiles, extra_packs=extra_packs)
    return build_kb_from_packs(packs)


def summarize_kb(kb: KnowledgeBase) -> Dict[str, Any]:
    """Return a small summary of predicates and rule counts in a KB."""

    predicates: Set[str] = set(kb.global_facts.keys())
    for rule in kb.rules:
        head_pred, _ = rule.head
        predicates.add(head_pred)
        for body_atom in rule.body:
            body_pred, _ = body_atom
            predicates.add(body_pred)

    return {
        "rule_count": float(len(kb.rules)),
        "predicate_count": float(len(predicates)),
    }


@dataclass(frozen=True)
class PlanCandidate:
    """Opaque candidate plan used by KRR-guided search.

    The Synth is free to encode its internal graph / configuration in
    `payload`. STRIKE/IKL only interact with a candidate through the
    facts function supplied to the search helper.
    """

    id: str
    meta: AgentPatchMeta
    profiles: ProfileSet
    payload: Any


def krr_guided_beam_search(
    initial: PlanCandidate,
    kb: KnowledgeBase,
    *,
    successor_fn: Callable[[PlanCandidate], Iterable[PlanCandidate]],
    facts_fn: Callable[[PlanCandidate], Dict[str, Set[Tuple[Any, ...]]]],
    score_fn: Callable[[PlanCandidate], Tuple[float, ...]],
    params: Dict[str, Any] | None = None,
    beam_width: int = 8,
    max_depth: int = 4,
    active_profiles: ProfileSet | None = None,
    explanations: Dict[str, List[str]] | None = None,
    stats: Dict[str, float] | None = None,
) -> List[Tuple[PlanCandidate, Tuple[float, ...]]]:
    """Run a bounded, KRR-guided beam search over plan candidates.

    Preconditions:
        - `successor_fn` generates a finite set of children for any
          candidate.
        - `facts_fn` produces IKL-style facts for a candidate.
        - `score_fn` returns a metric vector where higher is better in
          all dimensions.

    Postconditions:
        - Returns a list of (candidate, score) pairs for candidates
          that are KRR-valid under the provided KnowledgeBase.
        - All returned candidates passed `evaluate_with_krr` with no
          `violates_constraint` facts.
    """

    if beam_width <= 0 or max_depth < 0:
        return []

    profiles = initial.profiles if active_profiles is None else set(active_profiles)
    env_params: Dict[str, Any] = {} if params is None else dict(params)

    frontier: List[PlanCandidate] = [initial]
    seen_ids: Set[str] = {initial.id}
    accepted: List[Tuple[PlanCandidate, Tuple[float, ...]]] = []

    visited_count = 0
    pruned_count = 0
    max_frontier_size = float(len(frontier))
    depth_levels = 0

    for _depth in range(max_depth + 1):
        if not frontier:
            break

        next_frontier_scored: List[Tuple[PlanCandidate, Tuple[float, ...]]] = []

        for cand in frontier:
            visited_count += 1
            facts = facts_fn(cand)
            allowed, reasons = evaluate_with_krr(
                kb,
                base_facts=facts,
                params=env_params,
                active_profiles=profiles,
            )
            if not allowed:
                if explanations is not None:
                    explanations[cand.id] = list(reasons)
                pruned_count += 1
                continue

            score = score_fn(cand)
            accepted.append((cand, score))

            for child in successor_fn(cand):
                if child.id in seen_ids:
                    continue
                seen_ids.add(child.id)
                child_score = score_fn(child)
                next_frontier_scored.append((child, child_score))

        if not next_frontier_scored:
            break

        # FIX: Add tie-breaker by candidate ID to ensure deterministic sort
        # and avoid TypeError when comparing dataclass instances
        next_frontier_scored.sort(
            key=lambda item: (item[1], item[0].id),
            reverse=True,
        )
        frontier = [cand for cand, _score in next_frontier_scored[:beam_width]]
        if len(frontier) > max_frontier_size:
            max_frontier_size = float(len(frontier))
        depth_levels += 1

    if stats is not None:
        stats["visited"] = float(visited_count)
        stats["pruned_krr"] = float(pruned_count)
        stats["accepted"] = float(len(accepted))
        stats["frontier_max"] = max_frontier_size
        stats["depth_levels"] = float(depth_levels)

    return accepted


def demo_qagent_plan_search() -> List[Tuple[PlanCandidate, Tuple[float, ...]]]:
    """Run a tiny in-memory demo of KRR-guided planning for QAgents.

    This function is intended for developers. It builds a KnowledgeBase
    for a QAgent-style meta, defines a very small neighborhood of
    PlanCandidate objects whose payload encodes only two parameters
    (learning_rate, epsilon_start), and runs `krr_guided_beam_search` to
    show how KRR prunes unsafe configurations.
    """

    @dataclass(frozen=True)
    class Payload:
        learning_rate: float
        epsilon_start: float

    qmeta = QPatchMeta(
        name="demo-qagent-plan",
        description="Demo QAgent plan for KRR-guided search",
        version="0.0.1",
        tags=("qagent", "experimental"),
    )
    profiles: ProfileSet = {"conservative"}
    kb = build_kb_for_qpatch(qmeta, profiles)

    meta = AgentPatchMeta(
        id="demo-qagent-plan",
        name=qmeta.name,
        description=qmeta.description,
        version=qmeta.version,
        tags=qmeta.tags,
    )

    initial_payload = Payload(learning_rate=0.1, epsilon_start=0.3)
    initial = PlanCandidate(
        id="root",
        meta=meta,
        profiles=profiles,
        payload=initial_payload,
    )

    def successor_fn(plan: PlanCandidate) -> Iterable[PlanCandidate]:
        base: Payload = plan.payload
        children: List[PlanCandidate] = []

        lr_candidates = {base.learning_rate * 0.5, base.learning_rate * 1.5}
        eps_candidates = {
            max(0.0, base.epsilon_start - 0.2),
            min(1.0, base.epsilon_start + 0.2),
        }

        for idx, lr in enumerate(sorted(lr_candidates)):
            payload = Payload(learning_rate=lr, epsilon_start=base.epsilon_start)
            child_id = f"{plan.id}-lr{idx}"
            children.append(
                PlanCandidate(
                    id=child_id,
                    meta=plan.meta,
                    profiles=plan.profiles,
                    payload=payload,
                )
            )

        for idx, eps in enumerate(sorted(eps_candidates)):
            payload = Payload(learning_rate=base.learning_rate, epsilon_start=eps)
            child_id = f"{plan.id}-eps{idx}"
            children.append(
                PlanCandidate(
                    id=child_id,
                    meta=plan.meta,
                    profiles=plan.profiles,
                    payload=payload,
                )
            )

        return children

    def facts_fn(plan: PlanCandidate) -> Dict[str, Set[Tuple[Any, ...]]]:
        payload: Payload = plan.payload
        patch_id = plan.id
        return {
            "patch": {(patch_id,)},
            "param_value": {
                (patch_id, "learning_rate", payload.learning_rate),
                (patch_id, "epsilon_start", payload.epsilon_start),
            },
        }

    def score_fn(plan: PlanCandidate) -> Tuple[float, ...]:
        payload: Payload = plan.payload
        # Prefer lower learning_rate and epsilon_start; keep in [0,1].
        lr_score = 1.0 - float(payload.learning_rate)
        eps_score = 1.0 - float(payload.epsilon_start)
        return (lr_score, eps_score)

    return krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn,
        facts_fn=facts_fn,
        score_fn=score_fn,
        params={},
        beam_width=4,
        max_depth=2,
        active_profiles=profiles,
    )
