from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import json

from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.sape_q_patch import (
    QAgentPatch,
    QPatchMeta,
    qpatch_to_ikl_facts,
    mutate_patch,
    validate_patch_fast,
    evaluate_patch_stub,
)
from idi.devkit.experimental.synth_krr_planner import (
    PlanCandidate,
    ProfileSet,
    build_kb_for_qpatch,
    krr_guided_beam_search,
)
from idi.devkit.experimental.strike_krr import KnowledgeBase


EvaluatorFn = Callable[[QAgentPatch], Tuple[float, ...]]


MAX_BEAM_WIDTH = 32
MAX_SEARCH_DEPTH = 8


@dataclass(frozen=True)
class QAgentSynthConfig:
    beam_width: int = 4
    max_depth: int = 3


class QAgentSynthesizer:
    """Experimental QAgent-focused modular synthesizer.

    This class wraps the generic KRR planner for QAgentPatch objects,
    using STRIKE/IKL to prune unsafe patches and a pluggable evaluator
    to score candidates.
    """

    def __init__(
        self,
        base_patch: QAgentPatch,
        *,
        profiles: ProfileSet | None = None,
        evaluator: EvaluatorFn | None = None,
        kb: KnowledgeBase | None = None,
    ) -> None:
        if not validate_patch_fast(base_patch):
            raise ValueError("Base patch is structurally invalid")

        self._base_patch = base_patch
        self._profiles: ProfileSet = set() if profiles is None else set(profiles)
        self._evaluator: EvaluatorFn = evaluator or evaluate_patch_stub
        self._kb: KnowledgeBase = kb or build_kb_for_qpatch(
            base_patch.meta,
            self._profiles,
        )

    def _to_candidate(self, patch: QAgentPatch) -> PlanCandidate:
        meta = AgentPatchMeta(
            id=patch.identifier,
            name=patch.meta.name,
            description=patch.meta.description,
            version=patch.meta.version,
            tags=patch.meta.tags,
        )
        return PlanCandidate(
            id=patch.identifier,
            meta=meta,
            profiles=self._profiles,
            payload=patch,
        )

    def synthesize(self, config: QAgentSynthConfig | None = None) -> List[Tuple[QAgentPatch, Tuple[float, ...]]]:
        cfg = config or QAgentSynthConfig()
        beam_width = cfg.beam_width
        if beam_width < 1:
            beam_width = 1
        if beam_width > MAX_BEAM_WIDTH:
            beam_width = MAX_BEAM_WIDTH

        max_depth = cfg.max_depth
        if max_depth < 0:
            max_depth = 0
        if max_depth > MAX_SEARCH_DEPTH:
            max_depth = MAX_SEARCH_DEPTH
        initial = self._to_candidate(self._base_patch)
        score_cache: Dict[str, Tuple[float, ...]] = {}

        def successor_fn(plan: PlanCandidate) -> Iterable[PlanCandidate]:
            patch: QAgentPatch = plan.payload
            for neighbor in mutate_patch(patch):
                if not validate_patch_fast(neighbor):
                    continue
                yield self._to_candidate(neighbor)

        def facts_fn(plan: PlanCandidate) -> Dict[str, Set[Tuple[Any, ...]]]:
            patch: QAgentPatch = plan.payload
            return qpatch_to_ikl_facts(patch)

        def score_fn(plan: PlanCandidate) -> Tuple[float, ...]:
            patch: QAgentPatch = plan.payload
            key = patch.identifier
            if key not in score_cache:
                score_cache[key] = self._evaluator(patch)
            return score_cache[key]

        results = krr_guided_beam_search(
            initial,
            self._kb,
            successor_fn=successor_fn,
            facts_fn=facts_fn,
            score_fn=score_fn,
            params={},
            beam_width=beam_width,
            max_depth=max_depth,
            active_profiles=self._profiles,
        )

        patches: List[Tuple[QAgentPatch, Tuple[float, ...]]] = []
        seen_ids: Set[str] = set()
        for cand, score in results:
            patch: QAgentPatch = cand.payload
            if patch.identifier in seen_ids:
                continue
            seen_ids.add(patch.identifier)
            patches.append((patch, score))
        return patches


def demo_qagent_synth() -> List[Tuple[QAgentPatch, Tuple[float, ...]]]:
    """Run a small demo synthesis starting from a fixed QAgentPatch.

    Uses the synthetic evaluator by default and conservative profiles.
    """

    meta = QPatchMeta(
        name="demo-qagent-synth",
        description="Demo QAgentSynthesizer search",
        version="0.0.1",
        tags=("qagent", "experimental"),
    )
    base = QAgentPatch(
        identifier="base",
        num_price_bins=10,
        num_inventory_bins=10,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=1000,
        meta=meta,
    )

    synth = QAgentSynthesizer(base, profiles={"conservative"})
    return synth.synthesize()


def qagent_patch_from_dict(data: Dict[str, Any]) -> QAgentPatch:
    meta_data = data.get("meta", {}) or {}
    tags_raw = meta_data.get("tags", ("qagent", "preset"))
    if isinstance(tags_raw, (list, tuple)):
        tags = tuple(str(t) for t in tags_raw)
    else:
        tags = (str(tags_raw),)

    meta = QPatchMeta(
        name=str(meta_data.get("name", "unnamed")),
        description=str(meta_data.get("description", "qagent preset")),
        version=str(meta_data.get("version", "0.0.0")),
        tags=tags,
    )

    return QAgentPatch(
        identifier=str(data.get("identifier", "preset")),
        num_price_bins=int(data["num_price_bins"]),
        num_inventory_bins=int(data["num_inventory_bins"]),
        learning_rate=float(data["learning_rate"]),
        discount_factor=float(data["discount_factor"]),
        epsilon_start=float(data["epsilon_start"]),
        epsilon_end=float(data["epsilon_end"]),
        epsilon_decay_steps=int(data["epsilon_decay_steps"]),
        meta=meta,
    )


def load_qagent_patch_preset(path: Path) -> QAgentPatch:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("QAgent preset must be a JSON object at top level")
    patch = qagent_patch_from_dict(raw)
    if not validate_patch_fast(patch):
        raise ValueError("Loaded QAgent preset is structurally invalid")
    return patch
