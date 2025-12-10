from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.synth_krr_planner import PlanCandidate, ProfileSet, krr_guided_beam_search
from idi.devkit.experimental.strike_krr import KnowledgeBase

CandidatePayload = Any
SuccessorFn = Callable[[CandidatePayload], Iterable[CandidatePayload]]
FactsFn = Callable[[str, CandidatePayload], Dict[str, Set[Tuple[Any, ...]]]]
ScoreFn = Callable[[CandidatePayload], Tuple[float, ...]]

MAX_BEAM_WIDTH = 32
MAX_SEARCH_DEPTH = 8


@dataclass(frozen=True)
class AgentSynthConfig:
    beam_width: int = 4
    max_depth: int = 3


class AgentSynthesizer:
    def __init__(
        self,
        *,
        root_id: str,
        meta: AgentPatchMeta,
        kb: KnowledgeBase,
        profiles: ProfileSet | None,
        initial_payload: CandidatePayload,
        successor_fn: SuccessorFn,
        facts_fn: FactsFn,
        score_fn: ScoreFn,
    ) -> None:
        self._root_id = root_id
        self._meta = meta
        self._kb = kb
        self._profiles: ProfileSet = set() if profiles is None else set(profiles)
        self._initial_payload = initial_payload
        self._successor_fn = successor_fn
        self._facts_fn = facts_fn
        self._score_fn = score_fn

    def _make_candidate(self, cid: str, payload: CandidatePayload) -> PlanCandidate:
        return PlanCandidate(id=cid, meta=self._meta, profiles=self._profiles, payload=payload)

    def synthesize(self, config: AgentSynthConfig | None = None) -> List[Tuple[CandidatePayload, Tuple[float, ...]]]:
        cfg = config or AgentSynthConfig()
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

        initial = self._make_candidate(self._root_id, self._initial_payload)
        score_cache: Dict[str, Tuple[float, ...]] = {}

        def successor(plan: PlanCandidate) -> Iterable[PlanCandidate]:
            base_id = plan.id
            idx = 0
            for payload in self._successor_fn(plan.payload):
                cid = f"{base_id}-{idx}"
                idx += 1
                yield self._make_candidate(cid, payload)

        def facts(plan: PlanCandidate) -> Dict[str, Set[Tuple[Any, ...]]]:
            return self._facts_fn(plan.id, plan.payload)

        def score(plan: PlanCandidate) -> Tuple[float, ...]:
            cid = plan.id
            if cid not in score_cache:
                score_cache[cid] = self._score_fn(plan.payload)
            return score_cache[cid]

        results = krr_guided_beam_search(
            initial,
            self._kb,
            successor_fn=successor,
            facts_fn=facts,
            score_fn=score,
            params={},
            beam_width=beam_width,
            max_depth=max_depth,
            active_profiles=self._profiles,
        )

        out: List[Tuple[CandidatePayload, Tuple[float, ...]]] = []
        seen: Set[str] = set()
        for cand, s in results:
            if cand.id in seen:
                continue
            seen.add(cand.id)
            out.append((cand.payload, s))
        return out
