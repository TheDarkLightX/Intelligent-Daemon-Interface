from __future__ import annotations

from typing import Dict, Set, Tuple

from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.agent_synth import AgentSynthConfig, AgentSynthesizer
from idi.devkit.experimental.strike_krr import Constraint, KnowledgeBase, Rule


def _make_violation_kb() -> KnowledgeBase:
    def too_big(env: Dict[str, object]) -> bool:
        return int(env.get("?x", 0)) > 10

    rules = (
        Rule(
            head=("violates_constraint", ("?id",)),
            body=(("value", ("?id", "?x")),),
            constraints=(Constraint("too_big", too_big),),
        ),
    )

    return KnowledgeBase(rules=rules, global_facts={})


def test_agent_synth_prunes_invalid_payloads() -> None:
    kb = _make_violation_kb()
    meta = AgentPatchMeta(
        id="demo",
        name="demo",
        description="d",
        version="0.0.1",
        tags=("test",),
    )
    profiles: Set[str] = set()

    def successor(x: int):
        yield x
        yield x + 5
        yield x + 100

    def facts(cid: str, x: int) -> Dict[str, Set[Tuple[object, ...]]]:
        return {"value": {(cid, x)}}

    def score(x: int) -> Tuple[float, ...]:
        return (float(x),)

    synth = AgentSynthesizer(
        root_id="root",
        meta=meta,
        kb=kb,
        profiles=profiles,
        initial_payload=0,
        successor_fn=successor,
        facts_fn=facts,
        score_fn=score,
    )
    cfg = AgentSynthConfig(beam_width=4, max_depth=1)
    results = synth.synthesize(cfg)

    payloads = {p for p, _ in results}
    assert payloads
    assert all(v <= 10 for v in payloads)
