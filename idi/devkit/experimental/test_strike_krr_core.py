from __future__ import annotations

from typing import Dict, Set, Tuple

from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.strike_krr import (
    Constraint,
    KnowledgeBase,
    PackMeta,
    Rule,
    build_comms_base_pack,
    build_qagent_base_pack,
    build_risk_conservative_pack,
    build_zk_tau_invariants_pack,
    evaluate_with_krr,
    lint_pack_meta,
    run_strike_closure,
)
from idi.devkit.experimental.sape_q_patch import QAgentPatch, QPatchMeta
from idi.devkit.experimental.synth_krr_planner import PlanCandidate, krr_guided_beam_search
from idi.devkit.experimental.qagent_synth import QAgentSynthConfig, QAgentSynthesizer


Atom = Tuple[str, Tuple[object, ...]]


def _make_simple_kb() -> KnowledgeBase:
    """Build a minimal KB for testing STRIKE closure and evaluation."""

    def over_limit(env: Dict[str, object]) -> bool:
        return int(env.get("?x", 0)) > int(env.get("limit", 0))

    rules = (
        Rule(
            head=("flag", ("?id",)),
            body=(("value", ("?id", "?x")),),
            constraints=(Constraint("over_limit", over_limit),),
        ),
    )

    return KnowledgeBase(rules=rules, global_facts={})


def test_run_strike_closure_and_evaluate_with_krr_allows_when_no_violations() -> None:
    kb = _make_simple_kb()
    base_facts: Dict[str, Set[Tuple[object, ...]]] = {
        "value": {("p1", 1)},
    }
    params = {"limit": 10}

    closure = run_strike_closure(kb, base_facts, params, active_profiles=set())
    assert "flag" not in closure

    allowed, reasons = evaluate_with_krr(kb, base_facts, params, active_profiles=set(), violation_predicate="flag")
    assert allowed is True
    assert reasons == []


def test_run_strike_closure_and_evaluate_with_krr_reports_violations() -> None:
    kb = _make_simple_kb()
    base_facts: Dict[str, Set[Tuple[object, ...]]] = {
        "value": {("p1", 42)},
    }
    params = {"limit": 10}

    closure = run_strike_closure(kb, base_facts, params, active_profiles=set())
    assert closure.get("flag") == {("p1",)}

    allowed, reasons = evaluate_with_krr(kb, base_facts, params, active_profiles=set(), violation_predicate="flag")
    assert allowed is False
    assert reasons == ["Violation: ('p1',)"]


def test_evaluate_with_krr_handles_internal_errors() -> None:
    def bad_constraint(env: Dict[str, object]) -> bool:  # noqa: ARG001
        raise RuntimeError("boom")

    kb = KnowledgeBase(
        rules=(
            Rule(
                head=("x", ("a",)),
                body=(),
                constraints=(Constraint("bad", bad_constraint),),
            ),
        ),
        global_facts={},
    )

    allowed, reasons = evaluate_with_krr(kb, base_facts={}, params={}, active_profiles=set())
    assert allowed is False
    assert reasons and reasons[0].startswith("internal_error:")


def test_lint_pack_meta_rejects_empty_and_malformed_packs() -> None:
    ok, msg = lint_pack_meta(PackMeta(name="", version="0.1.0", rules=()))
    assert ok is False
    assert "Pack name" in msg

    ok, msg = lint_pack_meta(PackMeta(name="p", version="", rules=()))
    assert ok is False
    assert "empty version" in msg

    ok, msg = lint_pack_meta(PackMeta(name="p", version="0.1.0", rules=()))
    assert ok is False
    assert "has no rules" in msg

    bad_rule = Rule(head=("", ("a",)), body=())
    ok, msg = lint_pack_meta(PackMeta(name="p", version="0.1.0", rules=(bad_rule,)))
    assert ok is False
    assert "invalid head predicate" in msg


def test_lint_pack_meta_accepts_example_packs() -> None:
    for builder in (
        build_qagent_base_pack,
        build_risk_conservative_pack,
        build_comms_base_pack,
        build_zk_tau_invariants_pack,
    ):
        pack = builder()  # type: ignore[call-arg]
        meta = PackMeta(name=pack.name, version=pack.version, rules=pack.rules)
        ok, msg = lint_pack_meta(meta)
        assert ok is True, msg


def _make_violation_kb() -> KnowledgeBase:
    def too_big(env: Dict[str, object]) -> bool:
        return int(env.get("?x", 0)) > int(env.get("limit", 0))

    rules = (
        Rule(
            head=("violates_constraint", ("?id",)),
            body=(("value", ("?id", "?x")),),
            constraints=(Constraint("too_big", too_big),),
        ),
    )

    return KnowledgeBase(rules=rules, global_facts={})


def _make_empty_kb() -> KnowledgeBase:
    return KnowledgeBase(rules=(), global_facts={})


def test_krr_guided_beam_search_prunes_violations() -> None:
    kb = _make_violation_kb()
    meta = AgentPatchMeta(
        id="m",
        name="m",
        description="d",
        version="0.0.1",
        tags=("qagent",),
    )
    profiles: Set[str] = {"conservative"}
    initial = PlanCandidate(
        id="root",
        meta=meta,
        profiles=profiles,
        payload=1,
    )

    def successor_fn(plan: PlanCandidate):
        value = int(plan.payload)
        yield PlanCandidate(id=f"{plan.id}-safe", meta=plan.meta, profiles=plan.profiles, payload=value)
        yield PlanCandidate(id=f"{plan.id}-bad", meta=plan.meta, profiles=plan.profiles, payload=value + 100)

    def facts_fn(plan: PlanCandidate) -> Dict[str, Set[Tuple[object, ...]]]:
        value = int(plan.payload)
        return {"value": {(plan.id, value)}}

    def score_fn(plan: PlanCandidate) -> Tuple[float, ...]:
        return (float(plan.payload),)

    results = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn,
        facts_fn=facts_fn,
        score_fn=score_fn,
        params={"limit": 10},
        beam_width=4,
        max_depth=1,
        active_profiles=profiles,
    )

    ids = {cand.id for cand, _ in results}
    assert "root-bad" not in ids


def test_krr_guided_beam_search_respects_beam_width_and_is_deterministic() -> None:
    kb = _make_empty_kb()
    meta = AgentPatchMeta(
        id="m2",
        name="m2",
        description="d2",
        version="0.0.1",
        tags=("qagent",),
    )
    profiles: Set[str] = set()
    initial = PlanCandidate(
        id="root",
        meta=meta,
        profiles=profiles,
        payload=0,
    )

    def successor_fn(plan: PlanCandidate):
        value = int(plan.payload)
        for i in range(3):
            yield PlanCandidate(
                id=f"{plan.id}-{i}",
                meta=plan.meta,
                profiles=plan.profiles,
                payload=value + i + 1,
            )

    def facts_fn(plan: PlanCandidate) -> Dict[str, Set[Tuple[object, ...]]]:
        return {}

    def score_fn(plan: PlanCandidate) -> Tuple[float, ...]:
        return (float(plan.payload),)

    stats: Dict[str, float] = {}
    results1 = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn,
        facts_fn=facts_fn,
        score_fn=score_fn,
        params={},
        beam_width=2,
        max_depth=1,
        active_profiles=profiles,
        stats=stats,
    )
    results2 = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn,
        facts_fn=facts_fn,
        score_fn=score_fn,
        params={},
        beam_width=2,
        max_depth=1,
        active_profiles=profiles,
    )

    ids1 = [cand.id for cand, _ in results1]
    ids2 = [cand.id for cand, _ in results2]
    assert ids1 == ids2
    assert len(ids1) <= 1 + 2 * 1
    assert stats["visited"] >= len(ids1)
    assert stats["accepted"] == float(len(results1))


def _make_valid_qagent_patch() -> QAgentPatch:
    meta = QPatchMeta(
        name="q",
        description="qpatch",
        version="0.0.1",
        tags=("qagent", "experimental"),
    )
    return QAgentPatch(
        identifier="base",
        num_price_bins=4,
        num_inventory_bins=4,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        meta=meta,
    )


def test_qagent_synth_rejects_invalid_base_patch() -> None:
    meta = QPatchMeta(
        name="bad",
        description="bad",
        version="0.0.1",
        tags=("qagent",),
    )
    bad = QAgentPatch(
        identifier="bad",
        num_price_bins=0,
        num_inventory_bins=1,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=10,
        meta=meta,
    )

    raised = False
    try:
        QAgentSynthesizer(bad, profiles={"conservative"})
    except ValueError:
        raised = True
    assert raised is True


def test_qagent_synth_uses_custom_evaluator() -> None:
    base = _make_valid_qagent_patch()

    calls = []

    def evaluator(patch: QAgentPatch) -> Tuple[float, ...]:
        calls.append(patch.identifier)
        return (1.23,)

    synth = QAgentSynthesizer(base, profiles={"conservative"}, evaluator=evaluator)
    cfg = QAgentSynthConfig(beam_width=1, max_depth=0)
    results = synth.synthesize(config=cfg)
    assert results
    for _patch, score in results:
        assert score == (1.23,)
    assert calls


def test_qagent_synth_returns_unique_patch_identifiers() -> None:
    base = _make_valid_qagent_patch()
    synth = QAgentSynthesizer(base, profiles={"conservative"})
    cfg = QAgentSynthConfig(beam_width=4, max_depth=1)
    results = synth.synthesize(config=cfg)
    ids = [patch.identifier for patch, _ in results]
    assert len(ids) == len(set(ids))


def test_qagent_synth_integration_with_strike_packs() -> None:
    base = _make_valid_qagent_patch()
    synth = QAgentSynthesizer(base, profiles={"conservative"})
    cfg = QAgentSynthConfig(beam_width=2, max_depth=1)
    results = synth.synthesize(config=cfg)
    assert results
    for patch, score in results:
        assert isinstance(patch, QAgentPatch)
        assert isinstance(score, tuple)
        assert score

