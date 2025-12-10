"""Formal specification and KRR integration tests.

These tests verify that the STRIKE/IKL knowledge engine correctly:
1. Defines and enforces formal invariants for agent patches.
2. Prunes candidates that violate constraints.
3. Integrates with the synth pipeline to produce only valid configs.

Key invariants tested:
- I1: state_cells = price_bins × inventory_bins ≤ MAX_STATE_CELLS
- I2: discount_factor ≥ MIN_DISCOUNT
- I3: learning_rate ≤ MAX_LEARNING_RATE (conservative profile)
- I4: exploration ≤ MAX_EXPLORATION (conservative profile)
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pytest

from idi.devkit.experimental.strike_krr import (
    Constraint,
    KnowledgeBase,
    KnowledgePack,
    Rule,
    build_comms_base_pack,
    build_kb_from_packs,
    build_qagent_base_pack,
    build_risk_conservative_pack,
    build_zk_tau_invariants_pack,
    evaluate_with_krr,
    lint_pack_meta,
    PackMeta,
    run_strike_closure,
)
from idi.devkit.experimental.synth_krr_planner import (
    build_kb_for_qpatch,
    krr_guided_beam_search,
    PlanCandidate,
    select_packs_for_qpatch,
)
from idi.devkit.experimental.sape_q_patch import QPatchMeta
from idi.devkit.experimental.agent_patch import AgentPatchMeta


# ---------------------------------------------------------------------------
# Formal Invariant Definitions
# ---------------------------------------------------------------------------

# I1: Maximum state cells (prevents state explosion)
MAX_STATE_CELLS = 256
MIN_DISCOUNT = 0.9
MAX_LEARNING_RATE_CONSERVATIVE = 0.3
MAX_EXPLORATION_CONSERVATIVE = 0.7


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qagent_meta() -> QPatchMeta:
    """Standard QAgent metadata for testing."""
    return QPatchMeta(
        name="test-qagent",
        description="Test QAgent patch",
        version="1.0.0",
        tags=("qagent", "test"),
    )


@pytest.fixture
def base_kb() -> KnowledgeBase:
    """Knowledge base with qagent_base pack."""
    pack = build_qagent_base_pack(max_state_cells=MAX_STATE_CELLS, min_discount=MIN_DISCOUNT)
    return build_kb_from_packs([pack])


@pytest.fixture
def full_kb() -> KnowledgeBase:
    """Knowledge base with all standard packs."""
    packs = [
        build_qagent_base_pack(max_state_cells=MAX_STATE_CELLS, min_discount=MIN_DISCOUNT),
        build_risk_conservative_pack(),
        build_comms_base_pack(),
        build_zk_tau_invariants_pack(),
    ]
    return build_kb_from_packs(packs)


# ---------------------------------------------------------------------------
# Invariant I1: State Size Constraint Tests
# ---------------------------------------------------------------------------

class TestStateSizeInvariant:
    """Tests for Invariant I1: state_cells ≤ MAX_STATE_CELLS."""

    def test_valid_state_size_passes(self, base_kb: KnowledgeBase) -> None:
        """Patch with state size within limit should pass."""
        # 8 × 8 = 64 cells ≤ 256
        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "num_price_bins", 8),
                ("p1", "num_inventory_bins", 8),
            },
        }

        allowed, reasons = evaluate_with_krr(
            base_kb,
            base_facts=facts,
            params={"max_state_cells": MAX_STATE_CELLS},
            active_profiles=set(),
        )

        assert allowed is True
        assert len(reasons) == 0

    def test_excessive_state_size_rejected(self, base_kb: KnowledgeBase) -> None:
        """Patch with state size exceeding limit should be rejected."""
        # 20 × 20 = 400 cells > 256
        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "num_price_bins", 20),
                ("p1", "num_inventory_bins", 20),
            },
        }

        allowed, reasons = evaluate_with_krr(
            base_kb,
            base_facts=facts,
            params={"max_state_cells": MAX_STATE_CELLS},
            active_profiles=set(),
        )

        assert allowed is False
        assert any("state_size" in r for r in reasons)

    def test_boundary_state_size(self, base_kb: KnowledgeBase) -> None:
        """State size exactly at limit should pass."""
        # 16 × 16 = 256 cells == 256 (passes because constraint is >)
        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "num_price_bins", 16),
                ("p1", "num_inventory_bins", 16),
            },
        }

        allowed, reasons = evaluate_with_krr(
            base_kb,
            base_facts=facts,
            params={"max_state_cells": MAX_STATE_CELLS},
            active_profiles=set(),
        )

        # Exactly at limit should pass (> not >=)
        assert allowed is True


# ---------------------------------------------------------------------------
# Invariant I2: Discount Factor Constraint Tests
# ---------------------------------------------------------------------------

class TestDiscountInvariant:
    """Tests for Invariant I2: discount_factor ≥ MIN_DISCOUNT."""

    def test_valid_discount_passes(self, base_kb: KnowledgeBase) -> None:
        """Discount factor above threshold should pass."""
        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "discount_factor", 0.95),
            },
        }

        allowed, reasons = evaluate_with_krr(
            base_kb,
            base_facts=facts,
            params={"min_discount": MIN_DISCOUNT},
            active_profiles=set(),
        )

        assert allowed is True

    def test_low_discount_rejected(self, base_kb: KnowledgeBase) -> None:
        """Discount factor below threshold should be rejected."""
        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "discount_factor", 0.5),  # Too low
            },
        }

        allowed, reasons = evaluate_with_krr(
            base_kb,
            base_facts=facts,
            params={"min_discount": MIN_DISCOUNT},
            active_profiles=set(),
        )

        assert allowed is False
        assert any("discount" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Profile-Specific Constraint Tests
# ---------------------------------------------------------------------------

class TestProfileConstraints:
    """Tests for profile-specific constraints (e.g., conservative)."""

    def test_conservative_profile_rejects_high_learning_rate(self) -> None:
        """Conservative profile should reject high learning rates."""
        pack = build_risk_conservative_pack()
        kb = build_kb_from_packs([pack])

        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "learning_rate", 0.9),  # Too aggressive
            },
        }

        allowed, reasons = evaluate_with_krr(
            kb,
            base_facts=facts,
            params={},
            active_profiles={"conservative"},
        )

        assert allowed is False
        assert any("learning_rate" in r for r in reasons)

    def test_conservative_profile_allows_safe_learning_rate(self) -> None:
        """Conservative profile should allow low learning rates."""
        pack = build_risk_conservative_pack()
        kb = build_kb_from_packs([pack])

        facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "patch": {("p1",)},
            "param_value": {
                ("p1", "learning_rate", 0.1),  # Safe
            },
        }

        allowed, reasons = evaluate_with_krr(
            kb,
            base_facts=facts,
            params={},
            active_profiles={"conservative"},
        )

        assert allowed is True


# ---------------------------------------------------------------------------
# KRR-Guided Beam Search Integration Tests
# ---------------------------------------------------------------------------

class TestKRRGuidedBeamSearch:
    """Tests for KRR integration with beam search."""

    def test_beam_search_prunes_invalid_candidates(self, full_kb: KnowledgeBase) -> None:
        """Beam search should only return KRR-valid candidates."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Payload:
            price_bins: int
            inventory_bins: int
            discount: float
            learning_rate: float

        meta = AgentPatchMeta(
            id="root",
            name="test",
            description="",
            version="1.0",
        )

        # Initial candidate is valid
        initial = PlanCandidate(
            id="root",
            meta=meta,
            profiles={"conservative"},
            payload=Payload(price_bins=8, inventory_bins=8, discount=0.95, learning_rate=0.1),
        )

        def successor_fn(cand: PlanCandidate):
            base: Payload = cand.payload
            children = []

            # Generate some valid and invalid children
            configs = [
                (8, 8, 0.95, 0.1),   # Valid
                (20, 20, 0.95, 0.1), # Invalid: state size too large
                (8, 8, 0.5, 0.1),    # Invalid: discount too low
                (8, 8, 0.95, 0.8),   # Invalid: learning rate too high for conservative
            ]

            for idx, (pb, ib, d, lr) in enumerate(configs):
                children.append(PlanCandidate(
                    id=f"{cand.id}-{idx}",
                    meta=cand.meta,
                    profiles=cand.profiles,
                    payload=Payload(price_bins=pb, inventory_bins=ib, discount=d, learning_rate=lr),
                ))
            return children

        def facts_fn(cand: PlanCandidate):
            payload: Payload = cand.payload
            return {
                "patch": {(cand.id,)},
                "param_value": {
                    (cand.id, "num_price_bins", payload.price_bins),
                    (cand.id, "num_inventory_bins", payload.inventory_bins),
                    (cand.id, "discount_factor", payload.discount),
                    (cand.id, "learning_rate", payload.learning_rate),
                },
            }

        def score_fn(cand: PlanCandidate):
            return (1.0,)  # All same score for this test

        explanations: Dict[str, List[str]] = {}
        stats: Dict[str, float] = {}

        results = krr_guided_beam_search(
            initial,
            full_kb,
            successor_fn=successor_fn,
            facts_fn=facts_fn,
            score_fn=score_fn,
            beam_width=10,
            max_depth=1,
            active_profiles={"conservative"},
            explanations=explanations,
            stats=stats,
        )

        # Should have pruned some candidates
        assert stats["pruned_krr"] > 0

        # All returned candidates should be valid
        for cand, _score in results:
            facts = facts_fn(cand)
            allowed, _ = evaluate_with_krr(
                full_kb,
                base_facts=facts,
                params={},
                active_profiles={"conservative"},
            )
            assert allowed is True, f"Invalid candidate returned: {cand.id}"


# ---------------------------------------------------------------------------
# Pack Selection and Composition Tests
# ---------------------------------------------------------------------------

class TestPackSelection:
    """Tests for knowledge pack selection logic."""

    def test_select_packs_for_qagent_tag(self, qagent_meta: QPatchMeta) -> None:
        """Packs with 'qagent' domain tag should be selected."""
        packs = select_packs_for_qpatch(qagent_meta, set())

        pack_names = {p.name for p in packs}
        assert "qagent_base" in pack_names

    def test_select_packs_for_conservative_profile(self, qagent_meta: QPatchMeta) -> None:
        """Conservative profile should include risk_conservative pack."""
        packs = select_packs_for_qpatch(qagent_meta, {"conservative"})

        pack_names = {p.name for p in packs}
        assert "risk_conservative" in pack_names

    def test_build_kb_for_qpatch(self, qagent_meta: QPatchMeta) -> None:
        """build_kb_for_qpatch should return a valid KnowledgeBase."""
        kb = build_kb_for_qpatch(qagent_meta, {"conservative"})

        assert len(kb.rules) > 0


# ---------------------------------------------------------------------------
# Pack Linting Tests
# ---------------------------------------------------------------------------

class TestPackLinting:
    """Tests for pack metadata linting."""

    def test_lint_valid_pack(self) -> None:
        """Valid pack should pass linting."""
        pack = build_qagent_base_pack()
        meta = PackMeta(name=pack.name, version=pack.version, rules=pack.rules)

        ok, msg = lint_pack_meta(meta)

        assert ok is True
        assert msg == ""

    def test_lint_empty_name_rejected(self) -> None:
        """Pack with empty name should fail linting."""
        meta = PackMeta(
            name="",
            version="1.0",
            rules=(Rule(head=("test", ()), body=()),),
        )

        ok, msg = lint_pack_meta(meta)

        assert ok is False
        assert "name" in msg.lower()

    def test_lint_no_rules_rejected(self) -> None:
        """Pack with no rules should fail linting."""
        meta = PackMeta(name="test", version="1.0", rules=())

        ok, msg = lint_pack_meta(meta)

        assert ok is False
        assert "no rules" in msg.lower()


# ---------------------------------------------------------------------------
# Fixpoint Closure Tests
# ---------------------------------------------------------------------------

class TestStrikeClosure:
    """Tests for STRIKE fixpoint closure computation."""

    def test_closure_propagates_facts(self) -> None:
        """Closure should derive new facts from rules."""
        # A simple rule: derived(X) :- base(X)
        rule = Rule(
            head=("derived", ("?x",)),
            body=(("base", ("?x",)),),
        )
        kb = KnowledgeBase(rules=(rule,), global_facts={})

        base_facts = {"base": {("foo",), ("bar",)}}

        closure = run_strike_closure(kb, base_facts, {}, set())

        assert "derived" in closure
        assert ("foo",) in closure["derived"]
        assert ("bar",) in closure["derived"]

    def test_closure_respects_constraints(self) -> None:
        """Closure should not derive facts when constraints fail."""
        # Rule: positive(X) :- value(X), X > 0
        def check_positive(env: Dict[str, Any]) -> bool:
            x = env.get("?x", 0)
            return x > 0

        rule = Rule(
            head=("positive", ("?x",)),
            body=(("value", ("?x",)),),
            constraints=(Constraint("positive_check", check_positive),),
        )
        kb = KnowledgeBase(rules=(rule,), global_facts={})

        base_facts = {"value": {(10,), (-5,), (0,)}}

        closure = run_strike_closure(kb, base_facts, {}, set())

        assert "positive" in closure
        assert (10,) in closure["positive"]
        assert (-5,) not in closure["positive"]
        assert (0,) not in closure["positive"]


# ---------------------------------------------------------------------------
# ZK/Tau Invariants Pack Tests
# ---------------------------------------------------------------------------

class TestZKTauInvariants:
    """Tests for ZK/Tau invariants pack."""

    def test_zk_commitment_failure_detected(self) -> None:
        """ZK commitment failure should trigger violation."""
        pack = build_zk_tau_invariants_pack()
        kb = build_kb_from_packs([pack])

        facts = {"commitment_status": {("bundle1", "fail")}}

        allowed, reasons = evaluate_with_krr(kb, facts, {}, set())

        assert allowed is False
        assert any("commitment" in r.lower() for r in reasons)

    def test_zk_all_ok_passes(self) -> None:
        """All ZK statuses OK should pass."""
        pack = build_zk_tau_invariants_pack()
        kb = build_kb_from_packs([pack])

        facts = {
            "commitment_status": {("bundle1", "ok")},
            "method_id_status": {("bundle1", "ok")},
            "journal_status": {("bundle1", "ok")},
        }

        allowed, reasons = evaluate_with_krr(kb, facts, {}, set())

        assert allowed is True
