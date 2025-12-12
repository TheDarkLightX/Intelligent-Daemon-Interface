"""Tests for STRIKE closure iteration bounds.

These tests verify that the STRIKE closure computation:
1. Terminates within bounded iterations for normal rules.
2. Raises an error for cyclic or divergent rule sets.
3. Produces correct results within the iteration limit.
"""

from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import pytest

from idi.devkit.experimental.strike_krr import (
    Constraint,
    KnowledgeBase,
    Rule,
    run_strike_closure,
    MAX_CLOSURE_ITERATIONS,
)


class TestClosureIterationBounds:
    """Tests for iteration bounds in run_strike_closure."""

    def test_normal_closure_terminates(self) -> None:
        """Normal rule set should terminate well within limit."""
        # Simple transitive closure: if a -> b and b -> c, then a -> c
        rules = (
            Rule(
                head=("reachable", ("?x", "?z")),
                body=(
                    ("reachable", ("?x", "?y")),
                    ("edge", ("?y", "?z")),
                ),
            ),
        )
        kb = KnowledgeBase(rules=rules, global_facts={})

        # Small graph: a -> b -> c -> d
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "reachable": {("a", "b")},
            "edge": {("b", "c"), ("c", "d")},
        }

        closure = run_strike_closure(kb, base_facts, {}, set())

        # Should derive: a->c, a->d (via transitivity)
        assert ("a", "c") in closure.get("reachable", set())
        assert ("a", "d") in closure.get("reachable", set())

    def test_empty_rules_terminates_immediately(self) -> None:
        """KB with no rules should terminate in one iteration."""
        kb = KnowledgeBase(rules=(), global_facts={})
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {"fact": {("x",)}}

        closure = run_strike_closure(kb, base_facts, {}, set())

        assert closure.get("fact") == {("x",)}

    def test_self_referential_rule_with_constraint_terminates(self) -> None:
        """Self-referential rule with terminating constraint should complete."""
        # Rule: count(N+1) :- count(N), N < 10
        def under_limit(env: Dict[str, Any]) -> bool:
            n = env.get("?n", 0)
            return n is not None and int(n) < 10

        rules = (
            Rule(
                head=("count", ("?next",)),
                body=(("count", ("?n",)),),
                constraints=(Constraint("under_limit", under_limit),),
            ),
        )
        kb = KnowledgeBase(rules=rules, global_facts={})

        # This rule can't actually increment, but tests constraint-bounded termination
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {"count": {(0,)}}

        closure = run_strike_closure(kb, base_facts, {}, set())

        # Should terminate without error
        assert "count" in closure

    def test_max_iterations_constant_exists(self) -> None:
        """MAX_CLOSURE_ITERATIONS constant should be defined and reasonable."""
        assert MAX_CLOSURE_ITERATIONS >= 100
        assert MAX_CLOSURE_ITERATIONS <= 10000

    def test_divergent_rules_raise_error(self) -> None:
        """Rules that generate unbounded new facts should raise error."""
        # Rule that keeps generating new facts: gen(X) :- gen(Y), X = Y + 1
        # Simulated by a constraint that always returns True and modifies env
        call_count = [0]

        def always_true_and_generate(env: Dict[str, Any]) -> bool:
            call_count[0] += 1
            # Inject new value to simulate divergence
            if "?x" in env:
                env["?new"] = call_count[0]
            return True

        rules = (
            Rule(
                head=("diverge", ("?new",)),
                body=(("seed", ("?x",)),),
                constraints=(Constraint("gen", always_true_and_generate),),
            ),
        )
        kb = KnowledgeBase(rules=rules, global_facts={})
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {"seed": {(i,) for i in range(MAX_CLOSURE_ITERATIONS + 10)}}

        # This should raise due to iteration limit
        with pytest.raises(RuntimeError, match="iteration"):
            run_strike_closure(kb, base_facts, {}, set())


class TestClosureCorrectness:
    """Tests that closure produces correct results within bounds."""

    def test_profile_filtering(self) -> None:
        """Rules with profile tags should only fire for active profiles."""
        rules = (
            Rule(
                head=("derived", ("?x",)),
                body=(("base", ("?x",)),),
                profiles=("conservative",),
            ),
            Rule(
                head=("other", ("?x",)),
                body=(("base", ("?x",)),),
                profiles=("aggressive",),
            ),
        )
        kb = KnowledgeBase(rules=rules, global_facts={})
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {"base": {("a",)}}

        # Only conservative profile active
        closure = run_strike_closure(kb, base_facts, {}, {"conservative"})

        assert ("a",) in closure.get("derived", set())
        assert ("a",) not in closure.get("other", set())

    def test_constraint_parameters(self) -> None:
        """Constraints should receive params from caller."""
        def check_threshold(env: Dict[str, Any]) -> bool:
            val = env.get("?v", 0)
            threshold = env.get("threshold", 10)
            return int(val) > int(threshold)

        rules = (
            Rule(
                head=("above_threshold", ("?x",)),
                body=(("value", ("?x", "?v")),),
                constraints=(Constraint("check", check_threshold),),
            ),
        )
        kb = KnowledgeBase(rules=rules, global_facts={})
        base_facts: Dict[str, Set[Tuple[Any, ...]]] = {
            "value": {("a", 5), ("b", 15), ("c", 25)},
        }

        closure = run_strike_closure(kb, base_facts, {"threshold": 10}, set())

        # Only b and c are above threshold
        above = closure.get("above_threshold", set())
        assert ("a",) not in above
        assert ("b",) in above
        assert ("c",) in above
