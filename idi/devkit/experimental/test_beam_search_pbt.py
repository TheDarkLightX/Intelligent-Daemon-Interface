"""Property-based tests for krr_guided_beam_search.

Key properties verified:
- Determinism: same inputs yield same outputs.
- Monotonicity: tightening beam/depth never adds candidates.
- No duplicates: returned candidates have unique IDs.
- Stats consistency: stats match returned candidate counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

import pytest

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    def given(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator

    def settings(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator

    def assume(condition):  # type: ignore
        pass

    class st:  # type: ignore
        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None


from idi.devkit.experimental.agent_patch import AgentPatchMeta
from idi.devkit.experimental.strike_krr import KnowledgeBase
from idi.devkit.experimental.synth_krr_planner import (
    PlanCandidate,
    krr_guided_beam_search,
)


# ---------------------------------------------------------------------------
# Minimal fixtures for testing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimplePayload:
    """Minimal payload for testing."""
    value: float


def make_empty_kb() -> KnowledgeBase:
    """Create an empty KnowledgeBase that accepts all candidates."""
    return KnowledgeBase(global_facts={}, rules=[])


def simple_facts_fn(cand: PlanCandidate) -> Dict[str, Set[Tuple[Any, ...]]]:
    """Facts function that returns minimal facts."""
    return {"candidate": {(cand.id,)}}


def simple_score_fn(cand: PlanCandidate) -> Tuple[float, ...]:
    """Score based on payload value."""
    if isinstance(cand.payload, SimplePayload):
        return (cand.payload.value,)
    return (0.0,)


def successor_fn_linear(cand: PlanCandidate) -> Iterable[PlanCandidate]:
    """Generate one child with incremented value (limited depth)."""
    if isinstance(cand.payload, SimplePayload):
        if cand.payload.value < 10.0:
            child_value = cand.payload.value + 1.0
            child_id = f"{cand.id}-{int(child_value)}"
            meta = AgentPatchMeta(
                id=child_id,
                name="child",
                description="",
                version="1.0",
            )
            yield PlanCandidate(
                id=child_id,
                meta=meta,
                profiles=cand.profiles,
                payload=SimplePayload(value=child_value),
            )


def successor_fn_branching(cand: PlanCandidate) -> Iterable[PlanCandidate]:
    """Generate two children with different values."""
    if isinstance(cand.payload, SimplePayload):
        if cand.payload.value < 5.0:
            for delta in [1.0, 2.0]:
                child_value = cand.payload.value + delta
                child_id = f"{cand.id}-{delta}"
                meta = AgentPatchMeta(
                    id=child_id,
                    name="child",
                    description="",
                    version="1.0",
                )
                yield PlanCandidate(
                    id=child_id,
                    meta=meta,
                    profiles=cand.profiles,
                    payload=SimplePayload(value=child_value),
                )


def make_initial_candidate(value: float = 0.0) -> PlanCandidate:
    """Create an initial candidate with given value."""
    meta = AgentPatchMeta(
        id="root",
        name="root",
        description="",
        version="1.0",
    )
    return PlanCandidate(
        id="root",
        meta=meta,
        profiles=set(),
        payload=SimplePayload(value=value),
    )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    beam_width=st.integers(min_value=1, max_value=16),
    max_depth=st.integers(min_value=0, max_value=8),
    start_value=st.floats(min_value=0.0, max_value=5.0),
)
@settings(max_examples=50)
def test_beam_search_determinism(
    beam_width: int,
    max_depth: int,
    start_value: float,
) -> None:
    """Same inputs should yield same outputs."""
    kb = make_empty_kb()
    initial = make_initial_candidate(start_value)

    result1 = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
    )

    result2 = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
    )

    ids1 = [cand.id for cand, _ in result1]
    ids2 = [cand.id for cand, _ in result2]
    assert ids1 == ids2


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    beam_width=st.integers(min_value=2, max_value=16),
    max_depth=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=30)
def test_beam_search_monotonicity_width(
    beam_width: int,
    max_depth: int,
) -> None:
    """Smaller beam width never adds candidates not in larger search."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)

    result_large = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_branching,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
    )
    ids_large = {cand.id for cand, _ in result_large}

    result_small = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_branching,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=max(1, beam_width - 1),
        max_depth=max_depth,
    )
    ids_small = {cand.id for cand, _ in result_small}

    # Smaller search should be subset of larger (or equal)
    assert ids_small.issubset(ids_large)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    beam_width=st.integers(min_value=1, max_value=16),
    max_depth=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=30)
def test_beam_search_monotonicity_depth(
    beam_width: int,
    max_depth: int,
) -> None:
    """Smaller max_depth never adds candidates not in deeper search."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)

    result_deep = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
    )
    ids_deep = {cand.id for cand, _ in result_deep}

    result_shallow = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max(0, max_depth - 1),
    )
    ids_shallow = {cand.id for cand, _ in result_shallow}

    assert ids_shallow.issubset(ids_deep)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    beam_width=st.integers(min_value=1, max_value=16),
    max_depth=st.integers(min_value=0, max_value=8),
)
@settings(max_examples=50)
def test_beam_search_no_duplicates(
    beam_width: int,
    max_depth: int,
) -> None:
    """Returned candidates should have unique IDs."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)

    result = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_branching,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
    )

    ids = [cand.id for cand, _ in result]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    beam_width=st.integers(min_value=1, max_value=16),
    max_depth=st.integers(min_value=0, max_value=8),
)
@settings(max_examples=50)
def test_beam_search_stats_consistency(
    beam_width: int,
    max_depth: int,
) -> None:
    """Stats should be consistent with returned results."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)
    stats: Dict[str, float] = {}

    result = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=beam_width,
        max_depth=max_depth,
        stats=stats,
    )

    assert stats.get("accepted", 0.0) == float(len(result))
    assert stats.get("visited", 0.0) >= float(len(result))
    assert stats.get("pruned_krr", 0.0) >= 0.0


# ---------------------------------------------------------------------------
# Edge case tests (non-hypothesis)
# ---------------------------------------------------------------------------

def test_beam_search_zero_width_returns_empty() -> None:
    """beam_width=0 should return empty list."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)

    result = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=0,
        max_depth=5,
    )

    assert result == []


def test_beam_search_negative_depth_returns_empty() -> None:
    """Negative max_depth should return empty list."""
    kb = make_empty_kb()
    initial = make_initial_candidate(0.0)

    result = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=successor_fn_linear,
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=4,
        max_depth=-1,
    )

    assert result == []


def test_beam_search_initial_always_considered() -> None:
    """Initial candidate should always be in results if KRR-valid."""
    kb = make_empty_kb()
    initial = make_initial_candidate(42.0)

    result = krr_guided_beam_search(
        initial,
        kb,
        successor_fn=lambda _: [],  # No successors
        facts_fn=simple_facts_fn,
        score_fn=simple_score_fn,
        beam_width=4,
        max_depth=0,
    )

    assert len(result) == 1
    assert result[0][0].id == "root"
