"""Property-based tests for AgentPatch using hypothesis.

These tests verify key invariants for the AgentPatch engine:
- Round-trip consistency (to_dict -> from_dict yields equivalent patch).
- Validation under random but bounded inputs.
- Diff symmetry and self-diff properties.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Provide stubs so module loads without hypothesis
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
        def text(*args, **kwargs):
            return None

        @staticmethod
        def tuples(*args, **kwargs):
            return None

        @staticmethod
        def dictionaries(*args, **kwargs):
            return None

        @staticmethod
        def fixed_dictionaries(*args, **kwargs):
            return None

        @staticmethod
        def one_of(*args, **kwargs):
            return None

        @staticmethod
        def none():
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def booleans():
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

from idi.devkit.experimental.agent_patch import (
    AgentPatch,
    AgentPatchMeta,
    agent_patch_from_dict,
    agent_patch_to_dict,
    diff_agent_patches,
    validate_agent_patch,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    # Non-empty strings for required fields (avoid heavy filtering)
    nonempty_text = st.from_regex(r"[A-Za-z0-9_\-]{1,64}", fullmatch=True)

    # Safe payload values: JSON-serializable primitives
    safe_json_values = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        st.text(max_size=32),
    )

    identifier_text = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,15}", fullmatch=True)

    # Shallow dicts with safe values (for payload, spec_params, zk_profile)
    safe_json_dict = st.dictionaries(
        keys=identifier_text,
        values=safe_json_values,
        max_size=8,
    )

    # Tags: tuple of short strings (avoid heavy filtering)
    tags_strategy = st.lists(
        st.from_regex(r"[A-Za-z0-9]{1,16}", fullmatch=True),
        max_size=5,
    ).map(tuple)

    @st.composite
    def agent_patch_meta_strategy(draw):
        return AgentPatchMeta(
            id=draw(nonempty_text),
            name=draw(nonempty_text),
            description=draw(st.text(max_size=128)),
            version=draw(nonempty_text),
            tags=draw(tags_strategy),
        )

    @st.composite
    def agent_patch_strategy(draw):
        meta = draw(agent_patch_meta_strategy())
        return AgentPatch(
            meta=meta,
            agent_type=draw(nonempty_text),
            payload=draw(safe_json_dict),
            spec_backend=draw(st.sampled_from(["tau", "none", "custom"])),
            spec_params=draw(safe_json_dict),
            zk_profile=draw(safe_json_dict),
        )
else:
    # Fallback so tests can skip gracefully
    agent_patch_strategy = lambda: None  # type: ignore
    agent_patch_meta_strategy = lambda: None  # type: ignore


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(patch=agent_patch_strategy())
@settings(max_examples=100)
def test_agent_patch_roundtrip(patch: AgentPatch) -> None:
    """to_dict -> from_dict yields an equivalent AgentPatch."""
    data = agent_patch_to_dict(patch)
    restored = agent_patch_from_dict(data)

    # Meta fields
    assert restored.meta.id == patch.meta.id
    assert restored.meta.name == patch.meta.name
    assert restored.meta.description == patch.meta.description
    assert restored.meta.version == patch.meta.version
    assert restored.meta.tags == patch.meta.tags

    # Top-level fields
    assert restored.agent_type == patch.agent_type
    assert restored.spec_backend == patch.spec_backend
    assert dict(restored.payload) == dict(patch.payload)
    assert dict(restored.spec_params) == dict(patch.spec_params)
    assert dict(restored.zk_profile) == dict(patch.zk_profile)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(patch=agent_patch_strategy())
@settings(max_examples=100)
def test_agent_patch_validation_passes(patch: AgentPatch) -> None:
    """Generated patches should pass validation (strategy ensures non-empty)."""
    validate_agent_patch(patch)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(patch=agent_patch_strategy())
@settings(max_examples=50)
def test_agent_patch_self_diff_is_empty(patch: AgentPatch) -> None:
    """Diffing a patch with itself should yield no differences."""
    diff = diff_agent_patches(patch, patch)
    assert diff == {}


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(a=agent_patch_strategy(), b=agent_patch_strategy())
@settings(max_examples=50)
def test_agent_patch_diff_symmetry(a: AgentPatch, b: AgentPatch) -> None:
    """diff(a, b) and diff(b, a) should have the same keys."""
    diff_ab = diff_agent_patches(a, b)
    diff_ba = diff_agent_patches(b, a)
    assert set(diff_ab.keys()) == set(diff_ba.keys())


# ---------------------------------------------------------------------------
# Fuzz-style tests for from_dict with malformed input
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(
    data=st.dictionaries(
        keys=st.text(max_size=16),
        values=st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.text(max_size=32),
            st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=4),
        ),
        max_size=10,
    )
)
@settings(max_examples=100)
def test_agent_patch_from_dict_handles_malformed(data: Dict[str, Any]) -> None:
    """from_dict should either succeed or raise ValueError, never crash."""
    try:
        patch = agent_patch_from_dict(data)
        # If it succeeded, it must be valid
        validate_agent_patch(patch)
    except ValueError:
        # Expected for missing/empty required fields
        pass


# ---------------------------------------------------------------------------
# Edge case tests (non-hypothesis)
# ---------------------------------------------------------------------------

def test_agent_patch_empty_fields_rejected() -> None:
    """Empty required fields should raise ValueError."""
    meta = AgentPatchMeta(id="", name="n", description="", version="1.0")
    patch = AgentPatch(meta=meta, agent_type="test")
    with pytest.raises(ValueError, match="meta.id"):
        validate_agent_patch(patch)


def test_agent_patch_empty_agent_type_rejected() -> None:
    meta = AgentPatchMeta(id="p", name="n", description="", version="1.0")
    patch = AgentPatch(meta=meta, agent_type="")
    with pytest.raises(ValueError, match="agent_type"):
        validate_agent_patch(patch)
