"""Fuzz tests for AutoQAgentGoalSpec.from_dict.

These tests verify that the goal spec parser handles malformed,
adversarial, or unexpected JSON without crashing. It should either
parse successfully and validate, or raise a clear error.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
        def one_of(*args, **kwargs):
            return None

        @staticmethod
        def none():
            return None

        @staticmethod
        def booleans():
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def floats(*args, **kwargs):
            return None

        @staticmethod
        def text(*args, **kwargs):
            return None

        @staticmethod
        def dictionaries(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

        @staticmethod
        def fixed_dictionaries(*args, **kwargs):
            return None

        @staticmethod
        def just(*args, **kwargs):
            return None


from idi.devkit.experimental.auto_qagent import (
    AutoQAgentGoalSpec,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for fuzz testing
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    # Arbitrary JSON-like values
    json_primitives = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=64),
    )

    # Recursive JSON structure (limited depth)
    json_value = st.recursive(
        json_primitives,
        lambda children: st.one_of(
            st.lists(children, max_size=5),
            st.dictionaries(st.text(max_size=16), children, max_size=5),
        ),
        max_leaves=20,
    )

    # Random dicts that might look like goal specs
    random_dict = st.dictionaries(
        keys=st.text(max_size=32),
        values=json_value,
        max_size=15,
    )

    # Semi-valid goal spec structure with random values
    semi_valid_goal = st.fixed_dictionaries(
        {
            "agent_family": st.one_of(st.text(max_size=16), st.none(), st.integers()),
            "profiles": st.one_of(st.lists(st.text(max_size=8), max_size=5), st.none()),
            "packs": st.one_of(
                st.fixed_dictionaries({
                    "include": st.lists(st.text(max_size=16), max_size=3),
                }),
                st.none(),
                st.text(max_size=8),
            ),
            "objectives": st.one_of(
                st.lists(
                    st.fixed_dictionaries({
                        "id": st.text(max_size=16),
                        "direction": st.one_of(
                            st.just("maximize"),
                            st.just("minimize"),
                            st.text(max_size=8),
                        ),
                    }),
                    max_size=3,
                ),
                st.none(),
            ),
            "training": st.one_of(
                st.fixed_dictionaries({
                    "envs": st.lists(
                        st.fixed_dictionaries({
                            "id": st.text(max_size=8),
                            "weight": st.one_of(st.floats(allow_nan=False), st.integers()),
                        }),
                        max_size=3,
                    ),
                    "budget": st.one_of(
                        st.fixed_dictionaries({
                            "max_agents": st.integers(min_value=-100, max_value=1000),
                            "max_episodes_per_agent": st.integers(min_value=-100, max_value=10000),
                        }),
                        st.none(),
                    ),
                }),
                st.none(),
            ),
            "eval_mode": st.one_of(
                st.just("synthetic"),
                st.just("real"),
                st.text(max_size=16),
                st.none(),
            ),
            "outputs": st.one_of(
                st.fixed_dictionaries({
                    "num_final_agents": st.integers(min_value=-10, max_value=100),
                }),
                st.none(),
            ),
        },
        optional={
            "extra_field": json_value,
        },
    )


# ---------------------------------------------------------------------------
# Fuzz tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(data=random_dict)
@settings(max_examples=200)
def test_goal_spec_from_dict_handles_random(data: Dict[str, Any]) -> None:
    """from_dict should not crash on arbitrary dicts."""
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        # If it succeeded, basic fields should be accessible
        assert isinstance(spec.agent_family, str)
        assert isinstance(spec.profiles, tuple)
    except (ValueError, TypeError, KeyError) as e:
        # Expected for malformed input
        pass


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(data=semi_valid_goal)
@settings(max_examples=200)
def test_goal_spec_from_dict_semi_valid(data: Dict[str, Any]) -> None:
    """from_dict should handle semi-valid structures gracefully."""
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        # Verify structure is populated
        assert spec.agent_family is not None
        assert spec.training is not None
        assert spec.outputs is not None
    except (ValueError, TypeError, KeyError):
        # Expected for invalid values
        pass


# ---------------------------------------------------------------------------
# Specific edge case tests (non-hypothesis)
# ---------------------------------------------------------------------------

def test_goal_spec_empty_dict() -> None:
    """Empty dict should produce a default-ish spec or raise."""
    try:
        spec = AutoQAgentGoalSpec.from_dict({})
        # If it succeeds, should have defaults
        assert spec.agent_family == ""
    except (ValueError, TypeError, KeyError):
        pass


def test_goal_spec_invalid_eval_mode_raises() -> None:
    """Invalid eval_mode should raise ValueError."""
    data = {
        "agent_family": "qagent",
        "profiles": [],
        "eval_mode": "invalid_mode",
    }
    with pytest.raises(ValueError, match="eval_mode"):
        AutoQAgentGoalSpec.from_dict(data)


def test_goal_spec_nested_none_values() -> None:
    """None values in nested structures should be handled."""
    data = {
        "agent_family": "qagent",
        "profiles": None,
        "packs": None,
        "objectives": None,
        "training": None,
        "eval_mode": "synthetic",
        "outputs": None,
    }
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        assert spec.profiles == ()
    except (ValueError, TypeError, KeyError):
        pass


def test_goal_spec_negative_budget_values() -> None:
    """Negative budget values should be handled without crashing."""
    data = {
        "agent_family": "qagent",
        "training": {
            "envs": [],
            "budget": {
                "max_agents": -1,
                "max_episodes_per_agent": -100,
            },
        },
        "eval_mode": "synthetic",
    }
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        # Should parse, even if values are nonsensical
        assert spec.training.budget.max_agents == -1
    except (ValueError, TypeError, KeyError):
        pass


def test_goal_spec_extremely_large_values() -> None:
    """Very large values should not cause overflow or hangs."""
    data = {
        "agent_family": "qagent",
        "training": {
            "envs": [],
            "budget": {
                "max_agents": 10**15,
                "max_episodes_per_agent": 10**15,
            },
        },
        "eval_mode": "synthetic",
    }
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        # Should parse; actual clamping happens at runtime
        assert spec.training.budget.max_agents == 10**15
    except (ValueError, TypeError, KeyError, OverflowError):
        pass


def test_goal_spec_unicode_strings() -> None:
    """Unicode strings should be handled."""
    data = {
        "agent_family": "qagent_日本語",
        "profiles": ["保守的"],
        "eval_mode": "synthetic",
    }
    try:
        spec = AutoQAgentGoalSpec.from_dict(data)
        assert "日本語" in spec.agent_family
    except (ValueError, TypeError, KeyError):
        pass
