"""Tests for Tau spec generation from AgentPatch.

These tests verify that the tau_spec_generator module correctly:
1. Generates valid Tau spec syntax.
2. Encodes all formal invariants.
3. Detects violations during generation.
4. Validates patches against spec constraints.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from idi.devkit.experimental.tau_spec_generator import (
    TauSpecConfig,
    TauSpecResult,
    agent_patch_to_tau_spec,
    qagent_patch_to_tau_spec,
    save_tau_spec,
    validate_patch_against_spec,
)
from idi.devkit.experimental.sape_q_patch import QAgentPatch, QPatchMeta
from idi.devkit.experimental.agent_patch import AgentPatch, AgentPatchMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_qpatch() -> QAgentPatch:
    """A QAgentPatch that satisfies all invariants."""
    meta = QPatchMeta(
        name="valid-patch",
        description="Valid test patch",
        version="1.0.0",
        tags=("qagent", "test"),
    )
    return QAgentPatch(
        identifier="valid-patch-1",
        num_price_bins=8,
        num_inventory_bins=8,  # 64 cells, under 256
        learning_rate=0.1,      # Under 0.5
        discount_factor=0.95,   # Above 0.9
        epsilon_start=0.5,      # Under 1.0
        epsilon_end=0.1,        # Under epsilon_start
        epsilon_decay_steps=1000,
        meta=meta,
    )


@pytest.fixture
def invalid_qpatch() -> QAgentPatch:
    """A QAgentPatch that violates multiple invariants."""
    meta = QPatchMeta(
        name="invalid-patch",
        description="Invalid test patch",
        version="1.0.0",
        tags=("qagent", "test"),
    )
    return QAgentPatch(
        identifier="invalid-patch-1",
        num_price_bins=20,
        num_inventory_bins=20,  # 400 cells, over 256
        learning_rate=0.8,      # Over 0.5
        discount_factor=0.5,    # Under 0.9
        epsilon_start=0.3,
        epsilon_end=0.5,        # Over epsilon_start (invalid)
        epsilon_decay_steps=1000,
        meta=meta,
    )


@pytest.fixture
def valid_agent_patch() -> AgentPatch:
    """A generic AgentPatch with qtable type."""
    meta = AgentPatchMeta(
        id="agent-patch-1",
        name="Agent Patch",
        description="Test agent patch",
        version="1.0.0",
        tags=("qagent",),
    )
    return AgentPatch(
        meta=meta,
        agent_type="qtable",
        payload={
            "num_price_bins": 10,
            "num_inventory_bins": 10,
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "epsilon_start": 0.5,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 1000,
        },
    )


# ---------------------------------------------------------------------------
# Spec Generation Tests
# ---------------------------------------------------------------------------

class TestQAgentPatchToTauSpec:
    """Tests for qagent_patch_to_tau_spec function."""

    def test_generates_spec_for_valid_patch(self, valid_qpatch: QAgentPatch) -> None:
        """Valid patch should generate a spec with no warnings."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert isinstance(result, TauSpecResult)
        assert result.patch_id == "valid-patch-1"
        assert len(result.warnings) == 0
        assert len(result.invariants) == 5

    def test_spec_contains_patch_parameters(self, valid_qpatch: QAgentPatch) -> None:
        """Generated spec should contain patch parameters."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_price_bins := 8" in result.spec_text
        assert "param_inventory_bins := 8" in result.spec_text
        assert "param_state_cells := 64" in result.spec_text
        assert "param_discount := 0.95" in result.spec_text
        assert "param_learning_rate := 0.1" in result.spec_text

    def test_spec_contains_invariant_definitions(self, valid_qpatch: QAgentPatch) -> None:
        """Generated spec should contain invariant definitions."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "invariant_state_size_bound" in result.spec_text
        assert "invariant_discount_bound" in result.spec_text
        assert "invariant_learning_rate_bound" in result.spec_text
        assert "invariant_exploration_bound" in result.spec_text
        assert "invariant_exploration_decay_valid" in result.spec_text

    def test_spec_contains_combined_validity(self, valid_qpatch: QAgentPatch) -> None:
        """Generated spec should define patch_valid."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "patch_valid :=" in result.spec_text

    def test_warns_on_invalid_patch(self, invalid_qpatch: QAgentPatch) -> None:
        """Invalid patch should generate warnings."""
        result = qagent_patch_to_tau_spec(invalid_qpatch)

        assert len(result.warnings) > 0
        # Should warn about state size
        assert any("state" in w.lower() for w in result.warnings)
        # Should warn about discount
        assert any("discount" in w.lower() for w in result.warnings)
        # Should warn about learning rate
        assert any("learning" in w.lower() for w in result.warnings)
        # Should warn about epsilon ordering
        assert any("epsilon" in w.lower() for w in result.warnings)

    def test_custom_config(self, valid_qpatch: QAgentPatch) -> None:
        """Custom config should be respected."""
        config = TauSpecConfig(
            max_state_cells=32,  # Lower than 64 cells in patch
            min_discount=0.99,   # Higher than 0.95 in patch
        )

        result = qagent_patch_to_tau_spec(valid_qpatch, config)

        # Should now have warnings due to stricter config
        assert len(result.warnings) >= 2
        assert "bound_max_state_cells := 32" in result.spec_text
        assert "bound_min_discount := 0.99" in result.spec_text


class TestAgentPatchToTauSpec:
    """Tests for agent_patch_to_tau_spec function."""

    def test_generates_spec_for_qtable_patch(self, valid_agent_patch: AgentPatch) -> None:
        """Should generate spec for qtable agent type."""
        result = agent_patch_to_tau_spec(valid_agent_patch)

        assert isinstance(result, TauSpecResult)
        assert result.patch_id == "agent-patch-1"

    def test_rejects_non_qtable_agent_type(self) -> None:
        """Should raise error for unsupported agent types."""
        meta = AgentPatchMeta(
            id="other-agent",
            name="Other Agent",
            description="",
            version="1.0.0",
        )
        patch = AgentPatch(meta=meta, agent_type="deep_q")

        with pytest.raises(ValueError, match="not supported"):
            agent_patch_to_tau_spec(patch)


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidatePatchAgainstSpec:
    """Tests for validate_patch_against_spec function."""

    def test_valid_patch_passes(self, valid_qpatch: QAgentPatch) -> None:
        """Valid patch should pass validation."""
        is_valid, violations = validate_patch_against_spec(valid_qpatch)

        assert is_valid is True
        assert len(violations) == 0

    def test_invalid_patch_fails(self, invalid_qpatch: QAgentPatch) -> None:
        """Invalid patch should fail with violations."""
        is_valid, violations = validate_patch_against_spec(invalid_qpatch)

        assert is_valid is False
        assert len(violations) >= 4  # At least 4 invariants violated

    def test_specific_invariant_violations(self, invalid_qpatch: QAgentPatch) -> None:
        """Should report specific invariant violations."""
        is_valid, violations = validate_patch_against_spec(invalid_qpatch)

        violation_text = " ".join(violations)
        assert "I1" in violation_text  # State size
        assert "I2" in violation_text  # Discount
        assert "I3" in violation_text  # Learning rate
        assert "I5" in violation_text  # Epsilon ordering

    def test_custom_config_validation(self, valid_qpatch: QAgentPatch) -> None:
        """Custom config should affect validation."""
        # With default config, should pass
        is_valid_default, _ = validate_patch_against_spec(valid_qpatch)
        assert is_valid_default is True

        # With stricter config, should fail
        strict_config = TauSpecConfig(max_state_cells=32)
        is_valid_strict, violations = validate_patch_against_spec(valid_qpatch, strict_config)
        assert is_valid_strict is False
        assert any("I1" in v for v in violations)


# ---------------------------------------------------------------------------
# File I/O Tests
# ---------------------------------------------------------------------------

class TestSaveTauSpec:
    """Tests for save_tau_spec function."""

    def test_saves_spec_to_file(self, valid_qpatch: QAgentPatch, tmp_path: Path) -> None:
        """Should save spec to specified path."""
        result = qagent_patch_to_tau_spec(valid_qpatch)
        output_path = tmp_path / "specs" / "test.tau"

        save_tau_spec(result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "valid-patch-1" in content
        assert "invariant_state_size_bound" in content

    def test_creates_parent_directories(self, valid_qpatch: QAgentPatch, tmp_path: Path) -> None:
        """Should create parent directories if needed."""
        result = qagent_patch_to_tau_spec(valid_qpatch)
        output_path = tmp_path / "deep" / "nested" / "path" / "spec.tau"

        save_tau_spec(result, output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Invariant Coverage Tests
# ---------------------------------------------------------------------------

class TestInvariantCoverage:
    """Tests verifying all invariants are properly encoded."""

    def test_i1_state_size_encoded(self, valid_qpatch: QAgentPatch) -> None:
        """I1: State size bound should be encoded."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_state_cells <= bound_max_state_cells" in result.spec_text

    def test_i2_discount_encoded(self, valid_qpatch: QAgentPatch) -> None:
        """I2: Discount bound should be encoded."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_discount >= bound_min_discount" in result.spec_text

    def test_i3_learning_rate_encoded(self, valid_qpatch: QAgentPatch) -> None:
        """I3: Learning rate bound should be encoded."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_learning_rate <= bound_max_learning_rate" in result.spec_text

    def test_i4_exploration_encoded(self, valid_qpatch: QAgentPatch) -> None:
        """I4: Exploration bound should be encoded."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_epsilon_start <= bound_max_exploration" in result.spec_text

    def test_i5_decay_validity_encoded(self, valid_qpatch: QAgentPatch) -> None:
        """I5: Exploration decay validity should be encoded."""
        result = qagent_patch_to_tau_spec(valid_qpatch)

        assert "param_epsilon_end <= param_epsilon_start" in result.spec_text
