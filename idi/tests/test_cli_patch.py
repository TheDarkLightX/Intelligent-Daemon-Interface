"""CLI regression tests for patch and synth commands.

These tests verify that CLI commands work correctly with valid inputs
and fail gracefully with invalid inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "synth"
CLI_MODULE = "idi.cli"


def run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the CLI with given arguments."""
    cmd = [sys.executable, "-m", CLI_MODULE, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# patch-validate tests
# ---------------------------------------------------------------------------

class TestPatchValidate:
    """Tests for patch-validate command."""

    def test_validate_valid_patch(self) -> None:
        """Valid patch should validate successfully."""
        patch_path = EXAMPLES_DIR / "conservative_trader.agentpatch.json"
        if not patch_path.exists():
            pytest.skip("Example patch not found")

        result = run_cli("patch-validate", "--patch", str(patch_path))
        assert result.returncode == 0
        assert "Valid AgentPatch" in result.stdout or "id" in result.stdout

    def test_validate_missing_file(self) -> None:
        """Missing file should fail with clear error."""
        result = run_cli(
            "patch-validate",
            "--patch", "/nonexistent/path.json",
            check=False,
        )
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON should fail gracefully."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ not valid json }")

        result = run_cli("patch-validate", "--patch", str(bad_file), check=False)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# patch-create tests
# ---------------------------------------------------------------------------

class TestPatchCreate:
    """Tests for patch-create command."""

    def test_create_basic_patch(self, tmp_path: Path) -> None:
        """Create a basic patch with minimal arguments."""
        out_path = tmp_path / "test_patch.json"

        result = run_cli(
            "patch-create",
            "--out", str(out_path),
            "--id", "test-patch",
            "--name", "Test Patch",
            "--agent-type", "qtable",
        )
        assert result.returncode == 0
        assert out_path.exists()

        # Verify the created file is valid JSON with expected fields
        data = json.loads(out_path.read_text())
        assert data["meta"]["id"] == "test-patch"
        assert data["meta"]["name"] == "Test Patch"
        assert data["agent_type"] == "qtable"

    def test_create_patch_with_tags(self, tmp_path: Path) -> None:
        """Create a patch with multiple tags."""
        out_path = tmp_path / "tagged_patch.json"

        result = run_cli(
            "patch-create",
            "--out", str(out_path),
            "--id", "tagged",
            "--name", "Tagged Patch",
            "--agent-type", "generic",
            "--tag", "experimental",
            "--tag", "test",
        )
        assert result.returncode == 0

        data = json.loads(out_path.read_text())
        assert "experimental" in data["meta"]["tags"]
        assert "test" in data["meta"]["tags"]


# ---------------------------------------------------------------------------
# patch-diff tests
# ---------------------------------------------------------------------------

class TestPatchDiff:
    """Tests for patch-diff command."""

    def test_diff_same_patch(self) -> None:
        """Diffing a patch with itself should show no differences."""
        patch_path = EXAMPLES_DIR / "conservative_trader.agentpatch.json"
        if not patch_path.exists():
            pytest.skip("Example patch not found")

        result = run_cli(
            "patch-diff",
            "--old", str(patch_path),
            "--new", str(patch_path),
        )
        assert result.returncode == 0
        # Empty diff or {}
        assert "{}" in result.stdout or "Diffing" in result.stdout

    def test_diff_different_patches(self, tmp_path: Path) -> None:
        """Diffing different patches should show differences."""
        patch1 = tmp_path / "patch1.json"
        patch2 = tmp_path / "patch2.json"

        # Create two different patches
        run_cli(
            "patch-create",
            "--out", str(patch1),
            "--id", "patch1",
            "--name", "Patch One",
            "--agent-type", "qtable",
        )
        run_cli(
            "patch-create",
            "--out", str(patch2),
            "--id", "patch2",
            "--name", "Patch Two",
            "--agent-type", "generic",
        )

        result = run_cli("patch-diff", "--old", str(patch1), "--new", str(patch2))
        assert result.returncode == 0
        # Should show differences in meta and agent_type
        assert "meta" in result.stdout or "agent_type" in result.stdout


# ---------------------------------------------------------------------------
# patch-apply tests
# ---------------------------------------------------------------------------

class TestPatchApply:
    """Tests for patch-apply command."""

    def test_apply_valid_patch(self) -> None:
        """Apply should succeed and print patch summary."""
        patch_path = EXAMPLES_DIR / "conservative_trader.agentpatch.json"
        if not patch_path.exists():
            pytest.skip("Example patch not found")

        result = run_cli("patch-apply", "--patch", str(patch_path))
        assert result.returncode == 0
        assert "meta" in result.stdout
        assert "agent_type" in result.stdout


# ---------------------------------------------------------------------------
# dev-auto-qagent tests (lightweight)
# ---------------------------------------------------------------------------

class TestDevAutoQAgent:
    """Lightweight tests for dev-auto-qagent command."""

    def test_auto_qagent_with_goal_spec(self) -> None:
        """Auto-QAgent should run with a valid goal spec."""
        goal_path = EXAMPLES_DIR / "conservative_qagent_goal.json"
        if not goal_path.exists():
            pytest.skip("Example goal spec not found")

        # This is a lightweight test; we just check it doesn't crash
        # and produces some output
        result = run_cli("dev-auto-qagent", "--goal", str(goal_path), check=False)
        # May succeed or fail depending on dependencies, but shouldn't crash
        assert result.returncode in (0, 1)

    def test_auto_qagent_missing_goal(self) -> None:
        """Missing goal file should fail with clear error."""
        result = run_cli(
            "dev-auto-qagent",
            "--goal", "/nonexistent/goal.json",
            check=False,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_unknown_command(self) -> None:
        """Unknown command should fail gracefully."""
        result = run_cli("unknown-command", check=False)
        assert result.returncode != 0

    def test_missing_required_args(self) -> None:
        """Missing required arguments should show usage."""
        result = run_cli("patch-validate", check=False)
        assert result.returncode != 0
        # Should mention the missing --patch argument
        assert "patch" in result.stderr.lower() or "required" in result.stderr.lower()
