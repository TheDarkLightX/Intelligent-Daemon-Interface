"""Tests for spec validator.

Tests validate Tau CLI format:
- Input: varname:type = in file("path")
- Output: varname:type = out file("path")
- Run: r <wff> (formula directly)
- Stepping: Empty lines (not 'n' commands)
- Quit: q
"""

import pytest
from idi.devkit.tau_factory.spec_validator import validate_tau_spec


def test_valid_spec_passes():
    """Valid spec should pass validation."""
    # New Tau CLI format: r <wff> directly, empty lines for stepping
    spec = """i0:sbf = in file("inputs/test.in")
o0:sbf = out file("outputs/test.out")
r (o0[t] = i0[t])

q"""
    is_valid, errors = validate_tau_spec(spec)
    assert is_valid, f"Valid spec failed: {errors}"


def test_missing_run_command_fails():
    """Spec without run command should fail."""
    spec = """i0:sbf = in file("inputs/test.in")
o0:sbf = out file("outputs/test.out")

q"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("run" in e.lower() or "r " in e.lower() for e in errors)


def test_unbalanced_parens_fails():
    """Spec with unbalanced parentheses should fail."""
    spec = """i0:sbf = in file("inputs/test.in")
o0:sbf = out file("outputs/test.out")
r (o0[t] = i0[t]

q"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("parentheses" in e.lower() for e in errors)


def test_missing_q_fails():
    """Spec without q command should fail."""
    spec = """i0:sbf = in file("inputs/test.in")
o0:sbf = out file("outputs/test.out")
r (o0[t] = i0[t])
"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("q" in e.lower() for e in errors)

