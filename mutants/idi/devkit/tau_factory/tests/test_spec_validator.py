"""Tests for spec validator."""

import pytest
from idi.devkit.tau_factory.spec_validator import validate_tau_spec


def test_valid_spec_passes():
    """Valid spec should pass validation."""
    spec = """i0:sbf = in file("inputs/test.in").
o0:sbf = out file("outputs/test.out").
defs
r (
    (o0[t] = i0[t])
)
n
q"""
    is_valid, errors = validate_tau_spec(spec)
    assert is_valid, f"Valid spec failed: {errors}"


def test_missing_defs_fails():
    """Spec without defs should fail."""
    spec = """i0:sbf = in file("inputs/test.in").
o0:sbf = out file("outputs/test.out").
r (
    (o0[t] = i0[t])
)
n
q"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("defs" in e.lower() for e in errors)


def test_unbalanced_parens_fails():
    """Spec with unbalanced parentheses should fail."""
    spec = """i0:sbf = in file("inputs/test.in").
o0:sbf = out file("outputs/test.out").
defs
r (
    (o0[t] = i0[t]
)
n
q"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("parentheses" in e.lower() for e in errors)


def test_missing_q_fails():
    """Spec without q command should fail."""
    spec = """i0:sbf = in file("inputs/test.in").
o0:sbf = out file("outputs/test.out").
defs
r (
    (o0[t] = i0[t])
)
n"""
    is_valid, errors = validate_tau_spec(spec)
    assert not is_valid
    assert any("q" in e.lower() for e in errors)

