"""
Tests for DIV (Deterministic Invariant Verification) algorithm.

Covers:
- Invariant formula construction and evaluation
- Canonical serialization and hashing
- Certificate generation
- Edge cases and security properties
"""

import pytest
import secrets
from typing import Any, Dict

from ..div import (
    DIVInvariantChecker,
    Invariant,
    InvariantFormula,
    InvariantAtom,
    AtomOperator,
    FormulaOperator,
    InvariantCertificate,
    AttributeSchema,
    AttributeType,
    compute_attributes_root,
    create_simple_invariant,
    create_threshold_invariant,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockAgentPack:
    """Mock agent pack for testing."""
    def __init__(self, reward: float = 0.5, risk: float = 0.2, complexity: float = 0.1):
        self.reward = reward
        self.risk = risk
        self.complexity = complexity
        self.pack_hash = secrets.token_bytes(32)


class MockGoalSpec:
    """Mock goal spec for testing."""
    def __init__(self, goal_id: str = "TEST_GOAL"):
        self.goal_id = goal_id
        self.thresholds = type("Thresholds", (), {
            "min_reward": 0.0,
            "max_risk": 1.0,
            "max_complexity": 1.0,
        })()


# =============================================================================
# InvariantAtom Tests
# =============================================================================

class TestInvariantAtom:
    """Tests for InvariantAtom."""
    
    def test_create_atom(self):
        """Test basic atom creation."""
        atom = InvariantAtom("reward", AtomOperator.GE, 0.0)
        assert atom.attribute == "reward"
        assert atom.operator == AtomOperator.GE
        assert atom.threshold == 0.0
    
    def test_empty_attribute_rejected(self):
        """Test that empty attribute name is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InvariantAtom("", AtomOperator.EQ, 0)
    
    def test_in_range_requires_tuple(self):
        """Test that IN_RANGE operator requires tuple."""
        with pytest.raises(ValueError, match="tuple"):
            InvariantAtom("value", AtomOperator.IN_RANGE, 5.0)
    
    def test_in_range_valid(self):
        """Test valid IN_RANGE atom."""
        atom = InvariantAtom("value", AtomOperator.IN_RANGE, (0.0, 1.0))
        assert atom.threshold == (0.0, 1.0)
    
    @pytest.mark.parametrize("op,threshold,value,expected", [
        (AtomOperator.EQ, 5, 5, True),
        (AtomOperator.EQ, 5, 6, False),
        (AtomOperator.NE, 5, 6, True),
        (AtomOperator.NE, 5, 5, False),
        (AtomOperator.LT, 5, 4, True),
        (AtomOperator.LT, 5, 5, False),
        (AtomOperator.LE, 5, 5, True),
        (AtomOperator.LE, 5, 6, False),
        (AtomOperator.GT, 5, 6, True),
        (AtomOperator.GT, 5, 5, False),
        (AtomOperator.GE, 5, 5, True),
        (AtomOperator.GE, 5, 4, False),
    ])
    def test_evaluate_comparison(self, op, threshold, value, expected):
        """Test comparison operator evaluation."""
        atom = InvariantAtom("x", op, threshold)
        assert atom.evaluate(value) == expected
    
    def test_evaluate_in_range(self):
        """Test IN_RANGE evaluation."""
        atom = InvariantAtom("x", AtomOperator.IN_RANGE, (0.0, 1.0))
        assert atom.evaluate(0.5) is True
        assert atom.evaluate(0.0) is True
        assert atom.evaluate(1.0) is True
        assert atom.evaluate(-0.1) is False
        assert atom.evaluate(1.1) is False
    
    def test_canonical_bytes_deterministic(self):
        """Test that canonical bytes are deterministic."""
        atom1 = InvariantAtom("reward", AtomOperator.GE, 0.5)
        atom2 = InvariantAtom("reward", AtomOperator.GE, 0.5)
        assert atom1.canonical_bytes() == atom2.canonical_bytes()
    
    def test_canonical_bytes_different_atoms(self):
        """Test that different atoms have different canonical bytes."""
        atom1 = InvariantAtom("reward", AtomOperator.GE, 0.5)
        atom2 = InvariantAtom("reward", AtomOperator.GE, 0.6)
        atom3 = InvariantAtom("risk", AtomOperator.GE, 0.5)
        
        assert atom1.canonical_bytes() != atom2.canonical_bytes()
        assert atom1.canonical_bytes() != atom3.canonical_bytes()


# =============================================================================
# InvariantFormula Tests
# =============================================================================

class TestInvariantFormula:
    """Tests for InvariantFormula."""
    
    def test_atom_formula(self):
        """Test creating leaf formula from atom."""
        atom = InvariantAtom("x", AtomOperator.GT, 0)
        formula = InvariantFormula.atom(atom)
        assert formula.operator is None
        assert len(formula.children) == 1
    
    def test_and_formula(self):
        """Test AND formula."""
        f1 = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        f2 = InvariantFormula.atom(InvariantAtom("y", AtomOperator.LT, 10))
        formula = InvariantFormula.and_(f1, f2)
        
        assert formula.operator == FormulaOperator.AND
        assert len(formula.children) == 2
    
    def test_or_formula(self):
        """Test OR formula."""
        f1 = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        f2 = InvariantFormula.atom(InvariantAtom("y", AtomOperator.LT, 10))
        formula = InvariantFormula.or_(f1, f2)
        
        assert formula.operator == FormulaOperator.OR
    
    def test_not_formula(self):
        """Test NOT formula."""
        f1 = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        formula = InvariantFormula.not_(f1)
        
        assert formula.operator == FormulaOperator.NOT
    
    def test_implies_formula(self):
        """Test IMPLIES formula."""
        f1 = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        f2 = InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0))
        formula = InvariantFormula.implies(f1, f2)
        
        assert formula.operator == FormulaOperator.IMPLIES
    
    def test_evaluate_atom(self):
        """Test evaluating atom formula."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        
        result, _ = formula.evaluate({"x": 5})
        assert result is True
        
        result, _ = formula.evaluate({"x": -1})
        assert result is False
    
    def test_evaluate_missing_attribute(self):
        """Test evaluation with missing attribute returns False."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        result, reason = formula.evaluate({})
        
        assert result is False
        assert "missing" in reason.lower()
    
    def test_evaluate_and(self):
        """Test AND evaluation."""
        formula = InvariantFormula.and_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0)),
        )
        
        assert formula.evaluate({"x": 5, "y": 5})[0] is True
        assert formula.evaluate({"x": 5, "y": -1})[0] is False
        assert formula.evaluate({"x": -1, "y": 5})[0] is False
    
    def test_evaluate_or(self):
        """Test OR evaluation."""
        formula = InvariantFormula.or_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0)),
        )
        
        assert formula.evaluate({"x": 5, "y": -1})[0] is True
        assert formula.evaluate({"x": -1, "y": 5})[0] is True
        assert formula.evaluate({"x": -1, "y": -1})[0] is False
    
    def test_evaluate_not(self):
        """Test NOT evaluation."""
        formula = InvariantFormula.not_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        )
        
        assert formula.evaluate({"x": 5})[0] is False
        assert formula.evaluate({"x": -1})[0] is True
    
    def test_evaluate_implies(self):
        """Test IMPLIES evaluation (A -> B)."""
        formula = InvariantFormula.implies(
            InvariantFormula.atom(InvariantAtom("a", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("b", AtomOperator.GT, 0)),
        )
        
        # A false -> vacuously true
        assert formula.evaluate({"a": -1, "b": -1})[0] is True
        # A true, B true -> true
        assert formula.evaluate({"a": 5, "b": 5})[0] is True
        # A true, B false -> false
        assert formula.evaluate({"a": 5, "b": -1})[0] is False
    
    def test_complexity(self):
        """Test complexity calculation."""
        atom_formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        assert atom_formula.complexity() == 1
        
        and_formula = InvariantFormula.and_(atom_formula, atom_formula)
        assert and_formula.complexity() == 3  # 1 (AND) + 1 + 1
    
    def test_get_required_attributes(self):
        """Test extracting required attributes."""
        formula = InvariantFormula.and_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("y", AtomOperator.LT, 10)),
        )
        
        attrs = formula.get_required_attributes()
        assert attrs == frozenset(["x", "y"])
    
    def test_canonical_hash_deterministic(self):
        """Test that canonical hash is deterministic."""
        formula1 = InvariantFormula.and_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0)),
        )
        formula2 = InvariantFormula.and_(
            InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0)),
            InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0)),
        )
        
        assert formula1.canonical_hash() == formula2.canonical_hash()
    
    def test_canonical_hash_order_independent_for_and(self):
        """Test that AND children order doesn't affect hash."""
        f1 = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        f2 = InvariantFormula.atom(InvariantAtom("y", AtomOperator.GT, 0))
        
        formula1 = InvariantFormula.and_(f1, f2)
        formula2 = InvariantFormula.and_(f2, f1)
        
        assert formula1.canonical_hash() == formula2.canonical_hash()


# =============================================================================
# Invariant Tests
# =============================================================================

class TestInvariant:
    """Tests for Invariant."""
    
    def test_create_invariant(self):
        """Test basic invariant creation."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        
        invariant = Invariant(
            id=secrets.token_bytes(32),
            version=1,
            formula=formula,
            attribute_schema=schema,
        )
        
        assert invariant.version == 1
    
    def test_invalid_id_length(self):
        """Test that invalid ID length is rejected."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        
        with pytest.raises(ValueError, match="32 bytes"):
            Invariant(
                id=b"short",
                version=1,
                formula=formula,
                attribute_schema=schema,
            )
    
    def test_missing_schema_attribute(self):
        """Test that missing schema attribute is detected."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        schema = (AttributeSchema("y", AttributeType.FLOAT),)  # Wrong name
        
        with pytest.raises(ValueError, match="undefined attributes"):
            Invariant(
                id=secrets.token_bytes(32),
                version=1,
                formula=formula,
                attribute_schema=schema,
            )
    
    def test_canonical_hash_deterministic(self):
        """Test that invariant hash is deterministic."""
        formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        inv_id = secrets.token_bytes(32)
        
        inv1 = Invariant(id=inv_id, version=1, formula=formula, attribute_schema=schema)
        inv2 = Invariant(id=inv_id, version=1, formula=formula, attribute_schema=schema)
        
        assert inv1.canonical_hash() == inv2.canonical_hash()
    
    def test_requires_snark(self):
        """Test SNARK threshold detection."""
        simple_formula = InvariantFormula.atom(InvariantAtom("x", AtomOperator.GT, 0))
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        
        simple_inv = Invariant(
            id=secrets.token_bytes(32),
            version=1,
            formula=simple_formula,
            attribute_schema=schema,
        )
        
        assert simple_inv.requires_snark() is False


# =============================================================================
# DIVInvariantChecker Tests
# =============================================================================

class TestDIVInvariantChecker:
    """Tests for DIVInvariantChecker."""
    
    def test_no_invariants_passes(self):
        """Test that no configured invariants means pass."""
        checker = DIVInvariantChecker()
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        passed, reason = checker.check(agent, goal)
        assert passed is True
        assert "no invariants" in reason.lower()
    
    def test_register_and_check(self):
        """Test registering and checking invariant."""
        checker = DIVInvariantChecker()
        
        inv = create_simple_invariant(
            "min_reward",
            "reward",
            AtomOperator.GE,
            0.0,
        )
        checker.register_invariant(inv, goal_ids=["TEST_GOAL"])
        
        agent = MockAgentPack(reward=0.5)
        goal = MockGoalSpec(goal_id="TEST_GOAL")
        
        passed, reason = checker.check(agent, goal)
        assert passed is True
    
    def test_failing_invariant(self):
        """Test that failing invariant is detected."""
        checker = DIVInvariantChecker()
        
        inv = create_simple_invariant(
            "min_reward",
            "reward",
            AtomOperator.GE,
            0.9,  # High threshold
        )
        checker.register_invariant(inv, goal_ids=["TEST_GOAL"])
        
        agent = MockAgentPack(reward=0.5)  # Below threshold
        goal = MockGoalSpec(goal_id="TEST_GOAL")
        
        passed, reason = checker.check(agent, goal)
        assert passed is False
        assert "failed" in reason.lower()
    
    def test_threshold_invariant(self):
        """Test composite threshold invariant."""
        checker = DIVInvariantChecker()
        
        inv = create_threshold_invariant(
            min_reward=0.0,
            max_risk=0.5,
            max_complexity=0.5,
        )
        checker.register_invariant(inv, goal_ids=["TEST_GOAL"])
        
        # Passing case
        agent1 = MockAgentPack(reward=0.5, risk=0.2, complexity=0.1)
        goal = MockGoalSpec(goal_id="TEST_GOAL")
        
        passed, _ = checker.check(agent1, goal)
        assert passed is True
        
        # Failing case - risk too high
        agent2 = MockAgentPack(reward=0.5, risk=0.8, complexity=0.1)
        passed, reason = checker.check(agent2, goal)
        assert passed is False
    
    def test_check_with_certificate(self):
        """Test certificate generation."""
        checker = DIVInvariantChecker()
        
        inv = create_simple_invariant(
            "min_reward",
            "reward",
            AtomOperator.GE,
            0.0,
        )
        checker.register_invariant(inv, goal_ids=["TEST_GOAL"])
        
        agent = MockAgentPack(reward=0.5)
        goal = MockGoalSpec(goal_id="TEST_GOAL")
        contribution_hash = secrets.token_bytes(32)
        
        passed, reason, cert = checker.check_with_certificate(agent, goal, contribution_hash)
        
        assert passed is True
        assert cert is not None
        assert cert.result is True
        assert cert.contribution_hash == contribution_hash


# =============================================================================
# Merkle Tree Tests
# =============================================================================

class TestMerkleTree:
    """Tests for attribute Merkle tree."""
    
    def test_empty_attributes(self):
        """Test root for empty attributes."""
        root = compute_attributes_root({}, ())
        assert len(root) == 32
    
    def test_single_attribute(self):
        """Test root for single attribute."""
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        root = compute_attributes_root({"x": 1.0}, schema)
        assert len(root) == 32
    
    def test_deterministic(self):
        """Test that root is deterministic."""
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        root1 = compute_attributes_root({"x": 1.0}, schema)
        root2 = compute_attributes_root({"x": 1.0}, schema)
        assert root1 == root2
    
    def test_different_values_different_root(self):
        """Test that different values produce different roots."""
        schema = (AttributeSchema("x", AttributeType.FLOAT),)
        root1 = compute_attributes_root({"x": 1.0}, schema)
        root2 = compute_attributes_root({"x": 2.0}, schema)
        assert root1 != root2
    
    def test_order_independent(self):
        """Test that attribute order doesn't affect root."""
        schema = (
            AttributeSchema("a", AttributeType.FLOAT),
            AttributeSchema("b", AttributeType.FLOAT),
        )
        root1 = compute_attributes_root({"a": 1.0, "b": 2.0}, schema)
        root2 = compute_attributes_root({"b": 2.0, "a": 1.0}, schema)
        assert root1 == root2


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_simple_invariant(self):
        """Test simple invariant creation."""
        inv = create_simple_invariant(
            "test",
            "reward",
            AtomOperator.GE,
            0.5,
        )
        
        assert inv.version == 1
        assert len(inv.id) == 32
    
    def test_create_threshold_invariant_single(self):
        """Test threshold invariant with single constraint."""
        inv = create_threshold_invariant(min_reward=0.5)
        
        assert inv.complexity() == 1
    
    def test_create_threshold_invariant_multiple(self):
        """Test threshold invariant with multiple constraints."""
        inv = create_threshold_invariant(
            min_reward=0.0,
            max_risk=0.5,
            max_complexity=1.0,
        )
        
        # AND + 3 atoms
        assert inv.complexity() == 4
    
    def test_create_threshold_invariant_empty(self):
        """Test that empty threshold invariant is rejected."""
        with pytest.raises(ValueError, match="At least one"):
            create_threshold_invariant()


# =============================================================================
# Property-Based Tests
# =============================================================================

try:
    from hypothesis import given, strategies as st
    
    class TestPropertyBased:
        """Property-based tests using Hypothesis."""
        
        @given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
        def test_ge_consistent(self, threshold):
            """Test GE operator is consistent."""
            atom = InvariantAtom("x", AtomOperator.GE, threshold)
            
            # Value == threshold should pass
            assert atom.evaluate(threshold) is True
            # Value > threshold should pass
            if threshold < 1e6:
                assert atom.evaluate(threshold + 1) is True
            # Value < threshold should fail
            if threshold > -1e6:
                assert atom.evaluate(threshold - 1) is False
        
        @given(st.floats(min_value=0, max_value=1, allow_nan=False))
        def test_in_range_boundary(self, value):
            """Test IN_RANGE includes boundaries."""
            atom = InvariantAtom("x", AtomOperator.IN_RANGE, (0.0, 1.0))
            assert atom.evaluate(value) is True

except ImportError:
    pass  # Hypothesis not available
