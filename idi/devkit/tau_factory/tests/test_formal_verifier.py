"""Tests for the Formal Verification Tool Suite.

Verifies that the formal verifier correctly validates Tau specifications
against their documented formal claims.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from idi.devkit.tau_factory.formal_verifier import (
    TauSpecVerifier,
    TauFormulaParser,
    BooleanEvaluator,
    TruthTableGenerator,
    FormalClaimRegistry,
    CompletenessChecker,
    VerificationStatus,
    ParsedFormula,
)


class TestBooleanEvaluator:
    """Test Boolean formula evaluation."""
    
    def test_and_gate(self) -> None:
        """Test AND gate evaluation."""
        eval = BooleanEvaluator()
        assert eval.evaluate("i0[t] & i1[t]", {"i0": 0, "i1": 0}) == 0
        assert eval.evaluate("i0[t] & i1[t]", {"i0": 0, "i1": 1}) == 0
        assert eval.evaluate("i0[t] & i1[t]", {"i0": 1, "i1": 0}) == 0
        assert eval.evaluate("i0[t] & i1[t]", {"i0": 1, "i1": 1}) == 1
    
    def test_or_gate(self) -> None:
        """Test OR gate evaluation."""
        eval = BooleanEvaluator()
        assert eval.evaluate("i0[t] | i1[t]", {"i0": 0, "i1": 0}) == 0
        assert eval.evaluate("i0[t] | i1[t]", {"i0": 0, "i1": 1}) == 1
        assert eval.evaluate("i0[t] | i1[t]", {"i0": 1, "i1": 0}) == 1
        assert eval.evaluate("i0[t] | i1[t]", {"i0": 1, "i1": 1}) == 1
    
    def test_xor_gate(self) -> None:
        """Test XOR gate evaluation."""
        eval = BooleanEvaluator()
        assert eval.evaluate("i0[t] ^ i1[t]", {"i0": 0, "i1": 0}) == 0
        assert eval.evaluate("i0[t] ^ i1[t]", {"i0": 0, "i1": 1}) == 1
        assert eval.evaluate("i0[t] ^ i1[t]", {"i0": 1, "i1": 0}) == 1
        assert eval.evaluate("i0[t] ^ i1[t]", {"i0": 1, "i1": 1}) == 0
    
    def test_not_operator(self) -> None:
        """Test NOT (') operator."""
        eval = BooleanEvaluator()
        assert eval.evaluate("i0[t]'", {"i0": 0}) == 1
        assert eval.evaluate("i0[t]'", {"i0": 1}) == 0
    
    def test_nand_gate(self) -> None:
        """Test NAND gate: (a & b)'"""
        eval = BooleanEvaluator()
        assert eval.evaluate("(i0[t] & i1[t])'", {"i0": 0, "i1": 0}) == 1
        assert eval.evaluate("(i0[t] & i1[t])'", {"i0": 0, "i1": 1}) == 1
        assert eval.evaluate("(i0[t] & i1[t])'", {"i0": 1, "i1": 0}) == 1
        assert eval.evaluate("(i0[t] & i1[t])'", {"i0": 1, "i1": 1}) == 0
    
    def test_implication(self) -> None:
        """Test implication: a' | b (equivalent to a -> b)."""
        eval = BooleanEvaluator()
        # a=0, b=0: T (vacuously true)
        assert eval.evaluate("i0[t]' | i1[t]", {"i0": 0, "i1": 0}) == 1
        # a=0, b=1: T
        assert eval.evaluate("i0[t]' | i1[t]", {"i0": 0, "i1": 1}) == 1
        # a=1, b=0: F (antecedent true, consequent false)
        assert eval.evaluate("i0[t]' | i1[t]", {"i0": 1, "i1": 0}) == 0
        # a=1, b=1: T
        assert eval.evaluate("i0[t]' | i1[t]", {"i0": 1, "i1": 1}) == 1
    
    def test_xnor_equality(self) -> None:
        """Test XNOR (equality): (a ^ b)'"""
        eval = BooleanEvaluator()
        assert eval.evaluate("(i0[t] ^ i1[t])'", {"i0": 0, "i1": 0}) == 1
        assert eval.evaluate("(i0[t] ^ i1[t])'", {"i0": 0, "i1": 1}) == 0
        assert eval.evaluate("(i0[t] ^ i1[t])'", {"i0": 1, "i1": 0}) == 0
        assert eval.evaluate("(i0[t] ^ i1[t])'", {"i0": 1, "i1": 1}) == 1
    
    def test_majority_voting(self) -> None:
        """Test 2-of-3 majority: (a & b) | (a & c) | (b & c)."""
        eval = BooleanEvaluator()
        expr = "(i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t])"
        
        # All combinations
        assert eval.evaluate(expr, {"i0": 0, "i1": 0, "i2": 0}) == 0
        assert eval.evaluate(expr, {"i0": 0, "i1": 0, "i2": 1}) == 0
        assert eval.evaluate(expr, {"i0": 0, "i1": 1, "i2": 0}) == 0
        assert eval.evaluate(expr, {"i0": 0, "i1": 1, "i2": 1}) == 1  # 2 votes
        assert eval.evaluate(expr, {"i0": 1, "i1": 0, "i2": 0}) == 0
        assert eval.evaluate(expr, {"i0": 1, "i1": 0, "i2": 1}) == 1  # 2 votes
        assert eval.evaluate(expr, {"i0": 1, "i1": 1, "i2": 0}) == 1  # 2 votes
        assert eval.evaluate(expr, {"i0": 1, "i1": 1, "i2": 1}) == 1  # 3 votes
    
    def test_and_not(self) -> None:
        """Test AND-NOT: a & b'"""
        eval = BooleanEvaluator()
        assert eval.evaluate("i0[t] & i1[t]'", {"i0": 0, "i1": 0}) == 0
        assert eval.evaluate("i0[t] & i1[t]'", {"i0": 0, "i1": 1}) == 0
        assert eval.evaluate("i0[t] & i1[t]'", {"i0": 1, "i1": 0}) == 1
        assert eval.evaluate("i0[t] & i1[t]'", {"i0": 1, "i1": 1}) == 0


class TestTauFormulaParser:
    """Test Tau spec parsing."""
    
    def test_parse_simple_spec(self) -> None:
        """Parse a simple Tau spec."""
        spec = '''
i0:sbf = in file("inputs/a.in")
i1:sbf = in file("inputs/b.in")
o0:sbf = out file("outputs/result.out")
r o0[t] = i0[t] & i1[t]
q
'''
        parser = TauFormulaParser()
        formulas = parser.parse_spec(spec)
        
        assert len(formulas) == 1
        assert formulas[0].inputs == ["i0", "i1"]
        assert formulas[0].output == "o0"
        assert "i0[t] & i1[t]" in formulas[0].expression
    
    def test_parse_majority_spec(self) -> None:
        """Parse majority voting spec."""
        spec = '''
i0:sbf = in file("inputs/v1.in")
i1:sbf = in file("inputs/v2.in")
i2:sbf = in file("inputs/v3.in")
o0:sbf = out file("outputs/decision.out")
r (o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t]))
q
'''
        parser = TauFormulaParser()
        formulas = parser.parse_spec(spec)
        
        assert len(formulas) == 1
        assert formulas[0].inputs == ["i0", "i1", "i2"]
        assert formulas[0].output == "o0"


class TestTruthTableGenerator:
    """Test truth table generation."""
    
    def test_generate_1_input(self) -> None:
        """Generate truth table for 1 input."""
        gen = TruthTableGenerator()
        table = gen.generate(["a"])
        
        assert len(table) == 2
        assert {"a": 0} in table
        assert {"a": 1} in table
    
    def test_generate_2_inputs(self) -> None:
        """Generate truth table for 2 inputs."""
        gen = TruthTableGenerator()
        table = gen.generate(["a", "b"])
        
        assert len(table) == 4
        assert {"a": 0, "b": 0} in table
        assert {"a": 0, "b": 1} in table
        assert {"a": 1, "b": 0} in table
        assert {"a": 1, "b": 1} in table
    
    def test_generate_3_inputs(self) -> None:
        """Generate truth table for 3 inputs."""
        gen = TruthTableGenerator()
        table = gen.generate(["a", "b", "c"])
        
        assert len(table) == 8
    
    def test_max_inputs_limit(self) -> None:
        """Verify max inputs limit is enforced."""
        gen = TruthTableGenerator(max_inputs=3)
        
        with pytest.raises(ValueError, match="Too many inputs"):
            gen.generate(["a", "b", "c", "d"])


class TestFormalClaimRegistry:
    """Test formal claim registry."""
    
    def test_builtin_claims_registered(self) -> None:
        """Verify built-in claims are registered."""
        registry = FormalClaimRegistry()
        
        assert registry.get("mutual_exclusion") is not None
        assert registry.get("no_starvation") is not None
        assert registry.get("consensus_check") is not None
        assert registry.get("majority_2of3") is not None
    
    def test_claim_oracles_are_correct(self) -> None:
        """Verify claim oracles compute correct values."""
        registry = FormalClaimRegistry()
        
        # Mutual exclusion (NAND)
        me = registry.get("mutual_exclusion")
        assert me is not None
        assert me.oracle(0, 0) == 1
        assert me.oracle(0, 1) == 1
        assert me.oracle(1, 0) == 1
        assert me.oracle(1, 1) == 0
        
        # No starvation (implication)
        ns = registry.get("no_starvation")
        assert ns is not None
        assert ns.oracle(0, 0) == 1  # no request = fair
        assert ns.oracle(0, 1) == 1  # no request = fair
        assert ns.oracle(1, 0) == 0  # request but no grant = unfair
        assert ns.oracle(1, 1) == 1  # request and grant = fair
        
        # Majority 2-of-3
        maj = registry.get("majority_2of3")
        assert maj is not None
        assert maj.oracle(0, 0, 0) == 0
        assert maj.oracle(0, 1, 1) == 1
        assert maj.oracle(1, 1, 0) == 1
        assert maj.oracle(1, 1, 1) == 1


class TestTauSpecVerifier:
    """Test the main verification engine."""
    
    def test_verify_nand_formula(self) -> None:
        """Verify NAND formula against mutual_exclusion claim."""
        verifier = TauSpecVerifier()
        
        spec = '''
i0:sbf = in file("inputs/a.in")
i1:sbf = in file("inputs/b.in")
o0:sbf = out file("outputs/safe.out")
r o0[t] = (i0[t] & i1[t])'
q
'''
        result = verifier.verify_spec_content(spec, "mutual_exclusion", "test_nand")
        
        assert result.status == VerificationStatus.PASSED
        assert result.total_combinations == 4
        assert result.passed_combinations == 4
        assert result.failed_combinations == 0
    
    def test_verify_implication_formula(self) -> None:
        """Verify implication formula against no_starvation claim."""
        verifier = TauSpecVerifier()
        
        spec = '''
i0:sbf = in file("inputs/request.in")
i1:sbf = in file("inputs/grant.in")
o0:sbf = out file("outputs/fair.out")
r o0[t] = i0[t]' | i1[t]
q
'''
        result = verifier.verify_spec_content(spec, "no_starvation", "test_impl")
        
        assert result.status == VerificationStatus.PASSED
        assert result.total_combinations == 4
        assert result.passed_combinations == 4
    
    def test_verify_majority_formula(self) -> None:
        """Verify majority voting formula."""
        verifier = TauSpecVerifier()
        
        spec = '''
i0:sbf = in file("inputs/v1.in")
i1:sbf = in file("inputs/v2.in")
i2:sbf = in file("inputs/v3.in")
o0:sbf = out file("outputs/decision.out")
r (o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t]))
q
'''
        result = verifier.verify_spec_content(spec, "majority_2of3", "test_maj")
        
        assert result.status == VerificationStatus.PASSED
        assert result.total_combinations == 8
        assert result.passed_combinations == 8
    
    def test_detect_incorrect_formula(self) -> None:
        """Detect when formula doesn't match claim."""
        verifier = TauSpecVerifier()
        
        # AND gate trying to claim mutual exclusion (NAND)
        spec = '''
i0:sbf = in file("inputs/a.in")
i1:sbf = in file("inputs/b.in")
o0:sbf = out file("outputs/result.out")
r o0[t] = i0[t] & i1[t]
q
'''
        result = verifier.verify_spec_content(spec, "mutual_exclusion", "test_wrong")
        
        assert result.status == VerificationStatus.FAILED
        assert result.failed_combinations > 0
        assert len(result.counterexamples) > 0
    
    def test_list_claims(self) -> None:
        """List all registered claims."""
        verifier = TauSpecVerifier()
        claims = verifier.list_claims()
        
        assert "mutual_exclusion" in claims
        assert "no_starvation" in claims
        assert "consensus_check" in claims
        assert len(claims) >= 14  # At least the documented patterns
    
    def test_get_claim_info(self) -> None:
        """Get claim information."""
        verifier = TauSpecVerifier()
        info = verifier.get_claim_info("mutual_exclusion")
        
        assert info is not None
        assert info["name"] == "mutual_exclusion"
        assert "ltl" in info
        assert "description" in info


class TestCompletenessChecker:
    """Test completeness checking."""
    
    def test_complete_formula(self) -> None:
        """Verify a complete formula."""
        checker = CompletenessChecker()
        formula = ParsedFormula(
            raw="o0[t] = i0[t] & i1[t]",
            inputs=["i0", "i1"],
            output="o0",
            expression="i0[t] & i1[t]",
        )
        
        is_complete, failures = checker.check_completeness(formula)
        
        assert is_complete
        assert len(failures) == 0


class TestVerifyGoldenSmoke:
    """Verify golden smoke test fixtures."""
    
    @pytest.fixture
    def smoke_dir(self) -> Path:
        """Get path to golden smoke fixtures."""
        return Path(__file__).parent / "fixtures" / "golden_smoke"
    
    def test_smoke_majority(self, smoke_dir: Path) -> None:
        """Verify smoke_majority.tau implements majority voting."""
        verifier = TauSpecVerifier()
        spec_path = smoke_dir / "smoke_majority.tau"
        
        if not spec_path.exists():
            pytest.skip("smoke_majority.tau not found")
        
        report = verifier.verify_spec_file(spec_path, "majority_2of3")
        
        assert report.overall_status == VerificationStatus.PASSED
        assert len(report.results) == 1
        assert report.results[0].passed_combinations == 8
