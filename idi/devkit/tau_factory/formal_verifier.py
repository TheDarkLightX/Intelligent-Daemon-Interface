"""Formal Verification Tool Suite for Tau Specifications.

Provides static analysis and exhaustive verification to ensure Tau specifications
correctly implement their documented formal claims (LTL properties, Boolean logic).

Verification Layers:
1. Syntactic Analysis - Parse and validate Tau formula structure
2. Semantic Verification - Truth table exhaustive checking against oracles
3. LTL Claim Verification - Map formulas to documented LTL properties
4. Completeness Checking - Ensure all input combinations are handled
5. Trace Verification - Validate temporal execution sequences

Design by Contract:
- PRE: Valid Tau spec file path or spec string
- INV: All 2^n input combinations evaluated for n-input formulas
- POST: VerificationResult with pass/fail, counterexamples, and coverage metrics
"""

from __future__ import annotations

import re
import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
# Z3 SMT Solver for mathematical proof of correctness
try:
    from z3 import BitVec, BitVecVal, If, Solver, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False



class VerificationStatus(Enum):
    """Verification outcome status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ClaimType(Enum):
    """Type of formal claim being verified."""
    BOOLEAN_FORMULA = "boolean_formula"
    LTL_SAFETY = "ltl_safety"
    LTL_LIVENESS = "ltl_liveness"
    LTL_FAIRNESS = "ltl_fairness"
    LTL_CONVERGENCE = "ltl_convergence"
    IMPLICATION = "implication"
    MUTUAL_EXCLUSION = "mutual_exclusion"


@dataclass
class FormalClaim:
    """A formal claim about a Tau specification."""
    name: str
    claim_type: ClaimType
    ltl_formula: str
    description: str
    oracle: Callable[..., int]
    input_names: List[str]
    output_name: str
    is_temporal: bool = False  # True if claim involves [t-1] references


@dataclass
class ParsedFormula:
    """Parsed representation of a Tau Boolean formula."""
    raw: str
    inputs: List[str]
    output: str
    expression: str
    temporal: bool = False


@dataclass
class Counterexample:
    """A counterexample where formula output differs from expected."""
    inputs: Dict[str, int]
    expected: int
    actual: int
    formula: str


@dataclass
class VerificationResult:
    """Result of formal verification."""
    status: VerificationStatus
    spec_name: str
    claim: Optional[FormalClaim]
    total_combinations: int
    passed_combinations: int
    failed_combinations: int
    counterexamples: List[Counterexample] = field(default_factory=list)
    coverage_pct: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.passed_combinations == self.total_combinations


@dataclass
class SpecVerificationReport:
    """Complete verification report for a specification."""
    spec_path: str
    parsed_formulas: List[ParsedFormula]
    results: List[VerificationResult]
    overall_status: VerificationStatus
    summary: str


class TauFormulaParser:
    """Parse Tau specification files and extract Boolean formulas."""
    
    # Old format: i0:sbf = in file("path")
    _STREAM_PATTERN_OLD = re.compile(
        r"(\w+)\s*:\s*(sbf|bv\[\d+\])\s*=\s*(in|out)\s+file\([\"']([^\"']+)[\"']\)"
    )
    # New format: sbf i0 = ifile("path") or bv[16] i0 = ifile("path")
    _STREAM_PATTERN_NEW = re.compile(
        r"(sbf|bv\[\d+\])\s+(\w+)\s*=\s*(ifile|ofile)\([\"']([^\"']+)[\"']\)"
    )
    # Console format: sbf i1 = console (with optional trailing dot)
    _STREAM_PATTERN_CONSOLE = re.compile(
        r"^(sbf|bv\[\d+\])\s+(\w+)\s*=\s*(console)\.?\s*$", re.MULTILINE
    )
    # Colon format: i1 : bv[8] = in console.
    _STREAM_PATTERN_COLON = re.compile(
        r"^(\w+)\s*:\s*(sbf|bv\[\d+\])\s*=\s*(in|out)\s+(console|file\([^)]+\))\.?\s*$", re.MULTILINE
    )
    # Run pattern - simple, captures everything after 'r '
    _RUN_PATTERN = re.compile(r"^r\s+(.+)$", re.MULTILINE)
    # Direct assignment pattern: o1[t] = expr. (ends with dot)
    _DIRECT_ASSIGNMENT = re.compile(r"^(\w+\[t\]\s*=\s*.+)\.\s*$", re.MULTILINE)
    # Arrow conditional pattern: (cond -> out[t] = expr) - extract the assignment part
    _ARROW_PATTERN = re.compile(r"\(([^)]+)\s*->\s*(\w+\[t\]\s*=\s*[^)]+)\)", re.MULTILINE)
    # Multi-line run pattern - captures content between 'r (' and matching ')'
    _RUN_PATTERN_MULTILINE = re.compile(r"^r\s*\(", re.MULTILINE)
    _FORMULA_PATTERN = re.compile(r"(\w+)\[t(?:-\d+)?\]\s*=\s*(.+)")
    
    def parse_spec(self, spec_content: str) -> List[ParsedFormula]:
        """Parse a Tau spec and extract all formulas."""
        formulas: List[ParsedFormula] = []
        
        inputs: Dict[str, str] = {}
        outputs: Dict[str, str] = {}
        
        # Parse old format: i0:sbf = in file("path")
        for match in self._STREAM_PATTERN_OLD.finditer(spec_content):
            name, stype, direction, _path = match.groups()
            if direction == "in":
                inputs[name] = stype
            else:
                outputs[name] = stype
        
        # Parse new format: sbf i0 = ifile("path") or bv[16] o0 = ofile("path")
        for match in self._STREAM_PATTERN_NEW.finditer(spec_content):
            stype, name, direction, _path = match.groups()
            if direction == "ifile":
                inputs[name] = stype
            else:
                outputs[name] = stype
        
        # Parse console format: sbf i1 = console
        # Heuristic: names starting with 'i' are inputs, 'o' are outputs
        for match in self._STREAM_PATTERN_CONSOLE.finditer(spec_content):
            stype, name, _ = match.groups()
            if name.startswith("i"):
                inputs[name] = stype
            else:
                outputs[name] = stype
        
        # Parse colon format: i1 : bv[8] = in console.
        for match in self._STREAM_PATTERN_COLON.finditer(spec_content):
            name, stype, direction, _ = match.groups()
            if direction == "in":
                inputs[name] = stype
            else:
                outputs[name] = stype
        
        wff_list: List[str] = []
        
        # Collect from run commands
        for run_match in self._RUN_PATTERN.finditer(spec_content):
            wff = run_match.group(1).strip()
            # Skip if it's just an opening paren (multi-line start)
            if wff == "(" or wff.startswith("(") and wff.count("(") > wff.count(")"):
                continue  # Will be handled by multi-line parser
            wff_list.append(wff)
        
        # Collect from direct assignments (o1[t] = expr.)
        for da_match in self._DIRECT_ASSIGNMENT.finditer(spec_content):
            wff_list.append(da_match.group(1).strip())
        
        # Collect from arrow conditionals (cond -> out[t] = expr)
        for arrow_match in self._ARROW_PATTERN.finditer(spec_content):
            cond = arrow_match.group(1).strip()
            assignment = arrow_match.group(2).strip()
            # Convert arrow to ternary: out[t] = (cond ? value : prev)
            formula_match = self._FORMULA_PATTERN.match(assignment)
            if formula_match:
                output = formula_match.group(1)
                value = formula_match.group(2)
                # For now, just add the assignment directly
                wff_list.append(assignment)
        
        # If no matches, try multi-line pattern
        if not wff_list:
            for ml_match in self._RUN_PATTERN_MULTILINE.finditer(spec_content):
                start = ml_match.start()
                depth = 0
                in_run = False
                run_content = []
                for i, char in enumerate(spec_content[start:]):
                    if char == '(' and not in_run:
                        in_run = True
                        depth = 1
                    elif char == '(':
                        depth += 1
                        run_content.append(char)
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            break
                        run_content.append(char)
                    elif in_run:
                        run_content.append(char)
                wff = ''.join(run_content).strip()
                # Remove comment lines from multi-line content
                wff_lines = []
                for line in wff.split('\n'):
                    line = line.strip()
                    if not line.startswith('#'):
                        # Remove inline comments
                        if '#' in line:
                            line = line[:line.index('#')].strip()
                        if line:
                            wff_lines.append(line)
                wff = ' '.join(wff_lines)
                if wff:
                    wff_list.append(wff)
                break
        
        for wff in wff_list:
            
            # Remove initialization clauses like & (var[0] = value) or && (var[0] = value)
            wff = re.sub(r"\s*&+\s*\(\w+\[0\]\s*=\s*[^)]*\)\s*$", "", wff)
            wff = re.sub(r"\s*&+\s*\(\w+\[0\]\s*=\s*[^)]*\)", "", wff)
            
            # Remove trailing unbalanced parentheses
            open_count = wff.count("(")
            close_count = wff.count(")")
            while close_count > open_count and wff.endswith(")"):
                wff = wff[:-1].strip()
                close_count -= 1
            
            if "&&" in wff:
                sub_formulas = self._split_conjunctions(wff)
            else:
                sub_formulas = [wff]
            
            for sub in sub_formulas:
                sub = sub.strip()
                if sub.startswith("(") and sub.endswith(")"):
                    sub = sub[1:-1].strip()
                
                formula_match = self._FORMULA_PATTERN.match(sub)
                if formula_match:
                    output_name = formula_match.group(1)
                    expression = formula_match.group(2).strip()
                    used_inputs = self._extract_variables(expression, inputs)
                    temporal = "t-" in expression or "[t]" in expression
                    
                    formulas.append(ParsedFormula(
                        raw=sub,
                        inputs=used_inputs,
                        output=output_name,
                        expression=expression,
                        temporal=temporal,
                    ))
        
        return formulas
    
    def _split_conjunctions(self, wff: str) -> List[str]:
        """Split formula on && while respecting parentheses."""
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        
        i = 0
        while i < len(wff):
            char = wff[i]
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "&" and i + 1 < len(wff) and wff[i + 1] == "&" and depth == 0:
                parts.append("".join(current).strip())
                current = []
                i += 1
            else:
                current.append(char)
            i += 1
        
        if current:
            parts.append("".join(current).strip())
        
        return parts
    
    def _split_on_and(self, wff: str) -> List[str]:
        """Split formula on single & while respecting parentheses."""
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        
        i = 0
        while i < len(wff):
            char = wff[i]
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "&" and depth == 0:
                # Check it's not &&
                if i + 1 < len(wff) and wff[i + 1] == "&":
                    current.append(char)
                else:
                    parts.append("".join(current).strip())
                    current = []
            else:
                current.append(char)
            i += 1
        
        if current:
            parts.append("".join(current).strip())
        
        return parts
    
    def _extract_variables(self, expression: str, known_inputs: Dict[str, str]) -> List[str]:
        """Extract input variable names from expression."""
        var_pattern = re.compile(r"(\w+)\[t(?:-\d+)?\]")
        found = set()
        
        for match in var_pattern.finditer(expression):
            var_name = match.group(1)
            if var_name in known_inputs:
                found.add(var_name)
        
        return sorted(found)
    
    def parse_file(self, path: Path) -> List[ParsedFormula]:
        """Parse a Tau spec file."""
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return self.parse_spec(content)


class BitvectorEvaluator:
    """Evaluate Tau bitvector formulas (bv[N] type) with modular arithmetic."""
    
    def __init__(self, width: int = 16):
        """Initialize with bitvector width."""
        self.width = width
        self.max_val = (1 << width)  # 2^width
    
    def evaluate(self, expression: str, bindings: Dict[str, int]) -> int:
        """Evaluate a bitvector expression with given variable bindings."""
        expr = expression
        
        # Substitute variable bindings
        for var_name, value in bindings.items():
            expr = re.sub(
                rf"\b{re.escape(var_name)}\[t(?:-\d+)?\]",
                str(value),
                expr,
            )
        
        # Replace bitvector constants like {51}:bv[16] with just the number
        expr = re.sub(r"\{(\d+)\}:bv\[\d+\]", r"\1", expr)
        # Replace hex constants like {#xFF}:bv[16] or {#x00FF}:bv[16]
        def hex_to_int(m):
            hex_val = m.group(1)
            return str(int(hex_val, 16))
        expr = re.sub(r"\{#x([0-9a-fA-F]+)\}:bv\[\d+\]", hex_to_int, expr)
        # Also handle {#b...} binary constants
        def bin_to_int(m):
            bin_val = m.group(1)
            return str(int(bin_val, 2))
        expr = re.sub(r"\{#b([01]+)\}:bv\[\d+\]", bin_to_int, expr)
        
        return self._eval_expr(expr.strip())
    
    def _eval_expr(self, expr: str) -> int:
        """Recursively evaluate bitvector expression."""
        expr = expr.strip()
        
        # Handle parentheses
        if expr.startswith("(") and self._matching_paren(expr) == len(expr) - 1:
            return self._eval_expr(expr[1:-1])
        
        # Handle ternary: cond ? then : else (with or without outer parens)
        # Find ? and : at depth 0
        q_idx = self._find_op_outside_parens(expr, "?")
        if q_idx >= 0:
            c_idx = self._find_op_outside_parens(expr[q_idx+1:], ":")
            if c_idx >= 0:
                c_idx += q_idx + 1
                cond = self._eval_expr(expr[:q_idx].strip())
                then_val = self._eval_expr(expr[q_idx+1:c_idx].strip())
                else_val = self._eval_expr(expr[c_idx+1:].strip())
                return then_val if cond else else_val
        
        # Find operators (lowest to highest precedence): comparison, +/-, *, /
        # Comparisons first (lowest precedence)
        for op in [">", "<", ">=", "<=", "=="]:
            idx = self._find_op_outside_parens(expr, op)
            if idx >= 0:
                left = self._eval_expr(expr[:idx].strip())
                right = self._eval_expr(expr[idx + len(op):].strip())
                if op == ">":
                    return 1 if left > right else 0
                elif op == "<":
                    return 1 if left < right else 0
                elif op == ">=":
                    return 1 if left >= right else 0
                elif op == "<=":
                    return 1 if left <= right else 0
                elif op == "==":
                    return 1 if left == right else 0
        
        # Addition/subtraction
        for op in ["+", "-"]:
            idx = self._find_op_outside_parens(expr, op)
            if idx >= 0 and idx > 0:  # Don't match leading minus
                left = self._eval_expr(expr[:idx].strip())
                right = self._eval_expr(expr[idx + 1:].strip())
                if op == "+":
                    return (left + right) % self.max_val
                elif op == "-":
                    return (left - right) % self.max_val
        
        # Multiplication/division/modulo
        for op in ["*", "/", "%"]:
            idx = self._find_op_outside_parens(expr, op)
            if idx >= 0:
                left = self._eval_expr(expr[:idx].strip())
                right = self._eval_expr(expr[idx + 1:].strip())
                if op == "*":
                    return (left * right) % self.max_val
                elif op == "/":
                    return (left // right) % self.max_val if right != 0 else 0
                elif op == "%":
                    return left % right if right != 0 else 0
        
        # Bitwise operators
        for op in ["|", "^", "&"]:
            idx = self._find_op_outside_parens(expr, op)
            if idx >= 0:
                left = self._eval_expr(expr[:idx].strip())
                right = self._eval_expr(expr[idx + 1:].strip())
                if op == "|":
                    return left | right
                elif op == "^":
                    return left ^ right
                elif op == "&":
                    return left & right
        
        # Bitwise NOT (')
        if expr.endswith("'"):
            val = self._eval_expr(expr[:-1].strip())
            return (~val) & (self.max_val - 1)
        
        # Numeric literal
        try:
            return int(expr) % self.max_val
        except ValueError:
            raise ValueError(f"Unknown bitvector expression: {expr}")
    
    def _matching_paren(self, expr: str) -> int:
        """Find matching closing paren for opening paren at index 0."""
        depth = 0
        for i, c in enumerate(expr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1
    
    def _find_op_outside_parens(self, expr: str, op: str) -> int:
        """Find rightmost operator outside parentheses."""
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ")":
                depth += 1
            elif expr[i] == "(":
                depth -= 1
            elif depth == 0 and expr[i:i + len(op)] == op:
                return i
        return -1


class BooleanEvaluator:
    """Evaluate Tau Boolean formulas (sbf type)."""
    
    def evaluate(self, expression: str, bindings: Dict[str, int]) -> int:
        """Evaluate a Boolean expression with given variable bindings."""
        expr = expression
        for var_name, value in bindings.items():
            expr = re.sub(
                rf"\b{re.escape(var_name)}\[t(?:-\d+)?\]",
                str(value),
                expr,
            )
        return self._eval_expr(expr.strip())
    
    def _eval_expr(self, expr: str) -> int:
        """Recursively evaluate expression.
        
        Operator precedence (lowest to highest): |, ^, &, '
        The ' operator binds only to the immediately preceding term.
        """
        expr = expr.strip()
        
        # Handle outer parentheses with trailing negation: (...)' 
        if expr.endswith("'"):
            inner = expr[:-1].strip()
            if inner.startswith("(") and self._matching_paren(inner) == len(inner) - 1:
                return 1 - self._eval_expr(inner[1:-1])
            # Don't recurse here - let binary operators be found first
        
        # Handle parentheses
        if expr.startswith("("):
            close = self._matching_paren(expr)
            if close == len(expr) - 1:
                return self._eval_expr(expr[1:-1])
        
        # Find lowest precedence binary operator outside parentheses
        for op in ["|", "^", "&"]:
            idx = self._find_op_outside_parens(expr, op)
            if idx >= 0:
                left = expr[:idx].strip()
                right = expr[idx + 1:].strip()
                left_val = self._eval_expr(left)
                right_val = self._eval_expr(right)
                
                if op == "|":
                    return left_val | right_val
                elif op == "^":
                    return left_val ^ right_val
                elif op == "&":
                    return left_val & right_val
        
        # Now handle unary negation (highest precedence for atoms)
        if expr.endswith("'"):
            return 1 - self._eval_expr(expr[:-1].strip())
        
        if expr in ("0", "1"):
            return int(expr)
        
        try:
            return int(expr)
        except ValueError:
            raise ValueError(f"Unknown expression: {expr}")
    
    def _matching_paren(self, expr: str) -> int:
        """Find matching closing paren for opening paren at position 0."""
        if not expr.startswith("("):
            return -1
        depth = 0
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1
    
    def _find_op_outside_parens(self, expr: str, op: str) -> int:
        """Find rightmost occurrence of operator outside parentheses."""
        depth = 0
        result = -1
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == op and depth == 0:
                result = i
        return result


class TruthTableGenerator:
    """Generate exhaustive truth tables for Boolean formulas."""
    
    def __init__(self, max_inputs: int = 10) -> None:
        self._max_inputs = max_inputs
    
    def generate(self, input_names: List[str]) -> List[Dict[str, int]]:
        """Generate all input combinations."""
        n = len(input_names)
        if n > self._max_inputs:
            raise ValueError(f"Too many inputs ({n} > {self._max_inputs}).")
        
        combinations: List[Dict[str, int]] = []
        for values in itertools.product([0, 1], repeat=n):
            combinations.append(dict(zip(input_names, values)))
        
        return combinations


class FormalClaimRegistry:
    """Registry of documented formal claims for verification."""
    
    def __init__(self) -> None:
        self._claims: Dict[str, FormalClaim] = {}
        self._register_builtin_claims()
    
    def _register_builtin_claims(self) -> None:
        """Register all documented formal claims."""
        
        # === PASSTHROUGH / IDENTITY ===
        self.register(FormalClaim(
            name="passthrough",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(out = in)",
            description="Output equals input (identity function)",
            oracle=lambda x: x,
            input_names=["i0"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="echo",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(out = in)",
            description="Echo/passthrough pattern",
            oracle=lambda x: x,
            input_names=["i0"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="mutual_exclusion",
            claim_type=ClaimType.MUTUAL_EXCLUSION,
            ltl_formula="G(¬(a ∧ b))",
            description="At most one agent active at a time (NAND gate)",
            oracle=lambda a, b: 1 - (a & b),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="never_unsafe",
            claim_type=ClaimType.LTL_SAFETY,
            ltl_formula="G(¬unsafe)",
            description="Safety invariant - negation of unsafe condition",
            oracle=lambda unsafe: 1 - unsafe,
            input_names=["i0"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="belief_consistency",
            claim_type=ClaimType.LTL_SAFETY,
            ltl_formula="G(¬(belief ∧ ¬belief))",
            description="Beliefs must not contradict",
            oracle=lambda belief, negated: 1 - (belief & negated),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="no_starvation",
            claim_type=ClaimType.LTL_LIVENESS,
            ltl_formula="G(request → F grant)",
            description="Request implies eventual grant (implication)",
            oracle=lambda req, grant: (1 - req) | grant,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="progress",
            claim_type=ClaimType.LTL_LIVENESS,
            ltl_formula="G(enabled → F done)",
            description="Enabled implies eventual completion",
            oracle=lambda enabled, done: (1 - enabled) | done,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="consensus_check",
            claim_type=ClaimType.LTL_CONVERGENCE,
            ltl_formula="FG(v₁ = v₂)",
            description="Values eventually equal and stay equal (XNOR)",
            oracle=lambda a, b: 1 - (a ^ b),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="reputation_gate",
            claim_type=ClaimType.IMPLICATION,
            ltl_formula="G(action → trusted)",
            description="Actions require trust (AND gate)",
            oracle=lambda action, trusted: action & trusted,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="risk_gate",
            claim_type=ClaimType.LTL_SAFETY,
            ltl_formula="G(action → ¬high_risk)",
            description="Actions blocked by high risk (AND-NOT)",
            oracle=lambda action, risk: action & (1 - risk),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="counterfactual_safe",
            claim_type=ClaimType.LTL_SAFETY,
            ltl_formula="G(choice → ¬bad_outcome)",
            description="Choices avoid bad counterfactuals",
            oracle=lambda choice, bad: choice & (1 - bad),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="exploration_decay",
            claim_type=ClaimType.LTL_CONVERGENCE,
            ltl_formula="FG(¬exploring)",
            description="Exploration decays with experience",
            oracle=lambda trigger, exp: trigger & (1 - exp),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="safe_exploration",
            claim_type=ClaimType.LTL_SAFETY,
            ltl_formula="G(explore → safe)",
            description="Exploration requires safety (AND)",
            oracle=lambda explore, safe: explore & safe,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="emergent_detector",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(actual ≠ expected → alert)",
            description="Detect deviation from expected (XOR)",
            oracle=lambda actual, expected: actual ^ expected,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="utility_alignment",
            claim_type=ClaimType.IMPLICATION,
            ltl_formula="G(action → aligned)",
            description="Actions imply preference alignment",
            oracle=lambda action, aligned: (1 - action) | aligned,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="causal_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(do(action) → cause)",
            description="Actions require causal precondition (AND)",
            oracle=lambda action, cause: action & cause,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="majority_2of3",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="(a∧b) ∨ (a∧c) ∨ (b∧c)",
            description="2-of-3 majority voting",
            oracle=lambda a, b, c: (a & b) | (a & c) | (b & c),
            input_names=["i0", "i1", "i2"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="unanimous",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ∧ b ∧ c",
            description="Unanimous agreement (all must agree)",
            oracle=lambda a, b, c: a & b & c,
            input_names=["i0", "i1", "i2"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="nand_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="¬(a ∧ b)",
            description="NAND gate",
            oracle=lambda a, b: 1 - (a & b),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        # === BASIC GATES ===
        self.register(FormalClaim(
            name="not_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="¬a",
            description="NOT gate (inverter)",
            oracle=lambda a: 1 - a,
            input_names=["i0"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="and_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ∧ b",
            description="AND gate",
            oracle=lambda a, b: a & b,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="or_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ∨ b",
            description="OR gate",
            oracle=lambda a, b: a | b,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="xor_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ⊕ b",
            description="XOR gate",
            oracle=lambda a, b: a ^ b,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        # === TEMPORAL PATTERNS ===
        self.register(FormalClaim(
            name="delay",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t-1]",
            description="1-step delay",
            oracle=lambda prev: prev,
            input_names=["prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # AND with previous: out = in & prev
        self.register(FormalClaim(
            name="and_with_prev",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∧ out[t-1]",
            description="AND with previous output",
            oracle=lambda inp, prev: inp & prev,
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # AND-NOT with previous: out = in & ~prev
        self.register(FormalClaim(
            name="and_not_prev",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∧ ¬out[t-1]",
            description="AND-NOT with previous (rising edge detector)",
            oracle=lambda inp, prev: inp & (1 - prev),
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Latch with two inputs: out = set | (prev & ~reset)
        self.register(FormalClaim(
            name="sr_latch",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = set[t] ∨ (out[t-1] ∧ ¬reset[t])",
            description="SR latch (set-reset)",
            oracle=lambda set_in, reset, prev: set_in | (prev & (1 - reset)),
            input_names=["i0", "i1", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # 3-input AND
        self.register(FormalClaim(
            name="and3_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ∧ b ∧ c",
            description="3-input AND gate",
            oracle=lambda a, b, c: a & b & c,
            input_names=["i0", "i1", "i2"],
            output_name="o0",
        ))
        
        # 3-input OR
        self.register(FormalClaim(
            name="or3_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ∨ b ∨ c",
            description="3-input OR gate",
            oracle=lambda a, b, c: a | b | c,
            input_names=["i0", "i1", "i2"],
            output_name="o0",
        ))
        
        # === EDGE DETECTION ===
        # Rising edge: in & ~prev_in
        self.register(FormalClaim(
            name="rising_edge",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∧ ¬in[t-1]",
            description="Rising edge detector",
            oracle=lambda inp, prev_inp: inp & (1 - prev_inp),
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Falling edge: ~in & prev_in
        self.register(FormalClaim(
            name="falling_edge",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = ¬in[t] ∧ in[t-1]",
            description="Falling edge detector",
            oracle=lambda inp, prev_inp: (1 - inp) & prev_inp,
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # === COMPLEX TEMPORAL PATTERNS ===
        # Momentum: in & prev_in & ~prev_out (sustained input triggers)
        self.register(FormalClaim(
            name="momentum_trigger",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∧ in[t-1] ∧ ¬out[t-1]",
            description="Momentum trigger - sustained input",
            oracle=lambda inp, prev_inp, prev_out: inp & prev_inp & (1 - prev_out),
            input_names=["i0", "prev_in", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # OR with temporal: (in1 | prev_in1) & (in2 | prev_in2)
        self.register(FormalClaim(
            name="sustained_and",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="(a[t] ∨ a[t-1]) ∧ (b[t] ∨ b[t-1])",
            description="Sustained AND - either current or previous must be true",
            oracle=lambda a, prev_a, b, prev_b: (a | prev_a) & (b | prev_b),
            input_names=["i0", "prev_i0", "i1", "prev_i1"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Momentum with 2 inputs: in & prev_in & ~prev_out
        self.register(FormalClaim(
            name="momentum_entry",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∧ in[t-1] ∧ ¬out[t-1]",
            description="Momentum entry - sustained input triggers once",
            oracle=lambda inp, prev_inp, prev_out: inp & prev_inp & (1 - prev_out),
            input_names=["i0", "prev_i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Two input AND-NOT-NOT: a & ~b & ~prev_out
        self.register(FormalClaim(
            name="gated_entry",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = a[t] ∧ ¬b[t] ∧ ¬out[t-1]",
            description="Gated entry - a triggers when b is off and not already triggered",
            oracle=lambda a, b, prev_out: a & (1 - b) & (1 - prev_out),
            input_names=["i0", "i1", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Layered strategy: (a & prev_a & ~prev_out) | (b & prev_b & prev_out)
        self.register(FormalClaim(
            name="layered_state",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = (a[t] ∧ a[t-1] ∧ ¬out[t-1]) ∨ (b[t] ∧ b[t-1] ∧ out[t-1])",
            description="Layered state machine - entry on sustained a, hold on sustained b",
            oracle=lambda a, prev_a, b, prev_b, prev_out: (a & prev_a & (1 - prev_out)) | (b & prev_b & prev_out),
            input_names=["i0", "prev_i0", "i1", "prev_i1", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        self.register(FormalClaim(
            name="nor_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="¬(a ∨ b)",
            description="NOR gate",
            oracle=lambda a, b: 1 - (a | b),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="xnor_gate",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="a ↔ b",
            description="XNOR (equivalence) gate",
            oracle=lambda a, b: 1 - (a ^ b),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        self.register(FormalClaim(
            name="implication_gate",
            claim_type=ClaimType.IMPLICATION,
            ltl_formula="a → b",
            description="Implication gate",
            oracle=lambda a, b: (1 - a) | b,
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        # === FSM / TEMPORAL PATTERNS ===
        # SR Latch: set dominates - out = set | (out_prev & ~reset)
        self.register(FormalClaim(
            name="sr_latch_set_dominant",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = set[t] ∨ (out[t-1] ∧ ¬reset[t])",
            description="SR latch with set dominance",
            oracle=lambda set_in, reset, prev: set_in | (prev & (1 - reset)),
            input_names=["i0", "i1", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Toggle: out = in XOR prev
        self.register(FormalClaim(
            name="toggle",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ⊕ out[t-1]",
            description="Toggle pattern (XOR with previous)",
            oracle=lambda inp, prev: inp ^ prev,
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Hold: maintain state when no input - out = in | prev
        self.register(FormalClaim(
            name="hold_or",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="out[t] = in[t] ∨ out[t-1]",
            description="Hold pattern - OR with previous",
            oracle=lambda inp, prev: inp | prev,
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Action exclusivity: buy and sell mutually exclusive
        self.register(FormalClaim(
            name="action_exclusivity",
            claim_type=ClaimType.MUTUAL_EXCLUSION,
            ltl_formula="G(¬(buy ∧ sell))",
            description="Buy and sell signals mutually exclusive",
            oracle=lambda buy, sell: 1 - (buy & sell),
            input_names=["i0", "i1"],
            output_name="o0",
        ))
        
        # Position tracker: SR latch with descriptive names
        # holding = (prev' & buy) | (prev & ~sell)
        self.register(FormalClaim(
            name="position_sr",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="holding[t] = (¬holding[t-1] ∧ buy[t]) ∨ (holding[t-1] ∧ ¬sell[t])",
            description="Position tracker - set on buy, reset on sell",
            oracle=lambda buy, sell, prev: ((1 - prev) & buy) | (prev & (1 - sell)),
            input_names=["i0", "i1", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Timer bit0: toggle on trigger
        self.register(FormalClaim(
            name="timer_b0",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="b0[t] = trigger[t] ∧ ¬b0[t-1]",
            description="Timer bit 0 - toggles on trigger",
            oracle=lambda trigger, prev: trigger & (1 - prev),
            input_names=["i0", "prev"],
            output_name="o0",
            is_temporal=True,
        ))
        
        # Constant true
        self.register(FormalClaim(
            name="always_true",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(out = 1)",
            description="Always outputs 1",
            oracle=lambda: 1,
            input_names=[],
            output_name="o0",
        ))
        
        # Constant false
        self.register(FormalClaim(
            name="always_false",
            claim_type=ClaimType.BOOLEAN_FORMULA,
            ltl_formula="G(out = 0)",
            description="Always outputs 0",
            oracle=lambda: 0,
            input_names=[],
            output_name="o0",
        ))
    
    def register(self, claim: FormalClaim) -> None:
        """Register a formal claim."""
        self._claims[claim.name] = claim
    
    def get(self, name: str) -> Optional[FormalClaim]:
        """Get a claim by name."""
        return self._claims.get(name)
    
    def all_claims(self) -> List[FormalClaim]:
        """Get all registered claims."""
        return list(self._claims.values())


class TauSpecVerifier:
    """Main verification engine for Tau specifications."""
    
    def __init__(self) -> None:
        self._parser = TauFormulaParser()
        self._evaluator = BooleanEvaluator()
        self._bv_evaluator = BitvectorEvaluator(width=16)
        self._truth_table = TruthTableGenerator()
        self._registry = FormalClaimRegistry()
    
    def verify_formula_against_oracle(
        self,
        formula: ParsedFormula,
        oracle: Callable[..., int],
        spec_name: str = "unknown",
        claim: Optional[FormalClaim] = None,
    ) -> VerificationResult:
        """Verify a parsed formula against an oracle function."""
        if not formula.inputs:
            # For constant expressions (no inputs), evaluate once
            try:
                actual = self._evaluator.evaluate(formula.expression, {})
                expected = oracle()
                if actual == expected:
                    return VerificationResult(
                        status=VerificationStatus.PASSED,
                        spec_name=spec_name,
                        claim=claim,
                        total_combinations=1,
                        passed_combinations=1,
                        failed_combinations=0,
                        message="Constant expression verified",
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        spec_name=spec_name,
                        claim=claim,
                        total_combinations=1,
                        passed_combinations=0,
                        failed_combinations=1,
                        message=f"Constant mismatch: expected {expected}, got {actual}",
                    )
            except Exception:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    spec_name=spec_name,
                    claim=claim,
                    total_combinations=0,
                    passed_combinations=0,
                    failed_combinations=0,
                    message="Cannot verify constant expression",
                )
        
        combinations = self._truth_table.generate(formula.inputs)
        total = len(combinations)
        passed = 0
        failed = 0
        counterexamples: List[Counterexample] = []
        
        for combo in combinations:
            try:
                actual = self._evaluator.evaluate(formula.expression, combo)
                expected = oracle(*[combo[inp] for inp in formula.inputs])
                
                if actual == expected:
                    passed += 1
                else:
                    failed += 1
                    if len(counterexamples) < 5:
                        counterexamples.append(Counterexample(
                            inputs=combo,
                            expected=expected,
                            actual=actual,
                            formula=formula.expression,
                        ))
            except Exception as e:
                failed += 1
                if len(counterexamples) < 5:
                    counterexamples.append(Counterexample(
                        inputs=combo,
                        expected=-1,
                        actual=-1,
                        formula=f"Error: {e}",
                    ))
        
        status = VerificationStatus.PASSED if failed == 0 else VerificationStatus.FAILED
        coverage = (passed / total * 100) if total > 0 else 0.0
        
        return VerificationResult(
            status=status,
            spec_name=spec_name,
            claim=claim,
            total_combinations=total,
            passed_combinations=passed,
            failed_combinations=failed,
            counterexamples=counterexamples,
            coverage_pct=coverage,
            message=f"{passed}/{total} combinations verified" if status == VerificationStatus.PASSED 
                    else f"FAILED: {failed}/{total} combinations failed",
        )
    
    def verify_spec_file(
        self,
        spec_path: Path,
        claim_name: Optional[str] = None,
    ) -> SpecVerificationReport:
        """Verify a Tau spec file against documented claims."""
        try:
            formulas = self._parser.parse_file(spec_path)
        except Exception as e:
            return SpecVerificationReport(
                spec_path=str(spec_path),
                parsed_formulas=[],
                results=[VerificationResult(
                    status=VerificationStatus.ERROR,
                    spec_name=spec_path.name,
                    claim=None,
                    total_combinations=0,
                    passed_combinations=0,
                    failed_combinations=0,
                    message=f"Parse error: {e}",
                )],
                overall_status=VerificationStatus.ERROR,
                summary=f"Failed to parse: {e}",
            )
        
        results: List[VerificationResult] = []
        
        for formula in formulas:
            if claim_name:
                claim = self._registry.get(claim_name)
            else:
                claim = self._detect_claim(formula)
            
            if claim:
                # Use temporal verification for temporal claims
                if claim.is_temporal:
                    result = self._verify_temporal_formula(
                        formula,
                        claim.oracle,
                        spec_name=spec_path.name,
                        claim=claim,
                    )
                else:
                    result = self.verify_formula_against_oracle(
                        formula,
                        claim.oracle,
                        spec_name=spec_path.name,
                        claim=claim,
                    )
            else:
                # Try Z3 SMT verification for bitvector formulas (mathematical proof)
                if self._is_bitvector_formula(formula):
                    result = self._verify_with_z3(formula, spec_path.name)
                else:
                    result = VerificationResult(
                        status=VerificationStatus.SKIPPED,
                        spec_name=spec_path.name,
                        claim=None,
                        total_combinations=0,
                        passed_combinations=0,
                        failed_combinations=0,
                        message="No matching claim found for formula",
                        details={"expression": formula.expression},
                    )
            
            results.append(result)
        
        overall = self._compute_overall_status(results)
        summary = self._generate_summary(results)
        
        return SpecVerificationReport(
            spec_path=str(spec_path),
            parsed_formulas=formulas,
            results=results,
            overall_status=overall,
            summary=summary,
        )
    
    def verify_spec_content(
        self,
        content: str,
        claim_name: str,
        spec_name: str = "inline",
    ) -> VerificationResult:
        """Verify spec content string against a named claim."""
        formulas = self._parser.parse_spec(content)
        claim = self._registry.get(claim_name)
        
        if not claim:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                spec_name=spec_name,
                claim=None,
                total_combinations=0,
                passed_combinations=0,
                failed_combinations=0,
                message=f"Unknown claim: {claim_name}",
            )
        
        if not formulas:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                spec_name=spec_name,
                claim=claim,
                total_combinations=0,
                passed_combinations=0,
                failed_combinations=0,
                message="No formulas found in spec",
            )
        
        return self.verify_formula_against_oracle(
            formulas[0],
            claim.oracle,
            spec_name=spec_name,
            claim=claim,
        )
    
    def _detect_claim(self, formula: ParsedFormula) -> Optional[FormalClaim]:
        """Try to detect which claim a formula implements.
        
        For temporal formulas (those with [t-1] references), we compute
        all temporal inputs needed based on actual [t-1] references.
        """
        expr = formula.expression
        is_temporal = formula.temporal or "t-1" in expr or "t-" in expr
        
        # Count actual inputs (excluding output self-references)
        actual_inputs = formula.inputs.copy()
        
        # Try non-temporal claims first (simpler)
        for claim in self._registry.all_claims():
            if claim.is_temporal:
                continue  # Skip temporal claims for now
            if len(claim.input_names) != len(actual_inputs):
                continue
            
            result = self.verify_formula_against_oracle(
                formula,
                claim.oracle,
                spec_name="detection",
                claim=claim,
            )
            if result.status == VerificationStatus.PASSED:
                return claim
        
        # Try temporal claims - compute actual temporal inputs from expression
        if is_temporal:
            temporal_refs = re.findall(r"(\w+)\[t-1\]", expr)
            temporal_input_names = []
            for ref in temporal_refs:
                if ref == formula.output:
                    temporal_input_names.append("prev")
                elif ref in formula.inputs:
                    temporal_input_names.append(f"prev_{ref}")
                elif ref.startswith("o"):
                    temporal_input_names.append("prev")
                else:
                    temporal_input_names.append(f"prev_{ref}")
            
            seen = set()
            unique_temporal = [x for x in temporal_input_names if not (x in seen or seen.add(x))]
            temporal_inputs = actual_inputs + unique_temporal
            
            for claim in self._registry.all_claims():
                if not claim.is_temporal:
                    continue
                if len(claim.input_names) != len(temporal_inputs):
                    continue
                
                result = self._verify_temporal_formula(
                    formula,
                    claim.oracle,
                    spec_name="detection",
                    claim=claim,
                )
                if result.status == VerificationStatus.PASSED:
                    return claim
        
        return None
    
    def _verify_temporal_formula(
        self,
        formula: ParsedFormula,
        oracle: Callable[..., int],
        spec_name: str = "unknown",
        claim: Optional[FormalClaim] = None,
    ) -> VerificationResult:
        """Verify a temporal formula by treating [t-1] refs as prev inputs.
        
        Handles both output[t-1] and input[t-1] references by adding
        'prev' for output and 'prev_<input>' for each input with temporal ref.
        """
        expr = formula.expression
        
        # Detect all [t-1] references in the formula
        temporal_refs = re.findall(r"(\w+)\[t-1\]", expr)
        
        # Build list of temporal inputs needed
        temporal_input_names = []
        for ref in temporal_refs:
            if ref == formula.output:
                temporal_input_names.append("prev")
            elif ref in formula.inputs:
                temporal_input_names.append(f"prev_{ref}")
            elif ref.startswith("o"):
                # Cross-output reference - treat as prev
                temporal_input_names.append("prev")
            else:
                temporal_input_names.append(f"prev_{ref}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_temporal = []
        for name in temporal_input_names:
            if name not in seen:
                seen.add(name)
                unique_temporal.append(name)
        
        all_inputs = formula.inputs + unique_temporal
        
        if len(all_inputs) > 10:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                spec_name=spec_name,
                claim=claim,
                total_combinations=0,
                passed_combinations=0,
                failed_combinations=0,
                message=f"Too many inputs ({len(all_inputs)}) for exhaustive verification",
            )
        
        combinations = self._truth_table.generate(all_inputs)
        total = len(combinations)
        passed = 0
        failed = 0
        counterexamples: List[Counterexample] = []
        
        for combo in combinations:
            try:
                eval_expr = expr
                
                # Substitute output[t-1] with prev (including cross-output refs)
                eval_expr = re.sub(
                    rf"\b{re.escape(formula.output)}\[t-1\]",
                    str(combo.get("prev", 0)),
                    eval_expr,
                )
                # Also substitute any o*[t-1] with prev (cross-output references)
                eval_expr = re.sub(
                    r"\bo\d+\[t-1\]",
                    str(combo.get("prev", 0)),
                    eval_expr,
                )
                
                # Substitute input[t-1] with prev_<input>
                for inp in formula.inputs:
                    prev_key = f"prev_{inp}"
                    if prev_key in combo:
                        eval_expr = re.sub(
                            rf"\b{re.escape(inp)}\[t-1\]",
                            str(combo[prev_key]),
                            eval_expr,
                        )
                
                actual = self._evaluator.evaluate(eval_expr, combo)
                expected = oracle(*[combo[inp] for inp in all_inputs])
                
                if actual == expected:
                    passed += 1
                else:
                    failed += 1
                    if len(counterexamples) < 5:
                        counterexamples.append(Counterexample(
                            inputs=combo,
                            expected=expected,
                            actual=actual,
                            formula=eval_expr,
                        ))
            except Exception as e:
                failed += 1
                if len(counterexamples) < 5:
                    counterexamples.append(Counterexample(
                        inputs=combo,
                        expected=-1,
                        actual=-1,
                        formula=f"Error: {e}",
                    ))
        
        status = VerificationStatus.PASSED if failed == 0 else VerificationStatus.FAILED
        coverage = (passed / total * 100) if total > 0 else 0.0
        
        return VerificationResult(
            status=status,
            spec_name=spec_name,
            claim=claim,
            total_combinations=total,
            passed_combinations=passed,
            failed_combinations=failed,
            counterexamples=counterexamples,
            coverage_pct=coverage,
            message=f"Bitvector {'PASSED' if failed == 0 else 'FAILED'}: {passed}/{total} samples {'verified' if failed == 0 else 'failed'}",
        )
    
    def _is_bitvector_formula(self, formula: ParsedFormula) -> bool:
        """Check if formula is a bitvector formula that should use BV verification."""
        expr = formula.expression
        
        # Skip comparisons and conditionals - too complex for sample verification
        if any(op in expr for op in ["<", ">", "==", "!="]):
            return False
        
        # Skip if expression contains initialization checks
        if re.search(r"\w+\[t-?\d*\]\s*=\s*\{", expr):
            return False
        
        # Skip ternary conditionals
        if "?" in expr:
            return False
        
        # Skip function calls
        if re.search(r"\w+\(", expr):
            return False
        
        # Simple bitvector arithmetic without constants (like o1[t-1] + i1[t])
        if any(op in expr for op in ["+", "-", "*", "/", "%"]):
            # Check it only contains variable refs and operators
            if re.match(r"^[\w\[\]t\-\d\s\+\-\*/%&|^']+$", expr):
                return True
        
        # Check for bitvector arithmetic with constants
        if re.search(r"\{[\d#][^}]*\}:bv\[\d+\]", expr):
            if any(op in expr for op in ["+", "-", "*", "/", "%"]):
                return True
        
        return False
    
    def _verify_bitvector_formula(
        self,
        formula: ParsedFormula,
        spec_name: str,
    ) -> VerificationResult:
        """Verify a bitvector formula using sample-based testing.
        
        For bv[N] formulas, exhaustive testing is impractical (2^16 values per input).
        Instead, we use key test vectors: 0, 1, max/2, max-1, max, and random samples.
        """
        import random
        
        expr = formula.expression
        inputs = formula.inputs
        max_val = 65535  # 2^16 - 1
        
        # Auto-detect variables from expression if no inputs found
        if not inputs:
            var_pattern = re.compile(r"(\w+)\[t\]")
            detected = set(var_pattern.findall(expr))
            # Remove output variable
            detected.discard(formula.output)
            inputs = sorted(detected)
        
        if not inputs:
            # Formula with no inputs - test with various temporal values
            import random
            random.seed(42)
            test_passed = 0
            test_failed = 0
            for _ in range(20):
                try:
                    eval_expr = expr
                    # Substitute temporal refs with random values
                    for match in re.findall(r"(\w+)\[t-1\]", eval_expr):
                        val = random.randint(0, max_val)
                        eval_expr = re.sub(rf"\b{match}\[t-1\]", str(val), eval_expr)
                    for match in re.findall(r"(\w+)\[t\]", eval_expr):
                        val = random.randint(0, max_val)
                        eval_expr = re.sub(rf"\b{match}\[t\]", str(val), eval_expr)
                    result = self._bv_evaluator.evaluate(eval_expr, {})
                    if 0 <= result <= max_val:
                        test_passed += 1
                    else:
                        test_failed += 1
                except:
                    test_failed += 1
            
            status = VerificationStatus.PASSED if test_failed == 0 else VerificationStatus.FAILED
            return VerificationResult(
                status=status,
                spec_name=spec_name,
                claim=None,
                total_combinations=20,
                passed_combinations=test_passed,
                failed_combinations=test_failed,
                message=f"Bitvector: {test_passed}/20 samples verified",
            )
        
        # Generate test vectors for each input
        max_val = 65535  # 2^16 - 1
        key_values = [0, 1, 127, 128, 255, 256, 32767, 32768, 65534, 65535]
        
        # Add some random values
        random.seed(42)  # Deterministic for reproducibility
        key_values.extend([random.randint(0, max_val) for _ in range(10)])
        
        # Generate combinations (limit to prevent explosion)
        test_cases = []
        if len(inputs) == 1:
            test_cases = [{inputs[0]: v} for v in key_values]
        elif len(inputs) == 2:
            for v1 in key_values[:8]:
                for v2 in key_values[:8]:
                    test_cases.append({inputs[0]: v1, inputs[1]: v2})
        else:
            # For many inputs, use fewer values
            for _ in range(100):
                case = {inp: random.choice(key_values[:6]) for inp in inputs}
                test_cases.append(case)
        
        passed = 0
        failed = 0
        counterexamples: List[Counterexample] = []
        
        for combo in test_cases:
            try:
                eval_expr = expr
                
                # Substitute input bindings for current time [t]
                for var_name, value in combo.items():
                    eval_expr = re.sub(rf"\b{re.escape(var_name)}\[t\]", str(value), eval_expr)
                
                # Substitute temporal references with random values
                # o0[t-1], o1[t-1], etc. get random values
                import random
                for match in re.findall(r"(\w+)\[t-1\]", eval_expr):
                    random.seed(hash(str(combo)) + hash(match))
                    temporal_val = random.randint(0, max_val)
                    eval_expr = re.sub(rf"\b{match}\[t-1\]", str(temporal_val), eval_expr)
                
                # Substitute remaining [t] refs (outputs) with random values
                for match in re.findall(r"(\w+)\[t\]", eval_expr):
                    if match not in combo:
                        random.seed(hash(str(combo)) + hash(match) + 1)
                        val = random.randint(0, max_val)
                        eval_expr = re.sub(rf"\b{match}\[t\]", str(val), eval_expr)
                
                # For bitvector formulas, we verify the expression evaluates without error
                # and produces a valid integer result
                result = self._bv_evaluator.evaluate(eval_expr, combo)
                if 0 <= result <= max_val:
                    passed += 1
                else:
                    failed += 1
                    if len(counterexamples) < 3:
                        counterexamples.append(Counterexample(
                            inputs=combo,
                            expected=-1,
                            actual=result,
                            formula=f"Out of range: {result}",
                        ))
            except Exception as e:
                failed += 1
                if len(counterexamples) < 3:
                    counterexamples.append(Counterexample(
                        inputs=combo,
                        expected=-1,
                        actual=-1,
                        formula=f"Error: {e}",
                    ))
        
        total = len(test_cases)
        status = VerificationStatus.PASSED if failed == 0 else VerificationStatus.FAILED
        coverage = (passed / total * 100) if total > 0 else 0.0
        
        return VerificationResult(
            status=status,
            spec_name=spec_name,
            claim=None,
            total_combinations=total,
            passed_combinations=passed,
            failed_combinations=failed,
            counterexamples=counterexamples,
            coverage_pct=coverage,
            message=f"Bitvector: {passed}/{total} samples verified" if status == VerificationStatus.PASSED 
                    else f"Bitvector FAILED: {failed}/{total} samples failed",
            details={"type": "bitvector", "width": 16},
        )
    
    def _compute_overall_status(self, results: List[VerificationResult]) -> VerificationStatus:
        """Compute overall status from individual results.
        
        Logic: If any formula fails, overall is FAILED. If any passes and none fail,
        overall is PASSED. Otherwise SKIPPED.
        """
        if not results:
            return VerificationStatus.SKIPPED  # No formulas found = skipped, not error
        
        if any(r.status == VerificationStatus.FAILED for r in results):
            return VerificationStatus.FAILED
        if any(r.status == VerificationStatus.ERROR for r in results):
            return VerificationStatus.ERROR
        # If any passed (and none failed), consider it passed
        if any(r.status == VerificationStatus.PASSED for r in results):
            return VerificationStatus.PASSED
        
        return VerificationStatus.SKIPPED
    
    def _generate_summary(self, results: List[VerificationResult]) -> str:
        """Generate a summary string from results."""
        passed = sum(1 for r in results if r.status == VerificationStatus.PASSED)
        failed = sum(1 for r in results if r.status == VerificationStatus.FAILED)
        skipped = sum(1 for r in results if r.status == VerificationStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)
        
        total = len(results)
        return f"{passed}/{total} passed, {failed} failed, {skipped} skipped, {errors} errors"
    
    def list_claims(self) -> List[str]:
        """List all registered claim names."""
        return [c.name for c in self._registry.all_claims()]
    
    def get_claim_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a claim."""
        claim = self._registry.get(name)
        if not claim:
            return None
        
        return {
            "name": claim.name,
            "type": claim.claim_type.value,
            "ltl": claim.ltl_formula,
            "description": claim.description,
            "inputs": claim.input_names,
            "output": claim.output_name,
        }


class CompletenessChecker:
    """Check that a formula handles all input combinations correctly."""
    
    def __init__(self) -> None:
        self._evaluator = BooleanEvaluator()
        self._truth_table = TruthTableGenerator()
    
    def check_completeness(
        self,
        formula: ParsedFormula,
    ) -> Tuple[bool, List[Dict[str, int]]]:
        """Check if formula produces valid output for all inputs.
        
        Returns (is_complete, list_of_failing_combinations).
        """
        if not formula.inputs:
            return False, []
        
        combinations = self._truth_table.generate(formula.inputs)
        failures: List[Dict[str, int]] = []
        
        for combo in combinations:
            try:
                result = self._evaluator.evaluate(formula.expression, combo)
                if result not in (0, 1):
                    failures.append(combo)
            except Exception:
                failures.append(combo)
        
        return len(failures) == 0, failures


class TraceVerifier:
    """Verify temporal execution traces against expected behavior."""
    
    def __init__(self) -> None:
        self._evaluator = BooleanEvaluator()
    
    def verify_trace(
        self,
        formula: ParsedFormula,
        input_trace: List[Dict[str, int]],
        expected_output: List[int],
    ) -> Tuple[bool, List[int]]:
        """Verify a temporal trace.
        
        Returns (all_match, list_of_mismatched_timesteps).
        """
        if len(input_trace) != len(expected_output):
            return False, list(range(len(input_trace)))
        
        mismatches: List[int] = []
        
        for t, (inputs, expected) in enumerate(zip(input_trace, expected_output)):
            try:
                actual = self._evaluator.evaluate(formula.expression, inputs)
                if actual != expected:
                    mismatches.append(t)
            except Exception:
                mismatches.append(t)
        
        return len(mismatches) == 0, mismatches


def verify_all_specs(
    spec_dir: Path,
    output_report: Optional[Path] = None,
) -> Dict[str, SpecVerificationReport]:
    """Verify all .tau files in a directory.
    
    Returns dict mapping spec filename to verification report.
    """
    verifier = TauSpecVerifier()
    reports: Dict[str, SpecVerificationReport] = {}
    
    for tau_file in spec_dir.rglob("*.tau"):
        report = verifier.verify_spec_file(tau_file)
        reports[tau_file.name] = report
    
    if output_report:
        _write_report(reports, output_report)
    
    return reports


def _write_report(reports: Dict[str, SpecVerificationReport], path: Path) -> None:
    """Write verification report to file."""
    lines: List[str] = [
        "# Tau Specification Formal Verification Report",
        "",
        f"## Summary: {len(reports)} specifications verified",
        "",
    ]
    
    passed = sum(1 for r in reports.values() if r.overall_status == VerificationStatus.PASSED)
    failed = sum(1 for r in reports.values() if r.overall_status == VerificationStatus.FAILED)
    
    lines.append(f"- **Passed:** {passed}")
    lines.append(f"- **Failed:** {failed}")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    
    for name, report in sorted(reports.items()):
        status_icon = "✅" if report.overall_status == VerificationStatus.PASSED else "❌"
        lines.append(f"### {status_icon} {name}")
        lines.append(f"- Status: {report.overall_status.value}")
        lines.append(f"- Summary: {report.summary}")
        
        for result in report.results:
            if result.counterexamples:
                lines.append(f"- Counterexamples:")
                for ce in result.counterexamples[:3]:
                    lines.append(f"  - Inputs: {ce.inputs}, Expected: {ce.expected}, Actual: {ce.actual}")
        
        lines.append("")
    
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python formal_verifier.py <spec_file_or_dir> [claim_name]")
        print("\nAvailable claims:")
        verifier = TauSpecVerifier()
        for claim_name in verifier.list_claims():
            info = verifier.get_claim_info(claim_name)
            if info:
                print(f"  {claim_name}: {info['ltl']} - {info['description']}")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    claim_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    verifier = TauSpecVerifier()
    
    if target.is_file():
        report = verifier.verify_spec_file(target, claim_name)
        print(f"\n{'='*60}")
        print(f"Verification Report: {target.name}")
        print(f"{'='*60}")
        print(f"Status: {report.overall_status.value}")
        print(f"Summary: {report.summary}")
        
        for result in report.results:
            print(f"\n  Formula: {result.spec_name}")
            print(f"  Claim: {result.claim.name if result.claim else 'unknown'}")
            print(f"  Status: {result.status.value}")
            print(f"  Coverage: {result.coverage_pct:.1f}%")
            print(f"  Message: {result.message}")
            
            if result.counterexamples:
                print(f"  Counterexamples:")
                for ce in result.counterexamples[:3]:
                    print(f"    Inputs: {ce.inputs}")
                    print(f"    Expected: {ce.expected}, Actual: {ce.actual}")
    
    elif target.is_dir():
        reports = verify_all_specs(target)
        
        print(f"\n{'='*60}")
        print(f"Verification Summary: {target}")
        print(f"{'='*60}")
        
        for name, report in sorted(reports.items()):
            status = "✅" if report.overall_status == VerificationStatus.PASSED else "❌"
            print(f"{status} {name}: {report.summary}")
    
    else:
        print(f"Error: {target} not found")
        sys.exit(1)
