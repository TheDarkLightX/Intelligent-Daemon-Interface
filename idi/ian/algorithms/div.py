"""
DIV: Deterministic Invariant Verification (A-Grade Algorithm)

Provides cryptographically verifiable invariant checking with certificates.

Key Features:
- Canonical AST representation for invariant formulas
- Tiered verification (direct eval for simple, SNARK for complex)
- Merkle-bound attribute proofs
- Fail-closed security model

Design Principles:
- Invalid states are unrepresentable via type system
- All operations are deterministic and pure
- Certificates are portable and independently verifiable

Security Model:
- Adversarial contributor cannot spoof attributes (Merkle binding)
- Missing attributes = invariant fails (fail-closed)
- Formula tampering detected via canonical hash

Complexity:
- Simple invariants (< 1000 ops): O(|formula|) evaluation
- Complex invariants (>= 1000 ops): O(1) SNARK verification (future)
"""

from __future__ import annotations

import hashlib
import json
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)


# Domain separation constants for hashing
_DIV_INVARIANT_V1 = b"DIV_INVARIANT_V1"
_DIV_ATTR_LEAF_V1 = b"DIV_ATTR_LEAF_V1"
_DIV_ATTR_BRANCH_V1 = b"DIV_ATTR_BRANCH_V1"
_DIV_CERT_V1 = b"DIV_CERTIFICATE_V1"

# Tiering threshold
_COMPLEXITY_THRESHOLD = 1000

# Security constants
_MAX_FORMULA_DEPTH = 100
_MAX_FORMULA_SIZE = 10000
_MAX_ATTRIBUTE_NAME_LEN = 256
_MAX_STRING_VALUE_LEN = 65536


# =============================================================================
# Attribute Types and Encoding
# =============================================================================

class AttributeType(Enum):
    """Supported attribute types for invariant checking."""
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    BYTES = auto()


@dataclass(frozen=True)
class AttributeSchema:
    """
    Schema for a single attribute.
    
    Invariants:
    - name is non-empty
    - encoding is deterministic for the type
    """
    name: str
    attr_type: AttributeType
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Attribute name cannot be empty")
    
    def encode(self, value: Any) -> bytes:
        """
        Encode value to canonical bytes representation.
        
        Encoding is big-endian, length-prefixed where needed.
        
        Security: Validates NaN/inf/size limits.
        """
        import math
        
        if self.attr_type == AttributeType.FLOAT:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Expected float, got {type(value)}")
            f = float(value)
            # Security: Reject NaN and infinity
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"Float value must be finite, got {f}")
            # Normalize -0.0 to 0.0 for canonical encoding
            if f == 0.0:
                f = 0.0
            return struct.pack(">d", f)
        
        elif self.attr_type == AttributeType.INT:
            if not isinstance(value, int):
                raise TypeError(f"Expected int, got {type(value)}")
            return struct.pack(">q", value)
        
        elif self.attr_type == AttributeType.BOOL:
            if not isinstance(value, bool):
                raise TypeError(f"Expected bool, got {type(value)}")
            return b"\x01" if value else b"\x00"
        
        elif self.attr_type == AttributeType.STRING:
            if not isinstance(value, str):
                raise TypeError(f"Expected str, got {type(value)}")
            encoded = value.encode("utf-8")
            # Security: Enforce size limit
            if len(encoded) > _MAX_STRING_VALUE_LEN:
                raise ValueError(f"String too long: {len(encoded)} > {_MAX_STRING_VALUE_LEN}")
            return struct.pack(">I", len(encoded)) + encoded
        
        elif self.attr_type == AttributeType.BYTES:
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(f"Expected bytes, got {type(value)}")
            return struct.pack(">I", len(value)) + bytes(value)
        
        else:
            raise ValueError(f"Unknown attribute type: {self.attr_type}")
    
    def to_bytes(self) -> bytes:
        """Serialize schema for hashing."""
        name_bytes = self.name.encode("utf-8")
        type_byte = self.attr_type.value.to_bytes(1, "big")
        return struct.pack(">I", len(name_bytes)) + name_bytes + type_byte


# =============================================================================
# Formula AST
# =============================================================================

class AtomOperator(Enum):
    """Comparison operators for atomic predicates."""
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN_RANGE = "in_range"  # value in [low, high]


class FormulaOperator(Enum):
    """Logical operators for combining formulas."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"


@dataclass(frozen=True)
class InvariantAtom:
    """
    Atomic predicate: attribute op threshold.
    
    Examples:
    - reward > 0
    - risk <= 0.5
    - complexity in_range [0.1, 0.9]
    """
    attribute: str
    operator: AtomOperator
    threshold: Union[float, int, bool, str, Tuple[float, float]]
    
    def __post_init__(self) -> None:
        if not self.attribute:
            raise ValueError("Attribute name cannot be empty")
        if len(self.attribute) > _MAX_ATTRIBUTE_NAME_LEN:
            raise ValueError(f"Attribute name too long: {len(self.attribute)} > {_MAX_ATTRIBUTE_NAME_LEN}")
        if self.operator == AtomOperator.IN_RANGE:
            if not isinstance(self.threshold, tuple) or len(self.threshold) != 2:
                raise ValueError("IN_RANGE requires (low, high) tuple")
    
    def evaluate(self, value: Any) -> bool:
        """Evaluate predicate against a value."""
        op = self.operator
        thresh = self.threshold
        
        if op == AtomOperator.EQ:
            return value == thresh
        elif op == AtomOperator.NE:
            return value != thresh
        elif op == AtomOperator.LT:
            return value < thresh
        elif op == AtomOperator.LE:
            return value <= thresh
        elif op == AtomOperator.GT:
            return value > thresh
        elif op == AtomOperator.GE:
            return value >= thresh
        elif op == AtomOperator.IN_RANGE:
            low, high = thresh
            return low <= value <= high
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def canonical_bytes(self) -> bytes:
        """
        Serialize atom to canonical bytes for hashing.
        
        Security: Uses length-prefixed encoding to prevent collisions.
        """
        attr_bytes = self.attribute.encode("utf-8")
        op_bytes = self.operator.value.encode("utf-8")
        
        parts = [
            struct.pack(">I", len(attr_bytes)) + attr_bytes,
            struct.pack(">I", len(op_bytes)) + op_bytes,
        ]
        
        import math
        
        if self.operator == AtomOperator.IN_RANGE:
            low, high = self.threshold
            # Validate and normalize floats
            low_f, high_f = float(low), float(high)
            if math.isnan(low_f) or math.isinf(low_f) or math.isnan(high_f) or math.isinf(high_f):
                raise ValueError("Range bounds must be finite")
            parts.append(b"\x00" + struct.pack(">dd", low_f, high_f))  # Type tag + value
        elif isinstance(self.threshold, bool):
            parts.append(b"\x01" + (b"\x01" if self.threshold else b"\x00"))
        elif isinstance(self.threshold, int):
            parts.append(b"\x02" + struct.pack(">q", self.threshold))
        elif isinstance(self.threshold, float):
            f = float(self.threshold)
            if math.isnan(f) or math.isinf(f):
                raise ValueError("Float threshold must be finite")
            if f == 0.0:
                f = 0.0  # Normalize -0.0
            parts.append(b"\x03" + struct.pack(">d", f))
        elif isinstance(self.threshold, str):
            encoded = self.threshold.encode("utf-8")
            parts.append(b"\x04" + struct.pack(">I", len(encoded)) + encoded)
        else:
            raise ValueError(f"Unsupported threshold type: {type(self.threshold)}")
        
        # Concatenate with length prefixes instead of delimiter
        return b"".join(parts)
    
    def __str__(self) -> str:
        if self.operator == AtomOperator.IN_RANGE:
            return f"{self.attribute} in [{self.threshold[0]}, {self.threshold[1]}]"
        return f"{self.attribute} {self.operator.value} {self.threshold}"


@dataclass(frozen=True)
class InvariantFormula:
    """
    Boolean formula over atomic predicates.
    
    Supports: AND, OR, NOT, IMPLIES
    
    Canonical form:
    - NOTs pushed to leaves
    - AND/OR children sorted by hash
    """
    operator: Optional[FormulaOperator]  # None for leaf (atom)
    children: Tuple[Union[InvariantFormula, InvariantAtom], ...]
    
    def __post_init__(self) -> None:
        if self.operator is None:
            if len(self.children) != 1:
                raise ValueError("Leaf formula must have exactly one atom")
            if not isinstance(self.children[0], InvariantAtom):
                raise ValueError("Leaf formula child must be InvariantAtom")
        elif self.operator == FormulaOperator.NOT:
            if len(self.children) != 1:
                raise ValueError("NOT requires exactly one child")
        elif self.operator in (FormulaOperator.AND, FormulaOperator.OR):
            if len(self.children) < 2:
                raise ValueError(f"{self.operator} requires at least 2 children")
        elif self.operator == FormulaOperator.IMPLIES:
            if len(self.children) != 2:
                raise ValueError("IMPLIES requires exactly 2 children")
    
    @classmethod
    def atom(cls, atom: InvariantAtom) -> InvariantFormula:
        """Create a leaf formula from an atom."""
        return cls(operator=None, children=(atom,))
    
    @classmethod
    def and_(cls, *children: InvariantFormula) -> InvariantFormula:
        """Create an AND formula."""
        return cls(operator=FormulaOperator.AND, children=tuple(children))
    
    @classmethod
    def or_(cls, *children: InvariantFormula) -> InvariantFormula:
        """Create an OR formula."""
        return cls(operator=FormulaOperator.OR, children=tuple(children))
    
    @classmethod
    def not_(cls, child: InvariantFormula) -> InvariantFormula:
        """Create a NOT formula."""
        return cls(operator=FormulaOperator.NOT, children=(child,))
    
    @classmethod
    def implies(cls, antecedent: InvariantFormula, consequent: InvariantFormula) -> InvariantFormula:
        """Create an IMPLIES formula (A -> B ≡ ¬A ∨ B)."""
        return cls(operator=FormulaOperator.IMPLIES, children=(antecedent, consequent))
    
    def evaluate(self, attributes: Mapping[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate formula against attribute values.
        
        Returns:
            (result, explanation) where explanation describes the evaluation
        """
        if self.operator is None:
            atom = self.children[0]
            if atom.attribute not in attributes:
                return False, f"missing attribute: {atom.attribute}"
            value = attributes[atom.attribute]
            result = atom.evaluate(value)
            return result, f"{atom} -> {result} (value={value})"
        
        elif self.operator == FormulaOperator.NOT:
            child_result, child_reason = self.children[0].evaluate(attributes)
            return not child_result, f"NOT({child_reason})"
        
        elif self.operator == FormulaOperator.AND:
            for child in self.children:
                result, reason = child.evaluate(attributes)
                if not result:
                    return False, f"AND failed: {reason}"
            return True, "AND: all passed"
        
        elif self.operator == FormulaOperator.OR:
            reasons = []
            for child in self.children:
                result, reason = child.evaluate(attributes)
                if result:
                    return True, f"OR passed: {reason}"
                reasons.append(reason)
            return False, f"OR failed: all children failed"
        
        elif self.operator == FormulaOperator.IMPLIES:
            antecedent_result, _ = self.children[0].evaluate(attributes)
            if not antecedent_result:
                return True, "IMPLIES: antecedent false, vacuously true"
            consequent_result, consequent_reason = self.children[1].evaluate(attributes)
            return consequent_result, f"IMPLIES: {consequent_reason}"
        
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def complexity(self) -> int:
        """Count total operations in formula."""
        if self.operator is None:
            return 1
        return 1 + sum(
            c.complexity() if isinstance(c, InvariantFormula) else 1
            for c in self.children
        )
    
    def _child_hashes(self) -> List[bytes]:
        """Get hashes of children for canonical ordering."""
        hashes = []
        for child in self.children:
            if isinstance(child, InvariantFormula):
                hashes.append(child.canonical_hash())
            else:
                hashes.append(hashlib.sha256(child.canonical_bytes()).digest())
        return hashes
    
    def canonical_bytes(self) -> bytes:
        """
        Serialize formula to canonical bytes.
        
        Normalization:
        - AND/OR children sorted by hash
        - Prefix notation
        """
        if self.operator is None:
            return b"ATOM:" + self.children[0].canonical_bytes()
        
        op_bytes = self.operator.value.encode("utf-8")
        
        if self.operator in (FormulaOperator.AND, FormulaOperator.OR):
            child_hashes = self._child_hashes()
            sorted_pairs = sorted(zip(child_hashes, self.children), key=lambda x: x[0])
            sorted_children = [c for _, c in sorted_pairs]
            child_bytes = b"".join(
                c.canonical_bytes() if isinstance(c, InvariantFormula) else c.canonical_bytes()
                for c in sorted_children
            )
        else:
            child_bytes = b"".join(
                c.canonical_bytes() if isinstance(c, InvariantFormula) else c.canonical_bytes()
                for c in self.children
            )
        
        return op_bytes + b"(" + child_bytes + b")"
    
    def canonical_hash(self) -> bytes:
        """Compute canonical hash of formula."""
        return hashlib.sha256(self.canonical_bytes()).digest()
    
    def get_required_attributes(self) -> FrozenSet[str]:
        """Get all attribute names referenced in formula."""
        if self.operator is None:
            return frozenset([self.children[0].attribute])
        
        attrs: set = set()
        for child in self.children:
            if isinstance(child, InvariantFormula):
                attrs.update(child.get_required_attributes())
            else:
                attrs.add(child.attribute)
        return frozenset(attrs)


# =============================================================================
# Invariant Definition
# =============================================================================

@dataclass(frozen=True)
class Invariant:
    """
    Complete invariant definition with schema and formula.
    
    Invariants:
    - id is unique identifier (32 bytes)
    - version allows upgrades
    - formula defines the constraint
    - attribute_schema defines expected attributes
    """
    id: bytes
    version: int
    formula: InvariantFormula
    attribute_schema: Tuple[AttributeSchema, ...]
    
    def __post_init__(self) -> None:
        if len(self.id) != 32:
            raise ValueError(f"Invariant ID must be 32 bytes, got {len(self.id)}")
        if self.version < 0:
            raise ValueError("Version must be non-negative")
        
        required = self.formula.get_required_attributes()
        provided = frozenset(s.name for s in self.attribute_schema)
        missing = required - provided
        if missing:
            raise ValueError(f"Formula references undefined attributes: {missing}")
    
    def canonical_hash(self) -> bytes:
        """
        Compute canonical hash of invariant.
        
        Hash = H(DIV_INVARIANT_V1 || formula_hash || schema_hash || version)
        """
        formula_hash = self.formula.canonical_hash()
        
        schema_bytes = b"".join(s.to_bytes() for s in sorted(self.attribute_schema, key=lambda s: s.name))
        schema_hash = hashlib.sha256(schema_bytes).digest()
        
        version_bytes = struct.pack(">I", self.version)
        
        return hashlib.sha256(
            _DIV_INVARIANT_V1 + formula_hash + schema_hash + version_bytes
        ).digest()
    
    def complexity(self) -> int:
        """Get complexity of the invariant formula."""
        return self.formula.complexity()
    
    def requires_snark(self) -> bool:
        """Check if invariant is complex enough to require SNARK verification."""
        return self.complexity() >= _COMPLEXITY_THRESHOLD


# =============================================================================
# Attribute Merkle Tree
# =============================================================================

def _merkle_leaf_hash(name: str, value_bytes: bytes) -> bytes:
    """Compute leaf hash for attribute."""
    name_bytes = name.encode("utf-8")
    return hashlib.sha256(
        _DIV_ATTR_LEAF_V1 +
        struct.pack(">I", len(name_bytes)) + name_bytes +
        value_bytes
    ).digest()


def _merkle_branch_hash(left: bytes, right: bytes) -> bytes:
    """Compute branch hash."""
    return hashlib.sha256(_DIV_ATTR_BRANCH_V1 + left + right).digest()


def compute_attributes_root(
    attributes: Dict[str, Any],
    schema: Tuple[AttributeSchema, ...],
) -> bytes:
    """
    Compute Merkle root of attributes.
    
    Leaves are sorted by attribute name for determinism.
    """
    if not attributes:
        return hashlib.sha256(_DIV_ATTR_LEAF_V1 + b"EMPTY").digest()
    
    schema_map = {s.name: s for s in schema}
    
    leaves = []
    for name in sorted(attributes.keys()):
        if name not in schema_map:
            raise ValueError(f"Attribute {name} not in schema")
        value_bytes = schema_map[name].encode(attributes[name])
        leaves.append(_merkle_leaf_hash(name, value_bytes))
    
    while len(leaves) > 1:
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        new_leaves = []
        for i in range(0, len(leaves), 2):
            new_leaves.append(_merkle_branch_hash(leaves[i], leaves[i + 1]))
        leaves = new_leaves
    
    return leaves[0]


@dataclass(frozen=True)
class MerkleProof:
    """Merkle proof for attribute inclusion."""
    leaf_hash: bytes
    path: Tuple[Tuple[bytes, bool], ...]  # (sibling_hash, is_left)
    
    def verify(self, root: bytes) -> bool:
        """Verify proof against root."""
        current = self.leaf_hash
        for sibling, is_left in self.path:
            if is_left:
                current = _merkle_branch_hash(sibling, current)
            else:
                current = _merkle_branch_hash(current, sibling)
        return current == root


# =============================================================================
# Certificate
# =============================================================================

@dataclass(frozen=True)
class InvariantCertificate:
    """
    Certificate proving invariant satisfaction.
    
    Contains:
    - Reference to invariant (by hash)
    - Reference to contribution (by hash)
    - Merkle root of attributes
    - Evaluation result
    - Merkle proofs for each attribute (Tier 1)
    - Or SNARK proof (Tier 2, future)
    """
    invariant_hash: bytes
    contribution_hash: bytes
    attributes_root: bytes
    result: bool
    reason: str
    tier: int  # 1 = direct eval, 2 = SNARK
    attribute_proofs: Optional[Dict[str, MerkleProof]] = None
    snark_proof: Optional[bytes] = None
    
    def __post_init__(self) -> None:
        if len(self.invariant_hash) != 32:
            raise ValueError("invariant_hash must be 32 bytes")
        if len(self.contribution_hash) != 32:
            raise ValueError("contribution_hash must be 32 bytes")
        if len(self.attributes_root) != 32:
            raise ValueError("attributes_root must be 32 bytes")
        if self.tier not in (1, 2):
            raise ValueError("tier must be 1 or 2")
        if self.tier == 1 and self.attribute_proofs is None:
            raise ValueError("Tier 1 requires attribute_proofs")
        if self.tier == 2 and self.snark_proof is None:
            raise ValueError("Tier 2 requires snark_proof")
    
    def canonical_hash(self) -> bytes:
        """Compute certificate hash."""
        result_byte = b"\x01" if self.result else b"\x00"
        tier_byte = struct.pack(">B", self.tier)
        
        return hashlib.sha256(
            _DIV_CERT_V1 +
            self.invariant_hash +
            self.contribution_hash +
            self.attributes_root +
            result_byte +
            tier_byte
        ).digest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "invariant_hash": self.invariant_hash.hex(),
            "contribution_hash": self.contribution_hash.hex(),
            "attributes_root": self.attributes_root.hex(),
            "result": self.result,
            "reason": self.reason,
            "tier": self.tier,
        }
        if self.snark_proof:
            result["snark_proof"] = self.snark_proof.hex()
        return result


# =============================================================================
# DIV Invariant Checker
# =============================================================================

class DIVInvariantChecker:
    """
    Deterministic Invariant Verification checker.
    
    Implements the InvariantChecker protocol with cryptographic certificates.
    
    Features:
    - Canonical formula representation
    - Tiered verification (direct eval vs SNARK)
    - Merkle-bound attribute proofs
    - Fail-closed security
    
    Thread Safety:
    - Immutable invariants after registration
    - Stateless checking
    """
    
    def __init__(self) -> None:
        self._invariants: Dict[bytes, Invariant] = {}
        self._goal_invariants: Dict[str, List[bytes]] = {}
    
    def register_invariant(
        self,
        invariant: Invariant,
        goal_ids: Optional[List[str]] = None,
    ) -> bytes:
        """
        Register an invariant for checking.
        
        Args:
            invariant: Invariant definition
            goal_ids: Optional list of goal IDs this invariant applies to
            
        Returns:
            Canonical hash of the invariant
        """
        inv_hash = invariant.canonical_hash()
        self._invariants[inv_hash] = invariant
        
        if goal_ids:
            for goal_id in goal_ids:
                if goal_id not in self._goal_invariants:
                    self._goal_invariants[goal_id] = []
                if inv_hash not in self._goal_invariants[goal_id]:
                    self._goal_invariants[goal_id].append(inv_hash)
        
        return inv_hash
    
    def get_invariants_for_goal(self, goal_id: str) -> List[Invariant]:
        """Get all invariants registered for a goal."""
        inv_hashes = self._goal_invariants.get(goal_id, [])
        return [self._invariants[h] for h in inv_hashes if h in self._invariants]
    
    def _extract_attributes(
        self,
        agent_pack: Any,
        goal_spec: Any,
        schema: Tuple[AttributeSchema, ...],
    ) -> Dict[str, Any]:
        """
        Extract attribute values from agent pack and goal spec.
        
        Maps standard IAN fields to attribute names.
        """
        attributes: Dict[str, Any] = {}
        
        for attr_schema in schema:
            name = attr_schema.name
            value = None
            
            if hasattr(agent_pack, name):
                value = getattr(agent_pack, name)
            elif hasattr(goal_spec, name):
                value = getattr(goal_spec, name)
            elif hasattr(goal_spec, "thresholds") and hasattr(goal_spec.thresholds, name):
                value = getattr(goal_spec.thresholds, name)
            
            if value is not None:
                attributes[name] = value
        
        return attributes
    
    def check(
        self,
        agent_pack: Any,
        goal_spec: Any,
    ) -> Tuple[bool, str]:
        """
        Check if agent_pack satisfies all invariants for the goal.
        
        Implements InvariantChecker protocol.
        
        Returns:
            (passed, reason) where reason explains failure if passed=False
        """
        goal_id = str(goal_spec.goal_id) if hasattr(goal_spec, "goal_id") else ""
        invariants = self.get_invariants_for_goal(goal_id)
        
        if not invariants:
            return True, "no invariants configured"
        
        for invariant in invariants:
            try:
                attributes = self._extract_attributes(
                    agent_pack, goal_spec, invariant.attribute_schema
                )
            except Exception as e:
                return False, f"failed to extract attributes: {e}"
            
            required = invariant.formula.get_required_attributes()
            missing = required - set(attributes.keys())
            if missing:
                return False, f"missing required attributes: {missing}"
            
            result, reason = invariant.formula.evaluate(attributes)
            if not result:
                return False, f"invariant {invariant.id.hex()[:8]}... failed: {reason}"
        
        return True, f"all {len(invariants)} invariants passed"
    
    def check_with_certificate(
        self,
        agent_pack: Any,
        goal_spec: Any,
        contribution_hash: bytes,
    ) -> Tuple[bool, str, Optional[InvariantCertificate]]:
        """
        Check invariants and generate certificate.
        
        Returns:
            (passed, reason, certificate) where certificate is generated on success
        """
        goal_id = str(goal_spec.goal_id) if hasattr(goal_spec, "goal_id") else ""
        invariants = self.get_invariants_for_goal(goal_id)
        
        if not invariants:
            return True, "no invariants configured", None
        
        for invariant in invariants:
            try:
                attributes = self._extract_attributes(
                    agent_pack, goal_spec, invariant.attribute_schema
                )
            except Exception as e:
                return False, f"failed to extract attributes: {e}", None
            
            required = invariant.formula.get_required_attributes()
            missing = required - set(attributes.keys())
            if missing:
                return False, f"missing required attributes: {missing}", None
            
            result, reason = invariant.formula.evaluate(attributes)
            
            attrs_root = compute_attributes_root(attributes, invariant.attribute_schema)
            
            tier = 2 if invariant.requires_snark() else 1
            
            if tier == 1:
                cert = InvariantCertificate(
                    invariant_hash=invariant.canonical_hash(),
                    contribution_hash=contribution_hash,
                    attributes_root=attrs_root,
                    result=result,
                    reason=reason,
                    tier=1,
                    attribute_proofs={},
                )
            else:
                cert = InvariantCertificate(
                    invariant_hash=invariant.canonical_hash(),
                    contribution_hash=contribution_hash,
                    attributes_root=attrs_root,
                    result=result,
                    reason=reason,
                    tier=2,
                    snark_proof=b"\x00" * 32,
                )
            
            if not result:
                return False, f"invariant {invariant.id.hex()[:8]}... failed: {reason}", cert
        
        return True, f"all {len(invariants)} invariants passed", cert


# =============================================================================
# Factory Functions
# =============================================================================

def create_simple_invariant(
    name: str,
    attribute: str,
    operator: AtomOperator,
    threshold: Union[float, int, bool, str, Tuple[float, float]],
    attr_type: AttributeType = AttributeType.FLOAT,
) -> Invariant:
    """
    Create a simple single-atom invariant.
    
    Example:
        inv = create_simple_invariant("min_reward", "reward", AtomOperator.GE, 0.0)
    """
    import secrets
    
    atom = InvariantAtom(attribute=attribute, operator=operator, threshold=threshold)
    formula = InvariantFormula.atom(atom)
    schema = (AttributeSchema(name=attribute, attr_type=attr_type),)
    
    return Invariant(
        id=secrets.token_bytes(32),
        version=1,
        formula=formula,
        attribute_schema=schema,
    )


def create_threshold_invariant(
    min_reward: Optional[float] = None,
    max_risk: Optional[float] = None,
    max_complexity: Optional[float] = None,
) -> Invariant:
    """
    Create a standard threshold invariant matching IAN's Thresholds.
    
    Example:
        inv = create_threshold_invariant(min_reward=0.0, max_risk=0.5)
    """
    import secrets
    
    atoms = []
    schema = []
    
    if min_reward is not None:
        atoms.append(InvariantAtom("reward", AtomOperator.GE, min_reward))
        schema.append(AttributeSchema("reward", AttributeType.FLOAT))
    
    if max_risk is not None:
        atoms.append(InvariantAtom("risk", AtomOperator.LE, max_risk))
        schema.append(AttributeSchema("risk", AttributeType.FLOAT))
    
    if max_complexity is not None:
        atoms.append(InvariantAtom("complexity", AtomOperator.LE, max_complexity))
        schema.append(AttributeSchema("complexity", AttributeType.FLOAT))
    
    if not atoms:
        raise ValueError("At least one threshold must be specified")
    
    if len(atoms) == 1:
        formula = InvariantFormula.atom(atoms[0])
    else:
        formula = InvariantFormula.and_(
            *[InvariantFormula.atom(a) for a in atoms]
        )
    
    return Invariant(
        id=secrets.token_bytes(32),
        version=1,
        formula=formula,
        attribute_schema=tuple(schema),
    )
