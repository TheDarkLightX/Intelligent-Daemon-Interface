"""
VEP: Verifiable Evaluation Protocol (A-Grade Algorithm)

Provides commitment-traced evaluation with probabilistic auditing.

Key Features:
- Evaluation produces commitment trace (Merkle tree of steps)
- Threshold VRF determines audit indices after commit
- Targeted challenges for specific steps
- Economic security via staking and slashing

Design Principles:
- Build verification INTO evaluation, not after
- Zero proving overhead in happy path
- Retroactive verifiability via trace commitments
- Economic security via slashing

Security Model:
- Soundness >= 1 - 1e-6 from evaluator redundancy
- Audit selection grinding prevented by threshold VRF
- Selective abort prevented by pre-commit with slashing
- Data availability via encrypted trace on DA layer

Complexity:
- Trace generation: O(N) where N = steps
- Audit verification: O(k) where k = audit samples
- Certificate size: O(1) for result, O(log N) per audit
"""

from __future__ import annotations

import hashlib
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence


# Domain separation constants
_VEP_STEP_V1 = b"VEP_STEP_V1"
_VEP_TRACE_LEAF_V1 = b"VEP_TRACE_LEAF_V1"
_VEP_TRACE_BRANCH_V1 = b"VEP_TRACE_BRANCH_V1"
_VEP_COMMIT_V1 = b"VEP_COMMIT_V1"
_VEP_PRECOMMIT_V1 = b"VEP_PRECOMMIT_V1"
_VEP_AUDIT_SEED_V1 = b"VEP_AUDIT_SEED_V1"


# =============================================================================
# Trace Data Structures
# =============================================================================

@dataclass(frozen=True)
class EvaluationStep:
    """
    Single step in an evaluation trace.
    
    Captures state transition: (state, action) -> (next_state, reward)
    """
    step_index: int
    state_hash: bytes  # H(state) to avoid storing full state
    action: bytes  # Serialized action
    next_state_hash: bytes
    reward: float
    info: Optional[bytes] = None  # Optional metadata
    
    def __post_init__(self) -> None:
        if len(self.state_hash) != 32:
            raise ValueError("state_hash must be 32 bytes")
        if len(self.next_state_hash) != 32:
            raise ValueError("next_state_hash must be 32 bytes")
    
    def canonical_bytes(self) -> bytes:
        """Serialize step for hashing."""
        return (
            _VEP_STEP_V1 +
            struct.pack(">Q", self.step_index) +
            self.state_hash +
            struct.pack(">I", len(self.action)) + self.action +
            self.next_state_hash +
            struct.pack(">d", self.reward) +
            (struct.pack(">I", len(self.info)) + self.info if self.info else b"\x00\x00\x00\x00")
        )
    
    def step_hash(self) -> bytes:
        """Compute hash of this step."""
        return hashlib.sha256(self.canonical_bytes()).digest()


@dataclass
class EvaluationTrace:
    """
    Complete evaluation trace with Merkle commitment.
    
    Stores all steps and provides Merkle proofs for auditing.
    """
    steps: List[EvaluationStep] = field(default_factory=list)
    _leaf_hashes: List[bytes] = field(default_factory=list, repr=False)
    _merkle_tree: List[List[bytes]] = field(default_factory=list, repr=False)
    _root: Optional[bytes] = field(default=None, repr=False)
    
    def add_step(self, step: EvaluationStep) -> None:
        """Add a step to the trace."""
        if self._root is not None:
            raise RuntimeError("Cannot add steps after finalization")
        
        self.steps.append(step)
        leaf_hash = hashlib.sha256(_VEP_TRACE_LEAF_V1 + step.step_hash()).digest()
        self._leaf_hashes.append(leaf_hash)
    
    def finalize(self) -> bytes:
        """
        Finalize trace and compute Merkle root.
        
        Returns:
            Merkle root of the trace
        """
        if self._root is not None:
            return self._root
        
        if not self._leaf_hashes:
            self._root = hashlib.sha256(_VEP_TRACE_LEAF_V1 + b"EMPTY").digest()
            return self._root
        
        leaves = self._leaf_hashes.copy()
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        self._merkle_tree = [leaves]
        
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = hashlib.sha256(_VEP_TRACE_BRANCH_V1 + left + right).digest()
                next_level.append(parent)
            self._merkle_tree.append(next_level)
            current_level = next_level
        
        self._root = current_level[0]
        return self._root
    
    @property
    def root(self) -> bytes:
        """Get Merkle root (finalizes if needed)."""
        if self._root is None:
            return self.finalize()
        return self._root
    
    def get_step(self, index: int) -> EvaluationStep:
        """Get step by index."""
        if index < 0 or index >= len(self.steps):
            raise IndexError(f"Step index {index} out of range [0, {len(self.steps)})")
        return self.steps[index]
    
    def get_proof(self, index: int) -> TraceMerkleProof:
        """
        Get Merkle proof for step at index.
        
        Must be called after finalize().
        """
        if self._root is None:
            self.finalize()
        
        if index < 0 or index >= len(self.steps):
            raise IndexError(f"Step index {index} out of range")
        
        path: List[Tuple[bytes, bool]] = []
        idx = index
        
        for level in self._merkle_tree[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                is_left = False
            else:
                sibling_idx = idx - 1
                is_left = True
            
            if sibling_idx < len(level):
                path.append((level[sibling_idx], is_left))
            else:
                path.append((level[idx], is_left))
            
            idx //= 2
        
        return TraceMerkleProof(
            step_index=index,
            step_hash=self._leaf_hashes[index],
            path=tuple(path),
        )
    
    def __len__(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class TraceMerkleProof:
    """Merkle proof for a trace step."""
    step_index: int
    step_hash: bytes
    path: Tuple[Tuple[bytes, bool], ...]  # (sibling, is_left)
    
    def verify(self, root: bytes) -> bool:
        """Verify proof against root."""
        current = self.step_hash
        for sibling, is_left in self.path:
            if is_left:
                current = hashlib.sha256(_VEP_TRACE_BRANCH_V1 + sibling + current).digest()
            else:
                current = hashlib.sha256(_VEP_TRACE_BRANCH_V1 + current + sibling).digest()
        return current == root
    
    def to_bytes(self) -> bytes:
        """Serialize proof."""
        path_bytes = b"".join(
            sibling + (b"\x01" if is_left else b"\x00")
            for sibling, is_left in self.path
        )
        return (
            struct.pack(">Q", self.step_index) +
            self.step_hash +
            struct.pack(">I", len(self.path)) +
            path_bytes
        )


# =============================================================================
# Commitments
# =============================================================================

@dataclass(frozen=True)
class TraceCommitment:
    """
    Commitment to an evaluation trace.
    
    Published before audit selection to prevent grinding.
    """
    evaluator_id: bytes
    contribution_hash: bytes
    env_hash: bytes
    seed: int
    trace_root: bytes
    metrics_hash: bytes
    timestamp: int
    
    def __post_init__(self) -> None:
        if len(self.evaluator_id) != 32:
            raise ValueError("evaluator_id must be 32 bytes")
        if len(self.contribution_hash) != 32:
            raise ValueError("contribution_hash must be 32 bytes")
        if len(self.env_hash) != 32:
            raise ValueError("env_hash must be 32 bytes")
        if len(self.trace_root) != 32:
            raise ValueError("trace_root must be 32 bytes")
        if len(self.metrics_hash) != 32:
            raise ValueError("metrics_hash must be 32 bytes")
    
    def canonical_bytes(self) -> bytes:
        """Serialize for hashing."""
        return (
            _VEP_COMMIT_V1 +
            self.evaluator_id +
            self.contribution_hash +
            self.env_hash +
            struct.pack(">Q", self.seed) +
            self.trace_root +
            self.metrics_hash +
            struct.pack(">Q", self.timestamp)
        )
    
    def commitment_hash(self) -> bytes:
        """Compute commitment hash."""
        return hashlib.sha256(self.canonical_bytes()).digest()


@dataclass(frozen=True)
class PreCommitment:
    """
    Pre-commitment before evaluation (anti-abort).
    
    Published before starting evaluation.
    """
    evaluator_id: bytes
    contribution_hash: bytes
    slot: int
    env_hash: bytes
    timestamp: int
    
    def canonical_bytes(self) -> bytes:
        return (
            _VEP_PRECOMMIT_V1 +
            self.evaluator_id +
            self.contribution_hash +
            struct.pack(">Q", self.slot) +
            self.env_hash +
            struct.pack(">Q", self.timestamp)
        )
    
    def precommit_hash(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


# =============================================================================
# Audit Selection
# =============================================================================

def derive_audit_indices(
    trace_root: bytes,
    evaluator_set_hash: bytes,
    num_steps: int,
    num_samples: int,
    vrf_output: Optional[bytes] = None,
) -> List[int]:
    """
    Derive audit indices from trace commitment.
    
    Uses VRF output if available, otherwise deterministic derivation.
    Anti-grinding: indices depend on trace_root (committed before selection).
    
    Args:
        trace_root: Merkle root of trace
        evaluator_set_hash: Hash of evaluator set
        num_steps: Total steps in trace
        num_samples: Number of samples to select
        vrf_output: Optional threshold VRF output
        
    Returns:
        List of step indices to audit (deduplicated)
    """
    if num_steps == 0:
        return []
    
    if num_samples >= num_steps:
        return list(range(num_steps))
    
    seed = vrf_output if vrf_output else evaluator_set_hash
    audit_seed = hashlib.sha256(
        _VEP_AUDIT_SEED_V1 + trace_root + seed
    ).digest()
    
    indices: set = set()
    counter = 0
    
    while len(indices) < num_samples and counter < num_samples * 10:
        h = hashlib.sha256(audit_seed + struct.pack(">I", counter)).digest()
        candidate = int.from_bytes(h[:8], "big")
        
        max_valid = (2**64 // num_steps) * num_steps
        if candidate < max_valid:
            idx = candidate % num_steps
            indices.add(idx)
        
        counter += 1
    
    return sorted(indices)


# =============================================================================
# Audit Structures
# =============================================================================

class AuditStatus(Enum):
    """Status of an audit."""
    PENDING = auto()
    PASSED = auto()
    FAILED = auto()
    CHALLENGED = auto()


@dataclass
class AuditRequest:
    """Request to audit specific steps."""
    trace_commitment: TraceCommitment
    step_indices: List[int]
    requester_id: bytes
    timestamp: int
    
    def request_hash(self) -> bytes:
        indices_bytes = b"".join(struct.pack(">Q", i) for i in self.step_indices)
        return hashlib.sha256(
            self.trace_commitment.commitment_hash() +
            indices_bytes +
            self.requester_id +
            struct.pack(">Q", self.timestamp)
        ).digest()


@dataclass
class AuditResult:
    """Result of auditing steps."""
    request_hash: bytes
    status: AuditStatus
    verified_steps: List[int]
    failed_step: Optional[int] = None
    failure_reason: Optional[str] = None
    proofs: List[TraceMerkleProof] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_hash": self.request_hash.hex(),
            "status": self.status.name,
            "verified_steps": self.verified_steps,
            "failed_step": self.failed_step,
            "failure_reason": self.failure_reason,
        }


# =============================================================================
# VEP Evaluation Harness
# =============================================================================

class VEPEvaluationHarness:
    """
    Verifiable Evaluation Protocol harness.
    
    Wraps an underlying evaluation function and produces:
    - Evaluation metrics
    - Commitment trace
    - Merkle proofs for auditing
    
    Implements EvaluationHarness protocol.
    """
    
    def __init__(
        self,
        evaluator_id: bytes,
        env_hash: bytes,
        eval_fn: Optional[Callable] = None,
        num_audit_samples: int = 32,
    ) -> None:
        """
        Initialize VEP harness.
        
        Args:
            evaluator_id: 32-byte evaluator identifier
            env_hash: 32-byte hash of evaluation environment
            eval_fn: Optional evaluation function
            num_audit_samples: Number of steps to sample for auditing
        """
        if len(evaluator_id) != 32:
            raise ValueError("evaluator_id must be 32 bytes")
        if len(env_hash) != 32:
            raise ValueError("env_hash must be 32 bytes")
        
        self.evaluator_id = evaluator_id
        self.env_hash = env_hash
        self.eval_fn = eval_fn
        self.num_audit_samples = num_audit_samples
        
        self._traces: Dict[bytes, EvaluationTrace] = {}
        self._commitments: Dict[bytes, TraceCommitment] = {}
        self._precommits: Dict[bytes, PreCommitment] = {}
    
    def precommit(
        self,
        contribution_hash: bytes,
        slot: int,
    ) -> PreCommitment:
        """
        Create pre-commitment before evaluation.
        
        Must be published before evaluation starts.
        Failure to complete evaluation after precommit = slashing.
        """
        precommit = PreCommitment(
            evaluator_id=self.evaluator_id,
            contribution_hash=contribution_hash,
            slot=slot,
            env_hash=self.env_hash,
            timestamp=int(time.time() * 1000),
        )
        
        self._precommits[contribution_hash] = precommit
        return precommit
    
    def evaluate(
        self,
        agent_pack: Any,
        goal_spec: Any,
        seed: int,
    ) -> Optional[Any]:
        """
        Evaluate agent and produce traced metrics.
        
        Implements EvaluationHarness protocol.
        
        Returns:
            Metrics if evaluation succeeded, None on error
        """
        from ..models import Metrics
        
        contribution_hash = (
            agent_pack.pack_hash
            if hasattr(agent_pack, "pack_hash")
            else hashlib.sha256(str(agent_pack).encode()).digest()
        )
        
        trace = EvaluationTrace()
        
        try:
            if self.eval_fn:
                metrics, steps = self.eval_fn(agent_pack, goal_spec, seed, trace)
            else:
                metrics, steps = self._default_evaluation(agent_pack, goal_spec, seed, trace)
        except Exception as e:
            return None
        
        trace.finalize()
        self._traces[contribution_hash] = trace
        
        metrics_bytes = self._serialize_metrics(metrics)
        metrics_hash = hashlib.sha256(metrics_bytes).digest()
        
        commitment = TraceCommitment(
            evaluator_id=self.evaluator_id,
            contribution_hash=contribution_hash,
            env_hash=self.env_hash,
            seed=seed,
            trace_root=trace.root,
            metrics_hash=metrics_hash,
            timestamp=int(time.time() * 1000),
        )
        
        self._commitments[contribution_hash] = commitment
        
        return metrics
    
    def _default_evaluation(
        self,
        agent_pack: Any,
        goal_spec: Any,
        seed: int,
        trace: EvaluationTrace,
    ) -> Tuple[Any, int]:
        """Default mock evaluation for testing."""
        from ..models import Metrics
        
        pack_hash = (
            agent_pack.pack_hash
            if hasattr(agent_pack, "pack_hash")
            else hashlib.sha256(str(agent_pack).encode()).digest()
        )
        
        max_episodes = (
            goal_spec.eval_limits.max_episodes
            if hasattr(goal_spec, "eval_limits")
            else 10
        )
        
        total_reward = 0.0
        step_count = 0
        
        rng_state = seed
        for episode in range(max_episodes):
            state_hash = hashlib.sha256(pack_hash + struct.pack(">QQ", seed, episode)).digest()
            
            for step in range(100):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                action = struct.pack(">I", rng_state % 10)
                
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                reward = (rng_state % 100) / 100.0
                total_reward += reward
                
                next_state_hash = hashlib.sha256(state_hash + action).digest()
                
                trace.add_step(EvaluationStep(
                    step_index=step_count,
                    state_hash=state_hash,
                    action=action,
                    next_state_hash=next_state_hash,
                    reward=reward,
                ))
                
                state_hash = next_state_hash
                step_count += 1
        
        avg_reward = total_reward / step_count if step_count > 0 else 0.0
        
        metrics = Metrics(
            reward=avg_reward,
            risk=(pack_hash[1] / 255.0) * 0.3,
            complexity=(pack_hash[2] / 255.0) * 0.2 + 0.1,
            episodes_run=max_episodes,
            steps_run=step_count,
        )
        
        return metrics, step_count
    
    def _serialize_metrics(self, metrics: Any) -> bytes:
        """Serialize metrics for hashing."""
        return struct.pack(
            ">ddd",
            metrics.reward,
            metrics.risk,
            metrics.complexity,
        )
    
    def get_commitment(self, contribution_hash: bytes) -> Optional[TraceCommitment]:
        """Get commitment for a contribution."""
        return self._commitments.get(contribution_hash)
    
    def get_trace(self, contribution_hash: bytes) -> Optional[EvaluationTrace]:
        """Get trace for a contribution."""
        return self._traces.get(contribution_hash)
    
    def get_audit_indices(
        self,
        contribution_hash: bytes,
        vrf_output: Optional[bytes] = None,
    ) -> List[int]:
        """Get audit indices for a contribution."""
        trace = self._traces.get(contribution_hash)
        if trace is None:
            return []
        
        commitment = self._commitments.get(contribution_hash)
        if commitment is None:
            return []
        
        return derive_audit_indices(
            trace_root=commitment.trace_root,
            evaluator_set_hash=self.evaluator_id,
            num_steps=len(trace),
            num_samples=self.num_audit_samples,
            vrf_output=vrf_output,
        )
    
    def respond_to_audit(
        self,
        request: AuditRequest,
    ) -> AuditResult:
        """
        Respond to an audit request.
        
        Provides Merkle proofs for requested steps.
        """
        commitment = self._commitments.get(request.trace_commitment.contribution_hash)
        if commitment is None:
            return AuditResult(
                request_hash=request.request_hash(),
                status=AuditStatus.FAILED,
                verified_steps=[],
                failure_reason="No commitment found",
            )
        
        trace = self._traces.get(request.trace_commitment.contribution_hash)
        if trace is None:
            return AuditResult(
                request_hash=request.request_hash(),
                status=AuditStatus.FAILED,
                verified_steps=[],
                failure_reason="No trace found",
            )
        
        proofs = []
        verified = []
        
        for idx in request.step_indices:
            if idx < 0 or idx >= len(trace):
                return AuditResult(
                    request_hash=request.request_hash(),
                    status=AuditStatus.FAILED,
                    verified_steps=verified,
                    failed_step=idx,
                    failure_reason=f"Step index {idx} out of range",
                )
            
            proof = trace.get_proof(idx)
            if not proof.verify(commitment.trace_root):
                return AuditResult(
                    request_hash=request.request_hash(),
                    status=AuditStatus.FAILED,
                    verified_steps=verified,
                    failed_step=idx,
                    failure_reason="Merkle proof verification failed",
                    proofs=proofs,
                )
            
            proofs.append(proof)
            verified.append(idx)
        
        return AuditResult(
            request_hash=request.request_hash(),
            status=AuditStatus.PASSED,
            verified_steps=verified,
            proofs=proofs,
        )
    
    def verify_step(
        self,
        step: EvaluationStep,
        proof: TraceMerkleProof,
        trace_root: bytes,
        reference_eval_fn: Optional[Callable] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a single step against the trace commitment.
        
        Args:
            step: Step to verify
            proof: Merkle proof for the step
            trace_root: Expected trace root
            reference_eval_fn: Optional function to re-execute step
            
        Returns:
            (valid, reason)
        """
        leaf_hash = hashlib.sha256(_VEP_TRACE_LEAF_V1 + step.step_hash()).digest()
        
        if leaf_hash != proof.step_hash:
            return False, "Step hash mismatch"
        
        if not proof.verify(trace_root):
            return False, "Merkle proof invalid"
        
        if reference_eval_fn:
            expected_next_state, expected_reward = reference_eval_fn(
                step.state_hash, step.action
            )
            if expected_next_state != step.next_state_hash:
                return False, "Next state mismatch"
            if abs(expected_reward - step.reward) > 1e-9:
                return False, "Reward mismatch"
        
        return True, "Verified"


# =============================================================================
# Audit Verification
# =============================================================================

class VEPAuditor:
    """
    Auditor for VEP evaluations.
    
    Verifies trace commitments and step proofs.
    """
    
    def __init__(
        self,
        auditor_id: bytes,
        reference_env: Optional[Any] = None,
    ) -> None:
        if len(auditor_id) != 32:
            raise ValueError("auditor_id must be 32 bytes")
        
        self.auditor_id = auditor_id
        self.reference_env = reference_env
    
    def create_audit_request(
        self,
        commitment: TraceCommitment,
        num_steps: int,
        num_samples: int = 32,
        vrf_output: Optional[bytes] = None,
    ) -> AuditRequest:
        """Create an audit request for a commitment."""
        indices = derive_audit_indices(
            trace_root=commitment.trace_root,
            evaluator_set_hash=commitment.evaluator_id,
            num_steps=num_steps,
            num_samples=num_samples,
            vrf_output=vrf_output,
        )
        
        return AuditRequest(
            trace_commitment=commitment,
            step_indices=indices,
            requester_id=self.auditor_id,
            timestamp=int(time.time() * 1000),
        )
    
    def verify_audit_response(
        self,
        result: AuditResult,
        commitment: TraceCommitment,
        steps: List[EvaluationStep],
    ) -> Tuple[bool, str]:
        """
        Verify an audit response.
        
        Args:
            result: Audit result from evaluator
            commitment: Original commitment
            steps: Step data for verification
            
        Returns:
            (valid, reason)
        """
        if result.status == AuditStatus.FAILED:
            return False, f"Audit failed: {result.failure_reason}"
        
        if len(result.proofs) != len(steps):
            return False, "Proof count mismatch"
        
        for step, proof in zip(steps, result.proofs):
            leaf_hash = hashlib.sha256(_VEP_TRACE_LEAF_V1 + step.step_hash()).digest()
            
            if leaf_hash != proof.step_hash:
                return False, f"Step {step.step_index} hash mismatch"
            
            if not proof.verify(commitment.trace_root):
                return False, f"Step {step.step_index} proof invalid"
        
        return True, "All proofs verified"
