"""
Tests for VEP (Verifiable Evaluation Protocol) algorithm.

Covers:
- Evaluation trace construction
- Merkle proof generation and verification
- Commitment generation
- Audit request and response
- Edge cases and security properties
"""

import pytest
import secrets
import struct
import hashlib
from typing import List

from ..vep import (
    VEPEvaluationHarness,
    EvaluationTrace,
    EvaluationStep,
    TraceCommitment,
    PreCommitment,
    TraceMerkleProof,
    AuditRequest,
    AuditResult,
    AuditStatus,
    VEPAuditor,
    derive_audit_indices,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_step(index: int, seed: int = 0) -> EvaluationStep:
    """Create a test evaluation step."""
    state_hash = hashlib.sha256(f"state_{index}_{seed}".encode()).digest()
    action = struct.pack(">I", index % 10)
    next_state_hash = hashlib.sha256(f"next_{index}_{seed}".encode()).digest()
    reward = (index % 100) / 100.0
    
    return EvaluationStep(
        step_index=index,
        state_hash=state_hash,
        action=action,
        next_state_hash=next_state_hash,
        reward=reward,
    )


def make_trace(num_steps: int, seed: int = 0) -> EvaluationTrace:
    """Create a test trace with given number of steps."""
    trace = EvaluationTrace()
    for i in range(num_steps):
        trace.add_step(make_step(i, seed))
    return trace


class MockAgentPack:
    """Mock agent pack for testing."""
    def __init__(self):
        self.pack_hash = secrets.token_bytes(32)


class MockGoalSpec:
    """Mock goal spec for testing."""
    def __init__(self):
        self.eval_limits = type("Limits", (), {"max_episodes": 5})()


# =============================================================================
# EvaluationStep Tests
# =============================================================================

class TestEvaluationStep:
    """Tests for EvaluationStep."""
    
    def test_create_step(self):
        """Test basic step creation."""
        step = make_step(0)
        assert step.step_index == 0
        assert len(step.state_hash) == 32
        assert len(step.next_state_hash) == 32
    
    def test_invalid_state_hash_length(self):
        """Test that invalid state hash length is rejected."""
        with pytest.raises(ValueError, match="32 bytes"):
            EvaluationStep(
                step_index=0,
                state_hash=b"short",
                action=b"action",
                next_state_hash=secrets.token_bytes(32),
                reward=0.0,
            )
    
    def test_invalid_next_state_hash_length(self):
        """Test that invalid next_state hash length is rejected."""
        with pytest.raises(ValueError, match="32 bytes"):
            EvaluationStep(
                step_index=0,
                state_hash=secrets.token_bytes(32),
                action=b"action",
                next_state_hash=b"short",
                reward=0.0,
            )
    
    def test_step_hash_deterministic(self):
        """Test that step hash is deterministic."""
        step1 = make_step(0, seed=42)
        step2 = make_step(0, seed=42)
        assert step1.step_hash() == step2.step_hash()
    
    def test_step_hash_different_steps(self):
        """Test that different steps have different hashes."""
        step1 = make_step(0)
        step2 = make_step(1)
        assert step1.step_hash() != step2.step_hash()
    
    def test_canonical_bytes_deterministic(self):
        """Test that canonical bytes are deterministic."""
        step1 = make_step(5, seed=123)
        step2 = make_step(5, seed=123)
        assert step1.canonical_bytes() == step2.canonical_bytes()


# =============================================================================
# EvaluationTrace Tests
# =============================================================================

class TestEvaluationTrace:
    """Tests for EvaluationTrace."""
    
    def test_empty_trace(self):
        """Test empty trace."""
        trace = EvaluationTrace()
        assert len(trace) == 0
        root = trace.finalize()
        assert len(root) == 32
    
    def test_add_steps(self):
        """Test adding steps to trace."""
        trace = EvaluationTrace()
        trace.add_step(make_step(0))
        trace.add_step(make_step(1))
        assert len(trace) == 2
    
    def test_cannot_add_after_finalize(self):
        """Test that adding steps after finalize raises error."""
        trace = make_trace(5)
        trace.finalize()
        
        with pytest.raises(RuntimeError, match="finalization"):
            trace.add_step(make_step(5))
    
    def test_root_deterministic(self):
        """Test that Merkle root is deterministic."""
        trace1 = make_trace(10, seed=42)
        trace2 = make_trace(10, seed=42)
        
        assert trace1.finalize() == trace2.finalize()
    
    def test_root_different_traces(self):
        """Test that different traces have different roots."""
        trace1 = make_trace(10, seed=1)
        trace2 = make_trace(10, seed=2)
        
        assert trace1.finalize() != trace2.finalize()
    
    def test_get_step(self):
        """Test getting step by index."""
        trace = make_trace(5)
        step = trace.get_step(2)
        assert step.step_index == 2
    
    def test_get_step_out_of_range(self):
        """Test that out of range index raises error."""
        trace = make_trace(5)
        
        with pytest.raises(IndexError):
            trace.get_step(10)
    
    def test_get_proof(self):
        """Test getting Merkle proof."""
        trace = make_trace(10)
        trace.finalize()
        
        proof = trace.get_proof(5)
        assert proof.step_index == 5
        assert proof.verify(trace.root)
    
    def test_proof_all_steps(self):
        """Test that proofs work for all steps."""
        trace = make_trace(16)  # Power of 2
        root = trace.finalize()
        
        for i in range(16):
            proof = trace.get_proof(i)
            assert proof.verify(root), f"Proof failed for step {i}"
    
    def test_proof_non_power_of_two(self):
        """Test proofs with non-power-of-2 step count."""
        for n in [3, 7, 11, 15, 17, 100]:
            trace = make_trace(n)
            root = trace.finalize()
            
            for i in range(n):
                proof = trace.get_proof(i)
                assert proof.verify(root), f"Proof failed for step {i} with n={n}"


# =============================================================================
# TraceMerkleProof Tests
# =============================================================================

class TestTraceMerkleProof:
    """Tests for TraceMerkleProof."""
    
    def test_proof_verification(self):
        """Test proof verification."""
        trace = make_trace(8)
        root = trace.finalize()
        
        proof = trace.get_proof(3)
        assert proof.verify(root)
    
    def test_proof_invalid_root(self):
        """Test that proof fails with wrong root."""
        trace = make_trace(8)
        root = trace.finalize()
        
        proof = trace.get_proof(3)
        wrong_root = secrets.token_bytes(32)
        assert not proof.verify(wrong_root)
    
    def test_proof_serialization(self):
        """Test proof serialization."""
        trace = make_trace(8)
        trace.finalize()
        
        proof = trace.get_proof(3)
        serialized = proof.to_bytes()
        
        assert len(serialized) > 0
        assert isinstance(serialized, bytes)


# =============================================================================
# Commitment Tests
# =============================================================================

class TestTraceCommitment:
    """Tests for TraceCommitment."""
    
    def test_create_commitment(self):
        """Test basic commitment creation."""
        commitment = TraceCommitment(
            evaluator_id=secrets.token_bytes(32),
            contribution_hash=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
            seed=12345,
            trace_root=secrets.token_bytes(32),
            metrics_hash=secrets.token_bytes(32),
            timestamp=1000000,
        )
        
        assert len(commitment.commitment_hash()) == 32
    
    def test_invalid_field_lengths(self):
        """Test that invalid field lengths are rejected."""
        with pytest.raises(ValueError, match="32 bytes"):
            TraceCommitment(
                evaluator_id=b"short",
                contribution_hash=secrets.token_bytes(32),
                env_hash=secrets.token_bytes(32),
                seed=0,
                trace_root=secrets.token_bytes(32),
                metrics_hash=secrets.token_bytes(32),
                timestamp=0,
            )
    
    def test_commitment_hash_deterministic(self):
        """Test that commitment hash is deterministic."""
        fields = {
            "evaluator_id": secrets.token_bytes(32),
            "contribution_hash": secrets.token_bytes(32),
            "env_hash": secrets.token_bytes(32),
            "seed": 42,
            "trace_root": secrets.token_bytes(32),
            "metrics_hash": secrets.token_bytes(32),
            "timestamp": 1000,
        }
        
        c1 = TraceCommitment(**fields)
        c2 = TraceCommitment(**fields)
        
        assert c1.commitment_hash() == c2.commitment_hash()


class TestPreCommitment:
    """Tests for PreCommitment."""
    
    def test_create_precommitment(self):
        """Test basic pre-commitment creation."""
        precommit = PreCommitment(
            evaluator_id=secrets.token_bytes(32),
            contribution_hash=secrets.token_bytes(32),
            slot=1,
            env_hash=secrets.token_bytes(32),
            timestamp=1000,
        )
        
        assert len(precommit.precommit_hash()) == 32


# =============================================================================
# Audit Selection Tests
# =============================================================================

class TestAuditSelection:
    """Tests for audit index derivation."""
    
    def test_derive_indices_basic(self):
        """Test basic index derivation."""
        indices = derive_audit_indices(
            trace_root=secrets.token_bytes(32),
            evaluator_set_hash=secrets.token_bytes(32),
            num_steps=100,
            num_samples=10,
        )
        
        assert len(indices) == 10
        assert all(0 <= i < 100 for i in indices)
    
    def test_derive_indices_deterministic(self):
        """Test that indices are deterministic."""
        trace_root = secrets.token_bytes(32)
        eval_hash = secrets.token_bytes(32)
        
        indices1 = derive_audit_indices(trace_root, eval_hash, 100, 10)
        indices2 = derive_audit_indices(trace_root, eval_hash, 100, 10)
        
        assert indices1 == indices2
    
    def test_derive_indices_different_roots(self):
        """Test that different roots give different indices."""
        eval_hash = secrets.token_bytes(32)
        
        indices1 = derive_audit_indices(secrets.token_bytes(32), eval_hash, 100, 10)
        indices2 = derive_audit_indices(secrets.token_bytes(32), eval_hash, 100, 10)
        
        # Very unlikely to be equal by chance
        assert indices1 != indices2
    
    def test_derive_indices_all_when_samples_exceed_steps(self):
        """Test that all indices returned when samples >= steps."""
        indices = derive_audit_indices(
            trace_root=secrets.token_bytes(32),
            evaluator_set_hash=secrets.token_bytes(32),
            num_steps=5,
            num_samples=10,
        )
        
        assert indices == [0, 1, 2, 3, 4]
    
    def test_derive_indices_empty_trace(self):
        """Test empty trace returns empty indices."""
        indices = derive_audit_indices(
            trace_root=secrets.token_bytes(32),
            evaluator_set_hash=secrets.token_bytes(32),
            num_steps=0,
            num_samples=10,
        )
        
        assert indices == []
    
    def test_derive_indices_with_vrf(self):
        """Test indices derivation with VRF output."""
        trace_root = secrets.token_bytes(32)
        eval_hash = secrets.token_bytes(32)
        vrf_output = secrets.token_bytes(32)
        
        indices_no_vrf = derive_audit_indices(trace_root, eval_hash, 100, 10)
        indices_with_vrf = derive_audit_indices(trace_root, eval_hash, 100, 10, vrf_output)
        
        # Should be different when VRF is provided
        assert indices_no_vrf != indices_with_vrf


# =============================================================================
# VEPEvaluationHarness Tests
# =============================================================================

class TestVEPEvaluationHarness:
    """Tests for VEPEvaluationHarness."""
    
    def test_create_harness(self):
        """Test basic harness creation."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
        )
        
        assert harness.num_audit_samples == 32
    
    def test_invalid_evaluator_id(self):
        """Test that invalid evaluator ID is rejected."""
        with pytest.raises(ValueError, match="32 bytes"):
            VEPEvaluationHarness(
                evaluator_id=b"short",
                env_hash=secrets.token_bytes(32),
            )
    
    def test_precommit(self):
        """Test pre-commitment generation."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
        )
        
        contribution_hash = secrets.token_bytes(32)
        precommit = harness.precommit(contribution_hash, slot=1)
        
        assert precommit.contribution_hash == contribution_hash
        assert precommit.slot == 1
    
    def test_evaluate(self):
        """Test evaluation produces metrics."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        metrics = harness.evaluate(agent, goal, seed=42)
        
        assert metrics is not None
        assert hasattr(metrics, "reward")
        assert hasattr(metrics, "risk")
    
    def test_evaluate_generates_trace(self):
        """Test that evaluation generates trace."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        harness.evaluate(agent, goal, seed=42)
        
        trace = harness.get_trace(agent.pack_hash)
        assert trace is not None
        assert len(trace) > 0
    
    def test_evaluate_generates_commitment(self):
        """Test that evaluation generates commitment."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        harness.evaluate(agent, goal, seed=42)
        
        commitment = harness.get_commitment(agent.pack_hash)
        assert commitment is not None
        assert commitment.contribution_hash == agent.pack_hash
    
    def test_get_audit_indices(self):
        """Test getting audit indices."""
        harness = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
            num_audit_samples=10,
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        harness.evaluate(agent, goal, seed=42)
        
        indices = harness.get_audit_indices(agent.pack_hash)
        assert len(indices) == 10
    
    def test_respond_to_audit(self):
        """Test responding to audit request."""
        evaluator_id = secrets.token_bytes(32)
        harness = VEPEvaluationHarness(
            evaluator_id=evaluator_id,
            env_hash=secrets.token_bytes(32),
            num_audit_samples=5,
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        harness.evaluate(agent, goal, seed=42)
        
        commitment = harness.get_commitment(agent.pack_hash)
        indices = harness.get_audit_indices(agent.pack_hash)
        
        request = AuditRequest(
            trace_commitment=commitment,
            step_indices=indices,
            requester_id=secrets.token_bytes(32),
            timestamp=1000,
        )
        
        result = harness.respond_to_audit(request)
        
        assert result.status == AuditStatus.PASSED
        assert result.verified_steps == indices
        assert len(result.proofs) == len(indices)


# =============================================================================
# VEPAuditor Tests
# =============================================================================

class TestVEPAuditor:
    """Tests for VEPAuditor."""
    
    def test_create_auditor(self):
        """Test basic auditor creation."""
        auditor = VEPAuditor(auditor_id=secrets.token_bytes(32))
        assert auditor is not None
    
    def test_create_audit_request(self):
        """Test creating audit request."""
        auditor = VEPAuditor(auditor_id=secrets.token_bytes(32))
        
        commitment = TraceCommitment(
            evaluator_id=secrets.token_bytes(32),
            contribution_hash=secrets.token_bytes(32),
            env_hash=secrets.token_bytes(32),
            seed=42,
            trace_root=secrets.token_bytes(32),
            metrics_hash=secrets.token_bytes(32),
            timestamp=1000,
        )
        
        request = auditor.create_audit_request(
            commitment=commitment,
            num_steps=100,
            num_samples=10,
        )
        
        assert len(request.step_indices) == 10
    
    def test_verify_audit_response(self):
        """Test verifying audit response."""
        evaluator_id = secrets.token_bytes(32)
        harness = VEPEvaluationHarness(
            evaluator_id=evaluator_id,
            env_hash=secrets.token_bytes(32),
            num_audit_samples=5,
        )
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        harness.evaluate(agent, goal, seed=42)
        
        commitment = harness.get_commitment(agent.pack_hash)
        trace = harness.get_trace(agent.pack_hash)
        indices = harness.get_audit_indices(agent.pack_hash)
        
        auditor = VEPAuditor(auditor_id=secrets.token_bytes(32))
        
        request = AuditRequest(
            trace_commitment=commitment,
            step_indices=indices,
            requester_id=auditor.auditor_id,
            timestamp=1000,
        )
        
        result = harness.respond_to_audit(request)
        
        # Get the actual steps for verification
        steps = [trace.get_step(i) for i in indices]
        
        valid, reason = auditor.verify_audit_response(result, commitment, steps)
        
        assert valid is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestVEPIntegration:
    """Integration tests for VEP components."""
    
    def test_full_evaluation_audit_cycle(self):
        """Test complete evaluation and audit cycle."""
        # Setup
        evaluator_id = secrets.token_bytes(32)
        auditor_id = secrets.token_bytes(32)
        env_hash = secrets.token_bytes(32)
        
        harness = VEPEvaluationHarness(
            evaluator_id=evaluator_id,
            env_hash=env_hash,
            num_audit_samples=10,
        )
        auditor = VEPAuditor(auditor_id=auditor_id)
        
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        # 1. Pre-commit
        precommit = harness.precommit(agent.pack_hash, slot=1)
        assert precommit is not None
        
        # 2. Evaluate
        metrics = harness.evaluate(agent, goal, seed=42)
        assert metrics is not None
        
        # 3. Get commitment
        commitment = harness.get_commitment(agent.pack_hash)
        assert commitment is not None
        
        # 4. Create audit request
        trace = harness.get_trace(agent.pack_hash)
        request = auditor.create_audit_request(
            commitment=commitment,
            num_steps=len(trace),
            num_samples=10,
        )
        
        # 5. Respond to audit
        result = harness.respond_to_audit(request)
        assert result.status == AuditStatus.PASSED
        
        # 6. Verify audit response
        steps = [trace.get_step(i) for i in request.step_indices]
        valid, _ = auditor.verify_audit_response(result, commitment, steps)
        assert valid is True
    
    def test_deterministic_evaluation(self):
        """Test that evaluation is deterministic given same seed."""
        env_hash = secrets.token_bytes(32)
        
        harness1 = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=env_hash,
        )
        harness2 = VEPEvaluationHarness(
            evaluator_id=secrets.token_bytes(32),
            env_hash=env_hash,
        )
        
        # Same agent and seed
        agent = MockAgentPack()
        goal = MockGoalSpec()
        
        metrics1 = harness1.evaluate(agent, goal, seed=42)
        
        # Create new agent with same pack_hash
        agent2 = MockAgentPack()
        agent2.pack_hash = agent.pack_hash
        
        metrics2 = harness2.evaluate(agent2, goal, seed=42)
        
        # Same seed + same input should give same metrics
        assert metrics1.reward == metrics2.reward
        assert metrics1.risk == metrics2.risk
