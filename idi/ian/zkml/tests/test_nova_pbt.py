"""Property-based tests for Nova folding scheme using Hypothesis."""

import secrets
from typing import List, Tuple

import pytest
from hypothesis import given, settings, assume, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition, Bundle

from idi.ian.zkml import (
    NovaError,
    ProofVerificationError,
    CircuitError,
    CurveCycle,
    StepInput,
    StepOutput,
    TrainingStepCircuit,
    FoldingInstance,
    NovaProof,
    NovaProver,
    NovaVerifier,
    NovaTrainingSession,
    GradientTensor,
)


# =============================================================================
# Strategies
# =============================================================================

@st.composite
def valid_model_dimension(draw):
    """Generate valid model dimension."""
    return draw(st.integers(min_value=1, max_value=10000))


@st.composite
def valid_learning_rate(draw):
    """Generate valid learning rate in (0, 1]."""
    return draw(st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def state_hash(draw):
    """Generate a valid 32-byte state hash."""
    return draw(st.binary(min_size=32, max_size=32))


@st.composite
def curve_cycle(draw):
    """Generate a valid curve cycle."""
    return draw(st.sampled_from([CurveCycle.PASTA, CurveCycle.BN_GRUMPKIN, CurveCycle.BLS12_381]))


# =============================================================================
# TrainingStepCircuit Properties
# =============================================================================

class TestTrainingStepCircuitProperties:
    """Property-based tests for training step circuit."""
    
    @given(valid_model_dimension(), valid_learning_rate(), curve_cycle())
    @settings(max_examples=50)
    def test_circuit_creation_valid_params(self, dim: int, lr: float, curve: CurveCycle):
        """Property: Valid params create valid circuit."""
        circuit = TrainingStepCircuit(
            model_dimension=dim,
            learning_rate=lr,
            curve=curve,
        )
        assert circuit.model_dimension == dim
        assert circuit.learning_rate == lr
        assert circuit.curve == curve
    
    @given(st.integers(min_value=-1000, max_value=0))
    @settings(max_examples=30)
    def test_circuit_rejects_invalid_dimension(self, dim: int):
        """Property: Invalid dimension raises CircuitError."""
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=dim)
    
    @given(st.floats(min_value=-10, max_value=0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    def test_circuit_rejects_invalid_learning_rate(self, lr: float):
        """Property: Invalid learning rate raises CircuitError."""
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=100, learning_rate=lr)
    
    @given(valid_model_dimension(), state_hash(), state_hash(), state_hash())
    @settings(max_examples=30)
    def test_step_output_advances_epoch(
        self, dim: int, prev_state: bytes, batch_hash: bytes, weights_hash: bytes
    ):
        """Property: Step output always advances epoch by 1."""
        circuit = TrainingStepCircuit(model_dimension=dim)
        
        input = StepInput(
            epoch=0,
            prev_state_hash=prev_state,
            batch_hash=batch_hash,
            prev_weights_hash=weights_hash,
        )
        
        output = circuit.execute(input)
        assert output.epoch == input.epoch + 1
    
    @given(valid_model_dimension())
    @settings(max_examples=30)
    def test_constraints_positive(self, dim: int):
        """Property: Constraint count is always positive."""
        circuit = TrainingStepCircuit(model_dimension=dim)
        assert circuit.num_constraints() > 0


# =============================================================================
# FoldingInstance Properties
# =============================================================================

class TestFoldingInstanceProperties:
    """Property-based tests for folding instance."""
    
    @given(valid_model_dimension(), curve_cycle())
    @settings(max_examples=30)
    def test_genesis_is_epoch_zero(self, dim: int, curve: CurveCycle):
        """Property: Genesis instance is always epoch 0."""
        instance = FoldingInstance.genesis(model_dimension=dim, curve=curve)
        assert instance.epoch == 0
        assert instance.model_dimension == dim
        assert instance.curve == curve
    
    @given(valid_model_dimension(), curve_cycle(), st.integers(min_value=0, max_value=100))
    @settings(max_examples=30)
    def test_fold_preserves_curve_and_dimension(self, dim: int, curve: CurveCycle, epoch: int):
        """Property: Folding preserves curve and dimension."""
        inst1 = FoldingInstance.genesis(model_dimension=dim, curve=curve)
        inst2 = FoldingInstance(
            epoch=epoch,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=dim,
            curve=curve,
        )
        
        folded = inst1.fold(inst2)
        assert folded.curve == curve
        assert folded.model_dimension == dim
    
    @given(valid_model_dimension(), st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
    @settings(max_examples=30)
    def test_fold_epoch_is_max(self, dim: int, epoch1: int, epoch2: int):
        """Property: Folded epoch is max of both."""
        inst1 = FoldingInstance(
            epoch=epoch1,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=dim,
            curve=CurveCycle.PASTA,
        )
        inst2 = FoldingInstance(
            epoch=epoch2,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=dim,
            curve=CurveCycle.PASTA,
        )
        
        folded = inst1.fold(inst2)
        assert folded.epoch == max(epoch1, epoch2)
    
    @given(valid_model_dimension(), valid_model_dimension(), curve_cycle())
    @settings(max_examples=30)
    def test_fold_different_dimensions_fails(self, dim1: int, dim2: int, curve: CurveCycle):
        """Property: Folding different dimensions fails."""
        assume(dim1 != dim2)
        
        inst1 = FoldingInstance.genesis(model_dimension=dim1, curve=curve)
        inst2 = FoldingInstance(
            epoch=1,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=dim2,
            curve=curve,
        )
        
        with pytest.raises(NovaError, match="model dimensions"):
            inst1.fold(inst2)


# =============================================================================
# NovaVerifier Properties
# =============================================================================

class TestNovaVerifierProperties:
    """Property-based tests for Nova verifier."""
    
    @given(valid_model_dimension(), curve_cycle())
    @settings(max_examples=30)
    def test_verifier_rejects_dimension_mismatch(self, dim: int, curve: CurveCycle):
        """Property: Verifier rejects dimension mismatch."""
        verifier = NovaVerifier(
            expected_model_dimension=dim,
            expected_curve=curve,
            allow_simulation=True,
        )
        
        wrong_dim = dim + 1 if dim < 10000 else dim - 1
        
        proof = NovaProof(
            version=1,
            curve=curve,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=wrong_dim,
        )
        
        with pytest.raises(ProofVerificationError, match="Model dimension"):
            verifier.verify(proof)
    
    @given(valid_model_dimension(), st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_verifier_rejects_num_steps_mismatch(self, dim: int, actual: int, expected: int):
        """Property: Verifier rejects num_steps mismatch when specified."""
        assume(actual != expected)
        
        verifier = NovaVerifier(
            expected_model_dimension=dim,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=actual,
            model_dimension=dim,
        )
        
        with pytest.raises(ProofVerificationError, match="num_steps"):
            verifier.verify(proof, expected_num_steps=expected)
    
    @given(valid_model_dimension())
    @settings(max_examples=30)
    def test_verifier_requires_simulation_opt_in(self, dim: int):
        """Property: Without simulation flag, verifier raises error."""
        verifier = NovaVerifier(expected_model_dimension=dim)
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=dim,
        )
        
        with pytest.raises(ProofVerificationError, match="No Rust backend"):
            verifier.verify(proof)


# =============================================================================
# NovaTrainingSession Properties
# =============================================================================

class TestNovaTrainingSessionProperties:
    """Property-based tests for Nova training session."""
    
    @given(valid_model_dimension(), st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=5000)
    def test_epoch_tracks_steps(self, dim: int, num_steps: int):
        """Property: Session epoch equals number of steps taken."""
        session = NovaTrainingSession(model_dimension=dim)
        
        assert session.epoch == 0
        
        for i in range(num_steps):
            batch = secrets.token_bytes(100)
            gradient = GradientTensor(data=secrets.token_bytes(dim * 4), shape=(dim,))
            weights = secrets.token_bytes(32)
            session.add_training_step(batch, gradient, weights)
            assert session.epoch == i + 1
    
    @given(valid_model_dimension(), st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, deadline=5000)
    def test_proof_num_steps_matches_session(self, dim: int, num_steps: int):
        """Property: Proof num_steps matches session steps."""
        session = NovaTrainingSession(model_dimension=dim)
        
        for _ in range(num_steps):
            batch = secrets.token_bytes(100)
            gradient = GradientTensor(data=secrets.token_bytes(dim * 4), shape=(dim,))
            weights = secrets.token_bytes(32)
            session.add_training_step(batch, gradient, weights)
        
        proof = session.finalize()
        assert proof.num_steps == num_steps
    
    @given(valid_model_dimension(), valid_learning_rate(), curve_cycle())
    @settings(max_examples=20, deadline=5000)
    def test_proof_inherits_session_params(self, dim: int, lr: float, curve: CurveCycle):
        """Property: Proof inherits session parameters."""
        session = NovaTrainingSession(
            model_dimension=dim,
            learning_rate=lr,
            curve=curve,
        )
        
        batch = secrets.token_bytes(100)
        gradient = GradientTensor(data=secrets.token_bytes(dim * 4), shape=(dim,))
        weights = secrets.token_bytes(32)
        session.add_training_step(batch, gradient, weights)
        
        proof = session.finalize()
        assert proof.model_dimension == dim
        assert proof.curve == curve


# =============================================================================
# Nova State Machine Tests
# =============================================================================

class NovaSessionStateMachine(RuleBasedStateMachine):
    """
    State machine test for NovaTrainingSession.
    
    Tests that the session correctly tracks state through
    training steps and proof finalization.
    """
    
    def __init__(self):
        super().__init__()
        self.dim = 100
        self.session = NovaTrainingSession(model_dimension=self.dim)
        self.steps_taken = 0
        self.finalized = False
    
    @rule()
    @precondition(lambda self: not self.finalized)
    def add_training_step(self):
        """Add a training step."""
        batch = secrets.token_bytes(100)
        gradient = GradientTensor(data=secrets.token_bytes(self.dim * 4), shape=(self.dim,))
        weights = secrets.token_bytes(32)
        
        output = self.session.add_training_step(batch, gradient, weights)
        self.steps_taken += 1
        
        assert output.epoch == self.steps_taken
    
    @rule()
    @precondition(lambda self: self.steps_taken > 0 and not self.finalized)
    def finalize_proof(self):
        """Finalize and verify proof."""
        proof = self.session.finalize()
        self.finalized = True
        
        assert proof.num_steps == self.steps_taken
        assert proof.model_dimension == self.dim
        
        # Verify in simulation mode
        verifier = NovaVerifier(
            expected_model_dimension=self.dim,
            allow_simulation=True,
        )
        assert verifier.verify(proof, expected_num_steps=self.steps_taken)
    
    @rule()
    @precondition(lambda self: self.finalized)
    def noop_after_finalize(self):
        """After finalization, state machine can still progress (no-op)."""
        pass
    
    @invariant()
    def epoch_matches_steps(self):
        """Invariant: Session epoch always matches steps taken."""
        assert self.session.epoch == self.steps_taken
    
    @invariant()
    def state_hash_not_empty(self):
        """Invariant: State hash is always 32 bytes."""
        assert len(self.session.state_hash) == 32


TestNovaSessionStateMachine = NovaSessionStateMachine.TestCase
