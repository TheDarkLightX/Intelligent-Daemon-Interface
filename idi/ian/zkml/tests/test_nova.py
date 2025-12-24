"""Tests for Nova-IAN folding scheme integration."""

import secrets
import pytest

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


class TestTrainingStepCircuit:
    """Tests for training step circuit."""
    
    def test_circuit_creation(self):
        """Test circuit creation."""
        circuit = TrainingStepCircuit(
            model_dimension=1000,
            learning_rate=0.01,
            curve=CurveCycle.PASTA,
        )
        
        assert circuit.model_dimension == 1000
        assert circuit.learning_rate == 0.01
        assert circuit.curve == CurveCycle.PASTA
    
    def test_invalid_model_dimension(self):
        """Test that invalid model dimension is rejected."""
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=0)
        
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=-1)
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate is rejected."""
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=100, learning_rate=0)
        
        with pytest.raises(CircuitError):
            TrainingStepCircuit(model_dimension=100, learning_rate=2.0)
    
    def test_execute_step(self):
        """Test step execution."""
        circuit = TrainingStepCircuit(model_dimension=100)
        
        input = StepInput(
            epoch=0,
            prev_state_hash=b"\x00" * 32,
            batch_hash=secrets.token_bytes(32),
            prev_weights_hash=secrets.token_bytes(32),
        )
        
        output = circuit.execute(input)
        
        assert output.epoch == 1
        assert len(output.new_state_hash) == 32
        assert len(output.new_weights_hash) == 32
    
    def test_num_constraints(self):
        """Test constraint count estimation."""
        circuit = TrainingStepCircuit(model_dimension=1000)
        constraints = circuit.num_constraints()
        
        assert constraints > 0


class TestFoldingInstance:
    """Tests for folding instance."""
    
    def test_genesis_creation(self):
        """Test genesis instance creation."""
        instance = FoldingInstance.genesis(model_dimension=1000)
        
        assert instance.epoch == 0
        assert instance.state_hash == b"\x00" * 32
        assert instance.model_dimension == 1000
        assert instance.curve == CurveCycle.PASTA
    
    def test_fold_same_curve(self):
        """Test folding instances with same curve."""
        inst1 = FoldingInstance.genesis(model_dimension=1000)
        inst2 = FoldingInstance(
            epoch=1,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=1000,
            curve=CurveCycle.PASTA,
        )
        
        folded = inst1.fold(inst2)
        assert folded.epoch == 1
        assert folded.curve == CurveCycle.PASTA
    
    def test_fold_different_curve_fails(self):
        """Test that folding different curves fails."""
        inst1 = FoldingInstance.genesis(model_dimension=1000, curve=CurveCycle.PASTA)
        inst2 = FoldingInstance(
            epoch=1,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=1000,
            curve=CurveCycle.BN_GRUMPKIN,
        )
        
        with pytest.raises(NovaError, match="curves"):
            inst1.fold(inst2)
    
    def test_fold_different_model_dimension_fails(self):
        """Test that folding different model dimensions fails."""
        inst1 = FoldingInstance.genesis(model_dimension=1000)
        inst2 = FoldingInstance(
            epoch=1,
            state_hash=secrets.token_bytes(32),
            commitment=b"\x00" * 48,
            cross_term=b"\x00" * 48,
            model_dimension=2000,
            curve=CurveCycle.PASTA,
        )
        
        with pytest.raises(NovaError, match="model dimensions"):
            inst1.fold(inst2)
    
    def test_invalid_state_hash_length(self):
        """Test that invalid state hash length is rejected."""
        with pytest.raises(NovaError):
            FoldingInstance(
                epoch=0,
                state_hash=b"short",
                commitment=b"\x00" * 48,
                cross_term=b"\x00" * 48,
            )
    
    def test_negative_epoch_rejected(self):
        """Test that negative epoch is rejected."""
        with pytest.raises(NovaError):
            FoldingInstance(
                epoch=-1,
                state_hash=b"\x00" * 32,
                commitment=b"\x00" * 48,
                cross_term=b"\x00" * 48,
            )


class TestNovaVerifier:
    """Tests for Nova verifier."""
    
    def test_verifier_requires_simulation_opt_in(self):
        """Test that verifier requires explicit simulation opt-in."""
        verifier = NovaVerifier(expected_model_dimension=1000)
        
        # Create a minimal valid proof
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=1000,
        )
        
        with pytest.raises(ProofVerificationError, match="No Rust backend"):
            verifier.verify(proof)
    
    def test_verifier_with_simulation(self):
        """Test verifier with simulation mode enabled."""
        verifier = NovaVerifier(
            expected_model_dimension=1000,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=1000,
        )
        
        assert verifier.verify(proof) is True
    
    def test_verifier_rejects_wrong_model_dimension(self):
        """Test that wrong model dimension is rejected."""
        verifier = NovaVerifier(
            expected_model_dimension=2000,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=1000,
        )
        
        with pytest.raises(ProofVerificationError, match="Model dimension"):
            verifier.verify(proof)
    
    def test_verifier_rejects_wrong_curve(self):
        """Test that wrong curve is rejected."""
        verifier = NovaVerifier(
            expected_model_dimension=1000,
            expected_curve=CurveCycle.BN_GRUMPKIN,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=1000,
        )
        
        with pytest.raises(ProofVerificationError, match="Curve"):
            verifier.verify(proof)
    
    def test_verifier_rejects_wrong_num_steps(self):
        """Test that wrong num_steps is rejected."""
        verifier = NovaVerifier(
            expected_model_dimension=1000,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"IAN_NOVA_V1" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=5,
            model_dimension=1000,
        )
        
        with pytest.raises(ProofVerificationError, match="num_steps"):
            verifier.verify(proof, expected_num_steps=3)
    
    def test_verifier_rejects_invalid_domain_prefix(self):
        """Test that invalid domain prefix is rejected."""
        verifier = NovaVerifier(
            expected_model_dimension=1000,
            allow_simulation=True,
        )
        
        proof = NovaProof(
            version=1,
            curve=CurveCycle.PASTA,
            final_instance=b"WRONG_PREFIX" + b"\x00" * (32 + 48 + 48),
            snark_proof=b"\x00" * 32,
            num_steps=1,
            model_dimension=1000,
        )
        
        with pytest.raises(ProofVerificationError, match="domain prefix"):
            verifier.verify(proof)


class TestNovaTrainingSession:
    """Tests for Nova training session."""
    
    def test_session_creation(self):
        """Test session creation."""
        session = NovaTrainingSession(
            model_dimension=1000,
            learning_rate=0.01,
            curve=CurveCycle.PASTA,
        )
        
        assert session.epoch == 0
    
    def test_training_steps(self):
        """Test adding training steps."""
        session = NovaTrainingSession(model_dimension=1000)
        
        for i in range(3):
            batch = secrets.token_bytes(1024)
            gradient = GradientTensor(data=secrets.token_bytes(4000), shape=(1000,))
            weights_hash = secrets.token_bytes(32)
            
            output = session.add_training_step(batch, gradient, weights_hash)
            assert output.epoch == i + 1
        
        assert session.epoch == 3
    
    def test_finalize_proof(self):
        """Test proof finalization."""
        session = NovaTrainingSession(model_dimension=1000)
        
        for i in range(3):
            batch = secrets.token_bytes(1024)
            gradient = GradientTensor(data=secrets.token_bytes(4000), shape=(1000,))
            weights_hash = secrets.token_bytes(32)
            session.add_training_step(batch, gradient, weights_hash)
        
        proof = session.finalize()
        
        assert proof.version == 1
        assert proof.num_steps == 3
        assert proof.model_dimension == 1000
        assert proof.curve == CurveCycle.PASTA
    
    def test_batch_merkle_root(self):
        """Test batch Merkle root computation."""
        session = NovaTrainingSession(model_dimension=1000)
        
        # Empty session
        root1 = session.get_batch_merkle_root()
        assert len(root1) == 32
        
        # After steps
        for i in range(2):
            batch = secrets.token_bytes(1024)
            gradient = GradientTensor(data=secrets.token_bytes(4000), shape=(1000,))
            weights_hash = secrets.token_bytes(32)
            session.add_training_step(batch, gradient, weights_hash)
        
        root2 = session.get_batch_merkle_root()
        assert len(root2) == 32
        assert root2 != root1  # Should change after adding steps
    
    def test_end_to_end_verification(self):
        """Test end-to-end proof generation and verification."""
        session = NovaTrainingSession(model_dimension=1000)
        
        for i in range(3):
            batch = secrets.token_bytes(1024)
            gradient = GradientTensor(data=secrets.token_bytes(4000), shape=(1000,))
            weights_hash = secrets.token_bytes(32)
            session.add_training_step(batch, gradient, weights_hash)
        
        proof = session.finalize()
        
        verifier = NovaVerifier(
            expected_model_dimension=1000,
            expected_curve=CurveCycle.PASTA,
            allow_simulation=True,
        )
        
        assert verifier.verify(proof, expected_num_steps=3) is True


class TestNovaProof:
    """Tests for Nova proof validation."""
    
    def test_invalid_version(self):
        """Test that invalid version is rejected."""
        with pytest.raises(NovaError, match="version"):
            NovaProof(
                version=99,
                curve=CurveCycle.PASTA,
                final_instance=b"",
                snark_proof=b"",
                num_steps=1,
                model_dimension=1000,
            )
    
    def test_invalid_num_steps(self):
        """Test that invalid num_steps is rejected."""
        with pytest.raises(NovaError, match="num_steps"):
            NovaProof(
                version=1,
                curve=CurveCycle.PASTA,
                final_instance=b"",
                snark_proof=b"",
                num_steps=0,
                model_dimension=1000,
            )
    
    def test_invalid_model_dimension(self):
        """Test that invalid model_dimension is rejected."""
        with pytest.raises(NovaError, match="model_dimension"):
            NovaProof(
                version=1,
                curve=CurveCycle.PASTA,
                final_instance=b"",
                snark_proof=b"",
                num_steps=1,
                model_dimension=0,
            )
