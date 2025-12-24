"""
Nova-IAN Folding Scheme Integration for Incremental Training Proofs.

Nova is a zero-knowledge proof system for Incrementally Verifiable Computation (IVC)
that achieves O(1) prover work per step via a non-interactive folding scheme.

This module provides:
- Python interfaces for Nova step circuits
- Training proof generation and verification
- Integration with gradient commitments

ARCHITECTURE:
The actual Nova proving is done in Rust (microsoft/Nova or PSE sonobe).
This module provides Python bindings and orchestration.

IMPORTANT: SIMULATION MODE
--------------------------
Without a Rust backend, this module operates in SIMULATION MODE:
- Proofs are generated but NOT cryptographically secure
- Verification checks structure and binding, but NOT the SNARK
- Use ONLY for testing and development

For production, a Rust backend implementing actual Nova proofs is REQUIRED.
The verify() method will raise ProofVerificationError if no backend is
provided, unless explicitly opted into simulation mode.

Proof flow:
    Epoch 0 → Fold → Epoch 1 → Fold → Epoch 2 → ... → Epoch N → Final SNARK
       w₀              w₁              w₂               wₙ        [proof]

Each step proves:
1. Previous weights hash matches accumulated state
2. Gradient computed correctly from batch
3. New weights = old weights - η × gradient
4. New weights hash committed

References:
- Nova: Recursive SNARKs without trusted setup (Setty, CCS 2022)
- microsoft/Nova (MIT License)
- privacy-scaling-explorations/sonobe (MIT/Apache-2.0)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

try:
    import nova_ian
    _RUST_AVAILABLE = True
except ImportError:
    nova_ian = None
    _RUST_AVAILABLE = False

from .commitments import GradientCommitment, GradientTensor


# =============================================================================
# Constants
# =============================================================================

NOVA_DOMAIN = b"IAN_NOVA_V1"
STATE_HASH_LEN = 32
COMMITMENT_LEN = 48
CROSS_TERM_LEN = 48
PROOF_VERSION = 1
RUST_PROOF_VERSION = 2
SUPPORTED_PROOF_VERSIONS = (PROOF_VERSION, RUST_PROOF_VERSION)
MAX_STEPS = 1_000_000
MAX_MODEL_DIM = 1_000_000_000  # 1B parameters

# Supported curve cycles
class CurveCycle(Enum):
    """Supported elliptic curve cycles for Nova."""
    PASTA = auto()      # Pallas/Vesta - general IVC
    BN_GRUMPKIN = auto() # BN254/Grumpkin - Ethereum compatible
    BLS12_381 = auto()   # BLS12-381 - signature aggregation


# =============================================================================
# Errors
# =============================================================================


class NovaError(Exception):
    """Error in Nova proof operations."""
    pass


class ProofVerificationError(NovaError):
    """Proof verification failed."""
    pass


class CircuitError(NovaError):
    """Error in step circuit definition."""
    pass


# =============================================================================
# Rust FFI helpers
# =============================================================================

_RUST_CURVE_MAP = {
    CurveCycle.PASTA: 0,
    CurveCycle.BN_GRUMPKIN: 1,
    CurveCycle.BLS12_381: 2,
}
_RUST_CURVE_MAP_INV = {value: key for key, value in _RUST_CURVE_MAP.items()}


def _curve_to_rust(curve: CurveCycle) -> int:
    try:
        return _RUST_CURVE_MAP[curve]
    except KeyError as exc:
        raise NovaError(f"Unsupported curve for Rust backend: {curve}") from exc


def _curve_from_rust(curve: int) -> CurveCycle:
    try:
        return _RUST_CURVE_MAP_INV[curve]
    except KeyError as exc:
        raise NovaError(f"Unsupported curve from Rust backend: {curve}") from exc


# =============================================================================
# Step Circuit Interface
# =============================================================================


@dataclass(frozen=True)
class StepInput:
    """
    Input to a Nova step circuit.
    
    Contains the public and private inputs for one training step.
    """
    # Public inputs (visible to verifier)
    epoch: int
    prev_state_hash: bytes  # Hash of previous accumulated state
    batch_hash: bytes       # Hash of training batch
    
    # Private inputs (known only to prover)
    prev_weights_hash: bytes
    gradient: Optional[GradientTensor] = None
    learning_rate: float = 0.01
    
    def __post_init__(self) -> None:
        if len(self.prev_state_hash) != STATE_HASH_LEN:
            raise NovaError(f"prev_state_hash must be {STATE_HASH_LEN} bytes")
        if len(self.batch_hash) != STATE_HASH_LEN:
            raise NovaError(f"batch_hash must be {STATE_HASH_LEN} bytes")
        if len(self.prev_weights_hash) != STATE_HASH_LEN:
            raise NovaError(f"prev_weights_hash must be {STATE_HASH_LEN} bytes")


@dataclass(frozen=True)
class StepOutput:
    """
    Output from a Nova step circuit.
    
    Contains the new state after applying the training step.
    """
    epoch: int
    new_state_hash: bytes
    new_weights_hash: bytes
    gradient_commitment: Optional[GradientCommitment] = None
    
    def __post_init__(self) -> None:
        if len(self.new_state_hash) != STATE_HASH_LEN:
            raise NovaError(f"new_state_hash must be {STATE_HASH_LEN} bytes")
        if len(self.new_weights_hash) != STATE_HASH_LEN:
            raise NovaError(f"new_weights_hash must be {STATE_HASH_LEN} bytes")


class StepCircuit(Protocol):
    """
    Protocol for Nova step circuits.
    
    A step circuit defines the computation to be proven at each epoch.
    """
    
    def execute(self, input: StepInput) -> StepOutput:
        """Execute the step circuit and produce output."""
        ...
    
    def num_constraints(self) -> int:
        """Return the number of R1CS constraints in this circuit."""
        ...


# =============================================================================
# Training Step Circuit Implementation
# =============================================================================


class TrainingStepCircuit:
    """
    Nova step circuit for training proof.
    
    Proves correct gradient application:
    1. prev_weights_hash matches accumulated state
    2. gradient was computed from batch (via commitment)
    3. new_weights = prev_weights - lr * gradient
    4. new_weights_hash is correctly computed
    
    NOTE: Actual constraint generation is done in Rust.
    This Python class defines the interface and orchestrates proof generation.
    """
    
    def __init__(
        self,
        model_dimension: int,
        learning_rate: float = 0.01,
        curve: CurveCycle = CurveCycle.PASTA,
    ) -> None:
        """
        Initialize training step circuit.
        
        Args:
            model_dimension: Number of parameters in model
            learning_rate: Learning rate for gradient descent
            curve: Elliptic curve cycle to use
        """
        if model_dimension <= 0:
            raise CircuitError("model_dimension must be positive")
        if learning_rate <= 0 or learning_rate > 1:
            raise CircuitError("learning_rate must be in (0, 1]")
        
        self._model_dim = model_dimension
        self._learning_rate = learning_rate
        self._curve = curve
    
    @property
    def model_dimension(self) -> int:
        return self._model_dim
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def curve(self) -> CurveCycle:
        return self._curve
    
    def execute(self, input: StepInput) -> StepOutput:
        """
        Execute the training step (Python simulation).
        
        NOTE: This is for testing/simulation only. Real proofs are
        generated via the Rust FFI.
        """
        # Compute new state hash
        state_preimage = (
            NOVA_DOMAIN +
            b"STATE" +
            input.epoch.to_bytes(8, "big") +
            input.prev_state_hash +
            input.batch_hash
        )
        new_state_hash = hashlib.sha256(state_preimage).digest()
        
        # Compute new weights hash (simulated)
        weights_preimage = (
            NOVA_DOMAIN +
            b"WEIGHTS" +
            input.prev_weights_hash +
            (input.gradient.hash if input.gradient else b"\x00" * 32)
        )
        new_weights_hash = hashlib.sha256(weights_preimage).digest()
        
        return StepOutput(
            epoch=input.epoch + 1,
            new_state_hash=new_state_hash,
            new_weights_hash=new_weights_hash,
        )
    
    def num_constraints(self) -> int:
        """
        Estimate number of R1CS constraints.
        
        Actual value depends on:
        - Hash function constraints (~1000 per SHA256)
        - Model dimension (gradient check)
        - Field arithmetic
        """
        hash_constraints = 4 * 1000  # 4 hashes per step
        gradient_constraints = self._model_dim * 3  # mul, add, range
        return hash_constraints + gradient_constraints


# =============================================================================
# Folding Instance
# =============================================================================


@dataclass
class FoldingInstance:
    """
    A Nova folding instance representing accumulated computation.
    
    Each instance contains:
    - Accumulated state (commitment to all previous steps)
    - Current epoch number
    - Cross-term for folding verification
    """
    epoch: int
    state_hash: bytes
    commitment: bytes  # Relaxed R1CS commitment
    cross_term: bytes  # T value for folding
    
    # Metadata
    model_dimension: int = 0
    curve: CurveCycle = CurveCycle.PASTA
    
    def __post_init__(self) -> None:
        if len(self.state_hash) != STATE_HASH_LEN:
            raise NovaError(f"state_hash must be {STATE_HASH_LEN} bytes")
        if len(self.commitment) != COMMITMENT_LEN:
            raise NovaError(f"commitment must be {COMMITMENT_LEN} bytes")
        if len(self.cross_term) != CROSS_TERM_LEN:
            raise NovaError(f"cross_term must be {CROSS_TERM_LEN} bytes")
        if self.epoch < 0:
            raise NovaError("epoch cannot be negative")
        if self.model_dimension < 0:
            raise NovaError("model_dimension cannot be negative")
    
    @classmethod
    def genesis(cls, model_dimension: int, curve: CurveCycle = CurveCycle.PASTA) -> "FoldingInstance":
        """Create genesis (epoch 0) folding instance."""
        return cls(
            epoch=0,
            state_hash=b"\x00" * STATE_HASH_LEN,
            commitment=b"\x00" * 48,  # Identity point
            cross_term=b"\x00" * 48,
            model_dimension=model_dimension,
            curve=curve,
        )
    
    def fold(self, other: "FoldingInstance") -> "FoldingInstance":
        """
        Fold two instances into one.
        
        NOTE: This is a simplified simulation. Real folding requires
        computing the cross-term T and challenge r from the transcript.
        
        The folded epoch is the max of both instances (no additional +1).
        """
        if self.curve != other.curve:
            raise NovaError("Cannot fold instances with different curves")
        if self.model_dimension != other.model_dimension:
            raise NovaError("Cannot fold instances with different model dimensions")
        
        # Simplified folding (real impl in Rust)
        new_state = hashlib.sha256(
            NOVA_DOMAIN + b"FOLD" + self.state_hash + other.state_hash
        ).digest()
        
        # Epoch is max of both - the step already advanced epoch
        return FoldingInstance(
            epoch=max(self.epoch, other.epoch),
            state_hash=new_state,
            commitment=self.commitment,  # Simplified
            cross_term=other.cross_term,  # Simplified
            model_dimension=self.model_dimension,
            curve=self.curve,
        )


# =============================================================================
# Nova Prover/Verifier
# =============================================================================


@dataclass(frozen=True)
class NovaProof:
    """
    A Nova proof for incremental training verification.
    
    Contains:
    - Final folding instance (accumulated state)
    - SNARK proof for the final instance
    - Proof metadata
    """
    version: int
    curve: CurveCycle
    final_instance: bytes  # Serialized FoldingInstance
    snark_proof: bytes     # Final SNARK (e.g., Spartan)
    num_steps: int
    model_dimension: int
    
    # Binding to gradient commitments
    commitment_roots: Tuple[bytes, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_PROOF_VERSIONS:
            raise NovaError(f"Unsupported proof version: {self.version}")
        if self.num_steps <= 0:
            raise NovaError("num_steps must be positive")
        if self.num_steps > MAX_STEPS:
            raise NovaError(f"num_steps exceeds maximum {MAX_STEPS}")
        if self.model_dimension <= 0:
            raise NovaError("model_dimension must be positive")
        if self.model_dimension > MAX_MODEL_DIM:
            raise NovaError(f"model_dimension exceeds maximum {MAX_MODEL_DIM}")


class NovaProver:
    """
    Nova prover for generating training proofs.
    
    Orchestrates:
    1. Step circuit execution
    2. Folding instance accumulation
    3. Final SNARK generation
    
    NOTE: Actual cryptographic operations are delegated to Rust FFI.
    This class provides the Python interface.
    """
    
    def __init__(
        self,
        circuit: TrainingStepCircuit,
        rust_backend: Optional[Any] = None,
    ) -> None:
        """
        Initialize Nova prover.
        
        Args:
            circuit: Step circuit definition
            rust_backend: Optional Rust FFI backend for real proofs
        """
        self._circuit = circuit
        self._rust_backend = rust_backend
        self._current_instance = FoldingInstance.genesis(
            circuit.model_dimension,
            circuit.curve,
        )
        self._commitment_roots: List[bytes] = []
    
    @property
    def current_epoch(self) -> int:
        return self._current_instance.epoch
    
    @property
    def current_state_hash(self) -> bytes:
        return self._current_instance.state_hash
    
    def step(
        self,
        batch_hash: bytes,
        gradient: GradientTensor,
        prev_weights_hash: bytes,
    ) -> StepOutput:
        """
        Execute one training step and fold into accumulator.
        
        Args:
            batch_hash: Hash of training batch
            gradient: Computed gradient tensor
            prev_weights_hash: Hash of previous weights
            
        Returns:
            Step output with new state
        """
        input = StepInput(
            epoch=self._current_instance.epoch,
            prev_state_hash=self._current_instance.state_hash,
            batch_hash=batch_hash,
            prev_weights_hash=prev_weights_hash,
            gradient=gradient,
            learning_rate=self._circuit.learning_rate,
        )
        
        # Execute step circuit
        output = self._circuit.execute(input)
        
        # Create new instance and fold
        new_instance = FoldingInstance(
            epoch=output.epoch,
            state_hash=output.new_state_hash,
            commitment=b"\x00" * 48,  # Computed in Rust
            cross_term=b"\x00" * 48,  # Computed in Rust
            model_dimension=self._circuit.model_dimension,
            curve=self._circuit.curve,
        )
        
        self._current_instance = self._current_instance.fold(new_instance)
        
        # Track gradient commitment root
        if output.gradient_commitment:
            self._commitment_roots.append(output.gradient_commitment.commitment)
        
        return output
    
    def finalize(self) -> NovaProof:
        """
        Generate final Nova proof.
        
        Compresses the folding accumulator into a succinct SNARK proof.
        
        Returns:
            NovaProof that can be verified in O(1) time
        """
        if self._current_instance.epoch == 0:
            raise NovaError("Cannot finalize without any steps")
        
        # Serialize final instance
        instance_bytes = (
            NOVA_DOMAIN +
            self._current_instance.state_hash +
            self._current_instance.commitment +
            self._current_instance.cross_term
        )
        
        # Generate SNARK proof (delegated to Rust in production)
        snark_proof = self._generate_snark()
        
        return NovaProof(
            version=PROOF_VERSION,
            curve=self._circuit.curve,
            final_instance=instance_bytes,
            snark_proof=snark_proof,
            num_steps=self._current_instance.epoch,
            model_dimension=self._circuit.model_dimension,
            commitment_roots=tuple(self._commitment_roots),
        )
    
    def _generate_snark(self) -> bytes:
        """
        Generate final SNARK proof.
        
        In production, this calls the Rust FFI backend.
        Here we return a placeholder.
        """
        if self._rust_backend:
            # Call Rust backend
            return self._rust_backend.generate_snark(self._current_instance)
        
        # Placeholder for simulation
        return hashlib.sha256(
            NOVA_DOMAIN + b"SNARK" + self._current_instance.state_hash
        ).digest()


class NovaVerifier:
    """
    Nova verifier for checking training proofs.
    
    Verifies:
    1. Proof metadata matches expected values
    2. Final state hash matches expected
    3. num_steps and commitment_roots binding
    4. SNARK proof validity (requires Rust backend)
    
    WARNING: Without Rust backend, simulation mode must be explicitly enabled.
    """
    
    def __init__(
        self,
        expected_model_dimension: int,
        expected_curve: CurveCycle = CurveCycle.PASTA,
        rust_backend: Optional[Any] = None,
        allow_simulation: bool = False,
        use_rust: bool = True,
    ) -> None:
        """
        Initialize Nova verifier.
        
        Args:
            expected_model_dimension: Expected model dimension
            expected_curve: Expected curve cycle
            rust_backend: Rust FFI backend for real SNARK verification
            allow_simulation: If True, allow verification without Rust backend.
                              WARNING: Simulation does NOT verify SNARK proofs!
            use_rust: If True, auto-detect and use Rust backend when available.
        """
        if expected_model_dimension <= 0:
            raise NovaError("expected_model_dimension must be positive")
        
        self._expected_dim = expected_model_dimension
        self._expected_curve = expected_curve
        self._rust_verifier: Optional[Any] = None
        if rust_backend is not None:
            self._rust_verifier = rust_backend
        elif use_rust and _RUST_AVAILABLE:
            self._rust_verifier = nova_ian.RustNovaVerifier(
                expected_model_dimension,
                _curve_to_rust(expected_curve),
            )
        self._allow_simulation = allow_simulation
    
    def verify(
        self,
        proof: NovaProof,
        expected_final_state: Optional[bytes] = None,
        expected_num_steps: Optional[int] = None,
        expected_commitment_roots: Optional[Tuple[bytes, ...]] = None,
    ) -> bool:
        """
        Verify a Nova training proof.
        
        Args:
            proof: Nova proof to verify
            expected_final_state: Optional expected final state hash
            expected_num_steps: Optional expected number of training steps
            expected_commitment_roots: Optional expected commitment roots
            
        Returns:
            True if proof is valid
            
        Raises:
            ProofVerificationError: If verification fails or no backend
        """
        if self._rust_verifier:
            if expected_commitment_roots is not None:
                if proof.commitment_roots != expected_commitment_roots:
                    raise ProofVerificationError("commitment_roots mismatch")
            try:
                rust_proof = nova_ian.RustNovaProof(
                    proof.version,
                    _curve_to_rust(proof.curve),
                    proof.final_instance,
                    proof.snark_proof,
                    proof.num_steps,
                    proof.model_dimension,
                )
                return self._rust_verifier.verify(
                    rust_proof,
                    expected_final_state,
                    expected_num_steps,
                )
            except Exception as exc:
                raise ProofVerificationError(str(exc)) from exc

        # Check proof metadata
        if proof.model_dimension != self._expected_dim:
            raise ProofVerificationError(
                f"Model dimension mismatch: {proof.model_dimension} != {self._expected_dim}"
            )
        if proof.curve != self._expected_curve:
            raise ProofVerificationError(
                f"Curve mismatch: {proof.curve} != {self._expected_curve}"
            )
        
        # Check expected num_steps binding
        if expected_num_steps is not None and proof.num_steps != expected_num_steps:
            raise ProofVerificationError(
                f"num_steps mismatch: {proof.num_steps} != {expected_num_steps}"
            )
        
        # Check expected commitment_roots binding
        if expected_commitment_roots is not None:
            if proof.commitment_roots != expected_commitment_roots:
                raise ProofVerificationError("commitment_roots mismatch")
        
        # Validate final_instance structure
        min_instance_len = len(NOVA_DOMAIN) + STATE_HASH_LEN + COMMITMENT_LEN + CROSS_TERM_LEN
        if len(proof.final_instance) < min_instance_len:
            raise ProofVerificationError(
                f"Invalid final_instance length: {len(proof.final_instance)} < {min_instance_len}"
            )
        
        # Extract final state
        offset = len(NOVA_DOMAIN)
        final_state = proof.final_instance[offset:offset + STATE_HASH_LEN]
        
        # Validate NOVA_DOMAIN prefix
        if not proof.final_instance.startswith(NOVA_DOMAIN):
            raise ProofVerificationError("Invalid final_instance: missing domain prefix")
        
        # Check expected final state
        if expected_final_state is not None and final_state != expected_final_state:
            raise ProofVerificationError("Final state mismatch")
        
        # Verify SNARK proof with Rust backend
        # No Rust backend - require explicit simulation opt-in
        if not self._allow_simulation:
            raise ProofVerificationError(
                "No Rust backend available. For production, provide a Rust backend. "
                "For testing only, set allow_simulation=True."
            )
        
        # Simulation mode: structure verified, SNARK NOT verified
        return True


# =============================================================================
# Integration with Gradient Commitments
# =============================================================================


class NovaTrainingSession:
    """
    High-level session for Nova-verified training.
    
    Combines:
    - Nova folding for incremental proofs
    - Gradient commitments for verifiable contributions
    - Batch tracking for auditability
    """
    
    def __init__(
        self,
        model_dimension: int,
        learning_rate: float = 0.01,
        curve: CurveCycle = CurveCycle.PASTA,
        use_rust: bool = True,
    ) -> None:
        """
        Initialize training session.
        
        Args:
            model_dimension: Number of model parameters
            learning_rate: Learning rate
            curve: Elliptic curve cycle
            use_rust: If True, auto-detect and use Rust backend when available.
        """
        self._circuit = TrainingStepCircuit(
            model_dimension=model_dimension,
            learning_rate=learning_rate,
            curve=curve,
        )
        self._prover = NovaProver(self._circuit)
        self._rust_prover = None
        self._rust_weights_hash: Optional[bytes] = None
        if use_rust and _RUST_AVAILABLE:
            self._rust_prover = nova_ian.RustNovaProver(
                model_dimension,
                _curve_to_rust(curve),
            )
        self._batch_hashes: List[bytes] = []
        self._gradient_commitments: List[GradientCommitment] = []
    
    @property
    def epoch(self) -> int:
        if self._rust_prover:
            return self._rust_prover.current_epoch
        return self._prover.current_epoch
    
    @property
    def state_hash(self) -> bytes:
        if self._rust_prover:
            return self._rust_prover.current_state_hash
        return self._prover.current_state_hash

    def step(
        self,
        batch_data: bytes,
        gradient: GradientTensor,
        weights_hash: bytes,
        commitment: Optional[GradientCommitment] = None,
    ) -> StepOutput:
        """
        Add a training step to the session.
        
        Args:
            batch_data: Training batch data
            gradient: Computed gradient
            weights_hash: Hash of current weights
            commitment: Optional gradient commitment
            
        Returns:
            Step output with new state
        """
        batch_hash = hashlib.sha256(NOVA_DOMAIN + b"BATCH" + batch_data).digest()
        self._batch_hashes.append(batch_hash)
        
        if commitment:
            self._gradient_commitments.append(commitment)
        
        if self._rust_prover:
            prev_state = self._rust_prover.current_state_hash
            prev_weights = weights_hash
            if self._rust_weights_hash is not None:
                if weights_hash != self._rust_weights_hash:
                    raise NovaError("weights_hash mismatch with Rust prover state")
                prev_weights = self._rust_weights_hash
            epoch, new_state, new_weights = self._rust_prover.step(
                epoch=self._rust_prover.current_epoch,
                prev_state=prev_state,
                weights_hash=prev_weights,
                batch_hash=batch_hash,
                gradient_hash=gradient.hash,
            )
            new_state_bytes = bytes(new_state)
            new_weights_bytes = bytes(new_weights)
            self._rust_weights_hash = new_weights_bytes
            return StepOutput(
                epoch=epoch,
                new_state_hash=new_state_bytes,
                new_weights_hash=new_weights_bytes,
                gradient_commitment=commitment,
            )
        
        return self._prover.step(
            batch_hash=batch_hash,
            gradient=gradient,
            prev_weights_hash=weights_hash,
        )
    
    def add_training_step(
        self,
        batch_data: bytes,
        gradient: GradientTensor,
        weights_hash: bytes,
        commitment: Optional[GradientCommitment] = None,
    ) -> StepOutput:
        """Alias for step()."""
        return self.step(batch_data, gradient, weights_hash, commitment)
    
    def finalize(self) -> NovaProof:
        """
        Finalize training and generate proof.
        
        Returns:
            Nova proof for all training steps
        """
        if self._rust_prover:
            rust_proof = self._rust_prover.finalize()
            return NovaProof(
                version=rust_proof.version,
                curve=_curve_from_rust(rust_proof.curve),
                final_instance=bytes(rust_proof.final_instance),
                snark_proof=bytes(rust_proof.snark_proof),
                num_steps=rust_proof.num_steps,
                model_dimension=rust_proof.model_dimension,
            )
        return self._prover.finalize()
    
    def get_batch_merkle_root(self) -> bytes:
        """
        Get Merkle root of all batch hashes.
        
        Allows efficient verification that specific batches were used.
        """
        if not self._batch_hashes:
            return b"\x00" * STATE_HASH_LEN
        
        # Simple hash for now (real impl uses Merkle tree)
        all_hashes = b"".join(self._batch_hashes)
        return hashlib.sha256(NOVA_DOMAIN + b"BATCHES" + all_hashes).digest()
