"""Property-based tests for secure aggregation primitives using Hypothesis."""

import secrets
from typing import List, Tuple

import pytest
from hypothesis import given, settings, assume, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition, Bundle

from idi.ian.algorithms import BLSOperations
from idi.ian.zkml import (
    SecAggPhase,
    ShamirSecretSharing,
    PairwiseMasking,
    MaskedGradient,
    SecAggSession,
    SecAggParticipant,
)

def make_participant(round_id: bytes, gradient_size: int, threshold: int = 2) -> SecAggParticipant:
    bls = BLSOperations()
    sk, pk = bls.generate_keypair()
    return SecAggParticipant(pk, round_id, gradient_size, sk, threshold=threshold)


# =============================================================================
# Strategies
# =============================================================================

# Valid threshold parameters: 2 <= t <= n, n >= 2
@st.composite
def shamir_params(draw):
    """Generate valid (n, t) parameters for Shamir."""
    n = draw(st.integers(min_value=2, max_value=10))
    t = draw(st.integers(min_value=2, max_value=n))
    return (n, t)


@st.composite
def participant_id(draw):
    """Generate a valid 48-byte participant ID."""
    return draw(st.binary(min_size=48, max_size=48))


@st.composite
def round_id(draw):
    """Generate a valid 32-byte round ID."""
    return draw(st.binary(min_size=32, max_size=32))


@st.composite
def gradient_data(draw, size: int = 100):
    """Generate gradient data of specified size."""
    return draw(st.binary(min_size=size, max_size=size))


# =============================================================================
# Shamir Secret Sharing Properties
# =============================================================================

class TestShamirProperties:
    """Property-based tests for Shamir secret sharing."""
    
    @given(shamir_params())
    @settings(max_examples=50)
    def test_any_t_shares_reconstruct(self, params: Tuple[int, int]):
        """Property: Any t shares can reconstruct the secret."""
        n, t = params
        secret = secrets.token_bytes(32)
        
        shares = ShamirSecretSharing.split(secret, n=n, t=t)
        
        # Take first t shares
        reconstructed = ShamirSecretSharing.reconstruct(shares[:t])
        assert reconstructed == secret
    
    @given(shamir_params(), st.data())
    @settings(max_examples=50)
    def test_any_subset_of_t_shares_reconstructs(self, params: Tuple[int, int], data):
        """Property: Any subset of exactly t shares reconstructs correctly."""
        n, t = params
        secret = secrets.token_bytes(32)
        
        shares = ShamirSecretSharing.split(secret, n=n, t=t)
        
        # Draw t random indices without replacement
        indices = data.draw(st.lists(
            st.integers(min_value=0, max_value=n-1),
            min_size=t,
            max_size=t,
            unique=True,
        ))
        
        selected_shares = [shares[i] for i in indices]
        reconstructed = ShamirSecretSharing.reconstruct(selected_shares)
        assert reconstructed == secret
    
    @given(shamir_params())
    @settings(max_examples=30)
    def test_fewer_than_t_shares_cannot_reconstruct(self, params: Tuple[int, int]):
        """Property: Fewer than t shares cannot reconstruct correctly."""
        n, t = params
        assume(t >= 3)  # Need at least t=3 to test t-1 >= 2
        
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=n, t=t)
        
        # With t-1 shares, reconstruction should produce wrong result
        # (Lagrange interpolation with insufficient points gives wrong polynomial)
        wrong_result = ShamirSecretSharing.reconstruct(shares[:t-1])
        assert wrong_result != secret  # Should NOT match original
    
    @given(shamir_params())
    @settings(max_examples=30)
    def test_more_than_t_shares_still_works(self, params: Tuple[int, int]):
        """Property: More than t shares still reconstructs correctly."""
        n, t = params
        assume(t < n)  # Ensure we can have more than t shares
        
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=n, t=t)
        
        # Use all n shares
        reconstructed = ShamirSecretSharing.reconstruct(shares)
        assert reconstructed == secret
    
    @given(st.binary(min_size=32, max_size=32))
    @settings(max_examples=30)
    def test_split_is_deterministic_per_invocation(self, secret: bytes):
        """Property: Each split produces unique shares (due to random coefficients)."""
        shares1 = ShamirSecretSharing.split(secret, n=5, t=3)
        shares2 = ShamirSecretSharing.split(secret, n=5, t=3)
        
        # Shares should be different due to random coefficients
        # (unless we're astronomically unlucky)
        assert shares1 != shares2
    
    @given(shamir_params())
    @settings(max_examples=30)
    def test_shares_have_unique_indices(self, params: Tuple[int, int]):
        """Property: All shares have unique indices 1..n."""
        n, t = params
        secret = secrets.token_bytes(32)
        
        shares = ShamirSecretSharing.split(secret, n=n, t=t)
        
        indices = [s[0] for s in shares]
        assert len(set(indices)) == n
        assert all(1 <= i <= n for i in indices)


# =============================================================================
# Pairwise Masking Properties
# =============================================================================

class TestPairwiseMaskingProperties:
    """Property-based tests for pairwise masking."""
    
    @given(participant_id(), participant_id(), round_id(), st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_shared_secret_symmetry(self, id_a: bytes, id_b: bytes, round_id: bytes, mask_len: int):
        """Property: Shared secrets are symmetric between any two parties."""
        assume(id_a != id_b)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        pub_a = masking_a.ephemeral_public
        pub_b = masking_b.ephemeral_public
        
        shared_a = masking_a.compute_shared_secret(id_b, pub_b)
        shared_b = masking_b.compute_shared_secret(id_a, pub_a)
        
        assert shared_a == shared_b
    
    @given(participant_id(), participant_id(), round_id(), st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_mask_symmetry(self, id_a: bytes, id_b: bytes, round_id: bytes, mask_len: int):
        """Property: Masks are symmetric between any two parties."""
        assume(id_a != id_b)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        pub_a = masking_a.ephemeral_public
        pub_b = masking_b.ephemeral_public
        
        mask_a = masking_a.compute_pairwise_mask(id_b, pub_b, mask_len)
        mask_b = masking_b.compute_pairwise_mask(id_a, pub_a, mask_len)
        
        assert mask_a == mask_b
        assert len(mask_a) == mask_len
    
    @given(participant_id(), participant_id(), round_id(), st.binary(min_size=100, max_size=100))
    @settings(max_examples=50)
    def test_xor_mask_cancellation(self, id_a: bytes, id_b: bytes, round_id: bytes, gradient: bytes):
        """Property: XOR with same mask twice returns original."""
        assume(id_a != id_b)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        pub_b = masking_b.ephemeral_public
        mask = masking_a.compute_pairwise_mask(id_b, pub_b, len(gradient))
        
        # XOR twice should return original
        masked = bytes(g ^ m for g, m in zip(gradient, mask))
        unmasked = bytes(g ^ m for g, m in zip(masked, mask))
        
        assert unmasked == gradient
    
    @given(round_id(), round_id(), participant_id(), participant_id())
    @settings(max_examples=30)
    def test_different_rounds_produce_different_masks(
        self, round1: bytes, round2: bytes, id_a: bytes, id_b: bytes
    ):
        """Property: Different round IDs produce different masks."""
        assume(round1 != round2)
        assume(id_a != id_b)
        
        masking_a1 = PairwiseMasking(id_a, round1)
        masking_a2 = PairwiseMasking(id_a, round2)
        
        masking_b1 = PairwiseMasking(id_b, round1)
        masking_b2 = PairwiseMasking(id_b, round2)
        
        mask1 = masking_a1.compute_pairwise_mask(id_b, masking_b1.ephemeral_public, 100)
        mask2 = masking_a2.compute_pairwise_mask(id_b, masking_b2.ephemeral_public, 100)
        
        assert mask1 != mask2


# =============================================================================
# Multi-party Aggregation Properties
# =============================================================================

class TestAggregationProperties:
    """Property-based tests for multi-party aggregation."""
    
    @given(st.integers(min_value=2, max_value=3), st.integers(min_value=10, max_value=50))
    @settings(max_examples=3, deadline=None)  # BLS keygen is very slow with py_ecc
    def test_pairwise_masks_cancel_in_aggregate(self, num_participants: int, gradient_size: int):
        """Property: Pairwise masks cancel when all participants aggregate."""
        round_id = secrets.token_bytes(32)
        
        # Create participants
        participants = []
        for _ in range(num_participants):
            p = make_participant(round_id, gradient_size, 2)
            participants.append(p)
        
        # Exchange keys (all pairs)
        for p in participants:
            for other in participants:
                if p.keys.participant_id != other.keys.participant_id:
                    p.add_peer(other.keys)
        
        # Create random gradients
        gradients = [secrets.token_bytes(gradient_size) for _ in participants]
        
        # Mask gradients
        masked_gradients = [p.mask_gradient(g) for p, g in zip(participants, gradients)]
        
        # XOR all masked gradients
        aggregate = bytearray(gradient_size)
        for mg in masked_gradients:
            for i, b in enumerate(mg.masked_data):
                aggregate[i] ^= b
        
        # XOR of original gradients
        expected = bytearray(gradient_size)
        for g in gradients:
            for i, b in enumerate(g):
                expected[i] ^= b
        
        assert bytes(aggregate) == bytes(expected)


# =============================================================================
# SecAgg State Machine Tests
# =============================================================================

@pytest.mark.skip(reason="BLS keygen too slow for state machine exploration")
class SecAggStateMachine(RuleBasedStateMachine):
    """
    State machine test for SecAggSession phase transitions.
    
    Tests that the session correctly enforces phase ordering and
    rejects invalid operations in wrong phases.
    
    NOTE: Skipped by default due to slow BLS keygen with py_ecc.
    Run manually with: pytest -k SecAggStateMachine --run-slow
    """
    
    def __init__(self):
        super().__init__()
        self.round_id = secrets.token_bytes(32)
        self.gradient_size = 100
        self.bls = BLSOperations()
        self.session = SecAggSession(
            round_id=self.round_id,
            epoch=1,
            threshold=2,
            gradient_size=self.gradient_size,
        )
        self.registered_participants: List[ParticipantKeys] = []
        self.submitted_gradients: set = set()
    
    participants = Bundle("participants")
    
    @rule(target=participants)
    def register_participant(self):
        """Register a new participant."""
        if self.session.phase != SecAggPhase.SETUP:
            return None  # Can only register in SETUP
        
        sk, pid = self.bls.generate_keypair()
        participant = SecAggParticipant(
            pid,
            self.round_id,
            self.gradient_size,
            sk,
            threshold=2,
        )
        keys = participant.keys
        self.session.register_participant(keys)
        self.registered_participants.append(keys)
        return keys
    
    @rule()
    @precondition(lambda self: self.session.phase == SecAggPhase.SETUP and len(self.registered_participants) >= 3)
    def start_key_exchange(self):
        """Transition to key exchange phase."""
        self.session.start_key_exchange()
        assert self.session.phase == SecAggPhase.KEY_EXCHANGE
    
    @rule(participant=participants)
    @precondition(lambda self: self.session.phase in (SecAggPhase.KEY_EXCHANGE, SecAggPhase.MASKING, SecAggPhase.AGGREGATION))
    def submit_gradient(self, participant):
        """Submit a masked gradient."""
        if participant is None:
            return
        if participant.participant_id in self.submitted_gradients:
            return  # Already submitted
        
        masked = MaskedGradient(
            participant_id=participant.participant_id,
            round_id=self.round_id,
            epoch=1,
            masked_data=secrets.token_bytes(self.gradient_size),
            commitment_hash=secrets.token_bytes(32),
        )
        self.session.submit_masked_gradient(masked)
        self.submitted_gradients.add(participant.participant_id)
    
    @rule()
    @precondition(lambda self: len(self.submitted_gradients) == len(self.registered_participants) and len(self.registered_participants) > 0)
    def finalize(self):
        """Finalize aggregation."""
        if self.session.phase in (SecAggPhase.COMPLETE, SecAggPhase.FAILED):
            return
        
        result = self.session.finalize()
        if len(self.submitted_gradients) == len(self.registered_participants):
            assert result is not None
            assert self.session.phase == SecAggPhase.COMPLETE
    
    @invariant()
    def phase_is_valid(self):
        """Invariant: Phase is always a valid SecAggPhase."""
        assert isinstance(self.session.phase, SecAggPhase)
    
    @invariant()
    def participant_count_consistent(self):
        """Invariant: Session participant count matches our tracking."""
        assert self.session.num_participants == len(self.registered_participants)
    
    @invariant()
    def submitted_count_consistent(self):
        """Invariant: Submitted gradient count is consistent."""
        assert self.session.num_submitted == len(self.submitted_gradients)


# Skip: BLS keygen too slow for Hypothesis state machine exploration
# TestSecAggStateMachine = SecAggStateMachine.TestCase
