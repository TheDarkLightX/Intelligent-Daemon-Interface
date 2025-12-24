"""Tests for secure aggregation primitives."""

import secrets
import pytest

from idi.ian.zkml import (
    SecAggError,
    SecAggPhase,
    ShamirSecretSharing,
    ParticipantKeys,
    PairwiseMasking,
    MaskedGradient,
    SecAggSession,
    SecAggParticipant,
)


class TestShamirSecretSharing:
    """Tests for Shamir secret sharing."""
    
    def test_split_and_reconstruct_basic(self):
        """Test basic split and reconstruct."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        assert len(shares) == 5
        reconstructed = ShamirSecretSharing.reconstruct(shares[:3])
        assert reconstructed == secret
    
    def test_reconstruct_with_different_shares(self):
        """Test reconstruction with different share subsets."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        # Any 3 shares should work
        assert ShamirSecretSharing.reconstruct(shares[0:3]) == secret
        assert ShamirSecretSharing.reconstruct(shares[1:4]) == secret
        assert ShamirSecretSharing.reconstruct(shares[2:5]) == secret
        assert ShamirSecretSharing.reconstruct([shares[0], shares[2], shares[4]]) == secret
    
    def test_insufficient_shares_fails(self):
        """Test that reconstruction fails with too few shares."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        with pytest.raises(SecAggError):
            ShamirSecretSharing.reconstruct(shares[:1])
    
    def test_reject_duplicate_indices(self):
        """Test that duplicate share indices are rejected."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        # Create shares with duplicate index
        bad_shares = [(1, shares[0][1]), (1, shares[1][1]), (2, shares[2][1])]
        with pytest.raises(SecAggError, match="Duplicate"):
            ShamirSecretSharing.reconstruct(bad_shares)
    
    def test_reject_zero_index(self):
        """Test that zero index is rejected."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        bad_shares = [(0, shares[0][1]), (1, shares[1][1])]
        with pytest.raises(SecAggError, match="positive"):
            ShamirSecretSharing.reconstruct(bad_shares)
    
    def test_reject_negative_index(self):
        """Test that negative index is rejected."""
        secret = secrets.token_bytes(32)
        shares = ShamirSecretSharing.split(secret, n=5, t=3)
        
        bad_shares = [(-1, shares[0][1]), (1, shares[1][1])]
        with pytest.raises(SecAggError, match="positive"):
            ShamirSecretSharing.reconstruct(bad_shares)
    
    def test_invalid_threshold(self):
        """Test that invalid threshold is rejected."""
        secret = secrets.token_bytes(32)
        
        with pytest.raises(SecAggError):
            ShamirSecretSharing.split(secret, n=5, t=1)  # t must be >= 2
    
    def test_n_less_than_t(self):
        """Test that n < t is rejected."""
        secret = secrets.token_bytes(32)
        
        with pytest.raises(SecAggError):
            ShamirSecretSharing.split(secret, n=2, t=3)
    
    def test_wrong_secret_length(self):
        """Test that wrong secret length is rejected."""
        with pytest.raises(SecAggError):
            ShamirSecretSharing.split(b"short", n=5, t=3)


class TestPairwiseMasking:
    """Tests for pairwise masking."""
    
    def test_symmetric_shared_secret(self):
        """Test that shared secrets are symmetric."""
        id_a = secrets.token_bytes(48)
        id_b = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        # Exchange public keys
        pub_a = masking_a.ephemeral_public
        pub_b = masking_b.ephemeral_public
        
        # Compute shared secrets
        shared_a = masking_a.compute_shared_secret(id_b, pub_b)
        shared_b = masking_b.compute_shared_secret(id_a, pub_a)
        
        assert shared_a == shared_b
    
    def test_symmetric_mask(self):
        """Test that masks are symmetric."""
        id_a = secrets.token_bytes(48)
        id_b = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        pub_a = masking_a.ephemeral_public
        pub_b = masking_b.ephemeral_public
        
        mask_a = masking_a.compute_pairwise_mask(id_b, pub_b, 100)
        mask_b = masking_b.compute_pairwise_mask(id_a, pub_a, 100)
        
        assert mask_a == mask_b
    
    def test_mask_cancellation(self):
        """Test that masks cancel when XORed."""
        id_a = secrets.token_bytes(48)
        id_b = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        masking_a = PairwiseMasking(id_a, round_id)
        masking_b = PairwiseMasking(id_b, round_id)
        
        pub_a = masking_a.ephemeral_public
        pub_b = masking_b.ephemeral_public
        
        # Create gradients
        grad_a = secrets.token_bytes(100)
        grad_b = secrets.token_bytes(100)
        
        # Compute masks (same for both)
        mask = masking_a.compute_pairwise_mask(id_b, pub_b, 100)
        
        # Apply masks
        masked_a = bytes(a ^ m for a, m in zip(grad_a, mask))
        masked_b = bytes(b ^ m for b, m in zip(grad_b, mask))
        
        # XOR masked gradients
        aggregate = bytes(a ^ b for a, b in zip(masked_a, masked_b))
        
        # Should equal XOR of original gradients (masks cancel)
        expected = bytes(a ^ b for a, b in zip(grad_a, grad_b))
        assert aggregate == expected


class TestSecAggSession:
    """Tests for SecAgg session management."""
    
    def test_session_creation(self):
        """Test session creation."""
        round_id = secrets.token_bytes(32)
        session = SecAggSession(
            round_id=round_id,
            epoch=1,
            threshold=2,
            gradient_size=1000,
        )
        
        assert session.phase == SecAggPhase.SETUP
        assert session.num_participants == 0
    
    def test_participant_registration(self):
        """Test participant registration."""
        round_id = secrets.token_bytes(32)
        session = SecAggSession(
            round_id=round_id,
            epoch=1,
            threshold=2,
            gradient_size=1000,
        )
        
        for i in range(3):
            pid = secrets.token_bytes(48)
            keys = ParticipantKeys(
                participant_id=pid,
                ephemeral_public=secrets.token_bytes(32),
                round_id=round_id,
            )
            session.register_participant(keys)
        
        assert session.num_participants == 3
    
    def test_phase_transitions(self):
        """Test phase transitions."""
        round_id = secrets.token_bytes(32)
        session = SecAggSession(
            round_id=round_id,
            epoch=1,
            threshold=2,
            gradient_size=100,
        )
        
        # Register participants
        participants = []
        for i in range(3):
            pid = secrets.token_bytes(48)
            keys = ParticipantKeys(
                participant_id=pid,
                ephemeral_public=secrets.token_bytes(32),
                round_id=round_id,
            )
            session.register_participant(keys)
            participants.append(keys)
        
        assert session.phase == SecAggPhase.SETUP
        
        session.start_key_exchange()
        assert session.phase == SecAggPhase.KEY_EXCHANGE
    
    def test_aggregation_requires_all_participants(self):
        """Test that aggregation requires all participants."""
        round_id = secrets.token_bytes(32)
        session = SecAggSession(
            round_id=round_id,
            epoch=1,
            threshold=2,
            gradient_size=100,
        )
        
        participants = []
        for i in range(3):
            pid = secrets.token_bytes(48)
            keys = ParticipantKeys(
                participant_id=pid,
                ephemeral_public=secrets.token_bytes(32),
                round_id=round_id,
            )
            session.register_participant(keys)
            participants.append(keys)
        
        session.start_key_exchange()
        
        # Submit only 2 of 3 gradients
        for keys in participants[:2]:
            masked = MaskedGradient(
                participant_id=keys.participant_id,
                round_id=round_id,
                epoch=1,
                masked_data=secrets.token_bytes(100),
                commitment_hash=secrets.token_bytes(32),
            )
            session.submit_masked_gradient(masked)
        
        assert not session.can_aggregate
        result = session.finalize()
        assert result is None
        assert session.phase == SecAggPhase.FAILED


class TestSecAggParticipant:
    """Tests for SecAgg participant."""
    
    def test_participant_creation(self):
        """Test participant creation."""
        pid = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        participant = SecAggParticipant(
            participant_id=pid,
            round_id=round_id,
            gradient_size=1000,
            threshold=2,
        )
        
        assert participant.keys.participant_id == pid
        assert participant.keys.round_id == round_id
    
    def test_add_peer(self):
        """Test adding peer keys."""
        pid = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        participant = SecAggParticipant(pid, round_id, 1000, 2)
        
        peer_keys = ParticipantKeys(
            participant_id=secrets.token_bytes(48),
            ephemeral_public=secrets.token_bytes(32),
            round_id=round_id,
        )
        participant.add_peer(peer_keys)
    
    def test_cannot_add_self_as_peer(self):
        """Test that adding self as peer fails."""
        pid = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        participant = SecAggParticipant(pid, round_id, 1000, 2)
        
        with pytest.raises(SecAggError):
            participant.add_peer(participant.keys)
    
    def test_mask_gradient(self):
        """Test gradient masking."""
        pid = secrets.token_bytes(48)
        round_id = secrets.token_bytes(32)
        
        participant = SecAggParticipant(pid, round_id, 100, 2)
        
        # Add a peer
        peer_keys = ParticipantKeys(
            participant_id=secrets.token_bytes(48),
            ephemeral_public=secrets.token_bytes(32),
            round_id=round_id,
        )
        participant.add_peer(peer_keys)
        
        gradient = secrets.token_bytes(100)
        masked = participant.mask_gradient(gradient)
        
        assert len(masked.masked_data) == 100
        assert masked.participant_id == pid
        assert masked.round_id == round_id


class TestEndToEndSecAgg:
    """End-to-end secure aggregation tests."""
    
    def test_full_aggregation_flow(self):
        """Test complete aggregation with mask cancellation."""
        round_id = secrets.token_bytes(32)
        gradient_size = 100
        num_participants = 3
        
        # Create participants
        participants = []
        for _ in range(num_participants):
            pid = secrets.token_bytes(48)
            p = SecAggParticipant(pid, round_id, gradient_size, 2)
            participants.append(p)
        
        # Exchange keys
        for p in participants:
            for other in participants:
                if p.keys.participant_id != other.keys.participant_id:
                    p.add_peer(other.keys)
        
        # Create session
        session = SecAggSession(
            round_id=round_id,
            epoch=1,
            threshold=2,
            gradient_size=gradient_size,
        )
        
        for p in participants:
            session.register_participant(p.keys)
        
        session.start_key_exchange()
        
        # Create gradients and mask them
        gradients = [secrets.token_bytes(gradient_size) for _ in participants]
        
        for p, g in zip(participants, gradients):
            masked = p.mask_gradient(g, epoch=1)
            session.submit_masked_gradient(masked)
        
        # Finalize
        assert session.can_aggregate
        aggregate = session.finalize()
        
        assert aggregate is not None
        assert len(aggregate) == gradient_size
        assert session.phase == SecAggPhase.COMPLETE
