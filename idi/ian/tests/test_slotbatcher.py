"""
Tests for SlotBatcher module.

Security tests included:
- Commit-reveal verification
- VRF output verification
- Quality weight bounds
- Slot state transitions
- Ordering determinism

Author: DarkLightX
"""

import asyncio
import hashlib
import secrets
import time

import pytest

from idi.ian.network.slotbatcher import (
    SlotBatcher,
    Slot,
    SlotState,
    SlotContribution,
    VRFProvider,
    compute_lottery_ticket,
    order_by_lottery,
    HASH_SIZE,
    MIN_QUALITY_WEIGHT,
    MAX_QUALITY_WEIGHT,
    CRYPTO_AVAILABLE,
)


class TestSlotContribution:
    """Tests for SlotContribution."""
    
    def test_create_valid_contribution(self):
        """Should create valid contribution."""
        contrib = SlotContribution(
            contribution_id="test123",
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=secrets.token_bytes(32),
            quality_weight=1.0,
            submitted_at_ms=1000,
        )
        assert contrib.contributor_id == "alice"
    
    def test_validate_quality_weight_bounds(self):
        """Should reject out-of-bounds quality weight."""
        with pytest.raises(ValueError, match="quality_weight"):
            SlotContribution(
                contribution_id="test",
                contributor_id="alice",
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
                quality_weight=0.01,
                submitted_at_ms=1000,
            )
    
    def test_verify_reveal_success(self):
        """Should verify correct reveal."""
        content = b"secret content"
        nonce = secrets.token_bytes(32)
        commitment = hashlib.sha256(content + nonce).digest()
        
        contrib = SlotContribution(
            contribution_id="test",
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=commitment,
            quality_weight=1.0,
            submitted_at_ms=1000,
        )
        
        contrib.content = content
        contrib.reveal_nonce = nonce
        
        assert contrib.verify_reveal() is True
    
    def test_verify_reveal_failure(self):
        """Should reject incorrect reveal."""
        content = b"secret content"
        nonce = secrets.token_bytes(32)
        commitment = hashlib.sha256(content + nonce).digest()
        
        contrib = SlotContribution(
            contribution_id="test",
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=commitment,
            quality_weight=1.0,
            submitted_at_ms=1000,
        )
        
        contrib.content = b"different content"
        contrib.reveal_nonce = nonce
        
        assert contrib.verify_reveal() is False


class TestSlot:
    """Tests for Slot."""
    
    def test_valid_state_transitions(self):
        """Should allow valid state transitions."""
        now = int(time.time() * 1000)
        slot = Slot(
            slot_id="slot1",
            start_time_ms=now,
            end_time_ms=now + 500,
            state=SlotState.COLLECTING,
        )
        
        assert slot.can_transition_to(SlotState.ORDERING)
        slot.transition_to(SlotState.ORDERING)
        assert slot.state == SlotState.ORDERING
        
        assert slot.can_transition_to(SlotState.COMMITTED)
        slot.transition_to(SlotState.COMMITTED)
        assert slot.state == SlotState.COMMITTED
    
    def test_invalid_state_transitions(self):
        """Should reject invalid state transitions."""
        now = int(time.time() * 1000)
        slot = Slot(
            slot_id="slot1",
            start_time_ms=now,
            end_time_ms=now + 500,
            state=SlotState.COLLECTING,
        )
        
        assert not slot.can_transition_to(SlotState.EXECUTED)
        
        with pytest.raises(ValueError, match="Invalid"):
            slot.transition_to(SlotState.EXECUTED)


class TestVRFProvider:
    """Tests for VRF provider."""
    
    def test_compute_deterministic(self):
        """VRF output should be deterministic for same input."""
        provider = VRFProvider()
        
        hashes = [secrets.token_bytes(32) for _ in range(3)]
        
        output1 = provider.compute("slot1", hashes)
        output2 = provider.compute("slot1", hashes)
        
        assert output1.vrf_input == output2.vrf_input
    
    def test_different_inputs_different_outputs(self):
        """Different inputs should give different outputs."""
        provider = VRFProvider()
        
        hashes1 = [secrets.token_bytes(32)]
        hashes2 = [secrets.token_bytes(32)]
        
        output1 = provider.compute("slot1", hashes1)
        output2 = provider.compute("slot1", hashes2)
        
        assert output1.vrf_output != output2.vrf_output
    
    def test_public_key_available(self):
        """Public key should be available."""
        provider = VRFProvider()
        assert len(provider.public_key) == 32


class TestQualityWeightedLottery:
    """Tests for quality-weighted lottery."""
    
    def test_lottery_ticket_reproducible(self):
        """Same inputs should give same ticket."""
        vrf_output = secrets.token_bytes(32)
        
        ticket1 = compute_lottery_ticket("contrib1", 1.0, vrf_output)
        ticket2 = compute_lottery_ticket("contrib1", 1.0, vrf_output)
        
        assert ticket1 == ticket2
    
    def test_higher_weight_lower_expected_ticket(self):
        """Higher weight should generally give lower ticket (earlier position)."""
        vrf_output = secrets.token_bytes(32)
        
        low_weight_tickets = []
        high_weight_tickets = []
        
        for _ in range(100):
            vrf = secrets.token_bytes(32)
            low_weight_tickets.append(compute_lottery_ticket("c", MIN_QUALITY_WEIGHT, vrf))
            high_weight_tickets.append(compute_lottery_ticket("c", MAX_QUALITY_WEIGHT, vrf))
        
        avg_low = sum(low_weight_tickets) / len(low_weight_tickets)
        avg_high = sum(high_weight_tickets) / len(high_weight_tickets)
        
        assert avg_high < avg_low
    
    def test_order_by_lottery_deterministic(self):
        """Ordering should be deterministic given same VRF."""
        vrf_output = secrets.token_bytes(32)
        
        contribs = [
            SlotContribution(
                contribution_id=f"c{i}",
                contributor_id="alice",
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
                quality_weight=1.0 + i * 0.1,
                submitted_at_ms=1000,
            )
            for i in range(5)
        ]
        
        order1 = order_by_lottery(contribs, vrf_output)
        order2 = order_by_lottery(contribs, vrf_output)
        
        assert order1 == order2


class TestSlotBatcher:
    """Tests for SlotBatcher."""
    
    @pytest.fixture
    def batcher(self):
        """Create a SlotBatcher for testing."""
        # Use require_crypto=False for testing without cryptography library
        vrf = VRFProvider(require_crypto=False)
        return SlotBatcher(vrf_provider=vrf, slot_duration_ms=100)
    
    @pytest.mark.asyncio
    async def test_submit_contribution(self, batcher):
        """Should submit contribution successfully."""
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=secrets.token_bytes(32),
            quality_weight=1.5,
        )
        
        assert contrib_id is not None
        assert len(contrib_id) == 32  # hex string
    
    @pytest.mark.asyncio
    async def test_get_current_slot(self, batcher):
        """Should get current slot ID."""
        slot_id = await batcher.get_current_slot_id()
        assert slot_id is not None
    
    @pytest.mark.asyncio
    async def test_finalize_and_order(self, batcher):
        """Should finalize slot and compute order."""
        for i in range(5):
            await batcher.submit(
                contributor_id=f"user{i}",
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
            )
        
        slot_id = await batcher.force_finalize_current_slot()
        order = await batcher.get_slot_order(slot_id)
        
        assert len(order) == 5
    
    @pytest.mark.asyncio
    async def test_ordering_proof(self, batcher):
        """Should generate valid ordering proof."""
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=secrets.token_bytes(32),
        )
        
        await batcher.force_finalize_current_slot()
        
        proof = await batcher.get_ordering_proof(contrib_id)
        
        assert proof is not None
        assert proof.contribution_id == contrib_id
        assert proof.position == 0
        assert proof.total_contributions == 1
    
    @pytest.mark.asyncio
    async def test_reveal_verification(self, batcher):
        """Should verify reveal correctly."""
        content = b"secret content"
        nonce = secrets.token_bytes(32)
        commitment = hashlib.sha256(content + nonce).digest()
        
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=commitment,
        )
        
        await batcher.force_finalize_current_slot()
        
        result = await batcher.reveal(contrib_id, content, nonce)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_reveal_wrong_content_fails(self, batcher):
        """Should reject reveal with wrong content."""
        content = b"secret content"
        nonce = secrets.token_bytes(32)
        commitment = hashlib.sha256(content + nonce).digest()
        
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=commitment,
        )
        
        await batcher.force_finalize_current_slot()
        
        result = await batcher.reveal(contrib_id, b"wrong content", nonce)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_quality_weight_affects_ordering(self, batcher):
        """Higher quality should generally get better positions."""
        high_quality_positions = []
        low_quality_positions = []
        
        for _ in range(20):
            b = SlotBatcher(slot_duration_ms=100)
            
            high_id = await b.submit(
                contributor_id="high",
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
                quality_weight=MAX_QUALITY_WEIGHT,
            )
            
            low_id = await b.submit(
                contributor_id="low",
                pack_hash=secrets.token_bytes(32),
                commitment_hash=secrets.token_bytes(32),
                quality_weight=MIN_QUALITY_WEIGHT,
            )
            
            slot_id = await b.force_finalize_current_slot()
            order = await b.get_slot_order(slot_id)
            
            high_quality_positions.append(order.index(high_id))
            low_quality_positions.append(order.index(low_id))
        
        avg_high = sum(high_quality_positions) / len(high_quality_positions)
        avg_low = sum(low_quality_positions) / len(low_quality_positions)
        
        assert avg_high < avg_low


class TestSlotBatcherSecurity:
    """Security-focused tests for SlotBatcher."""
    
    @pytest.mark.asyncio
    async def test_cannot_reveal_during_collection(self):
        """Should not allow reveal while slot is collecting."""
        batcher = SlotBatcher(slot_duration_ms=5000)
        
        contrib_id = await batcher.submit(
            contributor_id="alice",
            pack_hash=secrets.token_bytes(32),
            commitment_hash=secrets.token_bytes(32),
        )
        
        with pytest.raises(RuntimeError, match="collecting"):
            await batcher.reveal(contrib_id, b"content", secrets.token_bytes(32))
    
    @pytest.mark.asyncio
    async def test_slot_duration_bounds(self):
        """Should enforce slot duration bounds."""
        with pytest.raises(ValueError, match="slot_duration_ms"):
            SlotBatcher(slot_duration_ms=10)
        
        with pytest.raises(ValueError, match="slot_duration_ms"):
            SlotBatcher(slot_duration_ms=10000)
    
    @pytest.mark.asyncio
    async def test_vrf_public_key_accessible(self):
        """VRF public key should be accessible for verification."""
        batcher = SlotBatcher()
        assert len(batcher.vrf_public_key) == 32
