"""
Tests for ESSO-verified model invariants.

These tests validate that the production code maintains the invariants
formally proven by ESSO's k-induction verification.
"""

import pytest
import asyncio
from unittest.mock import MagicMock

from idi.ian.network.p2p_manager import (
    TokenBucketRateLimiter,
    PeerSession,
    PeerState,
)


# =============================================================================
# TokenBucketRateLimiter Invariant Tests (rate_limiter.json)
# =============================================================================

class TestTokenBucketRateLimiterInvariants:
    """Test ESSO-proven invariants for TokenBucketRateLimiter."""

    def test_initial_state_satisfies_invariants(self):
        """ESSO invariant: init implies tokens <= burst."""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
        assert 0 <= limiter.tokens <= limiter.burst, "tokens_bounded_by_burst violated"
        assert limiter.tokens >= 0, "tokens_non_negative violated"

    def test_burst_domain_bounds(self):
        """ESSO domain: 1 <= burst <= 20."""
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate=10.0, burst=0)
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate=10.0, burst=21)

    @pytest.mark.asyncio
    async def test_acquire_preserves_invariants(self):
        """ESSO invariant: acquire action preserves tokens >= 0."""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

        # Drain all tokens
        for _ in range(10):
            await limiter.acquire()

        # Invariant must still hold
        assert limiter.tokens >= 0, "tokens_non_negative violated after exhaustion"
        assert limiter.tokens <= limiter.burst, "tokens_bounded_by_burst violated"

    def test_reset_preserves_invariants(self):
        """ESSO invariant: reset action sets tokens = burst."""
        limiter = TokenBucketRateLimiter(rate=10.0, burst=15)
        limiter.tokens = 0.0
        limiter.reset()

        assert limiter.tokens == limiter.burst, "reset did not set tokens to burst"
        assert 0 <= limiter.tokens <= limiter.burst


# =============================================================================
# PeerSession Invariant Tests (peer_state_fsm.json)
# =============================================================================

class TestPeerSessionInvariants:
    """Test ESSO-proven invariants for PeerSession."""

    def test_initial_state_is_disconnected(self):
        """ESSO init: state starts at DISCONNECTED."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)
        assert session.state == PeerState.DISCONNECTED

    def test_ready_requires_verified(self):
        """ESSO invariant: READY => verified."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)
        
        # Transition to HANDSHAKING state properly
        session.transition_to(PeerState.CONNECTING)
        session.transition_to(PeerState.CONNECTED)
        session.transition_to(PeerState.HANDSHAKING)

        # Try to transition to READY without verified=True
        session.verified = False
        with pytest.raises(AssertionError, match="ready_requires_verified"):
            session.transition_to(PeerState.READY)

        # With verified=True, should succeed
        session.verified = True
        session.handshake_completed = True
        session.transition_to(PeerState.READY)
        assert session.state == PeerState.READY

    def test_valid_state_transitions(self):
        """ESSO model: only declared transitions are valid."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)

        # DISCONNECTED -> CONNECTING is valid
        session.transition_to(PeerState.CONNECTING)
        assert session.state == PeerState.CONNECTING

        # CONNECTING -> CONNECTED is valid
        session.transition_to(PeerState.CONNECTED)
        assert session.state == PeerState.CONNECTED

    def test_invalid_state_transitions_rejected(self):
        """ESSO model: undeclared transitions raise ValueError."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)

        # DISCONNECTED -> READY is NOT valid (must go through handshake)
        with pytest.raises(ValueError, match="Invalid state transition"):
            session.transition_to(PeerState.READY)

    def test_full_happy_path_transition(self):
        """ESSO model: full CONNECTING->READY path is valid."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)

        # Full path
        session.transition_to(PeerState.CONNECTING)
        session.transition_to(PeerState.CONNECTED)
        session.transition_to(PeerState.HANDSHAKING)
        session.verified = True  # Required for READY
        session.handshake_completed = True
        session.transition_to(PeerState.READY)
        session.transition_to(PeerState.DISCONNECTING)
        session.transition_to(PeerState.DISCONNECTED)

        assert session.state == PeerState.DISCONNECTED


# =============================================================================
# Integration: Cross-Model Invariant Tests
# =============================================================================

class TestCrossModelInvariants:
    """Integration tests for multiple ESSO-verified components."""

    @pytest.mark.asyncio
    async def test_rate_limiting_in_peer_session(self):
        """Rate limiter in peer session maintains both sets of invariants."""
        session = PeerSession(node_id="test", address="127.0.0.1", port=9000)
        session.rate_limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

        # Transition to connected state
        session.transition_to(PeerState.CONNECTING)
        session.transition_to(PeerState.CONNECTED)

        # Use rate limiter
        for _ in range(10):
            await session.rate_limiter.acquire()

        # Both invariants hold
        assert 0 <= session.rate_limiter.tokens <= session.rate_limiter.burst
        assert session.state in (PeerState.CONNECTED, PeerState.HANDSHAKING, PeerState.READY)
