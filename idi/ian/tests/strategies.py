"""Hypothesis strategies for IAN P2P message fuzzing and security testing.

Provides deterministic, bounded generators for:
- P2P protocol messages (valid and malformed)
- Handshake challenge/response sequences
- Replay attack scenarios
- Adversarial event schedules

Design Constraints:
- All generators are seeded for reproducibility
- Bounded output sizes to prevent DoS during testing
- Separate strategies for valid vs adversarial inputs

Usage:
    from idi.ian.tests.strategies import valid_message_strategy, malformed_message_strategy
    
    @given(msg=valid_message_strategy())
    def test_message_roundtrip(msg):
        ...
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from hypothesis import assume, settings
    from hypothesis import strategies as st
    from hypothesis.strategies import SearchStrategy

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Stub for imports when hypothesis not available
    st = None  # type: ignore
    SearchStrategy = Any  # type: ignore

    def assume(condition: bool) -> None:  # type: ignore
        pass


# =============================================================================
# Constants / Bounds
# =============================================================================

MAX_NODE_ID_LEN = 40
MAX_NONCE_LEN = 64
MAX_SIGNATURE_LEN = 128
MAX_PAYLOAD_SIZE = 1024
MAX_TIMESTAMP_DRIFT_MS = 300_000  # 5 minutes
MAX_PEERS_IN_EXCHANGE = 10
MAX_ADDRESSES_PER_PEER = 5


# =============================================================================
# Primitive Strategies
# =============================================================================

if HAS_HYPOTHESIS:
    # Node ID: 40-char hex string (SHA256[:40] of public key)
    node_id_strategy: SearchStrategy[str] = st.from_regex(
        r"[0-9a-f]{40}", fullmatch=True
    )

    # Nonce: base64-encoded random bytes
    nonce_strategy: SearchStrategy[str] = st.binary(
        min_size=8, max_size=32
    ).map(lambda b: base64.b64encode(b).decode("utf-8"))

    # Timestamp: milliseconds, use a fixed reference point for determinism
    # Reference: 2024-01-01 00:00:00 UTC = 1704067200000 ms
    _REFERENCE_TIMESTAMP_MS = 1704067200000

    timestamp_strategy: SearchStrategy[int] = st.integers(
        min_value=_REFERENCE_TIMESTAMP_MS - MAX_TIMESTAMP_DRIFT_MS,
        max_value=_REFERENCE_TIMESTAMP_MS + MAX_TIMESTAMP_DRIFT_MS,
    )

    # Signature: 64-byte Ed25519 signature (or None)
    signature_strategy: SearchStrategy[Optional[bytes]] = st.one_of(
        st.none(),
        st.binary(min_size=64, max_size=64),
    )

    # Challenge nonce: 32-byte hex string
    challenge_nonce_strategy: SearchStrategy[str] = st.from_regex(
        r"[0-9a-f]{64}", fullmatch=True
    )

    # Public key: 32-byte base64
    public_key_b64_strategy: SearchStrategy[str] = st.binary(
        min_size=32, max_size=32
    ).map(lambda b: base64.b64encode(b).decode("utf-8"))

    # Address: tcp://host:port format
    address_strategy: SearchStrategy[str] = st.builds(
        lambda host, port: f"tcp://{host}:{port}",
        host=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
        port=st.integers(min_value=1024, max_value=65535),
    )


# =============================================================================
# Message Type Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def base_message_fields(draw: st.DrawFn) -> Dict[str, Any]:
        """Draw common fields for all message types."""
        sig = draw(signature_strategy)
        return {
            "sender_id": draw(node_id_strategy),
            "timestamp": draw(timestamp_strategy),
            "nonce": draw(nonce_strategy),
            "signature": sig.hex() if sig else None,
        }

    @st.composite
    def ping_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid PING message."""
        base = draw(base_message_fields())
        base["type"] = "ping"
        return base

    @st.composite
    def pong_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid PONG message."""
        base = draw(base_message_fields())
        base["type"] = "pong"
        base["ping_nonce"] = draw(nonce_strategy)
        return base

    @st.composite
    def handshake_challenge_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid HANDSHAKE_CHALLENGE message."""
        sender_id = draw(node_id_strategy)
        return {
            "type": "handshake_challenge",
            "sender_id": sender_id,
            "timestamp": draw(timestamp_strategy),
            "nonce": draw(nonce_strategy),
            "signature": None,
            "challenge_nonce": draw(challenge_nonce_strategy),
            "kx_public_key": draw(public_key_b64_strategy),
            "public_key": draw(public_key_b64_strategy),
        }

    @st.composite
    def handshake_response_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid HANDSHAKE_RESPONSE message."""
        return {
            "type": "handshake_response",
            "sender_id": draw(node_id_strategy),
            "timestamp": draw(timestamp_strategy),
            "nonce": draw(nonce_strategy),
            "signature": None,
            "challenge_nonce": draw(challenge_nonce_strategy),
            "response_nonce": draw(challenge_nonce_strategy),
            "kx_public_key": draw(public_key_b64_strategy),
            "public_key": draw(public_key_b64_strategy),
        }

    @st.composite
    def contribution_announce_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid CONTRIBUTION_ANNOUNCE message."""
        base = draw(base_message_fields())
        base["type"] = "contribution_announce"
        base["goal_id"] = draw(st.from_regex(r"[A-Z][A-Z0-9_]{2,31}", fullmatch=True))
        base["contribution_hash"] = draw(st.from_regex(r"[0-9a-f]{64}", fullmatch=True))
        base["contributor_id"] = draw(st.from_regex(r"[a-z][a-z0-9_]{2,31}", fullmatch=True))
        base["score"] = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        base["log_index"] = draw(st.integers(min_value=0, max_value=1_000_000))
        return base

    @st.composite
    def peer_exchange_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate valid PEER_EXCHANGE message."""
        base = draw(base_message_fields())
        base["type"] = "peer_exchange"
        base["peers"] = draw(
            st.lists(
                st.fixed_dictionaries({
                    "node_id": node_id_strategy,
                    "addresses": st.lists(address_strategy, min_size=1, max_size=MAX_ADDRESSES_PER_PEER),
                }),
                max_size=MAX_PEERS_IN_EXCHANGE,
            )
        )
        return base

    # Combined valid message strategy
    def valid_message_strategy() -> SearchStrategy[Dict[str, Any]]:
        """Generate any valid P2P message type."""
        return st.one_of(
            ping_message_strategy(),
            pong_message_strategy(),
            handshake_challenge_strategy(),
            handshake_response_strategy(),
            contribution_announce_strategy(),
            peer_exchange_strategy(),
        )


# =============================================================================
# Adversarial / Malformed Message Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def replay_message_strategy(draw: st.DrawFn) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a message and its exact replay (same nonce, same sender_id)."""
        original = draw(valid_message_strategy())
        # Replay is identical
        replay = dict(original)
        return original, replay

    @st.composite
    def stale_timestamp_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with timestamp outside acceptable window."""
        msg = draw(valid_message_strategy())
        # Make timestamp very old (> 5 minutes) - use fixed reference for determinism
        msg["timestamp"] = _REFERENCE_TIMESTAMP_MS - MAX_TIMESTAMP_DRIFT_MS - draw(
            st.integers(min_value=1000, max_value=3_600_000)
        )
        return msg

    @st.composite
    def future_timestamp_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with timestamp too far in the future."""
        msg = draw(valid_message_strategy())
        # Use fixed reference for determinism
        msg["timestamp"] = _REFERENCE_TIMESTAMP_MS + MAX_TIMESTAMP_DRIFT_MS + draw(
            st.integers(min_value=1000, max_value=3_600_000)
        )
        return msg

    @st.composite
    def malformed_json_strategy(draw: st.DrawFn) -> bytes:
        """Generate bytes that are not valid JSON."""
        return draw(st.one_of(
            st.binary(min_size=1, max_size=MAX_PAYLOAD_SIZE),
            st.text(min_size=1, max_size=100).map(lambda s: s.encode("utf-8")),
            st.just(b"{incomplete"),
            st.just(b"[1,2,3"),
            st.just(b'"unclosed string'),
        ))

    @st.composite
    def unknown_type_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with unknown type field."""
        msg = draw(valid_message_strategy())
        msg["type"] = draw(st.from_regex(r"unknown_[a-z]{4,8}", fullmatch=True))
        return msg

    @st.composite
    def missing_fields_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with required fields removed."""
        msg = draw(valid_message_strategy())
        # Remove a random required field
        required_fields = ["type", "sender_id", "timestamp", "nonce"]
        field_to_remove = draw(st.sampled_from(required_fields))
        msg.pop(field_to_remove, None)
        return msg

    @st.composite
    def oversized_payload_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with oversized payload field."""
        msg = draw(valid_message_strategy())
        # Add a large extra field
        msg["extra_data"] = "X" * draw(st.integers(min_value=100_000, max_value=1_000_000))
        return msg

    @st.composite
    def invalid_signature_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate message with invalid signature (wrong length or random bytes)."""
        msg = draw(valid_message_strategy())
        msg["signature"] = draw(st.one_of(
            st.binary(min_size=1, max_size=63).map(lambda b: b.hex()),  # Too short
            st.binary(min_size=65, max_size=128).map(lambda b: b.hex()),  # Too long
            st.binary(min_size=64, max_size=64).map(lambda b: b.hex()),  # Random (won't verify)
        ))
        return msg

    def adversarial_message_strategy() -> SearchStrategy[Dict[str, Any]]:
        """Generate any adversarial message type."""
        return st.one_of(
            stale_timestamp_message_strategy(),
            future_timestamp_message_strategy(),
            unknown_type_message_strategy(),
            missing_fields_message_strategy(),
            invalid_signature_message_strategy(),
        )


# =============================================================================
# Event Schedule Strategies (for simulation)
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def network_event_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate a single network event (connect, disconnect, message, delay)."""
        event_type = draw(st.sampled_from([
            "connect", "disconnect", "send_message", "delay", "partition", "heal"
        ]))

        event: Dict[str, Any] = {"type": event_type}

        if event_type == "connect":
            event["peer_id"] = draw(node_id_strategy)
            event["address"] = draw(address_strategy)
        elif event_type == "disconnect":
            event["peer_id"] = draw(node_id_strategy)
            event["reason"] = draw(st.sampled_from(["timeout", "error", "manual", "rate_limit"]))
        elif event_type == "send_message":
            event["message"] = draw(st.one_of(
                valid_message_strategy(),
                adversarial_message_strategy(),
            ))
        elif event_type == "delay":
            event["duration_ms"] = draw(st.integers(min_value=10, max_value=5000))
        elif event_type in ("partition", "heal"):
            event["peers"] = draw(st.lists(node_id_strategy, min_size=1, max_size=5))

        return event

    @st.composite
    def event_schedule_strategy(
        draw: st.DrawFn,
        max_events: int = 50,
    ) -> List[Dict[str, Any]]:
        """Generate a bounded schedule of network events for simulation."""
        return draw(st.lists(
            network_event_strategy(),
            min_size=1,
            max_size=max_events,
        ))


# =============================================================================
# Rate Limiting / DoS Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def message_burst_strategy(
        draw: st.DrawFn,
        burst_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate a burst of messages from same sender (DoS simulation)."""
        sender_id = draw(node_id_strategy)
        messages = []
        for i in range(burst_size):
            msg = draw(valid_message_strategy())
            msg["sender_id"] = sender_id
            # Each message needs unique nonce - use index-based deterministic nonce
            msg["nonce"] = draw(nonce_strategy)
            # Use fixed reference timestamp for determinism
            msg["timestamp"] = _REFERENCE_TIMESTAMP_MS + i
            messages.append(msg)
        return messages


# =============================================================================
# Helpers for Test Setup
# =============================================================================

def serialize_message(msg: Dict[str, Any]) -> bytes:
    """Serialize message dict to wire format (length-prefixed JSON)."""
    json_bytes = json.dumps(msg).encode("utf-8")
    length = len(json_bytes)
    return length.to_bytes(4, "big") + json_bytes


def make_deterministic_node_id(seed: int) -> str:
    """Generate deterministic node_id from seed for reproducible tests."""
    return hashlib.sha256(f"node_seed_{seed}".encode()).hexdigest()[:40]


def make_deterministic_nonce(seed: int) -> str:
    """Generate deterministic nonce from seed for reproducible tests."""
    return base64.b64encode(
        hashlib.sha256(f"nonce_seed_{seed}".encode()).digest()[:16]
    ).decode("utf-8")
