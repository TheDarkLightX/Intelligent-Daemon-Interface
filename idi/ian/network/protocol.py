"""
IAN P2P Protocol - Message definitions and serialization.

Message Types:
1. ContributionAnnounce - Gossip new contributions
2. ContributionRequest/Response - Fetch contribution bodies
3. StateRequest/Response - Sync coordinator state
4. PeerExchange - Share peer lists

Wire Format:
- Length-prefixed JSON for simplicity
- Future: Protocol Buffers or MessagePack for efficiency

Security:
- All messages include sender signature
- Recipients verify signatures before processing
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from idi.ian.models import Contribution, ContributionMeta

# =============================================================================
# Message Types
# =============================================================================

class MessageType(Enum):
    """P2P message types."""
    # Handshake
    HANDSHAKE_CHALLENGE = "handshake_challenge"
    HANDSHAKE_RESPONSE = "handshake_response"

    # Gossip
    CONTRIBUTION_ANNOUNCE = "contribution_announce"

    # Request/Response
    CONTRIBUTION_REQUEST = "contribution_request"
    CONTRIBUTION_RESPONSE = "contribution_response"
    STATE_REQUEST = "state_request"
    STATE_RESPONSE = "state_response"

    # State Sync
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

    # Discovery
    PEER_EXCHANGE = "peer_exchange"
    PING = "ping"
    PONG = "pong"


# =============================================================================
# Message Base
# =============================================================================

@dataclass
class Message:
    """
    Base P2P message.

    All messages include:
    - type: Message type
    - sender_id: Node ID of sender
    - timestamp: Unix timestamp (ms)
    - nonce: Random nonce for uniqueness
    - signature: Sender signature (optional)
    """
    type: MessageType
    sender_id: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    nonce: str = field(default_factory=lambda: base64.b64encode(hashlib.sha256(str(time.time_ns()).encode()).digest()[:8]).decode())
    signature: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": base64.b64encode(self.signature).decode() if self.signature else None,
        }

    def signing_payload(self) -> bytes:
        """Get payload for signing (excludes signature)."""
        data = self.to_dict()
        data.pop("signature", None)
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        msg_type = MessageType(cast(str, data["type"]))

        # Dispatch to specific message type
        type_map: dict[MessageType, type[Message]] = {
            MessageType.HANDSHAKE_CHALLENGE: HandshakeChallenge,
            MessageType.HANDSHAKE_RESPONSE: HandshakeResponse,
            MessageType.CONTRIBUTION_ANNOUNCE: ContributionAnnounce,
            MessageType.CONTRIBUTION_REQUEST: ContributionRequest,
            MessageType.CONTRIBUTION_RESPONSE: ContributionResponse,
            MessageType.STATE_REQUEST: StateRequest,
            MessageType.STATE_RESPONSE: StateResponse,
            MessageType.PEER_EXCHANGE: PeerExchange,
            MessageType.PING: Ping,
            MessageType.PONG: Pong,
        }

        msg_cls: type[Message] = type_map.get(msg_type, Message)
        return msg_cls._from_dict_impl(data)

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> Message:
        return cls(
            type=MessageType(data["type"]),
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
        )

    def to_wire(self) -> bytes:
        """Serialize to wire format (length-prefixed JSON)."""
        json_bytes = json.dumps(self.to_dict()).encode()
        length = len(json_bytes)
        return length.to_bytes(4, 'big') + json_bytes

    @classmethod
    def from_wire(cls, data: bytes) -> Message:
        """Deserialize from wire format."""
        length = int.from_bytes(data[:4], 'big')
        json_bytes = data[4:4 + length]
        parsed = json.loads(json_bytes)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid wire payload: expected JSON object")
        return cls.from_dict(cast(dict[str, Any], parsed))

    def message_id(self) -> str:
        """Unique message ID for deduplication."""
        return f"{self.sender_id}:{self.nonce}"


@dataclass
class HandshakeChallenge(Message):
    type: MessageType = field(default=MessageType.HANDSHAKE_CHALLENGE, init=False)

    challenge_nonce: str = ""
    kx_public_key: str = ""
    public_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "challenge_nonce": self.challenge_nonce,
                "kx_public_key": self.kx_public_key,
                "public_key": self.public_key,
            }
        )
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> HandshakeChallenge:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            challenge_nonce=data.get("challenge_nonce", ""),
            kx_public_key=data.get("kx_public_key", ""),
            public_key=data.get("public_key", ""),
        )


@dataclass
class HandshakeResponse(Message):
    type: MessageType = field(default=MessageType.HANDSHAKE_RESPONSE, init=False)

    challenge_nonce: str = ""
    response_nonce: str = ""
    kx_public_key: str = ""
    public_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "challenge_nonce": self.challenge_nonce,
                "response_nonce": self.response_nonce,
                "kx_public_key": self.kx_public_key,
                "public_key": self.public_key,
            }
        )
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> HandshakeResponse:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            challenge_nonce=data.get("challenge_nonce", ""),
            response_nonce=data.get("response_nonce", ""),
            kx_public_key=data.get("kx_public_key", ""),
            public_key=data.get("public_key", ""),
        )


# =============================================================================
# Contribution Messages
# =============================================================================

@dataclass
class ContributionAnnounce(Message):
    """
    Announce a new contribution (gossip).

    Contains metadata only - peers request full body if interested.
    """
    type: MessageType = field(default=MessageType.CONTRIBUTION_ANNOUNCE, init=False)

    goal_id: str = ""
    contribution_hash: str = ""  # SHA-256 of full contribution
    contributor_id: str = ""
    score: float | None = None
    log_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "goal_id": self.goal_id,
            "contribution_hash": self.contribution_hash,
            "contributor_id": self.contributor_id,
            "score": self.score,
            "log_index": self.log_index,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> ContributionAnnounce:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            goal_id=data["goal_id"],
            contribution_hash=data["contribution_hash"],
            contributor_id=data["contributor_id"],
            score=data.get("score"),
            log_index=data.get("log_index"),
        )

    @classmethod
    def from_contribution(cls, sender_id: str, contrib: Contribution, meta: ContributionMeta) -> ContributionAnnounce:
        """Create announce from contribution and metadata."""
        contrib_bytes = json.dumps(contrib.to_dict(), sort_keys=True).encode()
        contrib_hash = hashlib.sha256(contrib_bytes).hexdigest()

        return cls(
            sender_id=sender_id,
            goal_id=str(contrib.goal_id),
            contribution_hash=contrib_hash,
            contributor_id=contrib.contributor_id,
            score=meta.score,
            log_index=meta.log_index,
        )


@dataclass
class ContributionRequest(Message):
    """Request full contribution body by hash."""
    type: MessageType = field(default=MessageType.CONTRIBUTION_REQUEST, init=False)

    contribution_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["contribution_hash"] = self.contribution_hash
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> ContributionRequest:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            contribution_hash=data["contribution_hash"],
        )


@dataclass
class ContributionResponse(Message):
    """Response with full contribution body."""
    type: MessageType = field(default=MessageType.CONTRIBUTION_RESPONSE, init=False)

    contribution_hash: str = ""
    contribution: dict[str, Any] | None = None  # Serialized Contribution
    found: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "contribution_hash": self.contribution_hash,
            "contribution": self.contribution,
            "found": self.found,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> ContributionResponse:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            contribution_hash=data["contribution_hash"],
            contribution=data.get("contribution"),
            found=data.get("found", True),
        )


# =============================================================================
# State Sync Messages
# =============================================================================

@dataclass
class StateRequest(Message):
    """Request coordinator state for a goal."""
    type: MessageType = field(default=MessageType.STATE_REQUEST, init=False)

    goal_id: str = ""
    include_log: bool = False
    include_leaderboard: bool = True
    from_log_index: int = 0  # For incremental sync

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "goal_id": self.goal_id,
            "include_log": self.include_log,
            "include_leaderboard": self.include_leaderboard,
            "from_log_index": self.from_log_index,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> StateRequest:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            goal_id=data["goal_id"],
            include_log=data.get("include_log", False),
            include_leaderboard=data.get("include_leaderboard", True),
            from_log_index=data.get("from_log_index", 0),
        )


@dataclass
class StateResponse(Message):
    """Response with coordinator state."""
    type: MessageType = field(default=MessageType.STATE_RESPONSE, init=False)

    goal_id: str = ""
    log_root: str = ""
    log_size: int = 0
    leaderboard_root: str = ""
    leaderboard: list[dict[str, Any]] | None = None
    active_policy_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "goal_id": self.goal_id,
            "log_root": self.log_root,
            "log_size": self.log_size,
            "leaderboard_root": self.leaderboard_root,
            "leaderboard": self.leaderboard,
            "active_policy_hash": self.active_policy_hash,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> StateResponse:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            goal_id=data["goal_id"],
            log_root=data.get("log_root", ""),
            log_size=data.get("log_size", 0),
            leaderboard_root=data.get("leaderboard_root", ""),
            leaderboard=data.get("leaderboard"),
            active_policy_hash=data.get("active_policy_hash"),
        )


# =============================================================================
# Sync Messages
# =============================================================================

@dataclass
class SyncRequest(Message):
    """
    Request contributions for state sync.

    Used to request a batch of contributions by log index range.
    """
    type: MessageType = field(default=MessageType.SYNC_REQUEST, init=False)

    goal_id: str = ""
    from_index: int = 0  # Start log index (inclusive)
    to_index: int = 0    # End log index (exclusive)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "goal_id": self.goal_id,
            "from_index": self.from_index,
            "to_index": self.to_index,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> SyncRequest:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            goal_id=data["goal_id"],
            from_index=data.get("from_index", 0),
            to_index=data.get("to_index", 0),
        )


@dataclass
class SyncResponse(Message):
    """
    Response with contributions for state sync.

    Contains serialized contributions in order.
    """
    type: MessageType = field(default=MessageType.SYNC_RESPONSE, init=False)

    goal_id: str = ""
    from_index: int = 0
    contributions: list[dict[str, Any]] = field(default_factory=list)
    has_more: bool = False  # True if more contributions available

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "goal_id": self.goal_id,
            "from_index": self.from_index,
            "contributions": self.contributions,
            "has_more": self.has_more,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> SyncResponse:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            goal_id=data["goal_id"],
            from_index=data.get("from_index", 0),
            contributions=data.get("contributions", []),
            has_more=data.get("has_more", False),
        )


# =============================================================================
# Discovery Messages
# =============================================================================

@dataclass
class PeerExchange(Message):
    """Exchange peer lists."""
    type: MessageType = field(default=MessageType.PEER_EXCHANGE, init=False)

    peers: list[dict[str, Any]] = field(default_factory=list)  # List of NodeInfo dicts

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["peers"] = self.peers
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> PeerExchange:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            peers=data.get("peers", []),
        )


@dataclass
class Ping(Message):
    """Ping message for liveness check."""
    type: MessageType = field(default=MessageType.PING, init=False)

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> Ping:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
        )


@dataclass
class Pong(Message):
    """Pong response to ping."""
    type: MessageType = field(default=MessageType.PONG, init=False)

    ping_nonce: str = ""  # Nonce of the ping being responded to

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["ping_nonce"] = self.ping_nonce
        return data

    @classmethod
    def _from_dict_impl(cls, data: dict[str, Any]) -> Pong:
        return cls(
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            ping_nonce=data.get("ping_nonce", ""),
        )
