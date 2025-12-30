from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from idi.ian.network.node import NodeIdentity
from idi.ian.network.p2p_manager import P2PConfig, P2PManager, PeerSession, PeerState
from idi.ian.network.protocol import MessageType, Ping


def test_unsigned_non_handshake_messages_are_dropped_pre_handshake() -> None:
    async def scenario() -> None:
        now_s = 1704067200.0
        mgr = P2PManager(identity=NodeIdentity(private_key=b"\x01" * 32), config=P2PConfig())

        # Pre-handshake session (not READY)
        session = PeerSession(
            node_id="inbound_127.0.0.1:1234",
            address="127.0.0.1",
            port=1234,
            state=PeerState.CONNECTED,
        )

        # Unsigned ping-like dict (non-handshake)
        msg_dict = {
            "type": MessageType.PING.value,
            "sender_id": "0" * 40,
            "timestamp": int(now_s * 1000),
            "nonce": "n",
            "signature": None,
        }

        with patch("idi.ian.network.p2p_manager.time.time", lambda: now_s):
            await mgr._handle_message(session, json.dumps(msg_dict).encode("utf-8"))

        # Fail-closed: do not populate replay cache for unauthenticated pre-handshake traffic.
        assert mgr._peer_nonce_cache == {}

    asyncio.run(scenario())


def test_signed_messages_required_and_verified_for_ready_sessions() -> None:
    async def scenario() -> None:
        now_s = 1704067200.0
        local = NodeIdentity(private_key=b"\x01" * 32)
        peer = NodeIdentity(private_key=b"\x02" * 32)
        mgr = P2PManager(identity=local, config=P2PConfig())

        handled: list[str] = []

        async def handle_ping(msg: dict, from_id: str):
            handled.append(from_id)
            return None

        mgr.register_handler(MessageType.PING, handle_ping)

        session = PeerSession(
            node_id=peer.node_id,
            address="127.0.0.1",
            port=9001,
            state=PeerState.READY,
            verified=True,
            handshake_completed=True,
        )
        session.peer_public_key = peer.public_key
        mgr._peers[session.node_id] = session

        # Valid signed ping from peer
        ping = Ping(sender_id=peer.node_id, timestamp=int(now_s * 1000), nonce="nonce_ok")
        peer.sign_message(ping)

        with patch("idi.ian.network.p2p_manager.time.time", lambda: now_s):
            await mgr._handle_message(session, json.dumps(ping.to_dict()).encode("utf-8"))

        assert handled == [peer.node_id]

        # Now corrupt signature; must disconnect peer.
        bad = Ping(sender_id=peer.node_id, timestamp=int(now_s * 1000) + 1, nonce="nonce_bad")
        peer.sign_message(bad)
        assert bad.signature is not None
        bad.signature = bad.signature[:-1] + bytes([bad.signature[-1] ^ 0xFF])

        with patch("idi.ian.network.p2p_manager.time.time", lambda: now_s):
            await mgr._handle_message(session, json.dumps(bad.to_dict()).encode("utf-8"))
            await asyncio.sleep(0)  # allow scheduled disconnect to run

        assert peer.node_id not in mgr._peers

    asyncio.run(scenario())


