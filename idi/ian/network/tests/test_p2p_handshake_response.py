from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from idi.ian.network.node import NodeIdentity
from idi.ian.network.p2p_manager import P2PConfig, P2PManager, PeerSession, PeerState
from idi.ian.network.protocol import HandshakeResponse


def test_handle_handshake_response_accepts_valid_protocol_message() -> None:
    async def scenario() -> None:
        now_s = 1704067200.0
        now_ms = int(now_s * 1000)

        local = NodeIdentity(private_key=b"\x01" * 32)
        peer = NodeIdentity(private_key=b"\x02" * 32)
        mgr = P2PManager(identity=local, config=P2PConfig())

        # Session represents our local view of the connection (pre-handshake).
        session = PeerSession(
            node_id="pending_127.0.0.1:9001",
            address="127.0.0.1",
            port=9001,
            state=PeerState.CONNECTED,
            connected_at=now_s,
        )

        # Install deterministic pending challenge + local kx keypair (as if we sent HandshakeChallenge).
        session.pending_challenge = b"A" * 32
        our_kx_priv = X25519PrivateKey.from_private_bytes(b"\x11" * 32)
        our_kx_pub = our_kx_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        session.kx_private_key = our_kx_priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        session.kx_public_key = our_kx_pub

        # Peer kx key
        peer_kx_priv = X25519PrivateKey.from_private_bytes(b"\x22" * 32)
        peer_kx_pub = peer_kx_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Construct a valid HandshakeResponse message
        hs = HandshakeResponse(
            sender_id=peer.node_id,
            timestamp=now_ms,
            nonce="hs_nonce",
            challenge_nonce=session.pending_challenge.hex(),
            response_nonce=(b"B" * 32).hex(),
            kx_public_key=base64.b64encode(peer_kx_pub).decode("utf-8"),
            public_key=base64.b64encode(peer.public_key).decode("utf-8"),
        )
        peer.sign_message(hs)

        # Manager needs to know about this session ID to update mappings on success.
        mgr._peers[session.node_id] = session

        with patch("idi.ian.network.p2p_manager.time.time", lambda: now_s):
            ok = await mgr._handle_handshake_response(session, hs.to_dict())

        assert ok is True
        assert session.is_ready()
        assert session.node_id == peer.node_id
        assert session.peer_public_key == peer.public_key
        assert session.session_key is not None and len(session.session_key) == 32
        assert peer.node_id in mgr._peers  # mapping updated

    asyncio.run(scenario())


def test_handle_handshake_response_rejects_bad_signature_and_disconnects() -> None:
    async def scenario() -> None:
        now_s = 1704067200.0
        now_ms = int(now_s * 1000)

        local = NodeIdentity(private_key=b"\x01" * 32)
        peer = NodeIdentity(private_key=b"\x02" * 32)
        mgr = P2PManager(identity=local, config=P2PConfig())

        session = PeerSession(
            node_id="pending_127.0.0.1:9001",
            address="127.0.0.1",
            port=9001,
            state=PeerState.CONNECTED,
            connected_at=now_s,
        )
        session.pending_challenge = b"A" * 32
        our_kx_priv = X25519PrivateKey.from_private_bytes(b"\x11" * 32)
        our_kx_pub = our_kx_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        session.kx_private_key = our_kx_priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        session.kx_public_key = our_kx_pub

        peer_kx_priv = X25519PrivateKey.from_private_bytes(b"\x22" * 32)
        peer_kx_pub = peer_kx_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        hs = HandshakeResponse(
            sender_id=peer.node_id,
            timestamp=now_ms,
            nonce="hs_nonce",
            challenge_nonce=session.pending_challenge.hex(),
            response_nonce=(b"B" * 32).hex(),
            kx_public_key=base64.b64encode(peer_kx_pub).decode("utf-8"),
            public_key=base64.b64encode(peer.public_key).decode("utf-8"),
        )
        peer.sign_message(hs)
        assert hs.signature is not None
        hs.signature = hs.signature[:-1] + bytes([hs.signature[-1] ^ 0xFF])

        mgr._peers[session.node_id] = session

        with patch("idi.ian.network.p2p_manager.time.time", lambda: now_s):
            ok = await mgr._handle_handshake_response(session, hs.to_dict())
            await asyncio.sleep(0)  # allow scheduled disconnect to run

        assert ok is False
        assert session.node_id not in mgr._peers

    asyncio.run(scenario())


