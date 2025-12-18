from __future__ import annotations

import asyncio
import time

import pytest

from idi.ian.network.node import NodeIdentity
from idi.ian.network.p2p_manager import P2PConfig, P2PManager, PeerSession, PeerState


def test_p2p_handshake_timeout_disconnects_stalled_session():
    async def scenario() -> None:
        identity = NodeIdentity.generate()
        mgr = P2PManager(
            identity=identity,
            config=P2PConfig(
                handshake_timeout=0.1,
                max_peers=10,
                max_pending_connections=1,
                max_connections_per_ip=10,
            ),
        )

        session = PeerSession(
            node_id="inbound_127.0.0.1:1234",
            address="127.0.0.1",
            port=1234,
            state=PeerState.HANDSHAKING,
            connected_at=time.time(),
            handshake_started_at=time.time() - 10.0,
        )

        async with mgr._lock:
            mgr._peers[session.node_id] = session
            mgr._address_map[f"{session.address}:{session.port}"] = session.node_id

        mgr._running = True
        cleanup_task = asyncio.create_task(mgr._cleanup_loop())

        try:
            await asyncio.sleep(0.35)
            async with mgr._lock:
                assert session.node_id not in mgr._peers
        finally:
            mgr._running = False
            cleanup_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cleanup_task

    asyncio.run(scenario())
