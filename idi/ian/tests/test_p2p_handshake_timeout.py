from __future__ import annotations
 
import asyncio
from unittest.mock import patch
 
import pytest
 
from idi.ian.network.node import NodeIdentity
from idi.ian.network.p2p_manager import P2PConfig, P2PManager, PeerSession, PeerState

 
def test_p2p_handshake_timeout_disconnects_stalled_session():
    async def scenario() -> None:
        now_s = 1704067200.0
        real_sleep = asyncio.sleep
        identity = NodeIdentity(private_key=b"\x01" * 32)
        mgr = P2PManager(
            identity=identity,
            config=P2PConfig(
                handshake_timeout=0.1,
                max_peers=10,
                max_pending_connections=1,
                max_connections_per_ip=10,
            ),
        )
        setattr(mgr, "_now_s", lambda: now_s)
 
        session = PeerSession(
            node_id="inbound_127.0.0.1:1234",
            address="127.0.0.1",
            port=1234,
            state=PeerState.HANDSHAKING,
            connected_at=now_s,
            handshake_started_at=now_s - 10.0,
        )

        async with mgr._lock:
            mgr._peers[session.node_id] = session
            mgr._address_map[f"{session.address}:{session.port}"] = session.node_id
 
        mgr._running = True
 
        async def fast_sleep(_: float) -> None:
            await real_sleep(0)
 
        with patch("idi.ian.network.p2p_manager.asyncio.sleep", fast_sleep):
            cleanup_task = asyncio.create_task(mgr._cleanup_loop())
 
            try:
                removed = False
                for _ in range(8):
                    await real_sleep(0)
                    async with mgr._lock:
                        if session.node_id not in mgr._peers:
                            removed = True
                            break
                assert removed
            finally:
                mgr._running = False
                cleanup_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cleanup_task
 
    asyncio.run(scenario())
