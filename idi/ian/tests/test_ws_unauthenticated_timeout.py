from __future__ import annotations

import asyncio
import socket
from contextlib import closing

import pytest

try:
    import aiohttp
    from aiohttp import WSMsgType

    from idi.ian.network.node import NodeIdentity
    from idi.ian.network.websocket_transport import HAS_AIOHTTP, WebSocketConfig, WebSocketServer
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"WebSocket timeout test dependencies unavailable: {exc}", allow_module_level=True)


if not HAS_AIOHTTP:  # pragma: no cover
    pytest.skip("aiohttp not available", allow_module_level=True)


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ws_unauthenticated_timeout_disconnects_client():
    async def scenario() -> None:
        port = _find_free_port()
        server_identity = NodeIdentity.generate()
        server = WebSocketServer(
            identity=server_identity,
            config=WebSocketConfig(host="127.0.0.1", port=port, unauthenticated_timeout=0.1),
        )

        started = await server.start()
        assert started

        try:
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect(f"ws://127.0.0.1:{port}/ws")

                await ws.receive(timeout=1.0)

                await asyncio.sleep(0.35)

                msg = await ws.receive(timeout=1.0)
                assert msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}
        finally:
            await server.stop()

    asyncio.run(scenario())
