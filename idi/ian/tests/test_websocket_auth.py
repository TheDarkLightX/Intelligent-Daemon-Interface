from __future__ import annotations

import asyncio
import base64
import socket
from contextlib import closing

import pytest

try:
    from idi.ian.network.node import NodeIdentity
    from idi.ian.network.websocket_transport import (
        HAS_AIOHTTP,
        WebSocketClient,
        WebSocketConfig,
        WebSocketServer,
        _auth_payload,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"WebSocket auth test dependencies unavailable: {exc}", allow_module_level=True)


if not HAS_AIOHTTP:  # pragma: no cover
    pytest.skip("aiohttp not available", allow_module_level=True)


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _client_config() -> WebSocketConfig:
    return WebSocketConfig(auto_reconnect=False)


async def _run_with_server(test_coro):
    port = _find_free_port()
    server_identity = NodeIdentity.generate()
    server = WebSocketServer(
        identity=server_identity,
        config=WebSocketConfig(host="127.0.0.1", port=port),
    )

    started = await server.start()
    assert started

    try:
        await test_coro(f"ws://127.0.0.1:{port}/ws")
    finally:
        await server.stop()


def test_ws_subscribe_requires_auth():
    async def scenario(url: str) -> None:
        client = WebSocketClient(url=url, config=_client_config())
        await client.connect()

        try:
            response = await client.request(
                {"type": "subscribe", "topics": ["t1"]},
                timeout=2.0,
            )
            assert response is not None
            assert response.get("type") == "error"
            assert response.get("error") == "Not authenticated"
        finally:
            await client.disconnect()

    asyncio.run(_run_with_server(scenario))


def test_ws_authenticate_missing_fields_rejected():
    async def scenario(url: str) -> None:
        client = WebSocketClient(url=url, config=_client_config())
        await client.connect()

        try:
            response = await client.request({"type": "authenticate"}, timeout=2.0)
            assert response is not None
            assert response.get("type") == "error"
            assert response.get("error") == "Invalid authentication"
        finally:
            await client.disconnect()

    asyncio.run(_run_with_server(scenario))


def test_ws_authenticate_invalid_signature_rejected():
    async def scenario(url: str) -> None:
        signing_identity = NodeIdentity.generate()
        client_identity = NodeIdentity(public_key=signing_identity.public_key)

        client = WebSocketClient(url=url, identity=client_identity, config=_client_config())
        await client.connect()

        try:
            assert client._server_challenge is not None
            payload = _auth_payload(client._server_challenge, signing_identity.node_id)

            signature = bytearray(signing_identity.sign(payload))
            signature[0] ^= 1

            response = await client.request(
                {
                    "type": "authenticate",
                    "node_id": signing_identity.node_id,
                    "public_key": base64.b64encode(signing_identity.public_key).decode(),
                    "signature": base64.b64encode(bytes(signature)).decode(),
                },
                timeout=2.0,
            )

            assert response is not None
            assert response.get("type") == "error"
            assert response.get("error") == "Invalid authentication"
        finally:
            await client.disconnect()

    asyncio.run(_run_with_server(scenario))


def test_ws_authenticate_node_id_mismatch_rejected():
    async def scenario(url: str) -> None:
        signing_identity = NodeIdentity.generate()
        client_identity = NodeIdentity(public_key=signing_identity.public_key)
        wrong_node_id = NodeIdentity.generate().node_id

        client = WebSocketClient(url=url, identity=client_identity, config=_client_config())
        await client.connect()

        try:
            assert client._server_challenge is not None
            payload = _auth_payload(client._server_challenge, wrong_node_id)
            signature = signing_identity.sign(payload)

            response = await client.request(
                {
                    "type": "authenticate",
                    "node_id": wrong_node_id,
                    "public_key": base64.b64encode(signing_identity.public_key).decode(),
                    "signature": base64.b64encode(signature).decode(),
                },
                timeout=2.0,
            )

            assert response is not None
            assert response.get("type") == "error"
            assert response.get("error") == "node_id does not match public key"
        finally:
            await client.disconnect()

    asyncio.run(_run_with_server(scenario))


def test_ws_authenticate_success_enables_subscription():
    async def scenario(url: str) -> None:
        signing_identity = NodeIdentity.generate()
        client_identity = NodeIdentity(public_key=signing_identity.public_key)

        client = WebSocketClient(url=url, identity=client_identity, config=_client_config())
        await client.connect()

        try:
            assert client._server_challenge is not None
            payload = _auth_payload(client._server_challenge, signing_identity.node_id)
            signature = signing_identity.sign(payload)

            auth_response = await client.request(
                {
                    "type": "authenticate",
                    "node_id": signing_identity.node_id,
                    "public_key": base64.b64encode(signing_identity.public_key).decode(),
                    "signature": base64.b64encode(signature).decode(),
                },
                timeout=2.0,
            )

            assert auth_response is not None
            assert auth_response.get("type") == "authenticated"

            subscribe_response = await client.request(
                {"type": "subscribe", "topics": ["t1"]},
                timeout=2.0,
            )

            assert subscribe_response is not None
            assert subscribe_response.get("type") == "subscribed"
            assert "t1" in subscribe_response.get("topics", [])
        finally:
            await client.disconnect()

    asyncio.run(_run_with_server(scenario))


def test_ws_duplicate_node_id_rejected():
    async def scenario(url: str) -> None:
        signing_identity = NodeIdentity.generate()

        client_identity_1 = NodeIdentity(public_key=signing_identity.public_key)
        client1 = WebSocketClient(url=url, identity=client_identity_1, config=_client_config())
        await client1.connect()

        client_identity_2 = NodeIdentity(public_key=signing_identity.public_key)
        client2 = WebSocketClient(url=url, identity=client_identity_2, config=_client_config())
        await client2.connect()

        try:
            assert client1._server_challenge is not None
            payload_1 = _auth_payload(client1._server_challenge, signing_identity.node_id)
            signature_1 = signing_identity.sign(payload_1)

            auth1 = await client1.request(
                {
                    "type": "authenticate",
                    "node_id": signing_identity.node_id,
                    "public_key": base64.b64encode(signing_identity.public_key).decode(),
                    "signature": base64.b64encode(signature_1).decode(),
                },
                timeout=2.0,
            )

            assert auth1 is not None
            assert auth1.get("type") == "authenticated"

            assert client2._server_challenge is not None
            payload_2 = _auth_payload(client2._server_challenge, signing_identity.node_id)
            signature_2 = signing_identity.sign(payload_2)

            auth2 = await client2.request(
                {
                    "type": "authenticate",
                    "node_id": signing_identity.node_id,
                    "public_key": base64.b64encode(signing_identity.public_key).decode(),
                    "signature": base64.b64encode(signature_2).decode(),
                },
                timeout=2.0,
            )

            assert auth2 is not None
            assert auth2.get("type") == "error"
            assert auth2.get("error") == "node_id already connected"
        finally:
            await client2.disconnect()
            await client1.disconnect()

    asyncio.run(_run_with_server(scenario))
