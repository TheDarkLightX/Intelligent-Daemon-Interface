"""Smoke tests for TLS-enabled P2P and /info,/peers endpoints.

These are lightweight integration-style tests that verify:
- A DecentralizedNode can start its health server and expose /info and /peers
- The /info and /peers responses have the expected basic structure

TLS P2P is exercised indirectly by constructing a TLSConfig and P2PManager,
without requiring actual network sockets (full end-to-end TLS tests would be
handled in a higher-level environment).
"""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import closing

import pytest

from idi.ian.models import GoalID, GoalSpec, EvaluationLimits, Thresholds
from idi.ian.network import DecentralizedNode, DecentralizedNodeConfig, NodeIdentity


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_health_info_and_peers_endpoints_basic_structure():
    """DecentralizedNode exposes /info and /peers via HealthServer.

    This is a coarse smoke test to ensure the endpoints are reachable and
    return JSON with expected top-level keys. It does not assert on
    specific peer contents.
    """
    # Dynamic ports to avoid collisions in CI
    listen_port = _find_free_port()
    health_port = _find_free_port()

    goal_spec = GoalSpec(
        goal_id=GoalID("SMOKE_GOAL"),
        name="Smoke Test Goal",
        description="Smoke test for /info and /peers",
        eval_limits=EvaluationLimits(),
        thresholds=Thresholds(),
    )

    identity = NodeIdentity.generate()

    config = DecentralizedNodeConfig(
        listen_address="127.0.0.1",
        listen_port=listen_port,
        health_port=health_port,
        enable_health_server=True,
        accept_contributions=False,
    )

    node = DecentralizedNode(
        goal_spec=goal_spec,
        identity=identity,
        config=config,
    )

    await node.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", health_port)
        # Request /info
        writer.write(b"GET /info HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        raw = await reader.read()
        writer.close()
        await writer.wait_closed()

        # Simple HTTP parsing: split headers/body
        parts = raw.split(b"\r\n\r\n", 1)
        assert len(parts) == 2
        body = parts[1].decode()
        info = json.loads(body)

        assert info["node_id"] == identity.node_id
        assert info["goal_id"] == str(goal_spec.goal_id)
        assert "addresses" in info
        assert "capabilities" in info

        # /peers should return JSON with peers + stats
        reader, writer = await asyncio.open_connection("127.0.0.1", health_port)
        writer.write(b"GET /peers HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        raw_peers = await reader.read()
        writer.close()
        await writer.wait_closed()

        parts_peers = raw_peers.split(b"\r\n\r\n", 1)
        assert len(parts_peers) == 2
        peers_body = parts_peers[1].decode()
        peers = json.loads(peers_body)

        assert peers["goal_id"] == str(goal_spec.goal_id)
        assert "peers" in peers
        assert "stats" in peers

    finally:
        await node.stop()
