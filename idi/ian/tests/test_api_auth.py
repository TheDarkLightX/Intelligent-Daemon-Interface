from __future__ import annotations

import socket
from contextlib import closing

import pytest

try:
    import aiohttp
    from aiohttp import web
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"aiohttp not available: {exc}", allow_module_level=True)

from idi.ian.network.api import ApiConfig, IANApiHandlers, create_api_app


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_coordinator():
    from idi.ian import (
        IANCoordinator,
        CoordinatorConfig,
        EvaluationLimits,
        GoalID,
        GoalSpec,
        Thresholds,
    )

    goal_spec = GoalSpec(
        goal_id=GoalID("AUTH_TEST"),
        name="Auth Test Goal",
        description="Auth tests",
        eval_limits=EvaluationLimits(),
        thresholds=Thresholds(),
    )

    return IANCoordinator(goal_spec=goal_spec, config=CoordinatorConfig())


async def _start_app(config: ApiConfig):
    coordinator = _build_coordinator()
    handlers = IANApiHandlers(coordinator, config)
    app = create_api_app(handlers, config)

    runner = web.AppRunner(app)
    await runner.setup()

    port = _find_free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_contribute_requires_api_key_when_configured():
    config = ApiConfig(
        api_key="secret123",
        api_key_required=True,
        rate_limit_per_ip=1000,
    )
    runner, base = await _start_app(config)

    body = {
        "goal_id": "AUTH_TEST",
        "agent_pack": {
            "version": "1.0",
            "parameters": "dGVzdA==",
        },
        "contributor_id": "alice",
        "seed": 1,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base}/api/v1/contribute", json=body) as resp:
                assert resp.status == 401
                payload = await resp.json()
                assert payload["success"] is False
                assert "API key" in (payload.get("error") or "")

            async with session.post(
                f"{base}/api/v1/contribute",
                json=body,
                headers={"X-API-Key": "wrong"},
            ) as resp:
                assert resp.status == 401

            async with session.post(
                f"{base}/api/v1/contribute",
                json=body,
                headers={"X-API-Key": "secret123"},
            ) as resp:
                assert resp.status == 200
                payload = await resp.json()
                assert payload["success"] is True
                assert payload["data"]["accepted"] is True
    finally:
        await runner.cleanup()
