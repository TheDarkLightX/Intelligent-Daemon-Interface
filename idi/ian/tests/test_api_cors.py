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
        GoalSpec,
        GoalID,
        EvaluationLimits,
        Thresholds,
    )

    goal_spec = GoalSpec(
        goal_id=GoalID("CORS_TEST"),
        name="CORS Test Goal",
        description="CORS tests",
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
async def test_api_config_default_cors_is_deny_by_default():
    config = ApiConfig()
    assert config.cors_origins == []


@pytest.mark.asyncio
async def test_cors_disabled_emits_no_headers():
    runner, base = await _start_app(ApiConfig(cors_origins=[]))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/health", headers={"Origin": "https://evil.example"}) as resp:
                assert resp.status == 200
                assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cors_allow_all_emits_wildcard():
    runner, base = await _start_app(ApiConfig(cors_origins=["*"]))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/health", headers={"Origin": "https://a.example"}) as resp:
                assert resp.status == 200
                assert resp.headers.get("Access-Control-Allow-Origin") == "*"
                assert resp.headers.get("Access-Control-Allow-Methods") == "GET, POST, OPTIONS"
                assert "X-API-Key" in (resp.headers.get("Access-Control-Allow-Headers") or "")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cors_allowlist_echoes_origin_and_sets_vary():
    runner, base = await _start_app(ApiConfig(cors_origins=["https://allowed.example"]))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base}/health",
                headers={"Origin": "https://allowed.example"},
            ) as resp:
                assert resp.status == 200
                assert resp.headers.get("Access-Control-Allow-Origin") == "https://allowed.example"
                assert "Origin" in (resp.headers.get("Vary") or "")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cors_disallowed_origin_emits_no_acao():
    runner, base = await _start_app(ApiConfig(cors_origins=["https://allowed.example"]))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base}/health",
                headers={"Origin": "https://blocked.example"},
            ) as resp:
                assert resp.status == 200
                assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cors_preflight_options_allowed_origin_returns_204_with_headers():
    runner, base = await _start_app(ApiConfig(cors_origins=["https://allowed.example"]))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.options(
                f"{base}/health",
                headers={
                    "Origin": "https://allowed.example",
                    "Access-Control-Request-Method": "GET",
                },
            ) as resp:
                assert resp.status == 204
                assert resp.headers.get("Access-Control-Allow-Origin") == "https://allowed.example"
    finally:
        await runner.cleanup()
