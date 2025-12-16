from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .simulations.trading_agent_demo import DemoConfig, run_demo


def _build_demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IAN CLI - Trading Agent Competition Demo (Simulated Evaluation)",
    )
    parser.add_argument(
        "--contributors",
        type=int,
        default=5,
        help="Number of contributors",
    )
    parser.add_argument(
        "--contributions",
        type=int,
        default=3,
        help="Contributions per contributor",
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help="Disable security hardening",
    )
    parser.add_argument(
        "--no-tau",
        action="store_true",
        help="Disable Tau integration",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    return parser


def _run_demo(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_demo_parser()
    args = parser.parse_args(argv)

    config = DemoConfig(
        num_contributors=args.contributors,
        contributions_per_contributor=args.contributions,
        enable_security=not args.no_security,
        enable_tau=not args.no_tau,
        verbose=not args.quiet,
    )

    run_demo(config)


def _get_nested(config: Dict[str, Any], path: Sequence[str], default: Any) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return default


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load .yaml config files") from exc

    parsed = yaml.safe_load(path.read_text())
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("YAML config must be a mapping at top level")


def _default_config_path() -> Path:
    env_path = os.environ.get("IAN_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "deploy" / "config" / "development.yaml"


def _build_node_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IAN Node CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start an IAN node")
    start.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to IAN_CONFIG_PATH or deploy/config/development.yaml)",
    )
    return parser


async def _run_node_start(config_path: Path) -> None:
    from idi.ian import CoordinatorConfig, EvaluationLimits, GoalID, GoalSpec, IANCoordinator, Thresholds
    from idi.ian.network.api import ApiConfig, IANApiServer
    from idi.ian.network.node import NodeIdentity
    from idi.ian.network.p2p_manager import P2PConfig, P2PManager
    from idi.ian.network.websocket_transport import WebSocketConfig, WebSocketServer

    deploy_cfg = _load_yaml_config(config_path)

    node_cfg = _get_nested(deploy_cfg, ("node",), {})
    network_cfg = _get_nested(deploy_cfg, ("network",), {})
    security_cfg = _get_nested(deploy_cfg, ("security",), {})

    api_cfg = _get_nested(network_cfg, ("api",), {})
    ws_cfg = _get_nested(network_cfg, ("websocket",), {})
    p2p_cfg = _get_nested(network_cfg, ("p2p",), {})
    discovery_cfg = _get_nested(network_cfg, ("discovery",), {})

    listen_host = os.environ.get("IAN_LISTEN_HOST")

    api_host = os.environ.get("IAN_API_HOST") or listen_host or str(api_cfg.get("host") or "0.0.0.0")
    api_port = _parse_int(os.environ.get("IAN_API_PORT"), _parse_int(api_cfg.get("port"), 8000))
    cors_origins = _split_csv(os.environ.get("IAN_API_CORS_ORIGINS"))
    if not cors_origins:
        cors_origins_raw = api_cfg.get("cors_origins")
        if isinstance(cors_origins_raw, list):
            cors_origins = [str(item) for item in cors_origins_raw if str(item)]

    rate_limit_cfg = api_cfg.get("rate_limit") if isinstance(api_cfg, dict) else None
    requests_per_minute = 100
    if isinstance(rate_limit_cfg, dict):
        requests_per_minute = _parse_int(rate_limit_cfg.get("requests_per_minute"), requests_per_minute)

    api_key = os.environ.get("IAN_API_KEY")
    api_key_required = _parse_bool(os.environ.get("IAN_API_KEY_REQUIRED"), _parse_bool(security_cfg.get("api_key_required"), False))
    if api_key_required and not api_key:
        raise ValueError("API key required but IAN_API_KEY is not set")

    goal_id_raw = os.environ.get("IAN_GOAL_ID") or "DEMO_TRADING_AGENT"
    goal_spec = GoalSpec(
        goal_id=GoalID(goal_id_raw),
        name=str(goal_id_raw),
        description="",
        eval_limits=EvaluationLimits(),
        thresholds=Thresholds(),
    )

    coordinator = IANCoordinator(goal_spec=goal_spec, config=CoordinatorConfig())

    key_dir = Path(os.environ.get("IAN_KEY_DIR") or str(node_cfg.get("key_dir") or "./keys"))
    identity_path = key_dir / "node_identity.json"
    if identity_path.exists():
        identity = NodeIdentity.load(identity_path)
    else:
        identity = NodeIdentity.generate()
        identity.save(identity_path)

    api_server = IANApiServer(
        coordinator,
        config=ApiConfig(
            host=api_host,
            port=api_port,
            api_key=api_key,
            api_key_required=api_key_required,
            rate_limit_per_ip=requests_per_minute,
            cors_origins=cors_origins,
        ),
    )

    ws_enabled = _parse_bool(ws_cfg.get("enabled"), True)
    ws_host = listen_host or str(ws_cfg.get("host") or "0.0.0.0")
    ws_port = _parse_int(os.environ.get("IAN_WS_PORT"), _parse_int(ws_cfg.get("port"), 9001))
    ws_max_connections = _parse_int(ws_cfg.get("max_connections"), 1000)
    ws_server: Optional[WebSocketServer] = None
    if ws_enabled:
        ws_server = WebSocketServer(
            identity=identity,
            config=WebSocketConfig(host=ws_host, port=ws_port, max_connections=ws_max_connections),
        )

    p2p_host = listen_host or str(p2p_cfg.get("host") or "0.0.0.0")
    p2p_port = _parse_int(os.environ.get("IAN_P2P_PORT"), _parse_int(p2p_cfg.get("port"), 9000))
    p2p_max_peers = _parse_int(p2p_cfg.get("max_peers"), 50)
    p2p_timeout = float(p2p_cfg.get("connection_timeout") or 10.0)

    p2p_manager = P2PManager(
        identity=identity,
        config=P2PConfig(
            listen_host=p2p_host,
            listen_port=p2p_port,
            max_peers=p2p_max_peers,
            connection_timeout=p2p_timeout,
        ),
    )

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    await api_server.start()
    await p2p_manager.start()
    if ws_server is not None:
        await ws_server.start()

    seeds_raw = os.environ.get("IAN_SEED_NODES")
    if seeds_raw:
        seeds = _split_csv(seeds_raw)
        for seed in seeds[:32]:
            if ":" not in seed:
                continue
            host, port_str = seed.rsplit(":", 1)
            try:
                port_val = int(port_str)
            except ValueError:
                continue
            await p2p_manager.connect_to_peer(host, port_val)

    discovery_seeds = discovery_cfg.get("seed_nodes")
    if isinstance(discovery_seeds, list) and not seeds_raw:
        for seed in [str(item) for item in discovery_seeds][:32]:
            if ":" not in seed:
                continue
            host, port_str = seed.rsplit(":", 1)
            try:
                port_val = int(port_str)
            except ValueError:
                continue
            await p2p_manager.connect_to_peer(host, port_val)

    await stop_event.wait()

    if ws_server is not None:
        await ws_server.stop()
    await p2p_manager.stop()
    await api_server.stop()


def _run_node(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_node_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        config_path = Path(args.config) if args.config else _default_config_path()
        asyncio.run(_run_node_start(config_path))
        return
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if effective_argv and effective_argv[0] == "node":
        _run_node(effective_argv[1:])
        return
    _run_demo(effective_argv)


if __name__ == "__main__":
    main()

