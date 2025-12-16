from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.1)

    raise AssertionError(f"/health not reachable within {timeout_seconds}s: {last_error}")


def test_cli_node_start_exposes_health_and_exits_cleanly(tmp_path: Path) -> None:
    api_port = _find_free_port()
    p2p_port = _find_free_port()
    ws_port = _find_free_port()

    key_dir = tmp_path / "keys"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node:",
                "  key_dir: ./keys",
                "network:",
                "  api:",
                "    host: 127.0.0.1",
                "    port: 8000",
                "    cors_origins: []",
                "  websocket:",
                "    enabled: false",
                "  p2p:",
                "    host: 127.0.0.1",
                "    port: 9000",
                "security:",
                "  api_key_required: false",
            ]
        )
        + "\n"
    )

    env = os.environ.copy()
    env.update(
        {
            "IAN_LISTEN_HOST": "127.0.0.1",
            "IAN_API_HOST": "127.0.0.1",
            "IAN_API_PORT": str(api_port),
            "IAN_P2P_PORT": str(p2p_port),
            "IAN_WS_PORT": str(ws_port),
            "IAN_KEY_DIR": str(key_dir),
            "IAN_GOAL_ID": "CLI_SMOKE_GOAL",
            "IAN_API_KEY_REQUIRED": "0",
        }
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "idi.ian.cli",
            "node",
            "start",
            "--config",
            str(config_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_health(f"http://127.0.0.1:{api_port}/health", timeout_seconds=15.0)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10.0)

        output = ""
        if proc.stdout is not None:
            try:
                output = proc.stdout.read()
            except Exception:
                output = ""

        assert proc.returncode == 0, output
