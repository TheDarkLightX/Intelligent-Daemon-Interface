"""End-to-end Tau integration tests (stubbed) for ZK pipeline.

These tests ensure integration hooks are wired and behave as expected with
minimal dependencies. They focus on control flow rather than real proofs.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TAU_TESTNET_PATH = os.path.join(ROOT_DIR, "tau-testnet-ref")

if not os.path.isdir(TAU_TESTNET_PATH):
    pytest.skip(
        "tau-testnet-ref/ not present; skipping Tau integration tests",
        allow_module_level=True,
    )


class StubVerifier:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls = 0

    def verify(self, proof: Any) -> bool:
        self.calls += 1
        return self.result


def _minimal_payload():
    pk = "00" * 48
    return {
        "from": pk,
        "sender_pubkey": pk,
        "sequence_number": 0,
        "expiration_time": 9999999999,
        "operations": {},
        "fee_limit": 0,
        "signature": "00",
        "zk_proof": {"stub": True},
    }


def _load_sendtx(monkeypatch: pytest.MonkeyPatch, verifier: StubVerifier, zk_enabled: bool, zk_required: bool):
    if TAU_TESTNET_PATH not in sys.path:
        sys.path.insert(0, TAU_TESTNET_PATH)

    # Env flags
    monkeypatch.setenv("ZK_ENABLED", "1" if zk_enabled else "0")
    monkeypatch.setenv("ZK_REQUIRE_PROOFS", "1" if zk_required else "0")

    # Stub integration_config
    import idi.taunet_bridge.integration_config as ic

    monkeypatch.setattr(ic, "is_zk_enabled", lambda: zk_enabled)
    monkeypatch.setattr(ic, "is_zk_required", lambda: zk_required)
    monkeypatch.setattr(ic, "get_zk_verifier", lambda: verifier, raising=False)

    # Stub ValidationContext to accept dict proofs
    import idi.taunet_bridge.validation as validation

    monkeypatch.setattr(
        validation.ValidationContext,
        "__post_init__",
        lambda self: setattr(self, "zk_proof", self.payload.get("zk_proof")),
        raising=False,
    )

    # Stubs for db, network, tau_manager
    dummy_db = SimpleNamespace(
        get_string_id=lambda key: f"y{hash(key) & 0xffff}",
        add_mempool_tx=lambda blob: None,
        get_mempool_txs=lambda: [],
    )
    monkeypatch.setitem(sys.modules, "db", dummy_db)

    dummy_bus = SimpleNamespace(get=lambda: None, broadcast_transaction=lambda *args, **kwargs: None)
    dummy_network = SimpleNamespace(bus=dummy_bus)
    monkeypatch.setitem(sys.modules, "network", dummy_network)
    monkeypatch.setitem(sys.modules, "network.bus", dummy_bus)

    dummy_tau_manager = SimpleNamespace(communicate_with_tau=lambda **kwargs: "OK")
    monkeypatch.setitem(sys.modules, "tau_manager", dummy_tau_manager)

    # Reload sendtx
    if "commands.sendtx" in sys.modules:
        del sys.modules["commands.sendtx"]
    if "commands" in sys.modules:
        del sys.modules["commands"]

    import importlib.util

    sendtx_path = os.path.join(TAU_TESTNET_PATH, "commands", "sendtx.py")
    spec = importlib.util.spec_from_file_location("commands.sendtx", sendtx_path)
    assert spec and spec.loader
    sendtx = importlib.util.module_from_spec(spec)
    sys.modules["commands.sendtx"] = sendtx
    spec.loader.exec_module(sendtx)
    return sendtx


def test_e2e_tx_with_valid_proof(monkeypatch: pytest.MonkeyPatch):
    verifier = StubVerifier(result=True)
    sendtx = _load_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=True)
    payload = _minimal_payload()
    res = sendtx.queue_transaction(json.dumps(payload), propagate=False)
    assert "SUCCESS" in res
    assert verifier.calls == 1


def test_e2e_tx_rejected_on_invalid_proof(monkeypatch: pytest.MonkeyPatch):
    verifier = StubVerifier(result=False)
    sendtx = _load_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=True)
    payload = _minimal_payload()
    res = sendtx.queue_transaction(json.dumps(payload), propagate=False)
    assert "FAILURE" in res
    assert verifier.calls == 1


def test_e2e_tx_optional_without_proof(monkeypatch: pytest.MonkeyPatch):
    verifier = StubVerifier(result=True)
    sendtx = _load_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=False)
    payload = _minimal_payload()
    payload.pop("zk_proof", None)
    res = sendtx.queue_transaction(json.dumps(payload), propagate=False)
    assert "SUCCESS" in res


def test_e2e_tx_zk_disabled(monkeypatch: pytest.MonkeyPatch):
    verifier = StubVerifier(result=False)
    sendtx = _load_sendtx(monkeypatch, verifier, zk_enabled=False, zk_required=False)
    payload = _minimal_payload()
    res = sendtx.queue_transaction(json.dumps(payload), propagate=False)
    assert "SUCCESS" in res
    assert verifier.calls == 0

