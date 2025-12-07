"""Integration tests for Tau Testnet sendtx with ZK validation (TDD-first).

These tests assert the desired behavior once the ZK hook is added to
`tau-testnet-ref/commands/sendtx.py`. They use monkeypatching to avoid
network/crypto dependencies and to keep tests fast and deterministic.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import json


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TAU_TESTNET_PATH = os.path.join(ROOT_DIR, "tau-testnet-ref")


def _minimal_payload() -> Dict[str, Any]:
    """Create a minimal valid transaction payload."""
    dummy_pk = "00" * 48  # 96 hex chars
    return {
        "from": dummy_pk,
        "sender_pubkey": dummy_pk,
        "sequence_number": 0,
        "expiration_time": int(time.time()) + 3600,
        "operations": {},  # no transfers, no rules
        "fee_limit": 0,
        "signature": "00",
        "zk_proof": {"stub": True},  # placeholder proof payload
    }


class StubVerifier:
    """Stub ZK verifier with configurable result."""

    def __init__(self, result: bool = True):
        self.result = result
        self.calls = 0

    def verify(self, proof: Any) -> bool:
        self.calls += 1
        return self.result


def _import_sendtx(monkeypatch: pytest.MonkeyPatch, verifier: StubVerifier, zk_enabled: bool, zk_required: bool):
    """Import/reload sendtx with controlled environment and stubs."""
    # Ensure tau-testnet is importable
    if TAU_TESTNET_PATH not in sys.path:
        sys.path.insert(0, TAU_TESTNET_PATH)

    # Set env flags before import
    monkeypatch.setenv("ZK_ENABLED", "1" if zk_enabled else "0")
    monkeypatch.setenv("ZK_REQUIRE_PROOFS", "1" if zk_required else "0")

    # Stub out config mapping layer
    monkeypatch.setenv("ZK_PROOF_SYSTEM", "stub")

    # Patch bridge integration helpers
    import idi.taunet_bridge.integration_config as ic  # will be created in implementation
    import idi.taunet_bridge.validation as validation

    monkeypatch.setattr(ic, "is_zk_enabled", lambda: zk_enabled)
    monkeypatch.setattr(ic, "is_zk_required", lambda: zk_required)
    monkeypatch.setattr(ic, "get_zk_verifier", lambda: verifier, raising=False)

    # Simplify ValidationContext to accept dict proof directly
    monkeypatch.setattr(
        validation.ValidationContext,
        "__post_init__",
        lambda self: setattr(self, "zk_proof", self.payload.get("zk_proof")),
        raising=False,
    )

    # Stub chain_state to avoid real state mutations
    dummy_chain_state = SimpleNamespace(
        get_sequence_number=lambda _pk: 0,
        increment_sequence_number=lambda _pk: None,
        update_balances_after_transfer=lambda _from, _to, _amt: True,
    )
    monkeypatch.setitem(sys.modules, "chain_state", dummy_chain_state)

    # Stub db module functions used by sendtx
    dummy_db = SimpleNamespace(
        get_string_id=lambda key: f"y{hash(key) & 0xffff}",
        add_mempool_tx=lambda blob: None,
    )
    monkeypatch.setitem(sys.modules, "db", dummy_db)

    # Stub network module to avoid trio dependency
    dummy_bus = SimpleNamespace(get=lambda: None, broadcast_transaction=lambda *args, **kwargs: None)
    dummy_network = SimpleNamespace(bus=dummy_bus)
    monkeypatch.setitem(sys.modules, "network", dummy_network)
    monkeypatch.setitem(sys.modules, "network.bus", dummy_bus)

    # Stub tau_manager to avoid external calls
    dummy_tau_manager = SimpleNamespace(communicate_with_tau=lambda **kwargs: "OK")
    monkeypatch.setitem(sys.modules, "tau_manager", dummy_tau_manager)

    # Stub tau_defs (unused in these tests)
    monkeypatch.setitem(sys.modules, "tau_defs", SimpleNamespace())

    # Bypass BLS path
    monkeypatch.setenv("TAU_FORCE_TEST", "1")

    # Reload sendtx to pick up patches and env
    if "commands.sendtx" in sys.modules:
        del sys.modules["commands.sendtx"]
    if "commands" in sys.modules:
        del sys.modules["commands"]

    try:
        import commands.sendtx as sendtx
    except ModuleNotFoundError:
        # Fallback: load module directly from path
        import importlib.util
        sendtx_path = os.path.join(TAU_TESTNET_PATH, "commands", "sendtx.py")
        spec = importlib.util.spec_from_file_location("commands.sendtx", sendtx_path)
        assert spec and spec.loader
        sendtx = importlib.util.module_from_spec(spec)
        sys.modules["commands.sendtx"] = sendtx
        spec.loader.exec_module(sendtx)

    return sendtx


def test_sendtx_accepts_valid_proof(monkeypatch: pytest.MonkeyPatch):
    """When ZK is enabled and required, a valid proof should allow the tx."""
    verifier = StubVerifier(result=True)
    sendtx = _import_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=True)
    payload = _minimal_payload()
    payload_json = json.dumps(payload)

    result = sendtx.queue_transaction(payload_json, propagate=False)

    assert "SUCCESS" in result
    assert verifier.calls == 1


def test_sendtx_rejects_invalid_proof(monkeypatch: pytest.MonkeyPatch):
    """When ZK is required and proof is invalid, tx should be rejected."""
    verifier = StubVerifier(result=False)
    sendtx = _import_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=True)
    payload_json = json.dumps(_minimal_payload())

    result = sendtx.queue_transaction(payload_json, propagate=False)

    assert "FAILURE" in result
    assert verifier.calls == 1


def test_sendtx_optional_allows_no_proof(monkeypatch: pytest.MonkeyPatch):
    """When ZK is optional, transactions without proofs still succeed."""
    verifier = StubVerifier(result=True)
    sendtx = _import_sendtx(monkeypatch, verifier, zk_enabled=True, zk_required=False)
    payload_json = json.dumps(_minimal_payload())

    result = sendtx.queue_transaction(payload_json, propagate=False)

    assert "SUCCESS" in result


def test_sendtx_disabled_skips_zk(monkeypatch: pytest.MonkeyPatch):
    """When ZK is disabled, validation is skipped."""
    verifier = StubVerifier(result=False)  # would fail if called
    sendtx = _import_sendtx(monkeypatch, verifier, zk_enabled=False, zk_required=False)
    payload_json = json.dumps(_minimal_payload())

    result = sendtx.queue_transaction(payload_json, propagate=False)

    assert "SUCCESS" in result
    assert verifier.calls == 0

