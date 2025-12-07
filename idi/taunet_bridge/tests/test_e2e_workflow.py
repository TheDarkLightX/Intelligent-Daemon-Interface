"""End-to-end bridge workflow using stub proofs and adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.protocols import ZkProofBundle, InvalidZkProofError
from idi.taunet_bridge.validation import ValidationContext, ZkValidationStep
from idi.taunet_bridge.state_integration import apply_verified_transition, set_zk_verifier
from idi.zk.proof_manager import generate_proof


def _make_stub_bundle(tmp_path: Path) -> ZkProofBundle:
    """Generate a stub proof bundle from minimal manifest/streams."""
    streams_dir = tmp_path / "streams"
    streams_dir.mkdir()
    (streams_dir / "test.in").write_text("1\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"test": "data"}, sort_keys=True), encoding="utf-8")

    bundle = generate_proof(
        manifest_path=manifest_path,
        stream_dir=streams_dir,
        out_dir=tmp_path / "proof",
        prover_command=None,
    )

    return ZkProofBundle(
        proof_path=bundle.proof_path,
        receipt_path=bundle.receipt_path,
        manifest_path=bundle.manifest_path,
        tx_hash="tx123",
    )


def test_validation_and_state_apply_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Full path: proof -> validation step -> state transition with verifier."""
    proof_bundle = _make_stub_bundle(tmp_path)
    config = ZkConfig(enabled=True, require_proofs=True)
    verifier = TauNetZkAdapter(config)

    # Set env flags for state_integration fail-closed behavior
    monkeypatch.setenv("ZK_ENABLED", "1")
    monkeypatch.setenv("ZK_REQUIRE_PROOFS", "1")

    # Prepare validation context
    ctx = ValidationContext(tx_hash="tx123", payload={"zk_proof": proof_bundle})
    ZkValidationStep(verifier, required=True).run(ctx)  # should not raise

    # Stub chain_state for apply_verified_transition
    dummy_chain_state = type("C", (), {"update_balances_after_transfer": staticmethod(lambda *_: True)})
    monkeypatch.setitem(sys.modules, "chain_state", dummy_chain_state)

    set_zk_verifier(verifier)
    assert apply_verified_transition(proof_bundle, "addr1", "addr2", 10) is True


def test_validation_fails_on_tampered_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Tampering receipt digest causes validation to fail and state apply to reject."""
    proof_bundle = _make_stub_bundle(tmp_path)

    # Tamper receipt digest
    receipt = json.loads(proof_bundle.receipt_path.read_text())
    receipt["digest"] = "0" * 64
    proof_bundle.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    config = ZkConfig(enabled=True, require_proofs=True)
    verifier = TauNetZkAdapter(config)

    monkeypatch.setenv("ZK_ENABLED", "1")
    monkeypatch.setenv("ZK_REQUIRE_PROOFS", "1")

    with pytest.raises(InvalidZkProofError):
        ctx = ValidationContext(tx_hash="tx123", payload={"zk_proof": proof_bundle})
        ZkValidationStep(verifier, required=True).run(ctx)

    # Even if validation is skipped, state apply should reject
    dummy_chain_state = type("C", (), {"update_balances_after_transfer": staticmethod(lambda *_: True)})
    monkeypatch.setitem(sys.modules, "chain_state", dummy_chain_state)
    set_zk_verifier(verifier)
    assert apply_verified_transition(proof_bundle, "addr1", "addr2", 10) is False
