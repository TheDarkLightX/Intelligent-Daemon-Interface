"""TDD tests for ZK protocol interfaces and data models."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from idi.taunet_bridge.protocols import (
    ZkProofBundle,
    ZkWitness,
    ZkValidationResult,
    ZkVerifier,
    ZkProver,
    InvalidZkProofError,
)


class TestZkProofBundle:
    """Test ZkProofBundle dataclass."""

    def test_creation(self):
        """Test creating a ZkProofBundle."""
        bundle = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        assert bundle.proof_path == Path("/tmp/proof.bin")
        assert bundle.receipt_path == Path("/tmp/receipt.json")
        assert bundle.manifest_path == Path("/tmp/manifest.json")

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        bundle = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        serialized = bundle.serialize()
        deserialized = ZkProofBundle.deserialize(serialized)
        assert deserialized.proof_path == bundle.proof_path
        assert deserialized.receipt_path == bundle.receipt_path
        assert deserialized.manifest_path == bundle.manifest_path

    def test_to_idi_bundle(self):
        """Test conversion to IDI ProofBundle."""
        bundle = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        idi_bundle = bundle.to_idi_bundle()
        assert idi_bundle.proof_path == bundle.proof_path
        assert idi_bundle.receipt_path == bundle.receipt_path
        assert idi_bundle.manifest_path == bundle.manifest_path


class TestZkWitness:
    """Test ZkWitness dataclass."""

    def test_creation(self):
        """Test creating a ZkWitness."""
        witness = ZkWitness(
            state_key="state_0",
            q_table_root=bytes(32),
            selected_action=1,
            merkle_proof=None,
        )
        assert witness.state_key == "state_0"
        assert len(witness.q_table_root) == 32
        assert witness.selected_action == 1

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        witness = ZkWitness(
            state_key="state_0",
            q_table_root=b"\x01" * 32,
            selected_action=1,
            merkle_proof=None,
        )
        serialized = witness.serialize()
        deserialized = ZkWitness.deserialize(serialized)
        assert deserialized.state_key == witness.state_key
        assert deserialized.q_table_root == witness.q_table_root
        assert deserialized.selected_action == witness.selected_action


class TestZkValidationResult:
    """Test ZkValidationResult dataclass."""

    def test_success_result(self):
        """Test successful validation result."""
        result = ZkValidationResult(success=True, error=None)
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test failed validation result."""
        error = InvalidZkProofError("test_tx_hash")
        result = ZkValidationResult(success=False, error=error)
        assert result.success is False
        assert isinstance(result.error, InvalidZkProofError)


class TestZkVerifierProtocol:
    """Test ZkVerifier protocol compliance."""

    def test_protocol_interface(self):
        """Test that protocol defines correct interface."""
        # Check that protocol has verify method
        assert hasattr(ZkVerifier, "verify")
        # Check it's a Protocol (has __protocol_attrs__ in Python 3.8+)
        assert hasattr(ZkVerifier, "__protocol_attrs__") or hasattr(ZkVerifier, "__annotations__")


class TestZkProverProtocol:
    """Test ZkProver protocol compliance."""

    def test_protocol_interface(self):
        """Test that protocol defines correct interface."""
        # Check that protocol has prove method
        assert hasattr(ZkProver, "prove")
        # Check it's a Protocol
        assert hasattr(ZkProver, "__protocol_attrs__") or hasattr(ZkProver, "__annotations__")

