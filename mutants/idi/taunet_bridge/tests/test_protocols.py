"""TDD tests for ZK protocol interfaces and data models."""

from pathlib import Path

import pytest

from idi.taunet_bridge.protocols import (
    LocalZkProofBundle,
    NetworkZkProofBundle,
    ZkWitness,
    ZkValidationResult,
    ZkVerifier,
    ZkProver,
    InvalidZkProofError,
    deserialize_proof_bundle,
)


class TestZkProofBundles:
    """Test ZkProofBundle variants."""

    def test_network_roundtrip(self):
        """Network bundle base64 serialization/deserialization."""
        bundle = NetworkZkProofBundle(
            proof_bytes=b"proof",
            receipt_bytes=b'{"digest":"abc"}',
            manifest_bytes=b'{"manifest": true}',
            tx_hash="tx123",
        )
        serialized = bundle.serialize()
        deserialized = deserialize_proof_bundle(serialized)
        assert isinstance(deserialized, NetworkZkProofBundle)
        assert deserialized.proof_bytes == b"proof"
        assert deserialized.receipt_bytes.startswith(b"{")
        assert deserialized.tx_hash == "tx123"

    def test_local_serialize_uses_data_bytes(self, tmp_path: Path):
        """Local bundle serializes to the data, not paths."""
        proof = tmp_path / "proof.bin"
        receipt = tmp_path / "receipt.json"
        manifest = tmp_path / "manifest.json"
        proof.write_bytes(b"proof-bytes")
        receipt.write_bytes(b'{"digest":"abc"}')
        manifest.write_bytes(b"{}")

        bundle = LocalZkProofBundle(
            proof_path=proof,
            receipt_path=receipt,
            manifest_path=manifest,
            tx_hash="tx123",
        )
        serialized = bundle.serialize()
        deserialized = deserialize_proof_bundle(serialized)
        assert isinstance(deserialized, NetworkZkProofBundle)
        assert deserialized.proof_bytes == b"proof-bytes"
        assert deserialized.tx_hash == "tx123"

    def test_to_idi_bundle(self, tmp_path: Path):
        """Test conversion to IDI ProofBundle."""
        proof = tmp_path / "proof.bin"
        receipt = tmp_path / "receipt.json"
        manifest = tmp_path / "manifest.json"
        proof.write_bytes(b"proof")
        receipt.write_text("{}", encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")

        bundle = LocalZkProofBundle(
            proof_path=proof,
            receipt_path=receipt,
            manifest_path=manifest,
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
