"""TDD tests for TauNetZkAdapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.taunet_bridge.protocols import ZkProofBundle, InvalidZkProofError
from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.config import ZkConfig


@pytest.fixture
def tmp_path(tmp_path):
    """Pytest fixture for temporary directory."""
    return tmp_path


class TestTauNetZkAdapter:
    """Test TauNetZkAdapter implementation."""

    def test_adapter_creation(self):
        """Test creating an adapter with config."""
        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)
        assert adapter._config == config

    def test_verify_success(self, tmp_path):
        """Test successful proof verification."""
        import json
        from idi.zk.proof_manager import generate_proof

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

        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)

        proof = ZkProofBundle(
            proof_path=bundle.proof_path,
            receipt_path=bundle.receipt_path,
            manifest_path=bundle.manifest_path,
            tx_hash="tx123",
        )

        # This will use stub verification which should pass with valid files
        result = adapter.verify(proof)
        assert result is True

    def test_verify_disabled(self):
        """Test verification when ZK is disabled."""
        config = ZkConfig(enabled=False, proof_system="stub")
        adapter = TauNetZkAdapter(config)
        
        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        
        # When disabled, should always return True
        result = adapter.verify(proof)
        assert result is True

    def test_verify_rejects_tampered_receipt(self, tmp_path):
        """Receipt digest tampering fails verification."""
        import json
        from idi.zk.proof_manager import generate_proof

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

        receipt = json.loads(bundle.receipt_path.read_text())
        receipt["digest"] = "0" * 64
        bundle.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)

        proof = ZkProofBundle(
            proof_path=bundle.proof_path,
            receipt_path=bundle.receipt_path,
            manifest_path=bundle.manifest_path,
            tx_hash="tx123",
        )

        assert adapter.verify(proof) is False

    def test_integration_with_idi_modules(self):
        """Test adapter integrates with IDI ZK modules."""
        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)
        
        # Check that adapter has access to IDI modules
        assert hasattr(adapter, "_merkle_builder")
        assert isinstance(adapter._merkle_builder, MerkleTreeBuilder)
