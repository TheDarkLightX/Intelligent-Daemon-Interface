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
        import hashlib
        
        # Create temporary files
        receipt_path = tmp_path / "receipt.json"
        manifest_path = tmp_path / "manifest.json"
        proof_path = tmp_path / "proof.bin"
        streams_dir = tmp_path / "streams"
        streams_dir.mkdir()
        
        # Create manifest
        manifest_data = {"test": "data"}
        manifest_path.write_text(json.dumps(manifest_data))
        
        # Create stream file
        (streams_dir / "test.in").write_text("1\n")
        
        # Compute digest
        hasher = hashlib.sha256()
        hasher.update("manifest".encode())
        hasher.update(len(manifest_path.read_bytes()).to_bytes(8, "little"))
        hasher.update(manifest_path.read_bytes())
        hasher.update("streams/test.in".encode())
        hasher.update(len((streams_dir / "test.in").read_bytes()).to_bytes(8, "little"))
        hasher.update((streams_dir / "test.in").read_bytes())
        digest = hasher.hexdigest()
        
        # Create valid receipt matching proof_manager format
        receipt_path.write_text(json.dumps({
            "digest": digest,
            "manifest": str(manifest_path),
            "streams": str(streams_dir),
            "proof": str(proof_path),
        }))
        proof_path.write_bytes(b"test_proof")
        
        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)
        
        proof = ZkProofBundle(
            proof_path=proof_path,
            receipt_path=receipt_path,
            manifest_path=manifest_path,
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

    def test_integration_with_idi_modules(self):
        """Test adapter integrates with IDI ZK modules."""
        config = ZkConfig(enabled=True, proof_system="stub")
        adapter = TauNetZkAdapter(config)
        
        # Check that adapter has access to IDI modules
        assert hasattr(adapter, "_merkle_builder")
        assert isinstance(adapter._merkle_builder, MerkleTreeBuilder)

