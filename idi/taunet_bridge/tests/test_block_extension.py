"""TDD tests for Block extension with ZK proofs."""

from pathlib import Path

import pytest

from idi.taunet_bridge.protocols import ZkProofBundle
from idi.taunet_bridge.block_extension import (
    compute_zk_merkle_root,
    BlockZkExtension,
)


class TestBlockZkExtension:
    """Test Block ZK extension functionality."""

    def test_compute_zk_merkle_root_empty(self):
        """Test computing Merkle root with no proofs."""
        root = compute_zk_merkle_root([])
        assert isinstance(root, str)
        assert len(root) == 64  # SHA256 hex string

    def test_compute_zk_merkle_root_single(self):
        """Test computing Merkle root with single proof."""
        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        root = compute_zk_merkle_root([proof])
        assert isinstance(root, str)
        assert len(root) == 64

    def test_compute_zk_merkle_root_multiple(self):
        """Test computing Merkle root with multiple proofs."""
        proofs = [
            ZkProofBundle(
                proof_path=Path(f"/tmp/proof{i}.bin"),
                receipt_path=Path(f"/tmp/receipt{i}.json"),
                manifest_path=Path("/tmp/manifest.json"),
            )
            for i in range(3)
        ]
        root = compute_zk_merkle_root(proofs)
        assert isinstance(root, str)
        assert len(root) == 64

    def test_block_zk_extension_creation(self):
        """Test creating BlockZkExtension."""
        extension = BlockZkExtension()
        assert extension.zk_commitment is None
        assert extension.zk_proofs == []

    def test_block_zk_extension_with_proofs(self):
        """Test BlockZkExtension with proofs."""
        proofs = [
            ZkProofBundle(
                proof_path=Path("/tmp/proof.bin"),
                receipt_path=Path("/tmp/receipt.json"),
                manifest_path=Path("/tmp/manifest.json"),
            )
        ]
        extension = BlockZkExtension(zk_proofs=proofs)
        assert len(extension.zk_proofs) == 1
        assert extension.zk_commitment is not None

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        proofs = [
            ZkProofBundle(
                proof_path=Path("/tmp/proof.bin"),
                receipt_path=Path("/tmp/receipt.json"),
                manifest_path=Path("/tmp/manifest.json"),
            )
        ]
        extension = BlockZkExtension(zk_proofs=proofs)
        serialized = extension.serialize()
        deserialized = BlockZkExtension.deserialize(serialized)
        assert len(deserialized.zk_proofs) == len(extension.zk_proofs)
        assert deserialized.zk_commitment == extension.zk_commitment

