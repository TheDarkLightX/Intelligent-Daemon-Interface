"""TDD tests for Block extension with ZK proofs."""

from idi.taunet_bridge.protocols import NetworkZkProofBundle
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
        proof = NetworkZkProofBundle(
            proof_bytes=b"proof",
            receipt_bytes=b"receipt",
            manifest_bytes=b"manifest",
        )
        root = compute_zk_merkle_root([proof])
        assert isinstance(root, str)
        assert len(root) == 64

    def test_compute_zk_merkle_root_multiple(self):
        """Test computing Merkle root with multiple proofs."""
        proofs = [
            NetworkZkProofBundle(
                proof_bytes=f"proof{i}".encode(),
                receipt_bytes=f"receipt{i}".encode(),
                manifest_bytes=b"manifest",
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
            NetworkZkProofBundle(
                proof_bytes=b"proof",
                receipt_bytes=b"receipt",
                manifest_bytes=b"manifest",
            )
        ]
        extension = BlockZkExtension(zk_proofs=proofs)
        assert len(extension.zk_proofs) == 1
        assert extension.zk_commitment is not None

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        proofs = [
            NetworkZkProofBundle(
                proof_bytes=b"proof",
                receipt_bytes=b"receipt",
                manifest_bytes=b"manifest",
            )
        ]
        extension = BlockZkExtension(zk_proofs=proofs)
        serialized = extension.serialize()
        deserialized = BlockZkExtension.deserialize(serialized)
        assert len(deserialized.zk_proofs) == len(extension.zk_proofs)
        assert deserialized.zk_commitment == extension.zk_commitment
