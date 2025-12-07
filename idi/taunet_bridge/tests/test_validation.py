"""TDD tests for ZK validation step."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idi.taunet_bridge.protocols import ZkProofBundle, InvalidZkProofError
from idi.taunet_bridge.validation import ZkValidationStep, ValidationContext
from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.config import ZkConfig


class TestZkValidationStep:
    """Test ZkValidationStep chain-of-responsibility step."""

    def test_validation_with_valid_proof(self):
        """Test validation passes with valid proof."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True
        
        step = ZkValidationStep(mock_verifier)
        
        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        
        ctx = ValidationContext(
            tx_hash="test_hash",
            payload={"zk_proof": proof},
        )
        
        # Should not raise
        step.run(ctx)
        mock_verifier.verify.assert_called_once_with(proof)

    def test_validation_with_invalid_proof(self):
        """Test validation fails with invalid proof."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = False
        
        step = ZkValidationStep(mock_verifier)
        
        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        
        ctx = ValidationContext(
            tx_hash="test_hash",
            payload={"zk_proof": proof},
        )
        
        with pytest.raises(InvalidZkProofError) as exc_info:
            step.run(ctx)
        
        assert exc_info.value.tx_hash == "test_hash"

    def test_validation_without_proof(self):
        """Test validation passes when proof is optional and missing."""
        mock_verifier = MagicMock()
        step = ZkValidationStep(mock_verifier, required=False)
        
        ctx = ValidationContext(
            tx_hash="test_hash",
            payload={},  # No zk_proof
        )
        
        # Should not raise and not call verifier
        step.run(ctx)
        mock_verifier.verify.assert_not_called()

    def test_validation_required_proof_missing(self):
        """Test validation fails when proof is required but missing."""
        mock_verifier = MagicMock()
        step = ZkValidationStep(mock_verifier, required=True)
        
        ctx = ValidationContext(
            tx_hash="test_hash",
            payload={},  # No zk_proof
        )
        
        with pytest.raises(InvalidZkProofError) as exc_info:
            step.run(ctx)
        
        assert "missing" in str(exc_info.value.reason).lower()


class TestValidationContext:
    """Test ValidationContext data structure."""

    def test_context_creation(self):
        """Test creating a validation context."""
        ctx = ValidationContext(
            tx_hash="test_hash",
            payload={"test": "data"},
        )
        assert ctx.tx_hash == "test_hash"
        assert ctx.payload == {"test": "data"}

    def test_has_zk_proof(self):
        """Test checking for ZK proof in context."""
        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )
        
        ctx_with_proof = ValidationContext(
            tx_hash="test_hash",
            payload={"zk_proof": proof},
        )
        assert ctx_with_proof.has_zk_proof is True
        
        ctx_without_proof = ValidationContext(
            tx_hash="test_hash",
            payload={},
        )
        assert ctx_without_proof.has_zk_proof is False

