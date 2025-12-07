"""TDD tests for state integration with ZK proofs."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idi.taunet_bridge.protocols import ZkProofBundle
from idi.taunet_bridge.state_integration import (
    apply_verified_transition,
    get_zk_verifier,
    set_zk_verifier,
)
from idi.taunet_bridge.adapter import TauNetZkAdapter
from idi.taunet_bridge.config import ZkConfig


class TestStateIntegration:
    """Test ZK proof-verified state transitions."""

    def setup_method(self):
        """Reset verifier before each test."""
        set_zk_verifier(None)

    def test_apply_verified_transition_success(self):
        """Test successful verified transition."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True
        set_zk_verifier(mock_verifier)

        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )

        # Mock chain_state module (imported inside apply_verified_transition)
        # Note: This test requires chain_state module from Tau Testnet
        # In actual integration, chain_state would be available
        from unittest.mock import patch, MagicMock
        
        # Create a mock module for chain_state
        mock_chain_state = MagicMock()
        mock_chain_state.update_balances_after_transfer = MagicMock(return_value=True)
        
        # Patch sys.modules to inject mock chain_state
        import sys
        original_chain_state = sys.modules.get("chain_state")
        sys.modules["chain_state"] = mock_chain_state
        
        try:
            result = apply_verified_transition(
                proof=proof,
                from_addr="addr1",
                to_addr="addr2",
                amount=100,
            )

            assert result is True
            mock_verifier.verify.assert_called_once_with(proof)
            mock_chain_state.update_balances_after_transfer.assert_called_once_with("addr1", "addr2", 100)
        finally:
            # Restore original module
            if original_chain_state is not None:
                sys.modules["chain_state"] = original_chain_state
            elif "chain_state" in sys.modules:
                del sys.modules["chain_state"]

    def test_apply_verified_transition_failure(self):
        """Test failed verified transition."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = False
        set_zk_verifier(mock_verifier)

        proof = ZkProofBundle(
            proof_path=Path("/tmp/proof.bin"),
            receipt_path=Path("/tmp/receipt.json"),
            manifest_path=Path("/tmp/manifest.json"),
        )

        result = apply_verified_transition(
            proof=proof,
            from_addr="addr1",
            to_addr="addr2",
            amount=100,
        )

        assert result is False

    def test_get_set_verifier(self):
        """Test getting and setting verifier."""
        config = ZkConfig(enabled=True)
        verifier = TauNetZkAdapter(config)

        set_zk_verifier(verifier)
        retrieved = get_zk_verifier()

        assert retrieved == verifier

