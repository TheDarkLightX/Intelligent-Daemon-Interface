"""TDD tests for ZK gossip protocol."""

from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import pytest_asyncio
except ImportError:
    pytest_asyncio = None

from idi.taunet_bridge.protocols import NetworkZkProofBundle, InvalidZkProofError
from idi.taunet_bridge.gossip import ZkGossipProtocol, TAU_PROTOCOL_ZK_PROOFS
from idi.taunet_bridge.adapter import TauNetZkAdapter


class TestZkGossipProtocol:
    """Test ZK gossip protocol implementation."""

    @pytest.fixture
    def mock_verifier(self):
        """Create mock verifier."""
        verifier = MagicMock(spec=TauNetZkAdapter)
        return verifier

    @pytest.fixture
    def mock_gossip(self):
        """Create mock gossip service."""
        gossip = AsyncMock()
        return gossip

    @pytest.fixture
    def zk_gossip(self, mock_verifier, mock_gossip):
        """Create ZkGossipProtocol instance."""
        return ZkGossipProtocol(mock_verifier, mock_gossip)

    def test_protocol_constant(self):
        """Test protocol constant is defined."""
        assert TAU_PROTOCOL_ZK_PROOFS == "/tau/zkproofs/1.0.0"

    def test_broadcast_proof_sync(self, zk_gossip, mock_gossip):
        """Test broadcasting a proof (synchronous wrapper)."""
        import asyncio
        
        proof = NetworkZkProofBundle(
            proof_bytes=b"proof",
            receipt_bytes=b"receipt",
            manifest_bytes=b"manifest",
        )

        # Run async function synchronously for testing
        asyncio.run(zk_gossip.broadcast_proof(proof))

        # Verify publish was called
        mock_gossip.publish.assert_called_once()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == TAU_PROTOCOL_ZK_PROOFS
        assert isinstance(call_args[0][1], bytes)

    def test_handle_proof_valid_sync(self, zk_gossip, mock_verifier):
        """Test handling a valid proof (synchronous wrapper)."""
        import asyncio
        
        proof = NetworkZkProofBundle(
            proof_bytes=b"proof",
            receipt_bytes=b"receipt",
            manifest_bytes=b"manifest",
        )
        mock_verifier.verify.return_value = True

        data = proof.serialize()
        result = asyncio.run(zk_gossip.handle_proof(data))

        assert isinstance(result, NetworkZkProofBundle)
        mock_verifier.verify.assert_called_once()

    def test_handle_proof_invalid_sync(self, zk_gossip, mock_verifier):
        """Test handling an invalid proof (synchronous wrapper)."""
        import asyncio
        
        proof = NetworkZkProofBundle(
            proof_bytes=b"proof",
            receipt_bytes=b"receipt",
            manifest_bytes=b"manifest",
        )
        mock_verifier.verify.return_value = False

        data = proof.serialize()
        with pytest.raises(InvalidZkProofError):
            asyncio.run(zk_gossip.handle_proof(data))
