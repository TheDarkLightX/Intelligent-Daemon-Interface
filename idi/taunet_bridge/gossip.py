"""ZK proof gossip protocol for P2P propagation.

This module implements a Gossipsub protocol for propagating ZK proofs across
the Tau Testnet P2P network, following the Adapter pattern to integrate with
existing network infrastructure.
"""

from __future__ import annotations

from typing import Protocol

from idi.taunet_bridge.protocols import ZkProofBundle, ZkVerifier, InvalidZkProofError

TAU_PROTOCOL_ZK_PROOFS = "/tau/zkproofs/1.0.0"


class GossipService(Protocol):
    """Protocol for gossip service interface.

    This protocol abstracts the gossip service, allowing integration with
    different P2P implementations (e.g., libp2p gossipsub).
    """

    async def publish(self, topic: str, data: bytes) -> None:
        """Publish data to a gossip topic."""
        ...


class ZkGossipProtocol:
    """Gossipsub protocol for ZK proof propagation.

    This class implements ZK proof propagation over the Tau Testnet P2P network,
    following the Adapter pattern to bridge between our ZK infrastructure and
    the network layer.
    """

    def __init__(self, verifier: ZkVerifier, gossip_service: GossipService):
        """Initialize ZK gossip protocol.

        Args:
            verifier: ZK proof verifier (following DIP)
            gossip_service: Gossip service for P2P communication
        """
        self._verifier = verifier
        self._gossip = gossip_service

    async def broadcast_proof(self, proof: ZkProofBundle) -> None:
        """Broadcast a ZK proof to the network.

        Args:
            proof: The ZK proof bundle to broadcast
        """
        serialized = proof.serialize()
        await self._gossip.publish(TAU_PROTOCOL_ZK_PROOFS, serialized)

    async def handle_proof(self, data: bytes) -> ZkProofBundle:
        """Handle an incoming ZK proof from the network.

        Args:
            data: Serialized proof data

        Returns:
            Deserialized and verified proof bundle

        Raises:
            InvalidZkProofError: If proof is invalid or verification fails
        """
        try:
            proof = ZkProofBundle.deserialize(data)
        except Exception as e:
            raise InvalidZkProofError("unknown", reason=f"Deserialization failed: {e}")

        if not self._verifier.verify(proof):
            raise InvalidZkProofError(proof.tx_hash or "unknown", reason="Verification failed")

        return proof

