"""Gossip integration helpers for Tau Testnet ZK proofs.

Provides optional hooks to publish/subscribe ZK proofs over the Tau network
when ZK is enabled. Designed to keep Tau Testnet changes minimal and
feature-flagged.
"""

from __future__ import annotations

import logging
from typing import Optional

from idi.taunet_bridge.gossip import ZkGossipProtocol, TAU_PROTOCOL_ZK_PROOFS
from idi.taunet_bridge.integration_config import get_zk_verifier, is_zk_enabled

logger = logging.getLogger(__name__)


def attach_zk_gossip(network_service) -> Optional[ZkGossipProtocol]:
    """Attach ZK gossip protocol to a NetworkService instance if enabled."""
    if not is_zk_enabled():
        logger.info("ZK gossip disabled (ZK_ENABLED=0).")
        return None
    try:
        verifier = get_zk_verifier()
        zk_gossip = ZkGossipProtocol(verifier, network_service._gossip_manager)
        # Subscribe handler (bridge-level validation)
        network_service._gossip_manager.subscribe(TAU_PROTOCOL_ZK_PROOFS, zk_gossip.handle_proof)
        logger.info("ZK gossip protocol attached on topic %s", TAU_PROTOCOL_ZK_PROOFS)
        return zk_gossip
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to attach ZK gossip: %s", e)
        return None

