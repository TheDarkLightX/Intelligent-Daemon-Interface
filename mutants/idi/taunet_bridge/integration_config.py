"""Configuration bridge between Tau Testnet and IDI ZK integration.

Reads environment variables (or Tau config) and produces a `ZkConfig`
instance plus helper factories for dependency injection.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from idi.taunet_bridge.config import ZkConfig
from idi.taunet_bridge.adapter import TauNetZkAdapter


def is_zk_enabled() -> bool:
    """Return True if ZK verification is enabled via env."""
    return os.environ.get("ZK_ENABLED", "0") == "1"


def is_zk_required() -> bool:
    """Return True if ZK proofs are required for all transactions."""
    return os.environ.get("ZK_REQUIRE_PROOFS", "0") == "1"


def _proof_system() -> str:
    """Proof system selection (stub or risc0)."""
    ps = os.environ.get("ZK_PROOF_SYSTEM", "stub")
    if ps not in ("stub", "risc0"):
        raise ValueError(f"Invalid ZK_PROOF_SYSTEM: {ps}")
    return ps


@lru_cache(maxsize=1)
def get_zk_config() -> ZkConfig:
    """Return cached ZkConfig built from env variables."""
    return ZkConfig(
        enabled=is_zk_enabled(),
        proof_system=_proof_system(),
        require_proofs=is_zk_required(),
    )


@lru_cache(maxsize=1)
def get_zk_verifier() -> TauNetZkAdapter:
    """Factory for TauNetZkAdapter based on current config."""
    cfg = get_zk_config()
    return TauNetZkAdapter(cfg)

