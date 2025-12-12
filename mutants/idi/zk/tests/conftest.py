"""ZK-specific pytest fixtures.

These fixtures provide ZK proof-related test data including
proof bundles, witnesses, and Merkle trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def stub_proof_dir(tmp_path: Path, sample_manifest: Path, sample_streams: Path) -> Path:
    """Create a complete stub proof directory structure.
    
    Creates:
        tmp_path/
        ├── manifest.json
        ├── streams/
        │   ├── action.in
        │   └── reward.in
        └── proofs/
            ├── proof.bin
            └── receipt.json
    """
    from idi.zk.proof_manager import generate_proof
    
    proofs_dir = tmp_path / "proofs"
    proofs_dir.mkdir(exist_ok=True)
    
    # Generate a stub proof
    bundle = generate_proof(
        manifest_path=sample_manifest,
        stream_dir=sample_streams,
        out_dir=proofs_dir,
        prover_command=None,  # Use stub
        auto_detect_risc0=False,
    )
    
    return tmp_path


@pytest.fixture
def sample_qtable_entry() -> Dict[str, Any]:
    """Sample Q-table entry in various formats.
    
    Returns dict with:
    - float_values: Original float Q-values
    - fixed_point: Q16.16 fixed-point integers
    - state_key: The state identifier
    """
    return {
        "state_key": "state_0",
        "float_values": {"hold": 0.0, "buy": 0.5, "sell": -0.25},
        "fixed_point": {
            "q_hold": 0,          # 0.0 * 65536
            "q_buy": 32768,       # 0.5 * 65536
            "q_sell": -16384,     # -0.25 * 65536
        },
        "expected_action": 1,  # buy (highest Q-value)
    }


@pytest.fixture
def sample_merkle_leaves() -> list[bytes]:
    """Sample leaf hashes for Merkle tree testing."""
    import hashlib
    
    leaves = []
    for i in range(8):
        data = f"leaf_{i}".encode()
        leaves.append(hashlib.sha256(data).digest())
    return leaves


@pytest.fixture
def golden_commitment_vector() -> Dict[str, Any]:
    """Golden test vector for commitment verification.
    
    This vector ensures Python and Rust produce identical commitments.
    """
    return {
        "manifest_hex": "7b2276657273696f6e223a22312e30227d",  # {"version":"1.0"}
        "streams": {
            "action.in": "310a",  # "1\n"
        },
        "expected_digest": None,  # Will be computed and locked in
    }


@pytest.fixture  
def zk_config() -> Dict[str, Any]:
    """Default ZK configuration for testing."""
    return {
        "enabled": True,
        "proof_system": "stub",
        "require_proofs": False,
        "max_proof_bytes": 5 * 1024 * 1024,  # 5MB
        "max_receipt_bytes": 512 * 1024,     # 512KB
    }
