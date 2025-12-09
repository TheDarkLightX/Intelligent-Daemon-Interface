"""Policy commitment helpers using MerkleTreeBuilder.

Provides deterministic commitment and proof generation for Q-table policies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.zk.witness_generator import QTableEntry, Q16_16_SCALE


@dataclass(frozen=True)
class PolicyCommitment:
    """Commitment metadata for a Q-table policy."""

    root: bytes
    leaf_encoding: str
    q_scale: int
    size: int

    def to_json_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["root"] = self.root.hex()
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, object]) -> "PolicyCommitment":
        return cls(
            root=bytes.fromhex(data["root"]),  # type: ignore[arg-type]
            leaf_encoding=str(data["leaf_encoding"]),
            q_scale=int(data["q_scale"]),
            size=int(data["size"]),
        )


def canonical_leaf_bytes(state_key: str, entry: QTableEntry, q_scale: int = Q16_16_SCALE) -> bytes:
    """Deterministic encoding for a policy leaf.

    Encoding format (matches Risc0 guest `hash_q_entry`):
        b"qtable_entry" || state_key_bytes || i32_le(q_hold) || i32_le(q_buy) || i32_le(q_sell)
    
    IMPORTANT: This encoding MUST match the Risc0 guest's `hash_q_entry` function
    exactly, or Merkle proofs will fail to verify inside the ZK guest program.
    
    See: idi/zk/risc0/methods/idi-qtable/src/main.rs::hash_q_entry()
    """
    # Maintain the q_scale parameter for forwards compatibility even though the
    # current guest only consumes the fixed-point values directly.
    _ = q_scale

    return (
        b"qtable_entry"
        + state_key.encode("utf-8")
        + entry.q_hold.to_bytes(4, "little", signed=True)
        + entry.q_buy.to_bytes(4, "little", signed=True)
        + entry.q_sell.to_bytes(4, "little", signed=True)
    )


def build_policy_commitment(
    entries: Dict[str, QTableEntry], leaf_encoding: str = "qtable_entry_v1"
) -> Tuple[PolicyCommitment, Dict[str, List[Tuple[bytes, bool]]]]:
    """Build a Merkle commitment and proofs for the given Q-table entries."""
    builder = MerkleTreeBuilder()
    for state_key, entry in entries.items():
        builder.add_leaf(state_key, canonical_leaf_bytes(state_key, entry))
    root, proofs = builder.build()
    commitment = PolicyCommitment(
        root=root,
        leaf_encoding=leaf_encoding,
        q_scale=Q16_16_SCALE,
        size=len(entries),
    )
    return commitment, proofs


def save_policy_commitment(
    dir_path: Path,
    commitment: PolicyCommitment,
    proofs_index: Dict[str, List[Tuple[bytes, bool]]],
) -> None:
    """Persist commitment metadata and proofs index to disk."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "qtable.commit.json").write_text(
        json.dumps(commitment.to_json_dict(), indent=2), encoding="utf-8"
    )
    # Proofs: state -> list[[sibling_hex, is_right], ...]
    serializable = {
        state: [[sib.hex(), is_right] for sib, is_right in path] for state, path in proofs_index.items()
    }
    (dir_path / "qtable.proofs.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_policy_commitment(dir_path: Path) -> PolicyCommitment:
    data = json.loads((dir_path / "qtable.commit.json").read_text())
    return PolicyCommitment.from_json_dict(data)


def load_policy_proof(dir_path: Path, state_key: str) -> List[Tuple[bytes, bool]]:
    data = json.loads((dir_path / "qtable.proofs.json").read_text())
    if state_key not in data:
        raise KeyError(f"No proof for state {state_key}")
    return [(bytes.fromhex(item[0]), bool(item[1])) for item in data[state_key]]

