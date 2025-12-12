from pathlib import Path

from idi.zk.policy_commitment import (
    build_policy_commitment,
    canonical_leaf_bytes,
    load_policy_commitment,
    load_policy_proof,
    save_policy_commitment,
)
from idi.zk.witness_generator import QTableEntry
from idi.zk.merkle_tree import MerkleTreeBuilder


def _sample_entries():
    return {
        "state_0": QTableEntry.from_float(0.0, 0.5, 0.0),
        "state_1": QTableEntry.from_float(0.0, 0.0, 0.5),
        "state_2": QTableEntry.from_float(0.5, 0.0, 0.0),
    }


def test_policy_commitment_deterministic_root():
    entries = _sample_entries()
    commitment_a, proofs_a = build_policy_commitment(entries)
    commitment_b, proofs_b = build_policy_commitment(entries)

    assert commitment_a.root == commitment_b.root
    assert set(proofs_a.keys()) == set(proofs_b.keys())


def test_policy_proof_verification(tmp_path: Path):
    entries = _sample_entries()
    commitment, proofs = build_policy_commitment(entries)

    # Save/load roundtrip
    save_policy_commitment(tmp_path, commitment, proofs)
    loaded_commitment = load_policy_commitment(tmp_path)
    proof_path = load_policy_proof(tmp_path, "state_1")

    assert loaded_commitment.root == commitment.root
    assert proof_path == proofs["state_1"]

    builder = MerkleTreeBuilder()
    for k, entry in entries.items():
        builder.add_leaf(k, canonical_leaf_bytes(k, entry))
    root, _ = builder.build()

    assert builder.verify_proof(
        "state_1",
        canonical_leaf_bytes("state_1", entries["state_1"]),
        proof_path,
        root,
    )
