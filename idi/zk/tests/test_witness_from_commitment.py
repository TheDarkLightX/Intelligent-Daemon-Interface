from pathlib import Path

from idi.zk.policy_commitment import build_policy_commitment, save_policy_commitment
from idi.zk.witness_generator import (
    QTableEntry,
    generate_witness_from_commitment,
    serialize_witness,
)


def test_generate_witness_from_commitment_roundtrip(tmp_path: Path):
    entries = {
        "state_0": QTableEntry.from_float(0.0, 0.5, 0.0),
        "state_1": QTableEntry.from_float(0.0, 0.0, 0.5),
    }
    commitment, proofs = build_policy_commitment(entries)
    commit_dir = tmp_path / "policy"
    save_policy_commitment(commit_dir, commitment, proofs)

    witness = generate_witness_from_commitment(
        commit_dir=commit_dir,
        state_key="state_1",
        entry=entries["state_1"],
    )

    data = serialize_witness(witness)
    assert b"state_1" in data
    assert commitment.root.hex().encode() in data
