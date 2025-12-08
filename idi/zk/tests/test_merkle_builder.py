import random
from pathlib import Path

from idi.zk.merkle_tree import MerkleTreeBuilder
from idi.devkit.manifest import build_manifest, write_manifest
from idi.zk.proof_manager import compute_manifest_streams_digest


def _sample_entries():
    return {
        "state_0": b"entry0",
        "state_1": b"entry1",
        "state_2": b"entry2",
        "state_3": b"entry3",
    }


def test_merkle_builder_deterministic_root_and_proofs():
    entries = _sample_entries()

    # Build with natural order
    builder_a = MerkleTreeBuilder()
    for k, v in entries.items():
        builder_a.add_leaf(k, v)
    root_a, proofs_a = builder_a.build()

    # Build with shuffled order
    items = list(entries.items())
    random.shuffle(items)
    builder_b = MerkleTreeBuilder()
    for k, v in items:
        builder_b.add_leaf(k, v)
    root_b, proofs_b = builder_b.build()

    assert root_a == root_b, "Root should be deterministic regardless of insertion order"
    assert set(proofs_a.keys()) == set(proofs_b.keys())

    # Verify proofs from builder A using builder A's verify helper
    for state_key, data in entries.items():
        assert builder_a.verify_proof(state_key, data, proofs_a[state_key], root_a)


def test_compute_manifest_streams_digest_changes_on_tamper(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "q_buy.in").write_text("1\n", encoding="utf-8")

    manifest = build_manifest(config_path=cfg, stream_dir=streams, metadata={})
    manifest_path = tmp_path / "artifact_manifest.json"
    write_manifest(manifest, manifest_path)

    digest_original = compute_manifest_streams_digest(manifest_path, streams)

    # Tamper stream
    (streams / "q_buy.in").write_text("2\n", encoding="utf-8")
    digest_tampered = compute_manifest_streams_digest(manifest_path, streams)

    assert digest_original != digest_tampered
