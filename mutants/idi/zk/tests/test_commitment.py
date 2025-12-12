"""Tests for Commitment Spec V1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idi.zk.commitment import (
    compute_commitment_bytes,
    compute_commitment_fs,
    compute_commitment_preimage,
)


class TestCommitmentBytes:
    """Tests for bytes-first commitment computation."""
    
    def test_empty_manifest_no_streams(self):
        """Empty manifest with no streams produces consistent digest."""
        digest = compute_commitment_bytes(b"{}", {})
        
        # Should be deterministic
        assert len(digest) == 64  # SHA-256 hex
        assert digest == compute_commitment_bytes(b"{}", {})
    
    def test_manifest_changes_digest(self):
        """Different manifest content produces different digest."""
        digest1 = compute_commitment_bytes(b'{"a":1}', {})
        digest2 = compute_commitment_bytes(b'{"a":2}', {})
        
        assert digest1 != digest2
    
    def test_stream_changes_digest(self):
        """Different stream content produces different digest."""
        streams1 = {"action.in": b"1\n"}
        streams2 = {"action.in": b"2\n"}
        
        digest1 = compute_commitment_bytes(b"{}", streams1)
        digest2 = compute_commitment_bytes(b"{}", streams2)
        
        assert digest1 != digest2
    
    def test_stream_name_changes_digest(self):
        """Different stream names produce different digest."""
        digest1 = compute_commitment_bytes(b"{}", {"action.in": b"1\n"})
        digest2 = compute_commitment_bytes(b"{}", {"reward.in": b"1\n"})
        
        assert digest1 != digest2
    
    def test_extra_binding_changes_digest(self):
        """Extra bindings affect the digest."""
        digest1 = compute_commitment_bytes(b"{}", {})
        digest2 = compute_commitment_bytes(b"{}", {}, {"policy_root": b"abc"})
        
        assert digest1 != digest2
    
    def test_stream_order_independent(self):
        """Stream order doesn't affect digest (sorted by name)."""
        streams = {
            "z_stream.in": b"last",
            "a_stream.in": b"first",
            "m_stream.in": b"middle",
        }
        
        # Create dict in different order
        streams_reversed = dict(reversed(list(streams.items())))
        
        digest1 = compute_commitment_bytes(b"{}", streams)
        digest2 = compute_commitment_bytes(b"{}", streams_reversed)
        
        assert digest1 == digest2
    
    def test_case_insensitive_sort(self):
        """Streams sorted case-insensitively."""
        streams1 = {"Apple.in": b"1", "banana.in": b"2"}
        streams2 = {"banana.in": b"2", "Apple.in": b"1"}
        
        digest1 = compute_commitment_bytes(b"{}", streams1)
        digest2 = compute_commitment_bytes(b"{}", streams2)
        
        assert digest1 == digest2
    
    def test_preimage_contains_version_prefix(self):
        """Preimage starts with version prefix."""
        preimage = compute_commitment_preimage(b"{}", {})
        
        assert preimage.startswith(b"IDI_COMMITMENT_V1\x00")
    
    def test_preimage_contains_manifest(self):
        """Preimage contains manifest content."""
        manifest = b'{"test":"value"}'
        preimage = compute_commitment_preimage(manifest, {})
        
        assert manifest in preimage
    
    def test_preimage_contains_stream_content(self):
        """Preimage contains stream content."""
        streams = {"test.in": b"stream_data"}
        preimage = compute_commitment_preimage(b"{}", streams)
        
        assert b"stream_data" in preimage
        assert b"streams/test.in" in preimage


class TestCommitmentFs:
    """Tests for filesystem-based commitment."""
    
    def test_reads_manifest(self, tmp_path):
        """Reads manifest from path."""
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b'{"key":"value"}')
        
        streams = tmp_path / "streams"
        streams.mkdir()
        
        digest = compute_commitment_fs(manifest, streams)
        
        assert len(digest) == 64
    
    def test_reads_streams(self, tmp_path):
        """Reads stream files from directory."""
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b"{}")
        
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n2\n3\n")
        
        # Should include stream in digest
        digest_with = compute_commitment_fs(manifest, streams)
        
        # Compare with bytes version
        digest_expected = compute_commitment_bytes(
            b"{}",
            {"action.in": b"1\n2\n3\n"},
        )
        
        assert digest_with == digest_expected
    
    def test_only_in_files(self, tmp_path):
        """Only .in files are included."""
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b"{}")
        
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"included")
        (streams / "action.out").write_bytes(b"excluded")
        (streams / "readme.txt").write_bytes(b"excluded")
        
        # Only action.in should be included
        digest = compute_commitment_fs(manifest, streams)
        digest_expected = compute_commitment_bytes(
            b"{}",
            {"action.in": b"included"},
        )
        
        assert digest == digest_expected
    
    def test_matches_bytes_version(self, tmp_path):
        """FS and bytes versions produce identical results."""
        manifest_content = b'{"version":"1.0"}'
        stream_content = b"1\n0\n2\n"
        
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(manifest_content)
        
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(stream_content)
        
        digest_fs = compute_commitment_fs(manifest, streams)
        digest_bytes = compute_commitment_bytes(
            manifest_content,
            {"action.in": stream_content},
        )
        
        assert digest_fs == digest_bytes
    
    def test_extra_bindings(self, tmp_path):
        """Extra bindings work with FS version."""
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b"{}")
        
        streams = tmp_path / "streams"
        streams.mkdir()
        
        policy_root = bytes.fromhex("01" * 32)
        
        digest = compute_commitment_fs(
            manifest,
            streams,
            extra_bindings={"policy_root": policy_root},
        )
        
        # Should match bytes version
        expected = compute_commitment_bytes(
            b"{}",
            {},
            {"policy_root": policy_root},
        )
        
        assert digest == expected


class TestGoldenVectors:
    """Tests using golden vectors for cross-platform verification."""
    
    def test_argmax_vectors(self):
        """Test argmax tie-breaking matches expected behavior."""
        # Load golden vectors
        vectors_path = Path(__file__).parent / "data" / "golden_vectors.json"
        if not vectors_path.exists():
            pytest.skip("Golden vectors file not found")
        
        vectors = json.loads(vectors_path.read_text())
        
        for vec in vectors.get("argmax_vectors", []):
            # This tests the tie-breaking logic
            q_values = [vec["q_hold_fp"], vec["q_buy_fp"], vec["q_sell_fp"]]
            expected = vec["expected_action"]
            
            # Tie-breaking: buy > sell > hold
            # Find max value
            max_val = max(q_values)
            
            # Apply tie-breaking priority
            if q_values[1] == max_val:  # buy
                action = 1
            elif q_values[2] == max_val:  # sell
                action = 2
            else:  # hold
                action = 0
            
            assert action == expected, f"Failed for {vec['name']}"
    
    def test_commitment_deterministic(self):
        """Commitment is deterministic across calls."""
        manifest = b'{"test":"determinism"}'
        streams = {"a.in": b"1", "b.in": b"2"}
        
        digests = [compute_commitment_bytes(manifest, streams) for _ in range(10)]
        
        assert len(set(digests)) == 1, "Commitment should be deterministic"
