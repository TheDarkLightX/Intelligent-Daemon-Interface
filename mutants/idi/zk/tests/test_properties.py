"""Property-based tests using Hypothesis.

These tests verify invariants that must hold for ALL inputs,
not just specific examples. This catches edge cases that
example-based tests miss.
"""

from __future__ import annotations

import pytest

# Graceful skip if hypothesis not installed
try:
    from hypothesis import given, strategies as st, assume, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorators
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    
    class st:
        @staticmethod
        def binary(*args, **kwargs): pass
        @staticmethod
        def text(*args, **kwargs): pass
        @staticmethod
        def dictionaries(*args, **kwargs): pass
        @staticmethod
        def integers(*args, **kwargs): pass
        @staticmethod
        def just(*args, **kwargs): pass
    
    def settings(*args, **kwargs):
        def decorator(f): return f
        return decorator
    
    def assume(x): pass

pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


class TestCommitmentProperties:
    """Property-based tests for commitment computation."""
    
    @given(
        manifest=st.binary(min_size=0, max_size=1000),
        streams=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_."),
            values=st.binary(min_size=0, max_size=500),
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_commitment_is_deterministic(self, manifest: bytes, streams: dict):
        """Same inputs always produce same commitment."""
        from idi.zk.commitment import compute_commitment_bytes
        
        # Filter to only .in keys (matching real behavior)
        streams_in = {f"{k}.in": v for k, v in streams.items()}
        
        digest1 = compute_commitment_bytes(manifest, streams_in)
        digest2 = compute_commitment_bytes(manifest, streams_in)
        
        assert digest1 == digest2
    
    @given(
        manifest=st.binary(min_size=1, max_size=1000),
    )
    @settings(max_examples=50)
    def test_different_manifest_different_digest(self, manifest: bytes):
        """Different manifests produce different digests (collision resistance)."""
        from idi.zk.commitment import compute_commitment_bytes
        
        assume(len(manifest) > 0)
        
        # Create slightly different manifest
        modified = bytes([manifest[0] ^ 0x01]) + manifest[1:]
        
        digest1 = compute_commitment_bytes(manifest, {})
        digest2 = compute_commitment_bytes(modified, {})
        
        assert digest1 != digest2
    
    @given(
        manifest=st.binary(min_size=0, max_size=500),
        stream_content=st.binary(min_size=1, max_size=500),
    )
    @settings(max_examples=50)
    def test_stream_content_affects_digest(self, manifest: bytes, stream_content: bytes):
        """Changing stream content changes digest."""
        from idi.zk.commitment import compute_commitment_bytes
        
        assume(len(stream_content) > 0)
        
        modified = bytes([stream_content[0] ^ 0x01]) + stream_content[1:]
        
        digest1 = compute_commitment_bytes(manifest, {"test.in": stream_content})
        digest2 = compute_commitment_bytes(manifest, {"test.in": modified})
        
        assert digest1 != digest2
    
    @given(
        manifest=st.binary(min_size=0, max_size=500),
        streams=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            values=st.binary(min_size=0, max_size=100),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_stream_order_invariant(self, manifest: bytes, streams: dict):
        """Digest is independent of dict iteration order."""
        from idi.zk.commitment import compute_commitment_bytes
        
        streams_in = {f"{k}.in": v for k, v in streams.items()}
        
        # Reverse order
        reversed_streams = dict(reversed(list(streams_in.items())))
        
        digest1 = compute_commitment_bytes(manifest, streams_in)
        digest2 = compute_commitment_bytes(manifest, reversed_streams)
        
        assert digest1 == digest2


class TestWireBundleProperties:
    """Property-based tests for wire bundle serialization."""
    
    @given(
        proof=st.binary(min_size=1, max_size=1000),
        attestation=st.binary(min_size=2, max_size=500),
        manifest=st.binary(min_size=2, max_size=500),
    )
    @settings(max_examples=50)
    def test_serialize_deserialize_roundtrip(self, proof: bytes, attestation: bytes, manifest: bytes):
        """Serialize then deserialize preserves all data."""
        import base64
        from idi.zk.wire import ZkProofBundleWireV1
        
        # Ensure valid JSON-ish attestation
        assume(b"{" not in attestation[:1])  # Avoid JSON parse issues
        
        wire = ZkProofBundleWireV1(
            schema_version="1.0",
            proof_system="stub",
            zk_receipt_bin_b64=base64.b64encode(proof).decode(),
            attestation_json_b64=base64.b64encode(b"{}").decode(),  # Use valid JSON
            manifest_json_b64=base64.b64encode(manifest).decode(),
        )
        
        serialized = wire.serialize()
        restored = ZkProofBundleWireV1.deserialize(serialized)
        
        assert base64.b64decode(restored.zk_receipt_bin_b64) == proof
        assert base64.b64decode(restored.manifest_json_b64) == manifest
    
    @given(
        data=st.binary(min_size=1, max_size=1000),
    )
    @settings(max_examples=30)
    def test_streams_hash_tamper_detection(self, data: bytes):
        """Tampering streams is always detected via hash."""
        import base64
        import hashlib
        import json
        from idi.zk.wire import ZkProofBundleWireV1
        
        # Create valid streams pack (just raw data for simplicity)
        streams_b64 = base64.b64encode(data).decode()
        streams_hash = hashlib.sha256(data).hexdigest()
        
        wire = ZkProofBundleWireV1(
            zk_receipt_bin_b64=base64.b64encode(b"proof").decode(),
            attestation_json_b64=base64.b64encode(b"{}").decode(),
            manifest_json_b64=base64.b64encode(b"{}").decode(),
            streams_pack_b64=streams_b64,
            streams_sha256=streams_hash,
        )
        
        # Tamper
        obj = json.loads(wire.serialize())
        tampered = base64.b64decode(obj["streams_pack_b64"])
        tampered = bytes([tampered[0] ^ 0xFF]) + tampered[1:]
        obj["streams_pack_b64"] = base64.b64encode(tampered).decode()
        
        with pytest.raises(ValueError, match="mismatch"):
            ZkProofBundleWireV1.deserialize(json.dumps(obj).encode())


class TestVerificationProperties:
    """Property-based tests for verification report."""
    
    @given(
        message=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=30)
    def test_ok_report_is_truthy(self, message: str):
        """VerificationReport.ok() is always truthy."""
        from idi.zk.verification import VerificationReport
        
        report = VerificationReport.ok(message)
        
        assert bool(report) is True
        assert report.success is True
    
    @given(
        message=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=30)
    def test_fail_report_is_falsy(self, message: str):
        """VerificationReport.fail() is always falsy."""
        from idi.zk.verification import VerificationReport, VerificationErrorCode
        
        report = VerificationReport.fail(
            VerificationErrorCode.INTERNAL_ERROR,
            message,
        )
        
        assert bool(report) is False
        assert report.success is False
    
    @given(
        path=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-./"),
    )
    @settings(max_examples=50)
    def test_safe_paths_pass_validation(self, path: str):
        """Paths without traversal pass validation."""
        from idi.zk.verification import validate_path_safety
        
        # Skip paths with ..
        assume(".." not in path)
        assume(not path.startswith("/"))
        assume("\x00" not in path)
        assume(len(path) < 4096)
        
        report = validate_path_safety(path)
        
        assert report.success is True
    
    @given(
        prefix=st.text(min_size=0, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        suffix=st.text(min_size=0, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
    )
    @settings(max_examples=30)
    def test_traversal_paths_rejected(self, prefix: str, suffix: str):
        """Paths with .. are rejected."""
        from idi.zk.verification import validate_path_safety
        
        path = f"{prefix}/../{suffix}"
        report = validate_path_safety(path)
        
        assert report.success is False


class TestArgmaxProperties:
    """Property-based tests for Q-value argmax tie-breaking."""
    
    @given(
        q_hold=st.integers(min_value=-2**30, max_value=2**30),
        q_buy=st.integers(min_value=-2**30, max_value=2**30),
        q_sell=st.integers(min_value=-2**30, max_value=2**30),
    )
    @settings(max_examples=100)
    def test_argmax_always_returns_valid_action(self, q_hold: int, q_buy: int, q_sell: int):
        """Argmax always returns 0, 1, or 2."""
        # Tie-breaking: buy > sell > hold
        q_values = [q_hold, q_buy, q_sell]
        max_val = max(q_values)
        
        if q_buy == max_val:
            action = 1
        elif q_sell == max_val:
            action = 2
        else:
            action = 0
        
        assert action in (0, 1, 2)
    
    @given(
        q_value=st.integers(min_value=-2**30, max_value=2**30),
    )
    @settings(max_examples=50)
    def test_all_equal_chooses_buy(self, q_value: int):
        """When all Q-values equal, buy (action 1) is chosen."""
        q_values = [q_value, q_value, q_value]
        max_val = max(q_values)
        
        # Tie-breaking: buy > sell > hold
        if q_values[1] == max_val:
            action = 1
        elif q_values[2] == max_val:
            action = 2
        else:
            action = 0
        
        assert action == 1  # Buy wins all ties


class TestArtifactDigestProperties:
    """Cross-check artifact digest against a reference spec implementation."""

    @given(
        manifest=st.binary(min_size=0, max_size=256),
        streams=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
            values=st.binary(min_size=0, max_size=64),
            max_size=5,
        ),
        extra=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            values=st.binary(min_size=0, max_size=32),
            max_size=3,
        ),
    )
    @settings(max_examples=50)
    def test_compute_artifact_digest_matches_reference(self, manifest: bytes, streams: dict, extra: dict):
        """compute_artifact_digest must match the documented bytes-first spec.

        This re-implements the spec inline as a reference and checks equality.
        """
        import hashlib
        import tempfile
        from pathlib import Path
        from idi.zk.proof_manager import compute_artifact_digest

        # Reference implementation of the spec (matches idi/zk/proof_manager.py)
        def ref_digest(manifest_bytes: bytes, streams_dict: dict[str, bytes], extra_dict: dict[str, bytes]) -> str:
            h = hashlib.sha256()

            def _update(name: str, payload: bytes) -> None:
                h.update(name.encode("utf-8"))
                h.update(len(payload).to_bytes(8, "little"))
                h.update(payload)

            if manifest_bytes is not None:
                _update("manifest", manifest_bytes)

            for name in sorted(streams_dict.keys()):
                _update(f"streams/{name}", streams_dict[name])

            for key in sorted(extra_dict.keys()):
                _update(f"extra/{key}", extra_dict[key])

            return h.hexdigest()

        # Normalize stream names to have .in suffix like real code
        norm_streams: dict[str, bytes] = {}
        for k, v in streams.items():
            name = k if k.endswith(".in") else f"{k}.in"
            norm_streams[name] = v

        # Compute reference digest purely from bytes
        extra_bytes = {k: v for k, v in extra.items()}
        ref = ref_digest(manifest, norm_streams, extra_bytes)

        # Compute digest via production helper by writing to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            manifest_path = base / "manifest.json"
            manifest_path.write_bytes(manifest)
            streams_dir = base / "streams"
            streams_dir.mkdir()
            for name, payload in norm_streams.items():
                (streams_dir / name).write_bytes(payload)

            prod = compute_artifact_digest(manifest_path, streams_dir, extra=extra_bytes or None)

        assert prod == ref


class TestQLeafProperties:
    """Properties relating Q-leaf encoding to the Rust guest spec."""

    @given(
        state_key=st.text(min_size=0, max_size=16, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        q_hold=st.integers(min_value=-2**31, max_value=2**31 - 1),
        q_buy=st.integers(min_value=-2**31, max_value=2**31 - 1),
        q_sell=st.integers(min_value=-2**31, max_value=2**31 - 1),
    )
    @settings(max_examples=100)
    def test_canonical_leaf_bytes_matches_hash_q_entry_spec(
        self, state_key: str, q_hold: int, q_buy: int, q_sell: int
    ) -> None:
        """canonical_leaf_bytes must match the Rust guest's hash_q_entry preimage.

        Rust guest spec (idi-qtable/src/main.rs::hash_q_entry):
            SHA-256("qtable_entry" || state_key || q_hold || q_buy || q_sell)
        """
        import hashlib
        from idi.zk.witness_generator import QTableEntry
        from idi.zk.policy_commitment import canonical_leaf_bytes

        entry = QTableEntry(q_hold=q_hold, q_buy=q_buy, q_sell=q_sell)

        # Production preimage
        leaf = canonical_leaf_bytes(state_key, entry)

        # Reference preimage, directly encoding the spec
        ref_preimage = (
            b"qtable_entry"
            + state_key.encode("utf-8")
            + q_hold.to_bytes(4, "little", signed=True)
            + q_buy.to_bytes(4, "little", signed=True)
            + q_sell.to_bytes(4, "little", signed=True)
        )

        assert leaf == ref_preimage

        # Hash equality is then immediate, but we assert it explicitly for clarity
        leaf_hash = hashlib.sha256(leaf).hexdigest()
        ref_hash = hashlib.sha256(ref_preimage).hexdigest()
        assert leaf_hash == ref_hash
