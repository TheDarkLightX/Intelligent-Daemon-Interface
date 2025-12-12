"""Tests for Wire Bundle V1."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import gzip

import pytest

from idi.zk.wire import (
    MAX_MANIFEST_BYTES,
    ZkProofBundleLocal,
    ZkProofBundleWireV1,
    _pack_streams,
    _unpack_streams,
    _unpack_streams_to_dict,
)


class TestStreamsPacking:
    """Tests for stream packing/unpacking."""
    
    def test_pack_and_unpack_roundtrip(self, tmp_path):
        """Packed streams can be unpacked with same content."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n2\n3\n")
        (streams / "reward.in").write_bytes(b"0.5\n-0.1\n")
        
        packed = _pack_streams(streams)
        unpacked = _unpack_streams_to_dict(packed)
        
        assert unpacked["action.in"] == b"1\n2\n3\n"
        assert unpacked["reward.in"] == b"0.5\n-0.1\n"
    
    def test_only_in_files_packed(self, tmp_path):
        """Only .in files are included in pack."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"included")
        (streams / "action.out").write_bytes(b"excluded")
        (streams / "readme.txt").write_bytes(b"excluded")
        
        packed = _pack_streams(streams)
        unpacked = _unpack_streams_to_dict(packed)
        
        assert "action.in" in unpacked
        assert "action.out" not in unpacked
        assert "readme.txt" not in unpacked
    
    def test_pack_is_deterministic(self, tmp_path):
        """Same input produces same packed output."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "a.in").write_bytes(b"data1")
        (streams / "b.in").write_bytes(b"data2")
        
        pack1 = _pack_streams(streams)
        pack2 = _pack_streams(streams)
        
        assert pack1 == pack2


class TestZkProofBundleWireV1:
    """Tests for wire bundle serialization."""
    
    def test_serialize_deserialize_roundtrip(self):
        """Serialize and deserialize produces identical bundle."""
        original = ZkProofBundleWireV1(
            schema_version="1.0",
            proof_system="stub",
            zk_receipt_bin_b64=base64.b64encode(b"proof").decode(),
            attestation_json_b64=base64.b64encode(b'{"prover":"stub"}').decode(),
            manifest_json_b64=base64.b64encode(b'{"version":"1.0"}').decode(),
            streams_pack_b64=None,
            streams_sha256=None,
            tx_hash="0xabc",
            timestamp=12345,
        )
        
        serialized = original.serialize()
        restored = ZkProofBundleWireV1.deserialize(serialized)
        
        assert restored.schema_version == original.schema_version
        assert restored.proof_system == original.proof_system
        assert restored.zk_receipt_bin_b64 == original.zk_receipt_bin_b64
        assert restored.attestation_json_b64 == original.attestation_json_b64
        assert restored.manifest_json_b64 == original.manifest_json_b64
        assert restored.tx_hash == original.tx_hash
        assert restored.timestamp == original.timestamp
    
    def test_roundtrip_preserves_bytes(self, tmp_path):
        """Full roundtrip with streams preserves all bytes."""
        # Create local bundle
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n2\n3\n")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b'{"test":true}')
        
        proof = tmp_path / "proof.bin"
        proof.write_bytes(b"fake_proof_binary")
        
        attestation = tmp_path / "attestation.json"
        attestation.write_bytes(b'{"prover":"stub","digest":"abc"}')
        
        local = ZkProofBundleLocal(
            proof_path=proof,
            attestation_path=attestation,
            manifest_path=manifest,
            stream_dir=streams,
        )
        
        # Convert to wire
        wire = local.to_wire(include_streams=True)
        
        # Serialize and deserialize
        serialized = wire.serialize()
        restored = ZkProofBundleWireV1.deserialize(serialized)
        
        # Verify content
        assert restored.get_manifest_bytes() == b'{"test":true}'
        assert restored.get_streams()["action.in"] == b"1\n2\n3\n"
    
    def test_tamper_detection_streams(self, tmp_path):
        """Tampering streams is detected via hash mismatch."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"original")
        
        packed = _pack_streams(streams)
        streams_b64 = base64.b64encode(packed).decode()
        
        # Create wire with valid hash
        import hashlib
        streams_hash = hashlib.sha256(packed).hexdigest()
        
        wire = ZkProofBundleWireV1(
            zk_receipt_bin_b64=base64.b64encode(b"proof").decode(),
            attestation_json_b64=base64.b64encode(b"{}").decode(),
            manifest_json_b64=base64.b64encode(b"{}").decode(),
            streams_pack_b64=streams_b64,
            streams_sha256=streams_hash,
        )
        
        # Serialize
        data = json.loads(wire.serialize())
        
        # Tamper with streams
        tampered_pack = base64.b64decode(data["streams_pack_b64"])
        tampered = bytes([tampered_pack[0] ^ 0xFF]) + tampered_pack[1:]
        data["streams_pack_b64"] = base64.b64encode(tampered).decode()
        
        # Should fail to deserialize
        with pytest.raises(ValueError, match="streams_sha256 mismatch"):
            ZkProofBundleWireV1.deserialize(json.dumps(data).encode())
    
    def test_oversize_rejection(self):
        """Oversized blobs rejected before full parsing."""
        # Create bundle with oversized manifest
        huge_manifest = b"x" * (MAX_MANIFEST_BYTES + 1000)
        data = {
            "schema_version": "1.0",
            "proof_system": "stub",
            "zk_receipt_bin_b64": base64.b64encode(b"proof").decode(),
            "attestation_json_b64": base64.b64encode(b"{}").decode(),
            "manifest_json_b64": base64.b64encode(huge_manifest).decode(),
            "streams_pack_b64": None,
            "streams_sha256": None,
        }
        
        with pytest.raises(ValueError, match="exceeds size limit"):
            ZkProofBundleWireV1.deserialize(json.dumps(data).encode())
    
    def test_unsupported_schema_rejected(self):
        """Unsupported schema version is rejected."""
        data = {
            "schema_version": "2.0",
            "proof_system": "stub",
            "zk_receipt_bin_b64": "",
            "attestation_json_b64": "",
            "manifest_json_b64": "",
        }
        
        with pytest.raises(ValueError, match="Unsupported schema version"):
            ZkProofBundleWireV1.deserialize(json.dumps(data).encode())
    
    def test_missing_streams_still_parses(self):
        """Bundle without streams parses successfully."""
        wire = ZkProofBundleWireV1(
            zk_receipt_bin_b64=base64.b64encode(b"proof").decode(),
            attestation_json_b64=base64.b64encode(b"{}").decode(),
            manifest_json_b64=base64.b64encode(b"{}").decode(),
            streams_pack_b64=None,
            streams_sha256=None,
        )
        
        serialized = wire.serialize()
        restored = ZkProofBundleWireV1.deserialize(serialized)
        
        assert restored.streams_pack_b64 is None
        assert restored.get_streams() == {}
    
    def test_compute_commitment_without_filesystem(self, tmp_path):
        """Can compute commitment from wire bundle without filesystem."""
        # Create local bundle
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b'{"v":1}')
        
        proof = tmp_path / "proof.bin"
        proof.write_bytes(b"proof")
        
        attestation = tmp_path / "attestation.json"
        attestation.write_bytes(b'{}')
        
        local = ZkProofBundleLocal(
            proof_path=proof,
            attestation_path=attestation,
            manifest_path=manifest,
            stream_dir=streams,
        )
        
        # Convert to wire
        wire = local.to_wire()
        
        # Compute commitment from wire (no filesystem access)
        wire_commitment = wire.compute_commitment()
        
        # Compute commitment from filesystem
        from idi.zk.commitment import compute_commitment_fs
        fs_commitment = compute_commitment_fs(manifest, streams)
        
        assert wire_commitment == fs_commitment

    def test_verify_integrity_invalid_streams_base64(self):
        """verify_integrity reports RECEIPT_PARSE_ERROR on invalid streams base64."""
        from idi.zk.verification import VerificationErrorCode
        
        wire = ZkProofBundleWireV1(
            zk_receipt_bin_b64=base64.b64encode(b"proof").decode(),
            attestation_json_b64=base64.b64encode(b"{}").decode(),
            manifest_json_b64=base64.b64encode(b"{}").decode(),
            streams_pack_b64="not-base64",
            streams_sha256="0" * 64,
        )
        
        report = wire.verify_integrity()
        assert report.success is False
        assert report.error_code == VerificationErrorCode.RECEIPT_PARSE_ERROR

    def test_unpack_streams_rejects_path_traversal(self, tmp_path):
        """_unpack_streams rejects archives with path traversal entries."""
        import io
        import tarfile
        
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                info = tarfile.TarInfo(name="../evil.in")
                payload = b"data"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        pack_bytes = buf.getvalue()
        
        with pytest.raises(ValueError, match="Unsafe path in archive"):
            _unpack_streams(pack_bytes, tmp_path)


class TestZkProofBundleLocal:
    """Tests for local bundle operations."""
    
    def test_to_wire_includes_streams(self, tmp_path):
        """to_wire() includes streams by default."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "test.in").write_bytes(b"data")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b"{}")
        
        proof = tmp_path / "proof.bin"
        proof.write_bytes(b"proof")
        
        attestation = tmp_path / "attestation.json"
        attestation.write_bytes(b'{"prover":"stub"}')
        
        local = ZkProofBundleLocal(
            proof_path=proof,
            attestation_path=attestation,
            manifest_path=manifest,
            stream_dir=streams,
        )
        
        wire = local.to_wire()
        
        assert wire.streams_pack_b64 is not None
        assert wire.streams_sha256 is not None
        assert wire.get_streams()["test.in"] == b"data"
    
    def test_to_wire_without_streams(self, tmp_path):
        """to_wire(include_streams=False) omits streams."""
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "test.in").write_bytes(b"data")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b"{}")
        
        proof = tmp_path / "proof.bin"
        proof.write_bytes(b"proof")
        
        attestation = tmp_path / "attestation.json"
        attestation.write_bytes(b'{}')
        
        local = ZkProofBundleLocal(
            proof_path=proof,
            attestation_path=attestation,
            manifest_path=manifest,
            stream_dir=streams,
        )
        
        wire = local.to_wire(include_streams=False)
        
        assert wire.streams_pack_b64 is None
        assert wire.streams_sha256 is None
    
    def test_wire_to_local_roundtrip(self, tmp_path):
        """Wire bundle can be materialized back to local."""
        # Create original local bundle
        orig_streams = tmp_path / "orig" / "streams"
        orig_streams.mkdir(parents=True)
        (orig_streams / "action.in").write_bytes(b"1\n2\n3\n")
        
        orig_manifest = tmp_path / "orig" / "manifest.json"
        orig_manifest.write_bytes(b'{"original":true}')
        
        orig_proof = tmp_path / "orig" / "proof.bin"
        orig_proof.write_bytes(b"original_proof")
        
        orig_attestation = tmp_path / "orig" / "attestation.json"
        orig_attestation.write_bytes(b'{"prover":"stub"}')
        
        local = ZkProofBundleLocal(
            proof_path=orig_proof,
            attestation_path=orig_attestation,
            manifest_path=orig_manifest,
            stream_dir=orig_streams,
        )
        
        # Convert to wire
        wire = local.to_wire()
        
        # Materialize to different location
        dest = tmp_path / "dest"
        restored = wire.to_local(dest)
        
        # Verify files exist and match
        assert restored.proof_path.read_bytes() == b"original_proof"
        assert restored.manifest_path.read_bytes() == b'{"original":true}'
        assert (restored.stream_dir / "action.in").read_bytes() == b"1\n2\n3\n"
