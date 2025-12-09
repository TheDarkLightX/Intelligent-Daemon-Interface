"""End-to-end workflow tests for ZK proof infrastructure.

These tests verify the complete flow:
1. Generate proof (stub mode)
2. Pack to wire bundle
3. Verify from wire bundle
4. Detect tampering
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idi.zk.proof_manager import generate_proof
from idi.zk.wire import ZkProofBundleLocal, ZkProofBundleWireV1
from idi.zk.bundle_verify import (
    verify_commitment_only,
    verify_proof_bundle_local,
    verify_proof_bundle_wire,
)
from idi.zk.commitment import compute_commitment_bytes, compute_commitment_fs
from idi.zk.verification import VerificationErrorCode


class TestE2EStubWorkflow:
    """End-to-end tests for stub proof workflow."""
    
    def test_generate_pack_verify_roundtrip(self, tmp_path):
        """Complete workflow: generate → pack → verify."""
        # Setup
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n0\n2\n")
        (streams / "reward.in").write_bytes(b"0.5\n-0.1\n0.3\n")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "version": "1.0",
            "type": "e2e_test",
            "artifact_id": "test-e2e-001",
        }, sort_keys=True))
        
        proof_dir = tmp_path / "proofs"
        
        # Step 1: Generate proof (stub mode)
        bundle = generate_proof(
            manifest_path=manifest,
            stream_dir=streams,
            out_dir=proof_dir,
            prover_command=None,  # Stub mode
            auto_detect_risc0=False,
        )
        
        assert bundle.proof_path.exists()
        assert bundle.receipt_path.exists()
        
        # Step 2: Create local bundle and pack to wire
        local = ZkProofBundleLocal(
            proof_path=bundle.proof_path,
            attestation_path=bundle.receipt_path,
            manifest_path=manifest,
            stream_dir=streams,
        )
        
        wire = local.to_wire(include_streams=True)
        
        assert wire.schema_version == "1.0"
        assert wire.streams_pack_b64 is not None
        assert wire.streams_sha256 is not None
        
        # Step 3: Serialize and deserialize (simulates network transfer)
        wire_bytes = wire.serialize()
        restored_wire = ZkProofBundleWireV1.deserialize(wire_bytes)
        
        # Step 4: Verify from wire bundle
        report = verify_proof_bundle_wire(
            restored_wire,
            require_zk=False,  # Stub mode
        )
        
        assert report.success, f"Verification failed: {report.message}"
        assert report.details.get("proof_system") == "stub"
    
    def test_wire_commitment_matches_filesystem(self, tmp_path):
        """Wire bundle commitment matches filesystem computation."""
        # Setup
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "data.in").write_bytes(b"test_data\n")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"key":"value"}')
        
        # Generate proof
        bundle = generate_proof(
            manifest_path=manifest,
            stream_dir=streams,
            out_dir=tmp_path / "proof",
            prover_command=None,
            auto_detect_risc0=False,
        )
        
        # Create wire
        local = ZkProofBundleLocal(
            proof_path=bundle.proof_path,
            attestation_path=bundle.receipt_path,
            manifest_path=manifest,
            stream_dir=streams,
        )
        wire = local.to_wire()
        
        # Compute commitments both ways
        wire_commitment = wire.compute_commitment()
        fs_commitment = compute_commitment_fs(manifest, streams)
        
        assert wire_commitment == fs_commitment
    
    def test_tampering_detected_in_wire_flow(self, tmp_path):
        """Tampering is detected in wire bundle flow."""
        import base64
        
        # Setup and generate
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "action.in").write_bytes(b"1\n")
        
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"test":true}')
        
        bundle = generate_proof(
            manifest_path=manifest,
            stream_dir=streams,
            out_dir=tmp_path / "proof",
            prover_command=None,
            auto_detect_risc0=False,
        )
        
        local = ZkProofBundleLocal(
            proof_path=bundle.proof_path,
            attestation_path=bundle.receipt_path,
            manifest_path=manifest,
            stream_dir=streams,
        )
        wire = local.to_wire()
        
        # Tamper with manifest
        original_manifest = base64.b64decode(wire.manifest_json_b64)
        tampered_manifest = original_manifest.replace(b"true", b"false")
        wire.manifest_json_b64 = base64.b64encode(tampered_manifest).decode()
        
        # Verification should fail with commitment mismatch
        report = verify_proof_bundle_wire(wire, require_zk=False)
        
        assert not report.success
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH
    
    def test_local_bundle_roundtrip(self, tmp_path):
        """Local bundle → wire → local preserves all data."""
        # Setup
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "test.in").write_bytes(b"original_stream_data")
        
        manifest = tmp_path / "manifest.json"
        manifest_content = b'{"roundtrip":"test"}'
        manifest.write_bytes(manifest_content)
        
        bundle = generate_proof(
            manifest_path=manifest,
            stream_dir=streams,
            out_dir=tmp_path / "proof",
            prover_command=None,
            auto_detect_risc0=False,
        )
        
        # Create local → wire
        local = ZkProofBundleLocal(
            proof_path=bundle.proof_path,
            attestation_path=bundle.receipt_path,
            manifest_path=manifest,
            stream_dir=streams,
            tx_hash="0xtest123",
        )
        wire = local.to_wire()
        
        # Wire → new local
        dest = tmp_path / "restored"
        restored_local = wire.to_local(dest)
        
        # Verify content matches
        assert restored_local.manifest_path.read_bytes() == manifest_content
        assert (restored_local.stream_dir / "test.in").read_bytes() == b"original_stream_data"
        
        # Verify restored bundle
        report = verify_proof_bundle_local(restored_local, require_zk=False)
        assert report.success


class TestE2ECommitmentConsistency:
    """Tests for commitment consistency across different paths."""
    
    def test_commitment_bytes_matches_fs(self, tmp_path):
        """Bytes-first and FS commitment are identical."""
        # Setup files
        manifest = tmp_path / "manifest.json"
        manifest_content = b'{"consistency":"test"}'
        manifest.write_bytes(manifest_content)
        
        streams = tmp_path / "streams"
        streams.mkdir()
        (streams / "a.in").write_bytes(b"aaa")
        (streams / "b.in").write_bytes(b"bbb")
        
        # Compute both ways
        fs_digest = compute_commitment_fs(manifest, streams)
        bytes_digest = compute_commitment_bytes(
            manifest_content,
            {"a.in": b"aaa", "b.in": b"bbb"},
        )
        
        assert fs_digest == bytes_digest
    
    def test_commitment_order_independent(self, tmp_path):
        """Commitment doesn't depend on file creation order."""
        manifest = tmp_path / "manifest.json"
        manifest.write_bytes(b'{}')
        
        streams = tmp_path / "streams"
        streams.mkdir()
        
        # Create in one order
        (streams / "z.in").write_bytes(b"last")
        (streams / "a.in").write_bytes(b"first")
        (streams / "m.in").write_bytes(b"middle")
        
        digest1 = compute_commitment_fs(manifest, streams)
        
        # Remove and recreate in different order
        for f in streams.glob("*.in"):
            f.unlink()
        
        (streams / "a.in").write_bytes(b"first")
        (streams / "m.in").write_bytes(b"middle")
        (streams / "z.in").write_bytes(b"last")
        
        digest2 = compute_commitment_fs(manifest, streams)
        
        assert digest1 == digest2


class TestE2ECLIIntegration:
    """Test CLI integration (without actually running subprocess)."""
    
    def test_cli_parser_bundle_pack(self):
        """CLI parser accepts bundle pack arguments."""
        from idi.cli import build_parser
        
        parser = build_parser()
        args = parser.parse_args([
            "bundle", "pack",
            "--proof-dir", "/tmp/proof",
            "--out", "/tmp/bundle.json",
            "--tx-hash", "0xabc",
        ])
        
        assert args.command == "bundle"
        assert args.bundle_command == "pack"
        assert args.proof_dir == "/tmp/proof"
        assert args.tx_hash == "0xabc"
    
    def test_cli_parser_bundle_verify(self):
        """CLI parser accepts bundle verify arguments."""
        from idi.cli import build_parser
        
        parser = build_parser()
        args = parser.parse_args([
            "bundle", "verify",
            "--bundle", "/tmp/bundle.json",
            "--method-id", "abc123",
            "--require-zk",
        ])
        
        assert args.command == "bundle"
        assert args.bundle_command == "verify"
        assert args.method_id == "abc123"
        assert args.require_zk is True
