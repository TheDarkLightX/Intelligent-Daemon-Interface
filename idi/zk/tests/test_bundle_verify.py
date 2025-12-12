"""Tests for high-level bundle verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idi.zk.bundle_verify import (
    verify_commitment_only,
    verify_proof_bundle_local,
    verify_proof_bundle_wire,
)
from idi.zk.commitment import compute_commitment_bytes, compute_commitment_fs
from idi.zk.verification import VerificationErrorCode
from idi.zk.wire import ZkProofBundleLocal, ZkProofBundleWireV1


@pytest.fixture
def valid_local_bundle(tmp_path):
    """Create a valid local bundle for testing using proof_manager."""
    from idi.zk.proof_manager import generate_proof
    
    # Create files
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"test":"data"}')
    
    streams = tmp_path / "streams"
    streams.mkdir()
    (streams / "action.in").write_bytes(b"1\n2\n3\n")
    
    proof_dir = tmp_path / "proof"
    
    # Generate actual stub proof (computes digest correctly)
    bundle = generate_proof(
        manifest_path=manifest,
        stream_dir=streams,
        out_dir=proof_dir,
        prover_command=None,
        auto_detect_risc0=False,
    )
    
    return ZkProofBundleLocal(
        proof_path=bundle.proof_path,
        attestation_path=bundle.receipt_path,
        manifest_path=manifest,
        stream_dir=streams,
    )


class TestVerifyProofBundleLocal:
    """Tests for local bundle verification."""
    
    def test_valid_stub_bundle_passes(self, valid_local_bundle):
        """Valid stub bundle passes verification."""
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,  # Skip ZK for stub
        )
        
        assert report.success is True
        assert report.details.get("proof_system") == "stub"
    
    def test_missing_proof_fails(self, valid_local_bundle):
        """Missing proof file fails with RECEIPT_MISSING."""
        valid_local_bundle.proof_path.unlink()
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.RECEIPT_MISSING
    
    def test_missing_attestation_fails(self, valid_local_bundle):
        """Missing attestation fails with RECEIPT_MISSING."""
        valid_local_bundle.attestation_path.unlink()
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.RECEIPT_MISSING
    
    def test_missing_manifest_fails(self, valid_local_bundle):
        """Missing manifest fails with MANIFEST_MISSING."""
        valid_local_bundle.manifest_path.unlink()
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.MANIFEST_MISSING
    
    def test_commitment_mismatch_fails(self, valid_local_bundle):
        """Tampered manifest causes COMMITMENT_MISMATCH."""
        # Modify manifest after digest computed
        valid_local_bundle.manifest_path.write_bytes(b'{"tampered":true}')
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH
    
    def test_tx_hash_mismatch_fails(self, valid_local_bundle):
        """tx_hash mismatch fails."""
        # Add tx_hash to attestation
        attestation = json.loads(valid_local_bundle.attestation_path.read_text())
        attestation["tx_hash"] = "0xoriginal"
        valid_local_bundle.attestation_path.write_text(json.dumps(attestation))
        
        # Bundle has different tx_hash
        valid_local_bundle.tx_hash = "0xdifferent"
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.TX_HASH_MISMATCH
    
    def test_oversized_proof_fails(self, valid_local_bundle):
        """Oversized proof fails with SIZE_LIMIT_EXCEEDED."""
        # Write large proof
        valid_local_bundle.proof_path.write_bytes(b"x" * 1000)
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
            max_proof_bytes=500,  # Set low limit
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.SIZE_LIMIT_EXCEEDED
    
    def test_invalid_attestation_json_fails(self, valid_local_bundle):
        """Invalid attestation JSON fails with RECEIPT_PARSE_ERROR."""
        valid_local_bundle.attestation_path.write_bytes(b"not json")
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.RECEIPT_PARSE_ERROR
    
    def test_risc0_require_zk_needs_verifier(self, valid_local_bundle):
        """Risc0 with require_zk=True needs verifier binary."""
        # Set prover to risc0
        attestation = json.loads(valid_local_bundle.attestation_path.read_text())
        attestation["prover"] = "risc0"
        valid_local_bundle.attestation_path.write_text(json.dumps(attestation))
        
        report = verify_proof_bundle_local(
            valid_local_bundle,
            require_zk=True,
            expected_method_id="abc123",
        )
        
        # Should fail (either verifier missing or invalid receipt if a verifier exists)
        assert report.success is False
        assert report.error_code in {
            VerificationErrorCode.VERIFIER_UNAVAILABLE,
            VerificationErrorCode.ZK_RECEIPT_INVALID,
        }


class TestVerifyProofBundleWire:
    """Tests for wire bundle verification."""
    
    def test_valid_wire_bundle_passes(self, valid_local_bundle):
        """Valid wire bundle passes verification."""
        wire = valid_local_bundle.to_wire()
        
        report = verify_proof_bundle_wire(
            wire,
            require_zk=False,
        )
        
        assert report.success is True
    
    def test_tampered_streams_detected(self, valid_local_bundle):
        """Tampered streams hash is detected."""
        import base64
        import hashlib
        
        wire = valid_local_bundle.to_wire()
        
        # Tamper the hash
        wire.streams_sha256 = "0" * 64  # Wrong hash
        
        report = verify_proof_bundle_wire(
            wire,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.STREAMS_DIGEST_MISMATCH
    
    def test_commitment_mismatch_in_wire(self, valid_local_bundle):
        """Commitment mismatch detected in wire bundle."""
        import base64
        
        wire = valid_local_bundle.to_wire()
        
        # Modify attestation to have wrong digest
        attestation = wire.get_attestation()
        attestation["digest_hex"] = "wrong" * 16
        wire.attestation_json_b64 = base64.b64encode(
            json.dumps(attestation).encode()
        ).decode()
        
        report = verify_proof_bundle_wire(
            wire,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH

    def test_invalid_attestation_in_wire(self, valid_local_bundle):
        """Invalid attestation JSON in wire bundle fails with RECEIPT_PARSE_ERROR."""
        import base64
        
        wire = valid_local_bundle.to_wire()
        # Corrupt attestation so JSON parsing fails
        wire.attestation_json_b64 = base64.b64encode(b"not json").decode()
        
        report = verify_proof_bundle_wire(
            wire,
            require_zk=False,
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.RECEIPT_PARSE_ERROR


class TestVerifyCommitmentOnly:
    """Tests for commitment-only verification."""
    
    def test_matching_commitment_passes(self):
        """Matching commitment passes."""
        manifest = b'{"test":1}'
        streams = {"a.in": b"data"}
        
        expected = compute_commitment_bytes(manifest, streams)
        
        report = verify_commitment_only(manifest, streams, expected)
        
        assert report.success is True
    
    def test_mismatched_commitment_fails(self):
        """Mismatched commitment fails."""
        manifest = b'{"test":1}'
        streams = {"a.in": b"data"}
        
        report = verify_commitment_only(manifest, streams, "wrong" * 16)
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH
    
    def test_empty_inputs(self):
        """Empty manifest and streams work."""
        manifest = b"{}"
        streams = {}
        
        expected = compute_commitment_bytes(manifest, streams)
        
        report = verify_commitment_only(manifest, streams, expected)
        
        assert report.success is True
