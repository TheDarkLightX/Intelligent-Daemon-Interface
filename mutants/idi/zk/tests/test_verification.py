"""Tests for VerificationReport and path safety validation."""

from __future__ import annotations

import json

import pytest

from idi.zk.verification import (
    VerificationErrorCode,
    VerificationReport,
    validate_path_safety,
)


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""
    
    def test_ok_creates_successful_report(self):
        """VerificationReport.ok() creates success=True report."""
        report = VerificationReport.ok()
        
        assert report.success is True
        assert report.error_code == VerificationErrorCode.OK
        assert bool(report) is True
    
    def test_ok_with_details(self):
        """VerificationReport.ok() accepts optional details."""
        report = VerificationReport.ok(
            message="Verified successfully",
            digest="abc123",
            method_id="xyz789",
        )
        
        assert report.success is True
        assert report.details["digest"] == "abc123"
        assert report.details["method_id"] == "xyz789"
    
    def test_fail_creates_failed_report(self):
        """VerificationReport.fail() creates success=False report."""
        report = VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Digest does not match",
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH
        assert bool(report) is False
    
    def test_fail_with_details(self):
        """VerificationReport.fail() preserves details."""
        report = VerificationReport.fail(
            VerificationErrorCode.COMMITMENT_MISMATCH,
            "Mismatch detected",
            expected="abc123",
            actual="def456",
        )
        
        assert report.details["expected"] == "abc123"
        assert report.details["actual"] == "def456"
    
    def test_bool_conversion(self):
        """Report can be used in if statements."""
        ok_report = VerificationReport.ok()
        fail_report = VerificationReport.fail(
            VerificationErrorCode.INTERNAL_ERROR,
            "Test error",
        )
        
        # Should work in boolean context
        if ok_report:
            passed = True
        else:
            passed = False
        assert passed is True
        
        if fail_report:
            passed = True
        else:
            passed = False
        assert passed is False
    
    def test_to_dict_serialization(self):
        """Report can be serialized to dict."""
        report = VerificationReport.fail(
            VerificationErrorCode.METHOD_ID_MISMATCH,
            "Wrong method",
            expected="abc",
        )
        
        data = report.to_dict()
        
        assert data["success"] is False
        assert data["error_code"] == "method_id_mismatch"
        assert data["message"] == "Wrong method"
        assert data["details"]["expected"] == "abc"
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        assert "method_id_mismatch" in json_str
    
    def test_from_dict_deserialization(self):
        """Report can be deserialized from dict."""
        data = {
            "success": False,
            "error_code": "commitment_mismatch",
            "message": "Test",
            "details": {"key": "value"},
        }
        
        report = VerificationReport.from_dict(data)
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.COMMITMENT_MISMATCH
        assert report.details["key"] == "value"
    
    def test_error_code_is_str_enum(self):
        """Error codes can be used as strings."""
        code = VerificationErrorCode.RECEIPT_MISSING
        
        # Should work as string
        assert code == "receipt_missing"
        assert str(code.value) == "receipt_missing"


class TestPathSafetyValidation:
    """Tests for validate_path_safety function."""
    
    def test_valid_relative_path(self, tmp_path):
        """Normal relative paths pass validation."""
        report = validate_path_safety(
            "streams/action.in",
            base_dir=str(tmp_path),
        )
        
        assert report.success is True
    
    def test_null_byte_rejected(self):
        """Paths with null bytes are rejected."""
        report = validate_path_safety("streams/action\x00.in")
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.PATH_TRAVERSAL
        assert "null byte" in report.message.lower()
    
    def test_excessive_length_rejected(self):
        """Very long paths are rejected."""
        long_path = "a" * 5000
        report = validate_path_safety(long_path, max_length=4096)
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.PATH_TRAVERSAL
        assert "length" in report.message.lower()
    
    def test_absolute_path_rejected(self, tmp_path):
        """Absolute paths rejected when base_dir provided."""
        report = validate_path_safety(
            "/etc/passwd",
            base_dir=str(tmp_path),
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.PATH_TRAVERSAL
        assert "absolute" in report.message.lower()
    
    def test_parent_traversal_rejected(self, tmp_path):
        """Paths with .. are rejected."""
        report = validate_path_safety(
            "../../../etc/passwd",
            base_dir=str(tmp_path),
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.PATH_TRAVERSAL
        # Message may say "parent", "escapes", or contain ".."
        msg_lower = report.message.lower()
        assert "parent" in msg_lower or "escapes" in msg_lower or ".." in report.message
    
    def test_path_escape_rejected(self, tmp_path):
        """Paths that escape base directory are rejected."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        report = validate_path_safety(
            "subdir/../../outside",
            base_dir=str(tmp_path),
        )
        
        assert report.success is False
        assert report.error_code == VerificationErrorCode.PATH_TRAVERSAL
    
    def test_valid_nested_path(self, tmp_path):
        """Valid nested relative paths pass."""
        report = validate_path_safety(
            "dir1/dir2/file.txt",
            base_dir=str(tmp_path),
        )
        
        assert report.success is True
    
    def test_no_base_dir_allows_relative(self):
        """Without base_dir, relative paths pass basic checks."""
        report = validate_path_safety("relative/path/file.txt")
        
        assert report.success is True
