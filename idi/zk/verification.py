"""Verification reporting and error handling for ZK proofs.

This module provides structured verification results with detailed error codes,
replacing simple boolean returns with informative reports for debugging
and security auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class VerificationErrorCode(str, Enum):
    """Error codes for ZK proof verification.
    
    Using str as base class allows JSON serialization.
    """
    
    # Success
    OK = "ok"
    
    # File/resource errors
    RECEIPT_MISSING = "receipt_missing"
    RECEIPT_OVERSIZED = "receipt_oversized"
    RECEIPT_PARSE_ERROR = "receipt_parse_error"
    MANIFEST_MISSING = "manifest_missing"
    STREAMS_MISSING = "streams_missing"
    
    # Verification failures
    COMMITMENT_MISMATCH = "commitment_mismatch"
    METHOD_ID_MISSING = "method_id_missing"
    METHOD_ID_MISMATCH = "method_id_mismatch"
    JOURNAL_DIGEST_MISMATCH = "journal_digest_mismatch"
    ZK_RECEIPT_INVALID = "zk_receipt_invalid"
    TX_HASH_MISMATCH = "tx_hash_mismatch"
    
    # Security errors
    PATH_TRAVERSAL = "path_traversal"
    STREAMS_DIGEST_MISMATCH = "streams_digest_mismatch"
    SCHEMA_VERSION_UNSUPPORTED = "schema_version_unsupported"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"
    
    # System errors
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    VERIFIER_TIMEOUT = "verifier_timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class VerificationReport:
    """Structured verification result.
    
    Provides detailed information about verification outcomes,
    replacing simple boolean returns.
    
    Examples:
        >>> report = VerificationReport.ok()
        >>> if report:
        ...     print("Verification passed")
        
        >>> report = VerificationReport.fail(
        ...     VerificationErrorCode.COMMITMENT_MISMATCH,
        ...     "Digest mismatch",
        ...     expected="abc123",
        ...     actual="def456",
        ... )
        >>> print(report.error_code)
        VerificationErrorCode.COMMITMENT_MISMATCH
    """
    
    success: bool
    error_code: VerificationErrorCode
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, message: str = "Verification successful", **details: Any) -> "VerificationReport":
        """Create a successful verification report.
        
        Args:
            message: Optional success message
            **details: Optional additional details (e.g., digest, method_id)
            
        Returns:
            VerificationReport with success=True
        """
        return cls(
            success=True,
            error_code=VerificationErrorCode.OK,
            message=message,
            details=details or {},
        )
    
    @classmethod
    def fail(
        cls,
        code: VerificationErrorCode,
        message: str,
        **details: Any,
    ) -> "VerificationReport":
        """Create a failed verification report.
        
        Args:
            code: The specific error code
            message: Human-readable error message
            **details: Additional context (e.g., expected vs actual values)
            
        Returns:
            VerificationReport with success=False
        """
        return cls(
            success=False,
            error_code=code,
            message=message,
            details=details or {},
        )
    
    def __bool__(self) -> bool:
        """Allow `if report:` usage for checking success."""
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationReport":
        """Create report from dictionary."""
        return cls(
            success=data["success"],
            error_code=VerificationErrorCode(data["error_code"]),
            message=data.get("message", ""),
            details=data.get("details", {}),
        )


def validate_path_safety(
    path_str: str,
    base_dir: Optional[str] = None,
    max_length: int = 4096,
) -> VerificationReport:
    """Validate a path string for security.
    
    Checks for:
    - Null bytes
    - Excessive length
    - Absolute paths (if base_dir provided)
    - Path traversal attempts (../)
    
    Args:
        path_str: The path string to validate
        base_dir: Optional base directory for relative path resolution
        max_length: Maximum allowed path length
        
    Returns:
        VerificationReport indicating if path is safe
    """
    from pathlib import Path
    
    # Check for null bytes
    if "\x00" in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains null byte",
            path=path_str[:100],  # Truncate for safety
        )
    
    # Check length
    if len(path_str) > max_length:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            f"Path exceeds maximum length ({max_length})",
            length=len(path_str),
        )
    
    # Check for absolute path when relative expected
    if base_dir is not None:
        path = Path(path_str)
        if path.is_absolute():
            return VerificationReport.fail(
                VerificationErrorCode.PATH_TRAVERSAL,
                "Absolute path not allowed",
                path=path_str[:100],
            )
        
        # Resolve and check containment
        try:
            base = Path(base_dir).resolve()
            resolved = (base / path).resolve()
            
            if not str(resolved).startswith(str(base)):
                return VerificationReport.fail(
                    VerificationErrorCode.PATH_TRAVERSAL,
                    "Path escapes base directory",
                    path=path_str[:100],
                    base=str(base),
                )
        except (OSError, ValueError) as e:
            return VerificationReport.fail(
                VerificationErrorCode.PATH_TRAVERSAL,
                f"Path resolution failed: {e}",
                path=path_str[:100],
            )
    
    # Check for suspicious patterns
    if ".." in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")
