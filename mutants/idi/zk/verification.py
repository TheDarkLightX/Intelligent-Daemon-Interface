"""Verification reporting and error handling for ZK proofs.

This module provides structured verification results with detailed error codes,
replacing simple boolean returns with informative reports for debugging
and security auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg is not None:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


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


def x_validate_path_safety__mutmut_orig(
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


def x_validate_path_safety__mutmut_1(
    path_str: str,
    base_dir: Optional[str] = None,
    max_length: int = 4097,
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


def x_validate_path_safety__mutmut_2(
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
    if "XX\x00XX" in path_str:
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


def x_validate_path_safety__mutmut_3(
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
    if "\x00" not in path_str:
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


def x_validate_path_safety__mutmut_4(
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
            None,
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


def x_validate_path_safety__mutmut_5(
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
            None,
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


def x_validate_path_safety__mutmut_6(
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
            path=None,  # Truncate for safety
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


def x_validate_path_safety__mutmut_7(
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


def x_validate_path_safety__mutmut_8(
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


def x_validate_path_safety__mutmut_9(
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


def x_validate_path_safety__mutmut_10(
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
            "XXPath contains null byteXX",
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


def x_validate_path_safety__mutmut_11(
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
            "path contains null byte",
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


def x_validate_path_safety__mutmut_12(
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
            "PATH CONTAINS NULL BYTE",
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


def x_validate_path_safety__mutmut_13(
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
            path=path_str[:101],  # Truncate for safety
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


def x_validate_path_safety__mutmut_14(
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
    if len(path_str) >= max_length:
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


def x_validate_path_safety__mutmut_15(
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
            None,
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


def x_validate_path_safety__mutmut_16(
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
            None,
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


def x_validate_path_safety__mutmut_17(
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
            length=None,
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


def x_validate_path_safety__mutmut_18(
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


def x_validate_path_safety__mutmut_19(
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


def x_validate_path_safety__mutmut_20(
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


def x_validate_path_safety__mutmut_21(
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
    if base_dir is None:
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


def x_validate_path_safety__mutmut_22(
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
        path = None
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


def x_validate_path_safety__mutmut_23(
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
        path = Path(None)
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


def x_validate_path_safety__mutmut_24(
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
                None,
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


def x_validate_path_safety__mutmut_25(
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
                None,
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


def x_validate_path_safety__mutmut_26(
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
                path=None,
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


def x_validate_path_safety__mutmut_27(
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


def x_validate_path_safety__mutmut_28(
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


def x_validate_path_safety__mutmut_29(
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


def x_validate_path_safety__mutmut_30(
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
                "XXAbsolute path not allowedXX",
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


def x_validate_path_safety__mutmut_31(
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
                "absolute path not allowed",
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


def x_validate_path_safety__mutmut_32(
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
                "ABSOLUTE PATH NOT ALLOWED",
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


def x_validate_path_safety__mutmut_33(
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
                path=path_str[:101],
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


def x_validate_path_safety__mutmut_34(
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
            base = None
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


def x_validate_path_safety__mutmut_35(
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
            base = Path(None).resolve()
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


def x_validate_path_safety__mutmut_36(
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
            resolved = None
            
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


def x_validate_path_safety__mutmut_37(
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
            resolved = (base * path).resolve()
            
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


def x_validate_path_safety__mutmut_38(
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
            
            if str(resolved).startswith(str(base)):
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


def x_validate_path_safety__mutmut_39(
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
            
            if not str(resolved).startswith(None):
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


def x_validate_path_safety__mutmut_40(
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
            
            if not str(None).startswith(str(base)):
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


def x_validate_path_safety__mutmut_41(
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
            
            if not str(resolved).startswith(str(None)):
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


def x_validate_path_safety__mutmut_42(
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
                    None,
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


def x_validate_path_safety__mutmut_43(
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
                    None,
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


def x_validate_path_safety__mutmut_44(
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
                    path=None,
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


def x_validate_path_safety__mutmut_45(
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
                    base=None,
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


def x_validate_path_safety__mutmut_46(
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


def x_validate_path_safety__mutmut_47(
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


def x_validate_path_safety__mutmut_48(
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


def x_validate_path_safety__mutmut_49(
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


def x_validate_path_safety__mutmut_50(
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
                    "XXPath escapes base directoryXX",
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


def x_validate_path_safety__mutmut_51(
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
                    "path escapes base directory",
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


def x_validate_path_safety__mutmut_52(
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
                    "PATH ESCAPES BASE DIRECTORY",
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


def x_validate_path_safety__mutmut_53(
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
                    path=path_str[:101],
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


def x_validate_path_safety__mutmut_54(
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
                    base=str(None),
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


def x_validate_path_safety__mutmut_55(
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
                None,
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


def x_validate_path_safety__mutmut_56(
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
                None,
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


def x_validate_path_safety__mutmut_57(
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
                path=None,
            )
    
    # Check for suspicious patterns
    if ".." in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_58(
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


def x_validate_path_safety__mutmut_59(
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


def x_validate_path_safety__mutmut_60(
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
                )
    
    # Check for suspicious patterns
    if ".." in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_61(
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
                path=path_str[:101],
            )
    
    # Check for suspicious patterns
    if ".." in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_62(
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
    if "XX..XX" in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_63(
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
    if ".." not in path_str:
        return VerificationReport.fail(
            VerificationErrorCode.PATH_TRAVERSAL,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_64(
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
            None,
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_65(
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
            None,
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_66(
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
            path=None,
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_67(
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
            "Path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_68(
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
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_69(
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
            )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_70(
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
            "XXPath contains parent directory referenceXX",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_71(
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
            "path contains parent directory reference",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_72(
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
            "PATH CONTAINS PARENT DIRECTORY REFERENCE",
            path=path_str[:100],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_73(
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
            path=path_str[:101],
        )
    
    return VerificationReport.ok("Path is safe")


def x_validate_path_safety__mutmut_74(
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
    
    return VerificationReport.ok(None)


def x_validate_path_safety__mutmut_75(
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
    
    return VerificationReport.ok("XXPath is safeXX")


def x_validate_path_safety__mutmut_76(
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
    
    return VerificationReport.ok("path is safe")


def x_validate_path_safety__mutmut_77(
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
    
    return VerificationReport.ok("PATH IS SAFE")

x_validate_path_safety__mutmut_mutants : ClassVar[MutantDict] = {
'x_validate_path_safety__mutmut_1': x_validate_path_safety__mutmut_1, 
    'x_validate_path_safety__mutmut_2': x_validate_path_safety__mutmut_2, 
    'x_validate_path_safety__mutmut_3': x_validate_path_safety__mutmut_3, 
    'x_validate_path_safety__mutmut_4': x_validate_path_safety__mutmut_4, 
    'x_validate_path_safety__mutmut_5': x_validate_path_safety__mutmut_5, 
    'x_validate_path_safety__mutmut_6': x_validate_path_safety__mutmut_6, 
    'x_validate_path_safety__mutmut_7': x_validate_path_safety__mutmut_7, 
    'x_validate_path_safety__mutmut_8': x_validate_path_safety__mutmut_8, 
    'x_validate_path_safety__mutmut_9': x_validate_path_safety__mutmut_9, 
    'x_validate_path_safety__mutmut_10': x_validate_path_safety__mutmut_10, 
    'x_validate_path_safety__mutmut_11': x_validate_path_safety__mutmut_11, 
    'x_validate_path_safety__mutmut_12': x_validate_path_safety__mutmut_12, 
    'x_validate_path_safety__mutmut_13': x_validate_path_safety__mutmut_13, 
    'x_validate_path_safety__mutmut_14': x_validate_path_safety__mutmut_14, 
    'x_validate_path_safety__mutmut_15': x_validate_path_safety__mutmut_15, 
    'x_validate_path_safety__mutmut_16': x_validate_path_safety__mutmut_16, 
    'x_validate_path_safety__mutmut_17': x_validate_path_safety__mutmut_17, 
    'x_validate_path_safety__mutmut_18': x_validate_path_safety__mutmut_18, 
    'x_validate_path_safety__mutmut_19': x_validate_path_safety__mutmut_19, 
    'x_validate_path_safety__mutmut_20': x_validate_path_safety__mutmut_20, 
    'x_validate_path_safety__mutmut_21': x_validate_path_safety__mutmut_21, 
    'x_validate_path_safety__mutmut_22': x_validate_path_safety__mutmut_22, 
    'x_validate_path_safety__mutmut_23': x_validate_path_safety__mutmut_23, 
    'x_validate_path_safety__mutmut_24': x_validate_path_safety__mutmut_24, 
    'x_validate_path_safety__mutmut_25': x_validate_path_safety__mutmut_25, 
    'x_validate_path_safety__mutmut_26': x_validate_path_safety__mutmut_26, 
    'x_validate_path_safety__mutmut_27': x_validate_path_safety__mutmut_27, 
    'x_validate_path_safety__mutmut_28': x_validate_path_safety__mutmut_28, 
    'x_validate_path_safety__mutmut_29': x_validate_path_safety__mutmut_29, 
    'x_validate_path_safety__mutmut_30': x_validate_path_safety__mutmut_30, 
    'x_validate_path_safety__mutmut_31': x_validate_path_safety__mutmut_31, 
    'x_validate_path_safety__mutmut_32': x_validate_path_safety__mutmut_32, 
    'x_validate_path_safety__mutmut_33': x_validate_path_safety__mutmut_33, 
    'x_validate_path_safety__mutmut_34': x_validate_path_safety__mutmut_34, 
    'x_validate_path_safety__mutmut_35': x_validate_path_safety__mutmut_35, 
    'x_validate_path_safety__mutmut_36': x_validate_path_safety__mutmut_36, 
    'x_validate_path_safety__mutmut_37': x_validate_path_safety__mutmut_37, 
    'x_validate_path_safety__mutmut_38': x_validate_path_safety__mutmut_38, 
    'x_validate_path_safety__mutmut_39': x_validate_path_safety__mutmut_39, 
    'x_validate_path_safety__mutmut_40': x_validate_path_safety__mutmut_40, 
    'x_validate_path_safety__mutmut_41': x_validate_path_safety__mutmut_41, 
    'x_validate_path_safety__mutmut_42': x_validate_path_safety__mutmut_42, 
    'x_validate_path_safety__mutmut_43': x_validate_path_safety__mutmut_43, 
    'x_validate_path_safety__mutmut_44': x_validate_path_safety__mutmut_44, 
    'x_validate_path_safety__mutmut_45': x_validate_path_safety__mutmut_45, 
    'x_validate_path_safety__mutmut_46': x_validate_path_safety__mutmut_46, 
    'x_validate_path_safety__mutmut_47': x_validate_path_safety__mutmut_47, 
    'x_validate_path_safety__mutmut_48': x_validate_path_safety__mutmut_48, 
    'x_validate_path_safety__mutmut_49': x_validate_path_safety__mutmut_49, 
    'x_validate_path_safety__mutmut_50': x_validate_path_safety__mutmut_50, 
    'x_validate_path_safety__mutmut_51': x_validate_path_safety__mutmut_51, 
    'x_validate_path_safety__mutmut_52': x_validate_path_safety__mutmut_52, 
    'x_validate_path_safety__mutmut_53': x_validate_path_safety__mutmut_53, 
    'x_validate_path_safety__mutmut_54': x_validate_path_safety__mutmut_54, 
    'x_validate_path_safety__mutmut_55': x_validate_path_safety__mutmut_55, 
    'x_validate_path_safety__mutmut_56': x_validate_path_safety__mutmut_56, 
    'x_validate_path_safety__mutmut_57': x_validate_path_safety__mutmut_57, 
    'x_validate_path_safety__mutmut_58': x_validate_path_safety__mutmut_58, 
    'x_validate_path_safety__mutmut_59': x_validate_path_safety__mutmut_59, 
    'x_validate_path_safety__mutmut_60': x_validate_path_safety__mutmut_60, 
    'x_validate_path_safety__mutmut_61': x_validate_path_safety__mutmut_61, 
    'x_validate_path_safety__mutmut_62': x_validate_path_safety__mutmut_62, 
    'x_validate_path_safety__mutmut_63': x_validate_path_safety__mutmut_63, 
    'x_validate_path_safety__mutmut_64': x_validate_path_safety__mutmut_64, 
    'x_validate_path_safety__mutmut_65': x_validate_path_safety__mutmut_65, 
    'x_validate_path_safety__mutmut_66': x_validate_path_safety__mutmut_66, 
    'x_validate_path_safety__mutmut_67': x_validate_path_safety__mutmut_67, 
    'x_validate_path_safety__mutmut_68': x_validate_path_safety__mutmut_68, 
    'x_validate_path_safety__mutmut_69': x_validate_path_safety__mutmut_69, 
    'x_validate_path_safety__mutmut_70': x_validate_path_safety__mutmut_70, 
    'x_validate_path_safety__mutmut_71': x_validate_path_safety__mutmut_71, 
    'x_validate_path_safety__mutmut_72': x_validate_path_safety__mutmut_72, 
    'x_validate_path_safety__mutmut_73': x_validate_path_safety__mutmut_73, 
    'x_validate_path_safety__mutmut_74': x_validate_path_safety__mutmut_74, 
    'x_validate_path_safety__mutmut_75': x_validate_path_safety__mutmut_75, 
    'x_validate_path_safety__mutmut_76': x_validate_path_safety__mutmut_76, 
    'x_validate_path_safety__mutmut_77': x_validate_path_safety__mutmut_77
}

def validate_path_safety(*args, **kwargs):
    result = _mutmut_trampoline(x_validate_path_safety__mutmut_orig, x_validate_path_safety__mutmut_mutants, args, kwargs)
    return result 

validate_path_safety.__signature__ = _mutmut_signature(x_validate_path_safety__mutmut_orig)
x_validate_path_safety__mutmut_orig.__name__ = 'x_validate_path_safety'
