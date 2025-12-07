"""Secure error handling utilities.

Ensures that internal implementation details are never exposed to users,
preventing information leakage that could aid attackers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


class SecureError(Exception):
    """Base class for secure errors that don't leak internal details."""

    def __init__(self, user_message: str, internal_details: Optional[str] = None):
        """
        Args:
            user_message: Safe message to show to users
            internal_details: Internal details for logging only
        """
        super().__init__(user_message)
        self.user_message = user_message
        self.internal_details = internal_details

        # Log internal details securely (never expose to user)
        if internal_details:
            logging.warning(f"SecureError internal: {internal_details}")


def handle_file_operation_error(operation: str, path: str, exc: Exception) -> SecureError:
    """Convert file operation errors to secure errors."""
    return SecureError(
        f"File operation failed: {operation}",
        f"File operation '{operation}' failed on path '{path}': {type(exc).__name__}: {exc}"
    )


def handle_json_parse_error(data_source: str, exc: Exception) -> SecureError:
    """Convert JSON parsing errors to secure errors."""
    return SecureError(
        f"Invalid data format in {data_source}",
        f"JSON parse error in {data_source}: {type(exc).__name__}: {exc}"
    )


def handle_validation_error(field: str, value: Any = None) -> SecureError:
    """Convert validation errors to secure errors."""
    if value is not None:
        # Never expose the actual invalid value in user message
        return SecureError(
            f"Invalid value provided for {field}",
            f"Validation failed for {field}: invalid value '{value}'"
        )
    else:
        return SecureError(
            f"Validation failed for {field}",
            f"Validation failed for {field}: unspecified error"
        )


def safe_log_error(message: str, exc: Optional[Exception] = None, **context: Any) -> None:
    """Log errors securely without exposing sensitive information."""
    if exc:
        logging.error(f"{message} | Exception: {type(exc).__name__} | Context: {context}")
    else:
        logging.error(f"{message} | Context: {context}")
