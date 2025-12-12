from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class LoggingOptions:
    level: str = "INFO"
    format: str = "text"  # text | json
    file: Optional[str] = None
    redact: bool = True


_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "private_key",
    "secret",
    "token",
)

_RE_KV = re.compile(
    r"(?P<key>api[_-]?key|token|password|secret|authorization)\s*[:=]\s*(?P<value>[^\s,;]+)",
    flags=re.IGNORECASE,
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context is not None:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def _redact_str(value: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        key = match.group("key")
        return f"{key}=[REDACTED]"

    return _RE_KV.sub(_sub, value)


def _redact_any(value: Any, *, depth: int, max_depth: int) -> Any:
    """Redact secret-like values in nested structures.

    Preconditions:
        - max_depth >= 0

    Postconditions:
        - Returns a JSON-serializable structure when input is JSON-serializable
        - Secret-like keys have their values replaced with "[REDACTED]"

    Invariants:
        - Does not recurse beyond max_depth

    Complexity:
        - O(n) in size of traversed structure, bounded by max_depth
    """
    if depth > max_depth:
        return "[REDACTED]"

    if isinstance(value, str):
        return _redact_str(value)

    if isinstance(value, bytes):
        return "[REDACTED]"

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and _looks_secret_key(k):
                redacted[k] = "[REDACTED]"
                continue
            redacted[k] = _redact_any(v, depth=depth + 1, max_depth=max_depth)
        return redacted

    if isinstance(value, (list, tuple)):
        return [_redact_any(v, depth=depth + 1, max_depth=max_depth) for v in value]

    return value


class RedactionFilter(logging.Filter):
    def __init__(self, *, max_depth: int = 4):
        super().__init__()
        self._max_depth = max_depth

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact_str(record.msg)

        context = getattr(record, "context", None)
        if isinstance(context, MutableMapping):
            record.context = _redact_any(context, depth=0, max_depth=self._max_depth)

        return True


def _normalize_level(level: str) -> str:
    return level.strip().upper()


def _normalize_format(fmt: str) -> str:
    lowered = fmt.strip().lower()
    if lowered in {"text", "json"}:
        return lowered
    raise ValueError(f"Invalid log format: {fmt}")


def load_logging_options_from_env() -> LoggingOptions:
    """Load logging options from environment.

    Env vars:
        - IDI_LOG_LEVEL
        - IDI_LOG_FORMAT
        - IDI_LOG_FILE
        - IDI_LOG_REDACT ("0" disables redaction)
    """
    level = os.getenv("IDI_LOG_LEVEL", "INFO")
    fmt = os.getenv("IDI_LOG_FORMAT", "text")
    file = os.getenv("IDI_LOG_FILE")
    redact_env = os.getenv("IDI_LOG_REDACT", "1")
    redact = redact_env not in {"0", "false", "FALSE"}
    return LoggingOptions(level=level, format=fmt, file=file, redact=redact)


def configure_logging(options: LoggingOptions) -> None:
    """Configure logging for IDI.

    Preconditions:
        - options.level is a valid logging level name
        - options.format in {"text", "json"}

    Postconditions:
        - Logger hierarchy under "idi" is configured
        - Logs emit to stderr (and optional rotating file)
        - Redaction filter attached unless options.redact is False
    """
    logger = logging.getLogger("idi")
    logger.setLevel(getattr(logging, _normalize_level(options.level), logging.INFO))

    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if _normalize_format(options.format) == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    if options.redact:
        handler.addFilter(RedactionFilter())

    logger.addHandler(handler)

    if options.file:
        file_handler = RotatingFileHandler(
            options.file,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        if _normalize_format(options.format) == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        if options.redact:
            file_handler.addFilter(RedactionFilter())
        logger.addHandler(file_handler)
