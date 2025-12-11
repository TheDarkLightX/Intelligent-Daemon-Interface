"""
IAN Observability Module - Logging, metrics, and tracing.

Provides:
1. Structured JSON logging
2. Prometheus-compatible metrics
3. Contribution flow tracing
4. Performance monitoring

Usage:
    from idi.ian.observability import setup_logging, metrics
    
    # Setup logging
    setup_logging(format="json", level="INFO")
    
    # Record metrics
    metrics.contributions_total.inc()
    metrics.evaluation_duration.observe(0.5)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .config import LoggingConfig, MetricsConfig


# =============================================================================
# Structured Logging
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Output format:
    {
        "timestamp": "2025-12-10T15:30:00.000Z",
        "level": "INFO",
        "logger": "idi.ian.coordinator",
        "message": "Processing contribution",
        "context": {...}
    }
    """
    
    def __init__(self, include_timestamps: bool = True):
        super().__init__()
        self._include_timestamps = include_timestamps
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if self._include_timestamps:
            log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Include extra context
        if hasattr(record, "context") and record.context:
            log_entry["context"] = record.context
        
        # Include exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Include source location in debug mode
        if record.levelno <= logging.DEBUG:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        return json.dumps(log_entry)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to log records."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.get("extra", {})
        extra["context"] = {**self.extra, **extra.get("context", {})}
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    config: Optional[LoggingConfig] = None,
    format: str = "json",
    level: str = "INFO",
    file: Optional[str] = None,
) -> None:
    """
    Setup IAN logging configuration.
    
    Args:
        config: LoggingConfig object (overrides other args)
        format: "json" or "text"
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file: Optional log file path
    """
    if config:
        format = config.format
        level = config.level
        file = config.file
    
    # Derive file rotation settings
    max_bytes = 100 * 1024 * 1024  # 100 MB default
    backup_count = 5
    if config:
        max_bytes = config.max_size_mb * 1024 * 1024
        backup_count = config.backup_count
    
    # Get root logger for IAN
    logger = logging.getLogger("idi.ian")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if format == "json":
        formatter = JSONFormatter(include_timestamps=True)
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info("Logging initialized", extra={"context": {"format": format, "level": level}})


def get_logger(name: str, **context) -> ContextAdapter:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (will be prefixed with idi.ian.)
        **context: Additional context to include in all log messages
        
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(f"idi.ian.{name}")
    return ContextAdapter(logger, context)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = 0
    
    def inc(self, amount: float = 1) -> None:
        """Increment counter."""
        self._value += amount
    
    @property
    def value(self) -> float:
        return self._value
    
    def to_prometheus(self) -> str:
        """Format as Prometheus metric."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if labels_str:
            return f"{self.name}{{{labels_str}}} {self._value}"
        return f"{self.name} {self._value}"


@dataclass
class Gauge:
    """Simple gauge metric."""
    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = 0
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value
    
    def inc(self, amount: float = 1) -> None:
        """Increment gauge."""
        self._value += amount
    
    def dec(self, amount: float = 1) -> None:
        """Decrement gauge."""
        self._value -= amount
    
    @property
    def value(self) -> float:
        return self._value
    
    def to_prometheus(self) -> str:
        """Format as Prometheus metric."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if labels_str:
            return f"{self.name}{{{labels_str}}} {self._value}"
        return f"{self.name} {self._value}"


@dataclass
class Histogram:
    """Simple histogram metric."""
    name: str
    help: str
    buckets: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    labels: Dict[str, str] = field(default_factory=dict)
    _sum: float = 0
    _count: int = 0
    _bucket_counts: Dict[float, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        self._sum += value
        self._count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
        self._bucket_counts[float('inf')] += 1
    
    @property
    def sum(self) -> float:
        return self._sum
    
    @property
    def count(self) -> int:
        return self._count
    
    def to_prometheus(self) -> str:
        """Format as Prometheus metric."""
        lines = []
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        
        for bucket, count in sorted(self._bucket_counts.items()):
            le = "+Inf" if bucket == float('inf') else str(bucket)
            if labels_str:
                lines.append(f'{self.name}_bucket{{{labels_str},le="{le}"}} {count}')
            else:
                lines.append(f'{self.name}_bucket{{le="{le}"}} {count}')
        
        if labels_str:
            lines.append(f"{self.name}_sum{{{labels_str}}} {self._sum}")
            lines.append(f"{self.name}_count{{{labels_str}}} {self._count}")
        else:
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        
        return "\n".join(lines)


class IANMetrics:
    """
    IAN metrics collection.
    
    Provides Prometheus-compatible metrics for monitoring.
    """
    
    def __init__(self, namespace: str = "ian"):
        self._namespace = namespace
        
        # Contribution metrics
        self.contributions_total = Counter(
            f"{namespace}_contributions_total",
            "Total contributions processed",
        )
        self.contributions_accepted = Counter(
            f"{namespace}_contributions_accepted_total",
            "Total contributions accepted",
        )
        self.contributions_rejected = Counter(
            f"{namespace}_contributions_rejected_total",
            "Total contributions rejected",
        )
        
        # Rejection reason metrics
        self.rejections_by_reason: Dict[str, Counter] = {}
        
        # Performance metrics
        self.processing_duration = Histogram(
            f"{namespace}_processing_duration_seconds",
            "Contribution processing duration in seconds",
        )
        self.evaluation_duration = Histogram(
            f"{namespace}_evaluation_duration_seconds",
            "Agent evaluation duration in seconds",
        )
        
        # State metrics
        self.leaderboard_size = Gauge(
            f"{namespace}_leaderboard_size",
            "Current leaderboard size",
        )
        self.log_size = Gauge(
            f"{namespace}_log_size",
            "Current contribution log size",
        )
        self.active_goals = Gauge(
            f"{namespace}_active_goals",
            "Number of active goals",
        )
        
        # Rate limiting metrics
        self.rate_limited_total = Counter(
            f"{namespace}_rate_limited_total",
            "Total rate-limited requests",
        )
        
        # Tau metrics
        self.tau_transactions_total = Counter(
            f"{namespace}_tau_transactions_total",
            "Total Tau transactions sent",
        )
        self.tau_transactions_failed = Counter(
            f"{namespace}_tau_transactions_failed_total",
            "Total failed Tau transactions",
        )
    
    def record_rejection(self, reason: str) -> None:
        """Record a rejection by reason."""
        if reason not in self.rejections_by_reason:
            self.rejections_by_reason[reason] = Counter(
                f"{self._namespace}_rejections_total",
                f"Rejections by reason",
                labels={"reason": reason},
            )
        self.rejections_by_reason[reason].inc()
    
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        
        # Add all metrics
        metrics = [
            self.contributions_total,
            self.contributions_accepted,
            self.contributions_rejected,
            self.processing_duration,
            self.evaluation_duration,
            self.leaderboard_size,
            self.log_size,
            self.active_goals,
            self.rate_limited_total,
            self.tau_transactions_total,
            self.tau_transactions_failed,
        ]
        
        for metric in metrics:
            lines.append(f"# HELP {metric.name} {metric.help}")
            lines.append(f"# TYPE {metric.name} {'histogram' if isinstance(metric, Histogram) else 'counter' if isinstance(metric, Counter) else 'gauge'}")
            lines.append(metric.to_prometheus())
            lines.append("")
        
        # Add rejection reasons
        for counter in self.rejections_by_reason.values():
            lines.append(counter.to_prometheus())
        
        return "\n".join(lines)


# Global metrics instance
metrics = IANMetrics()


# =============================================================================
# Tracing
# =============================================================================

@dataclass
class Span:
    """A tracing span for measuring operations."""
    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.monotonic(),
            "attributes": attributes,
        })
    
    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status_message"] = message
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.monotonic()
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000


class Tracer:
    """
    Simple tracer for contribution flow.
    
    Not a full OpenTelemetry implementation, but provides basic tracing.
    """
    
    def __init__(self, service_name: str = "ian"):
        self._service_name = service_name
        self._current_span: Optional[Span] = None
        self._spans: List[Span] = []
    
    @contextmanager
    def span(self, name: str, **attributes):
        """Create a new span."""
        span = Span(name=name, attributes=attributes)
        parent = self._current_span
        self._current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            raise
        finally:
            span.end()
            self._current_span = parent
            self._spans.append(span)
    
    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans."""
        return self._spans[-limit:]
    
    def clear(self) -> None:
        """Clear recorded spans."""
        self._spans.clear()


# Global tracer instance
tracer = Tracer()


# =============================================================================
# Decorators
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def timed(metric: Optional[Histogram] = None) -> Callable[[F], F]:
    """
    Decorator to time function execution.
    
    Args:
        metric: Histogram to record duration (optional)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.monotonic() - start
                if metric:
                    metric.observe(duration)
        return wrapper  # type: ignore
    return decorator


def traced(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace function execution.
    
    Args:
        name: Span name (defaults to function name)
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.span(span_name):
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator
