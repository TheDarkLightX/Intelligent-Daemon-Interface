"""
IAN Resilience Utilities - Circuit breakers, retries, and structured logging.

This module provides production-grade resilience patterns:
1. CircuitBreaker - Prevents cascading failures from external services
2. StructuredLogger - JSON logging with correlation IDs
3. RetryWithCircuitBreaker - Combined retry + circuit breaker pattern

Circuit Breaker States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered

Usage:
    # Circuit breaker
    breaker = CircuitBreaker(name="tau_api", failure_threshold=5)
    
    async with breaker:
        result = await call_external_api()
    
    # Structured logging
    logger = StructuredLogger("idi.ian.node")
    logger.info("contribution_received", pack_hash=hash, size=size)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Correlation Context
# =============================================================================

# Context variable for request correlation ID
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    'correlation_id',
    default='',
)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID."""
    cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Rejecting requests
    HALF_OPEN = auto()   # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, until: float):
        self.name = name
        self.until = until
        wait_time = max(0, until - time.time())
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry after {wait_time:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes in half-open before closing
    timeout: float = 30.0            # Seconds to wait before half-open
    half_open_max_calls: int = 3     # Max concurrent calls in half-open


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_rejections: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.
    
    Prevents cascading failures by failing fast when a service is down.
    
    States:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Service might be recovering, limited calls allowed
    
    Example:
        breaker = CircuitBreaker(name="tau_net", failure_threshold=5)
        
        try:
            async with breaker:
                result = await send_to_tau(tx)
        except CircuitBreakerError:
            # Service is down, use fallback
            result = use_cached_value()
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0
        self._lock = asyncio.Lock()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats
    
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() >= self._opened_at + self._config.timeout:
                return False  # Will transition to half-open
            return True
        return False
    
    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry - check if call is allowed."""
        await self._before_call()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit - record result."""
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions
    
    async def _before_call(self) -> None:
        """Check if call is allowed, raise if circuit is open."""
        async with self._lock:
            self._stats.total_calls += 1
            
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if time.time() >= self._opened_at + self._config.timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._stats.total_rejections += 1
                    raise CircuitBreakerError(
                        self._name,
                        self._opened_at + self._config.timeout,
                    )
            
            if self._state == CircuitState.HALF_OPEN:
                # Limit concurrent calls in half-open
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._stats.total_rejections += 1
                    raise CircuitBreakerError(
                        self._name,
                        time.time() + 1.0,  # Short wait
                    )
                self._half_open_calls += 1
    
    async def _on_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                
                # Check if we should close the circuit
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                # Immediately open on failure in half-open
                self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._stats.consecutive_failures >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1
        
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            logger.warning(
                f"Circuit breaker '{self._name}' OPENED after "
                f"{self._stats.consecutive_failures} failures"
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self._name}' entering HALF_OPEN")
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            logger.info(f"Circuit breaker '{self._name}' CLOSED (recovered)")
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self._name}' manually reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary."""
        return {
            "name": self._name,
            "state": self._state.name,
            "stats": {
                "total_calls": self._stats.total_calls,
                "total_failures": self._stats.total_failures,
                "total_successes": self._stats.total_successes,
                "total_rejections": self._stats.total_rejections,
                "consecutive_failures": self._stats.consecutive_failures,
            },
        }


# Registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str):
    """
    Decorator to wrap async function with circuit breaker.
    
    Example:
        @circuit_breaker("tau_api")
        async def send_to_tau(tx):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(name)
            async with breaker:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs as JSON with standard fields:
    - timestamp
    - level
    - logger
    - message
    - correlation_id
    - (additional context fields)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID
        cid = correlation_id_var.get()
        if cid:
            log_data["correlation_id"] = cid
        
        # Add extra fields from record
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger with context and correlation IDs.
    
    Provides methods for logging with additional context fields
    that are automatically included in the JSON output.
    
    Example:
        logger = StructuredLogger("idi.ian.node")
        
        # Simple log
        logger.info("Node started")
        
        # Log with context
        logger.info("contribution_received", 
            pack_hash="abc123",
            size=1024,
            contributor="node456"
        )
        
        # With correlation ID
        with logger.correlation_context("req-123"):
            logger.info("Processing request")
            # All logs in this context share the correlation ID
    """
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def _log(
        self,
        level: int,
        event: str,
        **kwargs,
    ) -> None:
        """Internal logging method."""
        # Merge context with kwargs
        data = {**self._context, **kwargs}
        
        # Create record with structured data
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event,
            args=(),
            exc_info=None,
        )
        record.structured_data = data
        
        self._logger.handle(record)
    
    def debug(self, event: str, **kwargs) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, event, **kwargs)
    
    def info(self, event: str, **kwargs) -> None:
        """Log info message with context."""
        self._log(logging.INFO, event, **kwargs)
    
    def warning(self, event: str, **kwargs) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, event, **kwargs)
    
    def error(self, event: str, **kwargs) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, event, **kwargs)
    
    def critical(self, event: str, **kwargs) -> None:
        """Log critical message with context."""
        self._log(logging.CRITICAL, event, **kwargs)
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """
        Return a new logger with additional context bound.
        
        Example:
            node_logger = logger.bind(node_id="abc123", goal_id="goal1")
            node_logger.info("started")  # Includes node_id and goal_id
        """
        new_logger = StructuredLogger(self._logger.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    class correlation_context:
        """Context manager for correlation ID."""
        
        def __init__(self, cid: Optional[str] = None):
            self._cid = cid or str(uuid.uuid4())[:8]
            self._token = None
        
        def __enter__(self):
            self._token = correlation_id_var.set(self._cid)
            return self._cid
        
        def __exit__(self, *args):
            correlation_id_var.reset(self._token)


def setup_structured_logging(
    level: int = logging.INFO,
    json_output: bool = True,
) -> None:
    """
    Configure root logger for structured logging.
    
    Args:
        level: Logging level
        json_output: If True, use JSON formatter
    """
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
    
    root.addHandler(handler)


# =============================================================================
# Retry with Circuit Breaker
# =============================================================================

T = TypeVar('T')


async def retry_with_circuit_breaker(
    func: Callable[[], T],
    breaker_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
) -> T:
    """
    Execute function with retry logic and circuit breaker.
    
    Combines exponential backoff retry with circuit breaker pattern.
    
    Args:
        func: Async function to execute
        breaker_name: Name of circuit breaker to use
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exceptions to retry on
        
    Returns:
        Result of func
        
    Raises:
        CircuitBreakerError: If circuit is open
        Last exception: If all retries exhausted
    """
    import random
    
    breaker = get_circuit_breaker(breaker_name)
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            async with breaker:
                return await func()
                
        except CircuitBreakerError:
            # Don't retry if circuit is open
            raise
            
        except retryable_exceptions as e:
            last_error = e
            
            if attempt < max_retries - 1:
                # Calculate delay with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * 0.3 * random.uniform(-1, 1)
                await asyncio.sleep(max(0.1, delay + jitter))
    
    # All retries exhausted
    raise last_error


# =============================================================================
# Bulkhead Pattern
# =============================================================================

class Bulkhead:
    """
    Bulkhead pattern to limit concurrent calls to a resource.
    
    Prevents resource exhaustion by limiting parallelism.
    
    Example:
        bulkhead = Bulkhead(name="db", max_concurrent=10)
        
        async with bulkhead:
            result = await db_query()
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_waiting: int = 100,
        timeout: float = 30.0,
    ):
        self._name = name
        self._max_concurrent = max_concurrent
        self._max_waiting = max_waiting
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._active = 0
        self._total_calls = 0
        self._rejections = 0
    
    async def __aenter__(self) -> "Bulkhead":
        """Acquire slot in bulkhead."""
        self._total_calls += 1
        
        # Check if too many waiting
        if self._waiting >= self._max_waiting:
            self._rejections += 1
            raise RuntimeError(
                f"Bulkhead '{self._name}' queue full: "
                f"{self._waiting} waiting"
            )
        
        self._waiting += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            self._waiting -= 1
            self._rejections += 1
            raise RuntimeError(
                f"Bulkhead '{self._name}' timeout after {self._timeout}s"
            )
        
        self._waiting -= 1
        self._active += 1
        return self
    
    async def __aexit__(self, *args) -> None:
        """Release slot in bulkhead."""
        self._active -= 1
        self._semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self._name,
            "active": self._active,
            "waiting": self._waiting,
            "max_concurrent": self._max_concurrent,
            "total_calls": self._total_calls,
            "rejections": self._rejections,
        }


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================

class AsyncRateLimiter:
    """
    Async-safe token bucket rate limiter.
    
    Example:
        limiter = AsyncRateLimiter(rate=10.0, burst=5)
        
        async with limiter:
            await make_api_call()
    """
    
    def __init__(
        self,
        rate: float,
        burst: int = 1,
    ):
        """
        Args:
            rate: Tokens per second
            burst: Maximum tokens (bucket size)
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Returns:
            Wait time in seconds
        """
        async with self._lock:
            # Refill tokens
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                self._burst,
                self._tokens + elapsed * self._rate,
            )
            self._last_update = now
            
            # Calculate wait time
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            
            # Need to wait
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate
            self._tokens = 0
            
            return wait_time
    
    async def __aenter__(self) -> "AsyncRateLimiter":
        wait_time = await self.acquire()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        return self
    
    async def __aexit__(self, *args) -> None:
        pass


# =============================================================================
# Timeout Wrapper
# =============================================================================

async def with_timeout(
    coro,
    timeout: float,
    error_message: str = "Operation timed out",
) -> Any:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        error_message: Error message on timeout
        
    Returns:
        Result of coroutine
        
    Raises:
        TimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(error_message)


# =============================================================================
# Health Aggregator
# =============================================================================

@dataclass
class DependencyHealth:
    """Health status of a dependency."""
    name: str
    healthy: bool
    latency_ms: float = 0.0
    last_check: float = 0.0
    error: Optional[str] = None
    circuit_state: Optional[str] = None


class HealthAggregator:
    """
    Aggregates health status of multiple dependencies.
    
    Example:
        health = HealthAggregator()
        health.register("tau_net", check_tau_health)
        health.register("database", check_db_health)
        
        status = await health.check_all()
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._async_checks: Dict[str, Callable] = {}
        self._results: Dict[str, DependencyHealth] = {}
    
    def register(
        self,
        name: str,
        check: Callable,
        is_async: bool = False,
    ) -> None:
        """Register a health check."""
        if is_async:
            self._async_checks[name] = check
        else:
            self._checks[name] = check
    
    async def check_all(
        self,
        timeout: float = 5.0,
    ) -> Dict[str, DependencyHealth]:
        """Run all health checks."""
        results = {}
        
        # Run sync checks
        for name, check in self._checks.items():
            start = time.time()
            try:
                healthy = check()
                results[name] = DependencyHealth(
                    name=name,
                    healthy=healthy,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                )
            except Exception as e:
                results[name] = DependencyHealth(
                    name=name,
                    healthy=False,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                    error=str(e),
                )
        
        # Run async checks concurrently
        async def run_async_check(name: str, check) -> DependencyHealth:
            start = time.time()
            try:
                healthy = await asyncio.wait_for(check(), timeout=timeout)
                return DependencyHealth(
                    name=name,
                    healthy=healthy,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                )
            except Exception as e:
                return DependencyHealth(
                    name=name,
                    healthy=False,
                    latency_ms=(time.time() - start) * 1000,
                    last_check=time.time(),
                    error=str(e),
                )
        
        if self._async_checks:
            async_results = await asyncio.gather(*[
                run_async_check(name, check)
                for name, check in self._async_checks.items()
            ])
            for result in async_results:
                results[result.name] = result
        
        # Add circuit breaker states
        for name, breaker in _circuit_breakers.items():
            if name in results:
                results[name].circuit_state = breaker.state.name
        
        self._results = results
        return results
    
    def is_healthy(self) -> bool:
        """Check if all dependencies are healthy."""
        return all(r.healthy for r in self._results.values())
    
    def get_unhealthy(self) -> List[str]:
        """Get list of unhealthy dependencies."""
        return [
            name for name, result in self._results.items()
            if not result.healthy
        ]
