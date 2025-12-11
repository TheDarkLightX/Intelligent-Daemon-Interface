"""
IAN Production Utilities - Task supervision, metrics, health, and resilience.

This module provides production-grade infrastructure for running IAN nodes:
1. TaskSupervisor - Supervised async tasks with automatic restart
2. NodeMetrics - Prometheus-compatible metrics collection
3. HealthServer - HTTP health and readiness endpoints
4. PeerScoring - Peer reputation tracking
5. Backoff utilities - Exponential backoff with jitter

Usage:
    supervisor = TaskSupervisor()
    supervisor.spawn(my_coroutine, name="worker", restart_on_failure=True)
    await supervisor.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Backoff Utilities
# =============================================================================

def backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.3,
) -> float:
    """
    Calculate exponential backoff with jitter.
    
    Prevents thundering herd by adding randomness to retry delays.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter_factor: Fraction of delay to add as jitter (0.0-1.0)
        
    Returns:
        Delay in seconds with jitter applied
        
    Example:
        for attempt in range(5):
            delay = backoff_with_jitter(attempt)
            await asyncio.sleep(delay)
    """
    # Exponential: 1, 2, 4, 8, 16, ...
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    # Add jitter: ±30% by default
    jitter = delay * jitter_factor * random.uniform(-1, 1)
    
    return max(0.1, delay + jitter)


class BackoffStrategy:
    """
    Configurable backoff strategy with state tracking.
    
    Tracks consecutive failures and provides appropriate delays.
    Resets on success.
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 10,
        jitter_factor: float = 0.3,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter_factor = jitter_factor
        self._attempt = 0
    
    def next_delay(self) -> float:
        """Get next delay and increment attempt counter."""
        delay = backoff_with_jitter(
            self._attempt,
            self.base_delay,
            self.max_delay,
            self.jitter_factor,
        )
        self._attempt = min(self._attempt + 1, self.max_attempts)
        return delay
    
    def reset(self) -> None:
        """Reset on success."""
        self._attempt = 0
    
    @property
    def attempts(self) -> int:
        return self._attempt
    
    def should_give_up(self) -> bool:
        return self._attempt >= self.max_attempts


# =============================================================================
# Task Supervisor
# =============================================================================

class TaskState(Enum):
    """State of a supervised task."""
    PENDING = auto()
    RUNNING = auto()
    RESTARTING = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class SupervisedTask:
    """Metadata for a supervised task."""
    name: str
    coro_factory: Callable[[], Awaitable[Any]]
    task: Optional[asyncio.Task] = None
    state: TaskState = TaskState.PENDING
    restart_on_failure: bool = True
    restart_delay: float = 1.0
    max_restarts: int = 10
    restart_count: int = 0
    last_error: Optional[str] = None
    started_at: float = 0.0
    stopped_at: float = 0.0


class TaskSupervisor:
    """
    Supervisor for async tasks with automatic restart on failure.
    
    Features:
    - Automatic restart of failed tasks with backoff
    - Exception logging without crashing the event loop
    - Graceful shutdown with cancellation
    - Task state tracking and metrics
    
    Based on Erlang/OTP supervisor patterns adapted for asyncio.
    
    Example:
        supervisor = TaskSupervisor()
        
        async def worker():
            while True:
                await do_work()
        
        supervisor.spawn(worker, name="worker-1", restart_on_failure=True)
        
        # Later...
        await supervisor.shutdown()
    """
    
    def __init__(self, name: str = "supervisor"):
        self._name = name
        self._tasks: Dict[str, SupervisedTask] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    def spawn(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        name: str,
        restart_on_failure: bool = True,
        restart_delay: float = 1.0,
        max_restarts: int = 10,
    ) -> None:
        """
        Spawn a supervised task.
        
        Args:
            coro_factory: Callable that returns a coroutine (called on each restart)
            name: Unique name for the task
            restart_on_failure: Whether to restart on exception
            restart_delay: Delay before restart
            max_restarts: Maximum restart attempts before giving up
        """
        if name in self._tasks:
            raise ValueError(f"Task {name} already exists")
        
        supervised = SupervisedTask(
            name=name,
            coro_factory=coro_factory,
            restart_on_failure=restart_on_failure,
            restart_delay=restart_delay,
            max_restarts=max_restarts,
        )
        
        self._tasks[name] = supervised
        self._start_task(supervised)
    
    def _start_task(self, supervised: SupervisedTask) -> None:
        """Start or restart a supervised task."""
        async def wrapper():
            try:
                supervised.state = TaskState.RUNNING
                supervised.started_at = time.time()
                await supervised.coro_factory()
                # Normal exit
                supervised.state = TaskState.STOPPED
                supervised.stopped_at = time.time()
                logger.info(f"Task {supervised.name} exited normally")
            except asyncio.CancelledError:
                supervised.state = TaskState.STOPPED
                supervised.stopped_at = time.time()
                raise
            except Exception as e:
                supervised.last_error = f"{type(e).__name__}: {e}"
                supervised.stopped_at = time.time()
                logger.error(
                    f"Task {supervised.name} crashed: {e}",
                    exc_info=True
                )
                
                # Restart if configured
                if supervised.restart_on_failure and not self._shutdown_event.is_set():
                    if supervised.restart_count < supervised.max_restarts:
                        supervised.state = TaskState.RESTARTING
                        supervised.restart_count += 1
                        
                        # Calculate backoff delay
                        delay = backoff_with_jitter(
                            supervised.restart_count - 1,
                            supervised.restart_delay,
                            60.0,
                        )
                        
                        logger.info(
                            f"Restarting {supervised.name} in {delay:.1f}s "
                            f"(attempt {supervised.restart_count}/{supervised.max_restarts})"
                        )
                        
                        await asyncio.sleep(delay)
                        
                        if not self._shutdown_event.is_set():
                            self._start_task(supervised)
                    else:
                        supervised.state = TaskState.FAILED
                        logger.error(
                            f"Task {supervised.name} exceeded max restarts, giving up"
                        )
                else:
                    supervised.state = TaskState.FAILED
        
        supervised.task = asyncio.create_task(wrapper(), name=supervised.name)
        supervised.task.add_done_callback(self._task_done_callback)
    
    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Callback when a task completes."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        
        if exc:
            # Already logged in wrapper, but ensure it's captured
            logger.debug(f"Task {task.get_name()} done callback: {exc}")
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown all supervised tasks.
        
        Args:
            timeout: Maximum time to wait for tasks to stop
        """
        logger.info(f"Supervisor {self._name} shutting down...")
        self._shutdown_event.set()
        
        # Cancel all tasks
        for name, supervised in self._tasks.items():
            if supervised.task and not supervised.task.done():
                supervised.task.cancel()
        
        # Wait for all tasks with timeout
        tasks = [s.task for s in self._tasks.values() if s.task]
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            
            # Force cancel any still pending
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        
        logger.info(f"Supervisor {self._name} shutdown complete")
    
    def get_task_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all supervised tasks."""
        return {
            name: {
                "state": supervised.state.name,
                "restart_count": supervised.restart_count,
                "last_error": supervised.last_error,
                "started_at": supervised.started_at,
                "stopped_at": supervised.stopped_at,
            }
            for name, supervised in self._tasks.items()
        }
    
    def get_running_count(self) -> int:
        """Get count of currently running tasks."""
        return sum(
            1 for s in self._tasks.values()
            if s.state == TaskState.RUNNING
        )


# =============================================================================
# Node Metrics
# =============================================================================

@dataclass
class NodeMetrics:
    """
    Prometheus-compatible metrics for IAN node.
    
    Tracks counters, gauges, and histograms for node health monitoring.
    
    Usage:
        metrics = NodeMetrics()
        metrics.inc_counter("contributions_received")
        metrics.set_gauge("peer_count", 42)
        print(metrics.to_prometheus())
    """
    
    # Counters (monotonically increasing)
    _counters: Dict[str, int] = field(default_factory=dict)
    
    # Gauges (can go up or down)
    _gauges: Dict[str, float] = field(default_factory=dict)
    
    # Histograms (list of recent samples)
    _histograms: Dict[str, List[float]] = field(default_factory=dict)
    _histogram_max_samples: int = 1000
    
    # Labels
    node_id: str = ""
    goal_id: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self):
        # Initialize default counters
        default_counters = [
            "contributions_received_total",
            "contributions_processed_total",
            "contributions_rejected_total",
            "messages_sent_total",
            "messages_received_total",
            "tau_commits_total",
            "tau_commits_failed_total",
            "rate_limit_violations_total",
            "peer_connections_total",
            "peer_disconnections_total",
        ]
        for name in default_counters:
            self._counters.setdefault(name, 0)
        
        # Initialize default gauges
        default_gauges = [
            "log_size",
            "leaderboard_size",
            "peer_count",
            "sync_lag",
            "mempool_size",
            "uptime_seconds",
        ]
        for name in default_gauges:
            self._gauges.setdefault(name, 0.0)
    
    def inc_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value
    
    def observe_histogram(self, name: str, value: float) -> None:
        """Add a sample to a histogram."""
        if name not in self._histograms:
            self._histograms[name] = []
        
        samples = self._histograms[name]
        samples.append(value)
        
        # Keep only recent samples
        if len(samples) > self._histogram_max_samples:
            self._histograms[name] = samples[-self._histogram_max_samples:]
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a histogram."""
        samples = self._histograms.get(name, [])
        if not samples:
            return {"count": 0, "sum": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_samples = sorted(samples)
        count = len(sorted_samples)
        
        def percentile(p: float) -> float:
            idx = int(count * p / 100)
            return sorted_samples[min(idx, count - 1)]
        
        return {
            "count": count,
            "sum": sum(sorted_samples),
            "avg": sum(sorted_samples) / count,
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
        }
    
    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-compatible text format
        """
        lines = [
            "# IAN Node Metrics",
            f"# node_id: {self.node_id[:16]}...",
            f"# goal_id: {self.goal_id}",
            "",
        ]
        
        # Counters
        lines.append("# HELP ian_contributions_received_total Total contributions received")
        lines.append("# TYPE ian_contributions_received_total counter")
        for name, value in sorted(self._counters.items()):
            lines.append(f"ian_{name} {value}")
        
        lines.append("")
        
        # Gauges
        lines.append("# HELP ian_log_size Current log size")
        lines.append("# TYPE ian_log_size gauge")
        for name, value in sorted(self._gauges.items()):
            lines.append(f"ian_{name} {value}")
        
        lines.append("")
        
        # Histograms
        for name, samples in self._histograms.items():
            stats = self.get_histogram_stats(name)
            lines.append(f"# TYPE ian_{name} summary")
            lines.append(f'ian_{name}{{quantile="0.5"}} {stats["p50"]}')
            lines.append(f'ian_{name}{{quantile="0.95"}} {stats["p95"]}')
            lines.append(f'ian_{name}{{quantile="0.99"}} {stats["p99"]}')
            lines.append(f"ian_{name}_count {stats['count']}")
            lines.append(f"ian_{name}_sum {stats['sum']}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self._histograms
            },
            "node_id": self.node_id,
            "goal_id": self.goal_id,
        }


# =============================================================================
# Health Server
# =============================================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class HealthServer:
    """
    HTTP server for health and readiness checks.
    
    Provides endpoints:
    - GET /health - Liveness check (is the process running?)
    - GET /ready - Readiness check (is the node synced and serving?)
    - GET /metrics - Prometheus metrics
    - GET /status - Detailed status JSON
    - GET /info - Node info (if info provider registered)
    - GET /peers - Peer info (if peers provider registered)
    
    Usage:
        health = HealthServer(port=8080)
        health.register_check("sync", check_sync_status)
        health.register_check("peers", check_peer_count)
        health.set_info_provider(get_node_info_dict)
        health.set_peers_provider(get_peers_dict)
        await health.start()
    """
    
    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        metrics: Optional[NodeMetrics] = None,
    ):
        self._port = port
        self._host = host
        self._metrics = metrics or NodeMetrics()
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._start_time = time.time()
        self._info_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._peers_provider: Optional[Callable[[], Dict[str, Any]]] = None
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func
    
    def set_metrics(self, metrics: NodeMetrics) -> None:
        """Set the metrics instance to expose."""
        self._metrics = metrics
    
    def set_info_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """Set provider for node info (/info endpoint)."""
        self._info_provider = provider
    
    def set_peers_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """Set provider for peers info (/peers endpoint)."""
        self._peers_provider = provider
    
    async def start(self) -> None:
        """Start the health server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_request,
            self._host,
            self._port,
        )
        logger.info(f"Health server listening on {self._host}:{self._port}")
    
    async def stop(self) -> None:
        """Stop the health server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle HTTP request."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(),
                timeout=5.0,
            )
            
            if not request_line:
                return
            
            # Parse request
            parts = request_line.decode().strip().split()
            if len(parts) < 2:
                return
            
            method, path = parts[0], parts[1]
            
            # Read headers (skip for now)
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break
            
            # Route request
            if path == "/health":
                response = self._handle_health()
            elif path == "/ready":
                response = self._handle_ready()
            elif path == "/metrics":
                response = self._handle_metrics()
            elif path == "/status":
                response = self._handle_status()
            elif path == "/info":
                response = self._handle_info()
            elif path == "/peers":
                response = self._handle_peers()
            else:
                response = self._http_response(404, "Not Found")
            
            writer.write(response.encode())
            await writer.drain()
            
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Health server error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    def _handle_health(self) -> str:
        """Liveness check - is the process running?"""
        return self._http_response(200, "OK", content_type="text/plain")
    
    def _handle_ready(self) -> str:
        """Readiness check - run all registered checks."""
        all_checks = []
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self._checks.items():
            try:
                result = check_func()
                all_checks.append(result)
                
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                all_checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                ))
                overall_status = HealthStatus.UNHEALTHY
        
        if overall_status == HealthStatus.HEALTHY:
            return self._http_response(200, "READY", content_type="text/plain")
        elif overall_status == HealthStatus.DEGRADED:
            messages = [c.message for c in all_checks if c.status != HealthStatus.HEALTHY]
            return self._http_response(200, f"DEGRADED: {'; '.join(messages)}", content_type="text/plain")
        else:
            messages = [c.message for c in all_checks if c.status == HealthStatus.UNHEALTHY]
            return self._http_response(503, f"NOT READY: {'; '.join(messages)}", content_type="text/plain")
    
    def _handle_metrics(self) -> str:
        """Prometheus metrics endpoint."""
        # Update uptime
        self._metrics.set_gauge("uptime_seconds", time.time() - self._start_time)
        
        return self._http_response(
            200,
            self._metrics.to_prometheus(),
            content_type="text/plain; version=0.0.4",
        )
    
    def _handle_status(self) -> str:
        """Detailed status as JSON."""
        status = {
            "uptime_seconds": time.time() - self._start_time,
            "metrics": self._metrics.to_dict(),
            "checks": {},
        }
        
        for name, check_func in self._checks.items():
            try:
                result = check_func()
                status["checks"][name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                }
            except Exception as e:
                status["checks"][name] = {
                    "status": "error",
                    "message": str(e),
                }
        
        return self._http_response(
            200,
            json.dumps(status, indent=2),
            content_type="application/json",
        )
    
    def _handle_info(self) -> str:
        """Node info as JSON (if provider is set)."""
        if not self._info_provider:
            return self._http_response(404, "Not Found", content_type="application/json")
        
        try:
            info = self._info_provider() or {}
            body = json.dumps(info, indent=2)
            return self._http_response(200, body, content_type="application/json")
        except Exception as e:
            error_body = json.dumps({"error": str(e)}, indent=2)
            return self._http_response(500, error_body, content_type="application/json")
    
    def _handle_peers(self) -> str:
        """Peer info as JSON (if provider is set)."""
        if not self._peers_provider:
            return self._http_response(404, "Not Found", content_type="application/json")
        
        try:
            peers = self._peers_provider() or {}
            body = json.dumps(peers, indent=2)
            return self._http_response(200, body, content_type="application/json")
        except Exception as e:
            error_body = json.dumps({"error": str(e)}, indent=2)
            return self._http_response(500, error_body, content_type="application/json")
    
    def _http_response(
        self,
        status_code: int,
        body: str,
        content_type: str = "text/plain",
    ) -> str:
        """Build HTTP response."""
        status_text = {
            200: "OK",
            404: "Not Found",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status_code, "")
        return (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )


# =============================================================================
# Peer Scoring
# =============================================================================

@dataclass
class PeerScore:
    """
    Reputation score for a peer.
    
    Tracks positive and negative behaviors to determine peer quality.
    Used for:
    - Prioritizing good peers for connections
    - Disconnecting/banning bad peers
    - Selecting peers for sync
    """
    node_id: str
    score: float = 100.0  # Start neutral (0-200 range)
    
    # Positive behaviors
    valid_messages: int = 0
    valid_contributions: int = 0
    successful_syncs: int = 0
    
    # Negative behaviors
    invalid_messages: int = 0
    invalid_contributions: int = 0
    rate_limit_violations: int = 0
    protocol_violations: int = 0
    failed_syncs: int = 0
    
    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    banned_until: float = 0.0
    
    # Score weights
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "valid_message": 0.1,
        "valid_contribution": 1.0,
        "successful_sync": 5.0,
        "invalid_message": -5.0,
        "invalid_contribution": -10.0,
        "rate_limit_violation": -10.0,
        "protocol_violation": -20.0,
        "failed_sync": -5.0,
    })
    
    # Map event names to counter field names
    EVENT_TO_FIELD: Dict[str, str] = field(default_factory=lambda: {
        "valid_message": "valid_messages",
        "valid_contribution": "valid_contributions",
        "successful_sync": "successful_syncs",
        "invalid_message": "invalid_messages",
        "invalid_contribution": "invalid_contributions",
        "rate_limit_violation": "rate_limit_violations",
        "protocol_violation": "protocol_violations",
        "failed_sync": "failed_syncs",
    })
    
    def record_event(self, event: str, count: int = 1) -> None:
        """
        Record a behavior event and update score.
        
        Args:
            event: Event type (e.g., "valid_message", "rate_limit_violation")
            count: Number of occurrences
        """
        self.last_seen = time.time()
        
        # Update counter using field mapping
        field_name = self.EVENT_TO_FIELD.get(event)
        if field_name and hasattr(self, field_name):
            current = getattr(self, field_name)
            setattr(self, field_name, current + count)
        
        # Update score
        weight = self.WEIGHTS.get(event, 0)
        self.score += weight * count
        
        # Clamp score
        self.score = max(0.0, min(200.0, self.score))
    
    def should_disconnect(self) -> bool:
        """Should we disconnect this peer?"""
        return self.score < 30.0
    
    def should_ban(self, duration_hours: float = 24.0) -> bool:
        """Should we ban this peer?"""
        if self.score < 10.0:
            self.banned_until = time.time() + (duration_hours * 3600)
            return True
        return False
    
    def is_banned(self) -> bool:
        """Is this peer currently banned?"""
        return self.banned_until > time.time()
    
    def is_trusted(self) -> bool:
        """Is this peer trusted (high score)?"""
        return self.score >= 150.0
    
    def decay(self, hours_since_last_decay: float = 1.0) -> None:
        """
        Exponential decay of score towards neutral over time.
        
        Uses exponential decay formula: score = neutral + (score - neutral) * e^(-λt)
        
        Properties:
        - Good peers slowly lose reputation if inactive
        - Bad peers slowly regain reputation  
        - Exponential decay ensures bad actors recover slowly
        - Half-life is approximately 70 hours (ln(2)/0.01)
        
        Args:
            hours_since_last_decay: Time since last decay in hours
            
        Invariants:
            - Score always moves towards neutral (100.0)
            - Score never crosses neutral
            - Decay rate is proportional to distance from neutral
        """
        import math
        
        neutral = 100.0
        # Decay constant: λ = 0.01 per hour → half-life ≈ 69.3 hours
        decay_lambda = 0.01
        
        # Exponential decay: new_distance = old_distance * e^(-λt)
        decay_factor = math.exp(-decay_lambda * hours_since_last_decay)
        
        # Apply decay towards neutral
        distance_from_neutral = self.score - neutral
        self.score = neutral + (distance_from_neutral * decay_factor)


class PeerScoreManager:
    """
    Manager for peer reputation scores.
    
    Tracks scores for all known peers and provides methods for
    peer selection and management.
    
    Features:
    - Automatic persistence to disk
    - Periodic auto-save
    - Score decay over time
    - Peer banning and trust levels
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_save_interval: float = 300.0,  # 5 minutes
    ):
        self._scores: Dict[str, PeerScore] = {}
        self._persist_path = persist_path
        self._auto_save_interval = auto_save_interval
        self._last_save_time = time.time()
        self._dirty = False  # Track if changes need saving
    
    def get_score(self, node_id: str) -> PeerScore:
        """Get or create score for a peer."""
        if node_id not in self._scores:
            self._scores[node_id] = PeerScore(node_id=node_id)
        return self._scores[node_id]
    
    def record_event(self, node_id: str, event: str, count: int = 1) -> None:
        """Record an event for a peer."""
        score = self.get_score(node_id)
        score.record_event(event, count)
        self._dirty = True
        
        # Check for ban
        if score.should_ban():
            logger.warning(f"Banning peer {node_id[:16]}... (score: {score.score})")
        
        # Auto-save if interval elapsed
        self._maybe_auto_save()
    
    def _maybe_auto_save(self) -> None:
        """Auto-save if dirty and interval elapsed."""
        if not self._dirty or not self._persist_path:
            return
        
        if time.time() - self._last_save_time >= self._auto_save_interval:
            self.save()
    
    def get_best_peers(self, count: int = 10) -> List[str]:
        """Get the highest-scored peers."""
        active = [
            (node_id, score)
            for node_id, score in self._scores.items()
            if not score.is_banned()
        ]
        active.sort(key=lambda x: x[1].score, reverse=True)
        return [node_id for node_id, _ in active[:count]]
    
    def get_peers_for_sync(self, count: int = 3) -> List[str]:
        """Get trusted peers suitable for sync."""
        trusted = [
            node_id
            for node_id, score in self._scores.items()
            if score.is_trusted() and not score.is_banned()
        ]
        return trusted[:count]
    
    def get_peers_to_disconnect(self) -> List[str]:
        """Get peers that should be disconnected."""
        return [
            node_id
            for node_id, score in self._scores.items()
            if score.should_disconnect()
        ]
    
    def decay_all(self, hours: float = 1.0) -> None:
        """Apply decay to all peer scores."""
        for score in self._scores.values():
            score.decay(hours)
    
    def save(self) -> bool:
        """
        Persist scores to disk atomically.
        
        Uses atomic write (write to temp, then rename) to prevent
        corruption if process crashes during write.
        """
        if not self._persist_path:
            return False
        
        try:
            data = {
                "version": 1,
                "saved_at": time.time(),
                "peers": {
                    node_id: {
                        "score": score.score,
                        "valid_messages": score.valid_messages,
                        "valid_contributions": score.valid_contributions,
                        "successful_syncs": score.successful_syncs,
                        "invalid_messages": score.invalid_messages,
                        "invalid_contributions": score.invalid_contributions,
                        "rate_limit_violations": score.rate_limit_violations,
                        "protocol_violations": score.protocol_violations,
                        "failed_syncs": score.failed_syncs,
                        "banned_until": score.banned_until,
                        "first_seen": score.first_seen,
                        "last_seen": score.last_seen,
                    }
                    for node_id, score in self._scores.items()
                }
            }
            
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(data, indent=2))
            temp.rename(path)
            
            self._dirty = False
            self._last_save_time = time.time()
            logger.debug(f"Saved {len(self._scores)} peer scores to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save peer scores: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load scores from disk.
        
        Handles both v1 (versioned) and legacy formats.
        """
        if not self._persist_path:
            return False
        
        try:
            path = Path(self._persist_path)
            if not path.exists():
                return False
            
            raw_data = json.loads(path.read_text())
            
            # Handle versioned format (v1+)
            if "version" in raw_data:
                version = raw_data["version"]
                peers_data = raw_data.get("peers", {})
                saved_at = raw_data.get("saved_at", 0)
                logger.info(
                    f"Loading peer scores v{version} "
                    f"(saved {time.time() - saved_at:.0f}s ago)"
                )
            else:
                # Legacy format (peers at top level)
                peers_data = raw_data
            
            for node_id, values in peers_data.items():
                score = PeerScore(node_id=node_id)
                score.score = values.get("score", 100.0)
                score.valid_messages = values.get("valid_messages", 0)
                score.valid_contributions = values.get("valid_contributions", 0)
                score.successful_syncs = values.get("successful_syncs", 0)
                score.invalid_messages = values.get("invalid_messages", 0)
                score.invalid_contributions = values.get("invalid_contributions", 0)
                score.rate_limit_violations = values.get("rate_limit_violations", 0)
                score.protocol_violations = values.get("protocol_violations", 0)
                score.failed_syncs = values.get("failed_syncs", 0)
                score.banned_until = values.get("banned_until", 0.0)
                score.first_seen = values.get("first_seen", time.time())
                score.last_seen = values.get("last_seen", time.time())
                self._scores[node_id] = score
            
            self._dirty = False
            logger.info(f"Loaded {len(self._scores)} peer scores from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load peer scores: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about peer scores."""
        scores = list(self._scores.values())
        if not scores:
            return {"total": 0}
        
        return {
            "total": len(scores),
            "banned": sum(1 for s in scores if s.is_banned()),
            "trusted": sum(1 for s in scores if s.is_trusted()),
            "low_score": sum(1 for s in scores if s.should_disconnect()),
            "avg_score": sum(s.score for s in scores) / len(scores),
        }


# =============================================================================
# Graceful Shutdown Helper
# =============================================================================

class GracefulShutdown:
    """
    Helper for graceful shutdown with signal handling.
    
    Usage:
        shutdown = GracefulShutdown()
        shutdown.register_handler(save_state)
        shutdown.register_handler(close_connections)
        shutdown.setup_signals()
        
        # In your main loop
        while not shutdown.is_shutting_down():
            await do_work()
    """
    
    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._handlers: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_started = False
    
    def register_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register an async shutdown handler."""
        self._handlers.append(handler)
    
    def setup_signals(self) -> None:
        """Setup signal handlers for SIGTERM and SIGINT."""
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
    
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        if self._shutdown_started:
            logger.warning("Shutdown already in progress, ignoring signal")
            return
        
        self._shutdown_started = True
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
        
        # Run handlers in reverse order (LIFO)
        for handler in reversed(self._handlers):
            try:
                await asyncio.shield(handler())
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")
        
        self._shutdown_event.set()
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._shutdown_event.is_set()
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


# =============================================================================
# State Persistence
# =============================================================================

@dataclass
class NodeState:
    """Persistent node state."""
    node_id: str
    goal_id: str
    last_log_root: str = ""
    log_size: int = 0
    last_tau_commit: float = 0.0
    contributions_since_commit: int = 0
    last_checkpoint: float = 0.0
    
    def save(self, path: str) -> bool:
        """Save state to disk atomically."""
        try:
            data = {
                "node_id": self.node_id,
                "goal_id": self.goal_id,
                "last_log_root": self.last_log_root,
                "log_size": self.log_size,
                "last_tau_commit": self.last_tau_commit,
                "contributions_since_commit": self.contributions_since_commit,
                "last_checkpoint": time.time(),
            }
            
            file_path = Path(path)
            temp_path = file_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.rename(file_path)
            
            logger.debug(f"Saved node state to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save node state: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> Optional["NodeState"]:
        """Load state from disk."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return None
            
            data = json.loads(file_path.read_text())
            
            return cls(
                node_id=data["node_id"],
                goal_id=data["goal_id"],
                last_log_root=data.get("last_log_root", ""),
                log_size=data.get("log_size", 0),
                last_tau_commit=data.get("last_tau_commit", 0.0),
                contributions_since_commit=data.get("contributions_since_commit", 0),
                last_checkpoint=data.get("last_checkpoint", 0.0),
            )
            
        except Exception as e:
            logger.error(f"Failed to load node state: {e}")
            return None
