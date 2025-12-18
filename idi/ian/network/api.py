"""
IAN REST API Server - HTTP API for IAN operations.

Endpoints:
- POST /api/v1/contribute - Submit contribution
- GET  /api/v1/leaderboard/{goal_id} - Get leaderboard
- GET  /api/v1/status/{goal_id} - Get goal status
- GET  /api/v1/log/{goal_id} - Get log info
- GET  /api/v1/policy/{goal_id} - Get active policy
- GET  /api/v1/proof/{goal_id}/{log_index} - Get membership proof
- GET  /health - Health check
- GET  /metrics - Prometheus metrics

Authentication:
- API key in header (X-API-Key)
- Optional: signature-based auth

Rate Limiting:
- Per-IP rate limiting
- Per-contributor rate limiting (via security module)
"""

from __future__ import annotations

import base64
import hashlib
from importlib import resources as importlib_resources
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING

try:
    # Optional structured tracing support
    from .resilience import set_correlation_id, new_correlation_id
except ImportError:  # pragma: no cover - resilience not required for API
    set_correlation_id = None
    new_correlation_id = None

if TYPE_CHECKING:
    from idi.ian.coordinator import IANCoordinator
    from idi.ian.security import SecureCoordinator

logger = logging.getLogger(__name__)


def _cors_allow_origin_value(allowed_origins: List[str], origin: str) -> Optional[str]:
    """Return the Access-Control-Allow-Origin value for a request.

    Preconditions:
        - `origin` is a non-empty Origin header value.
        - `allowed_origins` is a non-empty allowlist.
    Postconditions:
        - Returns "*" iff wildcard is enabled.
        - Returns `origin` iff `origin` is allowlisted.
        - Returns None otherwise.
    """
    if "*" in allowed_origins:
        return "*"
    if origin in allowed_origins:
        return origin
    return None


def _cors_apply_headers(response: Any, allow_origin: str) -> None:
    """Apply CORS headers to a response.

    Preconditions:
        - `allow_origin` is either "*" or a validated, allowlisted origin.
    Postconditions:
        - Response includes standard CORS headers for allowed origins.
    """
    response.headers["Access-Control-Allow-Origin"] = allow_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    response.headers["Access-Control-Max-Age"] = "600"

    if allow_origin == "*":
        return

    vary = response.headers.get("Vary")
    if vary is None:
        response.headers["Vary"] = "Origin"
        return

    if "Origin" in vary:
        return
    response.headers["Vary"] = f"{vary}, Origin"


# =============================================================================
# API Response Types
# =============================================================================

@dataclass
class ApiResponse:
    """Standard API response."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time() * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def success_response(data: Any) -> ApiResponse:
    return ApiResponse(success=True, data=data)


def error_response(error: str, status_code: int = 400) -> ApiResponse:
    return ApiResponse(success=False, error=error)


# =============================================================================
# API Configuration
# =============================================================================

@dataclass
class ApiConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: Optional[str] = None  # If set, require X-API-Key header
    api_key_required: bool = False  # If true, reject requests when api_key is missing
    rate_limit_per_ip: int = 100  # Requests per minute
    cors_origins: List[str] = None  # CORS allowed origins
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []


def _parse_bool_env(value: Optional[str]) -> bool:
    """Parse a boolean environment value.

    Preconditions:
        - value is either None or a string.
    Postconditions:
        - Returns True iff value is in {"1","true","yes","on"} (case-insensitive).
    """
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_env(value: Optional[str]) -> List[str]:
    """Parse a comma-separated environment value into a list of strings."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def api_config_from_env(env: Optional[Mapping[str, str]] = None) -> ApiConfig:
    """Build ApiConfig from environment variables.

    Environment variables:
        - IAN_API_HOST (fallback: IAN_LISTEN_HOST)
        - IAN_API_PORT
        - IAN_API_KEY
        - IAN_API_KEY_REQUIRED
        - IAN_API_CORS_ORIGINS (comma-separated)

    Preconditions:
        - If IAN_API_KEY_REQUIRED is true, IAN_API_KEY must be set.
    Postconditions:
        - Returned config is safe-by-default (deny-by-default CORS).
    """
    src = env if env is not None else os.environ

    host = src.get("IAN_API_HOST") or src.get("IAN_LISTEN_HOST") or "0.0.0.0"

    port_raw = src.get("IAN_API_PORT")
    port = 8080
    if port_raw is not None:
        port = int(port_raw)

    api_key_raw = src.get("IAN_API_KEY")
    api_key = api_key_raw.strip() if api_key_raw is not None else None
    if api_key == "":
        api_key = None

    api_key_required = _parse_bool_env(src.get("IAN_API_KEY_REQUIRED"))
    if api_key_required and api_key is None:
        raise ValueError("IAN_API_KEY_REQUIRED is true but IAN_API_KEY is not set")

    cors_origins = _parse_csv_env(src.get("IAN_API_CORS_ORIGINS"))

    return ApiConfig(
        host=host,
        port=port,
        api_key=api_key,
        api_key_required=api_key_required,
        cors_origins=cors_origins,
    )


# =============================================================================
# API Handlers (Framework-agnostic)
# =============================================================================

class IANApiHandlers:
    """
    API request handlers.
    
    Framework-agnostic implementation that can be adapted to
    Flask, FastAPI, aiohttp, etc.
    """
    
    def __init__(
        self,
        coordinator: "IANCoordinator",
        config: Optional[ApiConfig] = None,
        use_secure: bool = True,
    ):
        self._config = config or ApiConfig()
        if use_secure:
            # Wrap coordinator with SecureCoordinator to enforce validation,
            # per-contributor rate limiting, and optional PoW at the API boundary.
            from idi.ian.security import SecureCoordinator
            if isinstance(coordinator, SecureCoordinator):
                self._coordinator = coordinator
            else:
                self._coordinator = SecureCoordinator(coordinator)
        else:
            self._coordinator = coordinator
        # Bounded IP tracking to prevent memory exhaustion from many unique IPs
        self._request_counts: OrderedDict[str, List[float]] = OrderedDict()
        self._max_tracked_ips = 10_000
    
    def _check_rate_limit(self, ip: str) -> bool:
        """Check if IP is rate limited."""
        now = time.time()
        window = 60  # 1 minute window
        
        # Clean old entries
        if ip in self._request_counts:
            self._request_counts[ip] = [
                t for t in self._request_counts[ip]
                if now - t < window
            ]
        else:
            self._request_counts[ip] = []
        
        # Check limit
        if len(self._request_counts[ip]) >= self._config.rate_limit_per_ip:
            return False
        
        # Record request
        self._request_counts[ip].append(now)
        # Move to end (most recent) and enforce max size
        self._request_counts.move_to_end(ip)
        while len(self._request_counts) > self._max_tracked_ips:
            self._request_counts.popitem(last=False)  # Evict oldest
        return True
    
    def _check_api_key(self, provided_key: Optional[str]) -> bool:
        """Check API key if configured."""
        if not self._config.api_key:
            if self._config.api_key_required:
                return False
            return True
        return provided_key == self._config.api_key
    
    # -------------------------------------------------------------------------
    # Contribution
    # -------------------------------------------------------------------------
    
    def handle_contribute(
        self,
        body: Dict[str, Any],
        ip: str,
        api_key: Optional[str] = None,
    ) -> ApiResponse:
        """
        Handle POST /api/v1/contribute
        
        Body:
        {
            "goal_id": "GOAL_ID",
            "agent_pack": {
                "version": "1.0",
                "parameters": "<base64>",
                "metadata": {}
            },
            "proofs": {},
            "contributor_id": "alice",
            "seed": 12345
        }
        """
        # Auth check
        if not self._check_api_key(api_key):
            return error_response("Invalid API key", 401)
        
        # Rate limit
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            from idi.ian.models import Contribution, AgentPack, GoalID
            
            # Parse contribution
            pack_data = body.get("agent_pack", {})
            parameters = pack_data.get("parameters", "")
            if isinstance(parameters, str):
                parameters = base64.b64decode(parameters)
            
            agent_pack = AgentPack(
                version=pack_data.get("version", "1.0"),
                parameters=parameters,
                metadata=pack_data.get("metadata"),
            )
            
            contribution = Contribution(
                goal_id=GoalID(body["goal_id"]),
                agent_pack=agent_pack,
                proofs=body.get("proofs", {}),
                contributor_id=body["contributor_id"],
                seed=int(body.get("seed", 0)),
            )
            
            # Process
            result = self._coordinator.process_contribution(contribution)
            
            return success_response({
                "accepted": result.accepted,
                "reason": result.reason,
                "rejection_type": result.rejection_type.name if result.rejection_type else None,
                "metrics": asdict(result.metrics) if result.metrics else None,
                "log_index": result.log_index,
                "score": result.score,
            })
            
        except KeyError as e:
            return error_response(f"Missing field: {e}")
        except ValueError as e:
            return error_response(f"Invalid value: {e}")
        except Exception as e:
            logger.exception("Contribute error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Leaderboard
    # -------------------------------------------------------------------------
    
    def handle_leaderboard(
        self,
        goal_id: str,
        limit: int = 100,
        ip: str = "",
    ) -> ApiResponse:
        """Handle GET /api/v1/leaderboard/{goal_id}"""
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            leaderboard = self._coordinator.get_leaderboard()
            
            entries = []
            for entry in leaderboard[:limit]:
                entries.append({
                    "rank": len(entries) + 1,
                    "pack_hash": entry.pack_hash.hex(),
                    "contributor_id": entry.contributor_id,
                    "score": entry.score,
                    "log_index": entry.log_index,
                    "timestamp_ms": entry.timestamp_ms,
                })
            
            return success_response({
                "goal_id": goal_id,
                "entries": entries,
                "total": len(leaderboard),
            })
            
        except Exception as e:
            logger.exception("Leaderboard error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    def handle_status(self, goal_id: str, ip: str = "") -> ApiResponse:
        """Handle GET /api/v1/status/{goal_id}"""
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            stats = self._coordinator.get_stats()
            
            return success_response({
                "goal_id": goal_id,
                "log_size": stats.get("log_size", 0),
                "leaderboard_size": stats.get("leaderboard_size", 0),
                "total_contributions": stats.get("total_contributions", 0),
                "accepted_contributions": stats.get("accepted_contributions", 0),
                "rejected_contributions": stats.get("rejected_contributions", 0),
                "log_root": stats.get("log_root", b"").hex() if isinstance(stats.get("log_root"), bytes) else stats.get("log_root", ""),
                "leaderboard_root": stats.get("leaderboard_root", b"").hex() if isinstance(stats.get("leaderboard_root"), bytes) else stats.get("leaderboard_root", ""),
            })
            
        except Exception as e:
            logger.exception("Status error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Log
    # -------------------------------------------------------------------------
    
    def handle_log(
        self,
        goal_id: str,
        from_index: int = 0,
        limit: int = 100,
        ip: str = "",
    ) -> ApiResponse:
        """Handle GET /api/v1/log/{goal_id}"""
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            log_root = self._coordinator.get_log_root()
            log_size = self._coordinator.state.log.size
            
            return success_response({
                "goal_id": goal_id,
                "log_root": log_root.hex(),
                "log_size": log_size,
                "from_index": from_index,
                "limit": limit,
            })
            
        except Exception as e:
            logger.exception("Log error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Active Policy
    # -------------------------------------------------------------------------
    
    def handle_policy(self, goal_id: str, ip: str = "") -> ApiResponse:
        """Handle GET /api/v1/policy/{goal_id}"""
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            policy = self._coordinator.get_active_policy()
            
            if policy is None:
                return success_response({
                    "goal_id": goal_id,
                    "active_policy": None,
                })
            
            return success_response({
                "goal_id": goal_id,
                "active_policy": {
                    "pack_hash": policy.pack_hash.hex(),
                    "contributor_id": policy.contributor_id,
                    "score": policy.score,
                    "log_index": policy.log_index,
                    "timestamp_ms": policy.timestamp_ms,
                },
            })
            
        except Exception as e:
            logger.exception("Policy error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Membership Proof
    # -------------------------------------------------------------------------
    
    def handle_proof(
        self,
        goal_id: str,
        log_index: int,
        ip: str = "",
    ) -> ApiResponse:
        """
        Handle GET /api/v1/proof/{goal_id}/{log_index}
        
        Returns a Merkle membership proof for a contribution in the log.
        """
        if not self._check_rate_limit(ip):
            return error_response("Rate limited", 429)
        
        try:
            log = self._coordinator.state.log
            
            if log_index < 0 or log_index >= log.size:
                return error_response(f"Log index {log_index} out of bounds [0, {log.size})", 404)
            
            proof = log.get_proof(log_index)
            
            return success_response({
                "goal_id": goal_id,
                "log_index": log_index,
                "leaf_hash": proof.leaf_hash.hex(),
                "siblings": [
                    {"hash": h.hex(), "is_right": r}
                    for h, r in proof.siblings
                ],
                "peaks_bag": [p.hex() for p in proof.peaks_bag],
                "mmr_size": proof.mmr_size,
                "log_root": log.get_root().hex(),
            })
            
        except IndexError as e:
            return error_response(f"Invalid log index: {e}", 404)
        except Exception as e:
            logger.exception("Proof error")
            return error_response(f"Internal error: {e}", 500)
    
    # -------------------------------------------------------------------------
    # Health & Metrics
    # -------------------------------------------------------------------------
    
    def handle_health(self) -> ApiResponse:
        """Handle GET /health"""
        return success_response({
            "status": "healthy",
            "timestamp": int(time.time() * 1000),
        })
    
    def handle_metrics(self) -> str:
        """Handle GET /metrics - Returns Prometheus format."""
        from idi.ian.observability import metrics
        return metrics.to_prometheus()
    
    def handle_openapi(self) -> Dict[str, Any]:
        """Handle GET /api/v1/openapi.json - Returns OpenAPI spec."""
        import yaml

        try:
            spec_text = (
                importlib_resources.files("idi.ian.network")
                .joinpath("openapi.yaml")
                .read_text(encoding="utf-8")
            )
            return yaml.safe_load(spec_text)
        except Exception:
            pass

        from pathlib import Path

        spec_path = Path(__file__).parent / "openapi.yaml"
        if spec_path.exists():
            with open(spec_path) as f:
                return yaml.safe_load(f)
        return {"error": "OpenAPI spec not found"}


# =============================================================================
# API Server (aiohttp-based)
# =============================================================================

class IANApiServer:
    """
    HTTP API server for IAN.
    
    Uses aiohttp for async HTTP handling.
    Falls back to simple HTTP server if aiohttp not available.
    """
    
    def __init__(
        self,
        coordinator: "IANCoordinator",
        config: Optional[ApiConfig] = None,
    ):
        self._coordinator = coordinator
        self._config = config or api_config_from_env()
        # Use SecureCoordinator-backed handlers by default for production safety.
        self._handlers = IANApiHandlers(coordinator, self._config, use_secure=True)
        self._app = None
        self._runner = None
        self._fallback_server: ThreadingHTTPServer | None = None
        self._fallback_thread: threading.Thread | None = None
    
    async def start(self) -> None:
        """Start API server."""
        try:
            from aiohttp import web
            
            self._app = create_api_app(self._handlers, self._config)
            
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            
            site = web.TCPSite(
                self._runner,
                self._config.host,
                self._config.port,
            )
            await site.start()
            
            logger.info(f"API server started on http://{self._config.host}:{self._config.port}")
            
        except ImportError:
            self._start_fallback_server()

    def _start_fallback_server(self) -> None:
        """Start a minimal HTTP server for /health and /metrics.

        Preconditions:
            - The configured host/port are bindable.
        Postconditions:
            - A background thread is serving HTTP responses.
        """
        handlers = self._handlers

        class FallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    resp = handlers.handle_health()
                    body = resp.to_json().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/metrics":
                    text = handlers.handle_metrics()
                    body = text.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        server = ThreadingHTTPServer((self._config.host, self._config.port), FallbackHandler)
        self._fallback_server = server

        thread = threading.Thread(target=server.serve_forever, name="ian-api-fallback", daemon=True)
        self._fallback_thread = thread
        thread.start()

        logger.warning(
            "aiohttp not available; started minimal fallback API server on "
            f"http://{self._config.host}:{self._config.port}"
        )
    
    async def stop(self) -> None:
        """Stop API server."""
        if self._runner:
            await self._runner.cleanup()

        if self._fallback_server is not None:
            self._fallback_server.shutdown()
            self._fallback_server.server_close()
            self._fallback_server = None

        if self._fallback_thread is not None:
            self._fallback_thread.join(timeout=1.0)
            self._fallback_thread = None
        logger.info("API server stopped")


def create_api_app(handlers: IANApiHandlers, config: ApiConfig):
    """Create aiohttp application with routes."""
    try:
        from aiohttp import web
    except ImportError:
        raise ImportError("aiohttp required for API server: pip install aiohttp")
    
    app = web.Application()
    
    # Helper to get client IP
    def get_ip(request: web.Request) -> str:
        return request.headers.get("X-Forwarded-For", request.remote or "unknown").split(",")[0].strip()
    
    def _bind_trace(request: web.Request) -> None:
        """Bind or create a correlation/trace ID for this request."""
        if new_correlation_id is None:
            return
        incoming = request.headers.get("X-Trace-Id")
        if incoming and set_correlation_id is not None:
            set_correlation_id(incoming)
        else:
            new_correlation_id()
    
    # Helper for JSON responses
    def json_response(resp: ApiResponse, status: int = 200) -> web.Response:
        return web.json_response(resp.to_dict(), status=status)
    
    # Routes
    async def contribute(request: web.Request) -> web.Response:
        _bind_trace(request)
        body = await request.json()
        api_key = request.headers.get("X-API-Key")
        resp = handlers.handle_contribute(body, get_ip(request), api_key)

        if resp.success:
            status = 200
        else:
            error = resp.error or ""
            if "API key" in error:
                status = 401
            elif "Rate limited" in error:
                status = 429
            elif error.startswith("Internal error:"):
                status = 500
            else:
                status = 400

        return json_response(resp, status)
    
    async def leaderboard(request: web.Request) -> web.Response:
        _bind_trace(request)
        goal_id = request.match_info["goal_id"]
        try:
            limit = int(request.query.get("limit", 100))
            limit = max(1, min(1000, limit))  # Clamp to valid range
        except (ValueError, TypeError):
            return json_response(error_response("Invalid limit parameter"), 400)
        resp = handlers.handle_leaderboard(goal_id, limit, get_ip(request))
        return json_response(resp)
    
    async def status(request: web.Request) -> web.Response:
        _bind_trace(request)
        goal_id = request.match_info["goal_id"]
        resp = handlers.handle_status(goal_id, get_ip(request))
        return json_response(resp)
    
    async def log(request: web.Request) -> web.Response:
        _bind_trace(request)
        goal_id = request.match_info["goal_id"]
        try:
            from_index = int(request.query.get("from", 0))
            limit = int(request.query.get("limit", 100))
            from_index = max(0, from_index)  # Non-negative
            limit = max(1, min(1000, limit))  # Clamp to valid range
        except (ValueError, TypeError):
            return json_response(error_response("Invalid from/limit parameter"), 400)
        resp = handlers.handle_log(goal_id, from_index, limit, get_ip(request))
        return json_response(resp)
    
    async def policy(request: web.Request) -> web.Response:
        _bind_trace(request)
        goal_id = request.match_info["goal_id"]
        resp = handlers.handle_policy(goal_id, get_ip(request))
        return json_response(resp)
    
    async def proof(request: web.Request) -> web.Response:
        _bind_trace(request)
        goal_id = request.match_info["goal_id"]
        try:
            log_index = int(request.match_info["log_index"])
            if log_index < 0:
                return json_response(error_response("log_index must be non-negative"), 400)
        except (ValueError, TypeError):
            return json_response(error_response("Invalid log_index parameter"), 400)
        resp = handlers.handle_proof(goal_id, log_index, get_ip(request))
        status = 200 if resp.success else (404 if "out of bounds" in (resp.error or "") else 400)
        return json_response(resp, status)
    
    async def health(request: web.Request) -> web.Response:
        _bind_trace(request)
        resp = handlers.handle_health()
        return json_response(resp)
    
    async def metrics(request: web.Request) -> web.Response:
        _bind_trace(request)
        text = handlers.handle_metrics()
        return web.Response(text=text, content_type="text/plain")
    
    async def openapi(request: web.Request) -> web.Response:
        _bind_trace(request)
        spec = handlers.handle_openapi()
        return web.json_response(spec)
    
    # Register routes
    app.router.add_post("/api/v1/contribute", contribute)
    app.router.add_get("/api/v1/leaderboard/{goal_id}", leaderboard)
    app.router.add_get("/api/v1/status/{goal_id}", status)
    app.router.add_get("/api/v1/log/{goal_id}", log)
    app.router.add_get("/api/v1/policy/{goal_id}", policy)
    app.router.add_get("/api/v1/proof/{goal_id}/{log_index}", proof)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/api/v1/openapi.json", openapi)
    
    # CORS middleware
    if config.cors_origins:
        allowed_origins = list(config.cors_origins)

        @web.middleware
        async def cors_middleware(request: web.Request, handler):
            origin = request.headers.get("Origin")
            if origin is None:
                return await handler(request)

            if request.method == "OPTIONS":
                response = web.Response(status=204)
            else:
                response = await handler(request)

            allow_origin = _cors_allow_origin_value(allowed_origins, origin)
            if allow_origin is None:
                return response

            _cors_apply_headers(response, allow_origin)
            return response
        
        app.middlewares.append(cors_middleware)
    
    return app
