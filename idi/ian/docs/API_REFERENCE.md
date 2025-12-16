# IAN API Reference

Complete API documentation for the IAN (Intelligent Agent Network) decentralized system.

## Table of Contents

1. [Overview](#overview)
2. [HTTP API](#http-api)
3. [P2P Protocol](#p2p-protocol)
4. [Python SDK](#python-sdk)
5. [Configuration](#configuration)
6. [Error Codes](#error-codes)
7. [WebSocket API](#websocket-api)

---

## Overview

IAN exposes multiple interfaces for interaction:

| Interface | Default Port | Purpose |
|-----------|--------------|---------|
| HTTP REST (`/api/v1/*`) | 8000 | Public API for contributions and queries |
| Health / Metrics | 8000 | Liveness and Prometheus metrics (`/health`, `/metrics`) |
| P2P TCP | 9000 | Node-to-node gossip and state sync |
| WebSocket | 9001 | Real-time subscriptions (optional) |

### Authentication

- **HTTP**: Optional API key via `X-API-Key` header (if configured in `ApiConfig.api_key`).
- **P2P**: Ed25519 signatures on messages and NodeIdentity; challenge-response handshake at the P2P layer.
- **WebSocket**: Token-based authentication on connect (pattern; implementation depends on deployment).

---

## HTTP API

There are **two HTTP surfaces**:

1. `HealthServer` (built-in): `/health`, `/ready`, `/metrics`, `/status` — used by `DecentralizedNode`.
2. `IANApiServer` (aiohttp-based, optional): `/api/v1/*` REST API defined in `idi.ian.network.api` and `openapi.yaml`.

### 1. Health & Readiness (HealthServer)

#### GET /health

Liveness check – is the node process running?

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/plain

OK
```

#### GET /ready

Readiness check – is the node synced and ready to serve?

**Response (Ready):**
```http
HTTP/1.1 200 OK
Content-Type: text/plain

READY
```

**Response (Degraded):**
```http
HTTP/1.1 200 OK
Content-Type: text/plain

DEGRADED: Sync lag: 6 blocks
```

**Response (Not Ready):**
```http
HTTP/1.1 503 Service Unavailable
Content-Type: text/plain

NOT READY: Sync lag: 15 blocks; Low peer count: 2
```

#### GET /metrics

Prometheus-compatible metrics.

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/plain; version=0.0.4

# IAN Node Metrics
ian_contributions_received_total 1523
ian_contributions_processed_total 1520
ian_log_size 1520
ian_peer_count 12
ian_sync_lag 0
ian_uptime_seconds 86400
```

#### GET /status

Detailed node status as JSON.

**Response:**
```json
{
  "uptime_seconds": 86400,
  "metrics": {
    "counters": {
      "contributions_received_total": 1523,
      "contributions_processed_total": 1520
    },
    "gauges": {
      "log_size": 1520,
      "peer_count": 12,
      "sync_lag": 0
    }
  },
  "checks": {
    "sync": {"status": "healthy", "message": "Synced"},
    "peers": {"status": "healthy", "message": "12 peers connected"},
    "consensus": {"status": "healthy", "message": "Consensus: ACTIVE"}
  }
}
```

---

### 2. REST API (`IANApiServer`)

These endpoints are provided by `idi.ian.network.api.IANApiServer` and documented by `idi/ian/network/openapi.yaml`.

#### 2.1 Contribution

##### POST /api/v1/contribute

Submit a contribution.

**Request body (JSON):**
```json
{
  "goal_id": "MY_GOAL",
  "agent_pack": {
    "version": "1.0",
    "parameters": "SGVsbG8gV29ybGQ=",
    "metadata": {"strategy": "momentum"}
  },
  "proofs": {},
  "contributor_id": "alice",
  "seed": 12345
}
```

- `parameters` is base64-encoded bytes.
- `proofs` is an optional map of proof bundles (base64 strings).

**Success response (wrapped in `ApiResponse`):**
```json
{
  "success": true,
  "data": {
    "accepted": true,
    "reason": "ACCEPTED",
    "rejection_type": null,
    "metrics": {
      "reward": 1.23,
      "risk": 0.10,
      "complexity": 0.40
    },
    "log_index": 1520,
    "score": 95.5
  },
  "error": null,
  "timestamp": 1702300000000
}
```

**Error responses:**

- **400 Bad Request** – malformed input, missing fields, invalid values.
- **401 Unauthorized** – missing/invalid `X-API-Key` when API key is configured.
- **429 Too Many Requests** – per-IP rate limit exceeded.
- **500 Internal Server Error** – unexpected internal failure.

Example 429:
```json
{
  "success": false,
  "data": null,
  "error": "Rate limited",
  "timestamp": 1702300000000
}
```

##### Authentication

If `ApiConfig.api_key` is set, all write operations require:

```http
X-API-Key: your_secret_key
```

#### 2.2 Leaderboard

##### GET /api/v1/leaderboard/{goal_id}

Get top-K leaderboard entries for a goal.

**Query parameters:**
- `limit` (int, optional, default 100, max 1000)

**Response:**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "entries": [
      {
        "rank": 1,
        "pack_hash": "abc123...",
        "contributor_id": "contributor-abc",
        "score": 99.5,
        "log_index": 1520,
        "timestamp_ms": 1702300000000
      }
    ],
    "total": 150
  },
  "error": null,
  "timestamp": 1702300000000
}
```

#### 2.3 Status

##### GET /api/v1/status/{goal_id}

Get high-level statistics for a goal.

**Response:**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "log_size": 1520,
    "leaderboard_size": 150,
    "total_contributions": 2000,
    "accepted_contributions": 1800,
    "rejected_contributions": 200,
    "log_root": "abc123...",
    "leaderboard_root": "def456..."
  },
  "error": null,
  "timestamp": 1702300000000
}
```

#### 2.4 Log Info

##### GET /api/v1/log/{goal_id}

Get log root and size for a goal.

**Query parameters:**
- `from` (int, optional, default 0)
- `limit` (int, optional, default 100, max 1000)

**Response:**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "log_root": "abc123...",
    "log_size": 1520,
    "from_index": 0,
    "limit": 100
  },
  "error": null,
  "timestamp": 1702300000000
}
```

#### 2.5 Active Policy

##### GET /api/v1/policy/{goal_id}

Get the current best (active) policy for a goal.

**Response (no active policy):**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "active_policy": null
  },
  "error": null,
  "timestamp": 1702300000000
}
```

**Response (active policy present):**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "active_policy": {
      "pack_hash": "abc123...",
      "contributor_id": "contributor-xyz",
      "score": 99.9,
      "log_index": 1519,
      "timestamp_ms": 1702300000000
    }
  },
  "error": null,
  "timestamp": 1702300000000
}
```

#### 2.6 Membership Proof

##### GET /api/v1/proof/{goal_id}/{log_index}

Get a Merkle Mountain Range membership proof for a log entry.

**Path parameters:**
- `goal_id` – goal identifier
- `log_index` – 0-based log index (must be within `[0, log_size)`).

**Success response:**
```json
{
  "success": true,
  "data": {
    "goal_id": "MY_GOAL",
    "log_index": 1520,
    "leaf_hash": "abc123...",
    "siblings": [
      {"hash": "hash1", "is_right": true},
      {"hash": "hash2", "is_right": false}
    ],
    "peaks_bag": ["peak1", "peak2"],
    "mmr_size": 2048,
    "log_root": "rootabc..."
  },
  "error": null,
  "timestamp": 1702300000000
}
```

**Error (out of bounds):**
- `404 Not Found` with error message `"Log index X out of bounds [0, N)"`.

---

### Node Info Endpoints (HealthServer)

These endpoints are served by `HealthServer` on the same port as `/health` and `/ready`.
They are enabled when a `DecentralizedNode` registers providers on the health server.

#### GET /info

Get node information.

**Response:**
```json
{
  "node_id": "a1b2c3d4e5f6...",
  "public_key": "base64-ed25519-key",
  "addresses": [
    "tcp://192.168.1.100:9000"
  ],
  "capabilities": {
    "accepts_contributions": true,
    "serves_leaderboard": true,
    "serves_log_proofs": true,
    "goal_ids": ["my-goal-123"],
    "protocol_version": "1.0",
    "software_version": "0.1.0"
  },
  "timestamp": 1702300000000,
  "consensus_state": "ACTIVE",
  "running": true,
  "goal_id": "my-goal-123"
}
```

#### GET /peers

List connected peers with reputation summary.

**Response:**
```json
{
  "goal_id": "my-goal-123",
  "total": 2,
  "stats": {
    "total": 2,
    "banned": 0,
    "trusted": 1,
    "low_score": 0,
    "avg_score": 120.5
  },
  "peers": [
    {
      "node_id": "peer1...",
      "public_key": "base64-ed25519-key",
      "addresses": [
        "tcp://10.0.0.1:9000"
      ],
      "capabilities": {
        "accepts_contributions": true,
        "serves_leaderboard": true,
        "serves_log_proofs": true,
        "goal_ids": ["my-goal-123"],
        "protocol_version": "1.0",
        "software_version": "0.1.0"
      },
      "timestamp": 1702300000000,
      "peer_score": {
        "score": 150.0,
        "banned": false,
        "trusted": true
      }
    }
  ]
}
```

---

## P2P Protocol

The P2P protocol is implemented in `idi.ian.network.protocol` and `idi.ian.network.p2p_manager`.

### Message Types

`protocol.py` defines the following logical message types:

| Type | Enum | Direction | Purpose |
|------|------|-----------|---------|
| `HandshakeChallenge` | `HANDSHAKE_CHALLENGE` | P2P | Authenticated handshake initiation (identity + ephemeral key agreement) |
| `HandshakeResponse` | `HANDSHAKE_RESPONSE` | P2P | Handshake completion (identity proof + per-peer session key derivation) |
| `ContributionAnnounce` | `CONTRIBUTION_ANNOUNCE` | Gossip | Announce new contribution |
| `ContributionRequest` | `CONTRIBUTION_REQUEST` | Request | Request full contribution data |
| `ContributionResponse` | `CONTRIBUTION_RESPONSE` | Response | Return contribution data |
| `StateRequest` | `STATE_REQUEST` | Request | Request coordinator state |
| `StateResponse` | `STATE_RESPONSE` | Response | Return coordinator state |
| `PeerExchange` | `PEER_EXCHANGE` | Both | Exchange peer lists |
| `Ping` | `PING` | Both | Liveness check |
| `Pong` | `PONG` | Both | Liveness response |

### Wire Format (Current Implementation)

The **current** wire format is a simple length-prefixed JSON frame:

```text
+----------------------+--------------------------+
| Length (4 bytes, BE) | JSON payload (UTF-8)     |
+----------------------+--------------------------+
```

- `Length` – big-endian uint32 of JSON payload size.
- `JSON payload` – `Message.to_dict()` output, including:
  - `type`: string enum value (e.g. `"contribution_announce"`).
  - `sender_id`, `timestamp`, `nonce`.
  - Optional `signature` (base64) and type-specific fields.

**Example:**
```python
from idi.ian.network.protocol import Ping, Message

ping = Ping(sender_id="node_abc123")
wire = ping.to_wire()
parsed = Message.from_wire(wire)
assert isinstance(parsed, Ping)
```

> **Note:** Earlier design docs referenced a binary header with `Magic` and `Type` bytes. That design is reserved for a potential future Protocol v2. The current implementation uses the simpler JSON framing above.

### Authentication & Handshake

- Each message includes a `nonce` and optional `signature` over the JSON fields.
- `NodeIdentity` (Ed25519) is used to sign and verify messages.
- `P2PManager` performs a handshake using `HandshakeChallenge` and `HandshakeResponse` messages.
- The handshake establishes a per-peer `session_key` (32 bytes) which can be used by higher layers for authenticated payloads (e.g., FrontierSync authenticated IBLT exchange).

### Rate Limiting

- Per-peer token-bucket limiter (see `TokenBucketRateLimiter` in `p2p_manager.py`).
- Configurable via `P2PConfig.max_messages_per_second` and `rate_limit_burst`.
- Violations increment peer statistics and can interact with peer scoring.

---

## Python SDK

### Basic Usage (Decentralized Node)

```python
from idi.ian.network import DecentralizedNode, DecentralizedNodeConfig
from idi.ian.network import NodeIdentity
from idi.ian.models import GoalSpec

# Create identity
identity = NodeIdentity.generate()

# Define goal
goal_spec = GoalSpec(
    goal_id="my-goal",
    name="My Optimization Goal",
    description="Optimize for X metric",
)

# Create node
config = DecentralizedNodeConfig(
    listen_port=9000,
    health_port=8080,
    seed_addresses=["tcp://seed1.ian.network:9000"],
)

node = DecentralizedNode(
    goal_spec=goal_spec,
    identity=identity,
    config=config,
)

# Start node
await node.start()

# Submit contribution
from idi.ian.models import Contribution

contrib = Contribution(
    goal_id="my-goal",
    contributor_id=identity.node_id,
    content={"code": "solution"},
)

success, reason = await node.submit_contribution(contrib)
print(f"Accepted: {success}, Reason: {reason}")

# Stop node
await node.stop()
```

### Production Utilities

```python
from idi.ian.network import (
    CircuitBreaker,
    StructuredLogger,
    PeerScoreManager,
    TLSConfig,
)

# Circuit breaker for external APIs
breaker = CircuitBreaker("tau_api")

async with breaker:
    result = await call_tau_api()

# Structured logging
logger = StructuredLogger("idi.ian.node")
logger.info("contribution_received", pack_hash="abc", size=1024)

# Peer scoring with persistence
scores = PeerScoreManager(persist_path="/data/peer_scores.json")
scores.load()
scores.record_event("peer123", "valid_message")
scores.save()

# TLS configuration for P2P
identity = NodeIdentity.generate()
config = P2PConfig(listen_port=9000)

tls = TLSConfig.generate(node_id=identity.node_id, output_dir="/data/certs")

p2p = P2PManager(
    identity=identity,
    config=config,
    tls_config=tls,  # Enables TLS for P2P
)
```

---

## Configuration

### DecentralizedNodeConfig

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DecentralizedNodeConfig:
    # Network
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    seed_addresses: List[str] = field(default_factory=list)
    max_peers: int = 50
    
    # Consensus
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    
    # Economics
    economics: EconomicConfig = field(default_factory=EconomicConfig)
    
    # Evaluation
    evaluation: EvaluationQuorumConfig = field(default_factory=EvaluationQuorumConfig)
    
    # Tau integration
    tau_commit_interval: float = 300.0  # 5 minutes
    tau_commit_threshold: int = 100     # contributions
    
    # Node capabilities
    accept_contributions: bool = True
    serve_evaluations: bool = False
    commit_to_tau: bool = True
    
    # Production
    health_port: int = 8080
    enable_health_server: bool = True
    state_persist_path: Optional[str] = None
    peer_scores_path: Optional[str] = None
    min_peers_for_ready: int = 3
    max_sync_lag_for_ready: int = 10
    
    # Logging
    log_level: str = "INFO"
```

### Environment Variables (Typical)

| Variable | Default | Description |
|----------|---------|-------------|
| `IAN_LISTEN_HOST` | "0.0.0.0" | Default listen host (used as fallback for network services) |
| `IAN_API_PORT` | 8000 | HTTP API port for `idi.ian.cli node start` |
| `IAN_P2P_PORT` | 9000 | P2P TCP port for `idi.ian.cli node start` |
| `IAN_WS_PORT` | 9001 | WebSocket port for `idi.ian.cli node start` |
| `IAN_SEED_NODES` | "" | Comma-separated seed addresses (host:port) |
| `IAN_LOG_LEVEL` | "INFO" | Log level |
| `IAN_TAU_HOST` | "localhost" | Tau Net host |
| `IAN_TAU_PORT` | 10330 | Tau Net port |

`DecentralizedNode` exposes a separate `HealthServer` (with `/ready`) and uses `DecentralizedNodeConfig.health_port` (default 8080).

---

## Error Codes

### Contribution Rejection Reasons (Logical)

The coordinator can reject contributions for various reasons. These are exposed via:

- `data.reason` – human-readable string.
- `data.rejection_type` – enum value (see `openapi.yaml`).

Common reasons include:

| Code | Reason | Description |
|------|--------|-------------|
| `DUPLICATE` | Duplicate | Already in log |
| `INVALID_SIGNATURE` | Bad signature | Signature verification failed |
| `INVALID_GOAL` | Wrong goal | Goal ID mismatch |
| `INVARIANT_VIOLATION` | Invariant | Goal invariant check failed |
| `PROOF_INVALID` | Bad proof | ZK proof verification failed |
| `SCORE_TOO_LOW` | Low score | Below minimum threshold |
| `RATE_LIMITED` | Rate limit | Too many submissions |

### P2P Error Conditions

P2P errors are not currently exposed as a separate error code enum, but typical conditions include:

- Timeout waiting for response.
- Rate limited by per-peer limiter.
- Invalid message (malformed JSON or signature failure).
- Authentication failure (invalid NodeIdentity signature).
- Peer banned by reputation system.

### HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not found (e.g., invalid log index) |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal server error |
| 503 | Service unavailable (not ready) |

---

## WebSocket API (Conceptual)

A WebSocket interface can be exposed via `websocket_transport.py` (or a gateway) to stream events.

**Example (conceptual):**

```javascript
const ws = new WebSocket("ws://localhost:9001/subscribe");

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "subscribe",
    channels: ["contributions", "leaderboard"]
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data);
};
```

### Possible Event Types

| Event | Channel | Description |
|-------|---------|-------------|
| `contribution_added` | contributions | New contribution accepted |
| `leaderboard_update` | leaderboard | Leaderboard changed |
| `peer_connected` | peers | Peer connected |
| `peer_disconnected` | peers | Peer disconnected |
| `sync_progress` | sync | Sync progress update |
| `tau_commit` | tau | Committed to Tau Net |

> WebSocket APIs are deployment-dependent and may be provided via a separate gateway service in front of the node.
