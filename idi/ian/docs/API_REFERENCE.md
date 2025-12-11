# IAN API Reference

Complete API documentation for the IAN (Intelligent Agent Network) decentralized system.

## Table of Contents

1. [Overview](#overview)
2. [HTTP API](#http-api)
3. [P2P Protocol](#p2p-protocol)
4. [Python SDK](#python-sdk)
5. [Configuration](#configuration)
6. [Error Codes](#error-codes)

---

## Overview

IAN provides multiple interfaces for interaction:

| Interface | Port | Purpose |
|-----------|------|---------|
| HTTP API | 8080 | Health, metrics, node info |
| P2P TCP | 9000 | Peer communication |
| WebSocket | 9001 | Real-time subscriptions |

### Authentication

- **P2P**: Ed25519 signature-based authentication with challenge-response handshake
- **HTTP**: Optional API key via `X-API-Key` header
- **WebSocket**: Token-based authentication on connect

---

## HTTP API

### Health Endpoints

#### GET /health

Liveness check - is the node process running?

**Response:**
```
HTTP/1.1 200 OK
Content-Type: text/plain

OK
```

#### GET /ready

Readiness check - is the node synced and ready to serve?

**Response (Ready):**
```
HTTP/1.1 200 OK
Content-Type: text/plain

READY
```

**Response (Not Ready):**
```
HTTP/1.1 503 Service Unavailable
Content-Type: text/plain

NOT READY: Sync lag: 15 blocks; Low peer count: 2
```

#### GET /metrics

Prometheus-compatible metrics.

**Response:**
```
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

### Node Info Endpoints

#### GET /info

Get node information.

**Response:**
```json
{
  "node_id": "a1b2c3d4e5f6...",
  "goal_id": "my-goal-123",
  "version": "1.0.0",
  "capabilities": {
    "accepts_contributions": true,
    "serves_leaderboard": true,
    "serves_log_proofs": true
  },
  "addresses": [
    "tcp://192.168.1.100:9000"
  ]
}
```

#### GET /peers

List connected peers.

**Response:**
```json
{
  "peers": [
    {
      "node_id": "peer1...",
      "address": "tcp://10.0.0.1:9000",
      "connected_at": 1702300000,
      "score": 150.0,
      "state": "ACTIVE"
    }
  ],
  "total": 12,
  "trusted": 5,
  "banned": 1
}
```

---

### Contribution Endpoints

#### POST /contribution

Submit a contribution.

**Request:**
```json
{
  "goal_id": "my-goal-123",
  "contributor_id": "contributor-abc",
  "content": {
    "code": "def solve(): return 42",
    "metadata": {"language": "python"}
  },
  "signature": "base64-encoded-signature"
}
```

**Response (Success):**
```json
{
  "accepted": true,
  "pack_hash": "abc123...",
  "log_index": 1520,
  "metrics": {
    "score": 95.5,
    "latency_ms": 12.3
  }
}
```

**Response (Rejected):**
```json
{
  "accepted": false,
  "reason": "DUPLICATE",
  "message": "Contribution already exists in log"
}
```

#### GET /contribution/{pack_hash}

Get contribution by pack hash.

**Response:**
```json
{
  "pack_hash": "abc123...",
  "log_index": 1520,
  "contributor_id": "contributor-abc",
  "timestamp_ms": 1702300000000,
  "metrics": {
    "score": 95.5
  }
}
```

---

### Leaderboard Endpoints

#### GET /leaderboard

Get current leaderboard.

**Query Parameters:**
- `limit` (int, default 10): Number of entries
- `offset` (int, default 0): Offset for pagination

**Response:**
```json
{
  "entries": [
    {
      "rank": 1,
      "pack_hash": "abc123...",
      "contributor_id": "contributor-abc",
      "score": 99.5,
      "timestamp_ms": 1702300000000
    }
  ],
  "total": 150,
  "leaderboard_root": "def456..."
}
```

---

### Proof Endpoints

#### GET /proof/log/{index}

Get Merkle proof for log entry.

**Response:**
```json
{
  "log_index": 1520,
  "pack_hash": "abc123...",
  "proof": {
    "root": "rootabc...",
    "path": ["hash1", "hash2", "hash3"],
    "leaf_index": 1520
  }
}
```

#### GET /proof/leaderboard/{pack_hash}

Get Merkle proof for leaderboard entry.

**Response:**
```json
{
  "pack_hash": "abc123...",
  "proof": {
    "root": "lbrootabc...",
    "path": ["hash1", "hash2"],
    "position": 0
  }
}
```

---

## P2P Protocol

### Message Types

| Type | Code | Direction | Purpose |
|------|------|-----------|---------|
| `HANDSHAKE` | 0x01 | Both | Initial connection |
| `HANDSHAKE_ACK` | 0x02 | Both | Handshake response |
| `PING` | 0x10 | Both | Keepalive |
| `PONG` | 0x11 | Both | Keepalive response |
| `CONTRIBUTION_ANNOUNCE` | 0x20 | Out | Announce new contribution |
| `CONTRIBUTION_REQUEST` | 0x21 | Out | Request contribution data |
| `CONTRIBUTION_RESPONSE` | 0x22 | In | Contribution data |
| `STATE_REQUEST` | 0x30 | Out | Request state info |
| `STATE_RESPONSE` | 0x31 | In | State info |
| `SYNC_REQUEST` | 0x40 | Out | Request log range |
| `SYNC_RESPONSE` | 0x41 | In | Log entries |
| `PEER_EXCHANGE` | 0x50 | Both | Share peer list |
| `FRAUD_PROOF` | 0x60 | Out | Submit fraud proof |

### Message Format

```
+--------+--------+--------+--------+
| Magic  | Type   | Length (4 bytes)|
+--------+--------+--------+--------+
| Payload (variable length)         |
+-----------------------------------+
| Signature (64 bytes, optional)    |
+-----------------------------------+
```

- **Magic**: `0x49414E` ("IAN")
- **Type**: Message type byte
- **Length**: Payload length (big-endian uint32)
- **Payload**: JSON or bytecode encoded data
- **Signature**: Ed25519 signature over (type + length + payload)

### Handshake Protocol

```
Client                          Server
  |                               |
  |------ HANDSHAKE ------------->|
  |   {node_id, version, nonce}   |
  |                               |
  |<----- HANDSHAKE_ACK ----------|
  |   {node_id, version,          |
  |    challenge, signature}      |
  |                               |
  |------ CHALLENGE_RESPONSE ---->|
  |   {signed_challenge}          |
  |                               |
  |<----- CONNECTION_READY -------|
  |   {verified: true}            |
```

### Rate Limiting

- Default: 100 messages/second per peer
- Burst: 20 messages
- Violations result in score penalty and potential disconnect

---

## Python SDK

### Basic Usage

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

result = await node.submit_contribution(contrib)
print(f"Accepted: {result.accepted}, Index: {result.log_index}")

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
breaker = CircuitBreaker("tau_api", failure_threshold=5)

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

# TLS configuration
tls = TLSConfig.generate(node_id="mynode", output_dir="/data/certs")
```

---

## Configuration

### DecentralizedNodeConfig

```python
@dataclass
class DecentralizedNodeConfig:
    # Network
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    seed_addresses: List[str] = []
    max_peers: int = 50
    
    # Consensus
    consensus: ConsensusConfig = ConsensusConfig()
    
    # Economics
    economics: EconomicConfig = EconomicConfig()
    
    # Evaluation
    evaluation: EvaluationQuorumConfig = EvaluationQuorumConfig()
    
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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IAN_LISTEN_PORT` | 9000 | P2P listen port |
| `IAN_HEALTH_PORT` | 8080 | Health endpoint port |
| `IAN_SEED_NODES` | "" | Comma-separated seed addresses |
| `IAN_DATA_DIR` | "/data" | Data directory |
| `IAN_LOG_LEVEL` | "INFO" | Log level |
| `IAN_TAU_HOST` | "localhost" | Tau Net host |
| `IAN_TAU_PORT` | 12345 | Tau Net port |

---

## Error Codes

### Contribution Rejection Reasons

| Code | Reason | Description |
|------|--------|-------------|
| `DUPLICATE` | Duplicate | Already in log |
| `INVALID_SIGNATURE` | Bad signature | Signature verification failed |
| `INVALID_GOAL` | Wrong goal | Goal ID mismatch |
| `INVARIANT_VIOLATION` | Invariant | Goal invariant check failed |
| `PROOF_INVALID` | Bad proof | ZK proof verification failed |
| `SCORE_TOO_LOW` | Low score | Below minimum threshold |
| `RATE_LIMITED` | Rate limit | Too many submissions |

### P2P Error Codes

| Code | Error | Description |
|------|-------|-------------|
| `E_TIMEOUT` | Timeout | Request timed out |
| `E_RATE_LIMIT` | Rate limited | Too many messages |
| `E_INVALID_MSG` | Invalid message | Malformed message |
| `E_AUTH_FAIL` | Auth failed | Handshake failed |
| `E_PEER_BANNED` | Peer banned | Peer is banned |
| `E_CIRCUIT_OPEN` | Circuit open | Circuit breaker open |

### HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not found |
| 429 | Rate limited |
| 500 | Internal server error |
| 503 | Service unavailable (not ready) |

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket("ws://localhost:9001/subscribe");

ws.onopen = () => {
  // Subscribe to events
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

### Event Types

| Event | Channel | Description |
|-------|---------|-------------|
| `contribution_added` | contributions | New contribution accepted |
| `leaderboard_update` | leaderboard | Leaderboard changed |
| `peer_connected` | peers | Peer connected |
| `peer_disconnected` | peers | Peer disconnected |
| `sync_progress` | sync | Sync progress update |
| `tau_commit` | tau | Committed to Tau Net |

### Event Format

```json
{
  "type": "contribution_added",
  "timestamp": 1702300000000,
  "data": {
    "pack_hash": "abc123...",
    "log_index": 1520,
    "contributor_id": "contributor-abc",
    "score": 95.5
  }
}
```
