# IAN Network Layer

The network layer provides peer-to-peer communication for IAN nodes, enabling distributed coordination of agent contributions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         IAN Node                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ NodeIdentity│  │  Discovery  │  │      API Server         │  │
│  │  (Ed25519)  │  │ (Seed-based)│  │  (aiohttp REST)         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          │                                       │
│                    ┌─────┴─────┐                                 │
│                    │ Transport │                                 │
│                    │   (TCP)   │                                 │
│                    └───────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

### `node.py` — Node Identity

Manages cryptographic identity for IAN nodes using Ed25519 keys.

```python
from idi.ian.network.node import NodeIdentity, NodeInfo

# Generate new identity
identity = NodeIdentity.generate()

# Sign and verify data
signature = identity.sign(b"message")
assert identity.verify(b"message", signature)

# Create signed node info for peer exchange
info = identity.create_node_info(
    addresses=["tcp://192.168.1.10:9000"],
    capabilities=NodeCapabilities(goal_ids=["GOAL_1", "GOAL_2"]),
)

# Persist identity
identity.save("node_identity.json")
loaded = NodeIdentity.load("node_identity.json")
```

**Security Note:** If the `cryptography` library is not installed, a warning is emitted and a fallback HMAC-based implementation is used. This fallback is **NOT suitable for production** as it cannot provide public-key verification.

### `protocol.py` — P2P Messages

Defines the message types for IAN P2P communication.

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `ContributionAnnounce` | Broadcast | Announce new contribution to peers |
| `ContributionRequest` | Request | Request full contribution data |
| `ContributionResponse` | Response | Return contribution data |
| `StateRequest` | Request | Request goal state (log root, leaderboard) |
| `StateResponse` | Response | Return goal state |
| `PeerExchange` | Broadcast | Share known peers |
| `Ping` / `Pong` | P2P | Liveness check with nonce |

```python
from idi.ian.network.protocol import Ping, Pong, Message

# Create and serialize message
ping = Ping(sender_id="node_abc123")
wire_bytes = ping.to_wire()  # Length-prefixed JSON

# Parse message
msg = Message.from_wire(wire_bytes)
assert isinstance(msg, Ping)
```

**Wire Format:**
```
┌──────────────────┬─────────────────────────────────────┐
│  Length (4 bytes)│  JSON payload (UTF-8)               │
│  (big-endian)    │                                     │
└──────────────────┴─────────────────────────────────────┘
```

### `discovery.py` — Peer Discovery

Seed-node based peer discovery with authenticated peer exchange.

```python
from idi.ian.network.discovery import SeedNodeDiscovery

discovery = SeedNodeDiscovery(
    identity=node_identity,
    seed_addresses=["tcp://seed1.example.com:9000", "tcp://seed2.example.com:9000"],
    max_peers=50,
)

# Start discovery (connects to seeds, requests peers)
await discovery.start()

# Get known peers
peers = discovery.get_peers()

# Handle incoming peer exchange (with signature verification)
discovery.handle_peer_exchange(peer_exchange_message)
```

**Security:** Peer exchange messages are signed when the `cryptography` library is available. Unsigned messages from unknown senders are rejected.

### `transport.py` — TCP Transport

Manages TCP connections with connection limits and read timeouts.

```python
from idi.ian.network.transport import TCPTransport

transport = TCPTransport(
    max_connections=1024,      # Global connection limit
    max_connections_per_ip=64, # Per-IP limit (DoS protection)
)

await transport.start("0.0.0.0", 9000)

# Connect to peer
conn_id = await transport.connect("192.168.1.20:9000")

# Send message
await transport.send(conn_id, message.to_wire())

# Messages arrive via callback
transport.on_message = lambda conn_id, data: handle_message(data)
```

**Security Features:**
- **Connection limits:** Prevents resource exhaustion from too many connections
- **Per-IP limits:** Mitigates single-source DoS
- **Read timeouts:** 60-second timeout on reads prevents slow-loris attacks
- **IPv6 support:** Handles `[::1]:port` format addresses

### `api.py` — REST API

Provides HTTP API for external clients using aiohttp.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/contribute` | Submit contribution |
| `GET` | `/api/v1/leaderboard/{goal_id}` | Get leaderboard |
| `GET` | `/api/v1/status/{goal_id}` | Get goal status |
| `GET` | `/api/v1/log/{goal_id}` | Get log entries |
| `GET` | `/api/v1/policy/{goal_id}` | Get active policy |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/api/v1/proof/{goal_id}/{log_index}` | Get membership proof |
| `GET` | `/api/v1/openapi.json` | OpenAPI specification |

```python
from idi.ian.network.api import create_api_app, ApiConfig

config = ApiConfig(
    host="0.0.0.0",
    port=8080,
    api_key="secret123",       # Optional authentication
    rate_limit_per_ip=100,     # Requests per minute per IP
)

app = create_api_app(coordinator, config)
await aiohttp.web.run_app(app, host=config.host, port=config.port)
```

#### Authentication

If `api_key` is configured, clients must include it in the header:

```bash
curl -X POST http://localhost:8080/api/v1/contribute \
  -H "X-API-Key: secret123" \
  -H "Content-Type: application/json" \
  -d '{"goal_id": "MY_GOAL", ...}'
```

#### Rate Limiting

- Per-IP rate limiting with configurable limit (default: 100 req/min)
- Bounded tracking (max 10,000 IPs) with LRU eviction

#### OpenAPI Documentation

The API is fully documented with OpenAPI 3.0. Access the spec:

```bash
# JSON format
curl http://localhost:8080/api/v1/openapi.json

# Or view the YAML source
cat idi/ian/network/openapi.yaml
```

You can import this into Swagger UI, Postman, or other API tools.

## Security Considerations

See [SECURITY.md](../SECURITY.md) for comprehensive security documentation.

### Key Points

1. **Use `cryptography` library in production** — Fallback is HMAC-based and insecure
2. **Configure connection limits** — Default 1024 global, 64 per-IP
3. **Enable API key authentication** — Prevent unauthorized contributions
4. **Use TLS in production** — The transport layer doesn't encrypt; use a reverse proxy

## Testing

```bash
# Run network tests
pytest idi/ian/tests/test_network.py -v

# Specific test classes
pytest idi/ian/tests/test_network.py::TestNodeIdentity -v
pytest idi/ian/tests/test_network.py::TestProtocol -v
pytest idi/ian/tests/test_network.py::TestDiscovery -v
pytest idi/ian/tests/test_network.py::TestApiHandlers -v
```

## Configuration

Environment variables:

```bash
export IAN_NETWORK_HOST=0.0.0.0
export IAN_NETWORK_PORT=9000
export IAN_API_PORT=8080
export IAN_API_KEY=your_secret_key
export IAN_NETWORK_MAX_CONNECTIONS=1024
export IAN_NETWORK_MAX_CONNECTIONS_PER_IP=64
```
