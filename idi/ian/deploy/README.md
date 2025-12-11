# IAN Deployment Guide

This directory contains deployment configurations for running IAN nodes in various environments.

## Quick Start

### Local Development (Docker Compose)

```bash
# Start a 3-node cluster with monitoring
cd deploy
docker-compose up -d

# View logs
docker-compose logs -f ian-seed

# Stop
docker-compose down
```

Services:
- **Seed Node**: http://localhost:8000 (API), ws://localhost:9001 (WebSocket)
- **Node 1**: http://localhost:8001
- **Node 2**: http://localhost:8002
- **Evaluator**: http://localhost:8003
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Kubernetes

```bash
# Apply manifests
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl -n ian get pods

# View logs
kubectl -n ian logs -f deployment/ian-node

# Scale
kubectl -n ian scale deployment/ian-node --replicas=5
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IAN_NODE_TYPE` | Node type: seed, full, evaluator | full |
| `IAN_SEED_NODES` | Comma-separated seed addresses | |
| `IAN_GOAL_ID` | Goal ID to serve | DEMO_TRADING_AGENT |
| `IAN_LOG_LEVEL` | Log level | INFO |
| `IAN_P2P_PORT` | P2P TCP port | 9000 |
| `IAN_WS_PORT` | WebSocket port | 9001 |
| `IAN_API_PORT` | REST API port | 8000 |
| `IAN_TAU_HOST` | Tau Testnet host | localhost |
| `IAN_TAU_PORT` | Tau Testnet port | 10330 |
| `IAN_API_KEY` | API key for authentication | |

### Config Files

- `config/production.yaml` - Production settings (secure, optimized)
- `config/development.yaml` - Development settings (relaxed, verbose)

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Load Balancer               │
                    │         (Ingress/LoadBalancer)           │
                    └───────────────┬─────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
    ┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
    │   IAN Node    │       │   IAN Node    │       │   IAN Node    │
    │   (Pod)       │       │   (Pod)       │       │   (Pod)       │
    │               │◄─────►│               │◄─────►│               │
    │  - API        │  P2P  │  - API        │  P2P  │  - API        │
    │  - WebSocket  │       │  - WebSocket  │       │  - WebSocket  │
    │  - P2P        │       │  - P2P        │       │  - P2P        │
    └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────┐
                    │            Seed Nodes                  │
                    │      (StatefulSet - 3 replicas)        │
                    └───────────────┬───────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────┐
                    │            Tau Testnet                 │
                    │            (Layer 1)                   │
                    └───────────────────────────────────────┘
```

## Node Types

### Seed Node
- Provides peer discovery bootstrap
- Stable, well-known addresses
- Should be geographically distributed
- Run as StatefulSet for stable DNS

### Full Node
- Processes contributions
- Maintains full state
- Participates in consensus
- Can commit to Tau Net (if bonded)

### Evaluator Node
- Runs contribution evaluations
- Participates in evaluation quorum
- Requires stake to participate

### Light Node
- Minimal state (headers only)
- Trusts full nodes for proofs
- Lower resource requirements

## Scaling

### Horizontal Scaling

IAN nodes scale horizontally. Add more replicas as load increases:

```bash
kubectl -n ian scale deployment/ian-node --replicas=10
```

Or use HPA for automatic scaling based on CPU/memory.

### Resource Requirements

| Node Type | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Seed | 100m-500m | 256MB-512MB | 5GB |
| Full | 200m-1000m | 512MB-1GB | 10GB |
| Evaluator | 500m-2000m | 1GB-4GB | 10GB |

## Security

### TLS

For production, enable TLS:

1. Obtain certificates
2. Create Kubernetes secrets
3. Configure Ingress with TLS

### Network Policies

The deployment includes NetworkPolicies that:
- Allow API/WebSocket from anywhere
- Restrict P2P to cluster-internal
- Allow egress to Tau Net

### API Authentication

Enable API key authentication in production:

```yaml
security:
  api_key_required: true
```

Set via environment:
```bash
export IAN_API_KEY=your-secret-key
```

## Monitoring

### Metrics

IAN exposes Prometheus metrics at `/metrics`:

- `ian_contributions_total` - Total contributions processed
- `ian_leaderboard_size` - Current leaderboard size
- `ian_consensus_state` - Current consensus state
- `ian_peer_count` - Number of connected peers

### Grafana Dashboards

Import dashboards from `grafana/dashboards/`:
- `ian-overview.json` - Overall system health
- `ian-consensus.json` - Consensus metrics
- `ian-economics.json` - Economic metrics

## Troubleshooting

### Node not connecting to peers

1. Check seed node addresses
2. Verify network connectivity
3. Check firewall rules for port 9000

### Consensus divergence

1. Check `ian_consensus_state` metric
2. Review logs for state mismatches
3. May need to sync from peers

### Tau commit failures

1. Verify Tau Net connectivity
2. Check committer bond status
3. Review transaction logs
