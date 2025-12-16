# IAN Operator Runbook

Operational guide for running and maintaining IAN (Intelligent Agent Network) nodes in production.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Deployment](#deployment)
3. [Monitoring](#monitoring)
4. [Common Operations](#common-operations)
5. [Troubleshooting](#troubleshooting)
6. [Emergency Procedures](#emergency-procedures)
7. [Maintenance](#maintenance)

---

## Quick Reference

### Important Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| Health | `http://localhost:8000/health` | Liveness check |
| Metrics | `http://localhost:8000/metrics` | Prometheus metrics |
| OpenAPI | `http://localhost:8000/api/v1/openapi.json` | API schema |
| Goal Status | `http://localhost:8000/api/v1/status/{goal_id}` | Goal status |

### Key Files

| File | Location | Purpose |
|------|----------|---------|
| Node state | `/data/node_state.json` | Persisted node state |
| Peer scores | `/data/peer_scores.json` | Peer reputation data |
| Node identity | `/data/identity.json` | Ed25519 keys |
| TLS cert | `/data/certs/node_cert.pem` | TLS certificate |
| TLS key | `/data/certs/node_key.pem` | TLS private key |
| Logs | `/var/log/ian/node.log` | Application logs |

### Key Metrics

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `ian_sync_lag` | > 10 | Blocks behind network |
| `ian_peer_count` | < 3 | Connected peers |
| `ian_contributions_rejected_total` | spike | Rejection rate |
| `ian_rate_limit_violations_total` | spike | Potential attack |

---

## Deployment

### Prerequisites

- Python 3.10+ (3.12 recommended; production images use 3.12)
- 2+ CPU cores
- 4+ GB RAM
- 50+ GB SSD storage
- Network: Ports 8000, 9000, 9001 accessible

### Docker Deployment

```bash
# Pull image
docker pull ian-network/ian-node:latest

# Run node
docker run -d \
  --name ian-node \
  -p 8000:8000 \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /data/ian:/data \
  -e IAN_SEED_NODES="tcp://seed1.ian.network:9000,tcp://seed2.ian.network:9000" \
  -e IAN_LOG_LEVEL="INFO" \
  ian-network/ian-node:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  ian-node:
    image: ian-network/ian-node:latest
    ports:
      - "8000:8000"
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data:/data
    environment:
      - IAN_SEED_NODES=tcp://seed1.ian.network:9000
      - IAN_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -n ian
kubectl logs -f deployment/ian-node -n ian
```

### Verify Deployment

```bash
# Check health
curl http://localhost:8000/health
# Expected: OK

# Check metrics
curl http://localhost:8000/metrics

# Check goal status
curl http://localhost:8000/api/v1/status/DEMO_TRADING_AGENT | jq
```

---

## Monitoring

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'ian-nodes'
    static_configs:
      - targets:
        - 'ian-node-1:8000'
        - 'ian-node-2:8000'
        - 'ian-node-3:8000'
    metrics_path: /metrics
    scrape_interval: 15s
```

### Key Dashboards

#### Node Health Dashboard

```
Panels:
1. Uptime (ian_uptime_seconds)
2. Sync Status (ian_sync_lag)
3. Peer Count (ian_peer_count)
4. Consensus State
5. Log Size (ian_log_size)
```

#### Throughput Dashboard

```
Panels:
1. Contributions/minute (rate(ian_contributions_received_total[1m]))
2. Acceptance Rate (ian_contributions_processed / ian_contributions_received)
3. Tau Commits (ian_tau_commits_total)
4. Message Rate (rate(ian_messages_received_total[1m]))
```

#### Security Dashboard

```
Panels:
1. Rate Limit Violations (ian_rate_limit_violations_total)
2. Invalid Messages (ian_invalid_messages_total)
3. Banned Peers (ian_peer_score_banned)
4. Circuit Breaker Status
```

### Alert Rules

```yaml
groups:
  - name: ian-alerts
    rules:
      # Node unhealthy
      - alert: IANNodeUnhealthy
        expr: up{job="ian-nodes"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "IAN node {{ $labels.instance }} is down"

      # Sync lag
      - alert: IANSyncLag
        expr: ian_sync_lag > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "IAN node {{ $labels.instance }} is {{ $value }} blocks behind"

      # Low peer count
      - alert: IANLowPeers
        expr: ian_peer_count < 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "IAN node {{ $labels.instance }} has only {{ $value }} peers"

      # High rejection rate
      - alert: IANHighRejectionRate
        expr: rate(ian_contributions_rejected_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High contribution rejection rate on {{ $labels.instance }}"

      # Rate limit violations (potential attack)
      - alert: IANRateLimitViolations
        expr: rate(ian_rate_limit_violations_total[5m]) > 100
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Possible DoS attack on {{ $labels.instance }}"
```

### Log Analysis

```bash
# View recent logs
docker logs --tail 100 ian-node

# Filter errors
docker logs ian-node 2>&1 | grep -i error

# Watch for specific events
docker logs -f ian-node 2>&1 | grep "contribution"

# Structured log query (if using JSON logging)
docker logs ian-node 2>&1 | jq 'select(.level == "ERROR")'
```

---

## Common Operations

### Restart Node

```bash
# Graceful restart (preserves state)
docker restart ian-node

# Or with docker-compose
docker-compose restart ian-node
```

### Check Node State

```bash
# Get current status
curl -s http://localhost:8000/api/v1/status/DEMO_TRADING_AGENT | jq

# Check log size
curl -s http://localhost:8000/metrics | grep ian_log_size
```

### Update Configuration

```bash
# Edit environment
docker-compose down
vim docker-compose.yml
docker-compose up -d
```

### Backup Data

```bash
# Stop node (optional but recommended)
docker stop ian-node

# Backup data directory
tar -czvf ian-backup-$(date +%Y%m%d).tar.gz /data/ian/

# Restart node
docker start ian-node
```

### Restore from Backup

```bash
# Stop node
docker stop ian-node

# Restore data
tar -xzvf ian-backup-20231201.tar.gz -C /

# Start node
docker start ian-node
```

### Add/Remove Seed Nodes

```bash
# Update seed list
export IAN_SEED_NODES="tcp://new-seed:9000,tcp://seed2:9000"
docker-compose up -d
```

### Manual Peer Management

```bash
# View peer scores
cat /data/ian/peer_scores.json | jq '.peers | to_entries | sort_by(.value.score) | reverse | .[0:10]'

# Manually ban a peer (edit peer_scores.json)
# Set banned_until to future timestamp
```

---

## Troubleshooting

### Node Won't Start

**Symptoms:** Container exits immediately or health check fails

**Diagnosis:**
```bash
# Check logs
docker logs ian-node

# Common issues:
# - Port already in use
# - Invalid configuration
# - Missing data directory
# - Permission issues
```

**Solutions:**
```bash
# Port conflict
netstat -tlnp | grep 9000
# Kill conflicting process or change port

# Permission issues
chown -R 1000:1000 /data/ian
chmod 700 /data/ian/certs
```

### Node Not Syncing

**Symptoms:** `ian_sync_lag` stays high

**Diagnosis:**
```bash
# Check peer count
curl -s http://localhost:8000/metrics | grep ian_peer_count

# Check if seeds are reachable
nc -zv seed1.ian.network 9000

# Check network
docker exec ian-node ping seed1.ian.network
```

**Solutions:**
```bash
# Add more seed nodes
# Check firewall rules
# Verify DNS resolution
# Check if peers are banning us (peer score)
```

### High Memory Usage

**Symptoms:** Container using excessive memory

**Diagnosis:**
```bash
# Check memory
docker stats ian-node

# Check log size (can grow large)
curl -s http://localhost:8000/metrics | grep ian_log_size
```

**Solutions:**
```bash
# Set memory limit in docker-compose
deploy:
  resources:
    limits:
      memory: 4G

# Check for memory leaks in logs
# Consider pruning old peer scores
```

### Connection Refused Errors

**Symptoms:** Peers can't connect to node

**Diagnosis:**
```bash
# Check if port is listening
netstat -tlnp | grep 9000

# Check firewall
iptables -L -n | grep 9000

# Test from outside
nc -zv your-node-ip 9000
```

**Solutions:**
```bash
# Open firewall port
ufw allow 9000/tcp

# Check docker port mapping
docker port ian-node

# Verify listen address (0.0.0.0 vs 127.0.0.1)
```

### Circuit Breaker Open

**Symptoms:** Tau Net commits failing, circuit breaker errors in logs

**Diagnosis:**
```bash
# Check logs for circuit breaker state
docker logs ian-node 2>&1 | grep "Circuit breaker"

# Check Tau Net connectivity
nc -zv tau-node 12345
```

**Solutions:**
```bash
# Wait for circuit breaker timeout (default 30s)
# Check Tau Net status
# Verify Tau Net configuration
# Manual reset (restart node)
```

### Peer Banned Unexpectedly

**Symptoms:** Legitimate peers being disconnected

**Diagnosis:**
```bash
# Check peer scores
cat /data/ian/peer_scores.json | jq '.peers["<peer_id>"]'
```

**Solutions:**
```bash
# Reset peer score (edit peer_scores.json, set score to 100)
# Review rate limiting thresholds
# Check if peer is actually misbehaving
```

---

## Emergency Procedures

### Node Under Attack (DoS)

**Symptoms:** High CPU, rate limit violations spiking, legitimate requests failing

**Immediate Actions:**
```bash
# 1. Check metrics
curl -s http://localhost:8000/metrics | grep rate_limit

# 2. Identify attacking peers
cat /data/ian/peer_scores.json | jq '.peers | to_entries | sort_by(.value.rate_limit_violations) | reverse | .[0:5]'

# 3. Temporarily increase rate limits or reduce peer count
# Edit config and restart

# 4. Consider blocking at firewall level
iptables -A INPUT -s <attacker_ip> -j DROP
```

### Data Corruption

**Symptoms:** Node crashes with state errors, invalid Merkle roots

**Immediate Actions:**
```bash
# 1. Stop node
docker stop ian-node

# 2. Backup current state (even if corrupted)
mv /data/ian /data/ian-corrupted-$(date +%Y%m%d)

# 3. Restore from backup
tar -xzvf ian-backup-latest.tar.gz -C /

# 4. Start node (will resync missing data)
docker start ian-node

# 5. Monitor sync progress
watch 'curl -s http://localhost:8000/api/v1/status/DEMO_TRADING_AGENT | jq'
```

### Network Partition Recovery

**Symptoms:** Node was isolated, now has different state than network

**Actions:**
```bash
# 1. Check sync status
curl -s http://localhost:8000/api/v1/status/DEMO_TRADING_AGENT

# 2. Node should automatically detect divergence and resync
# Monitor logs for "sync" messages

# 3. If stuck, force resync by clearing local state
docker stop ian-node
rm /data/ian/node_state.json
docker start ian-node
```

### Emergency Shutdown

```bash
# Graceful shutdown (saves state)
docker stop ian-node

# Force shutdown (if graceful fails)
docker kill ian-node

# Verify state was saved
ls -la /data/ian/node_state.json
```

---

## Maintenance

### Regular Tasks

| Task | Frequency | Command |
|------|-----------|---------|
| Check logs for errors | Daily | `docker logs --since 24h ian-node \| grep ERROR` |
| Verify backup | Weekly | `tar -tzvf backup.tar.gz \| head` |
| Review peer scores | Weekly | `cat peer_scores.json \| jq '.peers \| length'` |
| Update node software | Monthly | `docker pull && restart` |
| Rotate TLS certs | Yearly | See cert rotation section |

### Certificate Rotation

```bash
# 1. Generate new certificate (node does this automatically if expired)
# Or manually:
python3 -c "
from idi.ian.network import TLSConfig
config = TLSConfig.generate('$(cat /data/ian/identity.json | jq -r .node_id)', output_dir='/data/ian/certs-new')
"

# 2. Stop node
docker stop ian-node

# 3. Swap certificates
mv /data/ian/certs /data/ian/certs-old
mv /data/ian/certs-new /data/ian/certs

# 4. Start node
docker start ian-node

# 5. Verify peers can still connect
nc -zv localhost 9000
```

### Upgrading Node Software

```bash
# 1. Backup current state
./backup.sh

# 2. Pull new image
docker pull ian-network/ian-node:v2.0.0

# 3. Stop current node
docker stop ian-node

# 4. Start with new image
docker run -d \
  --name ian-node-new \
  -p 8000:8000 \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /data/ian:/data \
  ian-network/ian-node:v2.0.0

# 5. Verify health
curl http://localhost:8000/health

# 6. Remove old container
docker rm ian-node
docker rename ian-node-new ian-node
```

### Cleaning Up Old Data

```bash
# Check disk usage
du -sh /data/ian/*

# Prune old peer scores (peers not seen in 30 days)
python3 -c "
import json, time
with open('/data/ian/peer_scores.json') as f:
    data = json.load(f)

cutoff = time.time() - 30*24*3600
data['peers'] = {k:v for k,v in data['peers'].items() if v['last_seen'] > cutoff}

with open('/data/ian/peer_scores.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Kept {len(data[\"peers\"])} peers')
"
```

---

## Contact & Escalation

| Level | Contact | When |
|-------|---------|------|
| L1 | On-call operator | Alert fires |
| L2 | Senior operator | Can't resolve in 30min |
| L3 | Development team | Data corruption, bugs |

### Useful Links

- Documentation: `/docs/`
- Issue tracker: `github.com/project/issues`
- Monitoring: `grafana.internal/d/ian-nodes`
- Chat: `#ian-operators`
