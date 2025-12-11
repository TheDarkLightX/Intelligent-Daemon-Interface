# IAN Documentation Index

Comprehensive documentation for the IAN (Intelligent Agent Network) decentralized system.

## Quick Links

| Need To... | Read This |
|------------|-----------|
| Understand the architecture | [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) |
| Integrate with the API | [API_REFERENCE.md](API_REFERENCE.md) |
| Deploy to production | [OPERATOR_RUNBOOK.md](OPERATOR_RUNBOOK.md) |
| Understand L2 design | [IAN_L2_ARCHITECTURE.md](IAN_L2_ARCHITECTURE.md) |

---

## Documentation Overview

### 📐 Architecture Guide
**File:** [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)

Complete system architecture with ASCII diagrams covering:
- System overview and node types
- Component architecture (production, consensus, processing, network, storage layers)
- Data flow (contribution processing, Tau commits)
- Network topology (peer discovery, gossip protocol)
- State management (Merkle trees, persistence)
- Security architecture (authentication, defense layers, threat model)
- Resilience patterns (circuit breaker, task supervision)
- Module reference

---

### 🔌 API Reference
**File:** [API_REFERENCE.md](API_REFERENCE.md)

Complete API documentation covering:
- HTTP API (health, metrics, contributions, leaderboard, proofs)
- P2P Protocol (message types, format, handshake)
- Python SDK (usage examples, production utilities)
- Configuration (environment variables, config files)
- Error codes (contribution rejection, P2P errors, HTTP status)
- WebSocket API (subscriptions, event types)

---

### 🔧 Operator Runbook
**File:** [OPERATOR_RUNBOOK.md](OPERATOR_RUNBOOK.md)

Production operations guide covering:
- Quick reference (endpoints, files, metrics)
- Deployment (Docker, Compose, Kubernetes)
- Monitoring (Prometheus, dashboards, alerts)
- Common operations (restart, backup, restore)
- Troubleshooting (common issues, solutions)
- Emergency procedures (DoS, data corruption, partitions)
- Maintenance (regular tasks, cert rotation, upgrades)

---

### 🏗️ L2 Architecture
**File:** [IAN_L2_ARCHITECTURE.md](IAN_L2_ARCHITECTURE.md)

Layer 2 design on Tau Net covering:
- Design rationale
- State model
- Consensus mechanism
- Tau Net integration (transactions, rules)
- Economic security (bonding, slashing)
- Fraud proofs
- Implementation phases

---

## Related Documentation

| Location | Content |
|----------|---------|
| [../README.md](../README.md) | IAN module overview and quick start |
| [../SECURITY.md](../SECURITY.md) | Security architecture and hardening |
| [../network/README.md](../network/README.md) | P2P networking layer |
| [../deploy/README.md](../deploy/README.md) | Deployment configurations |
| [../tau_rules/README.md](../tau_rules/README.md) | Tau Language rules |

---

## Document Versions

| Document | Last Updated | Version |
|----------|--------------|---------|
| ARCHITECTURE_GUIDE.md | 2024-12-11 | 1.0 |
| API_REFERENCE.md | 2024-12-11 | 1.0 |
| OPERATOR_RUNBOOK.md | 2024-12-11 | 1.0 |
| IAN_L2_ARCHITECTURE.md | 2024-12-11 | 1.0 |
