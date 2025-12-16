# IAN Architecture Guide

Comprehensive architecture documentation for the IAN (Intelligent Agent Network) decentralized system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Network Topology](#network-topology)
5. [State Management](#state-management)
6. [Security Architecture](#security-architecture)
7. [Resilience Patterns](#resilience-patterns)
8. [Module Reference](#module-reference)

---

## System Overview

IAN is a decentralized Layer 2 system built on Tau Net that enables:
- **Contribution Processing**: Accept, validate, and order contributions
- **Leaderboard Management**: Maintain ranked contributions by score
- **Fraud Proofs**: Detect and penalize malicious behavior
- **Tau Net Anchoring**: Periodic state commits to L1

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              IAN Network                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Node A    │◄──►│   Node B    │◄──►│   Node C    │                │
│   │  (Seed)     │    │  (Full)     │    │  (Full)     │                │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │
│          │                  │                  │                         │
│          └──────────────────┼──────────────────┘                         │
│                             │                                            │
│                    ┌────────▼────────┐                                  │
│                    │  Tau Net (L1)   │                                  │
│                    │  State Anchor   │                                  │
│                    └─────────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Node Types

| Type | Role | Count |
|------|------|-------|
| **Seed Node** | Bootstrap, always online | 3-5 |
| **Full Node** | Process contributions, serve data | Many |
| **Evaluator Node** | Run evaluations, provide scores | Configurable |
| **Light Client** | Query only, no storage | Many |

---

## Component Architecture

### Node Internal Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DecentralizedNode                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Production Layer                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ TaskSupervsr │  │ HealthServer │  │   Metrics    │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ PeerScoring  │  │ CircuitBrkr  │  │  Shutdown    │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Consensus Layer                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  Consensus   │  │   Mempool    │  │    Gossip    │          │   │
│  │  │ Coordinator  │  │              │  │   Protocol   │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Processing Layer                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │     IAN      │  │  Fraud Proof │  │  Evaluation  │          │   │
│  │  │ Coordinator  │  │  Generator   │  │   Quorum     │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Network Layer                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  P2P Manager │  │  Discovery   │  │    Sync      │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ TCP/TLS Xport│  │  WebSocket   │  │  HTTP API    │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Storage Layer                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  Log Store   │  │ Leaderboard  │  │   State      │          │   │
│  │  │ (Merkle Tree)│  │    Store     │  │ Persistence  │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **TaskSupervisor** | Manage background tasks, auto-restart on failure |
| **HealthServer** | HTTP endpoints for health, ready, metrics |
| **PeerScoring** | Track peer reputation, ban bad actors |
| **CircuitBreaker** | Prevent cascading failures to Tau Net |
| **ConsensusCoordinator** | Order contributions deterministically |
| **Mempool** | Buffer pending contributions |
| **GossipProtocol** | Propagate contributions to peers |
| **IANCoordinator** | Process contributions, update state |
| **FraudProofGenerator** | Create proofs of invalid state |
| **EvaluationQuorum** | Coordinate distributed evaluation |
| **P2PManager** | Manage peer connections |
| **Discovery** | Find and connect to peers |
| **Sync** | Synchronize state with peers |

---

## Data Flow

### Contribution Processing Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────►│ HTTP API │────►│ Mempool  │────►│ Consensus│
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                        │
                                                        ▼
                                                  ┌──────────┐
                                                  │  Gossip  │
                                                  │  to Peers│
                                                  └────┬─────┘
                                                        │
                 ┌──────────────────────────────────────┘
                 │
                 ▼
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Validate │────►│ Evaluate │────►│ Log      │────►│Leaderbd  │
│Invariants│     │ (Sandbox)│     │ Append   │     │ Update   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                        │
                                        ▼
                                  ┌──────────┐
                                  │ Merkle   │
                                  │ Root Upd │
                                  └──────────┘
```

### Contribution Processing Steps

1. **Receive**: Client submits contribution via HTTP or P2P
2. **Deduplicate**: Check if already in log (by pack_hash)
3. **Validate**: Check goal invariants
4. **Verify Proof**: Validate ZK proof if required
5. **Evaluate**: Run in sandbox, compute metrics
6. **Order**: Consensus determines log position
7. **Append**: Add to log Merkle tree
8. **Update Leaderboard**: Insert if score qualifies
9. **Gossip**: Propagate to peers

### Tau Net Commit Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         IAN Node                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Commit Loop (5 min)                       │ │
│  │                                                              │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │ │
│  │   │ Gather  │───►│ Build   │───►│  Sign   │───►│  Send   │ │ │
│  │   │  State  │    │   Tx    │    │   Tx    │    │ to Tau  │ │ │
│  │   └─────────┘    └─────────┘    └─────────┘    └────┬────┘ │ │
│  │                                                      │      │ │
│  │        State:                                        │      │ │
│  │        - log_root                                    │      │ │
│  │        - log_size                                    │      │ │
│  │        - leaderboard_root                           │      │ │
│  │        - prev_commit_hash                           ▼      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                          Tau Net (L1)                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   IAN_LOG_COMMIT Transaction:                                     │
│   {                                                                │
│     goal_id: "my-goal",                                           │
│     log_root: "0xabc...",                                         │
│     log_size: 1520,                                               │
│     committer_id: "node123",                                      │
│     prev_hash: "0xdef...",                                        │
│     signature: "0x..."                                            │
│   }                                                                │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Network Topology

### Peer Discovery

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Discovery Process                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Bootstrap from Seeds                                            │
│                                                                      │
│     ┌──────────┐                                                    │
│     │ New Node │                                                    │
│     └────┬─────┘                                                    │
│          │                                                          │
│          ▼                                                          │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│     │  Seed 1  │    │  Seed 2  │    │  Seed 3  │                   │
│     └────┬─────┘    └────┬─────┘    └────┬─────┘                   │
│          │               │               │                          │
│          └───────────────┼───────────────┘                          │
│                          │                                          │
│                          ▼                                          │
│                    Connect to seeds                                 │
│                                                                      │
│  2. Peer Exchange                                                   │
│                                                                      │
│     ┌──────────┐         ┌──────────┐                              │
│     │ New Node │◄───────►│  Seed    │                              │
│     └────┬─────┘         └──────────┘                              │
│          │                    │                                     │
│          │    PEER_EXCHANGE   │                                     │
│          │◄───────────────────│                                     │
│          │   {peers: [...]}   │                                     │
│          │                                                          │
│          ▼                                                          │
│     Connect to discovered peers                                     │
│                                                                      │
│  3. Continuous Discovery                                            │
│                                                                      │
│     - Periodic peer exchange with connected peers                   │
│     - Health checks (ping/pong)                                     │
│     - Evict unhealthy peers                                         │
│     - Maintain target peer count                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Message Propagation (Gossip)

```
┌───────────────────────────────────────────────────────────────────┐
│                      Gossip Protocol                               │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Time T+0: Node A receives contribution                          │
│                                                                    │
│         ┌───┐                                                     │
│         │ A │  ◄── New contribution                               │
│         └─┬─┘                                                     │
│           │                                                        │
│   Time T+1: A forwards to neighbors                               │
│           │                                                        │
│           ├────────┬────────┐                                     │
│           ▼        ▼        ▼                                     │
│         ┌───┐    ┌───┐    ┌───┐                                   │
│         │ B │    │ C │    │ D │                                   │
│         └─┬─┘    └─┬─┘    └─┬─┘                                   │
│           │        │        │                                      │
│   Time T+2: B, C, D forward to their neighbors                    │
│           │        │        │                                      │
│           ▼        ▼        ▼                                     │
│         ┌───┐    ┌───┐    ┌───┐                                   │
│         │ E │    │ F │    │ G │                                   │
│         └───┘    └───┘    └───┘                                   │
│                                                                    │
│   Propagation: O(log N) hops for N nodes                          │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## State Management

### Merkle Tree Structure

```
                              Root Hash
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 Hash(0-3)                   Hash(4-7)
                    │                           │
            ┌───────┴───────┐           ┌───────┴───────┐
            │               │           │               │
         Hash(0-1)       Hash(2-3)   Hash(4-5)       Hash(6-7)
            │               │           │               │
         ┌──┴──┐         ┌──┴──┐     ┌──┴──┐         ┌──┴──┐
         │     │         │     │     │     │         │     │
       [C0]  [C1]      [C2]  [C3]  [C4]  [C5]      [C6]  [C7]
        │     │         │     │     │     │         │     │
      Contributions (pack_hashes)
```

### State Persistence

```
┌─────────────────────────────────────────────────────────────────┐
│                     Persisted State Files                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  /data/ian/                                                     │
│  ├── node_state.json      # Core node state                     │
│  │   {                                                          │
│  │     "node_id": "abc123...",                                  │
│  │     "goal_id": "my-goal",                                    │
│  │     "last_log_root": "def456...",                            │
│  │     "log_size": 1520,                                        │
│  │     "last_tau_commit": 1702300000.0,                         │
│  │     "contributions_since_commit": 45                         │
│  │   }                                                          │
│  │                                                              │
│  ├── peer_scores.json     # Peer reputation                     │
│  │   {                                                          │
│  │     "version": 1,                                            │
│  │     "saved_at": 1702300000.0,                                │
│  │     "peers": {                                               │
│  │       "peer123": {                                           │
│  │         "score": 150.0,                                      │
│  │         "valid_messages": 1000,                              │
│  │         "rate_limit_violations": 2,                          │
│  │         ...                                                  │
│  │       }                                                      │
│  │     }                                                        │
│  │   }                                                          │
│  │                                                              │
│  ├── identity.json        # Node identity (Ed25519 keys)        │
│  │                                                              │
│  └── certs/               # TLS certificates                    │
│      ├── node_cert.pem                                          │
│      └── node_key.pem                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    P2P Handshake (Challenge-Response)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client                              Server                     │
│     │                                   │                        │
│     │──── HANDSHAKE_CHALLENGE ─────────►│                        │
│     │     {sender_id, public_key,       │                        │
│     │      kx_public_key,               │                        │
│     │      challenge_nonce, signature}  │                        │
│     │                                   │                        │
│     │                      Verify signature + node_id binding    │
│     │                      Derive per-peer session key           │
│     │                                   │                        │
│     │◄─── HANDSHAKE_RESPONSE ──────────│                        │
│     │     {sender_id, public_key,       │                        │
│     │      kx_public_key,               │                        │
│     │      challenge_nonce,             │                        │
│     │      response_nonce, signature}   │                        │
│     │                                   │                        │
│     │     Verify signature + node_id binding                     │
│     │     Derive per-peer session key                            │
│     │                                   │                        │
│     │     Connection ready (verified peer)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Defense Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        Security Layers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Network                                               │
│  ├── TLS encryption (mutual TLS optional)                       │
│  ├── IP-based rate limiting                                     │
│  └── Connection limits per IP                                   │
│                                                                  │
│  Layer 2: Authentication                                        │
│  ├── Challenge-response handshake                               │
│  ├── Ed25519 signature verification                             │
│  └── Node ID validation                                         │
│                                                                  │
│  Layer 3: Protocol                                              │
│  ├── Message size limits (16MB max)                             │
│  ├── Per-peer rate limiting (100 msg/s)                         │
│  ├── Message signature verification                             │
│  └── Bounded reads (prevent memory exhaustion)                  │
│                                                                  │
│  Layer 4: Application                                           │
│  ├── Contribution hash verification                             │
│  ├── Invariant checking                                         │
│  ├── ZK proof verification                                      │
│  └── Merkle proof validation                                    │
│                                                                  │
│  Layer 5: Economic                                              │
│  ├── Committer bonding                                          │
│  ├── Slashing for fraud                                         │
│  └── Challenge system                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Threat Model

| Threat | Mitigation |
|--------|------------|
| **DoS (message flood)** | Rate limiting, peer banning |
| **Sybil attack** | Peer scoring, connection limits |
| **Eclipse attack** | Multiple seed nodes, peer diversity |
| **State corruption** | Merkle proofs, fraud proofs |
| **Front-running** | Deterministic ordering |
| **Man-in-middle** | TLS, signed messages |

---

## Resilience Patterns

### Circuit Breaker

```
┌─────────────────────────────────────────────────────────────────┐
│                     Circuit Breaker States                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────┐                              │
│                    │   CLOSED    │◄─────────────────────┐       │
│                    │ (Normal)    │                      │       │
│                    └──────┬──────┘                      │       │
│                           │                             │       │
│                    5 consecutive                  3 successes   │
│                      failures                    in half-open   │
│                           │                             │       │
│                           ▼                             │       │
│                    ┌─────────────┐              ┌──────┴──────┐ │
│                    │    OPEN     │──timeout────►│  HALF_OPEN  │ │
│                    │ (Rejecting) │   (30s)      │  (Testing)  │ │
│                    └─────────────┘              └──────┬──────┘ │
│                           ▲                            │        │
│                           │                            │        │
│                           └────── 1 failure ───────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Task Supervision

```
┌─────────────────────────────────────────────────────────────────┐
│                     Task Supervisor Tree                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌────────────────┐                           │
│                    │  TaskSupervisor │                           │
│                    │    (root)       │                           │
│                    └────────┬───────┘                           │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ tau_commit  │   │peer_state   │   │ challenge   │           │
│  │   loop      │   │check_loop   │   │final_loop   │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│    On crash:         On crash:         On crash:               │
│    restart w/        restart w/        restart w/              │
│    backoff           backoff           backoff                 │
│                                                                  │
│  Features:                                                      │
│  - Automatic restart on failure                                 │
│  - Exponential backoff with jitter                              │
│  - Max restart limit                                            │
│  - Exception logging                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| `coordinator` | `coordinator.py` | Core contribution processing |
| `models` | `models.py` | Data structures (Contribution, GoalSpec) |
| `merkle` | `merkle.py` | Merkle tree implementation |
| `leaderboard` | `leaderboard.py` | Ranked contribution storage |

### Network Modules

| Module | File | Purpose |
|--------|------|---------|
| `decentralized_node` | `network/decentralized_node.py` | Main node class |
| `consensus` | `network/consensus.py` | Ordering consensus |
| `p2p_manager` | `network/p2p_manager.py` | Peer connections |
| `protocol` | `network/protocol.py` | Message types |
| `discovery` | `network/discovery.py` | Peer discovery |
| `sync` | `network/sync.py` | State synchronization |
| `transport` | `network/transport.py` | TCP transport |
| `tls` | `network/tls.py` | TLS support |

### Production Modules

| Module | File | Purpose |
|--------|------|---------|
| `production` | `network/production.py` | Supervisor, metrics, health |
| `resilience` | `network/resilience.py` | Circuit breaker, logging |

### Economics Modules

| Module | File | Purpose |
|--------|------|---------|
| `economics` | `network/economics.py` | Bonding, slashing |
| `fraud` | `network/fraud.py` | Fraud proof generation |
| `evaluation` | `network/evaluation.py` | Evaluation quorum |

### Tau Integration

| Module | File | Purpose |
|--------|------|---------|
| `tau_sender` | `tau_sender.py` | Send transactions to Tau |
| `tau_bridge` | `tau_bridge.py` | Tau integration bridge |
| `tau_rules/` | `tau_rules/*.tau` | Tau Language rules |

---

## Glossary

| Term | Definition |
|------|------------|
| **Contribution** | A submission to a goal (code, data, etc.) |
| **Pack Hash** | SHA256 hash of contribution content |
| **Log** | Append-only Merkle tree of contributions |
| **Leaderboard** | Ranked list of top contributions by score |
| **Tau Net** | Layer 1 blockchain for state anchoring |
| **Fraud Proof** | Cryptographic proof of invalid state |
| **Committer** | Node authorized to commit state to Tau |
| **Bond** | Stake required to become a committer |
| **Slash** | Penalty for fraudulent behavior |
| **Gossip** | P2P message propagation protocol |
