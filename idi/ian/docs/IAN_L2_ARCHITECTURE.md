# IAN Layer 2 Architecture on Tau Net

**Status:** Implemented  
**Version:** 1.0  
**Date:** 2025-12-10

> **Implementation Complete:** All core L2 components are now implemented in the
> `idi/ian/network/` module. See the implementation summary at the end of this document.

---

## 1. Overview: IAN as Layer 2

IAN operates as a **Layer 2 (L2) coordination network** on top of Tau Net (Layer 1). This is analogous to how Optimistic Rollups or State Channels work on Ethereum, but tailored for agent coordination rather than general computation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              LAYER 2: IAN                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    IAN Node Network                              │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │ Node A  │  │ Node B  │  │ Node C  │  │ Node D  │  ...        │    │
│  │  │Coord+API│  │Coord+API│  │Coord+API│  │Coord+API│             │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │    │
│  │       │            │            │            │                   │    │
│  │       └────────────┴─────┬──────┴────────────┘                   │    │
│  │                          │                                       │    │
│  │                    ┌─────┴─────┐                                 │    │
│  │                    │ P2P Gossip │                                │    │
│  │                    │ (Contrib.) │                                │    │
│  │                    └───────────┘                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│                    State Root Commits (periodic)                         │
│                                 │                                        │
│                                 ▼                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                            BRIDGE LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        TauBridge                                 │    │
│  │  - IAN_GOAL_REGISTER: Register new goals                        │    │
│  │  - IAN_LOG_COMMIT: Commit log/leaderboard roots                 │    │
│  │  - IAN_UPGRADE: Update active policy                            │    │
│  │  - IAN_CHALLENGE: Fraud proof submission                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                 │                                        │
│                    Tau Transactions (sendtx)                             │
│                                 │                                        │
│                                 ▼                                        │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: TAU NET                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Tau Consensus Layer                          │    │
│  │  - Finality for state roots                                      │    │
│  │  - Governance for goal specs                                     │    │
│  │  - Economic security (staking, slashing)                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Tau Language Rules                           │    │
│  │  - rules/ian_goals.tau: Goal validation predicates               │    │
│  │  - rules/ian_upgrades.tau: Upgrade authorization logic           │    │
│  │  - rules/ian_governance.tau: Multi-sig / voting rules            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Tau State Streams                            │    │
│  │  - ian_goals[goal_id]: Registered goal metadata                  │    │
│  │  - ian_log_root[goal_id]: Current MMR root                       │    │
│  │  - ian_lb_root[goal_id]: Current leaderboard root                │    │
│  │  - ian_active_policy[goal_id]: Active policy hash                │    │
│  │  - ian_upgrade_count[goal_id]: Upgrade counter                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why Layer 2?

### 2.1 Scalability

| Operation | L1 (Tau) | L2 (IAN) |
|-----------|----------|----------|
| Contribution processing | Would require on-chain computation | Off-chain, parallel |
| Evaluation (RL/backtest) | Impossible on-chain | Sandboxed, distributed |
| Leaderboard updates | O(log K) on-chain per update | O(log K) off-chain, batch commit |
| State storage | All state on-chain | State off-chain, roots on-chain |

**Throughput:** IAN can process thousands of contributions per second off-chain, committing only periodic state roots to Tau.

### 2.2 Cost

Tau transactions have costs (gas/fees). By batching:
- 1000 contributions → 1 `IAN_LOG_COMMIT` transaction
- Cost reduction: ~1000x

### 2.3 Privacy

Agent packs and evaluation details remain off-chain. Only:
- `pack_hash` (commitment)
- `log_root` (Merkle root)
- `leaderboard_root` (state commitment)

are published on-chain.

---

## 3. L2 ↔ L1 Data Flow

### 3.1 State Commitment Model

IAN uses a **commit-chain** model where L2 state is periodically committed to L1:

```
Time →
L2: [C1] [C2] [C3] [C4] [C5] ... [C100]
         ↓                        ↓
L1:   COMMIT_1                 COMMIT_2
      (log_root_1,             (log_root_100,
       lb_root_1)               lb_root_100)
```

**Commit Triggers:**
1. **Time-based:** Every N seconds (default: 300s = 5 min)
2. **Count-based:** Every M contributions (default: 100)
3. **Upgrade:** When active policy changes

### 3.2 Transaction Types

#### `IAN_GOAL_REGISTER`
```json
{
  "type": "IAN_GOAL_REGISTER",
  "goal_id": "VC_AGENT_V1",
  "goal_spec_hash": "0x...",
  "name": "Ownerless VC Agent",
  "description": "...",
  "invariant_ids": ["I1", "I2", "I3", "I4", "I5"],
  "thresholds": {
    "min_reward": 0.1,
    "max_risk": 0.9,
    "max_complexity": 0.9
  },
  "governance_config": {
    "type": "multisig",
    "signers": ["0x...", "0x...", "0x..."],
    "threshold": 2
  },
  "timestamp_ms": 1702252800000,
  "signature": "0x..."
}
```

**L1 Effect:** Creates `ian_goals[goal_id]` stream with metadata.

#### `IAN_LOG_COMMIT`
```json
{
  "type": "IAN_LOG_COMMIT",
  "goal_id": "VC_AGENT_V1",
  "log_root": "0x...",
  "log_size": 1000,
  "leaderboard_root": "0x...",
  "leaderboard_size": 50,
  "prev_commit_hash": "0x...",
  "timestamp_ms": 1702253100000,
  "committer_signature": "0x..."
}
```

**L1 Effect:** Updates `ian_log_root[goal_id]` and `ian_lb_root[goal_id]`.

#### `IAN_UPGRADE`
```json
{
  "type": "IAN_UPGRADE",
  "goal_id": "VC_AGENT_V1",
  "new_pack_hash": "0x...",
  "new_score": 0.85,
  "log_root": "0x...",
  "log_index": 999,
  "prev_pack_hash": "0x...",
  "governance_signatures": ["0x...", "0x..."],
  "timestamp_ms": 1702253200000
}
```

**L1 Effect:** Updates `ian_active_policy[goal_id]` and increments `ian_upgrade_count[goal_id]`.

#### `IAN_CHALLENGE` (Fraud Proof)
```json
{
  "type": "IAN_CHALLENGE",
  "goal_id": "VC_AGENT_V1",
  "challenged_commit_hash": "0x...",
  "fraud_proof_type": "INVALID_LOG_ROOT",
  "fraud_proof_data": {
    "claimed_root": "0x...",
    "actual_leaves": [...],
    "merkle_proof": [...]
  },
  "challenger_signature": "0x..."
}
```

**L1 Effect:** If valid, slashes committer bond and reverts state.

---

## 4. Decentralized IAN Node Network

### 4.1 Node Roles

| Role | Responsibility | Incentive |
|------|----------------|-----------|
| **Contributor** | Submit agent contributions | Policy adoption |
| **Evaluator** | Run sandboxed evaluations | Evaluation fees |
| **Coordinator** | Process pipeline, maintain state | Commit rewards |
| **Watcher** | Monitor for fraud, submit challenges | Slashing rewards |

### 4.2 Multi-Node Consensus

Since IAN processing is **deterministic**, multiple nodes processing the same contributions in the same order will arrive at the same state. This enables:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Contribution Ordering                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Option A: Tau-Ordered (Strongest)                          ││
│  │  - Contributions submitted as L1 transactions               ││
│  │  - Tau consensus orders them                                ││
│  │  - All L2 nodes replay in same order                        ││
│  │  - Pro: Canonical ordering guaranteed                       ││
│  │  - Con: Higher latency, L1 costs                            ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Option B: Sequencer-Based (Faster, Less Decentralized)     ││
│  │  - Designated sequencer orders contributions                ││
│  │  - Other nodes follow sequencer order                       ││
│  │  - Sequencer posts ordering proof to L1                     ││
│  │  - Pro: Low latency                                         ││
│  │  - Con: Sequencer is semi-trusted                           ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Option C: Timestamp + Hash Ordering (Pragmatic)            ││
│  │  - Order by (timestamp_ms, pack_hash)                       ││
│  │  - Nodes gossip contributions                               ││
│  │  - Small reorg window until L1 commit                       ││
│  │  - Pro: Fully decentralized, no sequencer                   ││
│  │  - Con: Brief uncertainty before commit                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Recommended:** Option C for most use cases, with Option A available for high-value goals.

### 4.3 State Synchronization Protocol

```
┌─────────────┐                           ┌─────────────┐
│   Node A    │                           │   Node B    │
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       │  ──── ContributionAnnounce ────────►   │
       │       {pack_hash, contrib_data}         │
       │                                         │
       │  ◄──── ContributionAck ────────────    │
       │       {pack_hash, accepted}             │
       │                                         │
       │  ──── StateRequest ────────────────►   │
       │       {goal_id}                         │
       │                                         │
       │  ◄──── StateResponse ──────────────    │
       │       {log_root, lb_root, log_size}     │
       │                                         │
       │  [If roots differ]                      │
       │  ──── LogSyncRequest ──────────────►   │
       │       {goal_id, from_index}             │
       │                                         │
       │  ◄──── LogSyncResponse ────────────    │
       │       {contributions[from:to]}          │
       │                                         │
```

**Consistency Rule:** Nodes periodically compare state roots. If diverged beyond threshold, they sync from the majority or from L1 committed state.

---

## 5. Security Model

### 5.1 Trust Assumptions

| Component | Trust Level | Justification |
|-----------|-------------|---------------|
| Tau Net (L1) | Full | Byzantine fault tolerant consensus |
| IAN Nodes | Untrusted | Any node can be malicious |
| Committers | Economic | Must post bond, can be slashed |
| Evaluators | Untrusted | Results verified by quorum |
| Contributors | Untrusted | Inputs validated, rate limited |

### 5.2 Fraud Proof System

IAN uses **optimistic execution** with fraud proofs:

1. **Commit:** Coordinator commits state root to L1
2. **Challenge Period:** Watchers have T seconds to challenge
3. **Challenge:** If fraud detected, submit proof to L1
4. **Resolution:** L1 Tau rules verify fraud proof
5. **Slash:** If valid, committer bond slashed; state reverted

**Fraud Types:**

| Fraud | Detection | Proof |
|-------|-----------|-------|
| Invalid log root | Recompute MMR | Leaf hashes + Merkle proof |
| Invalid leaderboard | Recompute ranking | Contribution metas + scores |
| Skipped contribution | Missing in log | Contribution + gossip proof |
| Wrong evaluation | Re-evaluate | Agent pack + seed + expected metrics |

### 5.3 Economic Security

```
┌─────────────────────────────────────────────────────────────────┐
│                      Bonding & Slashing                          │
├─────────────────────────────────────────────────────────────────┤
│  Committer Bond:                                                 │
│    - Required: 1000 TAU per goal                                 │
│    - Slash: 50% for invalid commit                               │
│    - Slash: 100% for repeated fraud                              │
│                                                                  │
│  Challenger Reward:                                              │
│    - 25% of slashed amount goes to challenger                    │
│    - 25% burned                                                  │
│                                                                  │
│  Challenge Bond:                                                 │
│    - Required: 100 TAU to submit challenge                       │
│    - Returned if challenge valid                                 │
│    - Slashed if challenge invalid (griefing prevention)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Tau Language Integration

### 6.1 Tau State Streams

```tau
// rules/ian_state.tau

// Goal registry
ian_goal_exists(GoalID) :- ian_goals(GoalID, _, _, _).
ian_goals(GoalID, SpecHash, Name, GovernanceConfig).

// Per-goal state
ian_log_root(GoalID, Root, Size, CommitHeight).
ian_lb_root(GoalID, Root, Size, CommitHeight).
ian_active_policy(GoalID, PackHash, Score, LogIndex, UpgradeHeight).
ian_upgrade_count(GoalID, Count).

// Committer registry
ian_committer_bond(CommitterPubkey, GoalID, BondAmount).
ian_committer_slashed(CommitterPubkey, GoalID, SlashHeight).
```

### 6.2 Tau Rules for IAN Operations

```tau
// rules/ian_upgrades.tau

// Upgrade is valid if:
// 1. Goal exists
// 2. New score > current score
// 3. Log index exists in committed log
// 4. Governance approved (if required)
// 5. Cooldown elapsed

valid_upgrade(GoalID, NewPackHash, NewScore, LogIndex) :-
    ian_goal_exists(GoalID),
    current_score(GoalID, CurrentScore),
    NewScore > CurrentScore,
    log_index_valid(GoalID, LogIndex),
    governance_approved(GoalID, NewPackHash),
    cooldown_elapsed(GoalID).

current_score(GoalID, Score) :-
    ian_active_policy(GoalID, _, Score, _, _).
current_score(GoalID, 0) :-
    ~ian_active_policy(GoalID, _, _, _, _).

cooldown_elapsed(GoalID) :-
    ian_active_policy(GoalID, _, _, _, LastUpgrade),
    current_height(Height),
    Height - LastUpgrade >= 1440.  // ~24 hours in blocks

// Governance rules
governance_approved(GoalID, PackHash) :-
    ian_goals(GoalID, _, _, GovConfig),
    GovConfig.type = "none".

governance_approved(GoalID, PackHash) :-
    ian_goals(GoalID, _, _, GovConfig),
    GovConfig.type = "multisig",
    signature_count(GoalID, PackHash, Count),
    Count >= GovConfig.threshold.
```

### 6.3 Tau Rules for Fraud Proofs

```tau
// rules/ian_fraud.tau

// Challenge is valid if fraud proof verifies
valid_challenge(CommitHash, FraudProof) :-
    ian_commit(CommitHash, GoalID, ClaimedRoot, _),
    fraud_proof_verifies(FraudProof, ClaimedRoot).

// Slash committer and revert state
process_challenge(CommitHash, ChallengerPubkey) :-
    valid_challenge(CommitHash, _),
    ian_commit(CommitHash, GoalID, _, CommitterPubkey),
    slash_bond(CommitterPubkey, GoalID, ChallengerPubkey),
    revert_to_previous_commit(GoalID, CommitHash).
```

---

## 7. Implementation Roadmap

### Phase 1: Single-Node L2 (Current)
- [x] IANCoordinator with deterministic processing
- [x] TauBridge with transaction types
- [x] MMR for append-only log
- [x] Leaderboard with bounded capacity
- [ ] Basic L1 commit (mock sender → real Tau)

### Phase 2: Multi-Node L2
- [ ] P2P contribution gossip (partially done)
- [ ] State synchronization protocol
- [ ] Contribution ordering (Option C)
- [ ] Node discovery via DHT

### Phase 3: Economic Security
- [ ] Committer bonding on Tau
- [ ] Challenge/response protocol
- [ ] Fraud proof verification
- [ ] Slashing mechanism

### Phase 4: Decentralized Evaluation
- [ ] Multi-evaluator quorum
- [ ] Evaluation result consensus
- [ ] Optional ZK evaluation proofs

### Phase 5: Full Decentralization
- [ ] Governance on Tau
- [ ] Permissionless committers
- [ ] Economic incentive model

---

## 8. Data Availability

### 8.1 Where Data Lives

| Data | Location | Availability |
|------|----------|--------------|
| Agent packs (full) | IAN nodes, IPFS | Off-chain, hash-addressed |
| Contribution metadata | IAN nodes | Off-chain, replicated |
| MMR leaves | IAN nodes | Off-chain, Merkle provable |
| MMR root | Tau L1 | On-chain, finalized |
| Leaderboard entries | IAN nodes | Off-chain, Merkle provable |
| Leaderboard root | Tau L1 | On-chain, finalized |
| Active policy hash | Tau L1 | On-chain, finalized |

### 8.2 Data Retrieval

```python
# Retrieving contribution with proof
async def get_contribution_with_proof(goal_id: str, log_index: int):
    # 1. Get committed root from L1
    l1_root = await tau_client.get_ian_log_root(goal_id)
    
    # 2. Get contribution from any L2 node
    contrib, proof = await ian_node.get_contribution_with_proof(goal_id, log_index)
    
    # 3. Verify locally
    assert verify_mmr_proof(contrib.pack_hash, proof, l1_root)
    
    return contrib
```

---

## 9. Comparison to Other L2 Designs

| Feature | IAN on Tau | Optimistic Rollup | ZK Rollup | State Channel |
|---------|------------|-------------------|-----------|---------------|
| Finality | Commit + challenge | Commit + challenge | Instant (proof) | Instant (channel) |
| Data availability | Off-chain + proof | On-chain calldata | On-chain calldata | Off-chain |
| Computation | Off-chain eval | Off-chain EVM | Off-chain ZK circuit | Off-chain |
| Fraud proofs | Yes | Yes | No (validity proofs) | Yes |
| Multi-party | Yes (nodes) | Yes (sequencer) | Yes (prover) | 2-party |
| Use case | Agent coordination | General | General | Payments/games |

---

## 10. Summary

IAN is a **Layer 2 coordination network** that:

1. **Executes off-chain:** Contribution processing, evaluation, leaderboard maintenance
2. **Commits to L1:** Periodic state roots to Tau Net for finality
3. **Secures via fraud proofs:** Invalid commits can be challenged and slashed
4. **Scales horizontally:** Multiple nodes process in parallel, gossip contributions
5. **Inherits L1 security:** Final state is only what Tau consensus accepts

This design enables **high-throughput agent coordination** while maintaining the security guarantees of Tau Net's decentralized consensus.

---

## 11. Implementation Summary

All core L2 components are implemented in `idi/ian/network/`:

| Module | File | Description |
|--------|------|-------------|
| **Ordering** | `ordering.py` | Deterministic contribution ordering, mempool, ordering proofs |
| **Consensus** | `consensus.py` | Multi-node consensus coordinator, state synchronization |
| **Fraud Proofs** | `fraud.py` | Fraud proof types, generation, verification, challenge manager |
| **Economics** | `economics.py` | Committer bonding, slashing, challenger rewards |
| **Evaluation** | `evaluation.py` | Distributed evaluation quorum, evaluator registry |
| **Node** | `decentralized_node.py` | Unified decentralized node bringing all components together |

### Quick Start

```python
from idi.ian.network import (
    DecentralizedNode,
    DecentralizedNodeConfig,
    create_decentralized_node,
)
from idi.ian.models import GoalSpec, GoalID

# Create goal specification
goal_spec = GoalSpec(
    goal_id=GoalID("MY_AGENT_001"),
    name="My Decentralized Agent",
    # ... other config
)

# Create and run decentralized node
node = create_decentralized_node(
    goal_spec=goal_spec,
    seed_addresses=["tcp://seed1.ian.network:9000"],
)

# Start the node
await node.start()

# Submit contributions
contribution = Contribution(...)
success, reason = await node.submit_contribution(contribution)

# Node automatically:
# - Processes contributions in deterministic order
# - Maintains consensus with peers
# - Commits state to Tau Net periodically
# - Detects and proves fraud
```

### Running Tests

```bash
pytest idi/ian/tests/test_decentralized.py -v
```
