# IAN Tau Language Rules

This directory contains Tau Language specifications for validating IAN operations on Tau Net.

## Overview

IAN operates as a Layer 2 on Tau Net. These rules define the on-chain validation logic for:

1. **Goal Registration** (`IAN_GOAL_REGISTER`) - Register new coordination goals
2. **Log Commits** (`IAN_LOG_COMMIT`) - Commit state roots to L1
3. **Upgrades** (`IAN_UPGRADE`) - Update active policies
4. **Challenges** (`IAN_CHALLENGE`) - Submit fraud proofs

## State Streams

IAN uses the following Tau state streams:

| Stream | Description |
|--------|-------------|
| `ian_goals[goal_id]` | Registered goal metadata |
| `ian_log_root[goal_id]` | Current MMR log root |
| `ian_lb_root[goal_id]` | Current leaderboard root |
| `ian_active_policy[goal_id]` | Active policy hash |
| `ian_upgrade_count[goal_id]` | Upgrade counter |
| `ian_committer_bond[committer]` | Committer bond amounts |
| `ian_last_commit[goal_id]` | Timestamp of last commit |

## Files

| File | Description |
|------|-------------|
| `ian_state.tau` | State stream definitions |
| `ian_goal_register.tau` | Goal registration rules |
| `ian_log_commit.tau` | Log commit validation |
| `ian_upgrade.tau` | Policy upgrade rules |
| `ian_challenge.tau` | Fraud proof validation |
| `ian_governance.tau` | Multi-sig / voting rules |
| `ian_economics.tau` | Bonding / slashing rules |

## Integration

To integrate with Tau Testnet:

1. Copy rules to `tau-testnet/rules/` directory
2. Reference in genesis.tau or load via rule proposals
3. IAN transactions will be validated against these rules

## Example Transaction Flow

```
1. Node submits IAN_LOG_COMMIT transaction
2. Tau validates against ian_log_commit.tau rules:
   - Check committer has bond
   - Check prev_commit_hash matches
   - Check timestamp is valid
3. If valid, update ian_log_root and ian_lb_root streams
4. Challenge period begins
5. If challenged, ian_challenge.tau validates fraud proof
6. If valid fraud proof, slash committer via ian_economics.tau
```
