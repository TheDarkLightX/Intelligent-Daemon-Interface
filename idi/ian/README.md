# Intelligent Augmentation Network (IAN)

A decentralized coordination layer for IDI agent contributions with append-only logging, fair ranking, and Tau Net integration.

## Overview

IAN provides:
- **Append-only Contribution Logs** - Merkle Mountain Range (MMR) for auditable history
- **Fair Top-K Leaderboards** - With tie-breaking and optional Pareto ranking
- **Deterministic Processing** - Same inputs always produce same outputs
- **Security Hardening** - Rate limiting, input validation, timing attack mitigation
- **Bandwidth-Optimal P2P Sync** - FrontierSync + IBLT-based reconciliation for efficient log synchronization
- **Authenticated Reconciliation Data** - Optional IBLT HMAC authentication keyed by per-peer session keys
- **Tau Net Integration** - On-chain state commits and governance upgrades

## Quick Start

```python
from idi.ian import (
    GoalID, GoalSpec, GoalState,
    AgentPack, Contribution,
    IANCoordinator, CoordinatorConfig,
    EvaluationLimits, Thresholds,
)

# 1. Define a goal
goal_spec = GoalSpec(
    goal_id=GoalID("MY_GOAL"),
    name="My Agent Competition",
    description="Optimize for reward while minimizing risk",
    eval_limits=EvaluationLimits(
        max_episodes=100,
        max_steps_per_episode=1000,
        timeout_seconds=60,
        max_memory_mb=512,
    ),
    thresholds=Thresholds(
        min_reward=0.1,
        max_risk=0.9,
        max_complexity=0.9,
    ),
)

# 2. Create coordinator
coordinator = IANCoordinator(
    goal_spec=goal_spec,
    config=CoordinatorConfig(leaderboard_capacity=100),
)

# 3. Submit contributions
contribution = Contribution(
    goal_id=goal_spec.goal_id,
    agent_pack=AgentPack(
        version="1.0",
        parameters=b"...",  # Serialized model
    ),
    proofs={},
    contributor_id="alice",
    seed=12345,
)

result = coordinator.process_contribution(contribution)
print(f"Accepted: {result.accepted}, Score: {result.score}")

# 4. View leaderboard
for entry in coordinator.get_leaderboard()[:10]:
    print(f"Score: {entry.score}, Contributor: {entry.contributor_id}")

# 5. Get active policy
active = coordinator.get_active_policy()
print(f"Active policy: {active.pack_hash.hex()[:16]}...")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IANCoordinator                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Dedup Check    → BloomFilter + DedupIndex                   │
│  2. Validation     → InputValidator                             │
│  3. Invariants     → InvariantChecker                           │
│  4. Proof Verify   → ProofVerifier                              │
│  5. Evaluation     → SandboxedEvaluator                         │
│  6. Threshold      → Thresholds                                 │
│  7. Log Append     → MerkleMountainRange                        │
│  8. Leaderboard    → Leaderboard / ParetoFrontier               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TauBridge                               │
├─────────────────────────────────────────────────────────────────┤
│  • Goal Registration    (IAN_GOAL_REGISTER)                     │
│  • Log Commits          (IAN_LOG_COMMIT)                        │
│  • Policy Upgrades      (IAN_UPGRADE)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `models.py` | Core data models (GoalSpec, Contribution, Metrics, etc.) |
| `mmr.py` | Merkle Mountain Range implementation |
| `leaderboard.py` | Top-K leaderboard with Pareto support |
| `dedup.py` | Bloom filter + hash index de-duplication |
| `coordinator.py` | Main contribution processing pipeline |
| `ranking.py` | Ranking functions (scalar, Pareto) |
| `sandbox.py` | Sandboxed evaluation harness |
| `security.py` | Input validation, rate limiting, PoW |
| `tau_bridge.py` | Tau Net integration |
| `config.py` | Configuration management |
| `observability.py` | Logging, metrics, tracing |
| `network/` | P2P networking layer ([see docs](network/README.md)) |

## New Networking Features (How to Use / Try)

### FrontierSync + IBLT (Bandwidth-Optimal Sync)

Use case:
- Synchronize an append-only log between peers with minimal bandwidth when only a small delta differs.

Value proposition:
- Reduces reconciliation traffic from O(n) to approximately O(Δ) where Δ is the number of differing log entries.

### Authenticated IBLT Exchange (Tamper Resistance)

If the transport exposes a per-peer `session_key` (32 bytes), FrontierSync uses it as the `auth_key` for IBLT HMAC authentication.

To run the focused tests:
```bash
pytest -q idi/ian/tests/test_frontiersync.py::TestFrontierSyncIBLTAuthentication -q
```

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — quick start and API overview |
| [SECURITY.md](SECURITY.md) | Security architecture and hardening guide |

### Architecture & Design

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | **Complete architecture with diagrams** |
| [docs/IAN_L2_ARCHITECTURE.md](docs/IAN_L2_ARCHITECTURE.md) | Layer 2 design on Tau Net |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | **HTTP, P2P, and SDK API reference** |

### Operations

| Document | Description |
|----------|-------------|
| [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md) | **Production operations guide** |
| [deploy/README.md](deploy/README.md) | Docker & Kubernetes deployment |
| [network/README.md](network/README.md) | P2P networking layer docs |

### Tau Integration

| Document | Description |
|----------|-------------|
| [tau_rules/README.md](tau_rules/README.md) | Tau Language rules for IAN |

### Internal/Planning

| Document | Description |
|----------|-------------|
| [IAN_IMPLEMENTATION_PLAN.md](../../Internal/intelligent_augmentation_network/IAN_IMPLEMENTATION_PLAN.md) | Implementation checklist |
| [INTELLIGENT_AUGMENTATION_NETWORK_DESIGN.md](../../Internal/intelligent_augmentation_network/INTELLIGENT_AUGMENTATION_NETWORK_DESIGN.md) | Full design specification |

## Security

### Input Validation

All inputs are validated at the system boundary:

```python
from idi.ian import SecureCoordinator, SecurityLimits

limits = SecurityLimits(
    max_pack_parameters_size=10 * 1024 * 1024,  # 10 MB
    rate_limit_tokens=10,
    rate_limit_refill_per_second=0.1,
)

secure = SecureCoordinator(coordinator, limits=limits)
result = secure.process_contribution(contribution)
```

### Rate Limiting

Token bucket per-contributor rate limiting prevents abuse:

```python
# Check rate limit status
status = secure.get_rate_limit_status("alice")
print(f"Remaining: {status['remaining_tokens']}")
```

### Proof of Work (Optional)

Enable PoW for Sybil resistance:

```python
secure = SecureCoordinator(coordinator, limits=limits, enable_pow=True)

# Get challenge
challenge = secure.get_challenge("alice")

# Solve and submit
from idi.ian import ProofOfWork
pow = ProofOfWork.solve(challenge, difficulty=20)
result = secure.process_contribution(contribution, pow=pow)
```

## Tau Net Integration

### Register Goal

```python
from idi.ian import TauBridge, TauBridgeConfig

bridge = TauBridge(config=TauBridgeConfig(
    commit_interval_seconds=300,
    commit_threshold_contributions=100,
))

bridge.register_goal(goal_spec)
```

### Commit Log

```python
bridge.commit_log(
    goal_id=goal_spec.goal_id,
    log_root=coordinator.get_log_root(),
    log_size=coordinator.state.log.size,
    leaderboard_root=coordinator.get_leaderboard_root(),
    leaderboard_size=len(coordinator.state.leaderboard),
)
```

### Upgrade Policy

```python
bridge.upgrade_policy(
    goal_id=goal_spec.goal_id,
    new_policy=coordinator.get_active_policy(),
    log_root=coordinator.get_log_root(),
    governance_signatures=[],  # Or multi-sig for governance
)
```

## Configuration

### Environment Variables

All settings can be overridden via environment variables:

```bash
export IAN_COORDINATOR_LEADERBOARD_CAPACITY=100
export IAN_SECURITY_RATE_LIMIT_TOKENS=10
export IAN_TAU_ENABLED=true
export IAN_LOGGING_LEVEL=DEBUG
```

### Config File

```python
from idi.ian.config import IANConfig

config = IANConfig.load("config.json")
# or
config = IANConfig.load("config.toml")
```

Example `config.json`:
```json
{
  "coordinator": {
    "leaderboard_capacity": 100,
    "use_pareto": false
  },
  "security": {
    "rate_limit_enabled": true,
    "rate_limit_tokens": 10
  },
  "tau": {
    "enabled": true,
    "endpoint": "http://localhost:10330"
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  }
}
```

## Observability

### Structured Logging

```python
from idi.ian.observability import setup_logging, get_logger

setup_logging(format="json", level="INFO")
logger = get_logger("my_module", goal_id="MY_GOAL")
logger.info("Processing contribution", extra={"context": {"contributor": "alice"}})
```

### Metrics

```python
from idi.ian.observability import metrics

# Record metrics
metrics.contributions_total.inc()
metrics.processing_duration.observe(0.5)

# Export for Prometheus
print(metrics.to_prometheus())
```

## Running the Demo

```bash
# Full demo with security and Tau integration
python -m idi.ian.simulations.trading_agent_demo

# Options
python -m idi.ian.simulations.trading_agent_demo \
    --contributors 10 \
    --contributions 5 \
    --no-security \
    --no-tau \
    --quiet
```

## Testing

```bash
# Run all IAN tests
python -m pytest idi/ian/tests/ -v

# Run specific test file
python -m pytest idi/ian/tests/test_security.py -v

# Run with coverage
python -m pytest idi/ian/tests/ --cov=idi.ian --cov-report=html
```

## Rust Implementation

For maximum performance, a Rust implementation is available:

```bash
cd idi/ian/ian_core
cargo build --release
cargo test
```

Enable Python bindings:
```bash
cargo build --release --features python
```

## Troubleshooting

### Common Issues

#### `UserWarning: cryptography library not available`

The `cryptography` library is required for production use. Install it:

```bash
pip install cryptography
```

Without it, node identity uses an insecure HMAC-based fallback.

#### `Rate limited` errors

The default rate limit is 10 tokens per contributor, refilling at 0.1/second. Adjust in config:

```python
limits = SecurityLimits(
    rate_limit_tokens=100,
    rate_limit_refill_per_second=1.0,
)
```

#### Evaluation timeouts

If evaluations consistently timeout, check:

1. `eval_limits.timeout_seconds` in GoalSpec (default: 60s)
2. `max_memory_mb` — evaluation may be killed for memory
3. Agent code for infinite loops

#### Connection refused / timeout

For P2P networking issues:

1. Check firewall allows the configured port (default: 9000)
2. Verify seed nodes are reachable
3. Check `max_connections` limit hasn't been reached

#### `setrlimit` failures in sandbox

On some systems (Docker, certain Linux configs), `setrlimit` may fail. The sandbox now **aborts** evaluation if limits cannot be enforced. To run without limits (testing only):

```python
evaluator = InProcessEvaluator()  # No isolation
```

### Debug Logging

Enable verbose logging:

```python
from idi.ian.observability import setup_logging

setup_logging(level="DEBUG", format="text")
```

Or via environment:

```bash
export IAN_LOGGING_LEVEL=DEBUG
```

## License

Part of the IDI project. See repository root for license.
