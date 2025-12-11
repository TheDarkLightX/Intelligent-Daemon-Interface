# IAN Security Guide

This document describes the security architecture, threat model, and hardening measures for the Intelligent Augmentation Network (IAN).

## Threat Model

IAN operates in an **adversarial environment** where:

| Threat Actor | Capability | Goal |
|--------------|------------|------|
| **Malicious Contributor** | Submits crafted contributions | Game leaderboard, DoS, exploit evaluation |
| **Network Attacker** | Controls network traffic | DoS, man-in-middle, resource exhaustion |
| **Sybil Attacker** | Creates many identities | Flood rate limits, game rankings |
| **Timing Attacker** | Measures response times | Extract secret information |

## Security Architecture

```
                    ┌─────────────────────────────────────┐
                    │         SecureCoordinator           │
                    │  (Wraps IANCoordinator with guards) │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ InputValidator│          │  RateLimiter  │          │SybilResistance│
│ (Size, format)│          │(Token bucket) │          │  (PoW/PoS)    │
└───────────────┘          └───────────────┘          └───────────────┘
```

## Security Layers

### 1. Input Validation (`security.py`)

All inputs are validated at the system boundary **before** any processing.

```python
from idi.ian.security import SecurityLimits, InputValidator

limits = SecurityLimits(
    max_goal_id_length=64,
    max_contributor_id_length=256,
    max_pack_parameters_size=10 * 1024 * 1024,  # 10 MB
    max_pack_metadata_size=1 * 1024 * 1024,      # 1 MB
    max_proofs_size=1 * 1024 * 1024,             # 1 MB
)

validator = InputValidator(limits)
result = validator.validate_contribution(contribution)
if not result.valid:
    reject(result.reason)
```

**Validated Properties:**
- String lengths (goal_id, contributor_id, version)
- Binary sizes (parameters, proofs)
- UTF-8 encoding validity
- Nested structure depth

### 2. Rate Limiting (`security.py`)

Token bucket rate limiting per contributor prevents abuse.

```python
from idi.ian.security import RateLimiter

limiter = RateLimiter(
    tokens_per_contributor=10,           # Max burst
    refill_rate_per_second=0.1,          # 1 token per 10 seconds
    max_buckets=10_000,                  # Bounded memory
)

# Check before processing
allowed, remaining = limiter.check_and_consume("contributor_alice")
if not allowed:
    reject("Rate limited")
```

**Properties:**
- O(1) check and consume
- Bounded memory (LRU eviction at `max_buckets`)
- Per-contributor isolation

### 3. Sybil Resistance (`security.py`)

Optional proof-of-work prevents identity-based attacks.

```python
from idi.ian.security import SybilResistance, ProofOfWork

sybil = SybilResistance(
    difficulty=20,           # Leading zero bits required
    challenge_ttl_seconds=300,
    max_challenges=10_000,
)

# Server: Issue challenge
challenge = sybil.issue_challenge("contributor_id")

# Client: Solve challenge
pow = ProofOfWork.solve(challenge, difficulty=20)

# Server: Verify and consume
valid = sybil.verify_pow("contributor_id", pow)
```

**Properties:**
- Challenges are single-use (replay protection)
- TTL prevents stale challenges
- Difficulty adjustable per goal

### 4. Timing Attack Mitigation (`security.py`)

Security-sensitive operations are padded to constant time.

```python
from idi.ian.security import constant_time_compare, pad_execution_time
import time

# Constant-time string comparison
if constant_time_compare(provided_key, expected_key):
    allow()

# Pad execution to fixed time
start = time.monotonic()
result = sensitive_operation()
pad_execution_time(target_ms=100, start_time=start)
return result
```

**Async version available:**
```python
from idi.ian.security import pad_execution_time_async

await pad_execution_time_async(target_ms=100, start_time=start)
```

### 5. Sandboxed Evaluation (`sandbox.py`)

Agent evaluation runs in isolated subprocesses with strict limits.

```python
from idi.ian.sandbox import SandboxedEvaluator

evaluator = SandboxedEvaluator(
    max_memory_mb=512,
    cpu_time_limit=60,
    wall_time_limit=120,
)

# Runs in isolated subprocess
metrics = evaluator.evaluate(agent_pack, goal_spec, seed=42)
```

**Enforced Limits:**
- **Memory:** `setrlimit(RLIMIT_AS)` — Process killed if exceeded
- **CPU time:** `setrlimit(RLIMIT_CPU)` — SIGKILL on timeout
- **Wall time:** `multiprocessing.Process.join(timeout)` — Terminate if exceeded

**Security Fix (2025-12-10):** If `setrlimit` fails, evaluation is now **aborted** rather than continuing without limits.

### 6. Network Security (`network/`)

#### Connection Limits

```python
from idi.ian.network.transport import TCPTransport

transport = TCPTransport(
    max_connections=1024,        # Global limit
    max_connections_per_ip=64,   # Per-source limit
)
```

#### Read Timeouts

All socket reads have a 60-second timeout to prevent slow-loris attacks:

```python
# Internal implementation
data = await asyncio.wait_for(reader.readexactly(length), timeout=60.0)
```

#### API Rate Limiting

```python
from idi.ian.network.api import ApiConfig

config = ApiConfig(
    rate_limit_per_ip=100,  # Requests per minute
)
```

Rate limit tracking is bounded to 10,000 IPs with LRU eviction.

### 7. Cryptographic Identity (`network/node.py`)

Node identity uses Ed25519 for signing and verification.

```python
from idi.ian.network.node import NodeIdentity

identity = NodeIdentity.generate()
signature = identity.sign(message)
assert identity.verify(message, signature)
```

**⚠️ Warning:** If `cryptography` library is unavailable, a fallback HMAC implementation is used. This emits a warning at import time:

```
UserWarning: cryptography library not available. Node identity will use INSECURE fallback.
Install 'cryptography' for production use.
```

**In production, always install `cryptography`:**
```bash
pip install cryptography
```

## Secure Coordinator

The `SecureCoordinator` wraps `IANCoordinator` with all security measures:

```python
from idi.ian import IANCoordinator, SecureCoordinator, SecurityLimits

coordinator = IANCoordinator(goal_spec=spec)

secure = SecureCoordinator(
    coordinator,
    limits=SecurityLimits(
        max_pack_parameters_size=10 * 1024 * 1024,
        rate_limit_tokens=10,
        rate_limit_refill_per_second=0.1,
    ),
    enable_pow=True,  # Require proof-of-work
)

result = secure.process_contribution(contribution)
```

## Security Checklist

### Deployment

- [ ] Install `cryptography` library
- [ ] Configure TLS termination (reverse proxy)
- [ ] Set API key for authentication
- [ ] Configure rate limits appropriate for load
- [ ] Enable proof-of-work if Sybil attacks are a concern
- [ ] Set memory/CPU limits for evaluation
- [ ] Monitor connection counts and rate limit hits

### Code Review

- [ ] All user input validated before use
- [ ] No unbounded loops or recursion
- [ ] Secrets not logged
- [ ] Timing-sensitive operations use constant-time comparison
- [ ] Subprocess limits enforced (abort on failure)

## Vulnerability Reporting

If you discover a security vulnerability, please report it privately to the maintainers before public disclosure.

## Change Log

| Date | Issue | Fix | Signature |
|------|-------|-----|-----------|
| 2025-12-10 | Sandbox continued without limits on `setrlimit` failure | Abort evaluation if limits cannot be enforced | `0x6C9D2E4F3A8B5C7D` |
| 2025-12-10 | IPv6 address parsing failure | Handle `[::1]:port` format | `0x6C9D2E4F3A8B5C7D` |
| 2025-12-10 | Unbounded rate limit tracking | LRU eviction at 10,000 IPs | `0x6C9D2E4F3A8B5C7D` |
| 2025-12-10 | Slow-loris attack on TCP transport | 60-second read timeout | `0x4A7B9C2D1E8F3A6B` |
| 2025-12-10 | Silent fallback to insecure signing | Warning emitted at import time | `0x5B8C1D3E2F9A4B7C` |
