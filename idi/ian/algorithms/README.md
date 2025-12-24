# IAN Novel Algorithms

**A-Grade Cryptographic Algorithms for the Intelligent Augmentation Network**

This module contains novel algorithms designed through systematic research (2023-2025) and iterative refinement with Codex. Each algorithm addresses a specific gap in the IAN codebase.

---

## Quick Start

```python
from idi.ian.algorithms import (
    # Invariant Verification
    DIVInvariantChecker,
    create_threshold_invariant,
    
    # Verifiable Evaluation
    VEPEvaluationHarness,
    
    # Fair Randomness
    TVRFCoordinator,
    create_evaluator_set,
    
    # Fraud Proofs
    IFPDisputeManager,
    IFPState,
)
```

---

## Algorithms Overview

| Algorithm | Grade | Purpose | Key Feature |
|-----------|-------|---------|-------------|
| **DIV** | A | Invariant Verification | Canonical AST + Merkle proofs |
| **VEP** | A | Verifiable Evaluation | Commitment traces + probabilistic audit |
| **TVRF** | A- | Fair Randomness | Threshold VRF + anti-grinding |
| **IFP** | A- | Dispute Resolution | O(log N) bisection protocol |

---

## Algorithm Details

### 1. DIV: Deterministic Invariant Verification

**Problem:** `PassthroughInvariantChecker` always returns True.

**Solution:** Canonical formula representation with tiered verification.

```python
from idi.ian.algorithms import DIVInvariantChecker, create_threshold_invariant

# Create invariant: reward >= 0, risk <= 0.5
invariant = create_threshold_invariant(min_reward=0.0, max_risk=0.5)

# Register with checker
checker = DIVInvariantChecker()
checker.register_invariant(invariant, goal_ids=["MY_GOAL"])

# Check contribution
passed, reason = checker.check(agent_pack, goal_spec)

# Get verifiable certificate
passed, reason, certificate = checker.check_with_certificate(
    agent_pack, goal_spec, contribution_hash
)
```

**Security Properties:**
- Canonical encoding prevents hash collisions
- NaN/infinity values rejected
- Size limits prevent DoS
- Merkle-bound attributes prevent spoofing

---

### 2. VEP: Verifiable Evaluation Protocol

**Problem:** `DummyEvaluationHarness` returns mock metrics.

**Solution:** Commitment-traced evaluation with probabilistic auditing.

```python
from idi.ian.algorithms import VEPEvaluationHarness, VEPAuditor
import secrets

# Setup evaluator
harness = VEPEvaluationHarness(
    evaluator_id=secrets.token_bytes(32),
    env_hash=secrets.token_bytes(32),
    num_audit_samples=32,
)

# Pre-commit (anti-abort)
precommit = harness.precommit(contribution_hash, slot=1)

# Evaluate with trace
metrics = harness.evaluate(agent_pack, goal_spec, seed=42)

# Get commitment for verification
commitment = harness.get_commitment(agent_pack.pack_hash)

# Auditor can verify
auditor = VEPAuditor(auditor_id=secrets.token_bytes(32))
request = auditor.create_audit_request(commitment, num_steps=1000, num_samples=32)
result = harness.respond_to_audit(request)
```

**Security Properties:**
- Trace commitment binds evaluation to result
- Audit indices derived from trace root (anti-grinding)
- Pre-commit prevents selective abort
- Merkle proofs for step verification

---

### 3. TVRF: Threshold VRF for Evaluator Selection

**Problem:** `secrets.choice()` is predictable and manipulable.

**Solution:** Threshold BLS VRF with commit-reveal protocol.

```python
from idi.ian.algorithms import TVRFCoordinator, create_evaluator_set
import secrets

# Create evaluator set (t-of-n threshold)
evaluator_set, keypairs = create_evaluator_set(
    epoch_id=1,
    evaluator_count=25,
    threshold_ratio=2/3,  # t=17
)

# Setup coordinator
coordinator = TVRFCoordinator()
coordinator.register_epoch(evaluator_set)

# Create seed for contribution
seed = coordinator.create_seed(
    epoch_id=1,
    contribution_hash=secrets.token_bytes(32),
    slot=100,
)

# Participants submit commits, then reveals
# ... (threshold participants)

# Aggregate VRF output
vrf_output = coordinator.aggregate_vrf(seed, evaluator_set)

# Select evaluators
selected = coordinator.select_evaluators(
    vrf_output,
    evaluator_set,
    num_to_select=9,
    stake_weighted=True,
)
```

**Security Properties:**
- Unpredictability: < t parties cannot predict output
- Uniqueness: One valid output per seed
- Bias-resistance: Selective abort deterred by slashing
- Two-step VRF prevents preview attacks

**⚠️ Production Note:** Replace `MockBLSOperations` with real BLS library (e.g., `py_ecc`, `blst`).

---

### 4. IFP: Interactive Fraud Proofs

**Problem:** Full state replay for disputes is expensive.

**Solution:** O(log N) bisection protocol with SMT state proofs.

```python
from idi.ian.algorithms import IFPDisputeManager, IFPState, create_state_trace
import secrets

# Create state trace
initial_state = IFPState.genesis()
contributions = [secrets.token_bytes(32) for _ in range(1000)]
states, trace_root = create_state_trace(initial_state, contributions)

# Setup dispute manager
manager = IFPDisputeManager()

# Asserter submits assertion
assertion = manager.submit_assertion(
    asserter_id=secrets.token_bytes(32),
    state_0=initial_state,
    state_n=states[-1],
    trace_root=trace_root,
    trace_len=len(contributions),
    da_commitment=secrets.token_bytes(32),
    bond=1000,
)

# Challenger disputes
dispute, _ = manager.submit_challenge(
    challenger_id=secrets.token_bytes(32),
    assertion_hash=assertion.assertion_hash(),
    bond=1000,
)

# Bisection rounds narrow to single step
# ... (O(log N) rounds)

# Estimate gas cost
init, rounds, final = manager.estimate_gas_cost(len(contributions))
print(f"Total gas: {init + rounds + final}")  # ~0.8-2M for 1M contributions
```

**Security Properties:**
- O(log N) on-chain cost
- SMT for membership/non-membership proofs
- Round timeouts prevent stalling
- DA requirement ensures data availability

**⚠️ Production Note:** SMT implementation is simplified; use a proper sparse Merkle tree library for production.

---

## Security Notes

### Production Requirements

1. **Replace Mock Cryptography:**
   - `MockBLSOperations` → Real BLS library (`py_ecc`, `blst`)
   - SNARK proofs → Real ZK library (`snarkjs`, `bellman`)

2. **External Dependencies:**
   - Slashing enforcement requires staking contract
   - Timeout enforcement requires consensus layer
   - DA commitments require data availability layer

3. **Rate Limiting:**
   - Audit requests should be rate-limited
   - Dispute creation requires bond

### Validated Security Fixes

- **DIV:** Length-prefixed canonical encoding, NaN/inf rejection, size limits
- **VEP:** Audit index validation, trace size limits
- **TVRF:** Rejection sampling for unbiased selection
- **IFP:** Turn enforcement, trace-root binding

---

## Testing

```bash
# Run all algorithm tests
python -m pytest idi/ian/algorithms/tests/ -v

# Run specific algorithm tests
python -m pytest idi/ian/algorithms/tests/test_div.py -v
python -m pytest idi/ian/algorithms/tests/test_vep.py -v
```

---

## Research Sources

- **zkML:** DeepProve-1, JOLT/Lasso, JSTprove (2024-2025)
- **Consensus:** Mysticeti, Shoal, Bullshark (2023-2024)
- **Fraud Proofs:** Cannon, Arbitrum Nitro (2023-2024)
- **VRFs:** RFC 9381, Traceable VRFs (2025)

---

## License

See project root LICENSE file.
