# IDI Security Model and Threat Analysis

This document outlines the security model for the Intelligent Daemon Interface (IDI) training and proving pipeline.

## 1. System Overview

The IDI system consists of:
1. **Training Stack**: Python/Rust Q-learning with offline data
2. **Artifact Generation**: Q-tables, traces, manifests
3. **Proving Pipeline**: Risc0 zkVM for verifiable computation
4. **Tau Specs**: On-chain/verifiable agent logic
5. **Registry**: Experiment and policy tracking

## 2. Threat Model

### 2.1 Adversary Capabilities

We consider adversaries who may:
- **Corrupt Training Data**: Inject malicious samples to poison the learned policy
- **Tamper with Artifacts**: Modify Q-tables or traces after generation
- **Submit Malicious Inputs**: Provide adversarial states to deployed policies
- **Replay Old Artifacts**: Reuse outdated policies without proper versioning
- **Extract Private Information**: Learn proprietary Q-table strategies
- **Denial of Service**: Exhaust compute resources during proving

### 2.2 Trust Boundaries

| Boundary | Description | Trust Level |
|----------|-------------|-------------|
| Training Data | Offline logs from external sources | Untrusted |
| Training Code | Python/Rust training stack | Trusted (audited) |
| Generated Artifacts | Q-tables, traces | Verified via hashes |
| zkVM Guest | Risc0 guest program | Trusted (audited) |
| zkVM Proofs | Generated receipts | Cryptographically verified |
| Tau Specs | On-chain agent logic | Publicly auditable |

## 3. Security Controls

### 3.1 Data Integrity

#### 3.1.1 Hash Chains

All artifacts are linked via cryptographic hashes:

```
Training Config → Config Hash
    ↓
Q-Table → Q-Table Merkle Root
    ↓
Traces → Trace Stream Hashes
    ↓
Manifest → Manifest Hash (includes all above)
    ↓
zkVM Proof → Binds to Manifest Hash
```

#### 3.1.2 Manifest Schema

Every artifact bundle includes a manifest with:
- `schema_version`: Format version for compatibility
- `artifact_id`: Unique identifier derived from config+seed+timestamp
- `training_config`: Full config for reproducibility
- `policy_summary.states`: Q-table state count
- `trace_summary.stream_hashes`: SHA256 of each input stream
- `proof_policy`: Proof generation strategy

### 3.2 Input Validation

#### 3.2.1 Training-Time Validation

Before training:
1. **Config validation**: Check all parameters are within expected ranges
2. **Data schema validation**: Verify offline logs match expected format
3. **Seed validation**: Ensure deterministic seeds are provided

```python
def validate_config(config: TrainingConfig) -> None:
    assert 0 < config.discount <= 1.0
    assert 0 < config.learning_rate <= 1.0
    assert config.episodes > 0
    assert config.episode_length > 0
```

#### 3.2.2 Runtime Validation (Environments)

Environment implementations must:
1. Clamp state values to valid ranges
2. Validate action inputs
3. Bound rewards to prevent numerical overflow
4. Detect and flag anomalous transitions

```python
def step(self, action: str) -> Tuple[State, float]:
    if action not in self.ACTIONS:
        raise ValueError(f"Invalid action: {action}")
    # ... compute next state ...
    reward = max(-MAX_REWARD, min(MAX_REWARD, raw_reward))
    return state, reward
```

### 3.3 zkVM Integration Security

#### 3.3.1 Guest Program Constraints

The Risc0 guest program:
1. **Inputs**: Reads manifest JSON and trace stream bytes
2. **Computation**: Computes deterministic hash over all inputs
3. **Output**: Commits hash to journal (public output)

No private data is exposed in the proof; only the commitment is public.

#### 3.3.2 Proof Verification

Verification steps:
1. Retrieve proof receipt from storage
2. Verify receipt signature against known image ID
3. Extract committed hash from journal
4. Compare against expected manifest hash

### 3.4 Tau Spec Security

#### 3.4.1 Input Stream Validation

Tau specs should validate inputs:
- Check signal ranges (e.g., `q_regime < 32`)
- Enforce mutual exclusivity where needed
- Gate outputs on `risk_budget_ok`

#### 3.4.2 Safety Invariants

Encode safety rules directly in specs:
```tau
# Never buy and sell simultaneously
safety_ok = (buy_signal & sell_signal)'

# All outputs gated on safety
actual_buy = buy_signal & safety_ok & risk_budget_ok
```

## 4. Attack Scenarios and Mitigations

### 4.1 Data Poisoning

**Attack**: Adversary injects malicious training samples to bias the learned policy.

**Mitigations**:
1. Validate data source integrity before training
2. Use conservative Q-learning to distrust OOD samples
3. Monitor drift metrics between training and deployment data
4. Maintain behavior policy baselines for comparison

### 4.2 Artifact Tampering

**Attack**: Adversary modifies Q-table or traces after generation.

**Mitigations**:
1. Compute and store Merkle roots of Q-tables
2. Hash all trace streams and include in manifest
3. Verify hashes before any use (proving, deployment)
4. Store manifests in append-only registry

### 4.3 Replay Attacks

**Attack**: Adversary replays outdated policy to exploit known vulnerabilities.

**Mitigations**:
1. Version all artifacts with timestamps
2. Include git commit in experiment records
3. Enforce minimum freshness for deployed policies
4. Maintain revocation list for compromised artifacts

### 4.4 Model Extraction

**Attack**: Adversary queries deployed policy to reconstruct Q-table.

**Mitigations**:
1. Rate-limit policy queries
2. Add noise to outputs (differential privacy)
3. Use zk proofs to attest without revealing internals
4. Monitor for systematic probing patterns

### 4.5 Resource Exhaustion

**Attack**: Adversary submits large/complex inputs to exhaust proving resources.

**Mitigations**:
1. Enforce maximum trace length
2. Limit Q-table size in guest program
3. Set timeouts for proving operations
4. Implement proof caching for repeated requests

## 5. Operational Security

### 5.1 Key Management

| Key Type | Purpose | Storage |
|----------|---------|---------|
| Signing Key | Sign artifact manifests | Secure enclave / HSM |
| zkVM Image ID | Identify guest program | Version-controlled |
| Registry Access | Write to experiment registry | Role-based access |

### 5.2 Audit Logging

Log all:
- Training runs with full config
- Artifact generation events
- Proof generation/verification
- Policy deployments
- Registry modifications

### 5.3 Incident Response

1. **Detection**: Monitor drift metrics, verify proofs, check hashes
2. **Containment**: Revoke compromised artifacts, disable affected policies
3. **Recovery**: Regenerate from clean data, update keys if needed
4. **Post-mortem**: Document incident, update threat model

## 6. Implementation Checklist

### 6.1 Training Stack

- [ ] Config validation on startup
- [ ] Input schema validation for offline data
- [ ] Hash computation for generated Q-tables
- [ ] Deterministic training with seeded RNG
- [ ] Conservative Q-learning option

### 6.2 Artifact Pipeline

- [ ] Manifest generation with all hashes
- [ ] Schema validation for manifests
- [ ] Append-only registry storage
- [ ] Versioning with timestamps

### 6.3 Proving Pipeline

- [ ] Guest program input validation
- [ ] Size limits for inputs
- [ ] Timeout handling
- [ ] Proof caching
- [ ] Receipt verification

### 6.4 Deployment

- [ ] Freshness checks for policies
- [ ] Rate limiting
- [ ] Anomaly detection
- [ ] Revocation support

## 7. Future Considerations

1. **Multi-party Training**: Secure aggregation of Q-tables from multiple parties
2. **Differential Privacy**: Formal DP guarantees for published metrics
3. **Homomorphic Computation**: Policy evaluation without revealing Q-values
4. **Formal Verification**: Prove safety properties of Tau specs

