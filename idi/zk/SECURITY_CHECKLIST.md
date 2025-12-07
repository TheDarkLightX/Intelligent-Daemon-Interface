# ZK Security Audit Checklist

This checklist is used for systematic security auditing of the IDI ZK proof infrastructure. Each item should be verified and documented.

## Underconstrained Circuits

### Risc0 Guest Program (`idi-qtable/src/main.rs`)

- [ ] **Witness Variable Constraints**
  - [ ] All witness variables are used in constraints
  - [ ] No unused variables that could allow multiple satisfying assignments
  - [ ] Fixed-point arithmetic bounds are checked (Q16.16 overflow/underflow)

- [ ] **Merkle Proof Validation**
  - [ ] Proof path length is validated against expected tree depth
  - [ ] Sibling ordering is correctly constrained (left vs right)
  - [ ] Root hash comparison is properly constrained (not just asserted)

- [ ] **Action Selection Constraints**
  - [ ] Argmax computation is fully constrained (no ambiguity)
  - [ ] Tie-breaking rules are deterministic and constrained
  - [ ] Action index bounds are validated (0, 1, 2 only)

### Python Witness Generation (`witness_generator.py`)

- [ ] **Consistency Checks**
  - [ ] Witness generator produces same results as zkVM verification
  - [ ] Action selection matches between Python and Rust
  - [ ] Merkle proof generation matches Rust verification logic

## Public vs Private Data Leakage

### Rust Guest Programs

- [ ] **Commit Audit**
  - [ ] All `env::commit()` calls reviewed
  - [ ] No private Q-table entries committed
  - [ ] Only commitments (hashes) are committed, not raw values
  - [ ] Domain separation used in hash inputs (prefix tags)

- [ ] **Public Outputs**
  - [ ] Journal entries contain only public data
  - [ ] No debug output leaks private information
  - [ ] Error messages don't reveal sensitive data

### Python Modules

- [ ] **Serialization**
  - [ ] Witness serialization doesn't expose private data
  - [ ] Logging doesn't include Q-values
  - [ ] Debug mode doesn't print sensitive information

## Merkle Tree Security

### Leaf Encoding

- [ ] **Domain Separation**
  - [ ] Leaf encoding includes key prefix or domain tag
  - [ ] Different tree types use different encoding schemes
  - [ ] Encoding is deterministic and canonical

- [ ] **Consistency**
  - [ ] Python and Rust use identical leaf encoding
  - [ ] Sibling ordering matches between implementations
  - [ ] Tree depth calculation is consistent

### Tree Construction

- [ ] **Determinism**
  - [ ] Tree construction is deterministic (sorted keys)
  - [ ] Odd node handling is consistent (duplication vs padding)
  - [ ] Root hash is unique for each leaf set

- [ ] **Proof Generation**
  - [ ] Proof paths are correctly computed
  - [ ] Sibling hashes are correctly identified
  - [ ] Proof verification matches proof generation

### Index Validation

- [ ] **Bounds Checking**
  - [ ] Tree depth is validated
  - [ ] Leaf indices are within bounds
  - [ ] Proof path length matches tree depth

## Witness Generation Security

### Input Validation

- [ ] **State Key Validation**
  - [ ] State keys are validated before use
  - [ ] Invalid state keys are rejected
  - [ ] State key format is enforced

- [ ] **Q-Value Validation**
  - [ ] Q-values are within expected range
  - [ ] Fixed-point conversion doesn't overflow
  - [ ] NaN/Inf values are handled

### Action Selection

- [ ] **Determinism**
  - [ ] Same Q-values always produce same action
  - [ ] Tie-breaking is deterministic
  - [ ] No randomness in action selection

- [ ] **Correctness**
  - [ ] Action selection matches argmax(Q-values)
  - [ ] Python and Rust implementations match
  - [ ] Edge cases handled correctly (all equal, all zero)

## Cryptographic Primitives

### Hash Functions

- [ ] **SHA-256 Usage**
  - [ ] SHA-256 used correctly (no truncation)
  - [ ] Domain separation applied (prefix tags)
  - [ ] Hash inputs are properly formatted

- [ ] **Collision Resistance**
  - [ ] No known collision vulnerabilities
  - [ ] Input length is validated
  - [ ] Hash outputs are full 32 bytes

### Fixed-Point Arithmetic

- [ ] **Q16.16 Format**
  - [ ] Conversion preserves precision
  - [ ] Overflow/underflow handled correctly
  - [ ] Rounding is deterministic

- [ ] **Arithmetic Operations**
  - [ ] Addition/subtraction don't overflow
  - [ ] Comparisons are correct
  - [ ] Edge cases handled (INT32_MIN, INT32_MAX)

## Integration Security

### Tau Testnet Bridge

- [ ] **Validation Pipeline**
  - [ ] ZK validation step cannot be bypassed
  - [ ] Invalid proofs are rejected
  - [ ] Error handling doesn't leak information

- [ ] **Proof Propagation**
  - [ ] Proofs are validated before propagation
  - [ ] Invalid proofs are not broadcast
  - [ ] Replay protection (if implemented)

### Error Handling

- [ ] **Failure Modes**
  - [ ] Invalid inputs fail gracefully
  - [ ] Error messages don't reveal sensitive data
  - [ ] Exceptions don't expose internal state

- [ ] **Resource Limits**
  - [ ] Max tree depth enforced
  - [ ] Max proof size enforced
  - [ ] Memory limits respected

## Testing and Verification

### Property-Based Testing

- [ ] **Merkle Tree Properties**
  - [ ] Proof roundtrip tests (generate → verify)
  - [ ] Tamper detection tests
  - [ ] Consistency tests (Python vs Rust)

- [ ] **Witness Properties**
  - [ ] Action selection determinism
  - [ ] Fixed-point roundtrip
  - [ ] Cross-implementation consistency

### Fuzzing

- [ ] **Coverage**
  - [ ] Merkle tree construction fuzzed
  - [ ] Proof verification fuzzed
  - [ ] Witness generation fuzzed

- [ ] **Edge Cases**
  - [ ] Empty tables
  - [ ] Single entry tables
  - [ ] Power-of-2 sizes
  - [ ] Boundary Q-values

### Integration Testing

- [ ] **End-to-End**
  - [ ] Full workflow tested (witness → proof → verify)
  - [ ] Invalid proofs rejected
  - [ ] Cross-table proof rejection

## Code Quality

### Complexity

- [ ] **Cyclomatic Complexity**
  - [ ] All functions < 10 complexity
  - [ ] Nested loops extracted
  - [ ] Long if/elif chains refactored

- [ ] **Cognitive Complexity**
  - [ ] Code is readable
  - [ ] Intent is clear
  - [ ] Comments explain why, not what

### Type Safety

- [ ] **Type Hints**
  - [ ] All public functions typed
  - [ ] Security-critical data uses NewType
  - [ ] Immutable data uses frozen dataclasses

### Documentation

- [ ] **Security Documentation**
  - [ ] Security properties documented
  - [ ] Trust assumptions documented
  - [ ] Threat model documented

- [ ] **Code Comments**
  - [ ] Security rationale explained
  - [ ] Invariants documented
  - [ ] References to specs/papers included

## Operational Security

### Key Management

- [ ] **Proving Keys**
  - [ ] Keys stored securely
  - [ ] Key rotation plan documented
  - [ ] Key compromise procedures defined

### Logging

- [ ] **Security Logging**
  - [ ] Proof operations logged
  - [ ] Failed verifications logged
  - [ ] No sensitive data in logs

### Monitoring

- [ ] **Anomaly Detection**
  - [ ] Unusual proof patterns detected
  - [ ] Failed verification rate monitored
  - [ ] Performance metrics tracked

## Audit Findings Template

For each checklist item, document:

1. **Status**: ✅ Pass / ⚠️ Warning / ❌ Fail
2. **Evidence**: Code references, test results, analysis
3. **Risk Level**: Low / Medium / High / Critical
4. **Remediation**: Required fixes or mitigations
5. **Verification**: How fix was verified

## Example Audit Entry

```
Item: Merkle Proof Path Length Validation
Status: ⚠️ Warning
Evidence: 
  - idi-qtable/src/main.rs:60 - Proof path length not explicitly validated
  - merkle_tree.py:102 - get_proof() doesn't check depth
Risk Level: Medium
Remediation: Add explicit depth validation in Rust guest program
Verification: Added test_merkle_proof_depth_validation() - passes
```

