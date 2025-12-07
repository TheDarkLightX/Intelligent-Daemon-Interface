# Security Model for IDI ZK Proof Infrastructure

## Overview

This document defines the security model, threat assumptions, and trust boundaries for the Intelligent Daemon Interface (IDI) zero-knowledge proof system. The system enables privacy-preserving Q-table lookups and verifiable action selection for Tau Language agents.

## Security Properties

### 1. Soundness

**Definition**: A malicious prover cannot generate a valid proof for a false statement except with negligible probability.

**Application**:
- Merkle proof verification ensures Q-table entries match the committed root
- Action selection (argmax) verification ensures the selected action matches the Q-values
- Fixed-point arithmetic constraints prevent overflow/underflow attacks

**Threat**: Adversary attempts to prove incorrect Q-table lookups or action selections.

**Mitigation**: All critical computations are constrained within the Risc0 zkVM, ensuring mathematical correctness.

### 2. Completeness

**Definition**: An honest prover can always generate a valid proof for a true statement.

**Application**:
- Valid Q-table entries always produce valid Merkle proofs
- Correct action selections always verify successfully
- Witness generation succeeds for all valid inputs

**Threat**: Implementation bugs preventing honest provers from generating proofs.

**Mitigation**: Comprehensive testing, property-based tests, and fuzzing ensure all valid paths work correctly.

### 3. Zero-Knowledge

**Definition**: Proofs reveal no information about the private witness beyond what is explicitly committed.

**Application**:
- Q-table entries remain private (only Merkle roots are public)
- Individual Q-values are not revealed, only the selected action
- Full table structure is hidden from verifiers

**Threat**: Information leakage through public outputs or side channels.

**Mitigation**: Only commitments (hashes) are committed to the journal; no raw Q-values are exposed.

## Trust Boundaries

### 1. Guest vs Host (Risc0 zkVM)

**Guest Program** (`idi-qtable/src/main.rs`):
- **Trusted**: Executes in isolated zkVM environment
- **Proves**: Q-table lookup correctness, action selection correctness
- **Assumes**: Input data (witness) is provided correctly by host

**Host Program** (`risc0/host/src/main.rs`):
- **Trusted**: For proof generation (prover)
- **Untrusted**: For verification (verifier)
- **Responsibility**: Provides witness data, invokes prover, verifies receipts

**Security Boundary**: Guest program constraints ensure host cannot influence proof validity without providing correct witness.

### 2. Python vs Rust

**Python Modules** (`idi/zk/*.py`):
- **Trusted**: Witness generation, Merkle tree construction
- **Assumes**: Correct implementation of cryptographic primitives (SHA-256)
- **Risk**: Implementation bugs in Merkle tree logic

**Rust Guest Programs** (`risc0/methods/*/src/main.rs`):
- **Trusted**: Verification logic within zkVM
- **Proves**: Python witness generation was correct
- **Risk**: Mismatch between Python and Rust implementations

**Security Boundary**: Cross-implementation consistency tests ensure Python and Rust produce identical results.

### 3. On-Chain vs Off-Chain

**Off-Chain (Prover)**:
- **Trusted**: For generating proofs
- **Assumes**: Access to full Q-table, ability to compute Merkle proofs
- **Risk**: Malicious prover attempting to prove false statements

**On-Chain (Verifier)**:
- **Untrusted**: Verifies proofs without access to Q-table
- **Assumes**: Correct verification key, authentic root hash
- **Risk**: Accepting invalid proofs due to implementation bugs

**Security Boundary**: Cryptographic proof system ensures verifier cannot be fooled by malicious prover.

### 4. Tau Testnet Integration

**Bridge Module** (`idi/taunet_bridge/`):
- **Trusted**: For proof validation in transaction pipeline
- **Assumes**: Correct integration with Tau Testnet validation logic
- **Risk**: Bypass paths allowing unverified transactions

**Security Boundary**: ZK validation step is mandatory (when enabled) and cannot be skipped.

## Adversary Capabilities

### 1. Malicious Prover

**Capabilities**:
- Controls witness generation
- Can provide arbitrary Q-table entries
- Can attempt to forge Merkle proofs
- Can manipulate action selection logic

**Limitations**:
- Cannot break cryptographic hash functions (SHA-256)
- Cannot generate valid proofs without correct Merkle paths
- Cannot prove incorrect action selections due to zkVM constraints

**Attack Vectors**:
- Providing incorrect Q-table entries
- Attempting to use Merkle proofs from different trees
- Trying to prove actions that don't match Q-values

### 2. Malicious Verifier

**Capabilities**:
- Controls verification logic
- Can reject valid proofs
- Can attempt to extract information from proofs

**Limitations**:
- Cannot extract private Q-values from commitments
- Cannot forge proofs (doesn't have proving key)
- Cannot bypass zkVM verification

**Attack Vectors**:
- Rejecting valid proofs (availability attack)
- Attempting to correlate proofs to extract patterns
- Trying to use wrong verification keys

### 3. Network Adversary

**Capabilities**:
- Can intercept and modify network traffic
- Can replay proofs
- Can perform man-in-the-middle attacks

**Limitations**:
- Cannot forge proofs without proving key
- Cannot extract private data from proofs

**Attack Vectors**:
- Replaying old proofs (mitigated by nonces/timestamps)
- Modifying proof data in transit (detected by verification)
- Denial of service by flooding with invalid proofs

## Protected Assets

### 1. Q-Table Privacy

**Asset**: Full Q-table contents (state-action Q-values)

**Protection**: 
- Only Merkle roots are committed (public)
- Individual entries require Merkle proofs (private witness)
- No direct exposure of Q-values

**Threat**: Information leakage through proof correlation or side channels.

### 2. Proof Integrity

**Asset**: Correctness of Q-table lookups and action selections

**Protection**:
- Merkle proof verification ensures entry authenticity
- Action selection verification ensures correctness
- Cryptographic commitments prevent tampering

**Threat**: Forged proofs or incorrect computations.

### 3. Action Selection Correctness

**Asset**: Guarantee that selected action matches Q-table values

**Protection**:
- Argmax computation verified in zkVM
- Fixed-point arithmetic constraints prevent overflow
- Tie-breaking rules are deterministic

**Threat**: Incorrect action selection due to bugs or attacks.

## Security Assumptions

### 1. Cryptographic Assumptions

- **SHA-256**: Preimage resistance, collision resistance
- **Risc0 zkVM**: Soundness of STARK proof system
- **Fixed-Point Arithmetic**: Q16.16 representation is sufficient for Q-values

### 2. Implementation Assumptions

- **Python/Rust Consistency**: Both implementations produce identical Merkle roots
- **Determinism**: Same inputs always produce same outputs
- **No Side Channels**: Timing attacks don't leak information

### 3. Operational Assumptions

- **Key Management**: Proving keys are kept secure
- **Root Authenticity**: Verifiers have authentic Q-table root hashes
- **Network Security**: Proofs transmitted over secure channels

## Threat Mitigation Strategies

### 1. Underconstrained Circuits

**Threat**: Missing constraints allow multiple satisfying assignments.

**Mitigation**:
- Explicit bounds checking for all inputs
- Validation of Merkle proof path lengths
- Constraint coverage analysis

### 2. Public Data Leakage

**Threat**: Private Q-values accidentally committed to public outputs.

**Mitigation**:
- Audit all `env::commit()` calls
- Only commit hashes, never raw values
- Use domain separation in hash inputs

### 3. Implementation Bugs

**Threat**: Python/Rust mismatch or logic errors.

**Mitigation**:
- Cross-implementation consistency tests
- Property-based testing
- Fuzzing with edge cases

### 4. Replay Attacks

**Threat**: Reusing old proofs for new states.

**Mitigation**:
- Include state key in proof commitment
- Timestamp validation (future enhancement)
- Nonce-based replay protection (future enhancement)

## Security Guarantees

Given the security model above, the system provides:

1. **Privacy**: Q-table contents remain private; only commitments are revealed
2. **Correctness**: Proofs guarantee correct Q-table lookups and action selections
3. **Verifiability**: Anyone can verify proofs without access to private data
4. **Soundness**: Invalid proofs are rejected with high probability

## Limitations and Future Work

### Current Limitations

1. **No Replay Protection**: Proofs can be reused (mitigated by state key inclusion)
2. **No Timestamp Validation**: No freshness guarantees
3. **No Key Rotation**: Proving keys are static
4. **Limited Side-Channel Protection**: No explicit constant-time guarantees

### Future Enhancements

1. **Timestamped Proofs**: Include block height or timestamp in commitments
2. **Key Rotation**: Support for rotating proving keys
3. **Constant-Time Operations**: Explicit timing analysis
4. **Formal Verification**: Use Cryptol/SAW for Merkle tree verification

## References

- Risc0 zkVM Documentation: https://risc0.dev/
- Merkle Tree Security: Bitcoin BIP 34
- Zero-Knowledge Proofs: Goldwasser, Micali, Rackoff (1985)

