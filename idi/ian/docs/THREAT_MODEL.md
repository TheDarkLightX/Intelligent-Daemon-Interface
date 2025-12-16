# IAN Threat Model

**Version:** 1.0  
**Date:** 2025-12-15  
**Status:** Living Document  
**Methodology:** STRIDE + Attack Trees

---

## 1. System Overview

IAN (Intelligent Agent Network) is a decentralized coordination layer for AI agents, providing:
- **TwigMMR**: Append-only Merkle Mountain Range for verifiable logs
- **FrontierSync**: Peer-to-peer log synchronization with witness cosigning
- **SlotBatcher**: Fair transaction ordering using VRF-based randomness
- **IBLT**: Set reconciliation for efficient sync
- **MeritRank**: Sybil-resistant reputation using random walks

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL (Untrusted)                      │
│  • P2P Network Messages                                          │
│  • REST/WebSocket API Requests                                   │
│  • CLI Arguments                                                 │
│  • Configuration Files                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BOUNDARY VALIDATION LAYER                     │
│  • Input validation (size, format, bounds)                       │
│  • Authentication (Ed25519 signatures)                           │
│  • Rate limiting (per-peer token bucket)                         │
│  • Replay protection (nonce + timestamp)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INTERNAL (Trusted)                          │
│  • Core business logic                                           │
│  • Cryptographic operations                                      │
│  • State management                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         STORAGE                                  │
│  • TwigMMR files (JSON, integrity-hashed)                        │
│  • Kubernetes secrets (external)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. STRIDE Analysis

### 2.1 Spoofing (S)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| S-01 | P2P Network | Attacker impersonates legitimate node | Ed25519 signature verification on handshake | ✅ Implemented |
| S-02 | FrontierSync | Attacker forges witness signatures | Ed25519 verification + witness diversity check | ✅ Implemented |
| S-03 | SlotBatcher | Attacker predicts/forges VRF output | Ed25519-based VRF with proof verification | ✅ Implemented |
| S-04 | API | Attacker impersonates contributor | API key + optional Ed25519 challenge-response | ⚠️ Partial |

### 2.2 Tampering (T)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| T-01 | TwigMMR | Attacker modifies stored logs | SHA-256 integrity hash on each twig file | ✅ Implemented |
| T-02 | IBLT | Attacker modifies IBLT in transit | HMAC-SHA256 authentication | ✅ Implemented |
| T-03 | FrontierSync | Attacker modifies sync state | Cryptographic cosigning with threshold | ✅ Implemented |
| T-04 | Config | Attacker modifies K8s secrets | K8s RBAC + sealed secrets recommended | ⚠️ Operational |

### 2.3 Repudiation (R)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| R-01 | All | Actor denies actions | SecurityAuditLogger with structured JSON | ✅ Implemented |
| R-02 | FrontierSync | Witness denies signing | Signature stored in CosignedSyncState | ✅ Implemented |
| R-03 | MeritRank | Evaluator denies evaluation | contribution_hash in EvaluationEdge | ✅ Implemented |

### 2.4 Information Disclosure (I)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| I-01 | API | Private keys leaked in logs | SecurityAuditLogger redacts sensitive fields | ✅ Implemented |
| I-02 | K8s | Secrets exposed in manifests | Reference secrets by name, not value | ✅ Implemented |
| I-03 | Network | Traffic intercepted | TLS Ingress with HSTS | ✅ Implemented |
| I-04 | Error Messages | Stack traces expose internals | Production logging config | ⚠️ Operational |

### 2.5 Denial of Service (D)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| D-01 | P2P | Message flood | Rate limiting per peer | ✅ Implemented |
| D-02 | IBLT | Oversized IBLT decode | MAX_DECODE_ITERATIONS bound | ✅ Implemented |
| D-03 | MeritRank | Graph explosion | MAX_NODES, MAX_EDGES bounds | ✅ Implemented |
| D-04 | FrontierSync | Nonce cache exhaustion | Bounded LRU cache (MAX_NONCE_CACHE_SIZE) | ✅ Implemented |
| D-05 | TwigMMR | Disk exhaustion | Bounded twig count per goal | ⚠️ Partial |
| D-06 | API | Request flood | Ingress rate limiting (100 rps) | ✅ Implemented |

### 2.6 Elevation of Privilege (E)

| Threat ID | Component | Threat | Mitigation | Status |
|-----------|-----------|--------|------------|--------|
| E-01 | Container | Container escape | SecurityContext: runAsNonRoot, drop ALL caps | ✅ Implemented |
| E-02 | TwigMMR | Path traversal to escape storage | Path validation + canonicalization | ✅ Implemented |
| E-03 | Deserialization | Code execution via pickle | Replaced with JSON serialization | ✅ Implemented |
| E-04 | K8s | Pod to cluster escalation | NetworkPolicy + RBAC | ✅ Implemented |

---

## 3. Attack Trees

### 3.1 Compromise Log Integrity

```
[Goal: Corrupt TwigMMR Log]
├── [AND] Bypass integrity check
│   ├── Obtain write access to storage
│   │   ├── Container escape (E-01) - MITIGATED
│   │   └── Path traversal (E-02) - MITIGATED
│   └── Forge integrity hash
│       └── Requires SHA-256 preimage - INFEASIBLE
├── [OR] Forge cosigned state
│   ├── Compromise threshold witnesses
│   │   ├── Sybil attack (single entity) - MITIGATED (diversity check)
│   │   └── Compromise N/2+1 keys - OPERATIONAL RISK
│   └── Forge Ed25519 signatures - INFEASIBLE
└── [OR] Replay old valid state
    └── Replay protection (nonce + timestamp) - MITIGATED
```

### 3.2 Manipulate Transaction Ordering

```
[Goal: Gain Unfair Ordering Advantage]
├── [OR] Predict VRF output
│   ├── Compromise VRF private key - OPERATIONAL RISK
│   ├── Weak randomness source - MITIGATED (cryptography lib required)
│   └── Brute force VRF - INFEASIBLE
├── [OR] Front-run transactions
│   └── Observe mempool + inject - MITIGATED (commit-reveal + VRF)
└── [OR] Manipulate quality weights
    └── Sybil evaluations - MITIGATED (MeritRank decay)
```

### 3.3 Sybil Attack on Reputation

```
[Goal: Gain Undeserved Reputation]
├── [OR] Create many identities
│   ├── Cheap identity creation - MITIGATED (stake requirement)
│   └── Compromise existing identities - OPERATIONAL RISK
├── [OR] Circular evaluations
│   └── Transitivity decay (α=0.15) - MITIGATED
├── [OR] Bridge attack (connect clusters)
│   └── Bridge detection + slashing - MITIGATED
└── [OR] Control seed nodes
    ├── < 3 seeds - MITIGATED (min seed requirement)
    └── Compromise 50%+ seeds - OPERATIONAL RISK
```

---

## 4. Data Flow Diagram

```
┌─────────┐         ┌─────────────┐         ┌──────────────┐
│  Peer   │ ──P2P──▶│ P2PManager  │ ──────▶ │ FrontierSync │
│  Node   │◀────────│ (validate)  │ ◀────── │   (IBLT)     │
└─────────┘         └─────────────┘         └──────────────┘
                          │                        │
                          │                        ▼
┌─────────┐         ┌─────────────┐         ┌──────────────┐
│  API    │ ──REST─▶│ InputValid  │ ──────▶ │   TwigMMR    │
│ Client  │◀────────│ (security)  │ ◀────── │  (storage)   │
└─────────┘         └─────────────┘         └──────────────┘
                          │                        │
                          ▼                        ▼
                    ┌─────────────┐         ┌──────────────┐
                    │ SlotBatcher │ ◀─────▶ │  MeritRank   │
                    │   (VRF)     │         │ (reputation) │
                    └─────────────┘         └──────────────┘
```

---

## 5. Security Controls Summary

| Layer | Control | Implementation |
|-------|---------|----------------|
| **Network** | TLS 1.3 | Ingress with HSTS |
| **Network** | Rate limiting | 100 rps per IP |
| **Auth** | Signatures | Ed25519 |
| **Auth** | Replay protection | Nonce + timestamp |
| **Data** | Integrity | SHA-256 HMAC |
| **Data** | Serialization | JSON (no pickle) |
| **Container** | Isolation | SecurityContext |
| **Container** | Least privilege | Drop ALL caps |
| **Logging** | Audit trail | SecurityAuditLogger |
| **Consensus** | Witness diversity | min_unique_entities |
| **Randomness** | VRF | Ed25519 + require_crypto |

---

## 6. Residual Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Compromised VRF key | Low | High | Key rotation (24h default) |
| Compromised witness majority | Low | Critical | Threshold increase, stake slashing |
| Zero-day in cryptography lib | Very Low | Critical | Dependency scanning, SBOM |
| Insider threat (operator) | Low | High | RBAC, audit logs, key escrow |
| Supply chain attack | Low | Critical | Pinned deps, SBOM, provenance |

---

## 7. Review Schedule

- **Quarterly:** Review threat model against new features
- **On incident:** Update attack trees with new vectors
- **On dependency update:** Verify no new CVEs introduced
- **Annually:** External penetration test

---

## 8. References

- [OWASP Threat Modeling](https://owasp.org/www-community/Threat_Modeling)
- [STRIDE](https://docs.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats)
- [CWE Top 25](https://cwe.mitre.org/top25/archive/2023/2023_top25_list.html)
- [NIST SP 800-154](https://csrc.nist.gov/publications/detail/sp/800-154/draft)
