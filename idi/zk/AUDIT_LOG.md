# ZK Bridge Audit Log

This log tracks hardening work on `idi/taunet_bridge` and `idi/zk`.

## Environment
- Scope: our codebase only; no Tau Testnet edits.
- Target: production-grade, deterministic, low-complexity ZK bridge.

## Audits

### Dependency Audit
- Python: pip-audit/safety (pending run; add results here).
- Rust: cargo audit (pending run; add results here).

### Threat Model Refresh
- Boundaries: ZK verifier interface, config bridge, gossip helper, block/state helpers.
- Risks: malformed proofs, oversized payloads, replay attempts, unknown fields, missing domain separation.

## Planned Hardening (tracking)
- Input validation & caps (size/timeouts) for proofs, Merkle paths, configs.
- Determinism: canonical serialization, domain-separated hashes (Rust hash already done).
- Tests: property tests for validation/config/gossip/proof; fuzz targets (Atheris, cargo-fuzz); negative tests (replay/oversize/unknown fields).
- Logging/metrics: structured, redacted logs; optional metrics hooks.
- Docs: update INTEGRATION_GUIDE.md and EXAMPLES.md with limits, failure modes, operational guidance; maintainer patch notes.

