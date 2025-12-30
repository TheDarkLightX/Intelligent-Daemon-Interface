### Deep Security Analysis — IDI / IAN (Security Hardening + Decentralization)

**Date (retrieved):** 2025-12-30  
**Repo scope:** `Intelligent-Daemon-Interface/` (Python node + optional Next.js frontend + several Rust crates)

### 1) INVENTORY (from repo)

- **Primary runtime (Python)**
  - **Project config**: `pyproject.toml` (`requires-python = ">=3.10"`)  
    - **Core deps**: `aiohttp>=3.9.2`, `cryptography>=41.0.0`, `PyYAML>=6.0.1`, `zstandard>=0.22.0`, `py-ecc>=7.0.0`
    - **Optional extras**:
      - **`gui`**: `fastapi`, `uvicorn`, `websockets`
      - **`security`**: `keyring>=24.0.0` (optional OS keyring backend)
  - **Locked versions**: `uv.lock` (examples)
    - `aiohttp==3.13.2`, `cryptography==46.0.3`, `py-ecc==8.0.0`, `keyring==25.7.0`

- **Frontend (Node/Next.js)**
  - **App**: `idi/gui/frontend/package.json`
  - **Framework**: `next@16.0.10`, `react@19.2.1`, `react-dom@19.2.1`
  - **Lockfile**: `idi/gui/frontend/package-lock.json`

- **Rust components**
  - **Tau daemon**: `tau_daemon_alpha/` (`Cargo.toml`, `Cargo.lock`)
  - **IAN core crate**: `idi/ian/ian_core/` (`Cargo.toml`, `Cargo.lock`)
  - **ZK/RISC0**: `idi/zk/risc0/` (`Cargo.toml`, `Cargo.lock`)
  - **Devkit**: `idi/devkit/rust/` (`Cargo.toml`, `Cargo.lock`)

- **Containers**
  - **Root Docker build**: `Dockerfile` uses `python:3.12-slim` and `uv 0.7.13` (builder), installs wheels offline.
  - **IAN node image**: `idi/ian/deploy/Dockerfile` uses `python:3.12-slim`, creates non-root user `ian`, sets `IAN_CONFIG_PATH=/app/config/production.yaml`.

- **Networking / Auth / Crypto (application)**
  - **Node identity + Ed25519**: `idi/ian/network/node.py` (Ed25519 signatures; node_id derived from SHA-256(pubkey) prefix).
  - **P2P handshake key agreement**: `idi/ian/network/p2p_manager.py` uses X25519 + HKDF.
  - **TLS / mTLS support**: `idi/ian/network/tls.py` (min TLS 1.3 by default; optional pinning).
  - **Production API auth**: `idi/ian/deploy/config/production.yaml` sets `security.api_key_required: true`; `idi/ian/cli.py` requires `IAN_API_KEY` when `api_key_required` is enabled.

- **CI/CD**
  - No GitHub Actions workflows found under `.github/workflows/` in this repo snapshot.

### 2) THREAT MODEL (brief)

- **Assets**
  - **Node private key(s)** (Ed25519 identity; X25519 ephemeral/session keys).
  - **Economic state** (bonds / slashing) and **consensus state**.
  - **API key** (`IAN_API_KEY`) and any future signing keys used for Tau transactions.
  - **Contribution data** + logs + state snapshots.

- **Trust boundaries**
  - **Internet ↔ P2P listeners** (TCP / WebSocket transport; handshake and message parsing).
  - **Internet ↔ REST API** (auth + rate limiting + request parsing).
  - **Process ↔ filesystem / key store** (identity persistence; TLS key material).
  - **Build system ↔ dependency registries** (PyPI, crates.io, npm registry).

- **Attacker models**
  - Remote malicious peer (protocol abuse, replay, resource exhaustion).
  - Supply-chain attacker (dependency compromise, typosquatting, poisoned releases).
  - Host compromise / insider with filesystem access (private key exfil).
  - Malicious web client (WebSocket abuse, log injection, origin spoofing).

- **Worst outcomes**
  - Private-key compromise → identity spoofing, consensus manipulation, economic loss.
  - Remote DoS → node unavailability, liveness failure, stalled commit pipeline.
  - Supply-chain compromise → silent malicious code execution.

### 3) LATEST VULN INTEL (methods used)

- **Python deps**: `pip-audit` against frozen exports from `uv.lock` (prod + prod+security) → **no known vulnerabilities** found (scan date 2025-12-30).
- **Node deps**: `npm audit --package-lock-only` (frontend lockfile) → **0 vulnerabilities** found (scan date 2025-12-30).
- **Rust deps**: `cargo-audit` (RustSec DB) → **advisories found** (details below).  
  - RustSec DB reference: [RustSec advisory database](https://rustsec.org/advisory-database/)
- **CISA KEV cross-check**: CVE aliases from RustSec were checked against the official KEV JSON feed and were **not present** (as of 2025-12-30):  
  - [CISA KEV catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog)  
  - [KEV JSON feed](https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json)

### A) Vuln Digest (latest)

| ID | Component | Affected | Severity | KEV? | Fix / Mitigation | Evidence in repo | Source | Retrieved |
|---|---|---|---|---|---|---|---|---|
| RUSTSEC-2025-0047 (CVE-2025-55159 / GHSA-qx2v-8332-m4fv) | `slab` | `0.4.10` | N/A (no CVSS in advisory) | No | Upgrade to `slab >= 0.4.11` | `tau_daemon_alpha/Cargo.lock` (`slab 0.4.10`) | [GHSA-qx2v-8332-m4fv](https://github.com/tokio-rs/slab/security/advisories/GHSA-qx2v-8332-m4fv) | 2025-12-30 |
| RUSTSEC-2025-0055 (CVE-2025-58160 / GHSA-xwfj-jgwm-7wp5) | `tracing-subscriber` | `0.3.19` | N/A (no CVSS in advisory) | No | Upgrade to `tracing-subscriber >= 0.3.20` (ANSI escape sanitization) | `tau_daemon_alpha/Cargo.lock` (`tracing-subscriber 0.3.19`) | [GHSA-xwfj-jgwm-7wp5](https://github.com/advisories/GHSA-xwfj-jgwm-7wp5) | 2025-12-30 |
| YANKED | `rustls` | `0.23.30` | N/A | N/A | Move off yanked version (`cargo update` / pin a non-yanked release) | `tau_daemon_alpha/Cargo.lock` (`rustls 0.23.30`) | [crates.io rustls](https://crates.io/crates/rustls) | 2025-12-30 |
| YANKED | `slab` | `0.4.10` | N/A | N/A | Move to `>= 0.4.11` | `tau_daemon_alpha/Cargo.lock` (`slab 0.4.10`) | [crates.io slab](https://crates.io/crates/slab) | 2025-12-30 |
| RUSTSEC-2025-0020 (GHSA-pph8-gcv7-4qj5) | `pyo3` | `0.20.3` | N/A (no CVSS in advisory) | N/A | Upgrade to `pyo3 >= 0.24.1` or disable/remove Python bindings feature if not needed | `idi/ian/ian_core/Cargo.lock` (`pyo3 0.20.3`) | [PyO3 issue #5005](https://github.com/PyO3/pyo3/issues/5005) | 2025-12-30 |
| RUSTSEC-2023-0071 (CVE-2023-49092 / GHSA-c38w-74pg-36hr) | `rsa` | `0.9.9` | CVSS vector provided in advisory | No | **No patch** (per advisory): avoid using in contexts where remote timing observation is possible; re-evaluate crypto dependency | `idi/zk/risc0/Cargo.lock` (`rsa 0.9.9`) | [Marvin Attack note](https://people.redhat.com/~hkario/marvin/) | 2025-12-30 |
| RUSTSEC-2025-0137 (GHSA-9fjq-45qv-pcm7) | `ruint` | `1.17.0` | N/A (no CVSS in advisory) | N/A | Upgrade to `ruint >= 1.17.1` | `idi/zk/risc0/Cargo.lock` (`ruint 1.17.0`) | [recmo/uint issue #550](https://github.com/recmo/uint/issues/550) | 2025-12-30 |
| RUSTSEC-2025-0055 (CVE-2025-58160 / GHSA-xwfj-jgwm-7wp5) | `tracing-subscriber` | `0.2.25` | N/A (no CVSS in advisory) | No | Upgrade to `>= 0.3.20` (or otherwise ensure ANSI escaping in logs) | `idi/zk/risc0/Cargo.lock` (`tracing-subscriber 0.2.25`) | [GHSA-xwfj-jgwm-7wp5](https://github.com/advisories/GHSA-xwfj-jgwm-7wp5) | 2025-12-30 |
| RUSTSEC-2024-0388 (informational: unmaintained) | `derivative` | `2.2.0` | Informational | N/A | Replace with maintained alternative (`derive_more`, `derive-where`, `educe`, etc.) | `idi/devkit/rust/Cargo.lock` (`derivative 2.2.0`) | [rust-derivative issue #117](https://github.com/mcarton/rust-derivative/issues/117) | 2025-12-30 |
| RUSTSEC-2024-0384 (informational: unmaintained) | `instant` | `0.1.13` | Informational | N/A | Replace with maintained alternative (`web-time`, per advisory) | `idi/devkit/rust/Cargo.lock` (`instant 0.1.13`) | [crates.io instant](https://crates.io/crates/instant/0.1.13) | 2025-12-30 |
| RUSTSEC-2024-0436 (informational: unmaintained) | `paste` | `1.0.15` | Informational | N/A | Replace with maintained fork (e.g., `pastey`) or remove macro dependency | `idi/devkit/rust/Cargo.lock` (`paste 1.0.15`) | [dtolnay/paste](https://github.com/dtolnay/paste) | 2025-12-30 |
| YANKED | `flate2` | `1.1.7` | N/A | N/A | Move off yanked version (`cargo update`) | `idi/devkit/rust/Cargo.lock` (`flate2 1.1.7`) | [crates.io flate2](https://crates.io/crates/flate2) | 2025-12-30 |
| RUSTSEC-2024-0388 (informational: unmaintained) | `derivative` | `2.2.0` | Informational | N/A | Replace with maintained alternative | `idi/zk/risc0/Cargo.lock` (warned by `cargo-audit`) | [rust-derivative issue #117](https://github.com/mcarton/rust-derivative/issues/117) | 2025-12-30 |
| RUSTSEC-2024-0436 (informational: unmaintained) | `paste` | `1.0.15` | Informational | N/A | Replace with maintained fork or remove | `idi/zk/risc0/Cargo.lock` (warned by `cargo-audit`) | [dtolnay/paste](https://github.com/dtolnay/paste) | 2025-12-30 |
| YANKED | `toml_edit` | `0.23.8` | N/A | N/A | Move off yanked version (`cargo update`) | `idi/zk/risc0/Cargo.lock` (warned by `cargo-audit`) | [crates.io toml_edit](https://crates.io/crates/toml_edit) | 2025-12-30 |

### B) Security Hardening Checklist

| Priority | Control | Why | Exact change (repo/ops) | Verify | Sources |
|---|---|---|---|---|---|
| P0 | Store identity private key with least exposure | Key theft = identity spoofing | **DONE (repo):** `idi/ian/network/node.py` now does atomic `0600` writes + optional OS keyring via `keyring://...` refs; `IAN_IDENTITY_REF` supported in `idi/ian/cli.py` | Run `pytest -q idi/ian/tests/test_node_identity_key_storage.py`; verify file mode is `0600` and keyring path works | [OWASP Secrets Mgmt](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html), [NIST SP 800-57](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf), [keyring docs](https://keyring.readthedocs.io/en/latest/) |
| P0 | Use constant-time compares for auth bindings | Avoid timing side-channels | **DONE (repo):** constant-time comparisons already used in replay/identity checks where applicable (Python `hmac.compare_digest`) | Unit tests + code review; add microbench if desired | [Python hmac docs](https://docs.python.org/3/library/hmac.html) |
| P0 | Patch RustSec vulnerabilities in shipped Rust crates | Memory safety / log injection | **TODO (repo):** update `tau_daemon_alpha/Cargo.lock` off `slab 0.4.10` and `tracing-subscriber 0.3.19`; update `idi/zk/risc0/Cargo.lock` off `ruint 1.17.0` and `tracing-subscriber 0.2.25` | Re-run `cargo audit` in each crate dir; run crate tests/build | [RustSec DB](https://rustsec.org/advisory-database/) |
| P0 | Keep lockfiles in sync with declared deps | Reproducible builds; reduce supply-chain surprise | **DONE (repo):** refreshed `uv.lock` so `py-ecc` and the `security` extra are included | `uv lock --check` and Docker build should succeed | [SLSA](https://slsa.dev/) |
| P1 | Prefer keyless signing for artifacts + provenance | Reduce build/registry trust | **Ops/CI:** sign container images with Cosign; emit provenance (SLSA) | Verify with `cosign verify` + provenance verification | [Cosign docs](https://docs.sigstore.dev/cosign/), [SLSA](https://slsa.dev/) |
| P1 | Secure logging (avoid terminal/control injection) | Logs are an attacker-controlled channel | **Repo:** ensure Rust logging stack escapes untrusted content (upgrade `tracing-subscriber` per advisory); sanitize/structure logs | Add regression test logging hostile strings; verify output escapes | [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html), [GHSA-xwfj-jgwm-7wp5](https://github.com/advisories/GHSA-xwfj-jgwm-7wp5) |
| P1 | Enforce TLS 1.3 + mTLS for node↔node links (where feasible) | Protect transport confidentiality/integrity | **Ops/Repo:** ensure `TLSConfig.min_version=TLSv1_3`; configure CA + pinning for peer auth; require client certs for privileged endpoints | Integration test: connect with/without valid cert; verify rejection | [RFC 8446](https://datatracker.ietf.org/doc/html/rfc8446) |
| P2 | Rotate keys and secrets (planned) | Limit blast radius | **Ops:** rotation schedule + automation; support identity rollover + peer re-auth | Chaos test: rotate during operation; verify no split-brain | [NIST SP 800-57](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf) |
| P2 | Threat-intel prioritization with KEV | Patch what’s exploited first | **Ops/CI:** auto-check dependency CVEs against KEV feed and prioritize | Validate feed ingestion + alerting | [CISA KEV feed](https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json) |
| P3 | Add automated supply-chain scoring | Catch drift in OSS hygiene | **CI:** run OpenSSF Scorecard on key repos/components | Gate merges on minimum Scorecard thresholds | [OpenSSF Scorecard](https://github.com/ossf/scorecard), [scorecard.dev](https://scorecard.dev) |

### C) Decentralization Checklist

| Priority | Single point removed / trust reduced | Design change | Tradeoffs | Verify | Sources |
|---|---|---|---|---|---|
| P0 | Single-node key custody | Support optional OS keyring + future HSM/agent-based signing | Operational complexity | Cold-start restore works without plaintext key files | [NIST SP 800-57](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf), [OWASP Secrets Mgmt](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) |
| P1 | “One seed node” bootstrap dependence | Multi-seed default + rotating seed lists; consider DHT-based discovery | More networking complexity | Simulate seed outage; network still converges | [NIST SP 800-53](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf) (availability/control families) |
| P1 | Undocumented/unsigned upgrade channel | Signed release artifacts + provenance (SLSA) | Added release steps | Verify signatures + provenance during upgrade | [Cosign](https://docs.sigstore.dev/cosign/), [SLSA](https://slsa.dev/) |
| P2 | Centralized trust in a single CA | Certificate pinning set + explicit trust bundles; rotate trust bundles via governance | Operational overhead | Rotation drills; pinning prevents MitM | [RFC 8446](https://datatracker.ietf.org/doc/html/rfc8446) |
| P3 | Single implementation authority | Formalize upgrade governance (multi-party approval) and make policy auditable | Slower upgrades | Rehearse emergency patch process | [NIST SP 800-53](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf) |

### D) 30/60/90-day Plan

- **30 days (App/Sec, S–M)**
  - Patch RustSec findings in `tau_daemon_alpha/` and `idi/zk/risc0/` (upgrade `slab`, `tracing-subscriber`, `ruint`; move off yanked crates).
  - Document `IAN_IDENTITY_REF` usage (file vs `keyring://...`) in deployment docs; add “no plaintext keys” production guidance.
  - Add a simple “cargo audit” / “pip-audit” / “npm audit” runbook (CI integration can come later).

- **60 days (Infra/Sec, M)**
  - Ship container image signing + provenance in release pipeline (Cosign + SLSA).
  - Add rotation playbooks for `IAN_API_KEY`, TLS keys, and node identity (planned identity rollover).

- **90 days (App/Infra/Sec, M–L)**
  - Improve decentralized discovery (multi-seed → DHT/peer sampling), and verify via simulation.
  - Add explicit upgrade governance + signed policy bundles (reduce single-maintainer trust).


