# IDI / IAN Security Hardening Checklist (Internal)

**Owner:** Security / Platform

**Last updated:** 2025-12-13

## 0) Scope

This checklist focuses on **actionable security fixes** for the IDI repository components that are either:

- exposed to untrusted inputs (REST API, WebSockets, P2P), or
- materially affect security posture (secrets, deployment manifests, build/container hardening).

Primary targets (file evidence):

- `idi/ian/network/api.py` (aiohttp REST API; API key header; CORS middleware)
- `idi/ian/network/websocket_transport.py` (WebSocket server/client; currently trusts `node_id`)
- `idi/gui/backend/main.py` (FastAPI backend; permissive CORS)
- `idi/ian/deploy/kubernetes/deployment.yaml` (contains placeholder secret)
- `idi/ian/deploy/config/production.yaml` (CORS wildcard)
- `idi/ian/deploy/Dockerfile` (references missing `requirements.txt`)

## 1) Priority policy

- **P0 (Release blocker):** trivially exploitable, internet-reachable, or KEV-listed items; must be fixed before any production exposure.
- **P1 (≤30 days):** high severity issues or high-likelihood misconfigurations.
- **P2 (≤60 days):** medium severity, defense-in-depth, resilience improvements.
- **P3 (≤90 days):** backlog / hygiene improvements.

## 2) P0 Checklist (Release blockers)

### P0-1 Remove placeholder/default secrets from Kubernetes manifests

- **Risk:** default secrets are routinely deployed by accident; compromise becomes trivial.
- **Evidence:** `idi/ian/deploy/kubernetes/deployment.yaml` contains:
  - `stringData: IAN_API_KEY: "changeme-in-production"`

**Required change**

- **Delete** the `Secret` object from `idi/ian/deploy/kubernetes/deployment.yaml` OR remove the `IAN_API_KEY` field entirely.
- Require operators to create secrets out-of-band:
  - `kubectl -n ian create secret generic ian-secrets --from-literal=IAN_API_KEY='...'`
  - or ExternalSecrets / SealedSecrets (preferred).

**Acceptance criteria**

- The repo contains **no placeholder secret values**.
- Deploying without an operator-provided secret **fails closed** (pods do not start).

**Verification**

- `rg -n "changeme" idi/ian/deploy/kubernetes/deployment.yaml` returns no matches.
- `kubectl apply -f idi/ian/deploy/kubernetes/deployment.yaml` should fail or pods should crash if secret missing.

---

### P0-2 WebSocket authentication must be fail-closed (no trust of caller-provided `node_id`)

- **Risk:** current code sets `conn.node_id = node_id` without verifying possession of the private key or an API secret.
- **Evidence:** `idi/ian/network/websocket_transport.py`:
  - `_handle_authenticate()` accepts any `node_id`.

**Required change (minimum viable)**

- Introduce **explicit authentication modes** for WebSocket:
  - **API key mode:** require `X-API-Key` equivalent for WS messages (e.g., `api_key` field in the auth message) and compare against server-side configured key.
  - **Signature mode (preferred long-term):** implement a signed challenge/response similar to the existing TCP P2P handshake in `idi/ian/network/p2p_manager.py`.

**Hard requirement**

- **No privileged operation** is allowed unless `conn.is_authenticated()`.
  - At minimum: `subscribe`, `unsubscribe`, and any custom handlers must require authentication.

**Recommended design (signature-based)**

- **Server → client:** send a random `challenge_nonce` in the `welcome` message.
- **Client → server authenticate:** send:
  - `node_id`
  - `public_key` (base64)
  - `signature` over `H(challenge_nonce || node_id)` (or an equivalent canonical payload)
- **Server verifies:**
  - `node_id == sha256(public_key)[:40]`
  - signature verifies against `public_key`

**DbC specs**

- **Invariant:** `conn.node_id is None` iff connection is unauthenticated.
- **Preconditions:** `authenticate` message includes required fields, signature is correctly encoded.
- **Postconditions:** on success, `conn.node_id` is set and remains immutable for that connection.

**Verification**

- Add unit tests that assert:
  - authenticate with missing fields → rejected
  - authenticate with invalid signature → rejected
  - authenticate with mismatched `node_id` ↔ `public_key` → rejected
  - authenticate success enables subscription; unauth subscription fails

---

### P0-3 Tighten CORS (no wildcard in production)

- **Risk:** `Access-Control-Allow-Origin: *` broadens browser attack surface and enables drive-by interaction from arbitrary origins.

**Evidence**

- `idi/ian/deploy/config/production.yaml` has:
  - `cors_origins: ["*"]`
- `idi/ian/network/api.py` default:
  - `ApiConfig.cors_origins = ["*"]`
- `idi/gui/backend/main.py` has:
  - `allow_origins=["*"]`

**Required change**

- Production configs must specify an explicit allowlist.
- Runtime defaults should be **safe-by-default**:
  - Default CORS allowlist should be empty (deny cross-origin) unless explicitly configured.

**Verification**

- For IAN REST API:
  - Requests from unapproved `Origin` must not receive permissive CORS headers.
- For GUI backend:
  - If binding to non-localhost, cross-origin requests should be blocked unless origin explicitly allowlisted.

---

### P0-4 Fix REST CORS header behavior (must not emit comma-separated origins)

- **Risk:** `Access-Control-Allow-Origin` does not support multiple comma-separated origins. Emitting `a,b` is invalid and can lead to undefined client behavior.
- **Evidence:** `idi/ian/network/api.py` middleware sets:
  - `response.headers["Access-Control-Allow-Origin"] = ",".join(config.cors_origins)`

**Required change**

- Implement correct CORS behavior:
  - If allowlist is `*`, emit `*`.
  - Else, echo back the request `Origin` only if it is allowlisted.
  - Otherwise, emit no ACAO header.

**Verification**

- Unit tests for middleware:
  - allowlist `*` → `ACAO: *`
  - allowlist `[https://a]` and request origin `https://a` → `ACAO: https://a`
  - allowlist `[https://a]` and request origin `https://b` → no `ACAO`

## 3) P1 Checklist (≤30 days)

### P1-1 Ensure API key configuration is actually wired at runtime

- **Risk:** deployment docs/manifests reference `IAN_API_KEY`, but the server must load it into `ApiConfig(api_key=...)`.
- **Evidence:** `idi/ian/network/api.py` checks `ApiConfig.api_key`, but there is no canonical “from env/config” builder in `idi/ian/network`.

**Required change**

- Add a single, authoritative configuration builder that:
  - reads `IAN_API_KEY` from env (or secret mount) for REST + WS
  - applies port/host overrides
  - sets CORS allowlist

**Verification**

- Integration: start API server with `IAN_API_KEY` set; confirm unauthenticated `POST /api/v1/contribute` returns 401.

---

### P1-2 Deterministic Python dependencies (pin runtime deps)

- **Risk:** without pins/lock, CVE response is non-auditable and builds are non-reproducible.
- **Evidence:** root `pyproject.toml` only includes `zstandard` as runtime dependency.

**Required change**

- Decide one supported approach:
  - `pip-tools` (`requirements.in` → `requirements.txt` + hashes), or
  - Poetry/pdm/uv lock.

Minimum: pin internet-facing deps (e.g., `aiohttp`, `fastapi`, `uvicorn`, `starlette`, `pydantic`).

**Verification**

- Build produces consistent SBOM across runs.
- `pip-audit` / OSV scan runs in CI.

## 4) P2 Checklist (≤60 days)

### P2-1 Container hardening

- **Required changes**
  - ensure non-root runtime user (already present in `idi/ian/deploy/Dockerfile`)
  - drop Linux capabilities in K8s
  - prefer read-only root filesystem
  - set `securityContext` appropriately (seccomp/apparmor)

**Verification**

- Pod Security Admission / policy checks pass.
- Container scan (Trivy/Grype) is clean per policy.

---

### P2-2 Network policy tightening

- **Risk:** current NetworkPolicy allows API/WebSocket from anywhere.
- **Evidence:** `idi/ian/deploy/kubernetes/deployment.yaml` ingress allows ports 8000/9001 from all sources.

**Required change**

- Restrict ingress to:
  - ingress controller namespaces, or
  - internal VPC CIDRs (via LB firewall rules), or
  - explicitly approved namespaces/pods.

**Verification**

- Attempt access from disallowed namespace should fail.

## 5) P3 Checklist (≤90 days)

### P3-1 Align deployment artifacts with actual runnable entrypoints

- **Risk:** security settings in `deploy/config/*.yaml` and `idi/ian/deploy/Dockerfile` may not be enforced if the runtime does not load YAML config or the CLI entrypoint doesn’t implement the described subcommands.

**Required change**

- Either implement YAML config loading + `idi.ian.cli node start` command, or
- remove/replace non-functional deployment artifacts to avoid dangerous “assumed security”.

**Verification**

- Deploy paths are exercised in CI (smoke test): container boots, endpoints respond, auth enforced.

---

### P3-2 Documentation correctness

- **Risk:** docs that claim insecure crypto fallbacks exist can confuse operators.
- **Evidence:** `idi/ian/network/node.py` hard-requires `cryptography` and fails fast if it is missing.

**Required change**

- Update docs to match reality.

## 6) Recommended implementation order

1. P0-1: remove placeholder secrets
2. P0-2: WebSocket auth fail-closed
3. P0-3 / P0-4: CORS tightening + correct header behavior
4. P1-1: wire API key config
5. P1-2: pin runtime deps + CI scanning
6. P2/P3: container/network/documentation alignment

## 7) Verification command bundle (operator-friendly)

- `pytest idi/ian/tests/test_network.py -v`
- `pytest idi/ian/tests/test_security.py -v`
- `pytest -q idi/ian/tests/test_p2p_integration.py -q`
- `pytest -q idi/ian/tests/test_frontiersync.py::TestFrontierSyncIBLTAuthentication -q`
- `rg -n "changeme-in-production|allow_origins=\[\"\*\"\]|cors_origins:\s*\n\s*- \"\*\"" -S idi/`

