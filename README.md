# Deflationary Agents (Tau Network)

This repository contains the V35 Deflationary Agent kernel specification and an enhanced Rust daemon (Tau Daemon Alpha) that acts as an exoskeleton around a lean Tau kernel.

## Structure
- `specification/` — Tau specs, including `agent4_testnet_v35.tau`
- `tau_daemon_alpha/` — Rust workspace: daemon + core libs
- `inputs/`, `outputs/` — IO directories used by specs
- `verification/` — Formal checks (Z3 proofs, etc.)

## Quickstart
1) Build Tau locally (internal-only)
```bash
cd tau_daemon_alpha
scripts/build_tau_local.sh
```
2) Run daemon
```bash
cd tau_daemon_alpha
RUST_LOG=info cargo run -p daemon
```
3) Or run the spec directly
```bash
tau_daemon_alpha/bin/tau specification/agent4_testnet_v35.tau
```

## License notes
- Tau Language from IDNI: build locally for internal testing only; do not distribute built artifacts. See `LICENSE.txt` in `tau-lang` (`https://github.com/IDNI/tau-lang`).

## Development
- Config in `tau_daemon_alpha/tau_daemon.toml`
- Daemon writes ledger to `tau_daemon_alpha/ledger/`
- Kernel remains lean; daemon handles guards/monitors/actuation.
