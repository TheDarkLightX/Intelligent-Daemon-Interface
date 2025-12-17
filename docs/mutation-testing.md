# Mutation Testing for Security-Critical Code

This document describes how to use mutation testing to harden security properties in IDI/IAN.

## Scope (Security-Critical)

Run mutation testing only against modules where surviving mutants imply a security gap:

- `idi/ian/network/p2p_manager.py`
- `idi/ian/network/protocol.py`
- `idi/ian/network/node.py`

## Recommended Tooling

### Option A: mutmut

Install (dev-only):

```bash
pip install mutmut
```

Run (scoped):

```bash
mutmut run --paths-to-mutate idi/ian/network/p2p_manager.py \
           --paths-to-mutate idi/ian/network/protocol.py \
           --paths-to-mutate idi/ian/network/node.py

mutmut results
```

### Option B: cosmic-ray

Install (dev-only):

```bash
pip install cosmic-ray
```

Create a minimal `cosmic-ray` session and target the same modules.

## Security Mutant Policy (What Must Not Survive)

Treat surviving mutants in these categories as **P0**:

- Replay checks disabled (`if msg_id in cache` mutated)
- Timestamp freshness comparisons altered
- Signature verification bypassed or inverted
- Node ID ↔ public key binding removed
- Rate-limiter checks bypassed

## How to Respond to a Surviving Mutant

- Add a new **property** (preferred) instead of a one-off example.
- If the bug is sequence-dependent, add to the **stateful PBT** suite (`test_p2p_stateful_pbt.py`).
- If it is parsing-related, add to the **Atheris corpus** or a structured Hypothesis strategy.

## CI Guidance

Mutation testing is expensive; recommended approaches:

- Run on a nightly schedule
- Or run on demand for changes touching the scoped modules

Use a time budget (e.g., 5–15 minutes) and publish the mutant report as an artifact.
