# Native Tau 0.8 IAN boundary contracts

These six files replace earlier pseudocode with typed, executable constraints.
Each file's header defines its input mapping. Host code supplies verified facts
about one goal; Tau decides a bounded transition. Nothing here computes a
signature, Merkle proof, transaction hash, consensus result or finality.

| File | Decision |
| --- | --- |
| `ian_goal_register.tau` | Registration type, uniqueness, nonzero specification hash, name, governance kind and verified signature |
| `ian_log_commit.tau` | Registered goal, evidence flags, previous hash, bond, monotonic size and bounded leaderboard |
| `ian_upgrade.tau` | Policy identity, better unsigned fixed-point score, membership, monotonic clock, cooldown, governance and counter capacity |
| `ian_challenge.tau` | Open challenge window, bond, prior resolution and exclusive valid/invalid challenge settlement flags |
| `ian_economics.tau` | Authorized half-slash or checked deposit; exact remainder conservation |
| `ian_state.tau` | A one-goal fold of already-validated events; conflicting commit and upgrade hold state |

The economic example does not implement withdrawals or slash escalation. The
state fold does not provide a persistent registry. Governance never bypasses
cooldown in the new upgrade contract. These choices are explicit corrections to
incomplete historical sketches, not a compatibility claim for nonexistent
previous implementations.

The Python `TauBridge` uses independent pure preflight checks and an optional
governance-verifier callback. It does not automatically execute this entire Tau
rule set. In particular, its floating-point application score has no implicit
mapping to the Tau unsigned fixed-point score. A live Tau Net adapter must supply
a documented conversion, the cryptographic facts, current authoritative state,
and reconciliation of unknown transaction outcomes before claiming equivalence.

Native interpreter admission is in `qualification.json`; behavioral oracles are
in `../tests/test_tau08_contracts.py`. Acquire Tau separately under its license.
