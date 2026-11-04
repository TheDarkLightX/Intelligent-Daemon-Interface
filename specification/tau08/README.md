# Tau 0.8 migration status

The agent factory now emits Tau 0.8 file specifications by default: typed streams
use `:= in/out`, and the temporal requirement is an `always (...)` formula.
The runner uses file mode and `--charvar off`. `dialect="0.7"` and
`generate_legacy_tau_spec` retain the old REPL envelope explicitly.

All 122 recorded reference schemas are accepted by the native Tau 0.8 interpreter.
Their exact schemas, generated files, hashes and admission results are under
[`tau08_cases`](../../idi/devkit/tau_factory/tests/tau08_cases). This is one
reference schema per pattern, not a proof for every possible parameter set.
Eighteen native execution tests cover weighted votes, counter wrap and hold,
modular time boundaries, independent regions, proposal cancellation, entry/exit,
history restoration, a bounded TCP active-close trace, a complete 65,535-state LFSR cycle oracle,
and native transition traces for the two typed trading examples.

Eleven pattern emitters were rewritten as guarded equations. Transition guards
are disjoint, priorities are explicit, and stateful patterns have a hold case.
Combinational outputs no longer acquire an initial constraint that accidentally
restricts input values at step zero. Numeric widths are checked rather than
silently truncating impossible state sentinels. Single-input orthogonal regions
preserve the old level-selection behavior; they are not edge-triggered toggles.
Time-lock arithmetic remains modular in its declared width. The TCP example is
a minimal handshake/active-close model, not an implementation of the TCP RFCs.

## Historical inventory

[`inventory.json`](inventory.json) identifies all 106 original non-archive Tau
files and their original/current hashes. Fifty-eight received qualified envelope or type
ports in place. Six IAN sketches and three broken examples were replaced with explicit bounded contracts.
The remaining 39 files are historical sketches or exceeded the admission budget;
they are **not** presented as working Tau 0.8 applications. Their issues include
unsupported pseudocode, missing types and state, unimplemented cryptographic
operations, and ambiguous financial state transitions. Merely replacing syntax
would not establish their claimed behavior. The initial audit is retained in
`inventory-initial.json`.

The new [IAN rule set](../../idi/ian/tau_rules) separates host-verified facts from
Tau decisions. All six files load in Tau 0.8. Six native trace suites exercise
registration, monotonic commitments, upgrade evidence and clock boundaries,
challenge windows, conservative integer economics and the one-goal state fold.
There is no implementation of consensus, transaction finality, signature checks
or Merkle verification inside these contracts. Withdrawals and slash escalation
remain outside the economics contract, which covers half-slash and checked deposit.

## Recheck

Install Tau separately under its license, then from the repository root:

```sh
PYTHONPATH=. python -m idi.devkit.tau_factory.qualify_tau08
PYTHONPATH=. python -m pytest \
  idi/devkit/tau_factory/tests/test_tau08_execution.py \
  idi/ian/tests/test_tau08_contracts.py
```

`qualify_tau08` invokes the user-installed native Python API with a per-example
wall-clock budget. A refusal or timeout is recorded as such. Native tests skip
when that optional API is absent; the independent browser and pure-core tests
remain runnable without redistributing Tau.

The Python bridge now checks registration, hash lengths, monotonic log size,
integer bounds, finite scores and cooldowns before sending. Configured governance
requires an injected verifier returning exactly `True`; mere signature presence
is insufficient. Sender acknowledgement still does not establish chain finality.
