# Tau 0.8 and layered tables

Tau itself supports tables. Its current native `gen` / `compile` path can carry a
small strategy as a Mealy table, and `tau gen --machine` can synthesize a checked
bit-level strategy through Spot's `ltlsynt`. These facilities are distinct from
the four-entry Boolean tables learned by this experiment.

The inspected Tau source is
[`95a7d261ca3db5ff6f7c198a0b4d81c880816058`](https://github.com/IDNI/tau-lang/tree/95a7d261ca3db5ff6f7c198a0b4d81c880816058),
reporting version 0.8.0. Native qualification uses its official Python wheel
from CI artifact 13293842679. The artifact archive's SHA-256 is
`c58361a499c7d7f0de73d87cfd3bb6e0fce925f0286ff439794a0bc9f6fb5582`.

## Optional browser backend

The page accepts a matching `tau.esm.mjs` and `tau.esm.wasm` obtained independently
from IDNI. A `release-wasm-nothreads` build avoids a requirement for cross-origin
isolation headers on GitHub Pages. The loader requires an explicit license
acknowledgment and trusted local files. Loading a JavaScript module executes that
module in the page's worker; the files are not uploaded.

With Tau selected, the worker asks Tau to evaluate all 16 gate functions on all
four Boolean input pairs. It memoizes those 64 answers across the graph. The host
still extracts features, schedules nodes and sums integer class votes. This is
exact memoization of a finite function. It does not turn the browser into a
native Tau Mealy-table compiler or transfer proof authority to an unchecked host.

The current Wasm pack omits `bv`, `hsb`, `nlang` and external `ltlsynt`. The browser
adapter consequently uses `sbf`; native IAN bitvector contracts run separately.
The `set_charvar(false)` binding is required for the named stream syntax.

A local build from the pinned source, using Emscripten 6.0.13 and
`TAU_BAS=sbf,tau`, was loaded through the actual page in Chromium 157.
All 3,596 class votes across 58 validation inputs matched the Python fixtures.
Five conversation paths matched the independent evaluator, and desktop/mobile
layouts had no horizontal overflow. The runtime is about 6.3 MB plus a 117 KB
loader, supplied by the user separately. The adapter uses local Blob URLs through
Emscripten's `locateFile` option; this build rejects `wasmBinary` overrides.

To build the tested variant after checking out the pinned IDNI source and its
submodules, follow IDNI's prerequisites and run:

```sh
./dev preset release-wasm-nothreads -DTAU_BAS=sbf,tau -DTAU_LTO=OFF
```

Select the resulting matching `tau.esm.mjs` and `tau.esm.wasm` files in the page.
The page does not download or supply those framework files.

## Literal Tau graph export

`export_tau.py` emits the same frozen Boolean graph as 32-node Tau fragments.
The compressed manifest includes external signal indices, each fragment's source,
and the exact vote/bias map. It contains original specifications, not Tau code.

```sh
PYTHONPATH=. python -m idi.layered_chat.export_tau \
  --bundle idi/layered_chat/web/model.json --out /tmp/tau-fragments --qualify
```

`--qualify` needs a separately installed native `tau` Python module. It samples
32 fragments at fixed seed and checks two random global input assignments for
each against every emitted node bit. The complete folding certificate and the
exhaustive 64-entry gate-bank check provide separate, stronger finite guarantees.
No claim that the entire large graph was compiled by native `tau gen` is made.

## License boundary

The inspected [Tau license](https://github.com/IDNI/tau-lang/blob/95a7d261ca3db5ff6f7c198a0b4d81c880816058/LICENSE.md)
permits the stated educational/research use, excludes redistribution of the
framework or its portions, and describes downloading it from official channels
with end-user acceptance. User-loaded Tau remains subject to that license;
separate loading is not a blanket exemption for other uses.

The public page and this repository distribute our original tables, specifications,
independent evaluator, source attribution and measurements. They do not distribute
Tau source, runtime, wheels, Wasm, SDK, or a generated executable linked to Tau.
Native Tau and browser runtime build products remain outside the repository.
