# Performance Benchmark Guide

This guide explains how to run the Tau-native and Python simulation benchmarks defined in `verification/performance_benchmark.py`.

## Prerequisites
- Python 3.9+
- Optional: Tau binary built from `tau-lang` for native benchmarks
- Repository cloned locally (all paths in this guide are relative to the repo root)

## Environment configuration
- `TAU_PATH` (optional): Absolute or relative path to your Tau binary. Defaults to `tau` on the `PATH`.
- `TAU_SPEC_DIR` (optional): Directory containing Tau specifications if you want to point at an alternate spec set. Defaults to `specification/` in the repo.

## Running the suite
From the repository root:

```bash
python verification/performance_benchmark.py
```

Behavior:
- If `TAU_PATH` resolves to a valid binary, the Tau micro-benchmarks will run followed by the Python economic simulation benchmark.
- If the Tau binary is unavailable, the script will warn you and skip the Tau micro-benchmarks while still running the simulation benchmark.

## Outputs
- Human-readable benchmark summary is printed to stdout.
- Structured results are written to `verification/performance_results.json` for downstream analysis or dashboards.

## Interpreting results
- Average/min/max and standard deviation are reported for each operation.
- The “Performance Tiers” section in the console output provides quick heuristics:
  - `<10ms` Excellent (real-time)
  - `10–100ms` Good (interactive)
  - `100–1000ms` Acceptable (batch)
  - `>1000ms` Slow (optimization needed)

## Troubleshooting
- "Tau binary not found": Set `TAU_PATH` to your built binary, e.g. `export TAU_PATH=$HOME/.local/bin/tau`.
- Timeouts: Increase the timeout per call by adjusting the `timeout` argument in `run_tau_command` if your environment is slow.
- Empty results: If Tau benchmarks are skipped or fail, confirm the binary path and that your specs are accessible.

## Next steps
- Integrate the JSON output with your preferred dashboarding tool to track performance over time.
- Adjust iteration counts in `run_all_benchmarks` to trade off between runtime and statistical confidence.
