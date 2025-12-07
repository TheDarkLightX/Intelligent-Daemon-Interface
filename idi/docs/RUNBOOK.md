# IDI Operations Runbook

This runbook covers common issues, failure modes, and debug procedures for the Intelligent Daemon Interface (IDI) system.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Training Issues](#training-issues)
3. [Environment Issues](#environment-issues)
4. [Proof Generation Issues](#proof-generation-issues)
5. [Tau Spec Issues](#tau-spec-issues)
6. [Registry Issues](#registry-issues)
7. [Performance Issues](#performance-issues)

---

## Quick Reference

### Common Commands

```bash
# Run Python tests
cd idi/training/python && source .venv/bin/activate && python -m pytest

# Run Rust tests
cd idi/training/rust/idi_iann && cargo test

# Lint Python
ruff check idi_iann

# Lint Rust
cargo clippy -- -D warnings

# Generate traces
python run_idi_trainer.py --out build/traces --seed 42

# Run backtest
python backtest.py data.csv --config config.json --out outputs/

# Run OPE
python run_ope.py logged_data.json --policy policy.json --out ope_results.json
```

### Key Paths

| Component | Path |
|-----------|------|
| Python training | `idi/training/python/idi_iann/` |
| Rust training | `idi/training/rust/idi_iann/` |
| Tau specs | `idi/specs/` |
| zkVM host/guest | `idi/zk/risc0/` |
| Devkit | `idi/devkit/` |
| Configs | `idi/training/config_defaults.json` |
| Schemas | `idi/specs/schemas/` |

---

## Training Issues

### Issue: Training produces NaN rewards

**Symptoms**:
- `nan` values in episode rewards
- Q-values explode to infinity

**Causes**:
1. Learning rate too high
2. Reward values unbounded
3. Division by zero in environment

**Resolution**:
```python
# 1. Lower learning rate
config = TrainingConfig(learning_rate=0.05)  # Try 0.05 instead of 0.2

# 2. Add reward clipping in environment
reward = max(-100, min(100, raw_reward))

# 3. Check environment for division
# Look for / operations without zero checks
```

### Issue: Q-values don't converge

**Symptoms**:
- Episode rewards stay flat
- No improvement over baseline

**Causes**:
1. Exploration too low
2. State space too large
3. Discount factor misconfigured

**Resolution**:
```python
# 1. Increase exploration
config = TrainingConfig(exploration_decay=0.999)  # Slower decay

# 2. Use tile coding for state abstraction
config = TrainingConfig(
    tile_coder=TileCoderConfig(num_tilings=4, tile_sizes=(3,3,3,3,3))
)

# 3. Adjust discount
config = TrainingConfig(discount=0.95)  # Lower for short horizons
```

### Issue: OOM during training

**Symptoms**:
- Process killed
- MemoryError exception

**Causes**:
1. Q-table too large
2. Trace buffer unbounded
3. Memory leak in environment

**Resolution**:
```bash
# 1. Reduce state space
# Use coarser quantization buckets

# 2. Limit trace length
python run_idi_trainer.py --episode-length 64

# 3. Profile memory
python -m memory_profiler run_idi_trainer.py
```

---

## Environment Issues

### Issue: CryptoMarket regime stuck

**Symptoms**:
- Regime never changes from initial value
- Stress scenario forced regime

**Causes**:
1. Stress scenario has `forced_regime`
2. Low regime switch probability

**Resolution**:
```python
# Check if stress scenario is active
params = MarketParams(stress_scenario="normal")  # Use "normal" for dynamic regimes

# Verify regime switching
env = CryptoMarket(params)
for _ in range(100):
    env.step("hold")
print(f"Regime: {env.state.regime}")  # Should vary
```

### Issue: SyntheticMarketEnv determinism failure

**Symptoms**:
- Different results with same seed
- Non-reproducible traces

**Causes**:
1. Missing seed in config
2. Global RNG state pollution

**Resolution**:
```python
# 1. Pass explicit seed
env = SyntheticMarketEnv(quantizer, rewards, seed=42)

# 2. Isolate RNG
import random
env_rng = random.Random(42)  # Use dedicated RNG
```

---

## Proof Generation Issues

### Issue: Risc0 version mismatch

**Symptoms**:
- `Error: Your installation of the r0vm server is not compatible`

**Resolution**:
```bash
# 1. Check r0vm version
rzup --version

# 2. Update risc0-zkvm in Cargo.toml to match
# risc0-zkvm = "3.0.4"  # Match r0vm version

# 3. Rebuild
cargo clean && cargo build --release
```

### Issue: Proof generation timeout

**Symptoms**:
- Process hangs during proving
- No output from host

**Causes**:
1. Input too large
2. zkVM resource exhaustion

**Resolution**:
```bash
# 1. Reduce input size
# Limit trace length to 100 ticks

# 2. Use dev mode for testing
RISC0_DEV_MODE=1 cargo run --release -p idi_risc0_host

# 3. Add timeout
timeout 300 cargo run --release -p idi_risc0_host
```

### Issue: Proof verification fails

**Symptoms**:
- Receipt verification returns false
- Image ID mismatch

**Causes**:
1. Guest program changed
2. Wrong image ID stored
3. Corrupted receipt

**Resolution**:
```bash
# 1. Rebuild guest to get new image ID
cd idi/zk/risc0/methods && cargo build --release

# 2. Update stored image ID in host
# Check methods/src/lib.rs for IDI_MANIFEST_ID

# 3. Re-prove with correct guest
```

---

## Tau Spec Issues

### Issue: Stream not found error

**Symptoms**:
- `Failed to find output stream for stream 'i0'`

**Resolution**:
```bash
# Use REPL pipe pattern instead of file argument
echo -e "stream declarations\nr (spec)\nn" | tau
```

### Issue: Type mismatch in bitvector

**Symptoms**:
- `Invalid type for X: bv[N]`

**Causes**:
1. BV width mismatch
2. Operator not supported for BV

**Resolution**:
```tau
# Ensure consistent widths
i0 : bv[8] = in file("inputs/i0.in")
result : bv[8] = out file("outputs/result.out")

# Cast explicitly if needed
r (result[t] = i0[t] % { #x0F }:bv[8])
```

### Issue: Hex input parsing

**Symptoms**:
- `Unexpected 'x' at position N`

**Resolution**:
```bash
# Use #x prefix, not 0x
echo "#x0F" > inputs/value.in  # Correct
# NOT: echo "0x0F" > inputs/value.in
```

---

## Registry Issues

### Issue: Experiment not found

**Symptoms**:
- `None` returned from `get_experiment()`

**Causes**:
1. Wrong experiment ID
2. Registry directory changed
3. Index out of sync

**Resolution**:
```python
# 1. List all experiments
registry = ExperimentRegistry(Path("registry"))
for exp in registry.list_experiments():
    print(exp.experiment_id)

# 2. Check index file
import json
index = json.loads(Path("registry/index.json").read_text())
print(index)
```

### Issue: Policy bundle missing hashes

**Symptoms**:
- Empty `hashes` dict in PolicyBundle

**Resolution**:
```python
# Provide hashes when registering
import hashlib
policy_hash = hashlib.sha256(policy_json.encode()).hexdigest()
bundle = registry.register_policy(
    experiment_id=exp_id,
    policy_path="/path/to/policy.json",
    config_hash=config_hash,
    hashes={"policy": policy_hash},
)
```

---

## Performance Issues

### Issue: Slow training

**Symptoms**:
- Training takes hours for small configs
- Low CPU utilization

**Causes**:
1. Python interpreter overhead
2. Inefficient state encoding
3. Excessive logging

**Resolution**:
```python
# 1. Use Rust trainer for speed
cd idi/training/rust/idi_iann && cargo run --release --bin train

# 2. Profile hot spots
python -m cProfile -s cumtime run_idi_trainer.py

# 3. Disable verbose logging
import logging
logging.getLogger("idi_iann").setLevel(logging.WARNING)
```

### Issue: Large trace files

**Symptoms**:
- Gigabytes of `.in` files
- Slow trace export

**Resolution**:
```python
# 1. Compress traces
import gzip
with gzip.open("traces.json.gz", "wt") as f:
    json.dump(traces, f)

# 2. Reduce episode length
config = TrainingConfig(episode_length=64)  # Instead of 1000
```

---

## Debugging Checklist

1. **Check versions**: Python 3.12+, Rust stable, Risc0 3.0.4
2. **Check configs**: Validate against `config_schema.json`
3. **Check seeds**: Ensure deterministic seeds for reproducibility
4. **Check paths**: Use absolute paths to avoid working directory issues
5. **Check logs**: Enable debug logging for more context
6. **Check tests**: Run `pytest -v` and `cargo test` to verify setup

---

## Getting Help

1. Check existing documentation in `idi/docs/`
2. Search for similar issues in experiment registry
3. Review test cases for example usage
4. Check CI logs for recent failures

