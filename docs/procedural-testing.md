# Procedural Generation Testing Guide

This document describes the deterministic, seeded property-based testing (PBT) infrastructure for IDI/IAN security hardening and Tau spec generation.

## Overview

The testing infrastructure uses [Hypothesis](https://hypothesis.readthedocs.io/) for procedural generation of:
- **IAN P2P Messages**: Valid and adversarial network protocol messages
- **Tau Agent Schemas**: Bounded specifications for code generation

All generators are **deterministic** - given the same seed, they produce identical outputs.

## Quick Start

```bash
# Run all PBT tests
pytest idi/ian/tests/test_p2p_pbt.py idi/devkit/tau_factory/tests/test_generator_pbt.py -v

# Reproduce a specific failure with seed
pytest idi/ian/tests/test_p2p_pbt.py --hypothesis-seed=12345 -v

# Extended fuzzing (more examples)
pytest idi/ian/tests/test_p2p_pbt.py --hypothesis-profile=ci -v
```

## Seed Logging and Reproduction

### Automatic Seed Logging

When a test fails, Hypothesis automatically logs the seed and a `@reproduce_failure` decorator:

```
Falsifying example: test_replay_detected_same_message_id(
    sender_id='a1b2c3...',
    nonce='abc123...',
)
You can reproduce this example by adding @reproduce_failure(...) decorator
```

### Manual Seed Control

```bash
# Run with specific seed
pytest --hypothesis-seed=42 -v

# Show seed in output
pytest --hypothesis-verbosity=verbose -v
```

### Database-Based Reproduction

Hypothesis stores failing examples in `.hypothesis/examples/`. To preserve across runs:

```bash
# Keep database between CI runs
export HYPOTHESIS_DATABASE_FILE=.hypothesis/examples.db

# Or in pytest.ini / pyproject.toml:
[tool.hypothesis]
database = ".hypothesis/examples.db"
```

## CI Integration

### GitHub Actions Example

```yaml
# .github/workflows/pbt.yml
name: Property-Based Tests

on: [push, pull_request]

jobs:
  pbt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run PBT (time-budgeted)
        run: |
          pytest idi/ian/tests/test_p2p_pbt.py \
                 idi/devkit/tau_factory/tests/test_generator_pbt.py \
                 --hypothesis-profile=ci \
                 -v --tb=short
        env:
          HYPOTHESIS_SEED: ${{ github.run_id }}
      
      - name: Upload failure database
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: hypothesis-failures
          path: .hypothesis/
```

### Hypothesis Profiles

Add to `pyproject.toml`:

```toml
[tool.hypothesis]
# Default profile (local dev)
default = { max_examples = 100, deadline = null }

# CI profile (extended fuzzing)
ci = { max_examples = 500, deadline = null, suppress_health_check = ["too_slow"] }

# Quick profile (pre-commit)
quick = { max_examples = 20, deadline = 1000 }
```

Usage:
```bash
pytest --hypothesis-profile=ci -v
```

## Test Coverage

### IAN P2P Security Tests (`test_p2p_pbt.py`)

| Test | Invariant |
|------|-----------|
| `test_replay_detected_same_message_id` | Same (sender_id, nonce) = replay |
| `test_different_nonces_not_replay` | Different nonces accepted |
| `test_replay_cache_bounded_memory` | Cache ≤ max_size |
| `test_stale_timestamp_rejected` | timestamp > 5min old rejected |
| `test_future_timestamp_rejected` | timestamp > 5min future rejected |
| `test_fresh_timestamp_accepted` | timestamp within window accepted |
| `test_valid_message_roundtrip` | JSON serialize/deserialize preserves fields |
| `test_malformed_json_no_crash` | Invalid JSON doesn't crash |
| `test_missing_fields_handled` | Missing required fields handled gracefully |
| `test_token_bucket_bounds` | Rate limiter tokens ∈ [0, capacity] |
| `test_burst_exhausts_tokens` | Burst exhausts tokens correctly |
| `test_node_id_pubkey_binding` | node_id = hash(pubkey)[:40] |
| `test_signature_length_validation` | Only 64-byte sigs valid |
| `test_message_id_uniqueness` | msg_id = sender_id:nonce |
| `test_adversarial_messages_dont_crash` | Adversarial inputs don't raise |

### Tau Spec Generator Tests (`test_generator_pbt.py`)

| Test | Invariant |
|------|-----------|
| `test_valid_schema_passes_validation` | Generated schemas valid |
| `test_generation_produces_string` | Generator outputs non-empty string |
| `test_spec_size_bounded` | spec_bytes ≤ 100KB |
| `test_generation_time_bounded` | generation_time ≤ 5s |
| `test_spec_contains_required_sections` | Has inputs, outputs, defs, run, quit |
| `test_spec_input_count_matches_schema` | input declarations = schema.inputs |
| `test_spec_output_count_matches_schema` | output declarations = schema.outputs |
| `test_spec_step_count_matches_schema` | 'n' commands = num_steps |
| `test_balanced_parentheses` | count('(') == count(')') |
| `test_balanced_brackets` | count('[') == count(']') |
| `test_balanced_quotes` | count('"') % 2 == 0 |
| `test_bv_type_in_spec` | bv schemas produce bv[] types |
| `test_bv_width_matches_schema` | bv width in spec matches schema |

## Strategy Reference

### IAN P2P Strategies (`idi/ian/tests/strategies.py`)

```python
from idi.ian.tests.strategies import (
    # Primitives
    node_id_strategy,        # 40-char hex
    nonce_strategy,          # base64 bytes
    timestamp_strategy,      # fixed reference ± 5min
    signature_strategy,      # 64 bytes or None
    
    # Valid messages
    valid_message_strategy,  # Any valid P2P message
    ping_message_strategy,
    handshake_challenge_strategy,
    
    # Adversarial
    adversarial_message_strategy,
    stale_timestamp_message_strategy,
    malformed_json_strategy,
    missing_fields_message_strategy,
    
    # DoS simulation
    message_burst_strategy,
    
    # Deterministic helpers
    make_deterministic_node_id,
    make_deterministic_nonce,
)
```

### Tau Schema Strategies (`idi/devkit/tau_factory/tests/strategies.py`)

```python
from idi.devkit.tau_factory.tests.strategies import (
    # Schema generation
    agent_schema_strategy,           # Full schema with constraints
    minimal_agent_schema_strategy,   # 1 input, 1 output, 1 block
    bitvector_agent_schema_strategy, # bv[] types
    
    # Components
    stream_config_strategy,
    logic_block_strategy,
    
    # Deterministic helper
    make_deterministic_schema,
)
```

## Determinism Guarantees

All generators use:
- **Fixed reference timestamp**: `1704067200000` (2024-01-01 UTC)
- **Hypothesis-seeded randomness**: Reproducible via `--hypothesis-seed=X`
- **No external state**: No `time.time()`, `secrets.*`, or `random.*` calls

To verify determinism:
```bash
# Run twice with same seed, diff output
pytest --hypothesis-seed=42 -v 2>&1 | tee run1.log
pytest --hypothesis-seed=42 -v 2>&1 | tee run2.log
diff run1.log run2.log  # Should be empty
```

## Extending the Tests

### Adding a New Message Type Strategy

```python
@st.composite
def my_new_message_strategy(draw: st.DrawFn) -> Dict[str, Any]:
    base = draw(base_message_fields())
    base["type"] = "my_new_type"
    base["custom_field"] = draw(st.text(min_size=1, max_size=50))
    return base
```

### Adding a New Security Invariant Test

```python
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(msg=valid_message_strategy())
@settings(max_examples=100, deadline=None)
def test_my_invariant(self, msg: Dict[str, Any]) -> None:
    """My security invariant holds for all messages."""
    # Setup
    ...
    # Assert invariant
    assert my_invariant_holds(msg)
```

## Troubleshooting

### Flaky Tests
If Hypothesis reports "Inconsistent data generation":
1. Check for `time.time()`, `secrets.*`, `random.*` calls
2. Use fixed reference values instead
3. Ensure all randomness comes from `draw()` calls

### Slow Tests
```bash
# Reduce examples for local dev
pytest --hypothesis-profile=quick -v

# Or set deadline
@settings(max_examples=50, deadline=1000)  # 1 second deadline
```

### Reproducing CI Failures
1. Download `.hypothesis/` artifact from CI
2. Place in local project root
3. Run: `pytest --hypothesis-database=.hypothesis/examples.db -v`
