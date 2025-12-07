# IDI Data Contracts

This document defines the data contracts for offline logs, training datasets, and generated artifacts in the Intelligent Daemon Interface (IDI) system.

## Overview

Data contracts ensure consistency and compatibility between components:
- Python and Rust training stacks
- Tau spec inputs/outputs
- zkVM proving pipeline
- External tools and dashboards

## 1. Training Configuration Contract

**Schema**: `idi/training/config_schema.json`

Required fields:
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| episodes | integer | >= 1 | Number of training episodes |
| episode_length | integer | >= 1 | Maximum steps per episode |
| discount | float | (0, 1] | Discount factor (gamma) |
| learning_rate | float | (0, 1] | Learning rate (alpha) |
| exploration_decay | float | (0, 1] | Epsilon decay rate |

Nested objects:
- `quantizer`: State quantization bucket counts
- `rewards`: Reward component weights
- `emote`: Emotional expression configuration
- `layers`: Layered strategy configuration
- `communication`: Communication action set

**Versioning**: The schema includes a version field. Changes to required fields require version bump.

## 2. Trace Tick Contract

**Schema**: `idi/specs/schemas/trace_schema.json`

A trace consists of metadata and an array of ticks.

### 2.1 Trace Metadata

Required fields:
| Field | Type | Description |
|-------|------|-------------|
| episode_id | string | Globally unique episode identifier |
| policy_id | string | Identifier for generating policy |
| env_id | string | Environment identifier (e.g., "synthetic", "crypto") |
| data_version | string | Trace format version (semver) |

Optional fields:
| Field | Type | Description |
|-------|------|-------------|
| seed | integer | RNG seed for reproducibility |
| config_hash | string | SHA256 of training config |
| timestamp | datetime | Generation timestamp (ISO 8601) |

### 2.2 Tick Fields

Required signal fields:
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| step_id | integer | >= 0 | Step index within episode |
| q_buy | integer | 0-1 | Buy signal |
| q_sell | integer | 0-1 | Sell signal |
| risk_budget_ok | integer | 0-1 | Risk budget gate |
| q_emote_positive | integer | 0-1 | Positive emotion cue |
| q_emote_alert | integer | 0-1 | Alert emotion cue |
| q_regime | integer | 0-31 | Regime identifier (5-bit) |

Optional signal fields:
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| q_emote_persistence | integer | 0-1 | Emotion linger flag |
| price_up | integer | 0-1 | Price increased |
| price_down | integer | 0-1 | Price decreased |
| weight_momentum | integer | 0-1 | Momentum layer active |
| weight_contra | integer | 0-1 | Contrarian layer active |
| weight_trend | integer | 0-1 | Trend layer active |
| risk_event | integer | 0-1 | Risk event flag |

## 3. Logged Dataset Contract

**Schema**: `idi/training/python/idi_iann/ope.py` (LoggedDataset)

Used for off-policy evaluation and behavior analysis.

### 3.1 Episode Fields

| Field | Type | Description |
|-------|------|-------------|
| episode_id | string | Unique episode identifier |
| behavior_policy_id | string | Policy that generated the data |
| config_hash | string | Training config hash |
| data_version | string | Dataset format version |
| transitions | array | List of transitions |

### 3.2 Transition Fields

| Field | Type | Description |
|-------|------|-------------|
| state | array[int] | Quantized state tuple |
| action | string | Action taken ("hold", "buy", "sell") |
| reward | float | Reward received |
| next_state | array[int] | Next state tuple |
| behavior_prob | float | P(action) under behavior policy |
| done | boolean | Episode termination flag |

## 4. Policy Bundle Contract

**Schema**: `idi/specs/schemas/manifest_schema.json`

Defines the canonical format for trained policy artifacts.

Required fields:
| Field | Type | Description |
|-------|------|-------------|
| schema_version | string | Manifest format version |
| artifact_id | string | Unique artifact identifier |
| timestamp | datetime | Generation timestamp |
| training_config | object | Full training config used |
| policy_summary | object | Policy statistics |
| trace_summary | object | Trace file hashes |
| proof_policy | string | Proof strategy ("risc0", "stub", "none") |

### 4.1 Policy Summary

| Field | Type | Description |
|-------|------|-------------|
| states | integer | Number of states in Q-table |
| actions | array[string] | Available actions |

### 4.2 Trace Summary

| Field | Type | Description |
|-------|------|-------------|
| length | integer | Number of ticks |
| stream_hashes | object | SHA256 of each .in file |

## 5. Drift Report Contract

**Schema**: `idi/training/python/idi_iann/drift.py` (ShiftReport)

Used for distribution shift detection.

| Field | Type | Description |
|-------|------|-------------|
| reference_version | string | Reference dataset identifier |
| comparison_version | string | Comparison dataset identifier |
| overall_score | float | Aggregate drift score (PSI-based) |
| has_significant_drift | boolean | Any feature exceeds threshold |
| feature_metrics | array | Per-feature drift metrics |
| timestamp | datetime | Report generation time |

### 5.1 Feature Metrics

| Field | Type | Description |
|-------|------|-------------|
| feature_name | string | Feature identifier |
| ks_statistic | float | Kolmogorov-Smirnov statistic |
| psi | float | Population Stability Index |
| wasserstein_approx | float | Approximate Wasserstein distance |
| mean_diff | float | Mean difference |
| std_diff | float | Standard deviation difference |
| is_significant | boolean | Drift exceeds threshold |

## 6. Validation Rules

### 6.1 State Tuple

- Length must match quantizer dimensions (default: 5)
- All values must be non-negative integers
- Values must be within bucket range for each dimension

### 6.2 Actions

- Must be one of: "hold", "buy", "sell"
- No consecutive identical non-hold actions (optional guardrail)

### 6.3 Signals

- Binary signals (0/1) must not have values outside [0, 1]
- Regime must be in range [0, 31]
- Mutually exclusive signals (e.g., price_up and price_down) should not both be 1

## 7. Lineage Tracking

Every artifact should include lineage information:

```
Dataset -> Config -> Q-Table -> Traces -> Proofs -> Tau Specs
   |         |         |          |         |          |
   v         v         v          v         v          v
 hash      hash      hash       hash      hash       hash
```

The manifest ties these together with:
- `training_config` reference
- `trace_summary.stream_hashes` for trace files
- `prover_metadata` for proof artifacts

## 8. Versioning Policy

- Breaking changes to required fields: major version bump
- New optional fields: minor version bump
- Documentation/description changes: patch version bump

Current versions:
- Config schema: 1.0.0
- Trace schema: 1.1.0 (added episode-level tracing)
- Manifest schema: 1.0.0

