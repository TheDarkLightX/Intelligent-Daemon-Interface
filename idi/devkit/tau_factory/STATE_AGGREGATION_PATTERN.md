# State Aggregation Pattern

## Overview

The State Aggregation pattern combines multiple FSM states into a single superstate using various aggregation methods. This enables hierarchical state composition where multiple sub-states contribute to a higher-level state.

## Pattern Name

`state_aggregation`

## Description

This pattern aggregates the states of multiple FSMs into a single output state using methods like:
- **Majority**: Output is true if majority of inputs are true
- **Unanimous**: Output is true only if all inputs agree
- **Custom**: User-defined boolean expression
- **Mode**: Select mode based on which input is active

Common use cases:
- **Multi-Agent Coordination**: Aggregate decisions from multiple agents
- **State Composition**: Combine sub-states into superstate
- **Consensus Building**: Reach agreement from multiple sources
- **Mode Selection**: Choose mode based on active sub-states

## Use Cases

- **Ensemble Agents**: Combine multiple agent decisions
- **Hierarchical FSMs**: Build superstate from substates
- **Consensus Mechanisms**: Reach agreement from multiple sources
- **State Decomposition**: Aggregate decomposed states back together

## Schema Definition

```python
LogicBlock(
    pattern="state_aggregation",
    inputs=("fsm1_state", "fsm2_state", "fsm3_state"),
    output="aggregate_state",
    params={
        "method": "majority",  # or "unanimous", "custom", "mode"
        "threshold": 2,  # For majority: N-of-M threshold
        "total": 3,  # For majority: total number of inputs
        "expression": "...",  # For custom: boolean expression
        "initial_value": 0  # Initial output value
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `method` | str | No | `"majority"` | Aggregation method: "majority", "unanimous", "custom", "mode" |
| `threshold` | int | No | `(inputs+1)//2` | For majority: minimum number of true inputs |
| `total` | int | No | `len(inputs)` | For majority: total number of inputs |
| `expression` | str | Yes (custom) | None | For custom: boolean expression with input names |
| `initial_value` | int | No | 0 | Initial output value |

## Aggregation Methods

### 1. Majority

Output is true if at least `threshold` inputs are true.

**Example**: 2-of-3 majority
```python
params={
    "method": "majority",
    "threshold": 2,
    "total": 3
}
```

**Generated Logic**:
```tau
(o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t])) && (o0[0] = 0)
```

### 2. Unanimous

Output is true only if all inputs are true.

**Example**: All-agree consensus
```python
params={
    "method": "unanimous"
}
```

**Generated Logic**:
```tau
(o0[t] = i0[t] & i1[t] & i2[t]) && (o0[0] = 0)
```

### 3. Custom

User-defined boolean expression.

**Example**: Custom logic
```python
params={
    "method": "custom",
    "expression": "(fsm1_state[t] & fsm2_state[t]) | (fsm1_state[t]' & fsm2_state[t]')"
}
```

**Generated Logic**:
```tau
(o0[t] = (i0[t] & i1[t]) | (i0[t]' & i1[t]')) && (o0[0] = 0)
```

### 4. Mode

Select mode based on which input is active (first active input wins).

**Example**: Mode selection
```python
params={
    "method": "mode"
}
```

**Generated Logic**:
```tau
(o0[t] = (i0[t] ? {0}:bv[2] : (i1[t] ? {1}:bv[2] : (i2[t] ? {2}:bv[2] : {0}:bv[2])))) && (o0[0] = {0}:bv[2])
```

## Stream Requirements

### Input Streams

- **fsm1_state** (sbf or bv[N]): First FSM state
- **fsm2_state** (sbf or bv[N]): Second FSM state
- **fsmN_state** (sbf or bv[N]): Additional FSM states

### Output Stream

- **aggregate_state** (sbf or bv[N]): Aggregated state output

## Example Usage

### Majority Aggregation

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="ensemble_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="agent1_pos", stream_type="sbf"),
        StreamConfig(name="agent2_pos", stream_type="sbf"),
        StreamConfig(name="agent3_pos", stream_type="sbf"),
        StreamConfig(name="ensemble_pos", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="state_aggregation",
            inputs=("agent1_pos", "agent2_pos", "agent3_pos"),
            output="ensemble_pos",
            params={
                "method": "majority",
                "threshold": 2,
                "total": 3
            }
        ),
    ),
)
```

### Unanimous Consensus

```python
schema = AgentSchema(
    name="consensus_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="substate1", stream_type="sbf"),
        StreamConfig(name="substate2", stream_type="sbf"),
        StreamConfig(name="superstate", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="state_aggregation",
            inputs=("substate1", "substate2"),
            output="superstate",
            params={
                "method": "unanimous"
            }
        ),
    ),
)
```

### Custom Expression

```python
schema = AgentSchema(
    name="custom_aggregation_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="state_a", stream_type="sbf"),
        StreamConfig(name="state_b", stream_type="sbf"),
        StreamConfig(name="aggregate", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="state_aggregation",
            inputs=("state_a", "state_b"),
            output="aggregate",
            params={
                "method": "custom",
                "expression": "(state_a[t] & state_b[t]) | (state_a[t]' & state_b[t]')"
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Orthogonal Regions

```python
# Orthogonal regions for independent state tracking
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("execution_signal", "risk_signal"),
    output="execution_state",
    params={
        "regions": [...],
        "region_outputs": ["execution_state", "risk_state"]
    }
)

# Aggregate regions into superstate
LogicBlock(
    pattern="state_aggregation",
    inputs=("execution_state", "risk_state"),
    output="system_state",
    params={
        "method": "custom",
        "expression": "(execution_state[t] = {1}:bv[2]) & (risk_state[t] = {0}:bv[2])"
    }
)
```

### Combined with Multiple FSMs

```python
# Multiple independent FSMs
LogicBlock(pattern="fsm", inputs=("buy1", "sell1"), output="pos1")
LogicBlock(pattern="fsm", inputs=("buy2", "sell2"), output="pos2")
LogicBlock(pattern="fsm", inputs=("buy3", "sell3"), output="pos3")

# Aggregate FSM states
LogicBlock(
    pattern="state_aggregation",
    inputs=("pos1", "pos2", "pos3"),
    output="ensemble_position",
    params={
        "method": "majority",
        "threshold": 2,
        "total": 3
    }
)
```

## Limitations

1. **Simple Aggregation**: Currently supports basic boolean aggregation methods
2. **No Weighted Aggregation**: Cannot assign weights to different inputs (use Weighted Vote pattern instead)
3. **Mode Selection**: Mode method selects first active input, not most active

## Future Enhancements

1. **Weighted Aggregation**: Support weighted combination of states
2. **Complex Mode Selection**: Select mode based on priority or other criteria
3. **Temporal Aggregation**: Aggregate states over time windows
4. **State Transition Aggregation**: Aggregate state transitions, not just states

## Testing

See `tests/test_state_aggregation.py` for comprehensive test coverage.

## Related Patterns

- **Majority**: N-of-M voting pattern (similar logic)
- **Unanimous**: All-agree consensus pattern (similar logic)
- **Custom**: Custom boolean expressions (similar logic)
- **Weighted Vote**: Weighted aggregation with bitvector arithmetic
- **Orthogonal Regions**: Source of states to aggregate

## References

- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Ensemble Patterns](ENSEMBLE_PATTERNS.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [Tau Language Capabilities](../../specification/TAU_LANGUAGE_CAPABILITIES.md)

