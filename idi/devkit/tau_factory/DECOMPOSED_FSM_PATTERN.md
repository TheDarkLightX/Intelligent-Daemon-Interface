# Decomposed FSM Pattern

## Overview

The Decomposed FSM pattern implements hierarchical state decomposition: breaks down a superstate into multiple substates, generating separate FSMs for each substate and optionally aggregating them back into the superstate.

## Pattern Name

`decomposed_fsm`

## Description

This pattern decomposes a complex FSM into simpler substate FSMs:
- **Substate FSMs**: Each substate has its own FSM
- **Hierarchical Structure**: Substates belong to superstates
- **State Transitions**: Transitions between substates
- **Aggregation**: Optional aggregation of substates into superstate

**Key Operations:**
1. **Decompose**: Break superstate into substates
2. **Generate FSMs**: Create FSM for each substate
3. **Handle Transitions**: Manage transitions between substates
4. **Aggregate**: Combine substates into superstate (optional)

## Use Cases

- **Complex State Machines**: Break down complex FSMs into manageable parts
- **Hierarchical Design**: Model hierarchical state relationships
- **State Abstraction**: Abstract complex states into simpler components
- **Modular FSMs**: Build modular, reusable FSM components

## Schema Definition

```python
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell"),
    output="idle_low",
    params={
        "hierarchy": {
            "IDLE": {
                "substates": ["idle_low", "idle_high"],
                "initial": "idle_low"
            },
            "POSITION": {
                "substates": ["pos_low", "pos_high"],
                "initial": "pos_low"
            }
        },
        "transitions": [
            {"from": "idle_low", "to": "pos_low", "condition": "buy[t]"},
            {"from": "pos_low", "to": "idle_low", "condition": "sell[t]"}
        ],
        "substate_outputs": ["idle_low", "idle_high", "pos_low", "pos_high"],
        "aggregate_output": "position_state"  # Optional
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hierarchy` | dict | Yes | None | Hierarchy definition: {superstate: {substates: [...], initial: ...}} |
| `transitions` | list | No | [] | List of transitions: [{from, to, condition}] |
| `substate_outputs` | list | Yes | [] | Output stream names for each substate |
| `aggregate_output` | str | No | None | Output stream name for aggregated superstate |

## State Transitions

### Substate FSM Generation

For each substate, generate FSM logic:

```
substate[t] = (
    (condition1 ? target1 : 
     (condition2 ? target2 : 
      substate[t-1]))
)
```

**Logic:**
- Check transition conditions
- Transition to target if condition matches
- Maintain current state otherwise

### Aggregation

```
superstate[t] = substate1[t] | substate2[t] | ...
```

**Logic:**
- Superstate active if any substate active

## Generated Logic

The pattern generates Tau Language code:

```tau
# Substate: idle_low
(o0[t] = (buy[t] ? {2}:bv[2] : {0}:bv[2])) && (o0[0] = 1)

# Substate: idle_high
(o1[t] = o1[t-1]) && (o1[0] = 0)

# Substate: pos_low
(o2[t] = (sell[t] ? {0}:bv[2] : {2}:bv[2])) && (o2[0] = 0)

# Aggregate: position_state
(o3[t] = o0[t] | o1[t] | o2[t] | o3[t]) && (o3[0] = 0)
```

## Stream Requirements

### Input Streams

- **inputs** (sbf or bv[N]): Input signals for state transitions

### Output Streams

- **substate_outputs** (sbf or bv[N]): Output streams for each substate
- **aggregate_output** (sbf or bv[N], optional): Aggregated superstate output

## Example Usage

### Basic Decomposed FSM

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="decomposed_fsm_example",
    strategy="custom",
    streams=(
        StreamConfig(name="buy", stream_type="sbf"),
        StreamConfig(name="sell", stream_type="sbf"),
        StreamConfig(name="idle_low", stream_type="sbf", is_input=False),
        StreamConfig(name="idle_high", stream_type="sbf", is_input=False),
        StreamConfig(name="pos_low", stream_type="sbf", is_input=False),
        StreamConfig(name="pos_high", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="decomposed_fsm",
            inputs=("buy", "sell"),
            output="idle_low",
            params={
                "hierarchy": {
                    "IDLE": {
                        "substates": ["idle_low", "idle_high"],
                        "initial": "idle_low"
                    },
                    "POSITION": {
                        "substates": ["pos_low", "pos_high"],
                        "initial": "pos_low"
                    }
                },
                "transitions": [
                    {"from": "idle_low", "to": "pos_low", "condition": "buy[t]"},
                    {"from": "pos_low", "to": "idle_low", "condition": "sell[t]"}
                ],
                "substate_outputs": ["idle_low", "idle_high", "pos_low", "pos_high"]
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with History State

```python
# Decomposed FSM
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell"),
    output="idle_low",
    params={
        "hierarchy": {...},
        "transitions": [...]
    }
)

# History state for superstate
LogicBlock(
    pattern="history_state",
    inputs=("idle_low", "superstate_entry"),
    output="restored_idle_low",
    params={
        "substate_input": "idle_low",
        "superstate_entry": "superstate_entry"
    }
)
```

### Combined with State Aggregation

```python
# Decomposed FSM generates substates
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell"),
    output="idle_low",
    params={
        "hierarchy": {...},
        "substate_outputs": ["idle_low", "idle_high"]
    }
)

# Aggregate substates
LogicBlock(
    pattern="state_aggregation",
    inputs=("idle_low", "idle_high"),
    output="idle_state",
    params={"method": "majority"}
)
```

## Limitations

1. **Simple Decomposition**: Basic hierarchical decomposition
2. **No Nested Hierarchies**: Cannot handle deeply nested hierarchies
3. **Transition Complexity**: Complex transition conditions may be difficult
4. **State Encoding**: Uses simple encoding (one-hot or bitvector)

## Future Enhancements

1. **Deep Nesting**: Support deeply nested hierarchies
2. **Complex Transitions**: Better handling of complex transition conditions
3. **State Inheritance**: Inherit behavior from superstate
4. **Dynamic Decomposition**: Runtime decomposition

## Testing

See `tests/test_decomposed_fsm.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern (used internally)
- **History State**: Remember last substate
- **State Aggregation**: Combine substates into superstate
- **Supervisor-Worker**: Hierarchical coordination

## References

- [Hierarchical FSM Design](HIERARCHICAL_FSM_DESIGN.md)
- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)

