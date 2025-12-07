# History State Pattern

## Overview

The History State pattern implements memory of the last substate when returning to a superstate. When a superstate is entered, if history exists, the system restores to the last substate; otherwise, it uses the initial substate.

## Pattern Name

`history_state`

## Description

This pattern provides history state functionality for hierarchical FSMs:
- **Save History**: When exiting superstate, save current substate
- **Restore History**: When entering superstate, restore from history if exists
- **Initial State**: If no history exists, use initial substate

**Key Operations:**
1. **Enter Superstate**: Restore from history or use initial
2. **Track Substate**: Monitor current substate while in superstate
3. **Exit Superstate**: Save current substate to history

## Use Cases

- **Hierarchical FSMs**: Remember last substate when returning to superstate
- **State Persistence**: Maintain state across superstate transitions
- **Resume Behavior**: Continue from where left off
- **Nested State Machines**: Complex state hierarchies with memory

## Schema Definition

```python
LogicBlock(
    pattern="history_state",
    inputs=("substate", "superstate_entry"),
    output="restored_state",
    params={
        "substate_input": "substate",  # Current substate
        "superstate_entry": "superstate_entry",  # Signal entering superstate
        "superstate_exit": "superstate_exit",  # Optional: explicit exit signal
        "history_output": "history",  # Optional: external history storage
        "initial_substate": 0  # Initial substate when no history
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `substate_input` | str | Yes | First input | Input stream name for current substate |
| `superstate_entry` | str | Yes | Second input | Input stream name for superstate entry signal |
| `superstate_exit` | str | No | None | Input stream name for superstate exit signal |
| `history_output` | str | No | None | Output stream name for history storage (internal if not provided) |
| `initial_substate` | int | No | 0 | Initial substate when no history exists |

## State Transitions

### History State Logic

```
restored_state[t] = (
    superstate_exit ? save_to_history :
    (superstate_entry & history_exists ? history :
     (superstate_entry ? initial_substate :
      substate))
)
```

**Logic:**
1. **Exit**: Save current substate to history
2. **Enter with History**: Restore from history
3. **Enter without History**: Use initial substate
4. **Otherwise**: Use current substate

### History Storage

```
history[t] = (superstate_exit ? substate : history[t-1])
```

**Logic:**
- Save substate when exiting superstate
- Maintain history otherwise

## Generated Logic

The pattern generates Tau Language code:

```tau
# History state restoration
(o0[t] = (i1[t]' ? i0[t] : 
          (i1[t] & (o0[t-1] != {0}:bv[2]) ? o0[t-1] : 
           (i1[t] ? {0}:bv[2] : i0[t])))) && 
 (o0[0] = {0}:bv[2])
```

## Stream Requirements

### Input Streams

- **substate** (bv[N]): Current substate value
- **superstate_entry** (sbf): Signal indicating entry into superstate
- **superstate_exit** (sbf, optional): Signal indicating exit from superstate

### Output Streams

- **restored_state** (bv[N]): Restored state output (with history)
- **history** (bv[N], optional): History storage output

## Example Usage

### Basic History State

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="history_state_example",
    strategy="custom",
    streams=(
        StreamConfig(name="substate", stream_type="bv", width=2),
        StreamConfig(name="superstate_entry", stream_type="sbf"),
        StreamConfig(name="restored_state", stream_type="bv", width=2, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="history_state",
            inputs=("substate", "superstate_entry"),
            output="restored_state",
            params={
                "substate_input": "substate",
                "superstate_entry": "superstate_entry",
                "initial_substate": 0
            }
        ),
    ),
)
```

### History State with Explicit Exit

```python
schema = AgentSchema(
    name="history_state_with_exit",
    strategy="custom",
    streams=(
        StreamConfig(name="substate", stream_type="bv", width=2),
        StreamConfig(name="superstate_entry", stream_type="sbf"),
        StreamConfig(name="superstate_exit", stream_type="sbf"),
        StreamConfig(name="restored_state", stream_type="bv", width=2, is_input=False),
        StreamConfig(name="history", stream_type="bv", width=2, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="history_state",
            inputs=("substate", "superstate_entry"),
            output="restored_state",
            params={
                "substate_input": "substate",
                "superstate_entry": "superstate_entry",
                "superstate_exit": "superstate_exit",
                "history_output": "history",
                "initial_substate": 0
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Supervisor-Worker

```python
# Supervisor FSM
LogicBlock(
    pattern="fsm",
    inputs=("enable_worker", "disable_worker"),
    output="supervisor_state"
)

# Worker FSM with history
LogicBlock(
    pattern="history_state",
    inputs=("worker_state", "supervisor_state"),
    output="restored_worker_state",
    params={
        "substate_input": "worker_state",
        "superstate_entry": "supervisor_state",
        "initial_substate": 0
    }
)
```

### Combined with Decomposed FSM

```python
# Decomposed FSM with history
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell"),
    output="position_state",
    params={
        "hierarchy": {...},
        "history": True  # Enable history state
    }
)
```

## Limitations

1. **Single History**: Only remembers last substate (not full history)
2. **No Deep History**: Cannot remember nested substates
3. **State Width**: Limited by bitvector width
4. **Simple Logic**: Basic save/restore logic

## Future Enhancements

1. **Deep History**: Remember nested substate hierarchies
2. **Multiple Histories**: Track history for multiple superstates
3. **History Stack**: Stack-based history for nested states
4. **Selective History**: Choose which substates to remember

## Testing

See `tests/test_history_state.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern
- **Supervisor-Worker**: Hierarchical FSM coordination
- **Decomposed FSM**: State decomposition with history
- **State Aggregation**: Combining states (can use history)

## References

- [Hierarchical FSM Design](HIERARCHICAL_FSM_DESIGN.md)
- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)

