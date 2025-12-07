# Orthogonal Regions Pattern

## Overview

The Orthogonal Regions pattern implements parallel independent finite state machines that operate simultaneously. Each region maintains its own state independently, allowing complex systems to be decomposed into orthogonal concerns.

## Pattern Name

`orthogonal_regions`

## Description

This pattern models a system as multiple independent FSMs running in parallel. Each region:
- Operates independently of other regions
- Maintains its own state
- Responds to its own inputs
- Can be in different states simultaneously

Common use cases:
- **Execution Region**: Trading execution state (FLAT, LONG, SHORT)
- **Risk Region**: Risk management state (NORMAL, WARNING, CRITICAL)
- **Connectivity Region**: Network state (OK, DEGRADED, DISCONNECTED)

## Use Cases

- **Multi-Domain Agents**: Separate concerns (execution, risk, connectivity)
- **Parallel State Tracking**: Multiple independent state machines
- **Modular Design**: Decompose complex systems into orthogonal components
- **State Composition**: Combine independent regions for complex behavior

## Schema Definition

```python
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("execution_signal", "risk_signal", "connectivity_signal"),
    output="execution_state",  # First region output (required)
    params={
        "regions": [
            {
                "name": "execution",
                "inputs": ["execution_buy", "execution_sell"],
                "states": ["FLAT", "LONG"],
                "initial_state": 0
            },
            {
                "name": "risk",
                "inputs": ["risk_signal"],
                "states": ["NORMAL", "WARNING", "CRITICAL"],
                "initial_state": 0
            },
            {
                "name": "connectivity",
                "inputs": ["connectivity_ok", "connectivity_degraded"],
                "states": ["OK", "DEGRADED", "DISCONNECTED"],
                "initial_state": 0
            }
        ],
        "region_outputs": ["execution_state", "risk_state", "connectivity_state"]
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `regions` | List[Dict] | Yes | None | List of region definitions |
| `region_outputs` | List[str] | Yes | None | Output stream names for each region |

### Region Definition

Each region in the `regions` list is a dictionary with:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | str | No | Region name (for documentation) |
| `inputs` | List[str] | Yes | Input signal names for this region |
| `states` | List[str] | Yes | State names (2+ states) |
| `initial_state` | int | No | Initial state index (default: 0) |

### Input Handling

- **2+ inputs**: Treated as FSM with buy/sell (first=input, second=output)
- **1 input**: Treated as toggle FSM (switches between first two states)

## Generated Logic

The pattern generates independent FSM logic for each region:

### Multi-Input Region (FSM)

```tau
# Region 1: Execution
(o0[t] = i0[t] | (o0[t-1] & i1[t]')) && (o0[0] = {0}:bv[2])

# Region 2: Risk
(o1[t] = i2[t] ? {1}:bv[2] : {0}:bv[2]) && (o1[0] = {0}:bv[2])
```

### Single-Input Region (Toggle)

```tau
# Region with single input: toggle between states
(o0[t] = (i0[t] ? {1}:bv[2] : {0}:bv[2])) && (o0[0] = {0}:bv[2])
```

## Stream Requirements

### Input Streams

Each region requires its own input streams:
- **Region 1**: `execution_buy`, `execution_sell` (or single `execution_signal`)
- **Region 2**: `risk_signal` (or `risk_buy`, `risk_sell`)
- **Region N**: Region-specific inputs

### Output Streams

Each region produces its own output stream:
- **Region 1**: `execution_state` (bv[N])
- **Region 2**: `risk_state` (bv[N])
- **Region N**: Region-specific output

## Example Usage

### Basic Two-Region System

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="multi_region_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="execution_signal", stream_type="sbf"),
        StreamConfig(name="risk_signal", stream_type="sbf"),
        StreamConfig(name="execution_state", stream_type="bv", width=2, is_input=False),
        StreamConfig(name="risk_state", stream_type="bv", width=2, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="orthogonal_regions",
            inputs=("execution_signal", "risk_signal"),
            output="execution_state",
            params={
                "regions": [
                    {
                        "name": "execution",
                        "inputs": ["execution_signal"],
                        "states": ["FLAT", "LONG"],
                        "initial_state": 0
                    },
                    {
                        "name": "risk",
                        "inputs": ["risk_signal"],
                        "states": ["NORMAL", "WARNING"],
                        "initial_state": 0
                    }
                ],
                "region_outputs": ["execution_state", "risk_state"]
            }
        ),
    ),
)
```

### Three-Region System with FSM Inputs

```python
schema = AgentSchema(
    name="advanced_multi_region_agent",
    strategy="regime_aware",
    streams=(
        # Execution region inputs
        StreamConfig(name="execution_buy", stream_type="sbf"),
        StreamConfig(name="execution_sell", stream_type="sbf"),
        # Risk region inputs
        StreamConfig(name="risk_warning", stream_type="sbf"),
        StreamConfig(name="risk_critical", stream_type="sbf"),
        # Connectivity region inputs
        StreamConfig(name="connectivity_ok", stream_type="sbf"),
        StreamConfig(name="connectivity_degraded", stream_type="sbf"),
        # Outputs
        StreamConfig(name="execution_state", stream_type="bv", width=2, is_input=False),
        StreamConfig(name="risk_state", stream_type="bv", width=2, is_input=False),
        StreamConfig(name="connectivity_state", stream_type="bv", width=2, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="orthogonal_regions",
            inputs=("execution_buy", "execution_sell", "risk_warning", "risk_critical", 
                   "connectivity_ok", "connectivity_degraded"),
            output="execution_state",
            params={
                "regions": [
                    {
                        "name": "execution",
                        "inputs": ["execution_buy", "execution_sell"],
                        "states": ["FLAT", "LONG"],
                        "initial_state": 0
                    },
                    {
                        "name": "risk",
                        "inputs": ["risk_warning", "risk_critical"],
                        "states": ["NORMAL", "WARNING", "CRITICAL"],
                        "initial_state": 0
                    },
                    {
                        "name": "connectivity",
                        "inputs": ["connectivity_ok", "connectivity_degraded"],
                        "states": ["OK", "DEGRADED", "DISCONNECTED"],
                        "initial_state": 0
                    }
                ],
                "region_outputs": ["execution_state", "risk_state", "connectivity_state"]
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Custom Logic for Coordination

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

# Custom logic to coordinate regions
LogicBlock(
    pattern="custom",
    inputs=("execution_state", "risk_state"),
    output="safe_execution",
    params={
        "expression": "execution_state[t] & (risk_state[t] = {0}:bv[2])"
    }
)
```

### Combined with Supervisor-Worker

```python
# Orthogonal regions for worker states
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("worker1_signal", "worker2_signal"),
    output="worker1_state",
    params={
        "regions": [
            {"name": "worker1", "inputs": ["worker1_signal"], "states": ["IDLE", "ACTIVE"]},
            {"name": "worker2", "inputs": ["worker2_signal"], "states": ["IDLE", "ACTIVE"]}
        ],
        "region_outputs": ["worker1_state", "worker2_state"]
    }
)

# Supervisor coordinates workers
LogicBlock(
    pattern="supervisor_worker",
    inputs=("supervisor_mode",),
    output="supervisor_state",
    params={
        "supervisor_inputs": ["supervisor_mode"],
        "worker_inputs": ["worker1_state", "worker2_state"],
        "worker_enable_outputs": ["worker1_enable", "worker2_enable"],
        "worker_outputs": ["worker1_state", "worker2_state"]
    }
)
```

## Limitations

1. **No Inter-Region Communication**: Regions operate independently; no direct state dependencies
2. **Simple State Transitions**: Each region uses basic FSM logic (buy/sell or toggle)
3. **Fixed Region Count**: Number of regions must be specified at design time

## Future Enhancements

1. **Inter-Region Signals**: Allow regions to respond to other regions' states
2. **Complex State Transitions**: Support custom transition logic per region
3. **Dynamic Regions**: Add/remove regions at runtime
4. **Region Coordination**: Built-in coordination patterns

## Testing

See `tests/test_orthogonal_regions.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern (used internally)
- **Supervisor-Worker**: Hierarchical coordination
- **State Aggregation**: Combining region states
- **Custom**: Coordinating multiple regions

## References

- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Hierarchical FSM Design](HIERARCHICAL_FSM_DESIGN.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [Tau Language Capabilities](../../specification/TAU_LANGUAGE_CAPABILITIES.md)

