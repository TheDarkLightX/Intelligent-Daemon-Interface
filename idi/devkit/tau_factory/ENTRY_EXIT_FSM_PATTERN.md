# Entry-Exit FSM Pattern

## Overview

The Entry-Exit FSM pattern implements a multi-phase trade lifecycle for trading agents. It tracks the complete trading workflow from pre-trade scanning through active trading to post-trade reconciliation.

## Pattern Name

`entry_exit_fsm`

## Description

This pattern models a trading agent's lifecycle as a finite state machine with three main phases:

1. **PRE_TRADE** (0): Agent is scanning markets, validating conditions, preparing to enter
2. **IN_TRADE** (1): Agent has an active position, managing the trade
3. **POST_TRADE** (2): Agent has exited, reconciling results, preparing for next trade

## Use Cases

- **Trading Agents**: Complete trade lifecycle management
- **Risk Management**: Phase-based risk controls
- **Performance Tracking**: Phase-specific metrics
- **State Machine Composition**: Building complex trading workflows

## Schema Definition

```python
LogicBlock(
    pattern="entry_exit_fsm",
    inputs=("entry_signal", "exit_signal", "stop_loss", "take_profit"),
    output="phase",
    params={
        "phases": ["PRE_TRADE", "IN_TRADE", "POST_TRADE"],
        "entry_signal": "entry_signal",
        "exit_signal": "exit_signal",
        "stop_loss": "stop_loss",  # Optional
        "take_profit": "take_profit",  # Optional
        "phase_output": "phase",  # Default: same as output
        "position_output": "position"  # Optional: boolean position indicator
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `phases` | List[str] | No | `["PRE_TRADE", "IN_TRADE", "POST_TRADE"]` | List of phase names |
| `entry_signal` | str | Yes | First input | Input signal name for entering trade |
| `exit_signal` | str | Yes | Second input | Input signal name for exiting trade |
| `stop_loss` | str | No | None | Input signal name for stop loss trigger |
| `take_profit` | str | No | None | Input signal name for take profit trigger |
| `phase_output` | str | No | Same as `output` | Output stream name for phase state |
| `position_output` | str | No | None | Output stream name for position boolean |

## State Transitions

### Phase Transitions

```
PRE_TRADE (0) ──[entry_signal]──> IN_TRADE (1)
IN_TRADE (1) ──[exit_signal | stop_loss | take_profit]──> POST_TRADE (2)
POST_TRADE (2) ──[!entry_signal]──> PRE_TRADE (0)
```

### Transition Logic

1. **PRE_TRADE → IN_TRADE**: Triggered when `entry_signal` is true
2. **IN_TRADE → POST_TRADE**: Triggered when any of:
   - `exit_signal` is true (normal exit)
   - `stop_loss` is true (stop loss triggered)
   - `take_profit` is true (take profit triggered)
3. **POST_TRADE → PRE_TRADE**: Triggered when `entry_signal` is false (reset condition)

## Generated Logic

The pattern generates Tau Language code that implements:

1. **Phase State Machine**: Tracks current phase as bitvector
2. **Position Indicator**: Optional boolean output indicating active position (IN_TRADE phase)

### Example Generated Code

```tau
# Phase FSM
(o0[t] = ((o0[t-1] = {0}:bv[2]) & i0[t] ? {1}:bv[2] : 
          ((o0[t-1] = {1}:bv[2]) & (i1[t] | i2[t] | i3[t]) ? {2}:bv[2] : 
           ((o0[t-1] = {2}:bv[2]) & i0[t]' ? {0}:bv[2] : o0[t-1]))) && 
 (o0[0] = {0}:bv[2])

# Position indicator (optional)
(o1[t] = (o0[t] = {1}:bv[2])) && (o1[0] = 0)
```

## Stream Requirements

### Input Streams

- **entry_signal** (sbf): Signal to enter trade
- **exit_signal** (sbf): Signal to exit trade
- **stop_loss** (sbf, optional): Stop loss trigger
- **take_profit** (sbf, optional): Take profit trigger

### Output Streams

- **phase** (bv[2]): Current phase state (0=PRE_TRADE, 1=IN_TRADE, 2=POST_TRADE)
- **position** (sbf, optional): Boolean indicating active position (true when IN_TRADE)

## Example Usage

### Basic Entry-Exit FSM

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="trading_agent",
    strategy="momentum",
    streams=(
        StreamConfig(name="entry_signal", stream_type="sbf"),
        StreamConfig(name="exit_signal", stream_type="sbf"),
        StreamConfig(name="phase", stream_type="bv", width=2, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="entry_exit_fsm",
            inputs=("entry_signal", "exit_signal"),
            output="phase",
            params={
                "entry_signal": "entry_signal",
                "exit_signal": "exit_signal"
            }
        ),
    ),
)

spec = generate_tau_spec(schema)
```

### Advanced Entry-Exit FSM with Stop Loss and Take Profit

```python
schema = AgentSchema(
    name="advanced_trading_agent",
    strategy="regime_aware",
    streams=(
        StreamConfig(name="entry_signal", stream_type="sbf"),
        StreamConfig(name="exit_signal", stream_type="sbf"),
        StreamConfig(name="stop_loss", stream_type="sbf"),
        StreamConfig(name="take_profit", stream_type="sbf"),
        StreamConfig(name="phase", stream_type="bv", width=2, is_input=False),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="entry_exit_fsm",
            inputs=("entry_signal", "exit_signal", "stop_loss", "take_profit"),
            output="phase",
            params={
                "entry_signal": "entry_signal",
                "exit_signal": "exit_signal",
                "stop_loss": "stop_loss",
                "take_profit": "take_profit",
                "position_output": "position"
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Risk FSM

```python
# Entry-exit FSM for trade lifecycle
LogicBlock(pattern="entry_exit_fsm", inputs=("entry", "exit"), output="phase")

# Risk FSM for risk management
LogicBlock(pattern="risk_fsm", inputs=("warning", "critical", "normal"), output="risk_state")

# Gate entry on risk state
LogicBlock(
    pattern="custom",
    inputs=("entry", "risk_state"),
    output="safe_entry",
    params={"expression": "entry[t] & (risk_state[t] = {0}:bv[2])"}
)
```

### Combined with Multi-Bit Counter

```python
# Entry-exit FSM for trade lifecycle
LogicBlock(pattern="entry_exit_fsm", inputs=("entry", "exit"), output="phase")

# Counter for trade duration
LogicBlock(
    pattern="multi_bit_counter",
    inputs=("position",),
    output="trade_duration",
    params={"width": 4}
)
```

## Limitations

1. **Fixed Phase Count**: Currently supports 3 phases (PRE_TRADE, IN_TRADE, POST_TRADE)
2. **No Sub-States**: Phase sub-states (e.g., SCANNING, VALIDATING within PRE_TRADE) not yet supported
3. **Simple Reset**: POST_TRADE → PRE_TRADE transition is based on entry signal negation

## Future Enhancements

1. **Sub-State Support**: Allow sub-states within each phase
2. **Custom Phase Names**: Support user-defined phase names
3. **Phase-Specific Actions**: Trigger actions on phase transitions
4. **History Tracking**: Remember last phase for recovery

## Testing

See `tests/test_entry_exit_fsm.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern
- **Risk FSM**: Risk state management
- **Proposal FSM**: Governance lifecycle
- **Supervisor-Worker**: Hierarchical FSM coordination

## References

- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [Tau Language Capabilities](../../specification/TAU_LANGUAGE_CAPABILITIES.md)

