# UTXO State Machine Pattern

## Overview

The UTXO State Machine pattern implements Bitcoin's UTXO (Unspent Transaction Output) model as a state machine. It tracks which transaction outputs are unspent and validates transactions by checking that inputs exist in the UTXO set.

## Pattern Name

`utxo_state_machine`

## Description

This pattern models Bitcoin's UTXO set as a state machine:
- **UTXO Set**: Bitvector where each bit represents a UTXO (unspent transaction output)
- **Transaction Validation**: Inputs must exist in UTXO set
- **UTXO Set Update**: Remove spent outputs, add new outputs

**Key Operations:**
1. **Spend UTXO**: Remove UTXO from set when used as input
2. **Create UTXO**: Add new UTXO to set from transaction outputs
3. **Validate Transaction**: Check all inputs exist in UTXO set

## Use Cases

- **Bitcoin Protocol Implementation**: Core UTXO tracking
- **Cryptocurrency Wallets**: Track unspent outputs
- **Transaction Validation**: Verify transaction inputs
- **Blockchain State Management**: Maintain UTXO set state

## Schema Definition

```python
LogicBlock(
    pattern="utxo_state_machine",
    inputs=("tx_inputs", "tx_outputs"),
    output="utxo_set",
    params={
        "tx_inputs": "tx_inputs",  # Bitvector: which UTXOs are being spent
        "tx_outputs": "tx_outputs",  # Bitvector: new UTXOs being created
        "tx_valid": "tx_valid",  # Optional: external validation (signatures, etc.)
        "utxo_set_output": "utxo_set",  # Output: current UTXO set
        "tx_valid_output": "tx_valid_output",  # Optional: transaction validation result
        "initial_utxo_set": 0  # Initial UTXO set state
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tx_inputs` | str | Yes | First input | Input stream name for transaction inputs (UTXOs being spent) |
| `tx_outputs` | str | No | Second input | Input stream name for transaction outputs (new UTXOs) |
| `tx_valid` | str | No | None | External validation signal (signatures, etc.) |
| `utxo_set_output` | str | No | Same as `output` | Output stream name for UTXO set |
| `tx_valid_output` | str | No | None | Output stream name for transaction validation |
| `initial_utxo_set` | int | No | 0 | Initial UTXO set state |

## State Transitions

### UTXO Set Update

```
UTXO_Set[t] = (tx_valid ? 
    ((UTXO_Set[t-1] & !tx_inputs) | tx_outputs) : 
    UTXO_Set[t-1])
```

**Logic:**
1. Remove spent outputs: `UTXO_Set[t-1] & !tx_inputs` (clear bits for spent UTXOs)
2. Add new outputs: `tx_outputs` (set bits for new UTXOs)
3. Only if valid: `tx_valid` gates the update

### Transaction Validation

```
tx_valid = (tx_inputs & UTXO_Set[t-1]) = tx_inputs
```

**Logic:**
- All inputs must exist in UTXO set
- If external validation provided, use that instead

## Generated Logic

The pattern generates Tau Language code:

```tau
# UTXO set update
(o0[t] = (i2[t] ? ((o0[t-1] & i0[t]') | i1[t]) : o0[t-1])) && 
 (o0[0] = {0}:bv[32])

# Transaction validation (optional)
(o1[t] = ((i0[t] & o0[t-1]) = i0[t])) && (o1[0] = 0)
```

## Stream Requirements

### Input Streams

- **tx_inputs** (bv[N]): Bitvector indicating which UTXOs are being spent
- **tx_outputs** (bv[N], optional): Bitvector indicating new UTXOs being created
- **tx_valid** (sbf, optional): External validation signal (signatures, etc.)

### Output Streams

- **utxo_set** (bv[N]): Current UTXO set state
- **tx_valid_output** (sbf, optional): Transaction validation result

## Example Usage

### Basic UTXO State Machine

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="utxo_tracker",
    strategy="custom",
    streams=(
        StreamConfig(name="tx_inputs", stream_type="bv", width=32),
        StreamConfig(name="tx_outputs", stream_type="bv", width=32),
        StreamConfig(name="utxo_set", stream_type="bv", width=32, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="utxo_state_machine",
            inputs=("tx_inputs", "tx_outputs"),
            output="utxo_set",
            params={
                "tx_inputs": "tx_inputs",
                "tx_outputs": "tx_outputs",
                "initial_utxo_set": 0
            }
        ),
    ),
)
```

### UTXO State Machine with Validation

```python
schema = AgentSchema(
    name="utxo_validator",
    strategy="custom",
    streams=(
        StreamConfig(name="tx_inputs", stream_type="bv", width=32),
        StreamConfig(name="tx_outputs", stream_type="bv", width=32),
        StreamConfig(name="tx_valid", stream_type="sbf"),  # External validation
        StreamConfig(name="utxo_set", stream_type="bv", width=32, is_input=False),
        StreamConfig(name="tx_valid_output", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="utxo_state_machine",
            inputs=("tx_inputs", "tx_outputs"),
            output="utxo_set",
            params={
                "tx_inputs": "tx_inputs",
                "tx_outputs": "tx_outputs",
                "tx_valid": "tx_valid",
                "tx_valid_output": "tx_valid_output"
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Multi-Bit Counter for UTXO Count

```python
# UTXO state machine
LogicBlock(
    pattern="utxo_state_machine",
    inputs=("tx_inputs", "tx_outputs"),
    output="utxo_set"
)

# Count UTXOs (popcount)
LogicBlock(
    pattern="custom",
    inputs=("utxo_set",),
    output="utxo_count",
    params={"expression": "popcount(utxo_set[t])"}  # Would need popcount helper
)
```

### Combined with Time Lock for Time-Locked UTXOs

```python
# UTXO state machine
LogicBlock(
    pattern="utxo_state_machine",
    inputs=("tx_inputs", "tx_outputs"),
    output="utxo_set"
)

# Time lock for UTXO spending
LogicBlock(
    pattern="time_lock",
    inputs=("lock_start", "lock_duration", "current_time"),
    output="utxo_lock_active"
)

# Gate UTXO spending on time lock
LogicBlock(
    pattern="custom",
    inputs=("tx_inputs", "utxo_lock_active"),
    output="safe_tx_inputs",
    params={"expression": "tx_inputs[t] & utxo_lock_active[t]'"}
)
```

## Limitations

1. **Simplified UTXO Model**: Uses bitvector representation (each bit = one UTXO)
2. **No Value Tracking**: Does not track UTXO values (only existence)
3. **No Script Validation**: Script execution must be external
4. **No Cryptographic Validation**: Signature verification must be external
5. **Limited UTXO Count**: Bitvector width limits maximum UTXOs (32 bits = 32 UTXOs)

## Future Enhancements

1. **Value Tracking**: Track UTXO values using additional bitvectors
2. **Script Integration**: Integrate with Script Execution pattern
3. **Multi-UTXO Support**: Support multiple UTXO sets (different asset types)
4. **UTXO Indexing**: Efficient UTXO lookup and indexing
5. **Transaction Fee Calculation**: Calculate fees from input/output values

## Testing

See `tests/test_utxo_state_machine.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern (used internally)
- **Time Lock**: Time-locked UTXO spending
- **Script Execution**: Bitcoin Script validation (planned)
- **Accumulator**: Value accumulation for UTXO values

## References

- [Bitcoin Protocol](https://en.bitcoin.it/wiki/Protocol_documentation)
- [UTXO Model](https://en.bitcoin.it/wiki/UTXO)
- [TCP/IP & Bitcoin Analysis](TCPIP_BITCOIN_ANALYSIS.md)
- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)

