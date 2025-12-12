# Script Execution Pattern

## Overview

The Script Execution pattern implements a simplified Bitcoin Script execution engine (stack-based VM). It handles basic opcodes and can be extended with external predicates for cryptographic operations.

## Pattern Name

`script_execution`

## Description

This pattern implements a simplified Bitcoin Script VM:
- **Stack Operations**: Basic stack manipulation (OP_DUP, etc.)
- **Opcodes**: Configurable opcode set
- **External Predicates**: Cryptographic operations handled externally
- **Execution Result**: Outputs execution result

**Key Operations:**
1. **Parse Script**: Read opcodes from script input
2. **Execute Opcodes**: Process opcodes on stack
3. **Handle Crypto**: Use external predicates for cryptographic operations
4. **Output Result**: Return execution result

## Use Cases

- **Bitcoin Script Validation**: Validate Bitcoin transaction scripts
- **Smart Contracts**: Execute simple smart contract logic
- **Conditional Logic**: Stack-based conditional execution
- **Cryptographic Verification**: Verify signatures and hashes (external)

## Schema Definition

```python
LogicBlock(
    pattern="script_execution",
    inputs=("script", "stack"),
    output="execution_result",
    params={
        "script_input": "script",  # Script opcodes
        "stack_input": "stack",  # Stack state
        "opcodes": {
            "OP_DUP": "duplicate_top",
            "OP_HASH160": "external_hash",
            "OP_CHECKSIG": "external_sig"
        },
        "initial_value": 0
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `script_input` | str | Yes | First input | Input stream name for script opcodes |
| `stack_input` | str | No | Second input | Input stream name for stack state |
| `opcodes` | dict | No | {} | Opcode definitions and handlers |
| `initial_value` | int | No | 0 | Initial execution result |

## Opcodes

### Basic Opcodes

- **OP_DUP**: Duplicate top stack item
- **OP_HASH160**: Hash top stack item (external)
- **OP_EQUALVERIFY**: Verify equality
- **OP_CHECKSIG**: Verify signature (external)

### External Predicates

Cryptographic operations must be handled externally:
- **SHA-256**: External hash computation
- **ECDSA**: External signature verification
- **RIPEMD160**: External hash computation

## Generated Logic

The pattern generates simplified Tau Language code:

```tau
# Script execution (simplified)
(o0[t] = (i0[t] = {0}:bv[32] ? i1[t] : 
          (i0[t] = {1}:bv[32] ? i1[t] : i1[t]))) && 
 (o0[0] = {0}:bv[32])
```

**Note:** Full Bitcoin Script execution is very complex. This is a simplified version that demonstrates the pattern structure.

## Stream Requirements

### Input Streams

- **script** (bv[N]): Script opcodes
- **stack** (bv[N], optional): Stack state

### Output Streams

- **execution_result** (bv[N]): Script execution result

## Example Usage

### Basic Script Execution

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="script_execution_example",
    strategy="custom",
    streams=(
        StreamConfig(name="script", stream_type="bv", width=8),
        StreamConfig(name="stack", stream_type="bv", width=32),
        StreamConfig(name="execution_result", stream_type="bv", width=32, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="script_execution",
            inputs=("script", "stack"),
            output="execution_result",
            params={
                "script_input": "script",
                "stack_input": "stack",
                "opcodes": {
                    "OP_DUP": "duplicate_top"
                }
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with UTXO State Machine

```python
# UTXO state machine
LogicBlock(
    pattern="utxo_state_machine",
    inputs=("tx_inputs", "tx_outputs"),
    output="utxo_set"
)

# Script execution for transaction validation
LogicBlock(
    pattern="script_execution",
    inputs=("script", "stack"),
    output="script_valid",
    params={
        "opcodes": {
            "OP_CHECKSIG": "external_sig"
        }
    }
)

# Gate UTXO update on script validation
LogicBlock(
    pattern="custom",
    inputs=("utxo_set", "script_valid"),
    output="validated_utxo_set",
    params={"expression": "utxo_set[t] & script_valid[t]"}
)
```

## Limitations

1. **Simplified Implementation**: Full Bitcoin Script is much more complex
2. **External Crypto**: Cryptographic operations must be external
3. **Limited Opcodes**: Only basic opcodes implemented
4. **No Flow Control**: Limited conditional logic support
5. **Stack Size**: Limited by bitvector width

## Future Enhancements

1. **Full Opcode Set**: Implement all Bitcoin Script opcodes
2. **Flow Control**: Add OP_IF, OP_ELSE, OP_ENDIF support
3. **Stack Management**: Better stack manipulation
4. **Alt Stack**: Support alternative stack
5. **Crypto Integration**: Better integration with external crypto predicates

## Testing

See `tests/test_script_execution.py` for comprehensive test coverage.

## Related Patterns

- **UTXO State Machine**: Bitcoin UTXO tracking
- **FSM**: Basic state machine pattern
- **Custom**: Custom logic expressions

## References

- [Bitcoin Script](https://en.bitcoin.it/wiki/Script)
- [TCP/IP & Bitcoin Analysis](TCPIP_BITCOIN_ANALYSIS.md)
- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)

