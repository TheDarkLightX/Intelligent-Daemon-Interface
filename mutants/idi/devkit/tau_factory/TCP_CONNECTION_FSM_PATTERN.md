# TCP Connection FSM Pattern

## Overview

The TCP Connection FSM pattern implements the TCP connection state machine as defined in RFC 793. It models the complete lifecycle of a TCP connection from establishment through data transfer to termination.

## Pattern Name

`tcp_connection_fsm`

## Description

This pattern implements the standard TCP connection state machine with 11 states:
- **CLOSED** (0): No connection
- **LISTEN** (1): Waiting for connection request
- **SYN_SENT** (2): Waiting for matching connection request after sending SYN
- **SYN_RECEIVED** (3): Waiting for confirmation after receiving SYN
- **ESTABLISHED** (4): Connection established, data transfer
- **FIN_WAIT_1** (5): Waiting for connection termination request or acknowledgment
- **FIN_WAIT_2** (6): Waiting for connection termination request
- **CLOSE_WAIT** (7): Waiting for connection termination request from local user
- **CLOSING** (8): Waiting for connection termination acknowledgment
- **TIME_WAIT** (9): Waiting for enough time to pass to ensure remote received acknowledgment
- **LAST_ACK** (10): Waiting for acknowledgment of connection termination request

## Use Cases

- **Network Protocol Implementation**: TCP/IP stack implementation
- **Connection Management**: Track connection state in network applications
- **Protocol Testing**: Test TCP state transitions
- **Network Simulation**: Simulate TCP connections

## Schema Definition

```python
LogicBlock(
    pattern="tcp_connection_fsm",
    inputs=("syn_flag", "ack_flag", "fin_flag", "rst_flag"),
    output="tcp_state",
    params={
        "syn_flag": "syn_flag",
        "ack_flag": "ack_flag",
        "fin_flag": "fin_flag",
        "rst_flag": "rst_flag",
        "timeout_signal": "timeout_signal",  # Optional
        "initial_state": 0  # CLOSED
    }
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `syn_flag` | str | Yes | First input | Input signal name for SYN flag |
| `ack_flag` | str | Yes | Second input | Input signal name for ACK flag |
| `fin_flag` | str | Yes | Third input | Input signal name for FIN flag |
| `rst_flag` | str | Yes | Fourth input | Input signal name for RST flag |
| `timeout_signal` | str | No | None | Input signal name for timeout (TIME_WAIT → CLOSED) |
| `initial_state` | int | No | 0 | Initial state (0=CLOSED) |

## State Transitions

### Connection Establishment

```
CLOSED (0) ──[SYN]──> SYN_SENT (2)
LISTEN (1) ──[SYN]──> SYN_RECEIVED (3)
SYN_SENT (2) ──[SYN+ACK]──> ESTABLISHED (4)
SYN_RECEIVED (3) ──[ACK]──> ESTABLISHED (4)
```

### Connection Termination

```
ESTABLISHED (4) ──[FIN]──> FIN_WAIT_1 (5)
ESTABLISHED (4) ──[FIN]──> CLOSE_WAIT (7)
FIN_WAIT_1 (5) ──[ACK]──> FIN_WAIT_2 (6)
FIN_WAIT_1 (5) ──[FIN]──> CLOSING (8)
FIN_WAIT_2 (6) ──[FIN]──> TIME_WAIT (9)
CLOSE_WAIT (7) ──[FIN]──> LAST_ACK (10)
CLOSING (8) ──[ACK]──> TIME_WAIT (9)
TIME_WAIT (9) ──[timeout]──> CLOSED (0)
LAST_ACK (10) ──[ACK]──> CLOSED (0)
```

### Reset

```
Any State ──[RST]──> CLOSED (0)
```

## Generated Logic

The pattern generates Tau Language code that implements TCP state transitions:

```tau
# TCP Connection FSM
(o0[t] = (i3[t] ? {0}:bv[4] : 
          ((o0[t-1] = {0}:bv[4]) & i0[t] ? {2}:bv[4] : 
           ((o0[t-1] = {1}:bv[4]) & i0[t] ? {3}:bv[4] : 
            ((o0[t-1] = {2}:bv[4]) & i0[t] & i1[t] ? {4}:bv[4] : 
             ... : o0[t-1]))))) && 
 (o0[0] = {0}:bv[4])
```

## Stream Requirements

### Input Streams

- **syn_flag** (sbf): SYN flag signal
- **ack_flag** (sbf): ACK flag signal
- **fin_flag** (sbf): FIN flag signal
- **rst_flag** (sbf): RST flag signal
- **timeout_signal** (sbf, optional): Timeout signal for TIME_WAIT → CLOSED

### Output Stream

- **tcp_state** (bv[4]): Current TCP connection state (0-10)

## Example Usage

### Basic TCP Connection FSM

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec

schema = AgentSchema(
    name="tcp_connection",
    strategy="custom",
    streams=(
        StreamConfig(name="syn_flag", stream_type="sbf"),
        StreamConfig(name="ack_flag", stream_type="sbf"),
        StreamConfig(name="fin_flag", stream_type="sbf"),
        StreamConfig(name="rst_flag", stream_type="sbf"),
        StreamConfig(name="tcp_state", stream_type="bv", width=4, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="tcp_connection_fsm",
            inputs=("syn_flag", "ack_flag", "fin_flag", "rst_flag"),
            output="tcp_state",
            params={
                "syn_flag": "syn_flag",
                "ack_flag": "ack_flag",
                "fin_flag": "fin_flag",
                "rst_flag": "rst_flag"
            }
        ),
    ),
)
```

### TCP Connection FSM with Timeout

```python
schema = AgentSchema(
    name="tcp_connection_with_timeout",
    strategy="custom",
    streams=(
        StreamConfig(name="syn_flag", stream_type="sbf"),
        StreamConfig(name="ack_flag", stream_type="sbf"),
        StreamConfig(name="fin_flag", stream_type="sbf"),
        StreamConfig(name="rst_flag", stream_type="sbf"),
        StreamConfig(name="timeout_signal", stream_type="sbf"),
        StreamConfig(name="tcp_state", stream_type="bv", width=4, is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="tcp_connection_fsm",
            inputs=("syn_flag", "ack_flag", "fin_flag", "rst_flag"),
            output="tcp_state",
            params={
                "syn_flag": "syn_flag",
                "ack_flag": "ack_flag",
                "fin_flag": "fin_flag",
                "rst_flag": "rst_flag",
                "timeout_signal": "timeout_signal"
            }
        ),
    ),
)
```

## Integration with Other Patterns

### Combined with Multi-Bit Counter for Timeout

```python
# TCP connection FSM
LogicBlock(
    pattern="tcp_connection_fsm",
    inputs=("syn", "ack", "fin", "rst"),
    output="tcp_state",
    params={"timeout_signal": "timeout"}
)

# Timeout counter for TIME_WAIT state
LogicBlock(
    pattern="multi_bit_counter",
    inputs=("in_time_wait",),
    output="timeout_counter",
    params={"width": 8}
)

# Generate timeout signal when counter reaches threshold
LogicBlock(
    pattern="custom",
    inputs=("timeout_counter",),
    output="timeout",
    params={"expression": "timeout_counter[t] >= {60}:bv[8]"}
)
```

## Limitations

1. **Simplified Transitions**: Some edge cases and simultaneous flag combinations may not be fully handled
2. **No Sequence Numbers**: Does not track sequence numbers or window sizes
3. **No Data Transfer**: Focuses on connection state, not data transfer
4. **Simplified Timeout**: TIME_WAIT timeout uses external signal rather than internal timer

## Future Enhancements

1. **Sequence Number Tracking**: Add sequence number validation
2. **Window Size Management**: Track receive/send window sizes
3. **Internal Timers**: Built-in timeout handling for TIME_WAIT
4. **Simultaneous Open**: Handle simultaneous connection attempts
5. **Connection Pool**: Support multiple concurrent connections

## Testing

See `tests/test_tcp_connection_fsm.py` for comprehensive test coverage.

## Related Patterns

- **FSM**: Basic state machine pattern (used internally)
- **Entry-Exit FSM**: Similar lifecycle pattern
- **Multi-Bit Counter**: For timeout tracking
- **Risk FSM**: Similar state management pattern

## References

- [RFC 793 - Transmission Control Protocol](https://tools.ietf.org/html/rfc793)
- [TCP/IP Analysis](TCPIP_BITCOIN_ANALYSIS.md)
- [Pattern Landscape](PATTERN_LANDSCAPE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)

