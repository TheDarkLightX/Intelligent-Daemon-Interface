# Ensemble Voting Patterns

Phase 1 implementation complete! The Tau Agent Factory now supports ensemble voting patterns.

## Available Patterns

### 1. Majority Voting (`majority`)

N-of-M majority voting (e.g., 2-of-3, 3-of-5).

**Example: 2-of-3 Majority**
```python
LogicBlock(
    pattern="majority",
    inputs=("agent1", "agent2", "agent3"),
    output="majority_buy",
    params={"threshold": 2, "total": 3}  # 2-of-3
)
```

**Generated Logic:**
```tau
(o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t]))
```

**Parameters:**
- `threshold` (int): Minimum number of votes required (default: `len(inputs) // 2 + 1`)
- `total` (int): Total number of inputs to consider (default: `len(inputs)`)

### 2. Unanimous Consensus (`unanimous`)

All inputs must agree for output to be true.

**Example:**
```python
LogicBlock(
    pattern="unanimous",
    inputs=("agent1", "agent2", "agent3"),
    output="consensus"
)
```

**Generated Logic:**
```tau
(o0[t] = i0[t] & i1[t] & i2[t])
```

### 3. Custom Boolean Expression (`custom`)

Arbitrary boolean expression using stream names or indices.

**Example:**
```python
LogicBlock(
    pattern="custom",
    inputs=("a", "b", "c"),
    output="result",
    params={"expression": "(a[t] & b[t]) | (c[t] & a[t]')"}
)
```

**Expression Syntax:**
- Use stream names: `price_up[t] & volume_ok[t]`
- Use indices: `i0[t] & i1[t]`
- Use negation: `a[t]'` or `i0[t]'`
- Use parentheses for grouping

## Complete Example: Ensemble Trading Agent

```python
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock

schema = AgentSchema(
    name="ensemble_trading_agent",
    strategy="custom",
    streams=(
        # Inputs: Three independent trading agents
        StreamConfig(name="agent1_buy", stream_type="sbf"),
        StreamConfig(name="agent2_buy", stream_type="sbf"),
        StreamConfig(name="agent3_buy", stream_type="sbf"),
        StreamConfig(name="agent_sell", stream_type="sbf"),
        
        # Outputs: Voting results and position
        StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="unanimous_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        # 2-of-3 majority voting
        LogicBlock(
            pattern="majority",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="majority_buy",
            params={"threshold": 2, "total": 3}
        ),
        
        # Unanimous consensus (all agents agree)
        LogicBlock(
            pattern="unanimous",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="unanimous_buy"
        ),
        
        # Position FSM: buy on majority, sell on signal
        LogicBlock(
            pattern="fsm",
            inputs=("majority_buy", "agent_sell"),
            output="position"
        ),
    ),
)
```

**Generated Spec:**
```tau
# ensemble_trading_agent Agent (Auto-generated)
# Strategy: custom

i0:sbf = in file("inputs/agent1_buy.in").
i1:sbf = in file("inputs/agent2_buy.in").
i2:sbf = in file("inputs/agent3_buy.in").
i3:sbf = in file("inputs/agent_sell.in").

o0:sbf = out file("outputs/majority_buy.out").
o1:sbf = out file("outputs/unanimous_buy.out").
o2:sbf = out file("outputs/position.out").

defs
r (
    (o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t])) &&
    (o1[t] = i0[t] & i1[t] & i2[t]) &&
    (o2[t] = o0[t] | (o2[t-1] & o3[t]')) && (o2[0] = 0)
)
n
n
...
q
```

## Use Cases

### Ensemble Trading
- Multiple Q-learning agents vote on trades
- Majority voting reduces false positives
- Unanimous voting for high-confidence trades

### DAO Governance (Basic)
- Proposal voting with quorum
- Consensus detection
- Multi-signature requirements

### Redundancy & Safety
- Fault-tolerant systems
- Byzantine fault tolerance (with majority voting)
- Consensus protocols

## Limitations

**Current Support:**
- ✅ N-of-M majority voting
- ✅ Unanimous consensus
- ✅ Custom boolean expressions
- ✅ Outputs can be used as inputs to other blocks

**Not Yet Supported:**
- ❌ Weighted voting (different weights per agent)
- ❌ Quorum counting (requires arithmetic)
- ❌ Time-locks (requires time-based logic)
- ❌ Vote-escrow power (requires arithmetic)

See [ENSEMBLE_DAO_ANALYSIS.md](ENSEMBLE_DAO_ANALYSIS.md) for full analysis.

## Testing

All patterns are tested in `tests/test_ensemble_patterns.py`:

```bash
pytest idi/devkit/tau_factory/tests/test_ensemble_patterns.py -v
```

## Next Steps

**Phase 2:** Add quorum pattern (uses majority internally)
**Phase 3:** Add guarded FSM and time-lock patterns for full DAO support

See [ENHANCEMENT_PROPOSAL.md](ENHANCEMENT_PROPOSAL.md) for roadmap.

