# Enhancement Proposal: Ensemble & DAO Support

## Executive Summary

**Current Status:** The Tau Agent Factory can generate simple agents but **cannot generate ensemble voting agents or DAO governance specs**.

**Proposed Solution:** Add 6 new patterns to enable ensemble and basic DAO support.

## Proposed Patterns

### 1. Majority Voting Pattern
**Purpose:** Enable N-of-M majority voting (e.g., 2-of-3, 3-of-5)

**Syntax:**
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
# 2-of-3: (a & b) | (a & c) | (b & c)
(o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t]))
```

**Implementation:** Generate all combinations of `threshold` inputs from `total` inputs, OR them together.

### 2. Unanimous Pattern
**Purpose:** Detect when all agents agree

**Syntax:**
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

### 3. Custom Boolean Expression Pattern
**Purpose:** Allow arbitrary boolean expressions

**Syntax:**
```python
LogicBlock(
    pattern="custom",
    inputs=("a", "b", "c"),
    output="result",
    params={"expression": "(a & b) | (c & a')"}
)
```

**Generated Logic:**
```tau
(o0[t] = (i0[t] & i1[t]) | (i2[t] & i0[t]'))
```

**Note:** Requires parsing and validation of expression string.

### 4. Quorum Pattern
**Purpose:** Check if minimum votes are present

**Syntax:**
```python
LogicBlock(
    pattern="quorum",
    inputs=("vote1", "vote2", "vote3", "vote4", "vote5"),
    output="quorum_met",
    params={"threshold": 3}  # 3-of-5 quorum
)
```

**Generated Logic:**
```tau
# Count votes using majority pattern
# For 3-of-5: Generate all 3-element combinations
(o0[t] = (i0[t] & i1[t] & i2[t]) | (i0[t] & i1[t] & i3[t]) | ...)
```

**Implementation:** Use majority pattern internally with threshold.

### 5. Vote Counter Pattern
**Purpose:** Count number of votes (for quorum/weighted voting)

**Syntax:**
```python
LogicBlock(
    pattern="vote_counter",
    inputs=("vote1", "vote2", "vote3", "vote4", "vote5"),
    output="vote_count",
    params={"width": 3}  # 3-bit counter (0-7 votes)
)
```

**Generated Logic:**
```tau
# Multi-bit counter counting votes
# Requires bit manipulation to count 1s
```

**Implementation:** Complex - requires bit-level counting logic.

### 6. Guarded FSM Pattern
**Purpose:** Multi-state FSM with guard conditions

**Syntax:**
```python
LogicBlock(
    pattern="guarded_fsm",
    inputs=("action", "proposal_id", "amount"),
    output="state",
    params={
        "states": ["idle", "locked", "voting"],
        "transitions": [
            {"from": "idle", "to": "locked", "guard": "amount > 0"},
            {"from": "locked", "to": "voting", "guard": "proposal_id != 0"},
        ]
    }
)
```

**Generated Logic:**
```tau
# State encoding and transition logic with guards
# Requires comparisons (via solve blocks or external)
```

**Implementation:** Complex - requires state encoding and guard evaluation.

## Implementation Priority

### Phase 1: Boolean Logic (Easy - 1-2 days)
1. ✅ Majority pattern
2. ✅ Unanimous pattern
3. ✅ Custom expression pattern

**Impact:** Enables ensemble voting agents

### Phase 2: Vote Counting (Medium - 3-5 days)
4. ✅ Quorum pattern (uses majority internally)
5. ⚠️ Vote counter pattern (complex bit manipulation)

**Impact:** Enables basic DAO quorum checking

### Phase 3: Advanced DAO (Hard - 1-2 weeks)
6. ⚠️ Guarded FSM pattern
7. ⚠️ Time-lock pattern
8. ⚠️ Vote-escrow power (requires arithmetic)

**Impact:** Enables full DAO governance specs

## Example: Ensemble Agent After Enhancement

```python
schema = AgentSchema(
    name="ensemble_agent",
    strategy="custom",
    streams=(
        StreamConfig(name="agent1_buy", stream_type="sbf"),
        StreamConfig(name="agent2_buy", stream_type="sbf"),
        StreamConfig(name="agent3_buy", stream_type="sbf"),
        StreamConfig(name="majority_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="unanimous_buy", stream_type="sbf", is_input=False),
        StreamConfig(name="position", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="majority",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="majority_buy",
            params={"threshold": 2, "total": 3}
        ),
        LogicBlock(
            pattern="unanimous",
            inputs=("agent1_buy", "agent2_buy", "agent3_buy"),
            output="unanimous_buy"
        ),
        LogicBlock(
            pattern="fsm",
            inputs=("majority_buy", "unanimous_buy"),
            output="position"
        ),
    ),
)
```

**Generated Spec:**
```tau
# Majority: 2-of-3
(o0[t] = (i0[t] & i1[t]) | (i0[t] & i2[t]) | (i1[t] & i2[t])) &&
# Unanimous: all agree
(o1[t] = i0[t] & i1[t] & i2[t]) &&
# Position FSM
(o2[t] = o0[t] | (o2[t-1] & o1[t]')) && (o2[0] = 0)
```

## Example: DAO Quorum After Enhancement

```python
schema = AgentSchema(
    name="dao_proposal",
    strategy="custom",
    streams=(
        StreamConfig(name="vote1", stream_type="sbf"),
        StreamConfig(name="vote2", stream_type="sbf"),
        StreamConfig(name="vote3", stream_type="sbf"),
        StreamConfig(name="vote4", stream_type="sbf"),
        StreamConfig(name="vote5", stream_type="sbf"),
        StreamConfig(name="quorum_met", stream_type="sbf", is_input=False),
        StreamConfig(name="proposal_passed", stream_type="sbf", is_input=False),
    ),
    logic_blocks=(
        LogicBlock(
            pattern="quorum",
            inputs=("vote1", "vote2", "vote3", "vote4", "vote5"),
            output="quorum_met",
            params={"threshold": 3}  # 3-of-5 quorum
        ),
        LogicBlock(
            pattern="passthrough",
            inputs=("quorum_met",),
            output="proposal_passed"
        ),
    ),
)
```

## Complexity Assessment

| Pattern | Complexity | Implementation Time | Impact |
|---------|-----------|-------------------|--------|
| Majority | Low | 1 day | High (ensembles) |
| Unanimous | Low | 0.5 day | Medium |
| Custom | Medium | 2 days | High (flexibility) |
| Quorum | Low | 0.5 day | High (DAOs) |
| Vote Counter | High | 3-5 days | Medium |
| Guarded FSM | High | 1-2 weeks | High (DAOs) |

## Recommendation

**Immediate Action:** Implement Phase 1 (Majority, Unanimous, Custom) to enable ensemble support.

**Next Steps:** Add Quorum pattern for basic DAO support.

**Future:** Consider Guarded FSM and arithmetic patterns for full DAO support.

