# Hierarchical FSM Pattern Design

## Research Summary

Based on Perplexity research and codebase analysis, hierarchical FSMs require:

1. **State Decomposition** - Break complex states into sub-states
2. **State Aggregation** - Combine multiple FSMs into superstates
3. **Multi-Level Coordination** - Supervisor-worker relationships
4. **Orthogonal Regions** - Parallel independent FSMs
5. **History States** - Remember last substate

## Current Capabilities

### ✅ What Works Now

**Basic Hierarchy via Composition:**
```python
# Supervisor FSM
LogicBlock(pattern="fsm", inputs=("mode", "status"), output="supervisor")
# Worker FSM (enabled by supervisor)
LogicBlock(pattern="fsm", inputs=("supervisor", "signal"), output="worker")
```

**State Aggregation via Voting:**
```python
# Multiple FSMs vote
LogicBlock(pattern="majority", inputs=("fsm1", "fsm2", "fsm3"), output="aggregate")
```

**State Decomposition via Custom:**
```python
# Sub-states
LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate1")
LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate2")
# Aggregate
LogicBlock(pattern="custom", inputs=("substate1", "substate2"),
           output="position", params={"expression": "substate1[t] | substate2[t]"})
```

### ❌ What's Missing

1. **Explicit Parent-Child Relationships** - No automatic enable/disable
2. **State Decomposition Templates** - Must manually create each sub-state
3. **History States** - No memory of last substate
4. **Orthogonal Regions** - No explicit parallel FSM concept
5. **Behavioral Inheritance** - No superstate with common behavior

## Proposed Patterns

### Pattern 1: Supervisor-Worker FSM

**Purpose:** Parent FSM coordinates child FSMs

**Schema:**
```python
LogicBlock(
    pattern="supervisor_worker",
    supervisor_inputs=("global_mode", "worker_status"),
    worker_inputs=("worker1_signal", "worker2_signal"),
    outputs=("supervisor_state", "worker1_enable", "worker2_enable", 
             "worker1_state", "worker2_state"),
    params={
        "supervisor_states": ["IDLE", "ACTIVE", "PAUSED"],
        "worker_states": ["FLAT", "LONG"],
        "coordination": {
            "IDLE": {"workers": "disabled"},
            "ACTIVE": {"workers": "enabled"},
            "PAUSED": {"workers": "force_flat"}
        }
    }
)
```

**Generated Logic:**
```tau
# Supervisor FSM
(o0[t] = i0[t] | (o0[t-1] & i1[t]')) && (o0[0] = 0)
# Worker 1 enable: active when supervisor = ACTIVE
(o1[t] = o0[t])
# Worker 1 FSM (enabled by supervisor)
(o2[t] = o1[t] & i2[t] | (o2[t-1] & i3[t]')) && (o2[0] = 0)
# Worker 2 enable
(o3[t] = o0[t])
# Worker 2 FSM
(o4[t] = o3[t] & i4[t] | (o4[t-1] & i5[t]')) && (o4[0] = 0)
```

### Pattern 2: Decomposed FSM

**Purpose:** Break single FSM into sub-states with explicit hierarchy

**Schema:**
```python
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell", "regime"),
    output="position",
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
            {"from": "idle_low", "to": "pos_low", "condition": "buy & regime'"},
            {"from": "idle_high", "to": "pos_high", "condition": "buy & regime"},
            {"from": "pos_low", "to": "idle_low", "condition": "sell"},
            {"from": "pos_high", "to": "idle_high", "condition": "sell"}
        ]
    }
)
```

**Generated Logic:**
```tau
# Sub-state: idle_low
(o0[t] = (o0[t-1] & i2[t]' & i1[t]') | (o2[t-1] & i1[t]')) && (o0[0] = 1)
# Sub-state: idle_high
(o1[t] = (o1[t-1] & i2[t] & i1[t]') | (o3[t-1] & i1[t]')) && (o1[0] = 0)
# Sub-state: pos_low
(o2[t] = (o2[t-1] & i1[t]') | (o0[t-1] & i0[t] & i2[t]')) && (o2[0] = 0)
# Sub-state: pos_high
(o3[t] = (o3[t-1] & i1[t]') | (o1[t-1] & i0[t] & i2[t])) && (o3[0] = 0)
# Aggregate: position = any position state
(o4[t] = o2[t] | o3[t])
```

### Pattern 3: Orthogonal Regions

**Purpose:** Parallel independent FSMs (execution, risk, connectivity)

**Schema:**
```python
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("execution_signal", "risk_signal", "connectivity_signal"),
    outputs=("execution_state", "risk_state", "connectivity_state"),
    params={
        "regions": [
            {
                "name": "execution",
                "fsm": {"inputs": ["execution_signal"], "states": ["FLAT", "LONG"]}
            },
            {
                "name": "risk",
                "fsm": {"inputs": ["risk_signal"], "states": ["NORMAL", "WARNING", "CRITICAL"]}
            }
        ]
    }
)
```

**Generated Logic:**
```tau
# Region 1: Execution FSM
(o0[t] = i0[t] | (o0[t-1] & i1[t]')) && (o0[0] = 0)
# Region 2: Risk FSM (3-state)
(o1[t] = i2[t] | (o1[t-1] & i3[t]')) && (o1[0] = 0)
(o2[t] = o1[t-1] & i3[t] | (o2[t-1] & i4[t]')) && (o2[0] = 0)
```

## Implementation Plan

### Phase 3.1: Supervisor-Worker Pattern (3 days)
- [ ] Design schema for supervisor-worker relationships
- [ ] Implement generator logic
- [ ] Write unit tests
- [ ] Write verification tests
- [ ] Update wizard GUI

### Phase 3.2: Decomposed FSM Pattern (5 days)
- [ ] Design state hierarchy schema
- [ ] Implement sub-state generation
- [ ] Implement state aggregation
- [ ] Write comprehensive tests
- [ ] Update documentation

### Phase 3.3: Orthogonal Regions Pattern (3 days)
- [ ] Design parallel FSM schema
- [ ] Implement region generation
- [ ] Write tests
- [ ] Update wizard

## Verification Strategy

For each new pattern:
1. Generate spec with pattern
2. Create input files with all state combinations
3. Execute through Tau binary
4. Verify outputs match expected state transitions
5. Test edge cases (state conflicts, invalid transitions)

## Conclusion

**Current:** Can create hierarchical FSMs via composition, but it's manual and error-prone  
**With New Patterns:** Explicit, easy-to-use hierarchical FSM support  
**Impact:** Enables complex multi-level agent architectures

