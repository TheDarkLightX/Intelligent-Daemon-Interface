# Hierarchical FSM Analysis & Pattern Landscape

## Executive Summary

**Question:** Can we do hierarchical FSM agent structures? Can we design more patterns?

**Answer:** 
- ✅ **YES** - We CAN create hierarchical FSMs via pattern composition
- ✅ **YES** - We CAN design explicit hierarchical patterns for ease of use
- 📊 **Pattern Landscape:** 4-level taxonomy (Atomic → Composite → Hierarchical → Domain-Specific)

## Current Capabilities

### ✅ Hierarchical FSMs via Composition

**Verified Working:**
```python
# Supervisor FSM controls worker FSM
LogicBlock(pattern="fsm", inputs=("mode", "status"), output="supervisor")
LogicBlock(pattern="fsm", inputs=("supervisor", "signal"), output="worker")
```

**Result:** Worker FSM depends on supervisor state - hierarchy achieved!

**Limitations:**
- Manual composition (must wire outputs to inputs)
- No explicit parent-child relationships
- No automatic enable/disable of child FSMs
- No state decomposition templates

### ✅ State Aggregation via Voting

**Working:**
```python
# Multiple FSMs vote
LogicBlock(pattern="majority", inputs=("fsm1", "fsm2", "fsm3"), output="aggregate")
```

**Result:** Can aggregate multiple FSM states via voting patterns.

### ✅ State Decomposition via Custom Expressions

**Working:**
```python
# Sub-states
LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate1")
LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate2")
# Aggregate
LogicBlock(pattern="custom", inputs=("substate1", "substate2"),
           output="position", params={"expression": "substate1[t] | substate2[t]"})
```

**Result:** Can decompose states manually, but requires custom logic.

## Pattern Landscape

### Level 1: Atomic Patterns (9 patterns) ✅

**Current:**
1. FSM - Basic state machine
2. Counter - Toggle counter
3. Accumulator - Sum values
4. Passthrough - Direct mapping
5. Vote - OR-based voting
6. Majority - N-of-M voting
7. Unanimous - All-agree consensus
8. Custom - Boolean expressions
9. Quorum - Minimum votes

**Characteristics:**
- Single responsibility
- Can compose but don't explicitly support hierarchy

### Level 2: Composite Patterns ✅

**Current:** All Level 1 patterns can compose
- FSM → Majority → FSM
- FSM → Custom → FSM
- Counter → FSM

### Level 3: Hierarchical Patterns ❌ MISSING

**Needed:**
1. **Supervisor-Worker** - Parent FSM coordinates child FSMs
2. **Decomposed FSM** - Break single FSM into sub-states
3. **Orthogonal Regions** - Parallel independent FSMs
4. **State Aggregation** - Combine FSMs into superstate
5. **History State** - Remember last substate

**Impact:** High - Enables true hierarchical agent architectures

### Level 4: Domain-Specific Patterns ❌ MISSING

**Needed:**
1. **Entry-Exit FSM** - Multi-phase trade lifecycle
2. **Proposal FSM** - Governance proposal lifecycle
3. **Risk FSM** - Risk state machine
4. **Circuit Breaker FSM** - Safety circuit breaker

**Impact:** Medium - Specialized use cases

## Research Findings (Perplexity)

### Common Hierarchical FSM Patterns

1. **Composite State (Nested State)**
   - Superstate contains internal FSM
   - Example: TradingEnabled → Scanning, Entering, Managing, Exiting

2. **Behavioral Inheritance**
   - Superstate handles common behavior
   - Substates override only differences
   - Events bubble up to ancestors

3. **Orthogonal Regions**
   - Multiple independent FSMs in parallel
   - Example: Execution, Risk, Connectivity regions

4. **History State**
   - Remember last substate when re-entering
   - Resume instead of restart

5. **Supervisor-Worker**
   - Parent FSM coordinates workers
   - Mode changes propagate down
   - Status events propagate up

## Proposed New Patterns

### Priority 1: Supervisor-Worker Pattern

**Why:** Most common hierarchical pattern, enables multi-agent coordination

**Schema:**
```python
LogicBlock(
    pattern="supervisor_worker",
    supervisor_inputs=("global_mode",),
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

**Complexity:** Medium  
**Implementation:** 3 days  
**Impact:** High

### Priority 2: Decomposed FSM Pattern

**Why:** Enables state decomposition (idle_low/idle_high, pos_low/pos_high)

**Schema:**
```python
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell", "regime"),
    output="position",
    params={
        "hierarchy": {
            "IDLE": {"substates": ["idle_low", "idle_high"], "initial": "idle_low"},
            "POSITION": {"substates": ["pos_low", "pos_high"], "initial": "pos_low"}
        },
        "transitions": [
            {"from": "idle_low", "to": "pos_low", "condition": "buy & regime'"},
            {"from": "idle_high", "to": "pos_high", "condition": "buy & regime"}
        ]
    }
)
```

**Complexity:** High  
**Implementation:** 5 days  
**Impact:** High

### Priority 3: Orthogonal Regions Pattern

**Why:** Enables parallel FSMs (execution, risk, connectivity)

**Schema:**
```python
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("execution_signal", "risk_signal", "connectivity_signal"),
    outputs=("execution_state", "risk_state", "connectivity_state"),
    params={
        "regions": [
            {"name": "execution", "fsm": {"states": ["FLAT", "LONG"]}},
            {"name": "risk", "fsm": {"states": ["NORMAL", "WARNING", "CRITICAL"]}}
        ]
    }
)
```

**Complexity:** Medium  
**Implementation:** 3 days  
**Impact:** Medium

## Implementation Roadmap

### Phase 3: Hierarchical FSM Patterns (2-3 weeks)

**Week 1:**
- [ ] Supervisor-Worker Pattern (3 days)
- [ ] State Aggregation Pattern (2 days)

**Week 2:**
- [ ] Decomposed FSM Pattern (5 days)

**Week 3:**
- [ ] Orthogonal Regions Pattern (3 days)
- [ ] History State Pattern (2 days)

### Phase 4: Advanced Composition (1-2 weeks)
- [ ] Multi-Bit Counter Pattern (1 day)
- [ ] Streak Counter Pattern (2 days)
- [ ] Mode Switch Pattern (2 days)

### Phase 5: Domain-Specific (1-2 weeks)
- [ ] Entry-Exit FSM Pattern (2 days)
- [ ] Proposal FSM Pattern (1 day)
- [ ] Risk FSM Pattern (1 day)

## Verification Strategy

For each new pattern:
1. ✅ Generate spec with pattern
2. ✅ Create input files with all state combinations
3. ✅ Execute through Tau binary
4. ✅ Verify outputs match expected transitions
5. ✅ Test edge cases

## Conclusion

**Current State:**
- ✅ Can create hierarchical FSMs via composition
- ✅ 9 patterns available
- ❌ No explicit hierarchical patterns
- ❌ Manual composition is error-prone

**With New Patterns:**
- ✅ Explicit hierarchical FSM support
- ✅ State decomposition and aggregation
- ✅ Multi-level coordination
- ✅ Domain-specific templates

**Recommendation:** Implement Phase 3 (Hierarchical FSM Patterns) to enable true hierarchical agent architectures.

**Next Steps:**
1. Implement Supervisor-Worker Pattern (proof of concept)
2. Verify end-to-end with Tau binary
3. Update wizard GUI
4. Continue with remaining patterns

