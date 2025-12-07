# Tau Agent Factory - Pattern Landscape

> **"If patterns are lego blocks, what is the landscape?"**

## Executive Summary

**Current:** 9 patterns (FSM, Counter, Accumulator, Passthrough, Vote, Majority, Unanimous, Custom, Quorum)  
**Capability:** Can compose patterns to create hierarchical structures, but lacks explicit hierarchical patterns  
**Gap:** Need patterns for state decomposition, aggregation, and multi-level coordination

## Pattern Taxonomy

### Level 1: Atomic Patterns (Single Responsibility)

**Current:**
1. ✅ **FSM** - Basic state machine (buy/sell → position)
2. ✅ **Counter** - Toggle counter on event
3. ✅ **Accumulator** - Sum values over time
4. ✅ **Passthrough** - Direct input-to-output mapping

**Characteristics:**
- Single responsibility
- No internal state decomposition
- Can be composed but don't explicitly support hierarchy

### Level 2: Composite Patterns (Combine Atomic Patterns)

**Current:**
5. ✅ **Vote** - OR-based voting
6. ✅ **Majority** - N-of-M majority voting
7. ✅ **Unanimous** - All-agree consensus
8. ✅ **Quorum** - Minimum votes required
9. ✅ **Custom** - Arbitrary boolean expressions

**Characteristics:**
- Combine multiple inputs
- Stateless aggregation
- Enable coordination between patterns

### Level 3: Hierarchical Patterns (Multi-Level Coordination)

**Missing - Need to Design:**

#### 3.1 Supervisor-Worker Pattern
**Purpose:** Parent FSM coordinates child FSMs

**Example:**
```python
LogicBlock(
    pattern="supervisor_fsm",
    inputs=("global_mode", "worker_status"),
    outputs=("supervisor_state", "worker_enable"),
    params={
        "states": ["IDLE", "ACTIVE", "PAUSED"],
        "workers": ["worker1", "worker2", "worker3"],
        "coordination": "enable_on_active"
    }
)
```

**Generated Logic:**
```tau
# Supervisor FSM
(o0[t] = i0[t] | (o0[t-1] & i1[t]')) && (o0[0] = 0)
# Worker enable: active when supervisor = ACTIVE
(o1[t] = o0[t])
```

#### 3.2 State Decomposition Pattern
**Purpose:** Break single FSM into sub-states

**Example:**
```python
LogicBlock(
    pattern="decomposed_fsm",
    inputs=("buy", "sell", "regime"),
    output="position",
    params={
        "states": {
            "IDLE": ["idle_low", "idle_high"],
            "POSITION": ["pos_low", "pos_high"]
        },
        "transitions": {
            "idle_low": {"to": "pos_low", "on": "buy & regime"},
            "pos_low": {"to": "idle_low", "on": "sell"}
        }
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

#### 3.3 State Aggregation Pattern
**Purpose:** Combine multiple FSMs into superstate

**Example:**
```python
LogicBlock(
    pattern="aggregated_fsm",
    inputs=("fsm1_state", "fsm2_state", "fsm3_state"),
    output="superstate",
    params={
        "aggregation": "majority",  # or "unanimous", "any"
        "states": ["IDLE", "ACTIVE"]
    }
)
```

#### 3.4 History State Pattern
**Purpose:** Remember last substate when re-entering composite state

**Example:**
```python
LogicBlock(
    pattern="history_fsm",
    inputs=("enter", "exit", "substate_signal"),
    output="state",
    params={
        "remember": True,  # Resume last substate
        "substates": ["A", "B", "C"]
    }
)
```

#### 3.5 Orthogonal Regions Pattern
**Purpose:** Parallel independent FSMs (execution, risk, connectivity)

**Example:**
```python
LogicBlock(
    pattern="orthogonal_regions",
    inputs=("execution_signal", "risk_signal", "connectivity_signal"),
    outputs=("execution_state", "risk_state", "connectivity_state"),
    params={
        "regions": [
            {"name": "execution", "fsm": "position_fsm"},
            {"name": "risk", "fsm": "risk_fsm"},
            {"name": "connectivity", "fsm": "connectivity_fsm"}
        ]
    }
)
```

### Level 4: Domain-Specific Patterns

**Trading Domain:**
- **Entry-Exit FSM** - Multi-phase trade lifecycle
- **Risk FSM** - Risk state machine (Normal/Warning/Critical)
- **Regime FSM** - Market regime detection

**Governance Domain:**
- **Proposal FSM** - Proposal lifecycle (Draft/Voting/Executed)
- **Vote-Escrow FSM** - Lock/unlock/vote states
- **Delegation FSM** - Delegate/undelegate states

**Safety Domain:**
- **Circuit Breaker FSM** - Normal/Tripped/Reset states
- **Kill Switch FSM** - Active/Emergency/Recovery states
- **Health Monitor FSM** - Healthy/Degraded/Failed states

## Pattern Composition Matrix

| Pattern | Can Compose With | Enables |
|---------|------------------|---------|
| FSM | All | State machines |
| Counter | FSM, Custom | Timers, streaks |
| Accumulator | FSM, Custom | Accumulation |
| Passthrough | All | Signal routing |
| Vote | FSM, Custom | Coordination |
| Majority | FSM, Vote | Ensemble decisions |
| Unanimous | FSM, Vote | Consensus |
| Custom | All | Arbitrary logic |
| Quorum | FSM, Vote | DAO governance |

**Composition Examples:**
- **FSM → Majority → FSM**: Multiple FSMs vote, result feeds into another FSM
- **FSM → Custom → FSM**: FSM state transformed by custom logic, feeds into another FSM
- **Counter → FSM**: Timer counter gates FSM transitions

## Hierarchical FSM Capabilities

### ✅ What We CAN Do (Current)

1. **Basic Hierarchy via Composition**
   ```python
   # Supervisor FSM
   LogicBlock(pattern="fsm", inputs=("mode", "status"), output="supervisor_state")
   # Worker FSM (enabled by supervisor)
   LogicBlock(pattern="fsm", inputs=("supervisor_state", "signal"), output="worker_state")
   ```

2. **State Aggregation via Voting**
   ```python
   # Multiple FSMs vote
   LogicBlock(pattern="majority", inputs=("fsm1", "fsm2", "fsm3"), output="aggregate")
   ```

3. **State Decomposition via Intermediate Outputs**
   ```python
   # Sub-state 1
   LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate1")
   # Sub-state 2
   LogicBlock(pattern="fsm", inputs=("buy", "sell"), output="substate2")
   # Aggregate
   LogicBlock(pattern="custom", inputs=("substate1", "substate2"), 
              output="position", params={"expression": "substate1[t] | substate2[t]"})
   ```

### ❌ What We CANNOT Do (Missing Patterns)

1. **Explicit Parent-Child Relationships**
   - No way to declare "FSM A is parent of FSM B"
   - No automatic enable/disable of child FSMs

2. **State Decomposition Templates**
   - Must manually create each sub-state
   - No automatic aggregation of sub-states

3. **History States**
   - No memory of last substate
   - Always starts from initial state

4. **Orthogonal Regions**
   - Must manually coordinate parallel FSMs
   - No explicit region concept

5. **Behavioral Inheritance**
   - No superstate with common behavior
   - Must duplicate logic across substates

## Proposed New Patterns

### Priority 1: Hierarchical FSM Patterns

#### 1. Supervisor-Worker Pattern
**Complexity:** Medium  
**Impact:** High  
**Implementation:** 2-3 days

```python
LogicBlock(
    pattern="supervisor_worker",
    supervisor_inputs=("global_mode",),
    worker_inputs=("worker1_signal", "worker2_signal"),
    outputs=("supervisor_state", "worker1_enable", "worker2_enable", "worker1_state", "worker2_state"),
    params={
        "supervisor_states": ["IDLE", "ACTIVE", "PAUSED"],
        "worker_states": ["FLAT", "LONG", "SHORT"],
        "coordination": {
            "IDLE": {"workers": "disabled"},
            "ACTIVE": {"workers": "enabled"},
            "PAUSED": {"workers": "force_flat"}
        }
    }
)
```

#### 2. Decomposed FSM Pattern
**Complexity:** High  
**Impact:** High  
**Implementation:** 3-5 days

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

#### 3. Orthogonal Regions Pattern
**Complexity:** Medium  
**Impact:** Medium  
**Implementation:** 2-3 days

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
            },
            {
                "name": "connectivity",
                "fsm": {"inputs": ["connectivity_signal"], "states": ["OK", "DEGRADED", "DISCONNECTED"]}
            }
        ]
    }
)
```

### Priority 2: Advanced Composition Patterns

#### 4. Multi-Bit Counter Pattern
**Complexity:** Low  
**Impact:** Medium  
**Implementation:** 1 day

```python
LogicBlock(
    pattern="multi_bit_counter",
    inputs=("increment", "reset"),
    output="counter",
    params={"width": 3}  # 3-bit counter (0-7)
)
```

#### 5. Streak Counter Pattern
**Complexity:** Medium  
**Impact:** Medium  
**Implementation:** 1-2 days

```python
LogicBlock(
    pattern="streak_counter",
    inputs=("event", "reset"),
    output="streak_count",
    params={"width": 4, "reset_on": "opposite_event"}
)
```

#### 6. Mode Switch Pattern
**Complexity:** Medium  
**Impact:** High  
**Implementation:** 2 days

```python
LogicBlock(
    pattern="mode_switch",
    inputs=("aggressive_signal", "defensive_signal"),
    output="mode",
    params={
        "modes": ["AGGRESSIVE", "DEFENSIVE"],
        "transitions": {
            "AGGRESSIVE": {"to": "DEFENSIVE", "on": "defensive_signal"},
            "DEFENSIVE": {"to": "AGGRESSIVE", "on": "aggressive_signal"}
        }
    }
)
```

### Priority 3: Domain-Specific Patterns

#### 7. Entry-Exit FSM Pattern
**Complexity:** Medium  
**Impact:** High (Trading)  
**Implementation:** 2 days

```python
LogicBlock(
    pattern="entry_exit_fsm",
    inputs=("entry_signal", "exit_signal", "stop_loss", "take_profit"),
    outputs=("phase", "position"),
    params={
        "phases": ["PRE_TRADE", "IN_TRADE", "POST_TRADE"],
        "substates": {
            "PRE_TRADE": ["SCANNING", "VALIDATING", "READY"],
            "IN_TRADE": ["ENTERING", "MANAGING", "EXITING"],
            "POST_TRADE": ["RECONCILING", "REPORTING"]
        }
    }
)
```

#### 8. Proposal FSM Pattern
**Complexity:** Low  
**Impact:** High (Governance)  
**Implementation:** 1 day

```python
LogicBlock(
    pattern="proposal_fsm",
    inputs=("create", "vote", "execute", "cancel"),
    output="proposal_state",
    params={
        "states": ["DRAFT", "VOTING", "PASSED", "EXECUTED", "CANCELLED"],
        "transitions": {
            "DRAFT": {"to": "VOTING", "on": "vote"},
            "VOTING": {"to": "PASSED", "on": "quorum_met"},
            "PASSED": {"to": "EXECUTED", "on": "execute"}
        }
    }
)
```

## Pattern Landscape Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    PATTERN LANDSCAPE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LEVEL 1: ATOMIC PATTERNS                                  │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────────┐    │
│  │   FSM   │ │ Counter │ │Accumulator│ │Passthrough │    │
│  └─────────┘ └─────────┘ └──────────┘ └─────────────┘    │
│                                                             │
│  LEVEL 2: COMPOSITE PATTERNS                               │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐        │
│  │  Vote   │ │Majority │ │Unanimous │ │ Quorum  │        │
│  └─────────┘ └─────────┘ └──────────┘ └─────────┘        │
│  ┌─────────────────────────────────────────────────┐      │
│  │              Custom Expression                  │      │
│  └─────────────────────────────────────────────────┘      │
│                                                             │
│  LEVEL 3: HIERARCHICAL PATTERNS (MISSING)                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │Supervisor-   │ │Decomposed    │ │Orthogonal    │      │
│  │Worker        │ │FSM           │ │Regions       │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
│  ┌──────────────┐ ┌──────────────┐                        │
│  │History State │ │State         │                        │
│  │              │ │Aggregation   │                        │
│  └──────────────┘ └──────────────┘                        │
│                                                             │
│  LEVEL 4: DOMAIN-SPECIFIC PATTERNS                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │Entry-Exit    │ │Proposal FSM  │ │Risk FSM      │      │
│  │FSM           │ │              │ │              │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Roadmap

### Phase 3: Hierarchical FSM Patterns (2-3 weeks)
1. Supervisor-Worker Pattern (3 days)
2. Decomposed FSM Pattern (5 days)
3. Orthogonal Regions Pattern (3 days)
4. State Aggregation Pattern (2 days)
5. History State Pattern (2 days)

### Phase 4: Advanced Composition (1-2 weeks)
6. Multi-Bit Counter Pattern (1 day)
7. Streak Counter Pattern (2 days)
8. Mode Switch Pattern (2 days)

### Phase 5: Domain-Specific (1-2 weeks)
9. Entry-Exit FSM Pattern (2 days)
10. Proposal FSM Pattern (1 day)
11. Risk FSM Pattern (1 day)

## Conclusion

**Current State:**
- ✅ 9 patterns (atomic + composite)
- ✅ Can compose patterns for basic hierarchy
- ❌ No explicit hierarchical patterns
- ❌ No state decomposition templates
- ❌ No orthogonal regions support

**With New Patterns:**
- ✅ Full hierarchical FSM support
- ✅ State decomposition and aggregation
- ✅ Multi-level coordination
- ✅ Domain-specific templates

**Recommendation:** Implement Phase 3 (Hierarchical FSM Patterns) to enable true hierarchical agent structures.

