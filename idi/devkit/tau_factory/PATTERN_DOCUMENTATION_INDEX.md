# Pattern Documentation Index

Complete reference guide for all implemented patterns in the Tau Agent Factory.

> **NEW:** See [SPEC_FORMATS.md](SPEC_FORMATS.md) for pure vs human-readable spec examples.
> Human-readable specs require `tau --charvar off`.

> **VERIFIED:** See [EXHAUSTIVE_PATTERN_VERIFICATION.md](EXHAUSTIVE_PATTERN_VERIFICATION.md) for exhaustive testing results.
> 60 patterns verified with all input combinations against Tau binary.

> **NEW:** See [FORMAL_VERIFICATION_PATTERNS.md](FORMAL_VERIFICATION_PATTERNS.md) for LTL-based patterns with pure and human-readable forms.

> **TUTORIALS:** See [tutorials/README.md](tutorials/README.md) for beginner tutorials teaching 26 specifications with step-by-step explanations.

## Quick Reference - Original Patterns (26)

| Pattern | Status | Documentation | Use Case |
|---------|--------|---------------|----------|
| FSM | ✅ | [Basic Patterns](#basic-patterns) | Position tracking |
| Counter | ✅ | [Basic Patterns](#basic-patterns) | Event counting |
| Accumulator | ✅ | [Basic Patterns](#basic-patterns) | Value accumulation |
| Vote | ✅ | [Basic Patterns](#basic-patterns) | Simple voting |
| Passthrough | ✅ | [Basic Patterns](#basic-patterns) | Direct mapping |
| Majority | ✅ | [ENSEMBLE_PATTERNS.md](ENSEMBLE_PATTERNS.md) | N-of-M voting |
| Unanimous | ✅ | [ENSEMBLE_PATTERNS.md](ENSEMBLE_PATTERNS.md) | All-agree consensus |
| Custom | ✅ | [ENSEMBLE_PATTERNS.md](ENSEMBLE_PATTERNS.md) | Boolean expressions |
| Quorum | ✅ | [ENSEMBLE_PATTERNS.md](ENSEMBLE_PATTERNS.md) | Minimum votes |
| Supervisor-Worker | ✅ | [HIERARCHICAL_FSM_DESIGN.md](HIERARCHICAL_FSM_DESIGN.md) | Hierarchical FSMs |
| Weighted Vote | ✅ | [BITVECTOR_PATTERNS.md](BITVECTOR_PATTERNS.md) | Weighted voting |
| Time Lock | ✅ | [BITVECTOR_PATTERNS.md](BITVECTOR_PATTERNS.md) | Time-based locking |
| Hex Stake | ✅ | [HEX_SUMMARY.md](HEX_SUMMARY.md) | Time-lock staking |
| Multi-Bit Counter | ✅ | [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md) | Multi-bit timers |
| Streak Counter | ✅ | [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md) | Event streaks |
| Mode Switch | ✅ | [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md) | Mode switching |
| Proposal FSM | ✅ | [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md) | Governance |
| Risk FSM | ✅ | [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md) | Risk management |
| Entry-Exit FSM | ✅ | [ENTRY_EXIT_FSM_PATTERN.md](ENTRY_EXIT_FSM_PATTERN.md) | Trade lifecycle |
| Orthogonal Regions | ✅ | [ORTHOGONAL_REGIONS_PATTERN.md](ORTHOGONAL_REGIONS_PATTERN.md) | Parallel FSMs |
| State Aggregation | ✅ | [STATE_AGGREGATION_PATTERN.md](STATE_AGGREGATION_PATTERN.md) | State composition |
| TCP Connection FSM | ✅ | [TCP_CONNECTION_FSM_PATTERN.md](TCP_CONNECTION_FSM_PATTERN.md) | Network protocol |
| UTXO State Machine | ✅ | [UTXO_STATE_MACHINE_PATTERN.md](UTXO_STATE_MACHINE_PATTERN.md) | Bitcoin UTXO |
| History State | ✅ | [HISTORY_STATE_PATTERN.md](HISTORY_STATE_PATTERN.md) | State memory |
| Decomposed FSM | ✅ | [DECOMPOSED_FSM_PATTERN.md](DECOMPOSED_FSM_PATTERN.md) | Hierarchical decomposition |
| Script Execution | ✅ | [SCRIPT_EXECUTION_PATTERN.md](SCRIPT_EXECUTION_PATTERN.md) | Bitcoin Script VM |

## NEW: Signal Processing Patterns (7)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| edge_detector | Detect rising edge (0→1 transition) | 1 signal |
| falling_edge | Detect falling edge (1→0 transition) | 1 signal |
| toggle | Toggle state on each rising edge | 1 trigger |
| latch | SR latch (set/reset flip-flop) | 2 (set, reset) |
| debounce | Filter noisy signals | 1 signal |
| pulse_generator | Generate single-cycle pulse | 1 trigger |
| sample_hold | Sample data on trigger | 2 (data, trigger) |

## NEW: Data Flow / Routing Patterns (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| multiplexer | 2-to-1 MUX (select between inputs) | 3 (a, b, select) |
| demultiplexer | 1-to-2 DEMUX (route to outputs) | 2 (data, select) |
| priority_encoder | Select highest priority active input | 2+ candidates |
| arbiter | Grant access to requesters | 2+ requests |

## NEW: Safety / Watchdog Patterns (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| watchdog | Alarm if no heartbeat | 1 heartbeat |
| deadman_switch | Output only while held | 1 signal |
| safety_interlock | All safety conditions required | 2+ conditions |
| fault_detector | Detect disagreement (XOR) | 2+ signals |

## NEW: Protocol / Handshake Patterns (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| handshake | Request-acknowledge completion | 2 (request, ack) |
| sync_barrier | Wait for all ready signals | 2+ ready |
| token_ring | Token passing protocol | 2 (token_in, release) |

## NEW: Arithmetic / Comparison Patterns (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| comparator | Equality check (XNOR) | 2 signals |
| min_selector | AND all inputs (min for sbf) | 2+ signals |
| max_selector | OR all inputs (max for sbf) | 2+ signals |
| threshold_detector | N-of-M threshold | 2+ signals |

## NEW: Consensus / Distributed Patterns (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| byzantine_fault_tolerant | BFT consensus (2f+1 threshold) | 4+ nodes |
| leader_election | Priority-based leader selection | 2+ candidates |
| commit_protocol | Two-phase commit (all agree) | 2+ participants |

## NEW: Gate / Logic Patterns (5)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| nand_gate | NOT(AND) | 2+ signals |
| nor_gate | NOT(OR) | 2+ signals |
| xnor_gate | Equality (NOT XOR) | 2 signals |
| implication | a → b (if a then b) | 2 (a, b) |
| equivalence | a ↔ b (iff) | 2 (a, b) |

## NEW: Timing / Delay Patterns (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| delay_line | 1-cycle delay | 1 signal |
| hold | Hold value until load | 2 (data, load) |
| one_shot | Single pulse then lock | 2 (trigger, reset) |

## NEW: State Encoding Patterns (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| gray_code | Gray code transition | 1 signal |
| ring_counter | Rotating single-hot bit | 1 clock |
| sequence_detector | Detect bit patterns | 1 signal |

## NEW: Intelligent Agent - Decision Making (7)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| confidence_gate | Filter by confidence level | 2 (action, confidence) |
| action_selector | Priority-based action selection | 2+ candidates |
| exploration_exploit | Epsilon-greedy decision | 3 (explore, exploit, flag) |
| reward_accumulator | RL reward tracking | 2 (reward, reset) |
| goal_detector | Detect goal achievement | 1+ conditions |
| obstacle_detector | Detect blocking conditions | 1+ obstacles |
| policy_switch | Switch between policies | 3 (policy_a, policy_b, selector) |

## NEW: Intelligent Agent - Learning & Memory (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| experience_buffer | Store experience for replay | 2 (experience, store_trigger) |
| learning_gate | Enable/disable learning | 2 (update, enabled) |
| attention_focus | Attention mechanism | 3+ (data..., attention) |

## NEW: Intelligent Agent - Coordination (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| consensus_vote | Multi-agent consensus | 2+ votes |
| broadcast | Send to multiple outputs | 1 signal |
| message_filter | Filter by validity | 2 (message, valid) |
| role_assignment | Assign roles | 2 (capability, request) |

## NEW: Intelligent Agent - Safety (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| action_mask | Mask invalid actions | 2 (action, mask) |
| safety_override | Human-in-the-loop override | 3 (agent, safe, trigger) |
| constraint_checker | Verify constraints | 1+ constraints |
| budget_gate | Resource-constrained actions | 2 (action, budget) |

## NEW: Intelligent Agent - Inference (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| prediction_gate | Use prediction when obs unavailable | 3 (obs, pred, avail) |
| anomaly_detector | Detect out-of-distribution | 2 (current, expected) |
| state_classifier | Classify state from observations | 2+ conditions |

## NEW: Agent Fairness & Coordination Patterns (20)

Patterns for multi-agent coordination, verifiable randomness, and fair decision-making.

> **Inspiration:** General concepts of formal verification for trust-free coordination, as explored in the [Provably Fair Multiplayer Gaming](https://github.com/taumorrow/provably_fair_gaming/blob/master/PROVABLY_FAIR_GAMING_WHITEPAPER.MD) whitepaper by Tau Tomorrow (l0g1x). These patterns are **independent implementations for intelligent agents** (non-gaming), applying standard formal verification principles (LTL, safety invariants, fairness) to agent coordination. No gaming-specific code or specs were copied.

### Multi-Agent Randomness & Commitment (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| xor_combine | Multi-party randomness via XOR | 2+ random values |
| commitment_match | Verify revealed matches commitment | 2 (revealed, committed) |
| all_revealed | Check all parties revealed | 2+ reveal flags |
| phase_gate | Action only during specific phase | 2 (action, phase_active) |

### Agent State & Conflict Resolution (8)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| collision_detect | Detect conflicting agent states | 2 agent states |
| turn_gate | Enforce agent scheduling/ordering | 2 (action, my_turn) |
| fair_priority | Deterministic conflict resolution | 3 (action_a, action_b, priority) |
| capture | Resource acquisition on conflict | 2 (acquirer, holder) |
| territory_control | Track resource/zone ownership | 2 (claim, contest) |
| valid_move | Verify action satisfies constraints | 1+ rule conditions |
| win_condition | Check goal achievement | 1+ goal conditions |
| game_over | Terminal state latch | 1+ termination conditions |

### Agent Performance Tracking (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| streak_detector | Detect consecutive successes | 1 success signal |
| combo_counter | Track sustained performance | 1 success signal |
| score_gate | Conditional reward attribution | 2 (event, condition) |
| bonus_trigger | Activate performance bonus | 2 (streak, condition) |

### Multi-Agent Action Coordination (4)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| simultaneous_action | Detect synchronized agent actions | 2+ agent actions |
| any_action | Detect any agent activity | 2+ agent actions |
| exclusive_action | Mutual exclusion enforcement | 2 actions |
| cooldown | Rate limiting after action | 1 trigger |

## NEW: Formal Verification Invariant Patterns (19)

Novel patterns for safety, liveness, fairness, convergence, and learning guarantees in intelligent agent systems.

### Safety Invariants (3)

| Pattern | Use Case | Inputs | LTL Form |
|---------|----------|--------|----------|
| mutual_exclusion | At most one agent in critical section | 2 agent states | G(¬(a ∧ b)) |
| never_unsafe | Forbidden state prevention | 1 unsafe_condition | G(¬unsafe) |
| belief_consistency | Beliefs don't contradict | 2 (belief, negated) | G(consistent) |

### Liveness & Progress (4)

| Pattern | Use Case | Inputs | LTL Form |
|---------|----------|--------|----------|
| request_response | Every request gets response | 2 (request, grant) | G(req → F grant) |
| recovery | System recovers from failure | 2 (failure, recovered) | G(fail → F recover) |
| progress | System makes progress toward goal | 2 (enabled, done) | G(enabled → F done) |
| bounded_until | Condition holds until goal | 2 (condition, goal) | cond U goal |

### Fairness & Starvation Prevention (2)

| Pattern | Use Case | Inputs | LTL Form |
|---------|----------|--------|----------|
| no_starvation | Agent not starved of resources | 2 (request, grant) | GF req → GF grant |
| consensus_check | All agents have same value | 2+ agent values | F G (all equal) |

### Convergence & Stabilization (2)

| Pattern | Use Case | Inputs | LTL Form |
|---------|----------|--------|----------|
| stabilization | System reaches stable state | 1 state | F G stable |
| consensus_check | Consensus reached | 2+ values | F G (v₁ = v₂ = ...) |

### Trust & Reputation (2)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| trust_update | Update trust based on behavior | 2 (good_outcome, bad_outcome) |
| reputation_gate | Action only if trusted | 2 (action, trusted) |

### Risk-Aware Decision Making (2)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| risk_gate | Action only if risk below threshold | 2 (action, high_risk) |
| counterfactual_safe | Avoid bad counterfactual outcomes | 2 (choice, bad_counterfactual) |

### Learning & Exploration (3)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| exploration_decay | Exploration decreases over time | 2 (trigger, experience) |
| safe_exploration | Explore only when safe | 2 (explore, safe) |
| emergent_detector | Detect unexpected behavior | 2 (actual, expected) |

### Causal & Alignment (2)

| Pattern | Use Case | Inputs |
|---------|----------|--------|
| utility_alignment | Actions align with preferences | 2 (action, aligned) |
| causal_gate | Action only if causal precondition met | 2 (action, causal_condition) |

**Total: 122 patterns ✅**

## Pattern Categories

### Basic Patterns

Simple, foundational patterns for basic agent functionality.

- **FSM**: Finite state machine for position tracking (buy/sell → position)
- **Counter**: Toggle counter that flips on events
- **Accumulator**: Sums values over time (bitvector only)
- **Vote**: OR-based voting from multiple inputs
- **Passthrough**: Direct input-to-output mapping

**Documentation**: See [README.md](README.md) and [COMPLEXITY_ANALYSIS.md](COMPLEXITY_ANALYSIS.md)

### Composite Patterns

Patterns that combine multiple inputs or use complex logic.

- **Majority**: N-of-M majority voting (e.g., 2-of-3, 3-of-5)
- **Unanimous**: All inputs must agree (unanimous consensus)
- **Custom**: Custom boolean expressions with stream references
- **Quorum**: Minimum votes required (uses majority internally)

**Documentation**: [ENSEMBLE_PATTERNS.md](ENSEMBLE_PATTERNS.md)

### Hierarchical Patterns

Patterns for building complex, multi-level state machines.

- **Supervisor-Worker**: Hierarchical FSM where supervisor coordinates workers
- **Orthogonal Regions**: Parallel independent FSMs running simultaneously
- **State Aggregation**: Combining multiple FSM states into superstate

**Documentation**: 
- [HIERARCHICAL_FSM_DESIGN.md](HIERARCHICAL_FSM_DESIGN.md)
- [ORTHOGONAL_REGIONS_PATTERN.md](ORTHOGONAL_REGIONS_PATTERN.md)
- [STATE_AGGREGATION_PATTERN.md](STATE_AGGREGATION_PATTERN.md)

### Bitvector Patterns

Patterns that leverage bitvector arithmetic for complex calculations.

- **Weighted Vote**: Weighted sum of inputs compared to threshold
- **Time Lock**: Time-based locking using bitvector arithmetic

**Documentation**: [BITVECTOR_PATTERNS.md](BITVECTOR_PATTERNS.md)

### Domain-Specific Patterns

Patterns designed for specific application domains.

- **Hex Stake**: Time-lock staking system with rewards and penalties
- **Entry-Exit FSM**: Multi-phase trade lifecycle (PRE_TRADE → IN_TRADE → POST_TRADE)
- **Proposal FSM**: Governance proposal lifecycle (DRAFT → VOTING → PASSED → EXECUTED)
- **Risk FSM**: Risk state machine (NORMAL → WARNING → CRITICAL)
- **TCP Connection FSM**: TCP connection state machine (11 states)
- **UTXO State Machine**: Bitcoin UTXO set tracking

**Documentation**: 
- [HEX_SUMMARY.md](HEX_SUMMARY.md)
- [ENTRY_EXIT_FSM_PATTERN.md](ENTRY_EXIT_FSM_PATTERN.md)
- [TCP_CONNECTION_FSM_PATTERN.md](TCP_CONNECTION_FSM_PATTERN.md)
- [UTXO_STATE_MACHINE_PATTERN.md](UTXO_STATE_MACHINE_PATTERN.md)
- [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md)

### Advanced Patterns

Advanced patterns for sophisticated agent behavior.

- **Multi-Bit Counter**: Multi-bit counters with increment and reset
- **Streak Counter**: Consecutive event tracking with reset
- **Mode Switch**: Adaptive mode switching (e.g., AGGRESSIVE/DEFENSIVE)

**Documentation**: [HIGH_PRIORITY_PATTERNS_COMPLETE.md](HIGH_PRIORITY_PATTERNS_COMPLETE.md)

## Pattern Selection Guide

### For Trading Agents

**Basic Trading:**
- FSM (position tracking)
- Counter (trade counting)
- Accumulator (PnL tracking)

**Advanced Trading:**
- Entry-Exit FSM (trade lifecycle)
- Risk FSM (risk management)
- Multi-Bit Counter (timers)
- Streak Counter (win/loss tracking)
- Mode Switch (adaptive behavior)

**Multi-Agent Trading:**
- Supervisor-Worker (coordination)
- Majority/Unanimous (voting)
- Weighted Vote (weighted decisions)

### For Governance/DAO Agents

**Basic Governance:**
- Proposal FSM (proposal lifecycle)
- Quorum (minimum votes)
- Majority (voting)

**Advanced Governance:**
- Custom (complex voting rules)
- Time Lock (time-based actions)
- Hex Stake (staking mechanisms)

### For Risk Management

**Risk Monitoring:**
- Risk FSM (risk states)
- Custom (risk calculations)
- Accumulator (risk metrics)

**Risk Controls:**
- Time Lock (time-based limits)
- Supervisor-Worker (risk coordination)

## Pattern Combinations

### Common Combinations

**Trading Agent with Risk Management:**
```python
LogicBlock(pattern="entry_exit_fsm", inputs=("entry", "exit"), output="phase")
LogicBlock(pattern="risk_fsm", inputs=("warning", "critical", "normal"), output="risk")
LogicBlock(pattern="custom", inputs=("entry", "risk"), output="safe_entry",
           params={"expression": "entry[t] & (risk[t] = {0}:bv[2])"})
```

**Multi-Agent Ensemble:**
```python
LogicBlock(pattern="fsm", inputs=("buy1", "sell1"), output="pos1")
LogicBlock(pattern="fsm", inputs=("buy2", "sell2"), output="pos2")
LogicBlock(pattern="fsm", inputs=("buy3", "sell3"), output="pos3")
LogicBlock(pattern="majority", inputs=("pos1", "pos2", "pos3"), output="ensemble_pos",
           params={"threshold": 2, "total": 3})
```

**Timed Trading:**
```python
LogicBlock(pattern="entry_exit_fsm", inputs=("entry", "exit"), output="phase")
LogicBlock(pattern="multi_bit_counter", inputs=("position",), output="timer",
           params={"width": 4})
LogicBlock(pattern="custom", inputs=("timer",), output="timeout",
           params={"expression": "timer[t] >= {10}:bv[4]"})
```

## Testing Patterns

All patterns have comprehensive test coverage:

- Unit tests: `tests/test_*.py`
- Integration tests: `tests/test_e2e.py`
- Real Tau execution: `tests/test_real_tau_execution.py`
- Pattern-specific tests: `tests/test_*_pattern.py`

## Implementation Status

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for:
- Current progress (19/26 patterns)
- Remaining patterns
- Priority rankings
- Estimated completion times

## Related Documentation

- [Pattern Landscape](PATTERN_LANDSCAPE.md) - Complete pattern taxonomy
- [Limitations & Why](LIMITATIONS_AND_WHY.md) - What can't be done and why
- [Complexity Analysis](COMPLEXITY_ANALYSIS.md) - Current capabilities
- [Tau Language Capabilities](../../specification/TAU_LANGUAGE_CAPABILITIES.md) - Tau Language reference

## Contributing

When adding new patterns:

1. Implement pattern in `generator.py`
2. Add to `schema.py` valid patterns
3. Write tests in `tests/test_*_pattern.py`
4. Create pattern documentation (like this file)
5. Update this index
6. Update [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

