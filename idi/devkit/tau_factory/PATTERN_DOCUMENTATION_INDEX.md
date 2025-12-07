# Pattern Documentation Index

Complete reference guide for all implemented patterns in the Tau Agent Factory.

## Quick Reference

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

**Total: 22/26 patterns (85%)**

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

**Documentation**: 
- [HEX_SUMMARY.md](HEX_SUMMARY.md)
- [ENTRY_EXIT_FSM_PATTERN.md](ENTRY_EXIT_FSM_PATTERN.md)
- [TCP_CONNECTION_FSM_PATTERN.md](TCP_CONNECTION_FSM_PATTERN.md)
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

