# Tau Specification Tutorials

Learn to write formal specifications for intelligent agents using Tau Language.

## Quick Start

New to Tau? Start here:

1. **[TAU_SPECIFICATION_TUTORIAL.md](TAU_SPECIFICATION_TUTORIAL.md)** - Complete beginner tutorial covering:
   - What are invariants?
   - Anatomy of a Tau specification
   - Reading logic formulas
   - 26 detailed specification tutorials
   - Best practices

## Tutorial Index by Difficulty

### ⭐ Beginner (Start Here)

| Tutorial | What You'll Learn |
|----------|-------------------|
| [safety_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-1-safety_gate) | AND gate basics, gating actions |
| [mutual_exclusion](TAU_SPECIFICATION_TUTORIAL.md#tutorial-2-mutual_exclusion) | NAND gate, preventing conflicts |
| [never_unsafe](TAU_SPECIFICATION_TUTORIAL.md#tutorial-3-never_unsafe) | NOT operator basics |
| [confidence_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-6-confidence_gate) | Simple AND patterns |
| [turn_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-14-turn_gate) | Action gating |
| [consensus_check](TAU_SPECIFICATION_TUTORIAL.md#tutorial-11-consensus_check) | XNOR for equality |
| [emergent_detector](TAU_SPECIFICATION_TUTORIAL.md#tutorial-21-emergent_detector) | XOR for difference |
| [collision_detect](TAU_SPECIFICATION_TUTORIAL.md#tutorial-24-collision_detect) | XNOR for collision |

### ⭐⭐ Intermediate

| Tutorial | What You'll Learn |
|----------|-------------------|
| [risk_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-4-risk_gate) | AND-NOT pattern |
| [belief_consistency](TAU_SPECIFICATION_TUTORIAL.md#tutorial-5-belief_consistency) | NAND for consistency |
| [multiplexer](TAU_SPECIFICATION_TUTORIAL.md#tutorial-7-multiplexer) | Selection logic |
| [policy_switch](TAU_SPECIFICATION_TUTORIAL.md#tutorial-8-policy_switch) | Mode switching |
| [exploration_decay](TAU_SPECIFICATION_TUTORIAL.md#tutorial-9-exploration_decay) | AND-NOT for learning |
| [majority_vote](TAU_SPECIFICATION_TUTORIAL.md#tutorial-12-majority_vote) | Multi-input OR-AND |
| [sync_barrier](TAU_SPECIFICATION_TUTORIAL.md#tutorial-13-sync_barrier) | Multi-input AND |
| [progress](TAU_SPECIFICATION_TUTORIAL.md#tutorial-16-progress) | Implication pattern |
| [no_starvation](TAU_SPECIFICATION_TUTORIAL.md#tutorial-17-no_starvation) | Fairness implication |
| [request_response](TAU_SPECIFICATION_TUTORIAL.md#tutorial-18-request_response) | Temporal memory |
| [edge_detector](TAU_SPECIFICATION_TUTORIAL.md#tutorial-23-edge_detector) | Temporal change detection |
| [reputation_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-26-reputation_gate) | Trust-based gating |

### ⭐⭐⭐ Advanced

| Tutorial | What You'll Learn |
|----------|-------------------|
| [utility_alignment](TAU_SPECIFICATION_TUTORIAL.md#tutorial-10-utility_alignment) | AI safety implication |
| [fair_priority](TAU_SPECIFICATION_TUTORIAL.md#tutorial-15-fair_priority) | Complex conflict resolution |
| [recovery](TAU_SPECIFICATION_TUTORIAL.md#tutorial-19-recovery) | Temporal state machines |
| [bounded_until](TAU_SPECIFICATION_TUTORIAL.md#tutorial-20-bounded_until) | Temporal until patterns |
| [trust_update](TAU_SPECIFICATION_TUTORIAL.md#tutorial-25-trust_update) | Reputation state machine |

### 🔢 Bitvector Patterns (NEW)

| Tutorial | What You'll Learn |
|----------|-------------------|
| [weighted_vote](TAU_SPECIFICATION_TUTORIAL.md#tutorial-27-weighted_vote) | Ternary operator, weighted sums, comparisons |
| [time_lock](TAU_SPECIFICATION_TUTORIAL.md#tutorial-28-time_lock) | Arithmetic operations, time calculations |

## Tutorial Index by Category

### Safety Patterns
Learn to prevent dangerous states and enforce safety constraints.

- [safety_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-1-safety_gate) - Block unsafe actions
- [mutual_exclusion](TAU_SPECIFICATION_TUTORIAL.md#tutorial-2-mutual_exclusion) - Only one agent active
- [never_unsafe](TAU_SPECIFICATION_TUTORIAL.md#tutorial-3-never_unsafe) - Prevent forbidden states
- [risk_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-4-risk_gate) - Block high-risk actions
- [belief_consistency](TAU_SPECIFICATION_TUTORIAL.md#tutorial-5-belief_consistency) - No contradictory beliefs

### Decision Patterns
Learn to make decisions and switch between strategies.

- [confidence_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-6-confidence_gate) - Act only when confident
- [multiplexer](TAU_SPECIFICATION_TUTORIAL.md#tutorial-7-multiplexer) - Select between options
- [policy_switch](TAU_SPECIFICATION_TUTORIAL.md#tutorial-8-policy_switch) - Switch strategies
- [exploration_decay](TAU_SPECIFICATION_TUTORIAL.md#tutorial-9-exploration_decay) - Explore less over time
- [utility_alignment](TAU_SPECIFICATION_TUTORIAL.md#tutorial-10-utility_alignment) - Verify preference alignment

### Coordination Patterns
Learn to coordinate multiple agents.

- [consensus_check](TAU_SPECIFICATION_TUTORIAL.md#tutorial-11-consensus_check) - All agents agree
- [majority_vote](TAU_SPECIFICATION_TUTORIAL.md#tutorial-12-majority_vote) - Majority decides
- [sync_barrier](TAU_SPECIFICATION_TUTORIAL.md#tutorial-13-sync_barrier) - Wait for all ready
- [turn_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-14-turn_gate) - Enforce turn order
- [fair_priority](TAU_SPECIFICATION_TUTORIAL.md#tutorial-15-fair_priority) - Resolve conflicts fairly

### Progress Patterns
Learn to ensure the system makes progress.

- [progress](TAU_SPECIFICATION_TUTORIAL.md#tutorial-16-progress) - System advances toward goal
- [no_starvation](TAU_SPECIFICATION_TUTORIAL.md#tutorial-17-no_starvation) - Every request gets served
- [request_response](TAU_SPECIFICATION_TUTORIAL.md#tutorial-18-request_response) - Requests get responses
- [recovery](TAU_SPECIFICATION_TUTORIAL.md#tutorial-19-recovery) - System recovers from failure
- [bounded_until](TAU_SPECIFICATION_TUTORIAL.md#tutorial-20-bounded_until) - Condition holds until goal

### Detection Patterns
Learn to detect changes and anomalies.

- [emergent_detector](TAU_SPECIFICATION_TUTORIAL.md#tutorial-21-emergent_detector) - Detect unexpected behavior
- [anomaly_detector](TAU_SPECIFICATION_TUTORIAL.md#tutorial-22-anomaly_detector) - Detect anomalies
- [edge_detector](TAU_SPECIFICATION_TUTORIAL.md#tutorial-23-edge_detector) - Detect signal changes
- [collision_detect](TAU_SPECIFICATION_TUTORIAL.md#tutorial-24-collision_detect) - Detect conflicts

### Trust Patterns
Learn to manage trust and reputation.

- [trust_update](TAU_SPECIFICATION_TUTORIAL.md#tutorial-25-trust_update) - Track reputation
- [reputation_gate](TAU_SPECIFICATION_TUTORIAL.md#tutorial-26-reputation_gate) - Gate by trust level

## Quick Reference

### Logic Operators

| Symbol | Name | Meaning |
|--------|------|---------|
| `&` | AND | Both must be true |
| `\|` | OR | Either can be true |
| `^` | XOR | Exactly one true |
| `'` | NOT | Flip the value |

### Common Patterns

| Pattern | Formula | Use |
|---------|---------|-----|
| NAND | `(a & b)'` | Mutual exclusion |
| XNOR | `(a ^ b)'` | Equality check |
| Implication | `a' \| b` | If-then logic |
| AND-NOT | `a & b'` | Gate with negation |

### Temporal Operators

| Syntax | Meaning |
|--------|---------|
| `[t]` | Current time step |
| `[t-1]` | Previous time step |
| `[0]` | Initial value |

## Learning Path

1. **Week 1:** Complete beginner tutorials (⭐)
2. **Week 2:** Complete intermediate tutorials (⭐⭐)
3. **Week 3:** Complete advanced tutorials (⭐⭐⭐)
4. **Week 4:** Build your own specifications

## Related Documentation

- [FORMAL_VERIFICATION_PATTERNS.md](../FORMAL_VERIFICATION_PATTERNS.md) - LTL-based patterns with formal semantics
- [PATTERN_DOCUMENTATION_INDEX.md](../PATTERN_DOCUMENTATION_INDEX.md) - Complete pattern reference
- [EXHAUSTIVE_PATTERN_VERIFICATION.md](../EXHAUSTIVE_PATTERN_VERIFICATION.md) - Test results
