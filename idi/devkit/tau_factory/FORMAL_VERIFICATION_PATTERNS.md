# Formal Verification Patterns for Intelligent Agents

Patterns implementing formal verification invariants (safety, liveness, fairness, convergence) for intelligent agent systems.

> **Note:** These patterns apply standard formal verification concepts (LTL temporal logic, safety invariants) to intelligent agent coordination. All patterns exhaustively tested against Tau binary.

## Quick Reference: Tau Logic Operators

Understanding these operators is key to reading the specifications:

| Operator | Name | Meaning | Example |
|----------|------|---------|---------|
| `&` | AND | Both must be true | `a & b` = 1 only when both a=1 and b=1 |
| `\|` | OR | Either can be true | `a \| b` = 1 when a=1 or b=1 or both |
| `^` | XOR | Exactly one true | `a ^ b` = 1 when a≠b (difference detector) |
| `'` | NOT | Negation | `a'` = 1 when a=0, and vice versa |

### Common Compound Patterns

| Pattern | Formula | Meaning | Truth Table |
|---------|---------|---------|-------------|
| **NAND** | `(a & b)'` | NOT(both true) | (0,0)→1, (0,1)→1, (1,0)→1, (1,1)→0 |
| **NOR** | `(a \| b)'` | Neither true | (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→0 |
| **XNOR** | `(a ^ b)'` | Both same (equality) | (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→1 |
| **Implication** | `a' \| b` | If a then b | (0,0)→1, (0,1)→1, (1,0)→0, (1,1)→1 |
| **AND-NOT** | `a & b'` | a but not b | (0,0)→0, (0,1)→0, (1,0)→1, (1,1)→0 |

### Reading Tips

- **`'` at end of expression**: Negates the ENTIRE preceding expression
  - `(a & b)'` = NOT(a AND b)
  - `a'` = NOT(a)
- **Implication `a → b`**: Written as `a' | b` (NOT a OR b)
- **Equality check**: Use XNOR `(a ^ b)'` - outputs 1 when a equals b

---

## Pattern Reference

Each pattern shows:
1. **Pure form** - Default Tau output with indexed variables (`i0`, `o0`)
2. **Human-readable form** - Descriptive names (requires `tau --charvar off`)
3. **LTL semantics** - Formal temporal logic interpretation
4. **Python oracle** - Reference implementation for testing

---

## Safety Invariants

### mutual_exclusion

**Use Case:** At most one agent can hold a resource/enter critical section at a time.

**LTL:** `G(¬(agent_a ∧ agent_b))` - Globally, never both active simultaneously

**Pure Form:**
```tau
i0:sbf = in file("inputs/agent_a.in")
i1:sbf = in file("inputs/agent_b.in")
o0:sbf = out file("outputs/safe.out")

# NAND gate: safe = NOT(both active)
# The ' operator negates the entire expression (i0 & i1)
# Result: 1 when at most one agent active, 0 when both active
r o0[t] = (i0[t] & i1[t])'
```

**Human-Readable Form:**
```tau
agent_a:sbf = in file("inputs/agent_a.in")
agent_b:sbf = in file("inputs/agent_b.in")
safe:sbf = out file("outputs/safe.out")

# NAND: safe when NOT(agent_a AND agent_b)
# Truth table: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->0
r safe[t] = (agent_a[t] & agent_b[t])'
```

**Oracle:**
```python
def mutual_exclusion(agent_a, agent_b):
    return 1 - (agent_a & agent_b)  # NAND
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### never_unsafe

**Use Case:** Forbidden state prevention - safety invariant must always hold.

**LTL:** `G(¬unsafe)` - Globally, unsafe condition is never true

**Pure Form:**
```tau
i0:sbf = in file("inputs/unsafe.in")
o0:sbf = out file("outputs/safe.out")

# Simple NOT: safe = NOT(unsafe)
# The ' directly negates the input
r o0[t] = i0[t]'
```

**Human-Readable Form:**
```tau
unsafe_condition:sbf = in file("inputs/unsafe.in")
is_safe:sbf = out file("outputs/safe.out")

# Invert unsafe signal: 1 when safe, 0 when unsafe
r is_safe[t] = unsafe_condition[t]'
```

**Oracle:**
```python
def never_unsafe(unsafe):
    return 1 - unsafe  # NOT
```

**Test Result:** ✅ PASSED 8/8 combinations

---

### belief_consistency

**Use Case:** Agent beliefs must not contradict each other.

**LTL:** `G(¬(belief ∧ ¬belief))` - Beliefs are always consistent

**Pure Form:**
```tau
i0:sbf = in file("inputs/belief.in")
i1:sbf = in file("inputs/negated_belief.in")
o0:sbf = out file("outputs/consistent.out")
r o0[t] = (i0[t] & i1[t])'
```

**Human-Readable Form:**
```tau
belief:sbf = in file("inputs/belief.in")
negated_belief:sbf = in file("inputs/negated_belief.in")
consistent:sbf = out file("outputs/consistent.out")
r consistent[t] = (belief[t] & negated_belief[t])'
```

**Oracle:**
```python
def belief_consistency(belief, negated_belief):
    return 1 - (belief & negated_belief)  # NAND
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Liveness & Progress

### no_starvation

**Use Case:** If agent requests resource, it must eventually be granted.

**LTL:** `G(request → F grant)` - Simplified as implication at each step

**Pure Form:**
```tau
i0:sbf = in file("inputs/request.in")
i1:sbf = in file("inputs/grant.in")
o0:sbf = out file("outputs/fair.out")

# IMPLICATION: request -> grant  (equivalent to: NOT request OR grant)
# Logic: If you request, you must be granted. No request = automatically fair.
# Truth table: (0,0)->1, (0,1)->1, (1,0)->0, (1,1)->1
# Only fails (output=0) when request=1 but grant=0
r o0[t] = i0[t]' | i1[t]
```

**Human-Readable Form:**
```tau
request:sbf = in file("inputs/request.in")
grant:sbf = in file("inputs/grant.in")
fair_service:sbf = out file("outputs/fair.out")

# Implication: request implies grant
# Read as: "if requesting, then granted" OR "not requesting"
r fair_service[t] = request[t]' | grant[t]
```

**Oracle:**
```python
def no_starvation(request, grant):
    return (1 - request) | grant  # Implication: ¬req ∨ grant
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### progress

**Use Case:** System makes progress toward goal when enabled.

**LTL:** `G(enabled → F done)` - Simplified as implication

**Pure Form:**
```tau
i0:sbf = in file("inputs/enabled.in")
i1:sbf = in file("inputs/done.in")
o0:sbf = out file("outputs/progressing.out")
r o0[t] = i0[t]' | i1[t]
```

**Human-Readable Form:**
```tau
task_enabled:sbf = in file("inputs/enabled.in")
task_done:sbf = in file("inputs/done.in")
progressing:sbf = out file("outputs/progressing.out")
r progressing[t] = task_enabled[t]' | task_done[t]
```

**Oracle:**
```python
def progress(enabled, done):
    return (1 - enabled) | done  # Implication
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Convergence & Consensus

### consensus_check

**Use Case:** Verify all agents have converged to the same value.

**LTL:** `F G(v₁ = v₂)` - Eventually, all values equal and stay equal

**Pure Form:**
```tau
i0:sbf = in file("inputs/value_a.in")
i1:sbf = in file("inputs/value_b.in")
o0:sbf = out file("outputs/consensus.out")

# XNOR (equality check): consensus = NOT(a XOR b)
# XOR gives 1 when values differ, so NOT(XOR) gives 1 when values are SAME
# Truth table: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
r o0[t] = (i0[t] ^ i1[t])'
```

**Human-Readable Form:**
```tau
agent_a_value:sbf = in file("inputs/value_a.in")
agent_b_value:sbf = in file("inputs/value_b.in")
in_consensus:sbf = out file("outputs/consensus.out")

# XNOR: output 1 when both agents have same value (consensus reached)
# Think of it as "equality test" - same values = 1, different = 0
r in_consensus[t] = (agent_a_value[t] ^ agent_b_value[t])'
```

**Oracle:**
```python
def consensus_check(value_a, value_b):
    return 1 - (value_a ^ value_b)  # XNOR (equal)
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Trust & Reputation

### reputation_gate

**Use Case:** Only allow action if agent has sufficient trust/reputation.

**LTL:** `G(action → trusted)` - Actions require trust

**Pure Form:**
```tau
i0:sbf = in file("inputs/action.in")
i1:sbf = in file("inputs/trusted.in")
o0:sbf = out file("outputs/allowed.out")
r o0[t] = i0[t] & i1[t]
```

**Human-Readable Form:**
```tau
proposed_action:sbf = in file("inputs/action.in")
is_trusted:sbf = in file("inputs/trusted.in")
action_allowed:sbf = out file("outputs/allowed.out")
r action_allowed[t] = proposed_action[t] & is_trusted[t]
```

**Oracle:**
```python
def reputation_gate(action, trusted):
    return action & trusted  # AND
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Risk-Aware Decision Making

### risk_gate

**Use Case:** Only allow action if risk is below threshold.

**LTL:** `G(action → ¬high_risk)` - Actions require low risk

**Pure Form:**
```tau
i0:sbf = in file("inputs/action.in")
i1:sbf = in file("inputs/high_risk.in")
o0:sbf = out file("outputs/safe_action.out")

# AND-NOT gate: action passes through only when risk is LOW
# The ' negates i1, so i1[t]' = NOT(high_risk)
# Result: action AND NOT(high_risk) = action when safe
# Truth table: (action=1, risk=0)->1, (action=1, risk=1)->0
r o0[t] = i0[t] & i1[t]'
```

**Human-Readable Form:**
```tau
proposed_action:sbf = in file("inputs/action.in")
high_risk:sbf = in file("inputs/high_risk.in")
safe_action:sbf = out file("outputs/safe_action.out")

# Gate the action: only allow when NOT high_risk
# Read as: "do action AND it's not risky"
r safe_action[t] = proposed_action[t] & high_risk[t]'
```

**Oracle:**
```python
def risk_gate(action, high_risk):
    return action & (1 - high_risk)  # action AND NOT risk
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### counterfactual_safe

**Use Case:** Avoid actions with bad counterfactual outcomes.

**LTL:** `G(choice → ¬bad_outcome)` - Choices avoid bad counterfactuals

**Pure Form:**
```tau
i0:sbf = in file("inputs/choice.in")
i1:sbf = in file("inputs/bad_counterfactual.in")
o0:sbf = out file("outputs/safe_choice.out")

# AND-NOT: choice is safe only if counterfactual outcome is NOT bad
# Blocks choices that could lead to bad outcomes in alternate scenarios
r o0[t] = i0[t] & i1[t]'
```

**Human-Readable Form:**
```tau
agent_choice:sbf = in file("inputs/choice.in")
bad_counterfactual:sbf = in file("inputs/bad_counterfactual.in")
safe_choice:sbf = out file("outputs/safe_choice.out")

# Only allow choice if "what-if" analysis shows no bad outcome
# Read as: "make choice AND counterfactual is not bad"
r safe_choice[t] = agent_choice[t] & bad_counterfactual[t]'
```

**Oracle:**
```python
def counterfactual_safe(choice, bad_counterfactual):
    return choice & (1 - bad_counterfactual)
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Learning & Exploration

### exploration_decay

**Use Case:** Exploration rate decreases as experience accumulates.

**LTL:** `F G(¬exploring)` - Eventually, exploitation dominates

**Pure Form:**
```tau
i0:sbf = in file("inputs/explore_trigger.in")
i1:sbf = in file("inputs/experienced.in")
o0:sbf = out file("outputs/explore.out")

# AND-NOT: explore only when triggered AND NOT yet experienced
# As experience accumulates (i1=1), exploration stops (output=0)
# Models epsilon-greedy decay: explore early, exploit later
r o0[t] = i0[t] & i1[t]'
```

**Human-Readable Form:**
```tau
explore_trigger:sbf = in file("inputs/explore_trigger.in")
has_experience:sbf = in file("inputs/experienced.in")
should_explore:sbf = out file("outputs/explore.out")

# Exploration decays with experience
# When has_experience=1, exploration is blocked regardless of trigger
r should_explore[t] = explore_trigger[t] & has_experience[t]'
```

**Oracle:**
```python
def exploration_decay(trigger, experienced):
    return trigger & (1 - experienced)  # Explore only if not experienced
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### safe_exploration

**Use Case:** Explore only when it's safe to do so (shielded RL).

**LTL:** `G(explore → safe)` - Exploration requires safety

**Pure Form:**
```tau
i0:sbf = in file("inputs/explore.in")
i1:sbf = in file("inputs/safe.in")
o0:sbf = out file("outputs/safe_explore.out")
r o0[t] = i0[t] & i1[t]
```

**Human-Readable Form:**
```tau
want_explore:sbf = in file("inputs/explore.in")
is_safe:sbf = in file("inputs/safe.in")
safe_exploration:sbf = out file("outputs/safe_explore.out")
r safe_exploration[t] = want_explore[t] & is_safe[t]
```

**Oracle:**
```python
def safe_exploration(explore, safe):
    return explore & safe  # AND
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### emergent_detector

**Use Case:** Detect unexpected/emergent behavior deviating from expected.

**LTL:** `G(actual ≠ expected → alert)` - Detect deviations

**Pure Form:**
```tau
i0:sbf = in file("inputs/actual.in")
i1:sbf = in file("inputs/expected.in")
o0:sbf = out file("outputs/emergent.out")

# XOR: detect when actual differs from expected (deviation/anomaly)
# Output=1 when values are DIFFERENT, 0 when they match
# Useful for detecting unexpected/emergent behavior
r o0[t] = i0[t] ^ i1[t]
```

**Human-Readable Form:**
```tau
actual_behavior:sbf = in file("inputs/actual.in")
expected_behavior:sbf = in file("inputs/expected.in")
emergent_detected:sbf = out file("outputs/emergent.out")

# XOR detects deviation: 1 when actual != expected
# Alert fires when behavior differs from model prediction
r emergent_detected[t] = actual_behavior[t] ^ expected_behavior[t]
```

**Oracle:**
```python
def emergent_detector(actual, expected):
    return actual ^ expected  # XOR (difference)
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Causal & Alignment

### utility_alignment

**Use Case:** Verify agent actions align with desired utility/preferences.

**LTL:** `G(action → aligned)` - Actions imply alignment

**Pure Form:**
```tau
i0:sbf = in file("inputs/action.in")
i1:sbf = in file("inputs/aligned.in")
o0:sbf = out file("outputs/valid.out")

# IMPLICATION: action -> aligned (equivalent to: NOT action OR aligned)
# Validates that IF agent takes action, THEN it must be aligned with preferences
# No action (i0=0) is automatically valid; action requires alignment
r o0[t] = i0[t]' | i1[t]
```

**Human-Readable Form:**
```tau
agent_action:sbf = in file("inputs/action.in")
preference_aligned:sbf = in file("inputs/aligned.in")
action_valid:sbf = out file("outputs/valid.out")

# Implication: action implies alignment with human preferences
# Fails (output=0) only when action=1 but aligned=0
# Used for AI safety: verify actions match intended utility
r action_valid[t] = agent_action[t]' | preference_aligned[t]
```

**Oracle:**
```python
def utility_alignment(action, aligned):
    return (1 - action) | aligned  # Implication: ¬action ∨ aligned
```

**Test Result:** ✅ PASSED 64/64 combinations

---

### causal_gate

**Use Case:** Action only if causal precondition is satisfied.

**LTL:** `G(do(action) → belief(cause → effect))` - Causal reasoning

**Pure Form:**
```tau
i0:sbf = in file("inputs/action.in")
i1:sbf = in file("inputs/causal_condition.in")
o0:sbf = out file("outputs/valid_action.out")
r o0[t] = i0[t] & i1[t]
```

**Human-Readable Form:**
```tau
proposed_action:sbf = in file("inputs/action.in")
causal_precondition:sbf = in file("inputs/causal_condition.in")
valid_action:sbf = out file("outputs/valid_action.out")
r valid_action[t] = proposed_action[t] & causal_precondition[t]
```

**Oracle:**
```python
def causal_gate(action, causal_condition):
    return action & causal_condition  # AND
```

**Test Result:** ✅ PASSED 64/64 combinations

---

## Summary

| Category | Pattern | LTL | Combinations | Status |
|----------|---------|-----|--------------|--------|
| Safety | mutual_exclusion | G(¬(a∧b)) | 64 | ✅ |
| Safety | never_unsafe | G(¬unsafe) | 8 | ✅ |
| Safety | belief_consistency | G(consistent) | 64 | ✅ |
| Liveness | no_starvation | G(req→F grant) | 64 | ✅ |
| Liveness | progress | G(enabled→F done) | 64 | ✅ |
| Convergence | consensus_check | FG(v₁=v₂) | 64 | ✅ |
| Trust | reputation_gate | G(action→trusted) | 64 | ✅ |
| Risk | risk_gate | G(action→¬risk) | 64 | ✅ |
| Risk | counterfactual_safe | G(choice→¬bad) | 64 | ✅ |
| Learning | exploration_decay | FG(¬exploring) | 64 | ✅ |
| Learning | safe_exploration | G(explore→safe) | 64 | ✅ |
| Learning | emergent_detector | G(detect deviation) | 64 | ✅ |
| Alignment | utility_alignment | G(action→aligned) | 64 | ✅ |
| Alignment | causal_gate | G(do→cause) | 64 | ✅ |

**Total: 14/14 patterns passed exhaustive testing**
