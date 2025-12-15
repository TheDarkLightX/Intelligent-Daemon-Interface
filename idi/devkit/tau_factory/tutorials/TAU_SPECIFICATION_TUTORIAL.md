# Tau Specification Tutorial for Beginners

A comprehensive guide to writing Tau specifications for intelligent agents.

---

## Table of Contents

1. [What Are Invariants?](#1-what-are-invariants)
2. [Anatomy of a Tau Specification](#2-anatomy-of-a-tau-specification)
3. [Reading Logic Formulas](#3-reading-logic-formulas)
4. [Bitvectors: Arithmetic and Comparisons](#4-bitvectors-arithmetic-and-comparisons)
5. [Writing Your First Specification](#5-writing-your-first-specification)
6. [Tutorial Index: 28 Specifications](#6-tutorial-index-28-specifications)
7. [Best Practices](#7-best-practices)

---

## 1. What Are Invariants?

Before writing any specification, you need to understand **invariants** - they are the foundation of formal verification.

### The Formal Definition

> An **invariant** is a property or condition that remains true throughout the execution of a program, holding before and after every operation.

In temporal logic notation:
- **G(property)** means "Globally, this property is always true"
- **G(¬bad_state)** means "The system never enters a bad state"

### The Casual Explanation

Think of an invariant as a **non-negotiable rule** that your system must always obey - like a promise that never breaks.

**Real-world analogies:**

| System | Invariant | What It Means |
|--------|-----------|---------------|
| Bank Account | Balance ≥ 0 | You can't have negative money |
| Traffic Light | Only one direction green | Prevents crashes |
| Elevator | Doors closed when moving | Safety guarantee |
| Login System | Max 3 failed attempts | Security protection |

### Why Invariants Matter for Intelligent Agents

When building intelligent agents, invariants ensure:

1. **Safety**: The agent never takes harmful actions
2. **Fairness**: Resources are distributed properly among agents
3. **Progress**: The system eventually achieves its goals
4. **Consistency**: Agent beliefs don't contradict each other

### Example: Identifying Invariants

Let's say you're building a trading agent. Before writing any code, list your invariants:

```
INVARIANTS FOR TRADING AGENT:
1. Position size never exceeds maximum allowed
2. Only one trade can be active at a time (mutual exclusion)
3. Every buy order must eventually be matched with a sell
4. Risk exposure never exceeds threshold
5. Agent actions must align with user preferences
```

**This is your starting point.** Every Tau specification begins with identifying what must ALWAYS be true.

---

## 2. Anatomy of a Tau Specification

A Tau specification has three main parts:

### Part 1: Stream Declarations (Inputs and Outputs)

```tau
# INPUTS: What the agent receives from the environment
sensor_a:sbf = in file("inputs/sensor_a.in")
sensor_b:sbf = in file("inputs/sensor_b.in")

# OUTPUTS: What the agent produces
action:sbf = out file("outputs/action.out")
```

**Key concepts:**
- `sbf` = Simple Boolean Function (true/false values)
- `in` = Input stream (data coming in)
- `out` = Output stream (data going out)
- `file("...")` = Where the data is stored

### Part 2: The Recurrence Relation (The Logic)

```tau
# The formula that defines behavior
r (action[t] = sensor_a[t] & sensor_b[t])
```

**Key concepts:**
- `r` = Recurrence relation statement (defines stream values over time)
- `[t]` = Current time step
- `[t-1]` = Previous time step (for memory/state)
- `&` = AND operator
- `|` = OR operator
- `^` = XOR operator (exclusive or)
- `'` = NOT operator (negation)

### Part 3: Initial Conditions (Optional)

```tau
# Set initial state at t=0
r ((action[t] = sensor_a[t] & sensor_b[t]) && (action[0] = 0))
```

### Complete Example

```tau
# Stream declarations
sensor_a:sbf = in file("inputs/sensor_a.in")
sensor_b:sbf = in file("inputs/sensor_b.in")
safe_action:sbf = out file("outputs/safe_action.out")

# Logic: Act only when BOTH sensors agree
r (safe_action[t] = sensor_a[t] & sensor_b[t])
```

---

## 3. Reading Logic Formulas

### The Four Basic Operators

| Symbol | Name | Formal | Casual | Example |
|--------|------|--------|--------|---------|
| `&` | AND | Conjunction | "both must be true" | `a & b` = 1 only when a=1 AND b=1 |
| `\|` | OR | Disjunction | "either can be true" | `a \| b` = 1 when a=1 OR b=1 |
| `^` | XOR | Exclusive Or | "exactly one is true" | `a ^ b` = 1 when a≠b |
| `'` | NOT | Negation | "flip the value" | `a'` = 1 when a=0 |

### Truth Tables (Memorize These!)

**AND (`&`)**
```
a | b | a & b
--|---|------
0 | 0 |   0
0 | 1 |   0
1 | 0 |   0
1 | 1 |   1    ← Only 1 when BOTH are 1
```

**OR (`|`)**
```
a | b | a | b
--|---|------
0 | 0 |   0
0 | 1 |   1
1 | 0 |   1
1 | 1 |   1    ← 1 when EITHER is 1
```

**XOR (`^`)**
```
a | b | a ^ b
--|---|------
0 | 0 |   0
0 | 1 |   1
1 | 0 |   1
1 | 1 |   0    ← 1 when DIFFERENT
```

**NOT (`'`)**
```
a | a'
--|----
0 |  1    ← Flip!
1 |  0
```

### Compound Patterns

These are combinations you'll use frequently:

| Pattern | Formula | Meaning | When Output = 1 |
|---------|---------|---------|-----------------|
| **NAND** | `(a & b)'` | NOT(both true) | At most one is true |
| **NOR** | `(a \| b)'` | Neither true | Both are false |
| **XNOR** | `(a ^ b)'` | Same value | a equals b |
| **Implication** | `a' \| b` | If a then b | a=0 OR b=1 |
| **AND-NOT** | `a & b'` | a but not b | a=1 AND b=0 |

### Reading Formulas: Formal vs Casual

**Formula:** `safe[t] = (agent_a[t] & agent_b[t])'`

**Formal reading:**
> "The output 'safe' at time t equals the negation of the conjunction of agent_a at time t and agent_b at time t."

**Casual reading:**
> "We're safe when NOT both agents are active at the same time."

**Formula:** `allowed[t] = action[t]' | trusted[t]`

**Formal reading:**
> "The output 'allowed' at time t equals the disjunction of the negation of action at time t and trusted at time t."

**Casual reading:**
> "Action is allowed if: either we're not taking action, OR we're trusted. In other words: if we act, we must be trusted."

---

## 4. Bitvectors: Arithmetic and Comparisons

So far we've only covered **boolean** (true/false) logic. But Tau also supports **bitvectors** - fixed-width integers that enable arithmetic, weighted voting, and time-based logic.

### What Are Bitvectors?

A **bitvector** is a fixed-width binary number. Think of it as an integer with a specific bit size.

| Type | Bits | Range | Example Values |
|------|------|-------|----------------|
| `bv[8]` | 8 | 0 to 255 | Weights, small counts |
| `bv[16]` | 16 | 0 to 65,535 | Timestamps, medium values |
| `bv[32]` | 32 | 0 to 4 billion | Large values, addresses |
| `bv[64]` | 64 | 0 to 18 quintillion | Large integers, hashes |
| `bv[128]` | 128 | Very large | UUIDs, cryptographic values |
| `bv[256]` | 256 | Huge | Hash outputs (SHA-256) |

> **Note:** Tau uses the cvc5 SMT solver for bitvector operations. The default (when no size is specified) is 16 bits.

### ⚠️ Performance Warning: State Space Explosion

While Tau *syntactically* supports arbitrary bitvector widths, **practical limits exist** due to cvc5's bit-blasting approach:

| Width | Arithmetic | Comparisons | Bitwise (AND/OR/XOR) |
|-------|------------|-------------|----------------------|
| **8-16 bit** | ✅ Fast | ✅ Fast | ✅ Fast |
| **32 bit** | ✅ OK | ✅ OK | ✅ Fast |
| **64 bit** | ⚠️ Slow | ⚠️ Moderate | ✅ OK |
| **128 bit** | ❌ Very slow | ⚠️ Slow | ⚠️ Moderate |
| **256 bit** | ❌ Timeout likely | ❌ Slow | ⚠️ May work |
| **512 bit** | ❌ Not practical | ❌ Not practical | ❌ Risky |

**Why?** cvc5 converts bitvector operations to SAT problems (bit-blasting):
- **Addition**: Near-linear complexity - scales reasonably
- **Multiplication**: O(w²) gates for w-bit values - explodes quickly
- **Division/Modulo**: Even worse than multiplication
- **Comparisons**: Require full bit-width evaluation
- **Bitwise ops**: Most efficient, but still grow with width

### What This Means for Cryptography

**256-bit hash comparisons (SHA-256) are NOT practical** for verification:
- Simple equality checks may timeout
- Arithmetic on hashes will fail
- Use hashes as **opaque identifiers** only (compare externally, pass result as boolean)

**Practical approach for cryptographic verification:**
```tau
# DON'T do this - will timeout or explode
# hash_a:bv[256] = in file("inputs/hash_a.in")
# hash_b:bv[256] = in file("inputs/hash_b.in")  
# r match[t] = (hash_a[t] - hash_b[t]) = {0}:bv[256]  # BAD!

# DO this instead - compute comparison externally, pass boolean result
hashes_match:sbf = in file("inputs/hashes_match.in")  # Pre-computed
action_allowed:sbf = out file("outputs/allowed.out")
r (action_allowed[t] = hashes_match[t] & other_condition[t])  # GOOD!
```

### Recommended Bitvector Sizes

| Use Case | Recommended Width | Why |
|----------|-------------------|-----|
| Weights/votes | `bv[8]` | 0-255 sufficient |
| Timestamps | `bv[16]` or `bv[32]` | Fits most ranges |
| Counters | `bv[16]` | Rarely need more |
| Time locks | `bv[16]` | Keep arithmetic simple |
| Scores/balances | `bv[32]` | Max practical for math |

**Rule of thumb:** Keep bitvectors ≤32 bits for anything involving arithmetic. For larger values, compute externally and pass boolean results to Tau.

### The Hybrid Architecture: Crypto + Tau

Since Tau can't handle cryptographic-sized bitvectors efficiently, use a **hybrid architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR SYSTEM                               │
├─────────────────────────────┬───────────────────────────────────┤
│     EXTERNAL LAYER          │         TAU LAYER                 │
│     (Python/Rust/etc)       │         (Formal Logic)            │
├─────────────────────────────┼───────────────────────────────────┤
│                             │                                   │
│  ┌─────────────────────┐    │    ┌─────────────────────────┐   │
│  │ Compute SHA-256     │────┼───►│ hash_valid:sbf          │   │
│  │ hash(data) == expected   │    │                         │   │
│  └─────────────────────┘    │    │                         │   │
│                             │    │  Tau Specification:     │   │
│  ┌─────────────────────┐    │    │                         │   │
│  │ Verify Ed25519      │────┼───►│ sig_valid:sbf           │   │
│  │ signature           │    │    │                         │   │
│  └─────────────────────┘    │    │  r action_allowed[t] =  │   │
│                             │    │    hash_valid[t] &      │   │
│  ┌─────────────────────┐    │    │    sig_valid[t] &       │   │
│  │ Verify Merkle       │────┼───►│ proof_valid:sbf         │   │
│  │ inclusion proof     │    │    │    proof_valid[t]       │   │
│  └─────────────────────┘    │    │                         │   │
│                             │    └─────────────────────────┘   │
│  256-512 bit operations     │    Boolean logic only            │
│  (unlimited complexity)     │    (fast, verifiable)            │
└─────────────────────────────┴───────────────────────────────────┘
```

**Why this works:**
1. **External layer** handles computationally expensive crypto (unlimited bit widths)
2. **Tau layer** handles safety/coordination logic (boolean results only)
3. Tau verifies the **relationships** between results, not the crypto itself

### Example: Secure Transaction Authorization

**Goal:** Only authorize a transaction if:
- The sender's signature is valid
- The sender has sufficient balance
- The recipient is not blacklisted

**External code (Python):**
```python
# Compute crypto externally - Tau can't do this efficiently
sig_valid = verify_ed25519(signature, message, public_key)  # 256-bit
balance_ok = sender_balance >= amount  # Could be large integers
not_blacklisted = recipient not in blacklist

# Write boolean results for Tau
write_input("inputs/sig_valid.in", "1" if sig_valid else "0")
write_input("inputs/balance_ok.in", "1" if balance_ok else "0")
write_input("inputs/not_blacklisted.in", "1" if not_blacklisted else "0")
```

**Tau specification:**
```tau
# TRANSACTION AUTHORIZATION
# Tau handles the LOGIC, not the crypto

# Inputs: pre-computed boolean results from external layer
sig_valid:sbf = in file("inputs/sig_valid.in")
balance_ok:sbf = in file("inputs/balance_ok.in")
not_blacklisted:sbf = in file("inputs/not_blacklisted.in")

# Output: authorization decision
tx_authorized:sbf = out file("outputs/tx_authorized.out")

# Invariant: All three conditions must be true
# This is what Tau excels at - boolean logic and safety invariants
r (tx_authorized[t] = sig_valid[t] & balance_ok[t] & not_blacklisted[t])
```

### What Tau Adds to This Architecture

You might ask: "If crypto is computed externally, why use Tau at all?"

**Tau provides:**

1. **Formal verification** - The logic is mathematically proven correct
2. **Temporal reasoning** - Handle sequences of events over time
3. **Composition** - Combine multiple safety conditions reliably
4. **Auditability** - The spec is readable and verifiable by humans
5. **Determinism** - Same inputs always produce same outputs

**Example: Multi-step authorization with temporal logic**
```tau
# More complex: require signature valid AND no recent failed attempts

sig_valid:sbf = in file("inputs/sig_valid.in")
failed_attempt:sbf = in file("inputs/failed_attempt.in")
tx_authorized:sbf = out file("outputs/tx_authorized.out")

# Temporal state: track if there was a recent failure
recent_failure:sbf = out file("outputs/recent_failure.out")

# Recent failure persists for a while (simplified)
recent_failure[0] := 0
r (recent_failure[t] = failed_attempt[t] | recent_failure[t-1])

# Only authorize if sig valid AND no recent failures
r (tx_authorized[t] = sig_valid[t] & recent_failure[t]')
```

This temporal logic would be error-prone in regular code but is **provably correct** in Tau.

### Summary: When to Use What

| Task | Where to Compute | Why |
|------|------------------|-----|
| Hash computation | External | 256+ bits, complex |
| Signature verification | External | 256+ bits, complex |
| Merkle proof verification | External | Variable size trees |
| Balance comparisons | External if large | May exceed 32 bits |
| **AND/OR of results** | **Tau** | Boolean logic |
| **Temporal sequences** | **Tau** | State over time |
| **Safety invariants** | **Tau** | Formal verification |
| **Multi-condition gates** | **Tau** | Composition |

### Declaring Bitvector Streams

```tau
# Boolean stream (true/false)
sensor:sbf = in file("inputs/sensor.in")

# 8-bit bitvector stream (0-255)
weight:bv[8] = in file("inputs/weight.in")

# 16-bit bitvector stream
timestamp:bv[16] = in file("inputs/timestamp.in")
```

### Bitvector Arithmetic

Tau supports standard arithmetic operations **directly in recurrence relations**:

| Operator | Meaning | Example |
|----------|---------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Modulo | `a % b` |

**Example: Sum two values**
```tau
value_a:bv[8] = in file("inputs/a.in")
value_b:bv[8] = in file("inputs/b.in")
sum:bv[8] = out file("outputs/sum.out")

# Direct addition
r sum[t] = value_a[t] + value_b[t]
```

### Bitvector Comparisons

Comparisons return boolean (sbf) results:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equal | `a = b` |
| `!=` | Not equal | `a != b` |
| `<` | Less than | `a < b` |
| `<=` | Less or equal | `a <= b` |
| `>` | Greater than | `a > b` |
| `>=` | Greater or equal | `a >= b` |

**Example: Threshold check**
```tau
value:bv[8] = in file("inputs/value.in")
above_threshold:sbf = out file("outputs/above.out")

# Compare to constant threshold (written as {10}:bv[8])
r (above_threshold[t] = (value[t] >= {10}:bv[8]))
```

### Bitvector Literals

To write a constant bitvector value, use this syntax:

```
{value}:bv[width]
```

**Examples:**
- `{0}:bv[8]` - Zero as 8-bit value
- `{255}:bv[8]` - Max 8-bit value
- `{1000}:bv[16]` - 1000 as 16-bit value

### The Ternary Operator

The **ternary operator** converts booleans to bitvectors:

```
(condition ? value_if_true : value_if_false)
```

**Example: Convert vote to weight**
```tau
vote:sbf = in file("inputs/vote.in")
weighted:bv[8] = out file("outputs/weighted.out")

# If vote=1, output weight of 5; if vote=0, output 0
r weighted[t] = (vote[t] ? {5}:bv[8] : {0}:bv[8])
```

This is **crucial** for weighted voting - it converts boolean votes to numeric weights!

### Pattern: Weighted Vote

**The Problem:** Three agents vote, but their votes have different weights. Agent A has weight 3, Agent B has weight 2, Agent C has weight 1. Decision passes if total weight ≥ 4.

**The Solution:**
```tau
# Boolean votes from each agent
vote_a:sbf = in file("inputs/vote_a.in")
vote_b:sbf = in file("inputs/vote_b.in")
vote_c:sbf = in file("inputs/vote_c.in")
decision:sbf = out file("outputs/decision.out")

# Convert votes to weights, sum them, compare to threshold
# vote_a (weight 3) + vote_b (weight 2) + vote_c (weight 1) >= 4
r (decision[t] = (
    (
        (vote_a[t] ? {3}:bv[8] : {0}:bv[8]) +
        (vote_b[t] ? {2}:bv[8] : {0}:bv[8]) +
        (vote_c[t] ? {1}:bv[8] : {0}:bv[8])
    ) >= {4}:bv[8]
))
```

**How it works:**
1. Each vote is converted to its weight using ternary: `(vote ? weight : 0)`
2. Weights are summed using bitvector addition
3. Sum is compared to threshold using `>=`
4. Result is boolean: pass or fail

**Example scenarios:**
- A=1, B=1, C=0 → 3+2+0 = 5 ≥ 4 → **PASS**
- A=1, B=0, C=1 → 3+0+1 = 4 ≥ 4 → **PASS**
- A=0, B=1, C=1 → 0+2+1 = 3 < 4 → **FAIL**
- A=1, B=0, C=0 → 3+0+0 = 3 < 4 → **FAIL**

### Pattern: Time Lock

**The Problem:** An action is locked until a certain time. Check if the lock is still active.

**The Solution:**
```tau
# All inputs are 16-bit timestamps
lock_start:bv[16] = in file("inputs/lock_start.in")
lock_duration:bv[16] = in file("inputs/lock_duration.in")
current_time:bv[16] = in file("inputs/current_time.in")
lock_active:sbf = out file("outputs/lock_active.out")

# Lock is active if: current_time < (lock_start + lock_duration)
# IMPORTANT: Tau/cvc5 uses UNSIGNED bitvector comparisons.
# Using ((end_time - current_time) > 0) breaks under wrap-around.
r (lock_active[t] = (current_time[t] < (lock_start[t] + lock_duration[t])))
```

**How it works:**
1. Calculate unlock time: `lock_start + lock_duration`
2. Calculate remaining: `unlock_time - current_time`
3. Lock active if remaining > 0

**Example:**
- lock_start=100, duration=50, current=120 → 120 < 150 → **LOCKED**
- lock_start=100, duration=50, current=160 → 160 < 150 is false → **UNLOCKED**

### Why Bitvectors Matter

| Without Bitvectors | With Bitvectors |
|--------------------|-----------------|
| Only boolean (0/1) | Full integers |
| No weighted voting | Weighted voting ✅ |
| No time arithmetic | Time locks ✅ |
| Complex encodings | Direct arithmetic ✅ |

### Quick Reference

```tau
# Types
value:sbf = ...           # Boolean (0 or 1)
value:bv[8] = ...         # 8-bit integer (0-255)
value:bv[16] = ...        # 16-bit integer (0-65535)

# Literals
{42}:bv[8]                # Constant 42 as 8-bit
{1000}:bv[16]             # Constant 1000 as 16-bit

# Ternary (boolean to bitvector)
(condition ? {true_val}:bv[8] : {false_val}:bv[8])

# Arithmetic
a + b, a - b, a * b, a / b, a % b

# Comparisons (return boolean)
a = b, a != b, a < b, a <= b, a > b, a >= b
```

---

## 5. Writing Your First Specification

### Step 1: Identify Your Invariant

**Goal:** Build a safety gate that only allows actions when conditions are safe.

**Invariant:** "Actions are only permitted when the safety flag is true"

**Formal:** `G(action → safe)` (Globally, action implies safe)

### Step 2: Choose Your Streams

```
INPUTS:
- proposed_action: The action the agent wants to take (0 or 1)
- is_safe: Whether the environment is safe (0 or 1)

OUTPUT:
- allowed_action: The action that actually gets executed (0 or 1)
```

### Step 3: Write the Logic

The invariant "action implies safe" means:
- If proposed_action=1 AND is_safe=1 → allowed_action=1
- If proposed_action=1 AND is_safe=0 → allowed_action=0 (blocked!)
- If proposed_action=0 → allowed_action=0

This is simply: `allowed_action = proposed_action AND is_safe`

### Step 4: Write the Complete Spec

```tau
# SAFETY GATE SPECIFICATION
# Invariant: Actions only allowed when safe
# Formula: allowed = action AND safe

proposed_action:sbf = in file("inputs/proposed_action.in")
is_safe:sbf = in file("inputs/is_safe.in")
allowed_action:sbf = out file("outputs/allowed_action.out")

# Gate the action through the safety check
r (allowed_action[t] = proposed_action[t] & is_safe[t])
```

### Step 5: Test It

Create input files and verify the output matches your invariant:

```
proposed_action.in:    is_safe.in:    Expected output:
1                      1              1 (safe, action allowed)
1                      0              0 (unsafe, action BLOCKED)
0                      1              0 (no action proposed)
0                      0              0 (no action proposed)
```

---

## 6. Tutorial Index: 28 Specifications

Click on any specification to learn how to read, write, and use it.

### Safety Patterns (5)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 1 | [safety_gate](#tutorial-1-safety_gate) | Block unsafe actions | ⭐ Beginner |
| 2 | [mutual_exclusion](#tutorial-2-mutual_exclusion) | Only one agent active | ⭐ Beginner |
| 3 | [never_unsafe](#tutorial-3-never_unsafe) | Prevent forbidden states | ⭐ Beginner |
| 4 | [risk_gate](#tutorial-4-risk_gate) | Block high-risk actions | ⭐⭐ Intermediate |
| 5 | [belief_consistency](#tutorial-5-belief_consistency) | No contradictory beliefs | ⭐⭐ Intermediate |

### Decision Patterns (5)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 6 | [confidence_gate](#tutorial-6-confidence_gate) | Act only when confident | ⭐ Beginner |
| 7 | [multiplexer](#tutorial-7-multiplexer) | Select between options | ⭐⭐ Intermediate |
| 8 | [policy_switch](#tutorial-8-policy_switch) | Switch strategies | ⭐⭐ Intermediate |
| 9 | [exploration_decay](#tutorial-9-exploration_decay) | Explore less over time | ⭐⭐ Intermediate |
| 10 | [utility_alignment](#tutorial-10-utility_alignment) | Verify preference alignment | ⭐⭐⭐ Advanced |

### Coordination Patterns (5)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 11 | [consensus_check](#tutorial-11-consensus_check) | All agents agree | ⭐ Beginner |
| 12 | [majority_vote](#tutorial-12-majority_vote) | Majority decides | ⭐⭐ Intermediate |
| 13 | [sync_barrier](#tutorial-13-sync_barrier) | Wait for all ready | ⭐⭐ Intermediate |
| 14 | [turn_gate](#tutorial-14-turn_gate) | Enforce turn order | ⭐ Beginner |
| 15 | [fair_priority](#tutorial-15-fair_priority) | Resolve conflicts fairly | ⭐⭐⭐ Advanced |

### Progress Patterns (5)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 16 | [progress](#tutorial-16-progress) | System advances toward goal | ⭐⭐ Intermediate |
| 17 | [no_starvation](#tutorial-17-no_starvation) | Every request gets served | ⭐⭐ Intermediate |
| 18 | [request_response](#tutorial-18-request_response) | Requests get responses | ⭐⭐ Intermediate |
| 19 | [recovery](#tutorial-19-recovery) | System recovers from failure | ⭐⭐⭐ Advanced |
| 20 | [bounded_until](#tutorial-20-bounded_until) | Condition holds until goal | ⭐⭐⭐ Advanced |

### Detection Patterns (4)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 21 | [emergent_detector](#tutorial-21-emergent_detector) | Detect unexpected behavior | ⭐ Beginner |
| 22 | [anomaly_detector](#tutorial-22-anomaly_detector) | Detect anomalies | ⭐ Beginner |
| 23 | [edge_detector](#tutorial-23-edge_detector) | Detect signal changes | ⭐⭐ Intermediate |
| 24 | [collision_detect](#tutorial-24-collision_detect) | Detect conflicts | ⭐ Beginner |

### Trust Patterns (2)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 25 | [trust_update](#tutorial-25-trust_update) | Track reputation | ⭐⭐⭐ Advanced |
| 26 | [reputation_gate](#tutorial-26-reputation_gate) | Gate by trust level | ⭐⭐ Intermediate |

### Bitvector Patterns (2)

| # | Pattern | Use Case | Difficulty |
|---|---------|----------|------------|
| 27 | [weighted_vote](#tutorial-27-weighted_vote) | Stake-weighted voting | ⭐⭐ Intermediate |
| 28 | [time_lock](#tutorial-28-time_lock) | Time-based locking | ⭐⭐ Intermediate |

---

## Tutorial 1: safety_gate

### What It Does
Gates an action through a safety check. The action only passes through if the safety condition is met.

### The Invariant
**Formal:** `G(action → safe)` - Globally, if action then safe
**Casual:** "We only act when it's safe to do so"

### The Logic
```
allowed = proposed_action AND is_safe
```

If you propose an action (1) but it's not safe (0), the output is 0 (blocked).

### Human-Readable Specification
```tau
# SAFETY GATE
# Invariant: Actions only allowed when environment is safe

proposed_action:sbf = in file("inputs/proposed_action.in")
is_safe:sbf = in file("inputs/is_safe.in")
allowed_action:sbf = out file("outputs/allowed_action.out")

# AND gate: both must be true
# action passes through ONLY when safe=1
r (allowed_action[t] = proposed_action[t] & is_safe[t])
```

### Truth Table
```
proposed_action | is_safe | allowed_action
----------------|---------|---------------
       0        |    0    |       0
       0        |    1    |       0
       1        |    0    |       0  ← BLOCKED!
       1        |    1    |       1  ← Allowed
```

### How to Read It

**Formal:** "The allowed action at time t equals the conjunction of the proposed action at time t and the safety flag at time t."

**Casual:** "We're allowed to act when we want to act AND it's safe."

### When to Use It
- Gating agent actions through safety checks
- Enforcing preconditions before operations
- Human-in-the-loop approval systems

### Python Oracle
```python
def safety_gate(proposed_action, is_safe):
    return proposed_action & is_safe
```

---

## Tutorial 2: mutual_exclusion

### What It Does
Ensures at most one agent can be in a critical state at any time. This prevents conflicts and race conditions.

### The Invariant
**Formal:** `G(¬(agent_a ∧ agent_b))` - Globally, never both active
**Casual:** "Only one agent can hold the resource at a time"

### The Logic
```
safe = NOT(agent_a AND agent_b)
```

This is a NAND gate. Output is 1 (safe) unless BOTH agents are active.

### Human-Readable Specification
```tau
# MUTUAL EXCLUSION
# Invariant: At most one agent in critical section

agent_a_active:sbf = in file("inputs/agent_a.in")
agent_b_active:sbf = in file("inputs/agent_b.in")
is_safe:sbf = out file("outputs/is_safe.out")

# NAND gate: safe = NOT(both active)
# The ' negates the entire (a & b) expression
# Safe when: neither active, or exactly one active
# Unsafe when: BOTH active (output = 0)
r is_safe[t] = (agent_a_active[t] & agent_b_active[t])'
```

### Truth Table
```
agent_a | agent_b | is_safe
--------|---------|--------
   0    |    0    |    1   ← Neither active: SAFE
   0    |    1    |    1   ← Only B active: SAFE
   1    |    0    |    1   ← Only A active: SAFE
   1    |    1    |    0   ← BOTH active: UNSAFE!
```

### How to Read It

**The `'` operator:** When you see `(...)'`, read it as "NOT(everything inside)".

**Formal:** "Safe at time t equals the negation of the conjunction of agent A and agent B."

**Casual:** "We're safe when it's NOT the case that both agents are active."

### When to Use It
- Resource locking
- Critical section protection
- Preventing concurrent access
- Detecting conflicts

### Python Oracle
```python
def mutual_exclusion(agent_a, agent_b):
    return 1 - (agent_a & agent_b)  # NAND
```

---

## Tutorial 3: never_unsafe

### What It Does
Inverts an unsafe signal to produce a safe signal. Simplest safety pattern.

### The Invariant
**Formal:** `G(¬unsafe)` - Globally, not unsafe
**Casual:** "The unsafe condition is never true"

### The Logic
```
is_safe = NOT(unsafe_condition)
```

### Human-Readable Specification
```tau
# NEVER UNSAFE
# Invariant: Forbidden state never occurs

unsafe_condition:sbf = in file("inputs/unsafe.in")
is_safe:sbf = out file("outputs/is_safe.out")

# Simple NOT: invert the unsafe signal
# When unsafe=1, is_safe=0 (system detects danger)
# When unsafe=0, is_safe=1 (all clear)
r is_safe[t] = unsafe_condition[t]'
```

### Truth Table
```
unsafe | is_safe
-------|--------
   0   |    1   ← All clear
   1   |    0   ← DANGER!
```

### How to Read It
**Formal:** "Safe at time t equals the negation of the unsafe condition at time t."

**Casual:** "We're safe when the unsafe flag is off."

### When to Use It
- Inverting alarm signals
- Converting "bad" flags to "good" flags
- Safety monitoring

### Python Oracle
```python
def never_unsafe(unsafe):
    return 1 - unsafe  # NOT
```

---

## Tutorial 4: risk_gate

### What It Does
Blocks actions when risk is too high. Like a safety gate but specifically for risk management.

### The Invariant
**Formal:** `G(action → ¬high_risk)` - Globally, action implies not high risk
**Casual:** "We only act when risk is low"

### The Logic
```
safe_action = action AND NOT(high_risk)
```

This is an AND-NOT gate. The action passes through only when risk is LOW (0).

### Human-Readable Specification
```tau
# RISK GATE
# Invariant: Actions blocked when risk is high

proposed_action:sbf = in file("inputs/action.in")
high_risk:sbf = in file("inputs/high_risk.in")
safe_action:sbf = out file("outputs/safe_action.out")

# AND-NOT gate: action AND NOT(risk)
# The ' on high_risk inverts it: high_risk' means "low risk"
# Action passes through only when high_risk=0
# Truth: (1,0)->1, (1,1)->0, (0,x)->0
r safe_action[t] = proposed_action[t] & high_risk[t]'
```

### Truth Table
```
action | high_risk | safe_action
-------|-----------|------------
   0   |     0     |      0     ← No action proposed
   0   |     1     |      0     ← No action proposed
   1   |     0     |      1     ← Low risk: ALLOWED
   1   |     1     |      0     ← High risk: BLOCKED!
```

### How to Read It

**The AND-NOT pattern:** `a & b'` means "a but not b" or "a when b is false".

**Formal:** "Safe action at time t equals the conjunction of the proposed action and the negation of high risk."

**Casual:** "We can act when we want to AND the risk isn't high."

### When to Use It
- Risk management systems
- Trading agents with risk limits
- Any action that should be blocked in dangerous conditions

### Python Oracle
```python
def risk_gate(action, high_risk):
    return action & (1 - high_risk)  # AND-NOT
```

---

## Tutorial 5: belief_consistency

### What It Does
Checks that an agent's beliefs don't contradict each other. If belief_a is true, belief_not_a shouldn't also be true.

### The Invariant
**Formal:** `G(¬(belief ∧ ¬belief))` - No contradictions
**Casual:** "You can't believe something and its opposite at the same time"

### The Logic
```
consistent = NOT(belief AND negated_belief)
```

Same as mutual_exclusion - a NAND gate.

### Human-Readable Specification
```tau
# BELIEF CONSISTENCY
# Invariant: Beliefs don't contradict

believes_rain:sbf = in file("inputs/believes_rain.in")
believes_no_rain:sbf = in file("inputs/believes_no_rain.in")
consistent:sbf = out file("outputs/consistent.out")

# NAND: consistent when NOT(both beliefs true)
# If agent believes "rain" AND "no rain" simultaneously,
# that's a contradiction → consistent=0
r consistent[t] = (believes_rain[t] & believes_no_rain[t])'
```

### Truth Table
```
believes_rain | believes_no_rain | consistent
--------------|------------------|------------
      0       |        0         |     1      ← No strong belief: OK
      0       |        1         |     1      ← Believes no rain: OK
      1       |        0         |     1      ← Believes rain: OK
      1       |        1         |     0      ← CONTRADICTION!
```

### How to Read It
**Casual:** "The agent's beliefs are consistent unless it believes both P and NOT P."

### When to Use It
- BDI (Belief-Desire-Intention) agent architectures
- Knowledge base consistency checking
- Detecting conflicting information

### Python Oracle
```python
def belief_consistency(belief, negated_belief):
    return 1 - (belief & negated_belief)  # NAND
```

---

## Tutorial 6: confidence_gate

### What It Does
Gates an action through a confidence check. Only act when you're confident in your decision.

### The Invariant
**Formal:** `G(action → confident)` - Action implies confidence
**Casual:** "Only act when you're sure"

### The Logic
```
output = action AND confident
```

Simple AND gate - same as safety_gate but for confidence.

### Human-Readable Specification
```tau
# CONFIDENCE GATE
# Invariant: Only act when confident

proposed_action:sbf = in file("inputs/action.in")
is_confident:sbf = in file("inputs/confident.in")
final_action:sbf = out file("outputs/final_action.out")

# AND gate: both action and confidence required
r final_action[t] = proposed_action[t] & is_confident[t]
```

### When to Use It
- ML model inference (act only above confidence threshold)
- Decision-making under uncertainty
- Quality control in agent outputs

### Python Oracle
```python
def confidence_gate(action, confident):
    return action & confident
```

---

## Tutorial 7: multiplexer

### What It Does
Selects between two inputs based on a selector signal. Like a switch that routes one of two paths to the output.

### The Invariant
**Formal:** `G((sel → out=a) ∧ (¬sel → out=b))`
**Casual:** "When selector is 1, use input A; when 0, use input B"

### The Logic
```
output = (selector AND input_a) OR (NOT(selector) AND input_b)
```

### Human-Readable Specification
```tau
# MULTIPLEXER (2-to-1 MUX)
# Invariant: Output follows selected input

input_a:sbf = in file("inputs/input_a.in")
input_b:sbf = in file("inputs/input_b.in")
selector:sbf = in file("inputs/selector.in")
output:sbf = out file("outputs/output.out")

# MUX logic: 
# When selector=1: output = input_a
# When selector=0: output = input_b
# Formula: (sel & a) | (sel' & b)
r output[t] = (selector[t] & input_a[t]) | (selector[t]' & input_b[t])
```

### Truth Table (selector=1 means choose A)
```
sel | a | b | output
----|---|---|-------
 0  | 0 | 0 |   0    ← sel=0, choose B (which is 0)
 0  | 0 | 1 |   1    ← sel=0, choose B (which is 1)
 0  | 1 | 0 |   0    ← sel=0, choose B (which is 0)
 0  | 1 | 1 |   1    ← sel=0, choose B (which is 1)
 1  | 0 | 0 |   0    ← sel=1, choose A (which is 0)
 1  | 0 | 1 |   0    ← sel=1, choose A (which is 0)
 1  | 1 | 0 |   1    ← sel=1, choose A (which is 1)
 1  | 1 | 1 |   1    ← sel=1, choose A (which is 1)
```

### When to Use It
- Switching between strategies
- A/B selection
- Mode-dependent behavior

### Python Oracle
```python
def multiplexer(input_a, input_b, selector):
    return (selector & input_a) | ((1-selector) & input_b)
```

---

## Tutorial 8: policy_switch

### What It Does
Switches between two policies based on a mode flag. Same as multiplexer but with semantic meaning.

### Human-Readable Specification
```tau
# POLICY SWITCH
# Switch between aggressive and defensive policies

aggressive_action:sbf = in file("inputs/aggressive.in")
defensive_action:sbf = in file("inputs/defensive.in")
be_aggressive:sbf = in file("inputs/mode.in")
final_action:sbf = out file("outputs/action.out")

# When be_aggressive=1: use aggressive policy
# When be_aggressive=0: use defensive policy
r final_action[t] = (be_aggressive[t] & aggressive_action[t]) | (be_aggressive[t]' & defensive_action[t])
```

### When to Use It
- Adaptive agents that change strategy
- Market regime detection (bull/bear markets)
- Game AI (offense/defense modes)

---

## Tutorial 9: exploration_decay

### What It Does
Models exploration that decreases as experience accumulates. Early on, explore more; later, exploit knowledge.

### The Invariant
**Formal:** `F G(¬exploring)` - Eventually, always not exploring
**Casual:** "Eventually, we stop exploring and just exploit"

### Human-Readable Specification
```tau
# EXPLORATION DECAY
# Invariant: Exploration decreases with experience

explore_trigger:sbf = in file("inputs/trigger.in")
has_experience:sbf = in file("inputs/experienced.in")
should_explore:sbf = out file("outputs/explore.out")

# AND-NOT: explore when triggered AND NOT experienced
# As experience grows (has_experience=1), exploration stops
# Models epsilon-greedy decay in reinforcement learning
r should_explore[t] = explore_trigger[t] & has_experience[t]'
```

### Truth Table
```
trigger | experienced | should_explore
--------|-------------|---------------
   0    |      0      |       0        ← No trigger
   0    |      1      |       0        ← No trigger
   1    |      0      |       1        ← Explore! (new territory)
   1    |      1      |       0        ← Don't explore (we know this)
```

### When to Use It
- Reinforcement learning agents
- Bandit algorithms
- Any learning system that should settle on good solutions

### Python Oracle
```python
def exploration_decay(trigger, experienced):
    return trigger & (1 - experienced)
```

---

## Tutorial 10: utility_alignment

### What It Does
Verifies that agent actions align with user preferences. Critical for AI safety.

### The Invariant
**Formal:** `G(action → aligned)` - Actions imply alignment
**Casual:** "Every action the agent takes must align with what we want"

### The Logic
```
valid = NOT(action) OR aligned
```

This is **implication** - if you're not acting, you're fine; if you are acting, you must be aligned.

### Human-Readable Specification
```tau
# UTILITY ALIGNMENT (AI Safety Pattern)
# Invariant: Agent actions match human preferences

agent_action:sbf = in file("inputs/action.in")
is_aligned:sbf = in file("inputs/aligned.in")
action_valid:sbf = out file("outputs/valid.out")

# IMPLICATION: action -> aligned
# Written as: NOT(action) OR aligned
# Fails (output=0) only when action=1 but aligned=0
# This catches misaligned actions!
r action_valid[t] = agent_action[t]' | is_aligned[t]
```

### Truth Table
```
action | aligned | action_valid
-------|---------|-------------
   0   |    0    |      1      ← No action: automatically valid
   0   |    1    |      1      ← No action: automatically valid
   1   |    0    |      0      ← MISALIGNED ACTION!
   1   |    1    |      1      ← Action is aligned: valid
```

### How to Read Implication

The formula `a' | b` reads as "NOT a OR b", which is equivalent to "a IMPLIES b".

**Why?** Implication `a → b` is false ONLY when a is true but b is false. Check the truth table - that's exactly what `a' | b` produces!

### When to Use It
- AI safety verification
- Preference learning validation
- Ethical AI constraints

### Python Oracle
```python
def utility_alignment(action, aligned):
    return (1 - action) | aligned  # Implication
```

---

## Tutorial 11: consensus_check

### What It Does
Checks if multiple agents have converged to the same value (agreement).

### The Invariant
**Formal:** `F G(∀i,j: vᵢ = vⱼ)` - Eventually, all values equal
**Casual:** "All agents eventually agree"

### The Logic
```
in_consensus = NOT(value_a XOR value_b)
```

XOR gives 1 when values differ. NOT(XOR) gives 1 when values are the SAME.

### Human-Readable Specification
```tau
# CONSENSUS CHECK
# Invariant: Detect when all agents agree

agent_a_value:sbf = in file("inputs/value_a.in")
agent_b_value:sbf = in file("inputs/value_b.in")
in_consensus:sbf = out file("outputs/consensus.out")

# XNOR: equality check
# XOR gives 1 when different, so NOT(XOR) gives 1 when SAME
# This is an equality test: output=1 when a=b
r in_consensus[t] = (agent_a_value[t] ^ agent_b_value[t])'
```

### Truth Table
```
value_a | value_b | in_consensus
--------|---------|-------------
   0    |    0    |      1      ← Both 0: CONSENSUS
   0    |    1    |      0      ← Different: no consensus
   1    |    0    |      0      ← Different: no consensus
   1    |    1    |      1      ← Both 1: CONSENSUS
```

### When to Use It
- Distributed consensus protocols
- Multi-agent agreement checking
- Convergence detection

### Python Oracle
```python
def consensus_check(value_a, value_b):
    return 1 - (value_a ^ value_b)  # XNOR
```

---

## Tutorial 12: majority_vote

### What It Does
Implements majority voting among 3 agents. Output is 1 when at least 2 of 3 vote yes.

### The Invariant
**Casual:** "The majority wins"

### The Logic
```
output = (a AND b) OR (b AND c) OR (a AND c)
```

At least 2 of 3 must be true.

### Human-Readable Specification
```tau
# MAJORITY VOTE (3 voters)
# Invariant: Output reflects majority opinion

vote_a:sbf = in file("inputs/vote_a.in")
vote_b:sbf = in file("inputs/vote_b.in")
vote_c:sbf = in file("inputs/vote_c.in")
majority:sbf = out file("outputs/majority.out")

# Majority = at least 2 of 3 agree
# Any pair agreeing on 1 gives output 1
r majority[t] = (vote_a[t] & vote_b[t]) | (vote_b[t] & vote_c[t]) | (vote_a[t] & vote_c[t])
```

### When to Use It
- Byzantine fault tolerance
- Committee decisions
- Ensemble learning

### Python Oracle
```python
def majority_vote(a, b, c):
    return (a & b) | (b & c) | (a & c)
```

---

## Tutorial 13: sync_barrier

### What It Does
Outputs 1 only when ALL agents are ready. A synchronization point.

### The Invariant
**Casual:** "We all proceed together or not at all"

### Human-Readable Specification
```tau
# SYNC BARRIER
# Invariant: All must be ready to proceed

agent_a_ready:sbf = in file("inputs/a_ready.in")
agent_b_ready:sbf = in file("inputs/b_ready.in")
all_ready:sbf = out file("outputs/all_ready.out")

# AND gate: all must be ready
r all_ready[t] = agent_a_ready[t] & agent_b_ready[t]
```

### When to Use It
- Distributed systems synchronization
- Multi-agent coordination
- Barrier synchronization patterns

---

## Tutorial 14: turn_gate

### What It Does
Only allows an action when it's the agent's turn.

### Human-Readable Specification
```tau
# TURN GATE
# Invariant: Can only act on your turn

proposed_action:sbf = in file("inputs/action.in")
my_turn:sbf = in file("inputs/my_turn.in")
allowed:sbf = out file("outputs/allowed.out")

# AND gate: action only when it's your turn
r allowed[t] = proposed_action[t] & my_turn[t]
```

---

## Tutorial 15: fair_priority

### What It Does
Resolves conflicts between two agents using a priority signal for deterministic tiebreaking.

### Human-Readable Specification
```tau
# FAIR PRIORITY
# Invariant: Conflicts resolved deterministically

action_a:sbf = in file("inputs/action_a.in")
action_b:sbf = in file("inputs/action_b.in")
a_has_priority:sbf = in file("inputs/priority.in")
winner:sbf = out file("outputs/winner.out")

# If both want to act: priority wins
# If only one wants to act: they win
# (a & b & priority) → a wins when both compete and a has priority
# (a & b') → a wins when only a acts
# (a' & b) → b wins when only b acts
r winner[t] = (action_a[t] & action_b[t] & a_has_priority[t]) | (action_a[t] & action_b[t]') | (action_a[t]' & action_b[t])
```

---

## Tutorial 16: progress

### What It Does
Checks that the system makes progress - if something is enabled, it eventually completes.

### The Invariant
**Formal:** `G(enabled → F done)` - Enabled implies eventually done
**Casual:** "If we start something, we finish it"

### Human-Readable Specification
```tau
# PROGRESS
# Invariant: Enabled tasks eventually complete

task_enabled:sbf = in file("inputs/enabled.in")
task_done:sbf = in file("inputs/done.in")
making_progress:sbf = out file("outputs/progress.out")

# IMPLICATION: enabled -> done
# Written as: NOT(enabled) OR done
# We're making progress if: not working on anything, OR we finished
r making_progress[t] = task_enabled[t]' | task_done[t]
```

---

## Tutorial 17: no_starvation

### What It Does
Ensures that if an agent requests a resource, it eventually gets it.

### The Invariant
**Formal:** `GF request → GF grant` - Infinitely often requesting implies infinitely often granted
**Casual:** "If you keep asking, you'll eventually get served"

### Human-Readable Specification
```tau
# NO STARVATION
# Invariant: Requests eventually granted

request:sbf = in file("inputs/request.in")
grant:sbf = in file("inputs/grant.in")
fair:sbf = out file("outputs/fair.out")

# IMPLICATION: request -> grant (at this step)
# fair=0 only when requesting but not granted
r fair[t] = request[t]' | grant[t]
```

---

## Tutorial 18: request_response

### What It Does
Tracks pending requests and outputs whether the system is responsive.

### Human-Readable Specification
```tau
# REQUEST-RESPONSE (with memory)
# Tracks whether requests are being served

request:sbf = in file("inputs/request.in")
response:sbf = in file("inputs/response.in")
pending:sbf = out file("outputs/pending.out")

# pending = (new request OR was pending) AND NOT responded
# Uses [t-1] for memory of previous state
r (pending[t] = (request[t] | pending[t-1]) & response[t]') && (pending[0] = 0)
```

---

## Tutorial 19: recovery

### What It Does
Tracks system health - if failure occurs, it must eventually recover.

### Human-Readable Specification
```tau
# RECOVERY
# Invariant: System recovers from failures

failure:sbf = in file("inputs/failure.in")
recovered:sbf = in file("inputs/recovered.in")
healthy:sbf = out file("outputs/healthy.out")

# healthy = (was healthy AND no failure) OR just recovered
# Uses temporal memory [t-1]
r (healthy[t] = (healthy[t-1] & failure[t]') | recovered[t]) && (healthy[0] = 1)
```

---

## Tutorial 20: bounded_until

### What It Does
Verifies that a condition holds until a goal is reached.

### The Invariant
**Formal:** `condition U goal` - Condition until goal
**Casual:** "Keep the lights on until we reach the destination"

### Human-Readable Specification
```tau
# BOUNDED UNTIL
# Invariant: Condition holds until goal reached

condition:sbf = in file("inputs/condition.in")
goal:sbf = in file("inputs/goal.in")
valid:sbf = out file("outputs/valid.out")

# valid = goal reached OR (condition holds AND was valid)
r (valid[t] = goal[t] | (condition[t] & valid[t-1])) && (valid[0] = 1)
```

---

## Tutorial 21: emergent_detector

### What It Does
Detects when actual behavior differs from expected behavior.

### The Invariant
**Casual:** "Alert when reality doesn't match expectations"

### Human-Readable Specification
```tau
# EMERGENT DETECTOR
# Invariant: Detect unexpected behavior

actual:sbf = in file("inputs/actual.in")
expected:sbf = in file("inputs/expected.in")
deviation:sbf = out file("outputs/deviation.out")

# XOR: output 1 when actual != expected
# Detects anomalies, emergent behavior, or bugs
r deviation[t] = actual[t] ^ expected[t]
```

---

## Tutorial 22: anomaly_detector

Same as emergent_detector - XOR detects differences.

---

## Tutorial 23: edge_detector

### What It Does
Detects when a signal changes from 0 to 1 (rising edge).

### Human-Readable Specification
```tau
# EDGE DETECTOR (Rising Edge)
# Invariant: Detect signal transitions

signal:sbf = in file("inputs/signal.in")
rising_edge:sbf = out file("outputs/edge.out")

# Rising edge = signal is 1 now AND was 0 before
r (rising_edge[t] = signal[t] & signal[t-1]') && (rising_edge[0] = 0)
```

---

## Tutorial 24: collision_detect

### What It Does
Detects when two agents are in the same state (collision).

### Human-Readable Specification
```tau
# COLLISION DETECT
# Invariant: Detect when agents collide

position_a:sbf = in file("inputs/pos_a.in")
position_b:sbf = in file("inputs/pos_b.in")
collision:sbf = out file("outputs/collision.out")

# XNOR: collision when positions are SAME
r collision[t] = (position_a[t] ^ position_b[t])'
```

---

## Tutorial 25: trust_update

### What It Does
Updates trust/reputation based on observed behavior.

### Human-Readable Specification
```tau
# TRUST UPDATE
# Invariant: Trust evolves based on behavior

good_behavior:sbf = in file("inputs/good.in")
bad_behavior:sbf = in file("inputs/bad.in")
trusted:sbf = out file("outputs/trusted.out")

# Trust = good behavior OR (was trusted AND no bad behavior)
# Good behavior immediately grants trust
# Bad behavior immediately revokes trust
r (trusted[t] = good_behavior[t] | (trusted[t-1] & bad_behavior[t]')) && (trusted[0] = 0)
```

---

## Tutorial 26: reputation_gate

### What It Does
Gates actions based on trust level.

### Human-Readable Specification
```tau
# REPUTATION GATE
# Invariant: Actions require trust

proposed_action:sbf = in file("inputs/action.in")
is_trusted:sbf = in file("inputs/trusted.in")
allowed:sbf = out file("outputs/allowed.out")

# AND gate: action AND trusted
r allowed[t] = proposed_action[t] & is_trusted[t]
```

---

## 7. Best Practices

### 1. Start with Invariants

Before writing any spec:
```
1. What must ALWAYS be true? (Safety)
2. What must EVENTUALLY happen? (Liveness)
3. What must be FAIR? (Fairness)
```

### 2. Name Streams Semantically

```tau
# BAD: What do these mean?
i0:sbf = in file("inputs/a.in")
i1:sbf = in file("inputs/b.in")

# GOOD: Clear meaning
emergency_stop:sbf = in file("inputs/emergency_stop.in")
speed_limit_ok:sbf = in file("inputs/speed_limit_ok.in")
```

### 3. Comment Complex Logic

```tau
# IMPLICATION: action -> safe
# Written as NOT(action) OR safe
# Fails only when action=1 but safe=0
r output[t] = action[t]' | safe[t]
```

### 4. Use Truth Tables to Verify

Before finalizing, write out the truth table and check every case.

### 5. Test Exhaustively

For N binary inputs, test all 2^N combinations.

### 6. Keep Specs Simple

If a spec is hard to understand, break it into smaller parts.

### 7. Document the Invariant

Every spec should have a clear statement of what invariant it enforces.

---

## Tutorial 27: weighted_vote

### What It Does
Implements stake-weighted voting where different voters have different weights. Decision passes if weighted sum meets threshold.

### The Invariant
**Casual:** "The decision reflects the weighted consensus of voters"

### Why Bitvectors?
Simple boolean voting treats all votes equally. Bitvectors let us assign different **weights** (stakes, reputation, expertise) to each voter.

### Human-Readable Specification
```tau
# WEIGHTED VOTE
# Invariant: Decision = (weighted sum >= threshold)
# Agent A: weight 3, Agent B: weight 2, Agent C: weight 1
# Threshold: 4

vote_a:sbf = in file("inputs/vote_a.in")
vote_b:sbf = in file("inputs/vote_b.in")
vote_c:sbf = in file("inputs/vote_c.in")
decision:sbf = out file("outputs/decision.out")

# Convert each boolean vote to its weight using ternary operator
# Sum all weights, compare to threshold
# (vote ? weight : 0) converts vote to weight
r decision[t] = (
    (vote_a[t] ? {3}:bv[8] : {0}:bv[8]) +
    (vote_b[t] ? {2}:bv[8] : {0}:bv[8]) +
    (vote_c[t] ? {1}:bv[8] : {0}:bv[8])
) >= {4}:bv[8]
```

### How to Read It

**The ternary operator:** `(condition ? value_true : value_false)`
- If condition is true (1), use value_true
- If condition is false (0), use value_false

**Breaking down the formula:**
1. `(vote_a[t] ? {3}:bv[8] : {0}:bv[8])` → If A votes yes, contribute 3; else 0
2. `+ (vote_b[t] ? {2}:bv[8] : {0}:bv[8])` → Add B's contribution (2 or 0)
3. `+ (vote_c[t] ? {1}:bv[8] : {0}:bv[8])` → Add C's contribution (1 or 0)
4. `>= {4}:bv[8]` → Compare sum to threshold 4

### Example Scenarios

| A (w=3) | B (w=2) | C (w=1) | Sum | Decision |
|---------|---------|---------|-----|----------|
| 0 | 0 | 0 | 0 | 0 (FAIL) |
| 1 | 0 | 0 | 3 | 0 (FAIL) |
| 0 | 1 | 1 | 3 | 0 (FAIL) |
| 1 | 1 | 0 | 5 | 1 (PASS) |
| 1 | 0 | 1 | 4 | 1 (PASS) |
| 0 | 1 | 0 | 2 | 0 (FAIL) |

### When to Use It
- **Governance:** Stake-weighted voting in DAOs
- **Ensembles:** Weighted combination of ML models
- **Trading:** Weighted signals from multiple indicators
- **Reputation systems:** Higher-reputation users have more influence

### Python Oracle
```python
def weighted_vote(vote_a, vote_b, vote_c, weights=[3,2,1], threshold=4):
    total = (vote_a * weights[0]) + (vote_b * weights[1]) + (vote_c * weights[2])
    return 1 if total >= threshold else 0
```

---

## Tutorial 28: time_lock

### What It Does
Checks if a time-based lock is still active. Used for vesting, escrow, cooldowns, and rate limiting.

### The Invariant
**Formal:** `G(current_time < unlock_time → locked)`
**Casual:** "Can't access until the timer expires"

### Why Bitvectors?
Time requires arithmetic: adding durations, comparing timestamps. Bitvectors provide native arithmetic operations.

### Human-Readable Specification
```tau
# TIME LOCK
# Invariant: Lock active while current_time < (lock_start + duration)

lock_start:bv[16] = in file("inputs/lock_start.in")
lock_duration:bv[16] = in file("inputs/lock_duration.in")
current_time:bv[16] = in file("inputs/current_time.in")
lock_active:sbf = out file("outputs/lock_active.out")

# Calculate whether still locked
# lock_active = current_time < (lock_start + lock_duration)
r (lock_active[t] = (current_time[t] < (lock_start[t] + lock_duration[t])))
```

### How to Read It

**Breaking down the formula:**
1. `lock_start[t] + lock_duration[t]` → Calculate unlock time
2. `- current_time[t]` → Subtract current time to get remaining
3. `> {0}:bv[16]` → If remaining > 0, lock is still active

### Example Scenarios

| Lock Start | Duration | Current | Unlock Time | Remaining | Active? |
|------------|----------|---------|-------------|-----------|---------|
| 100 | 50 | 120 | 150 | 30 | YES |
| 100 | 50 | 150 | 150 | 0 | NO |
| 100 | 50 | 160 | 150 | -10 (wraps) | NO |
| 0 | 1000 | 500 | 1000 | 500 | YES |

### Overflow Handling
Bitvector arithmetic wraps around naturally (modular arithmetic). For time locks, prefer `current_time < end_time` over `(end_time - current_time) > 0` because comparisons are unsigned and wrap-around will otherwise mark expired locks as active.

### When to Use It
- **Vesting:** Tokens locked for a period
- **Escrow:** Funds released after deadline
- **Cooldowns:** Rate limiting actions
- **Governance:** Proposal voting periods
- **Security:** Timelock on admin actions

### Variations

**Check if unlocked (opposite):**
```tau
r unlocked[t] = ((lock_start[t] + lock_duration[t]) - current_time[t]) <= {0}:bv[16]
```

**Fixed duration from constant:**
```tau
# Always 100 units lock duration
r lock_active[t] = ((lock_start[t] + {100}:bv[16]) - current_time[t]) > {0}:bv[16]
```

### Python Oracle
```python
def time_lock(lock_start, lock_duration, current_time):
    end_time = lock_start + lock_duration
    return 1 if current_time < end_time else 0
```

---

## Summary

You've learned:

1. **Invariants** are non-negotiable rules your system must always obey
2. **Tau specs** have three parts: streams, logic, and initial conditions
3. **Logic formulas** use AND, OR, XOR, and NOT to express relationships
4. **Bitvectors** enable arithmetic, weighted voting, and time-based logic
5. **Common patterns** like NAND, XNOR, implication, and ternary solve recurring problems
6. **28 specifications** for safety, decisions, coordination, progress, detection, trust, and bitvector operations

**Next steps:**
- Practice by writing specs for your own agents
- Test your specs exhaustively
- Explore bitvector patterns for numeric logic
- Build up to more complex compositions

Happy specifying! 🎯
