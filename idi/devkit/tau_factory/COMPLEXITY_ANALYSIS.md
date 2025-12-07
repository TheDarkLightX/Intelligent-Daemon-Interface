# Tau Agent Factory - Complexity Analysis

## Current Capabilities

### Supported Patterns
1. **FSM** - Simple 2-input position state machine (buy/sell → position)
2. **Counter** - Toggle counter on event
3. **Accumulator** - Sum values over time (bitvector only)
4. **Passthrough** - Direct input-to-output mapping
5. **Vote** - OR-based voting from multiple inputs
6. **Majority** - N-of-M majority voting (e.g., 2-of-3, 3-of-5) ✅
7. **Unanimous** - Unanimous consensus (all inputs must agree) ✅
8. **Custom** - Custom boolean expressions with stream names/indices ✅
9. **Quorum** - Minimum votes required ✅
10. **Supervisor-Worker** - Hierarchical FSM coordination ✅
11. **Weighted Vote** - Weighted voting with bitvector arithmetic ✅
12. **Time Lock** - Time-based locking with bitvector arithmetic ✅
13. **Hex Stake** - Time-lock staking system ✅
14. **Multi-Bit Counter** - Multi-bit counters with increment/reset ✅ NEW
15. **Streak Counter** - Consecutive event tracking with reset ✅ NEW
16. **Mode Switch** - Adaptive mode switching ✅ NEW
17. **Proposal FSM** - Governance proposal lifecycle ✅ NEW
18. **Risk FSM** - Risk state machine (NORMAL/WARNING/CRITICAL) ✅ NEW

### Stream Types
- `sbf` - Simple Boolean Functions
- `bv[N]` - Bitvectors (1-32 bits)

### Limitations
- **No multi-bit timers** - Can't generate 2-bit+ counters
- **No state decomposition** - Can't break FSM into sub-states (idle_low, idle_high, etc.)
- ✅ **Custom boolean expressions** - NOW SUPPORTED via `custom` pattern
- **No intermediate outputs** - Can't create helper signals (but outputs can be inputs to other blocks)
- **No conditional logic** - No if-then-else patterns
- **No arithmetic operations** - Only accumulator pattern
- ✅ **Multi-agent coordination** - NOW SUPPORTED via `majority` and `unanimous` patterns
- **No streak tracking** - Can't track win/loss streaks
- **No mode switching** - Can't implement adaptive behavior

## Complex Agent Requirements

### Example: Intelligent Deflationary Agent
**Features Used:**
- Multi-bit timer (timer_b0, timer_b1)
- State decomposition (idle_low, idle_high, pos_low, pos_high)
- Smart entry/exit logic (multiple conditions)
- Trade counter (trade_b0, trade_b1)
- Cumulative accumulator
- Intermediate state signals

**Current Support:** ❌ **Cannot generate** - Missing multi-bit timers, state decomposition, custom conditions

### Example: Fractal Multi-Layer Agent
**Features Used:**
- Multiple Q-layers (momentum, mean-reversion, regime-aware)
- Layer voting/coordination
- Hierarchical state (coarse, medium, fine)
- Communication outputs (emotes, text)
- Bitvector arithmetic (EMA, RSI calculations)
- Weighted voting

**Current Support:** ❌ **Cannot generate** - Missing multi-layer coordination, hierarchical state, custom arithmetic

### Example: Advanced Agents (Multi-Indicator, Adaptive, LFSR)
**Features Used:**
- Multi-indicator fusion (9+ signals)
- Streak tracking (win/loss counters)
- Mode switching (aggressive/defensive)
- LFSR randomization
- Bayesian belief tracking
- Majority voting (N-of-M)

**Current Support:** ✅ **Majority voting supported** - Can do N-of-M majority, unanimous consensus, and custom expressions. Still missing streaks, modes, LFSR, belief tracking.

## Gap Analysis

### Critical Missing Features

#### 1. Multi-Bit Counters/Timers
**Needed for:**
- Position timers (2-bit, 3-bit)
- Trade counters
- Timeout mechanisms

**Current:** Only single-bit toggle counter

**Required:** Pattern like `multi_bit_counter` with width parameter

#### 2. State Decomposition
**Needed for:**
- FSM sub-states (idle_low/high, pos_low/high)
- Hierarchical state machines
- State tracking outputs

**Current:** Only single-position FSM

**Required:** Pattern like `decomposed_fsm` with state definitions

#### 3. Custom Boolean Expressions ✅ IMPLEMENTED
**Needed for:**
- Smart entry: `o0[t-1]' & i1[t] & i2[t] & i0[t]'`
- Complex conditions: `(cond1 & cond2) | (cond3 & cond4)`
- Gated logic

**Current:** ✅ **NOW SUPPORTED** - `custom` pattern with expression string

**Example:**
```python
LogicBlock(
    pattern="custom",
    inputs=("a", "b", "c"),
    output="result",
    params={"expression": "(a[t] & b[t]) | (c[t] & a[t]')"}
)
```

#### 4. Intermediate Outputs
**Needed for:**
- Helper signals (bull_strong, bear_signal)
- State tracking (fsm_idle_low, timer_b0)
- Debug outputs

**Current:** Only final outputs

**Required:** Support for intermediate streams that feed other blocks

#### 5. Multi-Layer Coordination ✅ PARTIALLY IMPLEMENTED
**Needed for:**
- Multiple Q-layers voting
- Weighted ensemble decisions
- Layer confidence tracking

**Current:** ✅ **NOW SUPPORTED** - `majority` and `unanimous` patterns enable ensemble voting

**Example:**
```python
LogicBlock(
    pattern="majority",
    inputs=("agent1", "agent2", "agent3"),
    output="majority_vote",
    params={"threshold": 2, "total": 3}  # 2-of-3 majority
)
```

**Still Missing:** Weighted voting, layer confidence tracking

#### 6. Arithmetic Operations
**Needed for:**
- EMA calculations: `ema[t] = alpha * price[t] + (1-alpha) * ema[t-1]`
- Weighted sums
- Comparisons (via solve blocks)

**Current:** Only accumulator (simple sum)

**Required:** Pattern like `weighted_average` or `ema` with parameters

#### 7. Streak Tracking
**Needed for:**
- Win/loss streaks
- Consecutive event counting
- Performance tracking

**Current:** Only toggle counter

**Required:** Pattern like `streak_counter` with reset condition

#### 8. Mode Switching
**Needed for:**
- Aggressive/defensive modes
- Adaptive behavior
- Regime-based strategies

**Current:** No support

**Required:** Pattern like `mode_switch` with mode conditions

## Complexity Score

### Current System: **8/10** (Updated from 7/10)
- ✅ Basic FSMs
- ✅ Simple counters
- ✅ Basic voting
- ✅ Majority voting (N-of-M)
- ✅ Unanimous consensus
- ✅ Custom boolean expressions
- ✅ Multi-agent coordination
- ❌ No multi-bit timers
- ❌ No state decomposition
- ❌ No weighted voting

### Required for Complex Agents: **8/10**
- ✅ Multi-bit timers (Multi-Bit Counter)
- ⚠️ State decomposition (partial - Supervisor-Worker)
- ✅ Custom boolean expressions
- ✅ Intermediate outputs (outputs can be inputs)
- ✅ Multi-layer coordination (Majority, Unanimous, Quorum)
- ✅ Arithmetic operations (Weighted Vote, Time Lock)
- ✅ Streak tracking (Streak Counter)
- ✅ Mode switching (Mode Switch)

## Recommendations

### Short-Term Improvements (High Priority)

1. **Add Multi-Bit Counter Pattern**
   ```python
   LogicBlock(
       pattern="multi_bit_counter",
       inputs=("event",),
       output="timer",
       params={"width": 2, "reset_on": "sell"}
   )
   ```

2. **Add Custom Logic Pattern** ✅ **IMPLEMENTED**
   ```python
   LogicBlock(
       pattern="custom",
       inputs=("cond1", "cond2", "cond3"),
       output="result",
       params={"expression": "(cond1 & cond2) | (cond3 & cond1')"}
   )
   ```
   **Status:** Available now via `custom` pattern

3. **Support Intermediate Streams**
   ```python
   StreamConfig(name="bull_strong", stream_type="sbf", is_input=False, is_intermediate=True)
   ```

4. **Add State Decomposition**
   ```python
   LogicBlock(
       pattern="decomposed_fsm",
       inputs=("buy", "sell", "regime"),
       output="position",
       params={"states": ["idle_low", "idle_high", "pos_low", "pos_high"]}
   )
   ```

### Medium-Term Improvements

5. **Add EMA/Weighted Average Pattern**
6. **Add Streak Counter Pattern**
7. **Add Mode Switch Pattern**
8. **Add Ensemble Coordinator Pattern**

### Long-Term Improvements

9. **Visual Pattern Builder** - GUI for constructing custom logic
10. **Pattern Library** - Community-shared patterns
11. **Pattern Composition** - Combine patterns hierarchically
12. **Code Generation** - Generate Python/Rust helpers for complex patterns

## Conclusion

**Current Status:** The parameterization system is **sufficient for simple agents** but **insufficient for complex agents**.

**To support complex agents, we need:**
- ✅ Multi-bit counters/timers
- ✅ Custom boolean expressions
- ✅ Intermediate outputs
- ✅ State decomposition
- ✅ Multi-layer coordination
- ✅ Arithmetic operations (EMA, weighted sums)

**Recommendation:** Implement the short-term improvements first, then gradually add medium-term features based on user needs.

