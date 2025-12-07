# Bitvector Intelligent Agent Best Practices

## Overview

This document captures best practices for designing intelligent deflationary agents using Tau Language's native bitvector (`bv[N]`) types. These practices were developed through the V42-V45 agent series.

## Evolution of Agent Intelligence

| Version | Intelligence Level | Key Features |
|---------|-------------------|--------------|
| V35-V41 | Binary | All `sbf` types, 1-bit price signals |
| V42 | Threshold | Native `bv[16]` prices, numeric thresholds |
| V43 | Trend | EMA crossover via recurrence relations |
| V44 | Momentum | RSI indicator, adaptive timeout |
| V45 | Regime | Market regime detection, multi-strategy |

## Key Design Principles

### 1. Hybrid Architecture (Recommended)

Use `sbf` for state machine logic and `bv[N]` for numeric computations:

```tau
# Good: Hybrid approach
sbf o0 = ofile("outputs/state.out").        # State machine (BDD-optimized)
bv[16] o14 = ofile("outputs/price.out").    # Price data (SMT-optimized)

# Entry uses both types efficiently
(o0[t] = ... & (i0[t] < i5[t]) & ...)       # sbf state + bv comparison
```

**Why:** BDDs excel at boolean logic, SMT solvers excel at arithmetic. Combining them gives best of both worlds.

### 2. Fixed-Point Arithmetic

Avoid floating point by scaling values:

```tau
# EMA calculation using fixed-point (scale by 256)
# α = 0.2 becomes 51/256, (1-α) = 0.8 becomes 205/256
(o15[t] = ({51}:bv[16] * i0[t] + {205}:bv[16] * o15[t-1]) / {256}:bv[16])
```

**Common scales:**
- Scale by 256: Simple bit shifts, α ∈ {0.004, 0.008, ..., 0.996}
- Scale by 100: Percentage-friendly, α ∈ {0.01, 0.02, ..., 0.99}
- Scale by 10000: High precision, α ∈ {0.0001, ..., 0.9999}

### 3. Avoid Division When Possible

Division is expensive in SMT. Use algebraic transformations:

```tau
# Bad: Direct RSI calculation
# rsi = 100 - 100/(1 + gain/loss)

# Good: Cross-multiply to avoid division
# oversold: gain/loss < 30/70 becomes gain * 70 < loss * 30
(o19[t] = ({70}:bv[16] * o16[t]) < ({30}:bv[16] * o17[t]))
```

### 4. Limit Lookback Window

Store only necessary price history:

```tau
# Good: Explicit history variables
bv[16] o15 = ofile("outputs/prev_price_1.out").
bv[16] o16 = ofile("outputs/prev_price_2.out").
bv[16] o17 = ofile("outputs/prev_price_3.out").

# Update chain
(o17[t] = o16[t-1]) &&
(o16[t] = o15[t-1]) &&
(o15[t] = i0[t-1])
```

**Rule of thumb:** More history = slower SMT solving. Keep lookback ≤ 5.

### 5. Use Conditional Assignment Carefully

Tau's conditional syntax for bitvectors:

```tau
# Correct: Ternary for bitvector assignment
(o14[t] = (o2[t] ? i0[t] : o14[t-1]))

# Alternative: Boolean-gated updates
(o14[t] = (o2[t] & i0[t]) | (o2[t]' & o14[t-1]))
```

### 6. Bitvector Size Selection

Choose appropriate bit widths:

| Use Case | Size | Range |
|----------|------|-------|
| Price | `bv[16]` | 0-65535 |
| Timer/Counter | `bv[4]` or `bv[8]` | 0-15 or 0-255 |
| Indicator | `bv[16]` | 0-65535 |
| High precision | `bv[32]` | 0-4B |

**Note:** Larger bit widths increase SMT solver time.

## Technical Indicator Patterns

### EMA (Exponential Moving Average)

```tau
# Fast EMA: α=0.2 (51/256)
(o15[t] = ({51}:bv[16] * i0[t] + {205}:bv[16] * o15[t-1]) / {256}:bv[16])

# Slow EMA: α=0.1 (26/256)  
(o16[t] = ({26}:bv[16] * i0[t] + {230}:bv[16] * o16[t-1]) / {256}:bv[16])

# Crossover signal
(o17[t] = o15[t] > o16[t])  # 1 = uptrend
```

### RSI (Relative Strength Index)

```tau
# Smoothed gain (when price goes up)
(o16[t] = (i0[t] > o15[t]) ? 
          ({64}:bv[16] * (i0[t] - o15[t]) + {192}:bv[16] * o16[t-1]) / {256}:bv[16] :
          ({192}:bv[16] * o16[t-1]) / {256}:bv[16])

# Smoothed loss (when price goes down)
(o17[t] = (i0[t] < o15[t]) ?
          ({64}:bv[16] * (o15[t] - i0[t]) + {192}:bv[16] * o17[t-1]) / {256}:bv[16] :
          ({192}:bv[16] * o17[t-1]) / {256}:bv[16])

# RSI signals (avoiding division)
(o19[t] = ({70}:bv[16] * o16[t]) < ({30}:bv[16] * o17[t]))  # Oversold
(o20[t] = ({30}:bv[16] * o16[t]) > ({70}:bv[16] * o17[t]))  # Overbought
```

### Volatility

```tau
# Simple absolute change
(o18[t] = (i0[t] > o15[t]) ? (i0[t] - o15[t]) : (o15[t] - i0[t]))

# High volatility flag
(o21[t] = o18[t] > {50}:bv[16])
```

### Trend Detection

```tau
# Uptrend: 2 consecutive higher prices
(o22[t] = (i0[t] > o15[t]) & (o15[t] > o16[t]))

# Downtrend: 2 consecutive lower prices
(o23[t] = (i0[t] < o15[t]) & (o15[t] < o16[t]))
```

## Performance Guidelines

### Clause Complexity

| Version | Clauses | Relative Speed |
|---------|---------|----------------|
| V38 (sbf only) | 10 | 1.0x (baseline) |
| V42 (thresholds) | 11 | ~0.9x |
| V43 (EMA) | 14 | ~0.8x |
| V44 (RSI) | 17 | ~0.7x |
| V45 (regime) | 20 | ~0.6x |

### Optimization Tips

1. **Minimize bitvector operations per tick**
   - Each bv operation requires SMT solving
   - Combine operations where possible

2. **Use constants where possible**
   - `{51}:bv[16]` is more efficient than a variable
   - Pre-compute threshold values

3. **Avoid deeply nested conditionals**
   - Flatten nested ternaries into separate clauses
   - Use helper predicates

4. **Benchmark before deploying**
   - Use `bv_trace_analyzer.py` to profile
   - Test with realistic market scenarios

## Daemon Integration

The daemon must provide bitvector inputs:

```python
# Daemon code example
def prepare_inputs():
    # V42: Price and thresholds
    write_bv_input("inputs/price.in", current_price)
    write_bv_input("inputs/entry_threshold.in", support_level)
    write_bv_input("inputs/exit_threshold.in", resistance_level)
    
    # V43: EMA values (computed off-chain for faster initialization)
    write_bv_input("inputs/ema_fast.in", calculate_ema(prices, 0.2))
    write_bv_input("inputs/ema_slow.in", calculate_ema(prices, 0.1))
    
    # V45: Regime parameters
    write_bv_input("inputs/support_level.in", rolling_min)
    write_bv_input("inputs/resistance_level.in", rolling_max)
    write_bv_input("inputs/volatility_threshold.in", atr * 2)
```

## Safety Invariants

All intelligent agents must preserve these invariants:

1. **Action Exclusivity**: Never buy AND sell simultaneously
   ```tau
   (o2[t] & o3[t])' = T  # Always true
   ```

2. **Timeout Enforcement**: Max hold time guaranteed
   ```tau
   timed_out(o7, o6) -> o0[t]' = T  # Timer forces exit
   ```

3. **Nonce Protection**: No immediate re-entry
   ```tau
   o9[t-1] -> entry' = T  # Nonce blocks entry
   ```

4. **Profit-Burn Coupling**: Burns require actual profit
   ```tau
   o12[t] -> o11[t] = T  # Burn implies profit
   ```

5. **Monotonic Burns**: Burn history never decreases
   ```tau
   o13[t] >= o13[t-1] = T  # Burns are monotonic
   ```

## Recommended Version Selection

| Market Condition | Recommended Version | Reason |
|------------------|---------------------|--------|
| Stable, predictable | V42 | Simple thresholds work well |
| Strong trends | V43 | EMA catches trend starts/ends |
| Choppy, oscillating | V44 | RSI identifies extremes |
| Volatile, regime shifts | V45 | Adapts to changing conditions |
| Production (safety-first) | V42 | Minimal complexity, proven |

## Testing Checklist

- [ ] Run all FSM states (8 states in base spec)
- [ ] Test edge cases: zero price, max price, overflow
- [ ] Verify safety invariants hold for all scenarios
- [ ] Benchmark against baseline (V38)
- [ ] Test daemon integration with mock inputs
- [ ] Review trace output for unexpected behavior

## Conclusion

Bitvector intelligence enables sophisticated trading strategies within Tau specifications. The key is balancing complexity against performance:

- **V42** is recommended for production due to simplicity
- **V43** is best for trending markets
- **V44** is best for ranging markets
- **V45** is best for volatile markets with regime changes

Always verify safety invariants and benchmark thoroughly before deployment.

