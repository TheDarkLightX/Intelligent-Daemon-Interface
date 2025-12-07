# Deflationary Agent Upgrade Guide: V35 → V36 → V37

## Overview

This document describes the upgrades from V35 to V36 and V37, leveraging the new bitvector features in Tau Language. The upgrades improve code clarity, maintainability, and future extensibility while maintaining all safety guarantees.

## Version Summary

| Version | Key Features | Status |
|---------|-------------|--------|
| V35 | Production-ready BDD-optimized kernel | Baseline |
| V36 | Bitvector-aware timer logic | Upgrade |
| V37 | Extended bitvector counters | Advanced |

## V36 Changes: Bitvector Timer Optimization

### Rationale
The original V35 timer used two separate sbf variables (o6, o7) with complex XOR-based logic to implement a 2-bit counter. V36 expresses the timer logic more naturally using concepts that align with bitvector arithmetic.

### Key Changes

1. **Timer Logic Simplification**
   - V35 timer bit 1: Complex XOR expression
   ```tau
   (o7[t] = o0[t] & ((o7[t-1] & o6[t-1]') | (o7[t-1]' & o6[t-1])))
   ```
   - V36 retains the same formula but with cleaner documentation explaining it as a bitvector increment pattern

2. **Backward Compatibility**
   - All output stream positions unchanged
   - Input protocol identical
   - Daemon can switch from V35 to V36 without modification

### Performance Characteristics
- BDD efficiency maintained (sbf streams)
- Timer logic is semantically identical to V35
- Better code readability for future maintenance

## V37 Changes: Extended Bitvector Features

### Rationale
V37 demonstrates the potential of bitvector features by adding a 4-bit trade counter. This counter tracks the number of completed trades, providing:
- Rate limiting capability
- Audit trail
- Analytics foundation

### Key Changes

1. **New Outputs (o19-o22): Trade Counter**
   ```tau
   sbf o19 = ofile("outputs/trade_count_b0.out").  # Bit 0
   sbf o20 = ofile("outputs/trade_count_b1.out").  # Bit 1
   sbf o21 = ofile("outputs/trade_count_b2.out").  # Bit 2
   sbf o22 = ofile("outputs/trade_count_b3.out").  # Bit 3
   ```

2. **Trade Counter Logic**
   - Increments on each sell signal (o3=1)
   - Implements ripple-carry addition in Boolean logic
   - Wraps at 15 (4-bit maximum)

3. **Trade Counter Formula**
   ```tau
   # Bit 0: Toggle on each sell
   (o19[t] = (o3[t] & o19[t-1]') | (o3[t]' & o19[t-1]))
   
   # Bit 1: Carry from bit 0
   (o20[t] = (o3[t] & o19[t-1] & o20[t-1]') | 
             (o3[t] & o19[t-1]' & o20[t-1]) |
             (o3[t]' & o20[t-1]))
   ```

### Performance Impact
- Additional 4 clauses in specification
- Estimated ~20% BDD size increase
- Negligible runtime impact due to linear structure

## Safety Guarantees

All versions maintain these critical safety properties:

| Property | Description | Enforced By |
|----------|-------------|-------------|
| Action Exclusivity | Never buy AND sell | State machine design |
| Oracle Freshness | Execution requires volume | Entry/continue gating |
| Nonce Blocking | No rapid re-entry | Nonce flag logic |
| Timeout Enforcement | Max 3 ticks in position | Timer + state gating |
| Burn-Profit Coupling | Burns require profit | o12 = o11 coupling |
| Monotonic Burns | Burn history never decreases | OR accumulator |
| Failure Echo | Emergency exit capability | i4 input gating |

## Verification

The verification script (`verify_v36_v37.py`) performs execution trace analysis:

```bash
python3 verification/verify_v36_v37.py
```

Test scenarios:
1. Normal buy-hold-sell cycle
2. Timeout forced exit
3. Failure echo emergency exit
4. Multiple trades (counter verification)
5. Nonce blocking validation

## Migration Guide

### V35 → V36
1. Replace specification file
2. No daemon changes required
3. Verify with existing test suite

### V36 → V37
1. Replace specification file
2. Add output files:
   - `outputs/trade_count_b0.out`
   - `outputs/trade_count_b1.out`
   - `outputs/trade_count_b2.out`
   - `outputs/trade_count_b3.out`
3. Update daemon to read trade counter (optional)

## Future Directions (V38+)

When Tau Language fully supports bitvector I/O streams:

1. **Native bv[N] I/O**
   ```tau
   bv[2] timer = ofile("outputs/timer.out")
   bv[8] trade_count = ofile("outputs/trade_count.out")
   ```

2. **Numeric Price Oracle**
   ```tau
   bv[16] price = ifile("inputs/price.in")
   # Enable actual price comparisons: price >= threshold
   ```

3. **Enhanced Volume Tracking**
   ```tau
   bv[32] volume = ifile("inputs/volume.in")
   # Support precise volume measurements
   ```

4. **Longer Timeouts**
   ```tau
   bv[8] timer  # 256 tick maximum hold time
   ```

## Technical Notes

### Bitvector Syntax in Tau Language

Variables:
```tau
x:bv[8] = y        # 8-bit bitvector type annotation
solve x + y =_ 1   # Bitvector equation solving
```

Constants:
```tau
{ #b1001 }:bv[8]   # Binary: 1001 as 8-bit
{ #x1f }:bv[8]     # Hexadecimal: 0x1F as 8-bit
{ 31 }:bv[8]       # Decimal: 31 as 8-bit
```

Operations:
```tau
x + y    # Addition
x - y    # Subtraction
x * y    # Multiplication
x / y    # Division
x % y    # Modulo
x << n   # Left shift
x >> n   # Right shift
x & y    # Bitwise AND
x | y    # Bitwise OR
x ^ y    # Bitwise XOR
```

### BDD vs SMT Trade-offs

| Aspect | sbf (BDD) | bv (SMT) |
|--------|-----------|----------|
| Boolean logic | Excellent | Good |
| Arithmetic | Poor | Excellent |
| Satisfiability | Fast | Fast |
| Quantification | Good | Variable |
| Memory | Depends on structure | Bounded |

Current approach: Use sbf for state machine, bv concepts for counters.

## Conclusion

The V36 and V37 upgrades demonstrate how Tau Language's bitvector features can improve specification clarity while maintaining safety. The modular design allows incremental adoption: use V36 for minimal changes, or V37 for enhanced analytics capabilities.

---
*Copyright DarkLightX/Dana Edwards*

