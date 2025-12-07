# Pattern Landscape Quick Reference

## Current Patterns (9)

### Atomic Patterns
1. **FSM** - Basic state machine
2. **Counter** - Toggle counter
3. **Accumulator** - Sum values
4. **Passthrough** - Direct mapping

### Composite Patterns
5. **Vote** - OR-based voting
6. **Majority** - N-of-M voting
7. **Unanimous** - All-agree consensus
8. **Custom** - Boolean expressions
9. **Quorum** - Minimum votes

## Hierarchical FSM Support

### ✅ What Works Now
- **Composition:** FSM outputs can feed into other FSMs
- **Aggregation:** Multiple FSMs can vote via majority/unanimous
- **Decomposition:** Can create sub-states manually via custom expressions

### ❌ What's Missing
- Explicit supervisor-worker relationships
- State decomposition templates
- Orthogonal regions (parallel FSMs)
- History states

## Pattern Landscape

```
Level 1: Atomic (9 patterns) ✅
Level 2: Composite (via composition) ✅
Level 3: Hierarchical (5 patterns) ❌ MISSING
Level 4: Domain-Specific (4 patterns) ❌ MISSING
```

## Documentation

- **PATTERN_LANDSCAPE.md** - Complete pattern taxonomy
- **HIERARCHICAL_FSM_DESIGN.md** - Hierarchical pattern designs
- **HIERARCHICAL_FSM_ANALYSIS.md** - Analysis and roadmap

## Next Steps

See **HIERARCHICAL_FSM_ANALYSIS.md** for implementation roadmap.

