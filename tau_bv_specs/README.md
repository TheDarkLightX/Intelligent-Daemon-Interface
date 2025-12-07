# Tau Language BitVector Experiments

This directory contains experiments with the Tau Language, focusing on building increasingly complex specifications culminating in a deflationary token agent.

## Overview

The Tau Language is a formal specification language that is:
- **Declarative**: You specify constraints, not procedures
- **Executable**: Specifications can be run directly
- **Decidable**: Satisfiability can be checked automatically

## Directory Structure

```
tau_bv_specs/
├── Simple Boolean Function Specs (sbf)
│   ├── 01_echo_simple.tau      - Output equals input
│   ├── 02_negation.tau         - Output is complement of input
│   ├── 03_memory_delay.tau     - Output equals previous input
│   ├── 04_toggle.tau           - Toggle on input signal
│   ├── 05_and_gate.tau         - AND of two inputs
│   └── 06_xor_gate.tau         - XOR of two inputs
│
├── BitVector Specs (bv)
│   ├── 10_bv_counter.tau       - 8-bit counter
│   ├── 11_bv_echo.tau          - BitVector echo
│   ├── 12_bv_add.tau           - BitVector addition
│   ├── 13_bv_comparator.tau    - Compare two BitVectors
│   ├── 14_bv_threshold.tau     - Threshold detector
│   └── 15_bv_accumulator.tau   - Running sum accumulator
│
├── Trading Specs
│   ├── 20_trading_signals.tau  - Buy/sell signal generation
│   ├── 21_position_tracker.tau - Position state machine
│   └── 22_profit_tracker.tau   - Profit calculation
│
├── Deflationary Agents
│   ├── 30_deflationary_agent_bv.tau     - Full BV agent (for new tau version)
│   └── 31_deflationary_agent_simple.tau - Simplified BV agent
│
├── Test Scripts
│   ├── run_tests.sh                     - Basic test runner
│   ├── test_deflationary_simple.sh      - Simple agent test
│   └── deflationary_agent_test_trace.sh - Full execution trace
│
└── Input/Output Files
    ├── inputs/                          - Test input files
    └── outputs/                         - Test output files
```

## Running Tests with Docker

The Tau Language is available via Docker:

```bash
# Basic REPL
docker run --rm -i --entrypoint /tau-lang/build-Release/tau tau:latest

# Run a spec file
docker run --rm -v "$(pwd):/work" -w /work --entrypoint /tau-lang/build-Release/tau tau:latest spec.tau
```

## Key Concepts

### Stream Types
- `sbf` - Simple Boolean Function (0 or 1)
- `tau` - Tau specifications (for meta-programming)
- `bv[N]` - BitVector of N bits (e.g., `bv[8]` for 8-bit)

### Stream Definition (older syntax)
```tau
sbf i1 = console.     # Input from console
sbf o1 = console.     # Output to console
sbf i2 = ifile("input.in").   # Input from file
sbf o2 = ofile("output.out"). # Output to file
```

### Basic Operators
- `&` - AND (conjunction)
- `|` - OR (disjunction)
- `^` or `+` - XOR
- `'` - NOT (complement)
- `&&` - Logical AND (for wff)
- `||` - Logical OR (for wff)

### BitVector Operators (new syntax)
- `+` - Addition
- `-` - Subtraction
- `*` - Multiplication
- `/` - Division
- `%` - Modulo
- `<<` - Left shift
- `>>` - Right shift

### Temporal References
- `o1[t]` - Current output
- `o1[t-1]` - Previous output
- `i1[t]` - Current input

## Deflationary Agent Spec

The core deflationary agent logic:

```tau
# Position management
o1[t] = (i1[t] & o1[t-1]') | (o1[t-1] & i2[t]')
# holding = (buy AND NOT was_holding) OR (was_holding AND NOT sell)

# Token burning
o2[t] = i2[t] & o1[t-1]
# burn = sell AND was_holding
```

Where:
- `i1` = buy signal
- `i2` = sell signal
- `o1` = holding position
- `o2` = burn trigger

## Verified Test Results

### Test 1: Echo
- Input: 0, 1, 0
- Output: 0, 1, 0 ✓

### Test 2: Negation
- Input: 0, 1, 0
- Output: 1, 0, 1 ✓

### Test 3: Toggle
- Input: 1, 0, 1, 1 (starting at 0)
- Output: 1, 1, 0, 1 ✓

### Test 4: AND Gate
- Input: (0,0), (0,1), (1,1), (1,0)
- Output: 0, 0, 1, 0 ✓

### Test 5: XOR Gate
- Input: (0,0), (0,1), (1,0), (1,1)
- Output: 0, 1, 1, 0 ✓

### Test 6: Deflationary Agent
- Buy signal -> Enter position
- Hold -> Maintain position
- Sell signal -> Exit position, BURN token ✓

## Future Work

1. Test with new Tau version that has full BitVector support
2. Implement multi-bit burn counter
3. Add profit tracking with BitVector price storage
4. Implement stop-loss mechanisms
5. Create meta-specification that can update its own trading rules

## References

- [Tau Language GitHub](https://github.com/IDNI/tau-lang)
- [Tau Documentation](https://github.com/IDNI/tau-lang/blob/main/README.md)
- [TABA Book](https://github.com/IDNI/tau-lang/blob/main/docs/Theories-and-Applications-of-Boolean-Algebras-0.25.pdf)

## Author

DarkLightX/Dana Edwards

