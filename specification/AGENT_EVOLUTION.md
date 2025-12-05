# Deflationary Agent Evolution (V35-V48)

## Complete Version History

| Version | Type | States | Key Features | Complexity |
|---------|------|--------|--------------|------------|
| V35 | Boolean | 2 | Original specification, sbf only | 16 clauses |
| V36 | Boolean | 2 | Clarified timer semantics | 16 clauses |
| V37 | Hybrid | 2 | Added 4-bit trade counter | 23 clauses |
| V38 | Boolean | 2 | Minimal core (optimized) | 10 clauses |
| V39a | Boolean | 2 | Daemon heuristic inputs | 10 clauses |
| V39b | Boolean | 2 | Multi-strategy selection | 11 clauses |
| V39c | Boolean | 2 | LTL temporal lookahead | 10 clauses |
| V40 | Modular | 2 | Core + audit modules | 10+7 clauses |
| V41 | Boolean | 2 | Debug state outputs | 14 clauses |
| **V42** | **Bitvector** | 2 | Native bv[16] price thresholds | 11 clauses |
| **V43** | **Bitvector** | 2 | EMA crossover trend detection | 14 clauses |
| **V44** | **Bitvector** | 2 | RSI + adaptive timeout | 17 clauses |
| **V45** | **Bitvector** | 2 | Regime detection + multi-strategy | 20 clauses |
| **V46** | **Risk Mgmt** | 5 | Stop-loss, take-profit, cooldown | 25 clauses |
| **V47** | **Consensus** | 2 | Multi-indicator voting | 25 clauses |
| **V48** | **Sizing** | 2 | Kelly criterion position sizing | 28 clauses |

## Intelligence Progression

```
V35-V41: Binary Intelligence (1-bit signals)
├── Price: 0=low, 1=high
├── Entry: Binary conditions
└── Exit: Timer or binary trigger

V42-V45: Numeric Intelligence (16-bit prices)
├── Price: 0-65535 range
├── Entry: Threshold comparisons
├── Exit: Numeric targets
└── Indicators: EMA, RSI, BB

V46-V48: Adaptive Intelligence
├── Risk Management: Stop-loss, take-profit
├── Performance Tracking: Win rate, streaks
├── Position Sizing: Kelly criterion
└── Multi-state FSM: 5 states
```

## Feature Matrix

| Feature | V35-V41 | V42-V45 | V46-V48 |
|---------|---------|---------|---------|
| Binary price | ✓ | - | - |
| Numeric price | - | ✓ | ✓ |
| Fixed timeout | ✓ | - | - |
| Adaptive timeout | - | V44 | ✓ |
| Stop-loss | - | - | V46 |
| Take-profit | - | - | V46 |
| Cooldown | - | - | V46 |
| EMA indicator | - | V43 | V47 |
| RSI indicator | - | V44 | V47 |
| Bollinger Bands | - | - | V47 |
| Consensus voting | - | - | V47 |
| Win rate tracking | - | - | V48 |
| Streak tracking | - | - | V48 |
| Position sizing | - | - | V48 |

## FSM State Diagrams

### Basic FSM (V35-V45, V47-V48)

```
     ┌────────────────────────┐
     │                        │
     ▼                        │
┌──────────┐    entry    ┌────┴─────┐
│   IDLE   │────────────►│EXECUTING │◄──┐
│   (0)    │             │   (1)    │───┘
└────┬─────┘             └────┬─────┘  continue
     │                        │
     │◄───────────────────────┘
           exit/timeout
```

### Risk FSM (V46)

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
              ┌──────────┐    entry_signal          ┌─────┴─────┐
         ┌───►│   IDLE   │─────────────────────────►│ EXECUTING │◄───┐
         │    │  (000)   │                          │   (001)   │────┘
         │    └──────────┘                          └───────────┘ continue
         │         ▲                                  │   │   
         │         │                      stop_loss◄──┘   └──►take_profit
         │         │                           │                 │
         │    cooldown_done                    ▼                 ▼
         │         │                     ┌──────────┐     ┌──────────┐
         │    ┌────┴─────┐               │STOP_LOSS │     │TAKE_PROFIT│
         │    │ COOLDOWN │◄──max_losses──│  (010)   │     │   (011)   │
         │    │  (100)   │               └──────────┘     └─────┬─────┘
         │    └──────────┘                     │                │
         │                                     ▼                │
         └─────────────────────────────────────┴────────────────┘
```

## Recommended Use Cases

| Version | Best For | Trade-off |
|---------|----------|-----------|
| V38 | Production, low latency | Simple, proven |
| V42 | Better precision | Minimal overhead |
| V43 | Trending markets | May whipsaw |
| V44 | Ranging markets | More computation |
| V45 | Volatile markets | Complex, slower |
| V46 | Capital preservation | 5 states, more logic |
| V47 | Reduce false signals | Consensus overhead |
| V48 | Maximize growth | Most complex |

## Performance Benchmarks

| Version | Clauses | Relative Speed | Memory |
|---------|---------|----------------|--------|
| V38 | 10 | 1.0x (baseline) | Low |
| V42 | 11 | 0.95x | Low |
| V43 | 14 | 0.85x | Medium |
| V44 | 17 | 0.75x | Medium |
| V45 | 20 | 0.65x | High |
| V46 | 25 | 0.55x | High |
| V47 | 25 | 0.55x | High |
| V48 | 28 | 0.50x | High |

## Safety Invariants (All Versions)

1. **Action Exclusivity**: Never buy AND sell simultaneously
2. **Volume Freshness**: Execution requires fresh volume
3. **Nonce Protection**: No immediate re-entry
4. **Timeout Enforcement**: Maximum hold time
5. **Profit-Burn Coupling**: Burns require profit
6. **Monotonic Burns**: Burn history never decreases

## Future Directions

1. **V49: Combined Agent**
   - Best features from V46-V48
   - Risk management + Consensus + Sizing

2. **V50: Machine Learning Integration**
   - Off-chain model predictions as inputs
   - On-chain verification of signals

3. **V51: Multi-Asset Support**
   - Correlation-aware entries
   - Portfolio-level risk management

## Files Created

```
specification/
├── agent4_testnet_v35.tau     # Original
├── agent4_testnet_v36.tau     # Timer clarification
├── agent4_testnet_v37.tau     # Trade counter
├── agent4_testnet_v38.tau     # Minimal core (source of truth now lives under `idi/specs/V38_Minimal_Core/`)
├── agent4_testnet_v39a.tau    # Daemon heuristics
├── agent4_testnet_v39b.tau    # Multi-strategy
├── agent4_testnet_v39c.tau    # LTL lookahead
├── agent4_testnet_v40_core.tau   # Modular core
├── agent4_testnet_v40_audit.tau  # Audit module
├── agent4_testnet_v41.tau     # Debug outputs
├── agent4_testnet_v42.tau     # BV thresholds
├── agent4_testnet_v43.tau     # EMA crossover
├── agent4_testnet_v44.tau     # RSI adaptive
├── agent4_testnet_v45.tau     # Regime detection
├── agent4_testnet_v46.tau     # Risk management
├── agent4_testnet_v47.tau     # Consensus
├── agent4_testnet_v48.tau     # Position sizing
├── BITVECTOR_AGENT_BEST_PRACTICES.md
└── AGENT_EVOLUTION.md         # This file

verification/
├── benchmark_versions.py      # Benchmark framework
├── bv_trace_analyzer.py       # BV trace analysis
├── comprehensive_fsm_analyzer.py  # FSM coverage
├── v46_v48_edge_tests.py      # Edge case tests
└── fsm_analysis_results.json  # Analysis output
```

## Conclusion

The agent series evolved from simple binary logic (V35) to sophisticated
adaptive trading systems (V48). Each version added specific capabilities:

- **V42-V45**: Introduced numeric intelligence with bitvector arithmetic
- **V46**: Added professional risk management (stop-loss, take-profit)
- **V47**: Reduced false signals through indicator consensus
- **V48**: Optimized position sizing using Kelly criterion

For production deployment:
- **Low complexity**: V38 or V42
- **Balanced**: V44 or V46
- **Maximum intelligence**: V47 + V48 features combined

