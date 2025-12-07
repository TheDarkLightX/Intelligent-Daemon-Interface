# Tau Q-Agents

Intelligent trading agents built with Tau Language executable specifications and multi-layer Q-learning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Intelligence Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Momentum   │  │ Contrarian  │  │   Trend     │         │
│  │  Q-Table    │  │  Q-Table    │  │  Q-Table    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                 │
│                    Weighted Vote                             │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────┐          │
│  │           Action Selection                     │          │
│  │     HOLD | BUY | SELL | WAIT                  │          │
│  └───────────────────────┬───────────────────────┘          │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Daemon Layer                              │
│  ┌─────────────┐              ┌─────────────┐               │
│  │   Python    │     OR       │    Rust     │               │
│  │   Daemon    │              │   Daemon    │               │
│  └──────┬──────┘              └──────┬──────┘               │
└─────────┼────────────────────────────┼──────────────────────┘
          │                            │
          └────────────┬───────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Tau Kernel                                │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Executable Specification (Boolean Logic)           │     │
│  │  • Action validation (legal moves only)             │     │
│  │  • State transitions                                │     │
│  │  • Burn mechanics                                   │     │
│  │  • Constraint enforcement                           │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
tau_q_agents/
├── phase1/                 # Foundation specs (echo, toggle, gates)
├── phase2/                 # Complex logic (FSM, edge detect, trading)
├── phase3/                 # Q-Agents (momentum, deflationary, ASCII)
├── phase4/                 # Multi-layer controlled agents
├── phase5_python/          # Python Q-daemon
├── phase6_rust/            # Rust Q-daemon (high-performance)
├── testing/                # Intelligence benchmarks
└── README.md               # This file
```

## Quick Start

### Run Intelligence Benchmark
```bash
cd testing
python3 intelligence_benchmark.py
```

### Run Rust Daemon
```bash
cd phase6_rust
cargo run --release
```

### Run Interactive Agent
```bash
cd testing
python3 interactive_agent.py
```

### Test Tau Specs
```bash
cat phase1/01_echo.tau | docker run --rm -i tau:latest
```

## Tau Specs

### Phase 1: Foundation
| Spec | Description |
|------|-------------|
| `01_echo.tau` | Pass-through (o = i) |
| `02_negation.tau` | Invert (o = NOT i) |
| `03_delay.tau` | Time delay (o[t] = i[t-1]) |
| `04_toggle.tau` | T flip-flop (XOR with prev) |
| `05_and_gate.tau` | 2-input AND gate |

### Phase 2: Complex Logic
| Spec | Description |
|------|-------------|
| `06_or_gate.tau` | 2-input OR gate |
| `07_edge_detect.tau` | Rising edge detector |
| `08_fsm_2state.tau` | 2-state FSM |
| `09_sr_latch.tau` | Set/Reset memory |
| `10_trading_signal.tau` | Buy/Sell signal generator |

### Phase 3: Q-Agents
| Spec | Description |
|------|-------------|
| `11_momentum_trader.tau` | Requires 2 consecutive signals |
| `12_deflationary_agent.tau` | Burns on every sell |
| `13_ascii_communicator.tau` | 3-bit action codes |
| `14_intelligent_q_agent.tau` | Full Q-policy with momentum |

### Phase 4: Multi-Layer Agents
| Spec | Description |
|------|-------------|
| `15_q_controlled_agent.tau` | Daemon sends Q-selected actions |
| `16_layered_strategy.tau` | Multiple Q-layers vote |
| `17_adaptive_burn.tau` | Volatility-aware burns |

## Intelligence Metrics

The benchmark compares strategies on:
- **Average Reward**: Mean reward per episode
- **Win Rate**: Profitable trades / total trades
- **Adaptation Score**: Performance in regime changes

### Results (typical run)
```
🏆 Intelligence Ranking
🥇 Contrarian           - Reward: 19.48
🥈 Momentum             - Reward: 19.37
🥉 Q-Table (Simple)     - Reward: 19.37
4. Q-Table (Layered)    - Reward: 19.37
5. Random               - Reward: 11.40
6. Buy & Hold           - Reward: 0.89
```

## Q-Table Structure

```python
# States (6 total)
IDLE_NEUTRAL = 0    # Not holding, no signal
IDLE_UP = 1         # Not holding, price up
IDLE_DOWN = 2       # Not holding, price down
HOLDING_NEUTRAL = 3 # Holding, no signal
HOLDING_UP = 4      # Holding, price up
HOLDING_DOWN = 5    # Holding, price down

# Actions (4 total)
HOLD = 0   # Do nothing
BUY = 1    # Enter position
SELL = 2   # Exit position
WAIT = 3   # Skip this step
```

## Deflationary Mechanism

The agents implement a burn-on-profit mechanism:
1. Every sell triggers a burn signal
2. Adaptive burn adds extra burns during volatility
3. Burn amount can be boosted by daemon Q-table

## Emoji/ASCII Communication

Agents output encoded actions:
- `000` = IDLE (.)
- `001` = WAIT (-)
- `010` = BUY (B)
- `011` = SELL (S)
- `100` = HOLD (H)
- `101` = BURN (!)
- `110` = WIN (+)
- `111` = LOSS (X)

## Requirements

- Docker with `tau:latest` image
- Python 3.8+ with numpy
- Rust 1.70+ (for Rust daemon)

## Limitations

- Current Tau Docker (v0.7.0-alpha) does NOT support BitVectors
- All specs use Boolean (SBF) logic only
- Numeric operations require daemon computation

## Next Steps (when Tau BitVector support available)

1. Port specs to use `bv[16]` for prices
2. Implement Q-values directly in Tau
3. Add Bellman equation updates in spec
4. Enable on-chain Q-table verification

