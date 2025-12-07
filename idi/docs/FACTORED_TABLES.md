# Factored and Hierarchical Q-Tables Design

This document outlines the design for factored and hierarchical Q-table architectures that maintain compatibility with Tau specs and zkVM proofs while improving scalability and interpretability.

## Motivation

Monolithic Q-tables face scalability challenges:
- State space explosion: 5 features × 4 buckets each = 1,024 states × 3 actions = 3,072 entries
- Adding features multiplies the state space exponentially
- Many states may be rarely visited, leading to poor generalization

Factored tables decompose the value function into smaller, more manageable components.

## Approach 1: Additive Decomposition

Decompose Q(s, a) as a sum of component Q-values:

```
Q(s, a) = Q_base(a) + Q_price(price_bucket, a) + Q_regime(regime, a) + Q_position(position, a)
```

### Benefits
- Each component table is small (e.g., 4 × 3 = 12 entries for price)
- Components can be interpreted independently
- Easy to add new features without exponential blowup

### Implementation

```python
class FactoredQTable:
    def __init__(self):
        self.base_q = {}  # action -> value
        self.price_q = {}  # (price_bucket, action) -> value
        self.regime_q = {}  # (regime, action) -> value
        self.position_q = {}  # (position, action) -> value

    def q_value(self, state, action):
        price, volume, trend, scarcity, mood = state
        return (
            self.base_q.get(action, 0.0)
            + self.price_q.get((price, action), 0.0)
            + self.regime_q.get((trend, action), 0.0)  # trend as regime proxy
            + self.position_q.get((mood, action), 0.0)  # mood as position proxy
        )
```

### Tau Integration

For Tau specs, decompose into separate input streams:

```tau
# Component weights from factored Q-tables
i_base_buy : sbf = in file("inputs/base_buy.in")
i_price_buy : sbf = in file("inputs/price_buy.in")
i_regime_buy : sbf = in file("inputs/regime_buy.in")

# Combined decision
buy_signal = i_base_buy & i_price_buy & i_regime_buy
```

## Approach 2: Hierarchical (Regime → Micro-State)

Two-level hierarchy where regime selects a sub-policy:

```
Level 1: Regime Selector
  - Input: market regime (bull, bear, chop, panic)
  - Output: sub-policy index

Level 2: Micro-State Policies
  - One small Q-table per regime
  - Input: micro-state (price, volume, position)
  - Output: action
```

### Benefits
- Natural interpretation: different strategies for different market conditions
- Each sub-policy is small and focused
- Regime transitions are explicit

### Implementation

```python
class HierarchicalQTable:
    def __init__(self, n_regimes=4):
        self.regime_policies = [LookupPolicy() for _ in range(n_regimes)]

    def q_value(self, state, action):
        regime = self._get_regime(state)
        micro_state = self._get_micro_state(state)
        return self.regime_policies[regime].q_value(micro_state, action)

    def _get_regime(self, state):
        # Map state to regime index
        _, _, trend, _, _ = state
        return min(trend, 3)  # 0-3 regime index

    def _get_micro_state(self, state):
        # Extract micro-state features
        price, volume, _, scarcity, mood = state
        return (price, volume, scarcity)
```

### Tau Integration

For Tau specs, use regime as a selector:

```tau
# Regime identifier
i_regime : bv[2] = in file("inputs/regime.in")

# Per-regime buy signals
i_buy_bull : sbf = in file("inputs/buy_bull.in")
i_buy_bear : sbf = in file("inputs/buy_bear.in")
i_buy_chop : sbf = in file("inputs/buy_chop.in")
i_buy_panic : sbf = in file("inputs/buy_panic.in")

# Regime-based selection (simplified)
buy_signal = (i_regime = 0 & i_buy_bull)
           | (i_regime = 1 & i_buy_bear)
           | (i_regime = 2 & i_buy_chop)
           | (i_regime = 3 & i_buy_panic)
```

## Approach 3: Low-Rank Approximation

Approximate the Q-table as a low-rank matrix:

```
Q(s, a) ≈ sum_k w_k * f_k(s) * g_k(a)
```

Where f_k are state features and g_k are action features.

### Benefits
- Compact representation for large state spaces
- Generalization across similar states
- Interpretable feature interactions

### Implementation

```python
class LowRankQTable:
    def __init__(self, n_state_features=5, n_action_features=3, rank=3):
        self.rank = rank
        self.state_weights = [[0.0] * n_state_features for _ in range(rank)]
        self.action_weights = [[0.0] * n_action_features for _ in range(rank)]
        self.coefficients = [0.0] * rank

    def q_value(self, state, action):
        action_idx = {"hold": 0, "buy": 1, "sell": 2}[action]
        total = 0.0
        for k in range(self.rank):
            state_feature = sum(s * w for s, w in zip(state, self.state_weights[k]))
            action_feature = self.action_weights[k][action_idx]
            total += self.coefficients[k] * state_feature * action_feature
        return total
```

## zkVM Compatibility

All approaches maintain zkVM compatibility:

1. **Additive**: Prove each component lookup separately, sum in-circuit
2. **Hierarchical**: Prove regime selection, then sub-policy lookup
3. **Low-rank**: Prove dot products are correct

The key constraint is that all operations must be expressible as:
- Table lookups (proven via Merkle inclusion)
- Simple arithmetic (addition, multiplication)
- Boolean logic

## Prototype Plan

### Phase 1: Additive Decomposition
1. Implement `FactoredQTable` class
2. Update trainer to learn component weights
3. Generate separate input streams for Tau
4. Test with existing V38 spec (minor modifications)

### Phase 2: Hierarchical Extension
1. Implement `HierarchicalQTable` class
2. Train regime-specific sub-policies
3. Create regime-switching Tau spec
4. Benchmark vs. monolithic approach

### Phase 3: zkVM Integration
1. Define proof structure for factored lookups
2. Implement Risc0 guest for additive decomposition
3. Validate proofs on sample traces

## Metrics

Track:
- **Table size**: Total entries across all components
- **Learning efficiency**: Episodes to convergence
- **Generalization**: Performance on unseen states
- **Proof size**: zkVM proof complexity

Target: 10x reduction in table size with <5% performance loss.

