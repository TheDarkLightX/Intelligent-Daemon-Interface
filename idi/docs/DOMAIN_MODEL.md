# IDI/IAN Shared Domain Model

This document defines the core domain concepts shared across Python, Rust, zkVM, and Tau specs.

## Core Domain Types

### Action
**Purpose**: Represents a trading action that an agent can take.

**Values**:
- `Hold` - No action (maintain current position)
- `Buy` - Enter or increase long position
- `Sell` - Exit or enter short position

**Representation**:
- Python: `Literal["hold", "buy", "sell"]` or enum (to be standardized)
- Rust: `enum Action { Hold, Buy, Sell }`
- Tau: `sbf` streams (`q_buy.in`, `q_sell.in`)

**Serialization**: JSON strings `"hold"`, `"buy"`, `"sell"`

### Regime
**Purpose**: Market regime identifier for regime-aware policies.

**Values**:
- `Bull` - Bull market (positive drift)
- `Bear` - Bear market (negative drift)
- `Chop` - Choppy/ranging market (neutral drift)
- `Panic` - High volatility panic regime

**Representation**:
- Python: `Literal["bull", "bear", "chop", "panic"]` or enum
- Rust: `enum Regime { Bull, Bear, Chop, Panic }`
- Tau: `bv[5]` stream (`q_regime.in`)

**Serialization**: JSON strings `"bull"`, `"bear"`, `"chop"`, `"panic"`

### StateKey
**Purpose**: Unique identifier for a discrete state in the Q-table.

**Structure**: Tuple of quantized features:
- `(price_bucket, volume_bucket, trend_bucket, scarcity_bucket, mood_bucket)`

**Constraints**:
- Each bucket is a non-negative integer
- Bounds defined by `QuantizerConfig`:
  - `price_bucket ∈ [0, price_buckets)`
  - `volume_bucket ∈ [0, volume_buckets)`
  - `trend_bucket ∈ [0, trend_buckets)`
  - `scarcity_bucket ∈ [0, scarcity_buckets)`
  - `mood_bucket ∈ [0, mood_buckets)`

**Representation**:
- Python: `Tuple[int, int, int, int, int]`
- Rust: `(u8, u8, u8, u8, u8)`
- Serialization: JSON array `[price, volume, trend, scarcity, mood]`

### Transition
**Purpose**: Represents a state-action-reward-next_state transition.

**Structure**:
```python
@dataclass
class Transition:
    state: StateKey
    action: Action
    reward: float
    next_state: StateKey
    done: bool  # True if episode terminated
```

**Rust equivalent**:
```rust
pub struct Transition {
    pub state: StateKey,
    pub action: Action,
    pub reward: f32,
    pub next_state: StateKey,
    pub done: bool,
}
```

**Serialization**: JSON object with fields `state`, `action`, `reward`, `next_state`, `done`

### Policy
**Purpose**: Maps states to actions (via Q-values or direct mapping).

**Interface**:
- `q_value(state: StateKey, action: Action) -> float` - Get Q-value
- `update(state: StateKey, action: Action, delta: float) -> None` - Update Q-value
- `best_action(state: StateKey) -> Action` - Get best action for state

**Implementation**:
- Python: `LookupPolicy` with `Dict[StateKey, PolicyEntry]`
- Rust: `LookupPolicy` with `HashMap<StateKey, PolicyEntry>`
- Both use `PolicyEntry` containing `Dict/HashMap<Action, float>`

### Environment
**Purpose**: Simulates market dynamics and returns observations/rewards.

**Interface**:
- `reset() -> Observation` - Reset to initial state
- `step(action: Action) -> (Observation, float)` - Execute action, return next obs and reward

**Implementations**:
- `SyntheticMarketEnv` - Simple synthetic market
- `CryptoMarket` - Realistic crypto market with regimes

### Observation
**Purpose**: Represents the current state of the environment.

**Structure**: Quantized features matching `StateKey`:
- `price: int` - Price bucket
- `volume: int` - Volume bucket
- `trend: int` - Trend bucket
- `scarcity: int` - Scarcity bucket
- `mood: int` - Mood bucket

**Methods**:
- `as_state() -> StateKey` - Convert to state key tuple

### Trace
**Purpose**: Sequence of transitions/episodes for training or replay.

**Structure**:
- `TraceTick` - Single timestep:
  - `q_buy: u8` - Buy signal (0 or 1)
  - `q_sell: u8` - Sell signal (0 or 1)
  - `risk_budget_ok: u8` - Risk budget flag
  - `q_emote_positive: u8` - Positive emotion cue
  - `q_emote_alert: u8` - Alert emotion cue
  - `q_regime: u8` - Regime identifier

**Serialization**: JSON array of `TraceTick` objects

### Spec
**Purpose**: Tau language specification defining agent FSM behavior.

**Structure**:
- Stream declarations (inputs/outputs)
- Recurrence relations (FSM logic)
- Helper predicates (optional)

**Format**: Tau language text file (`.tau`)

### Proof
**Purpose**: Zero-knowledge proof bundle for verifiable computation.

**Structure**:
- `proof.bin` - Binary proof data
- `receipt.json` - Receipt with journal digest and metadata
- `manifest.json` - Artifact manifest with stream hashes

**Verification**: Receipt digest must match host-computed hash of manifest + streams

## Configuration Schema

### TrainingConfig
**Purpose**: High-level training hyperparameters.

**Fields**:
- `episodes: int` - Number of training episodes (default: 128)
- `episode_length: int` - Steps per episode (default: 64)
- `discount: float` - Discount factor γ ∈ (0, 1] (default: 0.92)
- `learning_rate: float` - Learning rate α ∈ (0, 1] (default: 0.2)
- `exploration_decay: float` - Exploration decay rate ∈ (0, 1] (default: 0.995)
- `quantizer: QuantizerConfig` - State quantization parameters
- `rewards: RewardWeights` - Reward component weights
- `emote: EmoteConfig` - Emotion/mood configuration (Python only, to be added to Rust)
- `layers: LayerConfig` - Layer weight configuration (Python only, to be added to Rust)
- `tile_coder: Optional[TileCoderConfig]` - Tile coding parameters (Python only, to be added to Rust)
- `communication: CommunicationConfig` - Communication policy config (Python only, to be added to Rust)

### QuantizerConfig
**Purpose**: Defines state space quantization buckets.

**Fields**:
- `price_buckets: int` - Number of price buckets (default: 4)
- `volume_buckets: int` - Number of volume buckets (default: 4)
- `trend_buckets: int` - Number of trend buckets (default: 4)
- `scarcity_buckets: int` - Number of scarcity buckets (default: 8)
- `mood_buckets: int` - Number of mood buckets (default: 4)

**Constraints**: All buckets must be > 0

### RewardWeights
**Purpose**: Multi-objective reward component weights.

**Fields**:
- `pnl: float` - PnL reward weight (default: 1.0)
- `scarcity_alignment: float` - Scarcity alignment weight (default: 0.5)
- `ethics_bonus: float` - Ethics bonus weight (default: 0.75)
- `communication_clarity: float` - Communication clarity weight (default: 0.2)

## Alignment Checklist

- [x] Action enum defined in Rust
- [ ] Action enum defined in Python (currently uses strings)
- [x] Regime enum defined in Rust
- [ ] Regime enum defined in Python (currently uses strings)
- [x] StateKey type aligned (both use 5-tuple)
- [x] Config schema shared (JSON schema exists)
- [ ] Python config includes all fields (emote, layers, tile_coder, communication)
- [ ] Rust config includes all fields (currently missing emote, layers, tile_coder, communication)
- [x] TraceTick structure aligned
- [ ] Transition type explicitly defined in both languages
- [ ] Observation type explicitly defined in both languages

