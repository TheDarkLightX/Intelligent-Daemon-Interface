# Hex Pattern Implementation Plan

## Executive Summary

**Hex Pattern:** Time-lock staking system (certificate of deposit style)  
**Feasibility:** ✅ **Highly Feasible** - Core mechanics fit Tau's capabilities  
**Implementation:** 3 phases (Basic FSM → Share & Rewards → Advanced Features)

## Phase 1: Basic Hex FSM Pattern ✅

### Goal
Implement core FSM state machine for Hex staking.

### States
- **UNSTAKED (0)** - No active stake
- **ACTIVE_STAKE (1)** - Stake is active and locked
- **MATURED_STAKE (2)** - Stake has reached maturity
- **EARLY_ENDED_STAKE (3)** - Stake ended before maturity

### Inputs
- `stake_amount` (bv[256]) - Amount to stake
- `stake_duration` (bv[16]) - Duration in days
- `current_time` (bv[32]) - Current time in days
- `action_stake` (sbf) - Stake action signal
- `action_end` (sbf) - End stake action signal

### Outputs
- `stake_state` (bv[2]) - Current state (0-3)
- `lock_active` (sbf) - Is stake active
- `remaining_days` (bv[16]) - Days remaining
- `is_matured` (sbf) - Is stake matured
- `is_early` (sbf) - Is early exit

### Logic
```tau
# State transitions
UNSTAKED → (action_stake & valid_amount & valid_duration) → ACTIVE_STAKE
ACTIVE_STAKE → (current_time >= end_time & action_end) → MATURED_STAKE
ACTIVE_STAKE → (current_time < end_time & action_end) → EARLY_ENDED_STAKE
MATURED_STAKE → (action_end) → UNSTAKED
EARLY_ENDED_STAKE → (action_end) → UNSTAKED

# Time calculations
end_time = lock_start + lock_duration
remaining_days = end_time - current_time
is_matured = current_time >= end_time
is_early = current_time < end_time
```

### Complexity: **Low** ✅
- Uses existing FSM pattern
- Uses existing time_lock pattern concepts
- All boolean/bitvector logic

## Phase 2: Share & Reward Calculation ⚠️

### Goal
Add share calculation and reward accrual.

### Additional Inputs
- `total_shares` (bv[256]) - Total shares in system (external)
- `daily_inflation` (bv[256]) - Daily inflation amount (external)
- `lock_start` (bv[32]) - Lock start time

### Additional Outputs
- `user_shares` (bv[256]) - User's share count
- `accrued_rewards` (bv[256]) - Accrued rewards
- `penalty_amount` (bv[256]) - Penalty if early exit

### Logic
```tau
# Share calculation (linear or sqrt scaling)
user_shares = stake_amount * duration_multiplier
duration_multiplier = duration / max_duration  # Linear
# OR: duration_multiplier = sqrt(duration / max_duration)  # Sqrt (needs external)

# Reward accrual
elapsed_days = current_time - lock_start
user_share_ratio = user_shares / total_shares
daily_reward = user_share_ratio * daily_inflation
accrued_rewards = daily_reward * elapsed_days

# Penalty calculation (early exit)
penalty_rate = base_penalty_rate * (remaining_days / total_days)
penalty_amount = stake_amount * penalty_rate / 100
```

### Complexity: **Medium** ⚠️
- Requires bitvector division (supported)
- Requires external inputs (total_shares, daily_inflation)
- Sqrt scaling may need external computation

## Phase 3: Advanced Features ⚠️

### Goal
Add extend, compound, and governance features.

### Additional Features
- **Extend Stake:** Increase duration while active
- **Compound Rewards:** Re-stake accrued rewards
- **Governance Voting:** Vote while stake is active

### Complexity: **Medium-High** ⚠️
- Extend: Similar to stake, but updates existing stake
- Compound: Requires reward calculation + new stake
- Governance: Requires voting pattern integration

## Implementation Strategy

### Step 1: Basic Hex Pattern (Phase 1)
1. Add `hex_stake` to pattern Literal
2. Implement `_generate_hex_stake_logic()`
3. Create FSM with 4 states
4. Add time calculations
5. Write unit tests
6. Write verification tests

### Step 2: Share Calculation (Phase 2a)
1. Add share calculation logic
2. Support linear and sqrt scaling (sqrt may need external)
3. Add user_shares output
4. Test share calculation

### Step 3: Reward Accrual (Phase 2b)
1. Add reward accrual logic
2. Require external inputs (total_shares, daily_inflation)
3. Add accrued_rewards output
4. Test reward calculation

### Step 4: Penalty Calculation (Phase 2c)
1. Add penalty calculation logic
2. Support early exit penalties
3. Add penalty_amount output
4. Test penalty calculation

### Step 5: Advanced Features (Phase 3)
1. Add extend stake logic
2. Add compound rewards logic
3. Integrate governance voting
4. Test advanced features

## Pattern Schema

```python
LogicBlock(
    pattern="hex_stake",
    inputs=("stake_amount", "stake_duration", "current_time", "action_stake", "action_end"),
    output="stake_state",
    params={
        "max_duration": 3650,  # Max stake duration (days)
        "base_penalty_rate": 50,  # Base penalty rate (percentage)
        "duration_scaling": "linear",  # "linear" or "sqrt"
        "include_shares": True,  # Include share calculation
        "include_rewards": True,  # Include reward accrual
        "include_penalties": True,  # Include penalty calculation
    }
)
```

## What Can Be Implemented

### ✅ Fully Implementable
- FSM state transitions (4 states)
- Time calculations (remaining_days, is_matured, is_early)
- Basic share calculation (linear scaling)
- Lock active tracking

### ⚠️ Implementable with External Inputs
- Reward accrual (needs total_shares, daily_inflation)
- Share ratio calculation (needs total_shares)
- System-level metrics

### ⚠️ Implementable but Complex
- Sqrt scaling (may need external computation)
- Penalty calculation (bitvector division)
- Extend stake (updates existing state)

### ❌ Requires External Computation
- Complex reward formulas (log, exp)
- System-level aggregation (total_shares calculation)
- Multiple stakes per user (requires multiple instances)

## Comparison with Existing Patterns

### Similar to Time Lock Pattern ✅
- Both use time-based locking
- Both use bitvector arithmetic
- Both check maturity

### Similar to Supervisor-Worker Pattern ✅
- Both use FSM state machines
- Both track state transitions

### New Capabilities
- Share calculation (amount * duration_multiplier)
- Reward accrual (share_ratio * inflation * elapsed_days)
- Penalty calculation (early exit penalties)

## Testing Strategy

### Unit Tests
1. FSM state transitions
2. Time calculations
3. Share calculation
4. Reward accrual (with mock external inputs)
5. Penalty calculation

### Integration Tests
1. End-to-end stake lifecycle
2. Early exit with penalties
3. Matured stake with rewards
4. Extend stake functionality

### Verification Tests
1. Execute with Tau binary
2. Verify state transitions
3. Verify time calculations
4. Verify share/reward calculations

## Conclusion

**Hex pattern is highly feasible** for Tau Agent Factory:

✅ **Core FSM:** Fully implementable (Phase 1)  
⚠️ **Share & Rewards:** Implementable with external inputs (Phase 2)  
⚠️ **Advanced Features:** Implementable but complex (Phase 3)  

**Recommendation:** Start with Phase 1 (Basic FSM), then add Phase 2 features incrementally.

