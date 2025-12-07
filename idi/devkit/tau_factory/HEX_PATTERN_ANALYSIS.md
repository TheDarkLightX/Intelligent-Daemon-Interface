# Hex Pattern Analysis & Implementation Plan

## Hex Crypto Overview

**What is Hex?**
- ERC-20 token on Ethereum (launched Dec 2019)
- Time-lock "certificate of deposit" system
- Users lock HEX tokens for chosen duration
- Earn inflationary rewards (newly minted HEX)
- Bonuses for longer, larger, and earlier stakes
- Early exit penalties

## Core Mechanics

### 1. Staking (Creating a Stake)
- User selects:
  - **Amount** of HEX to lock
  - **Length** of lock (up to many years)
- Contract records stake position
- HEX removed from liquid circulation until stake ends

### 2. Rewards & Bonuses
- Rewards paid in **newly minted HEX** (inflation)
- Bonuses favor:
  - **Longer stakes** → higher effective yield
  - **Larger stakes** → additional bonus weighting
  - **Earlier stakes** → better relative opportunity (limited daily distributions)

### 3. Penalties
- Ending stake **before maturity** incurs penalties
- Penalties may be:
  - Redistributed to other stakers (increasing their yield)
  - Burned or removed
- Late end (after maturity) may have additional penalties

### 4. Ending a Stake
- At **maturity date**: user can end stake
  - Returns **original principal HEX**
  - Mints and credits **accrued interest HEX**
  - Minus any penalties
- Early end: penalties apply

## FSM State Model

### States

1. **UNSTAKED** (Liquid)
   - HEX is normal ERC-20 balance
   - Transition: `stake(amount, duration)` → **ACTIVE_STAKE**

2. **ACTIVE_STAKE**
   - Stake record exists: amount, start day, end day, share count
   - HEX non-transferable until stake ends
   - Transitions:
     - Normal maturity: `current_day >= end_day` + `endStake()` → **MATURED_STAKE**
     - Early end: `endStake()` before `end_day` → **EARLY_ENDED_STAKE**

3. **MATURED_STAKE**
   - Logical state: time commitment fulfilled
   - Ready to pay principal + interest
   - Transition: `endStake()` → **UNSTAKED** (principal + interest - penalties)

4. **EARLY_ENDED_STAKE**
   - Logical state: stake closed before term
   - Transition: `endStake()` → **UNSTAKED** (principal - penalties + residual interest)

### FSM Diagram

```
UNSTAKED
  │ stake(amount, duration)
  ▼
ACTIVE_STAKE
  │
  ├─ current_day >= end_day + endStake() → MATURED_STAKE → endStake() → UNSTAKED
  │
  └─ current_day < end_day + endStake() → EARLY_ENDED_STAKE → endStake() → UNSTAKED
```

## Key Calculations

### 1. Share Calculation
- Shares determine reward allocation
- Formula: `shares = amount * duration_multiplier * size_bonus`
- Longer duration = more shares
- Larger amount = more shares

### 2. Reward Calculation
- Rewards = `(user_shares / total_shares) * daily_inflation`
- Accrues over time
- Paid at stake end

### 3. Penalty Calculation
- Early exit: `penalty = base_penalty * (remaining_days / total_days)`
- Late exit: `penalty = late_penalty * (days_late / grace_period)`

### 4. Time Calculations
- `remaining_days = end_day - current_day`
- `elapsed_days = current_day - start_day`
- `is_matured = current_day >= end_day`
- `is_early = current_day < end_day`

## Tau Language Feasibility

### ✅ What CAN Be Implemented

1. **FSM State Transitions**
   - ✅ State machine (UNSTAKED → ACTIVE_STAKE → MATURED/EARLY_ENDED)
   - ✅ Time-based transitions (current_day >= end_day)
   - ✅ Action-based transitions (stake, endStake)

2. **Time Calculations**
   - ✅ Bitvector arithmetic (`+`, `-`, `>=`, `<`)
   - ✅ Remaining days: `(lock_start + lock_duration) - current_time`
   - ✅ Maturity check: `current_time >= (lock_start + lock_duration)`

3. **Share Calculation**
   - ✅ Bitvector multiplication: `shares = amount * duration_multiplier`
   - ✅ Duration multiplier: `multiplier = duration / max_duration` (or sqrt scaling)
   - ✅ Size bonus: `bonus = amount / base_amount`

4. **Penalty Calculation**
   - ✅ Early exit penalty: `penalty = base_penalty * (remaining_days / total_days)`
   - ✅ Bitvector division and multiplication
   - ✅ Comparison: `is_early = current_time < (lock_start + lock_duration)`

5. **Reward Accrual**
   - ✅ Daily reward: `daily_reward = (shares / total_shares) * daily_inflation`
   - ✅ Accumulated reward: `accrued = daily_reward * elapsed_days`
   - ✅ Requires external `total_shares` and `daily_inflation` inputs

### ⚠️ What Needs External Computation

1. **Total Shares (System-Level)**
   - Requires aggregating all active stakes
   - Can be provided as external input (`i_total_shares`)

2. **Daily Inflation**
   - Depends on total supply and inflation rate
   - Can be provided as external input (`i_daily_inflation`)

3. **Share Distribution**
   - Calculating user's share of total shares
   - Can compute: `user_share_ratio = user_shares / total_shares`

4. **Complex Reward Formulas**
   - If rewards depend on complex formulas (e.g., sqrt, log)
   - Tau supports basic arithmetic, but complex math may need external computation

## Implementation Plan

### Phase 1: Basic Hex Pattern (Core FSM)

**Pattern Name:** `hex_stake`

**Inputs:**
- `stake_amount` (bv[256]) - Amount to stake
- `stake_duration` (bv[16]) - Duration in days
- `current_time` (bv[32]) - Current time in days
- `action_stake` (sbf) - Stake action signal
- `action_end` (sbf) - End stake action signal

**Outputs:**
- `stake_state` (bv[2]) - State: 0=UNSTAKED, 1=ACTIVE_STAKE, 2=MATURED_STAKE, 3=EARLY_ENDED_STAKE
- `lock_active` (sbf) - Is stake active
- `remaining_days` (bv[16]) - Days remaining
- `is_matured` (sbf) - Is stake matured
- `is_early` (sbf) - Is early exit

**Logic:**
- FSM transitions based on actions and time
- Time calculations using bitvector arithmetic
- State tracking

### Phase 2: Share & Reward Calculation

**Additional Inputs:**
- `total_shares` (bv[256]) - Total shares in system (external)
- `daily_inflation` (bv[256]) - Daily inflation amount (external)

**Additional Outputs:**
- `user_shares` (bv[256]) - User's share count
- `accrued_rewards` (bv[256]) - Accrued rewards
- `penalty_amount` (bv[256]) - Penalty if early exit

**Logic:**
- Share calculation: `user_shares = stake_amount * duration_multiplier`
- Reward accrual: `accrued = (user_shares / total_shares) * daily_inflation * elapsed_days`
- Penalty calculation: `penalty = base_penalty * (remaining_days / total_days)` if early

### Phase 3: Advanced Features

**Additional Features:**
- Extend stake duration
- Compound rewards (re-stake rewards)
- Multiple stakes per user
- Governance voting (if stake active)

## Pattern Schema Design

```python
LogicBlock(
    pattern="hex_stake",
    inputs=("stake_amount", "stake_duration", "current_time", "action_stake", "action_end"),
    output="stake_state",
    params={
        "max_duration": 3650,  # Max stake duration (days)
        "base_penalty_rate": 50,  # Base penalty rate (percentage)
        "duration_multiplier": "linear",  # or "sqrt" for sqrt scaling
        "include_rewards": True,  # Include reward calculation
        "include_penalties": True,  # Include penalty calculation
    }
)
```

## Complexity Analysis

### What's Easy ✅
- FSM state transitions
- Time calculations (bitvector arithmetic)
- Basic share calculation
- Maturity checks

### What's Medium ⚠️
- Reward accrual (requires external inputs)
- Penalty calculation (bitvector division)
- Share ratio calculation (division)

### What's Hard ❌
- Complex reward formulas (sqrt, log) - may need external computation
- System-level aggregation (total_shares) - must be external input
- Multiple stakes per user - requires multiple pattern instances

## Conclusion

**Hex pattern CAN be implemented** as a Tau Agent Factory pattern:

✅ **Core FSM:** Fully implementable  
✅ **Time calculations:** Fully implementable  
✅ **Share calculation:** Fully implementable  
⚠️ **Reward accrual:** Implementable with external inputs  
⚠️ **Penalty calculation:** Implementable with bitvector arithmetic  
❌ **System-level aggregation:** Requires external computation  

**Recommendation:** Implement Phase 1 (Basic FSM) first, then Phase 2 (Share & Rewards) with external inputs for system-level data.

