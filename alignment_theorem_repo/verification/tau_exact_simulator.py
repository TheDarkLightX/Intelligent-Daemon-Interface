#!/usr/bin/env python3
"""
Tau Exact Simulator

A faithful implementation of the Tau specifications in Python.
This acts as a reference implementation/VM to execute the logic defined in:
- infinite_deflation_engine.tau
- ethical_ai_alignment.tau
- agent4_testnet_v54.tau

It enforces bit-widths, overflow behavior, and exact logic flow.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List

# =============================================================================
# Helper Functions
# =============================================================================

def mask(val: int, bits: int) -> int:
    """Mask integer to N bits"""
    return val & ((1 << bits) - 1)

def clamp(val: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(val, max_val))

# =============================================================================
# Infinite Deflation Engine (V52)
# =============================================================================

class InfiniteDeflationEngine:
    # Constants
    C_BASE_DEFLATION = 2000  # 20%
    C_MAX_DEFLATION = 5000   # 50%
    C_MIN_DEFLATION = 100    # 1%
    C_EETF_TARGET = 100      # 1.0
    C_POWER_EXPONENT = 200   # 2.0
    C_HALVING_PERIOD = 216000
    SCALE_FACTOR = 100_000_000 # 100^4
    BURN_SCALE = 10000       # 100.00%

    # FSM States
    ST_GENESIS = 0
    ST_ACTIVE = 1
    ST_ACCELERATING = 2
    ST_HALVING = 3
    ST_PAUSED = 4
    ST_TERMINAL = 5

    def __init__(self):
        # State variables
        self.s_total_burned = 0
        self.s_deflation_era = 0
        self.s_era_start_supply = 0
        self.s_ethical_streak = 0
        self.s_state = self.ST_GENESIS
        
        # Previous inputs/state for edge detection
        self.prev_deflation_era = 0
        self.initialized = False

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Unpack inputs
        i_current_supply = mask(inputs['current_supply'], 256)
        i_eetf_network = mask(inputs['eetf_network'], 16)
        i_tx_volume = mask(inputs['tx_volume'], 256)
        i_time_period = mask(inputs['time_period'], 32)
        i_circuit_ok = bool(inputs['circuit_ok'])

        # Initialize era start supply on first run
        if not self.initialized:
            self.s_era_start_supply = i_current_supply
            self.initialized = True

        # ---------------------------------------------------------------------
        # Internal State Updates (Recurrence Relations)
        # ---------------------------------------------------------------------

        # Era calculation
        current_era = mask(i_time_period // self.C_HALVING_PERIOD, 8)
        
        # Era start supply update check
        if current_era > self.prev_deflation_era:
            self.s_era_start_supply = i_current_supply
        
        # Ethical streak
        if i_eetf_network > self.C_EETF_TARGET:
            self.s_ethical_streak = mask(self.s_ethical_streak + 1, 16)
        else:
            self.s_ethical_streak = 0

        # ---------------------------------------------------------------------
        # Core Calculations
        # ---------------------------------------------------------------------

        # EETF Multiplier (Power Law)
        # (1 + excess)^2
        eetf_excess = max(0, i_eetf_network - self.C_EETF_TARGET)
        # Scaled 100. (100 + excess)^2 / 100
        eetf_mult = ((100 + eetf_excess) ** 2) // 100
        eetf_mult = mask(eetf_mult, 16)

        # Era Adjusted Rate
        era_rate = self.C_BASE_DEFLATION
        if current_era == 1: era_rate //= 2
        elif current_era == 2: era_rate //= 4
        elif current_era == 3: era_rate //= 8
        elif current_era == 4: era_rate //= 16
        elif current_era >= 5: era_rate //= 32
        
        # Streak Bonus
        streak_mult = 100
        if self.s_ethical_streak > 100: streak_mult = 300
        elif self.s_ethical_streak > 50: streak_mult = 200
        elif self.s_ethical_streak > 10: streak_mult = 150
        
        # Volume Factor
        vol_mult = 100 # Base
        if i_tx_volume > i_current_supply:
            vol_mult += 200 # Cap at 2x (+100 base = 300 total? No, vol factor returns additonal)
            # Tau: volume_factor(...) + 100.
            # Volume factor logic: cap at 200 (2.0x) if vol > supply?
            # Tau code: (volume > supply) ? 200 : (vol * 100 / supply)
            # Then + 100. So max is 300.
        else:
            if i_current_supply > 0:
                v_factor = (i_tx_volume * 100) // i_current_supply
                vol_mult += mask(v_factor, 16)
            else:
                vol_mult += 0 # Should not happen if supply > 0

        # Effective Rate
        # base * era * eetf * streak * vol / 100,000,000
        # All inputs are scaled integers.
        # era_rate (~2000), eetf_mult (~100), streak (~100), vol (~100)
        # 2000 * 100 * 100 * 100 = 2e9. fits in bv[32].
        # But max could be higher.
        # We use Python arbitrary precision integers to simulate bv[256] safety.
        raw_rate = (era_rate * eetf_mult * streak_mult * vol_mult) // self.SCALE_FACTOR
        raw_rate = mask(raw_rate, 256)

        # Clamping (using the fixed logic: clamp BEFORE cast)
        clamped_rate_256 = clamp(raw_rate, self.C_MIN_DEFLATION, self.C_MAX_DEFLATION)
        effective_rate = mask(clamped_rate_256, 16)

        # Burn Amount
        burn_amount = 0
        if i_circuit_ok:
            burn_amount = (i_current_supply * effective_rate) // self.BURN_SCALE
            burn_amount = mask(burn_amount, 256)
        
        # New Supply
        new_supply = mask(i_current_supply - burn_amount, 256)
        
        # Scarcity Multiplier
        if new_supply > 0:
            scarcity = self.s_era_start_supply // new_supply
        else:
            scarcity = mask(-1, 256) # Max int
        scarcity = mask(scarcity, 256)

        # Update running totals
        self.s_total_burned = mask(self.s_total_burned + burn_amount, 256)
        
        # ---------------------------------------------------------------------
        # FSM State Transition
        # ---------------------------------------------------------------------
        prev_era = self.prev_deflation_era
        
        if not i_circuit_ok:
            next_state = self.ST_PAUSED
        elif current_era > prev_era:
            next_state = self.ST_HALVING
        elif i_eetf_network > 200: # > 2.0
            next_state = self.ST_ACCELERATING
        elif new_supply < 1:
            next_state = self.ST_TERMINAL
        else:
            next_state = self.ST_ACTIVE
            
        self.s_state = next_state
        self.prev_deflation_era = current_era
        self.s_deflation_era = current_era

        return {
            'burn_amount': burn_amount,
            'new_supply': new_supply,
            'total_burned': self.s_total_burned,
            'effective_rate': effective_rate,
            'current_era': current_era,
            'ethical_streak': self.s_ethical_streak,
            'scarcity_multiplier': scarcity,
            'burn_executed': burn_amount > 0,
            'state': self.s_state
        }

# =============================================================================
# Ethical AI Alignment (V53)
# =============================================================================

class EthicalAIAlignment:
    # Constants
    C_EETF_TIER_1 = 100
    C_EETF_TIER_2 = 150
    C_EETF_TIER_3 = 200
    C_EETF_MINIMUM = 80
    C_AI_EETF_PREMIUM = 20
    
    C_ALIGNMENT_BASE = 100
    C_ALIGNMENT_TIER_2 = 300
    C_ALIGNMENT_TIER_3 = 500
    
    SCALE_REWARD = 100_000_000 # 100^4
    SCALE_PRESSURE = 10000

    # FSM States
    ST_UNALIGNED = 0
    ST_BASIC = 1
    ST_ALIGNED = 2
    ST_HIGHLY_ALIGNED = 3
    ST_EXEMPLARY = 4
    ST_AI_ALIGNED = 5
    ST_PENALIZED = 6
    ST_RECOVERING = 7

    def __init__(self):
        self.s_alignment_score = 0
        self.s_ethical_streak = 0
        self.s_account_tier = 0
        self.s_total_penalties = 0
        self.s_state = self.ST_UNALIGNED

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        i_account_eetf = mask(inputs['account_eetf'], 16)
        i_account_lthf = mask(inputs['account_lthf'], 16)
        i_account_balance = mask(inputs['account_balance'], 256)
        i_network_eetf = mask(inputs['network_eetf'], 16)
        i_scarcity_mult = mask(inputs['scarcity_mult'], 256)
        i_tx_value = mask(inputs['tx_value'], 256)
        i_tx_type = mask(inputs['tx_type'], 16)
        i_is_ai_agent = bool(inputs['is_ai_agent'])

        # ---------------------------------------------------------------------
        # Logic
        # ---------------------------------------------------------------------

        # Effective EETF
        effective_eetf = i_account_eetf
        if i_is_ai_agent:
            effective_eetf = max(0, effective_eetf - self.C_AI_EETF_PREMIUM)
        
        # Tier Calculation
        tier = 0
        if effective_eetf >= self.C_EETF_TIER_3: tier = 3
        elif effective_eetf >= self.C_EETF_TIER_2: tier = 2
        elif effective_eetf >= self.C_EETF_TIER_1: tier = 1
        
        self.s_account_tier = tier

        # Ethical Transaction Check
        is_ethical = False
        if i_account_eetf >= self.C_EETF_MINIMUM:
            # Check conditions
            cond1 = (i_tx_value < (i_account_balance // 10)) or (i_account_eetf >= self.C_EETF_TIER_2)
            cond2 = (i_tx_type == 2) or (i_tx_type == 3) or (i_account_eetf >= self.C_EETF_TIER_1)
            if cond1 and cond2:
                is_ethical = True
        
        # Streak Update
        if is_ethical:
            self.s_ethical_streak = mask(self.s_ethical_streak + 1, 32)
        else:
            # Streak resets on unethical? Tau says: (is_ethical) ? prev+1 : 0. Yes.
            self.s_ethical_streak = 0

        # Multipliers
        tier_mult = 0
        if tier == 3: tier_mult = self.C_ALIGNMENT_TIER_3
        elif tier == 2: tier_mult = self.C_ALIGNMENT_TIER_2
        elif tier == 1: tier_mult = self.C_ALIGNMENT_BASE
        
        ai_bonus = 0
        if i_is_ai_agent and i_account_eetf >= self.C_EETF_TIER_3 and self.s_ethical_streak > 100:
            ai_bonus = 50
            
        effective_mult = tier_mult + ai_bonus

        # Reward Calculation
        reward = 0
        if is_ethical:
            reward = (i_account_balance * i_scarcity_mult * effective_mult * i_account_lthf) // self.SCALE_REWARD
            reward = mask(reward, 256)
            
        # Penalty Calculation
        penalty = 0
        if not is_ethical:
            if i_account_eetf < self.C_EETF_MINIMUM:
                # Penalty = tx_value * (min - eetf) / 100
                diff = self.C_EETF_MINIMUM - i_account_eetf
                penalty = (i_tx_value * diff) // 100
                penalty = mask(penalty, 256)
                
        # Economic Pressure
        # Scarcity * NetworkEETF * (100 + penalty_rate) / 10000
        penalty_rate = 50 # C_PENALTY_RATE
        pressure = (i_scarcity_mult * i_network_eetf * (100 + penalty_rate)) // self.SCALE_PRESSURE
        pressure = mask(pressure, 256)

        # Update Stats
        if is_ethical:
            self.s_alignment_score = mask(self.s_alignment_score + reward, 256)
        else:
            self.s_alignment_score = mask(self.s_alignment_score - penalty, 256)
            
        self.s_total_penalties = mask(self.s_total_penalties + penalty, 256)

        # ---------------------------------------------------------------------
        # FSM State Transition
        # ---------------------------------------------------------------------
        prev_state = self.s_state
        
        if not is_ethical and penalty > 0:
            next_state = self.ST_PENALIZED
        elif prev_state == self.ST_PENALIZED and is_ethical:
            next_state = self.ST_RECOVERING
        elif i_is_ai_agent and tier >= 2 and self.s_ethical_streak > 50:
            next_state = self.ST_AI_ALIGNED
        elif tier == 3 and self.s_ethical_streak > 100:
            next_state = self.ST_EXEMPLARY
        elif tier == 3:
            next_state = self.ST_HIGHLY_ALIGNED
        elif tier == 2:
            next_state = self.ST_ALIGNED
        elif tier == 1:
            next_state = self.ST_BASIC
        else:
            next_state = self.ST_UNALIGNED
            
        self.s_state = next_state

        return {
            'is_ethical_tx': is_ethical,
            'alignment_reward': reward,
            'penalty_amount': penalty,
            'account_tier': tier,
            'economic_pressure': pressure,
            'state': self.s_state,
            'total_penalties': self.s_total_penalties
        }

# =============================================================================
# Agent V54 (Unified Predictor)
# =============================================================================

class AgentV54:
    # FSM States
    ST_ANALYZING = 0
    ST_CONFIDENT_ENTRY = 1
    ST_IN_POSITION_BULLISH = 2
    ST_IN_POSITION_NEUTRAL = 3
    ST_IN_POSITION_BEARISH = 4
    ST_EETF_OPTIMIZATION = 5
    ST_EXITING = 6

    # Constants
    C_SIGNAL_STRONG_BUY = 170
    C_SIGNAL_BUY = 140
    C_SIGNAL_NEUTRAL = 100
    C_SCARCITY_STRONG = 10
    
    C_EETF_BULL = 100
    C_EETF_BEAR = 200
    C_EETF_VOLATILE = 150

    def __init__(self):
        self.s_in_position = False
        self.s_entry_price = 0
        self.s_entry_scarcity = 1
        self.s_eetf_ema = 100 # Scaled 1.0
        self.s_state = self.ST_ANALYZING

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Inputs
        i_price = mask(inputs['price'], 64)
        i_current_supply = mask(inputs['current_supply'], 256)
        i_initial_supply = mask(inputs['initial_supply'], 256)
        i_agent_eetf = mask(inputs['agent_eetf'], 16)
        i_predicted_scarcity_short = mask(inputs['predicted_scarcity_short'], 256)
        i_predicted_scarcity_long = mask(inputs['predicted_scarcity_long'], 256)
        i_ema_bullish = bool(inputs['ema_bullish'])
        i_market_regime = mask(inputs['market_regime'], 3)
        i_composite_signal = mask(inputs['composite_signal'], 8)
        i_circuit_ok = bool(inputs['circuit_ok'])

        # ---------------------------------------------------------------------
        # Logic
        # ---------------------------------------------------------------------

        # Current Scarcity
        if i_current_supply > 0:
            current_scarcity = i_initial_supply // i_current_supply
        else:
            current_scarcity = mask(-1, 256)
        current_scarcity = mask(current_scarcity, 256)

        # Scarcity Trend
        if i_predicted_scarcity_short > current_scarcity: trend = 1 # Increasing
        elif i_predicted_scarcity_short < current_scarcity: trend = 2 # Decreasing
        else: trend = 0

        # Prediction Confidence
        confidence = 0
        if i_composite_signal > self.C_SIGNAL_STRONG_BUY and trend == 1 and i_ema_bullish:
            confidence = 3
        elif i_composite_signal > self.C_SIGNAL_BUY and i_ema_bullish:
            confidence = 2
        elif i_composite_signal > self.C_SIGNAL_NEUTRAL:
            confidence = 1
        else:
            confidence = 0

        # Regime Requirement
        regime_req = self.C_EETF_BULL
        if i_market_regime == 3: regime_req = self.C_EETF_BEAR # Markdown
        elif i_market_regime == 4: regime_req = self.C_EETF_VOLATILE # Vol Bull
        elif i_market_regime == 5: regime_req = self.C_EETF_VOLATILE # Vol Bear

        # Entry Signal
        # circuit & ~in_position & confidence >= 2 & eetf >= req
        # (Updated logic: Require >= 2 (Buy/Strong Buy))
        entry_signal = (i_circuit_ok and not self.s_in_position and
                        confidence >= 2 and i_agent_eetf >= regime_req)

        # Exit Signal
        exit_signal = False
        if self.s_in_position:
            # Scarcity gain >= 2x
            cond1 = current_scarcity >= (self.s_entry_scarcity * 2)
            # Price gain > 50%
            cond2 = i_price > (self.s_entry_price * 150 // 100)
            # Confidence collapse
            cond3 = confidence == 0
            
            if cond1 or cond2 or cond3:
                exit_signal = True

        # Should Improve EETF
        should_improve = False
        if (i_predicted_scarcity_long > self.C_SCARCITY_STRONG and i_agent_eetf < 200) or \
           (i_agent_eetf < regime_req):
            should_improve = True

        # Update Position State
        if entry_signal:
            self.s_in_position = True
            self.s_entry_price = i_price
            self.s_entry_scarcity = current_scarcity
        elif exit_signal:
            self.s_in_position = False
            # Reset entry stats? Not strictly needed by spec, but good for sim
            # Spec says s_entry_price keeps value if not entry signal.
            pass

        # ---------------------------------------------------------------------
        # FSM State Transition
        # ---------------------------------------------------------------------
        
        if should_improve and not self.s_in_position:
            next_state = self.ST_EETF_OPTIMIZATION
        elif exit_signal:
            next_state = self.ST_EXITING
        elif self.s_in_position:
            if confidence >= 2: next_state = self.ST_IN_POSITION_BULLISH
            elif confidence == 1: next_state = self.ST_IN_POSITION_NEUTRAL
            elif confidence == 0: next_state = self.ST_IN_POSITION_BEARISH
            else: next_state = self.ST_IN_POSITION_NEUTRAL # Fallback
        elif confidence >= 2 and not self.s_in_position:
            next_state = self.ST_CONFIDENT_ENTRY
        else:
            next_state = self.ST_ANALYZING
            
        self.s_state = next_state

        return {
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
            'in_position': self.s_in_position,
            'confidence': confidence,
            'should_improve_eetf': should_improve,
            'state': self.s_state,
            'current_scarcity': current_scarcity
        }

# =============================================================================
# Main Verification Runner
# =============================================================================

def run_verification():
    print("=" * 80)
    print("TAU EXACT SIMULATOR - EXECUTION TRACE ANALYSIS")
    print("=" * 80)
    
    # 1. Verify Infinite Deflation Engine
    print("\n--- Verifying Infinite Deflation Engine ---")
    engine = InfiniteDeflationEngine()
    
    # Trace 1: Normal Decay
    supply = 10**27
    states_visited = set()
    
    for t in range(100):
        inputs = {
            'current_supply': supply,
            'eetf_network': 100,
            'tx_volume': 10**24,
            'time_period': t,
            'circuit_ok': True
        }
        res = engine.step(inputs)
        states_visited.add(res['state'])
        
        # Invariant checks
        assert res['new_supply'] <= supply, "Supply must imply monotonic decrease"
        assert res['effective_rate'] >= 100, "Rate >= min"
        
        supply = res['new_supply']
        
    print(f"  Passed 100 steps of normal decay. Final supply: {supply}")
    print(f"  States visited: {states_visited}")
    
    # Trace 2: Halving
    # Jump to era 1
    inputs['time_period'] = 220000
    res = engine.step(inputs)
    states_visited.add(res['state'])
    assert res['state'] == engine.ST_HALVING, "Must enter HALVING state"
    print("  Passed halving transition check.")
    
    # Trace 3: Circuit Breaker
    inputs['circuit_ok'] = False
    res = engine.step(inputs)
    states_visited.add(res['state'])
    assert res['state'] == engine.ST_PAUSED, "Must enter PAUSED state"
    assert res['burn_amount'] == 0, "Burn must be 0 when paused"
    print("  Passed circuit breaker check.")
    
    # 2. Verify Agent V54 Logic
    print("\n--- Verifying Agent V54 Logic ---")
    agent = AgentV54()
    
    # Test Entry
    inputs = {
        'price': 100,
        'current_supply': 1000,
        'initial_supply': 1000,
        'agent_eetf': 150,
        'predicted_scarcity_short': 2, # > 1 (current)
        'predicted_scarcity_long': 5,
        'ema_bullish': True,
        'market_regime': 1,
        'composite_signal': 150, # Buy
        'circuit_ok': True
    }
    
    # Step 1: Analyze -> In Position (Immediate Entry)
    # Since all conditions met, we transition directly to position state
    res = agent.step(inputs)
    print(f"  State: {res['state']} (Expected: {agent.ST_IN_POSITION_BULLISH})")
    assert res['state'] == agent.ST_IN_POSITION_BULLISH
    assert res['entry_signal'] == True
    
    # Step 1b: Verify CONFIDENT_ENTRY (Waiting state)
    # If EETF is good but circuit is broken, we wait in CONFIDENT_ENTRY
    agent_blocked = AgentV54()
    inputs_blocked = inputs.copy()
    inputs_blocked['circuit_ok'] = False
    res_blocked = agent_blocked.step(inputs_blocked)
    print(f"  Blocked State: {res_blocked['state']} (Expected: {agent.ST_CONFIDENT_ENTRY})")
    assert res_blocked['state'] == agent.ST_CONFIDENT_ENTRY
    assert res_blocked['entry_signal'] == False
    
    # Step 2: In Position
    # Inputs remain same, but internal state updated
    res = agent.step(inputs)
    print(f"  State: {res['state']} (Expected: {agent.ST_IN_POSITION_BULLISH})")
    assert res['state'] == agent.ST_IN_POSITION_BULLISH
    assert res['in_position'] == True
    
    # Step 3: Exit Condition (Confidence Drop)
    inputs['composite_signal'] = 50 # Sell
    inputs['ema_bullish'] = False
    res = agent.step(inputs)
    print(f"  State: {res['state']} (Expected: {agent.ST_EXITING})")
    assert res['state'] == agent.ST_EXITING
    assert res['exit_signal'] == True
    
    print("\n✓ ALL TRACE CHECKS PASSED")

if __name__ == "__main__":
    run_verification()

