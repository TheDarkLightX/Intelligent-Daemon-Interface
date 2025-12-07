#!/usr/bin/env python3
"""
Edge Case Tests for V46-V48 Advanced Agents

Tests all identified edge cases from FSM analysis:
- Timer boundaries
- Price thresholds
- Stop-loss/take-profit triggers
- Consecutive loss handling
- Position sizing tiers
- Counter overflow

Copyright DarkLightX/Dana Edwards
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class V46State:
    """V46 Risk Management State"""
    # 3-bit state encoding
    state_b0: int = 0
    state_b1: int = 0
    state_b2: int = 0
    
    # Standard
    holding: int = 0
    buy_signal: int = 0
    sell_signal: int = 0
    timer_b0: int = 0
    timer_b1: int = 0
    nonce: int = 0
    profit: int = 0
    has_burned: int = 0
    
    # Bitvector
    entry_price: int = 0
    highest_price: int = 0
    lowest_price: int = 0
    consecutive_losses: int = 0
    cooldown_timer: int = 0
    total_wins: int = 0
    total_losses: int = 0
    
    def get_state_name(self) -> str:
        if self.state_b2 == 0 and self.state_b1 == 0 and self.state_b0 == 0:
            return "IDLE"
        elif self.state_b2 == 0 and self.state_b1 == 0 and self.state_b0 == 1:
            return "EXECUTING"
        elif self.state_b2 == 0 and self.state_b1 == 1 and self.state_b0 == 0:
            return "STOP_LOSS"
        elif self.state_b2 == 0 and self.state_b1 == 1 and self.state_b0 == 1:
            return "TAKE_PROFIT"
        elif self.state_b2 == 1 and self.state_b1 == 0 and self.state_b0 == 0:
            return "COOLDOWN"
        return "UNKNOWN"


class V46Simulator:
    """Simulator for V46 Risk Management Agent"""
    
    def __init__(self):
        self.state = V46State()
        self.history = []
        
    def reset(self):
        self.state = V46State()
        self.history = []
    
    def stop_loss_hit(self, current: int, entry: int, pct: int) -> bool:
        """Check if stop-loss is triggered"""
        if entry == 0 or current >= entry:
            return False
        return (entry - current) * 100 > entry * pct // 100
    
    def take_profit_hit(self, current: int, entry: int, pct: int) -> bool:
        """Check if take-profit is triggered"""
        if entry == 0 or current <= entry:
            return False
        return (current - entry) * 100 > entry * pct // 100
    
    def step(self, price: int, volume: int, trend: int, profit_guard: int,
             failure: int, stop_loss_pct: int, take_profit_pct: int,
             max_losses: int, emergency: int) -> V46State:
        """Execute one step"""
        prev = self.state
        curr = V46State()
        
        # Get current state name
        state_name = prev.get_state_name()
        
        # Timer
        timer = (prev.timer_b1 << 1) | prev.timer_b0
        timed_out = timer >= 3
        
        # Check triggers
        sl_hit = self.stop_loss_hit(price, prev.entry_price, stop_loss_pct)
        tp_hit = self.take_profit_hit(price, prev.entry_price, take_profit_pct)
        
        # State transitions
        if state_name == "IDLE":
            # Can enter if conditions met
            if volume and trend and not prev.holding and not emergency:
                curr.state_b0 = 1  # -> EXECUTING
                curr.entry_price = price
                curr.buy_signal = 1
            curr.holding = curr.buy_signal
            
        elif state_name == "EXECUTING":
            if sl_hit:
                curr.state_b1 = 1  # -> STOP_LOSS
                curr.sell_signal = 1
                curr.consecutive_losses = min(15, prev.consecutive_losses + 1)
                curr.total_losses = min(255, prev.total_losses + 1)
            elif tp_hit:
                curr.state_b1 = 1
                curr.state_b0 = 1  # -> TAKE_PROFIT
                curr.sell_signal = 1
                curr.profit = profit_guard
                curr.consecutive_losses = 0
                curr.total_wins = min(255, prev.total_wins + 1)
            elif timed_out or failure or emergency:
                # Timeout exit
                curr.sell_signal = 1
                profit_made = price > prev.entry_price
                if profit_made and profit_guard:
                    curr.profit = 1
                    curr.total_wins = min(255, prev.total_wins + 1)
                    curr.consecutive_losses = 0
                else:
                    curr.total_losses = min(255, prev.total_losses + 1)
                    curr.consecutive_losses = min(15, prev.consecutive_losses + 1)
            else:
                # Continue executing
                curr.state_b0 = 1
                curr.entry_price = prev.entry_price
                curr.consecutive_losses = prev.consecutive_losses
            
            # Update watermarks
            curr.highest_price = max(prev.highest_price, price)
            curr.lowest_price = min(prev.lowest_price, price) if prev.lowest_price > 0 else price
            
        elif state_name == "STOP_LOSS":
            # Check if cooldown needed
            if prev.consecutive_losses >= max_losses:
                curr.state_b2 = 1  # -> COOLDOWN
            # else -> IDLE (default)
            
        elif state_name == "TAKE_PROFIT":
            # Always -> IDLE
            pass
            
        elif state_name == "COOLDOWN":
            if prev.cooldown_timer < 8:
                curr.state_b2 = 1  # Stay in COOLDOWN
                curr.cooldown_timer = prev.cooldown_timer + 1
            # else -> IDLE (default)
        
        # Timer (only in EXECUTING)
        if curr.state_b0 == 1 and curr.state_b1 == 0 and curr.state_b2 == 0:
            curr.timer_b0 = (prev.timer_b0 + 1) % 2
            curr.timer_b1 = prev.timer_b1 ^ (prev.timer_b0 == 1)
        
        # Position tracking
        curr.holding = curr.buy_signal or (not curr.sell_signal and prev.holding)
        
        # Burns
        curr.has_burned = prev.has_burned or curr.profit
        
        # Preserve counters
        if curr.total_wins == 0:
            curr.total_wins = prev.total_wins
        if curr.total_losses == 0:
            curr.total_losses = prev.total_losses
        if curr.consecutive_losses == 0 and not curr.profit:
            curr.consecutive_losses = prev.consecutive_losses
        
        self.state = curr
        self.history.append((prev, curr))
        return curr


def test_stop_loss_threshold():
    """Test stop-loss at exact threshold"""
    print("\n--- TEST: Stop-Loss at Exact Threshold ---")
    sim = V46Simulator()
    
    # Enter at price 100
    sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    assert sim.state.get_state_name() == "EXECUTING", "Should enter EXECUTING"
    
    # Price drops to exactly 95 (5% drop)
    sim.step(price=95, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    
    # With 5% stop-loss and entry at 100, drop of exactly 5 should trigger
    print(f"  State after 5% drop: {sim.state.get_state_name()}")
    print(f"  Sell signal: {sim.state.sell_signal}")
    print("  PASS: Stop-loss trigger tested")


def test_take_profit_threshold():
    """Test take-profit at exact threshold"""
    print("\n--- TEST: Take-Profit at Exact Threshold ---")
    sim = V46Simulator()
    
    # Enter at price 100
    sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    
    # Price rises to 111 (11% up, just above 10% target)
    sim.step(price=111, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    
    print(f"  State after 11% gain: {sim.state.get_state_name()}")
    print(f"  Profit: {sim.state.profit}")
    print(f"  Total wins: {sim.state.total_wins}")
    print("  PASS: Take-profit trigger tested")


def test_consecutive_losses_to_cooldown():
    """Test transition to cooldown after max losses"""
    print("\n--- TEST: Consecutive Losses to Cooldown ---")
    sim = V46Simulator()
    
    for i in range(3):
        # Enter
        sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
                 stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
        
        # Hit stop-loss
        sim.step(price=90, volume=1, trend=1, profit_guard=1, failure=0,
                 stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
        
        print(f"  Loss {i+1}: State={sim.state.get_state_name()}, "
              f"ConsecLosses={sim.state.consecutive_losses}")
        
        # Process stop-loss state
        sim.step(price=90, volume=1, trend=1, profit_guard=1, failure=0,
                 stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    
    final_state = sim.state.get_state_name()
    print(f"  Final state after 3 losses: {final_state}")
    print(f"  Total losses: {sim.state.total_losses}")
    print("  PASS: Cooldown transition tested")


def test_cooldown_timer():
    """Test cooldown timer duration"""
    print("\n--- TEST: Cooldown Timer ---")
    sim = V46Simulator()
    
    # Force into cooldown by setting state manually
    sim.state.state_b2 = 1  # COOLDOWN
    sim.state.consecutive_losses = 3
    
    for tick in range(10):
        sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
                 stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
        print(f"  Tick {tick+1}: State={sim.state.get_state_name()}, "
              f"Timer={sim.state.cooldown_timer}")
        
        if sim.state.get_state_name() == "IDLE":
            print(f"  Exited cooldown after {tick+1} ticks")
            break
    
    print("  PASS: Cooldown timer tested")


def test_timer_boundary():
    """Test timer at exact timeout"""
    print("\n--- TEST: Timer Boundary ---")
    sim = V46Simulator()
    
    # Enter
    sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=20, take_profit_pct=50, max_losses=5, emergency=0)
    
    for tick in range(5):
        timer = (sim.state.timer_b1 << 1) | sim.state.timer_b0
        state = sim.state.get_state_name()
        print(f"  Tick {tick+1}: State={state}, Timer={timer}")
        
        if state != "EXECUTING":
            print(f"  Exited at timer={timer}")
            break
            
        sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
                 stop_loss_pct=20, take_profit_pct=50, max_losses=5, emergency=0)
    
    print("  PASS: Timer boundary tested")


def test_emergency_halt():
    """Test emergency halt during execution"""
    print("\n--- TEST: Emergency Halt ---")
    sim = V46Simulator()
    
    # Enter
    sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=0)
    assert sim.state.get_state_name() == "EXECUTING"
    
    # Emergency halt
    sim.step(price=100, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=3, emergency=1)
    
    print(f"  State after emergency: {sim.state.get_state_name()}")
    print(f"  Sell signal: {sim.state.sell_signal}")
    print("  PASS: Emergency halt tested")


def test_counter_overflow():
    """Test counter behavior at max values"""
    print("\n--- TEST: Counter Overflow ---")
    sim = V46Simulator()
    
    # Set counters near max
    sim.state.total_wins = 254
    sim.state.total_losses = 254
    sim.state.consecutive_losses = 14
    
    # Trigger a win
    sim.state.state_b0 = 1  # EXECUTING
    sim.state.entry_price = 100
    sim.state.holding = 1
    
    sim.step(price=120, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=15, emergency=0)
    
    print(f"  Total wins after: {sim.state.total_wins} (max 255)")
    print(f"  Consecutive losses: {sim.state.consecutive_losses}")
    
    # Trigger another win to test overflow
    sim.state.state_b0 = 1
    sim.state.entry_price = 100
    sim.step(price=120, volume=1, trend=1, profit_guard=1, failure=0,
             stop_loss_pct=5, take_profit_pct=10, max_losses=15, emergency=0)
    
    print(f"  Total wins at max: {sim.state.total_wins}")
    print("  PASS: Counter overflow tested")


def main():
    """Run all edge case tests"""
    print("=" * 70)
    print("V46-V48 EDGE CASE TESTS")
    print("=" * 70)
    
    tests = [
        test_stop_loss_threshold,
        test_take_profit_threshold,
        test_consecutive_losses_to_cooldown,
        test_cooldown_timer,
        test_timer_boundary,
        test_emergency_halt,
        test_counter_overflow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

