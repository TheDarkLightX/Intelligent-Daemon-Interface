#!/usr/bin/env python3
"""
Realistic Input Simulator for Deflationary Agent

Simulates real Tau Net / blockchain user behavior patterns:
1. Market oscillation - price and volume correlation
2. Trend persistence - bullish/bearish streaks
3. Oracle staleness - periodic volume drops
4. Adversarial inputs - attack patterns
5. Failure injection - daemon failures

Copyright DarkLightX/Dana Edwards
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Generator, Tuple, Optional
from enum import Enum
import json

from fsm_model import InputVector


class MarketCondition(Enum):
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class MarketState:
    """Current market simulation state"""
    price_level: float  # 0.0 to 1.0, threshold at 0.5
    volume_level: float  # 0.0 to 1.0, threshold at 0.5
    trend_momentum: float  # -1.0 to 1.0, threshold at 0.0
    volatility: float  # 0.0 to 1.0
    condition: MarketCondition = MarketCondition.SIDEWAYS
    
    def to_input(self, profit_guard: bool = True, failure: bool = False) -> InputVector:
        """Convert market state to input vector"""
        return InputVector(
            i0=1 if self.price_level >= 0.5 else 0,
            i1=1 if self.volume_level >= 0.5 else 0,
            i2=1 if self.trend_momentum >= 0.0 else 0,
            i3=1 if profit_guard else 0,
            i4=1 if failure else 0
        )


class InputSimulator:
    """
    Generate realistic input sequences simulating Tau Net behavior
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.market = MarketState(
            price_level=0.3,
            volume_level=0.6,
            trend_momentum=0.0,
            volatility=0.2
        )
        self.tick = 0
    
    def reset(self, seed: Optional[int] = None):
        """Reset simulation state"""
        if seed is not None:
            random.seed(seed)
        self.market = MarketState(
            price_level=0.3,
            volume_level=0.6,
            trend_momentum=0.0,
            volatility=0.2
        )
        self.tick = 0
    
    def step_market(self) -> MarketState:
        """Advance market by one tick with realistic dynamics"""
        m = self.market
        
        # Price follows random walk with trend bias
        price_change = random.gauss(m.trend_momentum * 0.05, m.volatility * 0.1)
        m.price_level = max(0.0, min(1.0, m.price_level + price_change))
        
        # Volume correlates with price movement magnitude
        volume_base = 0.5 + abs(price_change) * 2
        m.volume_level = max(0.0, min(1.0, volume_base + random.gauss(0, 0.1)))
        
        # Trend momentum has inertia but can reverse
        momentum_change = random.gauss(0, 0.1)
        if m.price_level > 0.7:
            momentum_change -= 0.02  # Mean reversion at high prices
        elif m.price_level < 0.3:
            momentum_change += 0.02  # Mean reversion at low prices
        m.trend_momentum = max(-1.0, min(1.0, m.trend_momentum + momentum_change))
        
        # Volatility clustering
        vol_change = random.gauss(0, 0.02)
        if abs(price_change) > 0.1:
            vol_change += 0.05  # Volatility increases after big moves
        m.volatility = max(0.05, min(0.5, m.volatility + vol_change))
        
        self.tick += 1
        return m
    
    def generate_scenario_bull_market(self, ticks: int = 20) -> List[InputVector]:
        """Generate bull market scenario - trending up"""
        self.reset(seed=42)
        self.market.trend_momentum = 0.5
        self.market.price_level = 0.3
        
        inputs = []
        for _ in range(ticks):
            self.step_market()
            inputs.append(self.market.to_input(profit_guard=True))
        return inputs
    
    def generate_scenario_bear_market(self, ticks: int = 20) -> List[InputVector]:
        """Generate bear market scenario - trending down"""
        self.reset(seed=43)
        self.market.trend_momentum = -0.5
        self.market.price_level = 0.7
        
        inputs = []
        for _ in range(ticks):
            self.step_market()
            inputs.append(self.market.to_input(profit_guard=True))
        return inputs
    
    def generate_scenario_flash_crash(self, ticks: int = 15) -> List[InputVector]:
        """Generate flash crash - sudden drop then recovery"""
        self.reset(seed=44)
        self.market.price_level = 0.6
        self.market.trend_momentum = 0.2
        
        inputs = []
        for i in range(ticks):
            if i == 5:  # Crash starts
                self.market.price_level = 0.2
                self.market.trend_momentum = -0.8
                self.market.volatility = 0.4
            elif i == 10:  # Recovery starts
                self.market.trend_momentum = 0.6
            
            self.step_market()
            inputs.append(self.market.to_input(profit_guard=True))
        return inputs
    
    def generate_scenario_oracle_stale(self, ticks: int = 15) -> List[InputVector]:
        """Generate oracle staleness - periodic volume drops"""
        self.reset(seed=45)
        inputs = []
        
        for i in range(ticks):
            self.step_market()
            # Every 4th tick, oracle goes stale (volume = 0)
            if i % 4 == 3:
                inp = InputVector(
                    i0=1 if self.market.price_level >= 0.5 else 0,
                    i1=0,  # Stale oracle
                    i2=1 if self.market.trend_momentum >= 0.0 else 0,
                    i3=1,
                    i4=0
                )
            else:
                inp = self.market.to_input(profit_guard=True)
            inputs.append(inp)
        return inputs
    
    def generate_scenario_adversarial_double_entry(self, ticks: int = 10) -> List[InputVector]:
        """Adversarial: Attempt rapid re-entry to exploit nonce"""
        inputs = []
        
        # Setup for entry
        entry_input = InputVector(i0=0, i1=1, i2=1, i3=1, i4=0)
        exit_input = InputVector(i0=1, i1=1, i2=1, i3=1, i4=0)
        
        # First trade
        inputs.append(entry_input)  # Enter
        inputs.append(exit_input)   # Exit
        
        # Rapid re-entry attempts (should be blocked by nonce)
        for _ in range(4):
            inputs.append(entry_input)
        
        # Wait and try again
        inputs.append(InputVector(i0=1, i1=0, i2=0, i3=1, i4=0))  # Idle tick
        inputs.append(entry_input)  # Should still be blocked
        
        return inputs
    
    def generate_scenario_adversarial_mev(self, ticks: int = 12) -> List[InputVector]:
        """Adversarial: MEV-style sandwich attack simulation"""
        inputs = []
        
        # Attacker front-runs: creates favorable entry condition
        inputs.append(InputVector(i0=0, i1=1, i2=1, i3=1, i4=0))  # Low price, entry
        
        # User trade executes
        inputs.append(InputVector(i0=0, i1=1, i2=1, i3=1, i4=0))  # Continue
        
        # Attacker back-runs: manipulates price up quickly
        inputs.append(InputVector(i0=1, i1=1, i2=0, i3=1, i4=0))  # Price spikes
        inputs.append(InputVector(i0=1, i1=1, i2=0, i3=1, i4=0))  # Stay high
        
        # Agent forced to exit at "good" price (but trend reversed)
        inputs.append(InputVector(i0=1, i1=1, i2=0, i3=1, i4=0))  # Exit
        
        # More manipulation
        for _ in range(3):
            inputs.append(InputVector(i0=0, i1=1, i2=1, i3=1, i4=0))
            inputs.append(InputVector(i0=1, i1=1, i2=0, i3=0, i4=0))  # No profit guard
        
        return inputs
    
    def generate_scenario_failure_injection(self, ticks: int = 15) -> List[InputVector]:
        """Random failure echo injection"""
        self.reset(seed=46)
        inputs = []
        
        for i in range(ticks):
            self.step_market()
            # Random failure at 20% probability, more likely during volatile periods
            failure_prob = 0.1 + self.market.volatility * 0.2
            failure = random.random() < failure_prob
            
            inputs.append(self.market.to_input(profit_guard=True, failure=failure))
        return inputs
    
    def generate_scenario_profit_guard_unreliable(self, ticks: int = 15) -> List[InputVector]:
        """Profit guard fails periodically"""
        self.reset(seed=47)
        inputs = []
        
        for i in range(ticks):
            self.step_market()
            # Profit guard fails 30% of the time
            guard_works = random.random() > 0.3
            inputs.append(self.market.to_input(profit_guard=guard_works))
        return inputs
    
    def generate_scenario_high_volatility(self, ticks: int = 20) -> List[InputVector]:
        """High volatility market - rapid price swings"""
        self.reset(seed=48)
        self.market.volatility = 0.4
        
        inputs = []
        for _ in range(ticks):
            # Force high volatility behavior
            self.market.volatility = max(0.3, self.market.volatility)
            self.step_market()
            inputs.append(self.market.to_input(profit_guard=True))
        return inputs
    
    def generate_scenario_sideways(self, ticks: int = 20) -> List[InputVector]:
        """Sideways/ranging market - no clear trend"""
        self.reset(seed=49)
        self.market.trend_momentum = 0.0
        self.market.price_level = 0.5
        self.market.volatility = 0.1
        
        inputs = []
        for _ in range(ticks):
            # Keep momentum near zero
            self.market.trend_momentum *= 0.5
            self.step_market()
            inputs.append(self.market.to_input(profit_guard=True))
        return inputs
    
    def generate_all_32_inputs(self) -> List[Tuple[InputVector, str]]:
        """Generate all 32 possible input combinations with descriptions"""
        inputs = []
        for i in range(32):
            inp = InputVector.from_int(i)
            desc = self._describe_input(inp)
            inputs.append((inp, desc))
        return inputs
    
    def _describe_input(self, inp: InputVector) -> str:
        """Human-readable input description"""
        parts = []
        parts.append("PRICE_HIGH" if inp.i0 else "price_low")
        parts.append("VOL_OK" if inp.i1 else "vol_low")
        parts.append("BULL" if inp.i2 else "bear")
        parts.append("GUARD" if inp.i3 else "no_guard")
        parts.append("FAIL!" if inp.i4 else "ok")
        return " ".join(parts)
    
    def generate_exhaustive_from_states(self) -> Dict[str, List[Tuple[List[InputVector], str]]]:
        """
        Generate input sequences that reach each reachable state,
        then exhaustively test all 32 inputs from that state.
        """
        from fsm_model import DeflationaryAgentFSM, AgentState
        
        fsm = DeflationaryAgentFSM()
        fsm.enumerate_reachable_states()
        fsm.build_transition_table()
        
        result = {}
        
        for state in fsm.reachable_states:
            state_key = str(state)
            result[state_key] = []
            
            # Find path to state
            path = self._find_path_to_state(fsm, state)
            setup_inputs = [inp for inp, _ in (path or [])]
            
            # Test all 32 inputs
            for i in range(32):
                test_inp = InputVector.from_int(i)
                full_seq = setup_inputs + [test_inp]
                desc = f"State {state_key} + input {i:02d}: {self._describe_input(test_inp)}"
                result[state_key].append((full_seq, desc))
        
        return result
    
    def _find_path_to_state(self, fsm, target) -> Optional[List[Tuple[InputVector, 'AgentState']]]:
        """BFS to find path to target state"""
        from fsm_model import AgentState
        from collections import deque
        
        initial = AgentState.initial()
        if target == initial:
            return []
        
        queue = deque([(initial, [])])
        visited = {initial}
        
        while queue:
            state, path = queue.popleft()
            
            for i in range(32):
                inp = InputVector.from_int(i)
                next_state, _, _ = fsm.next_state(state, inp)
                
                if next_state == target:
                    return path + [(inp, next_state)]
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [(inp, next_state)]))
        
        return None


def main():
    """Generate and save all simulation scenarios"""
    simulator = InputSimulator(seed=42)
    
    scenarios = {
        'bull_market': simulator.generate_scenario_bull_market(),
        'bear_market': simulator.generate_scenario_bear_market(),
        'flash_crash': simulator.generate_scenario_flash_crash(),
        'oracle_stale': simulator.generate_scenario_oracle_stale(),
        'adversarial_double_entry': simulator.generate_scenario_adversarial_double_entry(),
        'adversarial_mev': simulator.generate_scenario_adversarial_mev(),
        'failure_injection': simulator.generate_scenario_failure_injection(),
        'profit_guard_unreliable': simulator.generate_scenario_profit_guard_unreliable(),
        'high_volatility': simulator.generate_scenario_high_volatility(),
        'sideways_market': simulator.generate_scenario_sideways(),
    }
    
    print("=" * 60)
    print("INPUT SIMULATOR - Generated Scenarios")
    print("=" * 60)
    
    for name, inputs in scenarios.items():
        print(f"\n{name}: {len(inputs)} ticks")
        print("  First 5 inputs:")
        for i, inp in enumerate(inputs[:5]):
            print(f"    {i}: {simulator._describe_input(inp)}")
    
    # Save to JSON
    output_data = {}
    for name, inputs in scenarios.items():
        output_data[name] = [inp.to_tuple() for inp in inputs]
    
    output_file = "/home/trevormoc/Downloads/DeflationaryAgent/verification/simulated_inputs.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nScenarios saved to {output_file}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total input ticks: {sum(len(v) for v in scenarios.values())}")
    
    # Print all 32 input combinations
    print("\n" + "=" * 60)
    print("ALL 32 INPUT COMBINATIONS")
    print("=" * 60)
    all_inputs = simulator.generate_all_32_inputs()
    for inp, desc in all_inputs:
        print(f"  {inp.to_tuple()}: {desc}")
    
    return scenarios


if __name__ == "__main__":
    main()

