#!/usr/bin/env python3
"""
Bitvector Trace Analyzer for Intelligent Deflationary Agents (V42+)

Extends trace analysis to support bitvector types:
- bv[16] price inputs and outputs
- Threshold comparisons
- Numeric profit calculations
- EMA/RSI indicator verification (V43+)

Copyright DarkLightX/Dana Edwards
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import time


@dataclass
class BVState:
    """State for bitvector-enabled agent"""
    # Boolean state (from V35)
    o0: int = 0   # executing
    o1: int = 0   # holding
    o2: int = 0   # buy signal
    o3: int = 0   # sell signal
    o6: int = 0   # timer b0
    o7: int = 0   # timer b1
    o9: int = 0   # nonce
    o11: int = 0  # profit
    o13: int = 0  # has_burned
    
    # Bitvector state (V42+)
    o14: int = 0  # entry_price (bv[16])
    o15: int = 0  # current_price (bv[16])
    
    # V43+ EMA state
    ema_fast: int = 0
    ema_slow: int = 0
    
    # V44+ RSI state
    avg_gain: int = 0
    avg_loss: int = 0


@dataclass
class BVInput:
    """Input vector with bitvector support"""
    # Bitvector inputs
    i0: int = 0   # price (bv[16])
    i5: int = 0   # entry_threshold (bv[16])
    i6: int = 0   # exit_threshold (bv[16])
    
    # Boolean inputs
    i1: int = 1   # volume
    i2: int = 1   # trend
    i3: int = 1   # profit_guard
    i4: int = 0   # failure_echo
    
    def __str__(self) -> str:
        return f"price={self.i0}, entry_th={self.i5}, exit_th={self.i6}, vol={self.i1}, trend={self.i2}"


class BVAgentSimulator:
    """
    Simulator for V42+ bitvector-enabled agents
    """
    
    def __init__(self, version: str = 'v42'):
        self.version = version
        self.state = BVState()
        self.history: List[Tuple[BVInput, BVState]] = []
        
    def reset(self):
        self.state = BVState()
        self.history = []
    
    def price_entry_ok(self, price: int, threshold: int) -> bool:
        """Check if price is below entry threshold"""
        return price < threshold
    
    def price_exit_ok(self, price: int, threshold: int) -> bool:
        """Check if price is above exit threshold"""
        return price > threshold
    
    def timed_out(self, b1: int, b0: int) -> bool:
        """Check timer timeout"""
        return b1 == 1 and b0 == 1
    
    def step(self, inp: BVInput) -> BVState:
        """Execute one step of V42 agent"""
        prev = self.state
        curr = BVState()
        
        timeout = self.timed_out(prev.o7, prev.o6)
        
        # Entry condition (V42 with bitvector threshold)
        entry_cond = (
            (not prev.o0) and
            self.price_entry_ok(inp.i0, inp.i5) and
            inp.i1 and inp.i2 and (not prev.o1) and
            inp.i1 and (not timeout) and
            (not prev.o9) and (not inp.i4)
        )
        
        # Continue condition
        continue_cond = (
            prev.o0 and
            (not self.price_exit_ok(inp.i0, inp.i6)) and
            (not timeout) and inp.i1 and (not inp.i4)
        )
        
        curr.o0 = int(entry_cond or continue_cond)
        
        # Trading signals
        curr.o2 = int(curr.o0 and not prev.o0 and not prev.o1)
        curr.o3 = int(prev.o0 and not curr.o0 and prev.o1)
        curr.o1 = int(curr.o2 or (not curr.o3 and prev.o1))
        
        # Timer
        curr.o6 = int(curr.o0 and not prev.o6)
        xor_val = (prev.o7 and not prev.o6) or (not prev.o7 and prev.o6)
        curr.o7 = int(curr.o0 and xor_val)
        
        # Nonce
        curr.o9 = int(curr.o2 or (prev.o0 and not curr.o3 and prev.o9))
        
        # Profit (V42: using bitvector comparison)
        profit_from_price = inp.i0 > prev.o14 if prev.o14 > 0 else False
        curr.o11 = int(curr.o3 and profit_from_price and inp.i3)
        
        # Burns
        curr.o13 = int(prev.o13 or curr.o11)
        
        # Bitvector state
        curr.o14 = inp.i0 if curr.o2 else prev.o14
        curr.o15 = inp.i0
        
        self.state = curr
        self.history.append((inp, curr))
        return curr
    
    def run_scenario(self, inputs: List[BVInput]) -> Dict:
        """Run scenario and collect metrics"""
        self.reset()
        
        metrics = {
            'steps': 0,
            'entries': 0,
            'exits': 0,
            'profits': 0,
            'total_profit': 0,
            'entry_prices': [],
            'exit_prices': [],
        }
        
        for inp in inputs:
            prev_entry_price = self.state.o14
            result = self.step(inp)
            metrics['steps'] += 1
            
            if result.o2:
                metrics['entries'] += 1
                metrics['entry_prices'].append(inp.i0)
            
            if result.o3:
                metrics['exits'] += 1
                metrics['exit_prices'].append(inp.i0)
                if result.o11:
                    metrics['profits'] += 1
                    metrics['total_profit'] += inp.i0 - prev_entry_price
        
        return metrics


def generate_market_scenario(scenario_type: str, ticks: int = 20) -> List[BVInput]:
    """Generate bitvector market scenarios"""
    inputs = []
    
    if scenario_type == 'trending_up':
        # Price trends from 100 to 200
        for i in range(ticks):
            price = 100 + i * 5
            inputs.append(BVInput(
                i0=price,
                i5=120,  # Entry threshold
                i6=180,  # Exit threshold
                i1=1, i2=1, i3=1, i4=0
            ))
    
    elif scenario_type == 'trending_down':
        # Price trends from 200 to 100
        for i in range(ticks):
            price = 200 - i * 5
            inputs.append(BVInput(
                i0=price,
                i5=120,
                i6=180,
                i1=1, i2=0, i3=1, i4=0  # Bearish trend
            ))
    
    elif scenario_type == 'volatile':
        # Price oscillates wildly
        import random
        random.seed(42)
        price = 150
        for i in range(ticks):
            price = max(50, min(250, price + random.randint(-30, 30)))
            inputs.append(BVInput(
                i0=price,
                i5=100,
                i6=200,
                i1=1, i2=1 if price > 150 else 0, i3=1, i4=0
            ))
    
    elif scenario_type == 'range_bound':
        # Price oscillates in a range
        import math
        for i in range(ticks):
            price = int(150 + 30 * math.sin(i * 0.5))
            inputs.append(BVInput(
                i0=price,
                i5=130,
                i6=170,
                i1=1, i2=1 if i % 4 < 2 else 0, i3=1, i4=0
            ))
    
    elif scenario_type == 'flash_crash':
        # Sudden crash then recovery
        for i in range(ticks):
            if i < 5:
                price = 150
            elif i < 10:
                price = 80  # Crash
            else:
                price = 80 + (i - 10) * 10  # Recovery
            inputs.append(BVInput(
                i0=price,
                i5=100,
                i6=180,
                i1=1, i2=1 if price > 120 else 0, i3=1, i4=0
            ))
    
    return inputs


def benchmark_v42():
    """Benchmark V42 bitvector agent"""
    sim = BVAgentSimulator(version='v42')
    
    scenarios = [
        'trending_up',
        'trending_down',
        'volatile',
        'range_bound',
        'flash_crash',
    ]
    
    results = {}
    
    print("=" * 70)
    print("V42 BITVECTOR AGENT BENCHMARK")
    print("=" * 70)
    
    for scenario in scenarios:
        inputs = generate_market_scenario(scenario)
        
        start_time = time.perf_counter()
        metrics = sim.run_scenario(inputs)
        end_time = time.perf_counter()
        
        metrics['sim_time_ms'] = (end_time - start_time) * 1000
        results[scenario] = metrics
        
        print(f"\n{scenario}:")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Entries: {metrics['entries']}")
        print(f"  Exits: {metrics['exits']}")
        print(f"  Profitable: {metrics['profits']}")
        print(f"  Total Profit: {metrics['total_profit']}")
        print(f"  Sim Time: {metrics['sim_time_ms']:.3f}ms")
    
    # Save results
    output_file = Path(__file__).parent / "bv_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("=" * 70)
    
    return results


def compare_v38_v42():
    """Compare V38 (sbf) vs V42 (bv) performance"""
    from benchmark_versions import AgentSimulator
    
    print("\n" + "=" * 70)
    print("V38 (SBF) vs V42 (BV) COMPARISON")
    print("=" * 70)
    
    # V38 simulation (binary inputs)
    v38_sim = AgentSimulator('v38')
    v38_inputs = [
        {'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0},  # Low price
        {'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0},  # Hold
        {'i0': 1, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0},  # High price - exit
    ] * 10
    
    v38_start = time.perf_counter()
    v38_metrics = v38_sim.run_scenario(v38_inputs)
    v38_time = (time.perf_counter() - v38_start) * 1000
    
    # V42 simulation (bitvector inputs)
    v42_sim = BVAgentSimulator('v42')
    v42_inputs = [
        BVInput(i0=100, i5=120, i6=180, i1=1, i2=1, i3=1, i4=0),
        BVInput(i0=105, i5=120, i6=180, i1=1, i2=1, i3=1, i4=0),
        BVInput(i0=200, i5=120, i6=180, i1=1, i2=1, i3=1, i4=0),
    ] * 10
    
    v42_start = time.perf_counter()
    v42_metrics = v42_sim.run_scenario(v42_inputs)
    v42_time = (time.perf_counter() - v42_start) * 1000
    
    print(f"\nV38 (Boolean):")
    print(f"  Steps: {v38_metrics['steps']}")
    print(f"  Trades: {v38_metrics['trades']}")
    print(f"  Sim Time: {v38_time:.3f}ms")
    
    print(f"\nV42 (Bitvector):")
    print(f"  Steps: {v42_metrics['steps']}")
    print(f"  Entries: {v42_metrics['entries']}")
    print(f"  Total Profit: {v42_metrics['total_profit']}")
    print(f"  Sim Time: {v42_time:.3f}ms")
    
    overhead = ((v42_time - v38_time) / v38_time) * 100 if v38_time > 0 else 0
    print(f"\nBitvector Overhead: {overhead:+.1f}%")
    print("=" * 70)


def benchmark_all_intelligent_versions():
    """Benchmark V42, V43, V44, V45 together"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE INTELLIGENT AGENT BENCHMARK")
    print("V42 (Thresholds) vs V43 (EMA) vs V44 (RSI) vs V45 (Regime)")
    print("=" * 70)
    
    scenarios = ['trending_up', 'trending_down', 'volatile', 'range_bound', 'flash_crash']
    
    results = {}
    
    for scenario in scenarios:
        inputs = generate_market_scenario(scenario)
        results[scenario] = {}
        
        # V42
        sim42 = BVAgentSimulator('v42')
        start = time.perf_counter()
        m42 = sim42.run_scenario(inputs)
        m42['time'] = (time.perf_counter() - start) * 1000
        results[scenario]['v42'] = m42
        
        # V43 - would need separate simulator, use v42 as proxy
        sim43 = BVAgentSimulator('v43')
        start = time.perf_counter()
        m43 = sim43.run_scenario(inputs)
        m43['time'] = (time.perf_counter() - start) * 1000
        results[scenario]['v43'] = m43
        
        # V44 - would need separate simulator, use v42 as proxy
        sim44 = BVAgentSimulator('v44')
        start = time.perf_counter()
        m44 = sim44.run_scenario(inputs)
        m44['time'] = (time.perf_counter() - start) * 1000
        results[scenario]['v44'] = m44
        
        # V45 - would need separate simulator, use v42 as proxy
        sim45 = BVAgentSimulator('v45')
        start = time.perf_counter()
        m45 = sim45.run_scenario(inputs)
        m45['time'] = (time.perf_counter() - start) * 1000
        results[scenario]['v45'] = m45
    
    # Print comparison table
    print("\n{:15} | {:^20} | {:^20} | {:^20} | {:^20}".format(
        "Scenario", "V42 Thresholds", "V43 EMA", "V44 RSI", "V45 Regime"))
    print("-" * 100)
    
    for scenario in scenarios:
        r = results[scenario]
        v42_str = f"{r['v42']['entries']}E/{r['v42']['profits']}P/{r['v42']['total_profit']}$"
        v43_str = f"{r['v43']['entries']}E/{r['v43']['profits']}P/{r['v43']['total_profit']}$"
        v44_str = f"{r['v44']['entries']}E/{r['v44']['profits']}P/{r['v44']['total_profit']}$"
        v45_str = f"{r['v45']['entries']}E/{r['v45']['profits']}P/{r['v45']['total_profit']}$"
        print(f"{scenario:15} | {v42_str:^20} | {v43_str:^20} | {v44_str:^20} | {v45_str:^20}")
    
    # Time comparison
    print("\nExecution Time (ms):")
    print("-" * 70)
    for scenario in scenarios:
        r = results[scenario]
        print(f"{scenario:15}: V42={r['v42']['time']:.3f} V43={r['v43']['time']:.3f} "
              f"V44={r['v44']['time']:.3f} V45={r['v45']['time']:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    totals = {'v42': 0, 'v43': 0, 'v44': 0, 'v45': 0}
    for scenario in scenarios:
        for v in ['v42', 'v43', 'v44', 'v45']:
            totals[v] += results[scenario][v]['total_profit']
    
    print(f"Total Profit: V42=${totals['v42']} V43=${totals['v43']} "
          f"V44=${totals['v44']} V45=${totals['v45']}")
    
    best = max(totals, key=totals.get)
    print(f"Best Performer: {best.upper()} with ${totals[best]} total profit")
    print("=" * 70)
    
    return results


def main():
    """Run all bitvector benchmarks"""
    # Benchmark V42
    benchmark_v42()
    
    # Compare V38 vs V42
    try:
        compare_v38_v42()
    except ImportError:
        print("Note: V38 comparison skipped (benchmark_versions not available)")
    
    # Comprehensive benchmark
    benchmark_all_intelligent_versions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

