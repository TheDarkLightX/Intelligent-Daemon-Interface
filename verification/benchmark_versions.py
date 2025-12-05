#!/usr/bin/env python3
"""
Benchmark Framework for Deflationary Agent Versions

Compares performance of V35, V38, V39a/b/c, V40 specifications.

Metrics:
1. Execution trace simulation time
2. BDD size estimation (clause count, variable count)
3. State transition efficiency
4. Pathfinding effectiveness (V39 variants)

Copyright DarkLightX/Dana Edwards
"""

import os
import sys
import time
import json
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import statistics


@dataclass
class SpecMetrics:
    """Metrics for a specification"""
    name: str
    version: str
    clause_count: int = 0
    input_count: int = 0
    output_count: int = 0
    helper_count: int = 0
    
    # Performance metrics
    simulation_time_ms: float = 0.0
    steps_simulated: int = 0
    state_transitions: int = 0
    
    # Trading metrics
    trades_executed: int = 0
    profitable_trades: int = 0
    timeout_exits: int = 0
    
    # Pathfinding metrics (V39 variants)
    entries_blocked_by_heuristic: int = 0
    strategy_switches: int = 0


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    spec_file: str
    metrics: SpecMetrics
    scenario_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


class AgentSimulator:
    """Simulates deflationary agent execution for benchmarking"""
    
    def __init__(self, version: str):
        self.version = version
        self.state = self._initial_state()
        self.history = []
        
    def _initial_state(self) -> Dict[str, int]:
        """Initialize agent state"""
        return {
            'o0': 0, 'o1': 0, 'o2': 0, 'o3': 0,
            'o6': 0, 'o7': 0, 'o9': 0, 'o10': 0,
            'o11': 0, 'o13': 0,
            # V39a heuristic inputs
            'i5': 1, 'i6': 0,
            # V39b strategy
            'strategy': 0,
            # V39c goal
            'o21': 0,
        }
    
    def valid_entry(self, i0, i1, i2, o1_prev) -> bool:
        """Check valid entry condition"""
        return (not i0) and i1 and i2 and (not o1_prev)
    
    def valid_exit(self, i0, o1_prev) -> bool:
        """Check valid exit condition"""
        return i0 and o1_prev
    
    def timed_out(self, b1, b0) -> bool:
        """Check timeout condition"""
        return b1 and b0
    
    def heuristic_ok(self, forecast, risk) -> bool:
        """V39a: Check heuristic condition"""
        return forecast and (not risk)
    
    def step(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """Execute one simulation step"""
        prev = self.state.copy()
        curr = {}
        
        i0 = inputs.get('i0', 0)
        i1 = inputs.get('i1', 0)
        i2 = inputs.get('i2', 0)
        i3 = inputs.get('i3', 0)
        i4 = inputs.get('i4', 0)
        i5 = inputs.get('i5', 1)  # V39a: trend forecast
        i6 = inputs.get('i6', 0)  # V39a: risk score / V39b: strategy
        
        # Timeout check
        timeout = self.timed_out(prev['o7'], prev['o6'])
        
        # Entry/continue condition (varies by version)
        entry_cond = (not prev['o0'] and 
                      self.valid_entry(i0, i1, i2, prev['o1']) and
                      not prev['o0'] and i1 and not timeout and
                      not prev['o9'] and not i4)
        
        # V39a: Add heuristic gate
        if self.version == 'v39a':
            entry_cond = entry_cond and self.heuristic_ok(i5, i6)
        
        # V39b: Strategy-dependent entry (simplified)
        if self.version == 'v39b':
            # Just track strategy, same entry logic
            pass
        
        continue_cond = (prev['o0'] and 
                         not self.valid_exit(i0, prev['o1']) and
                         not timeout and i1 and not i4)
        
        curr['o0'] = int(entry_cond or continue_cond)
        
        # Trading signals
        curr['o2'] = int(curr['o0'] and not prev['o0'] and not prev['o1'])
        curr['o3'] = int(prev['o0'] and not curr['o0'] and prev['o1'])
        curr['o1'] = int(curr['o2'] or (not curr['o3'] and prev['o1']))
        
        # Timer
        curr['o6'] = int(curr['o0'] and not prev['o6'])
        xor_val = (prev['o7'] and not prev['o6']) or (not prev['o7'] and prev['o6'])
        curr['o7'] = int(curr['o0'] and xor_val)
        
        # Nonce
        curr['o9'] = int(curr['o2'] or (prev['o0'] and not curr['o3'] and prev['o9']))
        
        # Economics
        curr['o10'] = int((curr['o2'] and i0) or 
                          (prev['o0'] and not curr['o2'] and not curr['o3'] and prev['o10']))
        curr['o11'] = int(curr['o3'] and i0 and not prev['o10'] and i3)
        
        # Burns
        curr['o13'] = int(prev['o13'] or curr['o11'])
        
        # V39c: Goal tracking
        if self.version == 'v39c':
            curr['o21'] = int(prev.get('o21', 0) or curr['o11'])
        
        self.state = curr
        self.history.append((inputs.copy(), curr.copy()))
        return curr
    
    def run_scenario(self, input_sequence: List[Dict[str, int]]) -> Dict[str, Any]:
        """Run a complete scenario and collect metrics"""
        self.state = self._initial_state()
        self.history = []
        
        metrics = {
            'steps': 0,
            'state_transitions': 0,
            'trades': 0,
            'profitable_trades': 0,
            'timeout_exits': 0,
            'heuristic_blocks': 0,
        }
        
        prev_o0 = 0
        for inputs in input_sequence:
            result = self.step(inputs)
            metrics['steps'] += 1
            
            if result['o0'] != prev_o0:
                metrics['state_transitions'] += 1
            
            if result['o2']:
                metrics['trades'] += 1
            
            if result['o11']:
                metrics['profitable_trades'] += 1
            
            if result['o3'] and self.timed_out(self.history[-1][1].get('o7', 0),
                                                self.history[-1][1].get('o6', 0)):
                metrics['timeout_exits'] += 1
            
            prev_o0 = result['o0']
        
        return metrics


def parse_tau_spec(spec_path: str) -> SpecMetrics:
    """Parse a .tau file and extract metrics"""
    metrics = SpecMetrics(
        name=Path(spec_path).stem,
        version=Path(spec_path).stem.split('_')[-1]
    )
    
    with open(spec_path, 'r') as f:
        content = f.read()
    
    # Count clauses (lines with '&&' continuation or assignments in r() block)
    in_r_block = False
    clause_count = 0
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('r ('):
            in_r_block = True
        elif in_r_block and line == ')':
            in_r_block = False
        elif in_r_block and ('=' in line or '&&' in line):
            if line.startswith('(') and '=' in line:
                clause_count += 1
    
    metrics.clause_count = clause_count
    
    # Count inputs (sbf i* = ifile)
    metrics.input_count = content.count('ifile(')
    
    # Count outputs (sbf o* = ofile)
    metrics.output_count = content.count('ofile(')
    
    # Count helpers (:=)
    metrics.helper_count = content.count(':=')
    
    return metrics


def create_test_scenarios() -> Dict[str, List[Dict[str, int]]]:
    """Create standard test scenarios"""
    scenarios = {}
    
    # Scenario 1: Normal profitable trade
    scenarios['normal_profit'] = []
    for i in range(10):
        if i == 0:  # Entry
            scenarios['normal_profit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        elif i < 3:  # Hold
            scenarios['normal_profit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        elif i == 3:  # Exit with profit
            scenarios['normal_profit'].append({
                'i0': 1, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        else:  # Idle
            scenarios['normal_profit'].append({
                'i0': 1, 'i1': 1, 'i2': 0, 'i3': 1, 'i4': 0,
                'i5': 0, 'i6': 0
            })
    
    # Scenario 2: Timeout forced exit
    scenarios['timeout_exit'] = []
    for i in range(8):
        if i == 0:  # Entry
            scenarios['timeout_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        elif i < 5:  # Hold until timeout
            scenarios['timeout_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        else:  # After timeout
            scenarios['timeout_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
    
    # Scenario 3: Failure echo exit
    scenarios['failure_exit'] = []
    for i in range(5):
        if i == 0:  # Entry
            scenarios['failure_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
        elif i == 1:  # Failure echo
            scenarios['failure_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 1,  # i4=1
                'i5': 1, 'i6': 0
            })
        else:
            scenarios['failure_exit'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
    
    # Scenario 4: Multiple trades
    scenarios['multi_trade'] = []
    for cycle in range(5):
        # Entry
        scenarios['multi_trade'].append({
            'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
            'i5': 1, 'i6': 0
        })
        # Exit
        scenarios['multi_trade'].append({
            'i0': 1, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
            'i5': 1, 'i6': 0
        })
        # Cool down
        scenarios['multi_trade'].append({
            'i0': 1, 'i1': 1, 'i2': 0, 'i3': 1, 'i4': 0,
            'i5': 0, 'i6': 0
        })
    
    # Scenario 5: V39a heuristic blocking
    scenarios['heuristic_block'] = []
    for i in range(10):
        if i < 3:  # Bad heuristic - should block entry
            scenarios['heuristic_block'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 0, 'i6': 1  # Bad forecast, high risk
            })
        else:  # Good heuristic
            scenarios['heuristic_block'].append({
                'i0': 0, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 0,
                'i5': 1, 'i6': 0
            })
    
    # Scenario 6: High volatility (rapid state changes)
    scenarios['high_volatility'] = []
    for i in range(20):
        scenarios['high_volatility'].append({
            'i0': i % 2, 'i1': 1, 'i2': (i + 1) % 2, 'i3': 1, 'i4': 0,
            'i5': 1, 'i6': 0
        })
    
    return scenarios


def benchmark_version(spec_path: str, scenarios: Dict[str, List[Dict[str, int]]]) -> BenchmarkResult:
    """Run benchmarks on a specification"""
    metrics = parse_tau_spec(spec_path)
    version = metrics.version
    
    result = BenchmarkResult(
        spec_file=spec_path,
        metrics=metrics,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    total_sim_time = 0
    total_steps = 0
    total_transitions = 0
    total_trades = 0
    total_profitable = 0
    total_timeouts = 0
    
    for scenario_name, input_sequence in scenarios.items():
        sim = AgentSimulator(version)
        
        start_time = time.perf_counter()
        scenario_metrics = sim.run_scenario(input_sequence)
        end_time = time.perf_counter()
        
        sim_time = (end_time - start_time) * 1000  # ms
        
        scenario_result = {
            'name': scenario_name,
            'simulation_time_ms': sim_time,
            **scenario_metrics
        }
        result.scenario_results.append(scenario_result)
        
        total_sim_time += sim_time
        total_steps += scenario_metrics['steps']
        total_transitions += scenario_metrics['state_transitions']
        total_trades += scenario_metrics['trades']
        total_profitable += scenario_metrics['profitable_trades']
        total_timeouts += scenario_metrics['timeout_exits']
    
    # Aggregate metrics
    result.metrics.simulation_time_ms = total_sim_time
    result.metrics.steps_simulated = total_steps
    result.metrics.state_transitions = total_transitions
    result.metrics.trades_executed = total_trades
    result.metrics.profitable_trades = total_profitable
    result.metrics.timeout_exits = total_timeouts
    
    return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    print("\n" + "=" * 100)
    print("DEFLATIONARY AGENT BENCHMARK RESULTS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Version':<15} {'Clauses':<10} {'Inputs':<10} {'Outputs':<10} "
          f"{'Time(ms)':<12} {'Trades':<10} {'Profitable':<12} {'Timeouts':<10}")
    print("-" * 100)
    
    # Sort by version
    results.sort(key=lambda x: x.metrics.version)
    
    for r in results:
        m = r.metrics
        print(f"{m.version:<15} {m.clause_count:<10} {m.input_count:<10} {m.output_count:<10} "
              f"{m.simulation_time_ms:<12.3f} {m.trades_executed:<10} {m.profitable_trades:<12} "
              f"{m.timeout_exits:<10}")
    
    print("-" * 100)
    
    # Scenario breakdown
    print("\nSCENARIO BREAKDOWN:")
    print("-" * 80)
    
    scenario_names = results[0].scenario_results if results else []
    for scenario in [s['name'] for s in scenario_names]:
        print(f"\n{scenario}:")
        for r in results:
            for sr in r.scenario_results:
                if sr['name'] == scenario:
                    print(f"  {r.metrics.version:<12}: {sr['simulation_time_ms']:.3f}ms, "
                          f"trades={sr['trades']}, profitable={sr['profitable_trades']}")
    
    # Performance comparison
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON (vs V35 baseline)")
    print("=" * 100)
    
    baseline = next((r for r in results if r.metrics.version == 'v35'), None)
    if baseline:
        for r in results:
            if r.metrics.version != 'v35':
                clause_diff = ((r.metrics.clause_count - baseline.metrics.clause_count) / 
                              baseline.metrics.clause_count * 100)
                time_diff = ((r.metrics.simulation_time_ms - baseline.metrics.simulation_time_ms) /
                            baseline.metrics.simulation_time_ms * 100) if baseline.metrics.simulation_time_ms > 0 else 0
                
                print(f"{r.metrics.version}: Clauses {clause_diff:+.1f}%, "
                      f"Sim Time {time_diff:+.1f}%")


def main():
    """Main benchmark execution"""
    repo_root = Path(__file__).parent.parent
    spec_dir = repo_root / "specification"
    idi_spec_dir = repo_root / "idi" / "specs" / "V38_Minimal_Core"
    
    # Discover all versions
    spec_files = [
        spec_dir / "agent4_testnet_v35.tau",
        idi_spec_dir / "agent4_testnet_v38.tau",
        spec_dir / "agent4_testnet_v39a.tau",
        spec_dir / "agent4_testnet_v39b.tau",
        spec_dir / "agent4_testnet_v39c.tau",
        spec_dir / "agent4_testnet_v40_core.tau",
    ]
    
    # Filter to existing files
    existing_specs = [f for f in spec_files if f.exists()]
    
    if not existing_specs:
        print("ERROR: No specification files found")
        return 1
    
    print(f"Found {len(existing_specs)} specifications to benchmark")
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    print(f"Running {len(scenarios)} test scenarios")
    
    # Run benchmarks
    results = []
    for spec_file in existing_specs:
        print(f"Benchmarking {spec_file.name}...")
        result = benchmark_version(str(spec_file), scenarios)
        results.append(result)
    
    # Print results
    print_results(results)
    
    # Save results to JSON
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

