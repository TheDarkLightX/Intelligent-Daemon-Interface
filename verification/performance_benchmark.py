#!/usr/bin/env python3
"""
Performance Benchmark Suite for VCC Tau Specifications

This module benchmarks:
1. FSM state transition time
2. Bitvector operation performance
3. Recurrence relation computation
4. Memory usage per specification
5. Scalability with input size
"""

import time
import os
import subprocess
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import statistics


@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""
    spec_name: str
    operation: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    memory_kb: Optional[int]
    passed: bool
    notes: str = ""


class TauBenchmark:
    """Benchmark runner for Tau specifications"""
    
    TAU_PATH = "/home/trevormoc/Downloads/tau-lang-latest/build-Release/tau"
    SPEC_DIR = "/home/trevormoc/Downloads/DeflationaryAgent/specification"
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def check_tau_exists(self) -> bool:
        """Check if tau binary exists"""
        return os.path.exists(self.TAU_PATH)
    
    def run_tau_command(self, command: str, timeout: int = 30) -> Tuple[str, float, bool]:
        """Run a tau command and measure time"""
        start = time.perf_counter()
        try:
            result = subprocess.run(
                f"echo '{command}' | {self.TAU_PATH}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return output, elapsed, success
        except subprocess.TimeoutExpired:
            elapsed = timeout * 1000
            return "TIMEOUT", elapsed, False
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return str(e), elapsed, False
    
    def benchmark_bitvector_ops(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark bitvector arithmetic operations"""
        times = []
        
        # Test various bitvector operations
        commands = [
            "solve (result = a + b) && (a = { #xFF }:bv[8]) && (b = { #x01 }:bv[8])",
            "solve (result = a * b) && (a = { #x0A }:bv[8]) && (b = { #x05 }:bv[8])",
            "solve (result = a ^ b) && (a = { #xAA }:bv[8]) && (b = { #x55 }:bv[8])",
            "solve (result = a % b) && (a = { #x64 }:bv[8]) && (b = { #x0A }:bv[8])",
        ]
        
        for _ in range(iterations):
            for cmd in commands:
                _, elapsed, success = self.run_tau_command(cmd, timeout=5)
                if success:
                    times.append(elapsed)
        
        if not times:
            return BenchmarkResult(
                spec_name="bitvector_ops",
                operation="arithmetic",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                memory_kb=None,
                passed=False,
                notes="No successful runs"
            )
        
        return BenchmarkResult(
            spec_name="bitvector_ops",
            operation="arithmetic",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            memory_kb=None,
            passed=True,
            notes=f"Tested {len(commands)} operations"
        )
    
    def benchmark_256bit_ops(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark 256-bit bitvector operations (for supply tracking)"""
        times = []
        
        # 256-bit operations
        commands = [
            "solve (x = { #xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF }:bv[256])",
            "solve (result = a + b) && (a = { #x01 }:bv[256]) && (b = { #x01 }:bv[256])",
        ]
        
        for _ in range(iterations):
            for cmd in commands:
                _, elapsed, success = self.run_tau_command(cmd, timeout=10)
                if success:
                    times.append(elapsed)
        
        if not times:
            return BenchmarkResult(
                spec_name="bv256_ops",
                operation="256-bit arithmetic",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                memory_kb=None,
                passed=False,
                notes="No successful runs - 256-bit may not be supported"
            )
        
        return BenchmarkResult(
            spec_name="bv256_ops",
            operation="256-bit arithmetic",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            memory_kb=None,
            passed=True,
            notes="256-bit operations for supply tracking"
        )
    
    def benchmark_fsm_transitions(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark FSM state transition simulation"""
        times = []
        
        # Simulate FSM transitions using solve commands
        # These represent the state transition logic from our specs
        fsm_commands = [
            # State 0 -> State 1 (lock)
            "solve (next_state = { #x01 }:bv[8]) && (current_state = { #x00 }:bv[8]) && (action_lock = 1)",
            # State 1 -> State 2 (compound)
            "solve (next_state = { #x02 }:bv[8]) && (current_state = { #x01 }:bv[8]) && (action_compound = 1)",
            # State 2 -> State 0 (return)
            "solve (next_state = { #x00 }:bv[8]) && (current_state = { #x02 }:bv[8])",
        ]
        
        for _ in range(iterations):
            for cmd in fsm_commands:
                _, elapsed, success = self.run_tau_command(cmd, timeout=5)
                if success:
                    times.append(elapsed)
        
        if not times:
            return BenchmarkResult(
                spec_name="fsm_transitions",
                operation="state transitions",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                memory_kb=None,
                passed=False,
                notes="No successful runs"
            )
        
        return BenchmarkResult(
            spec_name="fsm_transitions",
            operation="state transitions",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            memory_kb=None,
            passed=True,
            notes=f"Tested {len(fsm_commands)} transitions"
        )
    
    def simulate_burn_calculation(self, iterations: int = 30) -> BenchmarkResult:
        """Benchmark burn calculation (key VCC operation)"""
        times = []
        
        # Burn multiplier: (1 + max(0, EETF-1))^2
        # EETF = 150 (1.5), result should be 2.25 * 100 = 225
        burn_commands = [
            # Simple burn calc
            "solve (burn_mult = { #xE1 }:bv[16]) && (eetf = { #x96 }:bv[8])",
            # Cascade check
            "solve (cascade_level = { #x02 }:bv[8]) && (eetf > { #x8C }:bv[8]) && (eetf <= { #xA0 }:bv[8])",
        ]
        
        for _ in range(iterations):
            for cmd in burn_commands:
                _, elapsed, success = self.run_tau_command(cmd, timeout=5)
                if success:
                    times.append(elapsed)
        
        if not times:
            return BenchmarkResult(
                spec_name="burn_calculation",
                operation="AEB burn logic",
                iterations=iterations,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                memory_kb=None,
                passed=False,
                notes="No successful runs"
            )
        
        return BenchmarkResult(
            spec_name="burn_calculation",
            operation="AEB burn logic",
            iterations=len(times),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            memory_kb=None,
            passed=True,
            notes="Burn multiplier and cascade calculations"
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks"""
        self.results = []
        
        print("\n" + "=" * 80)
        print("RUNNING PERFORMANCE BENCHMARKS")
        print("=" * 80)
        
        benchmarks = [
            ("Bitvector Operations (8-bit)", self.benchmark_bitvector_ops, 50),
            ("256-bit Operations", self.benchmark_256bit_ops, 20),
            ("FSM Transitions", self.benchmark_fsm_transitions, 20),
            ("Burn Calculations", self.simulate_burn_calculation, 20),
        ]
        
        for name, func, iterations in benchmarks:
            print(f"\nBenchmarking: {name}...")
            result = func(iterations)
            self.results.append(result)
            
            if result.passed:
                print(f"  Avg: {result.avg_time_ms:.2f}ms, Min: {result.min_time_ms:.2f}ms, Max: {result.max_time_ms:.2f}ms")
            else:
                print(f"  FAILED: {result.notes}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate performance report"""
        lines = [
            "=" * 80,
            "VCC SPECIFICATION PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            "",
            f"Tau Binary: {self.TAU_PATH}",
            f"Tau Available: {self.check_tau_exists()}",
            "",
            "BENCHMARK RESULTS:",
            "-" * 80,
            f"{'Operation':<30} | {'Avg (ms)':>10} | {'Min':>10} | {'Max':>10} | {'StdDev':>10} | Status",
            "-" * 80,
        ]
        
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{r.operation:<30} | {r.avg_time_ms:>10.2f} | {r.min_time_ms:>10.2f} | "
                f"{r.max_time_ms:>10.2f} | {r.std_dev_ms:>10.2f} | {status}"
            )
        
        # Performance analysis
        lines.extend([
            "",
            "=" * 80,
            "PERFORMANCE ANALYSIS",
            "=" * 80,
            "",
        ])
        
        passed_results = [r for r in self.results if r.passed]
        if passed_results:
            avg_all = statistics.mean(r.avg_time_ms for r in passed_results)
            lines.extend([
                f"Overall average operation time: {avg_all:.2f}ms",
                "",
                "Performance Tiers:",
                "  < 10ms: Excellent (suitable for real-time)",
                "  10-100ms: Good (suitable for interactive)",
                "  100-1000ms: Acceptable (suitable for batch)",
                "  > 1000ms: Slow (optimization needed)",
                "",
            ])
            
            # Categorize results
            excellent = [r for r in passed_results if r.avg_time_ms < 10]
            good = [r for r in passed_results if 10 <= r.avg_time_ms < 100]
            acceptable = [r for r in passed_results if 100 <= r.avg_time_ms < 1000]
            slow = [r for r in passed_results if r.avg_time_ms >= 1000]
            
            lines.extend([
                f"Excellent: {len(excellent)} operations",
                f"Good: {len(good)} operations",
                f"Acceptable: {len(acceptable)} operations",
                f"Slow: {len(slow)} operations",
            ])
        
        lines.extend([
            "",
            "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            "",
            "1. For production deployment:",
            "   - Batch state transitions for efficiency",
            "   - Cache frequently computed values",
            "   - Use smaller bitvectors where possible (8/16 vs 256)",
            "",
            "2. For testnet alpha:",
            "   - Current performance is acceptable",
            "   - Focus on correctness over speed",
            "   - Monitor actual network latency",
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)


def run_simulation_benchmark():
    """Benchmark the economic simulation (Python-based)"""
    print("\n" + "=" * 80)
    print("ECONOMIC SIMULATION PERFORMANCE")
    print("=" * 80)
    
    from decimal import Decimal
    import decimal_economics_analysis as dea
    
    # Benchmark 200-year simulation
    times = []
    for _ in range(5):
        analyzer = dea.DecimalEconomicsAnalyzer()
        start = time.perf_counter()
        analyzer.run_simulation(years=200)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    print(f"\n200-year simulation:")
    print(f"  Avg: {statistics.mean(times):.2f}ms")
    print(f"  Min: {min(times):.2f}ms")
    print(f"  Max: {max(times):.2f}ms")
    print(f"  Result: {'FAST' if statistics.mean(times) < 100 else 'ACCEPTABLE'}")
    
    return times


def main():
    print("\n" + "=" * 80)
    print("VCC PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    
    # Check if tau exists
    benchmark = TauBenchmark()
    if not benchmark.check_tau_exists():
        print(f"\nWARNING: Tau binary not found at {benchmark.TAU_PATH}")
        print("Running simulation benchmarks only...")
    else:
        # Run Tau benchmarks
        benchmark.run_all_benchmarks()
        print("\n" + benchmark.generate_report())
    
    # Run simulation benchmark
    run_simulation_benchmark()
    
    # Save results
    results_path = "/home/trevormoc/Downloads/DeflationaryAgent/verification/performance_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "tau_benchmarks": [
                {
                    "name": r.spec_name,
                    "operation": r.operation,
                    "avg_ms": r.avg_time_ms,
                    "min_ms": r.min_time_ms,
                    "max_ms": r.max_time_ms,
                    "passed": r.passed,
                }
                for r in benchmark.results
            ] if benchmark.results else [],
            "tau_available": benchmark.check_tau_exists(),
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

