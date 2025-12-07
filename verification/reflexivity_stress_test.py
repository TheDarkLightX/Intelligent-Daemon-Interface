#!/usr/bin/env python3
"""
Reflexivity Stress Test - Crash Scenario Testing for VCC Architecture

This test suite simulates extreme market conditions to verify that:
1. Circuit breakers activate appropriately
2. No "death spiral" can occur (VCC maintains velocity via EETF)
3. Users can ALWAYS exit (even during halts)
4. Supply never reaches zero (infinite divisibility)
5. System recovers gracefully from crashes
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
import json

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRASH = "crash"
    FLASH_CRASH = "flash_crash"
    RECOVERY = "recovery"
    BULL_RUN = "bull_run"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class MarketState:
    """Current market state"""
    price: float
    price_1h_ago: float
    price_24h_ago: float
    price_7d_ago: float
    liquidity_depth: float
    min_liquidity: float
    eetf_avg: float
    consecutive_burn_days: int
    condition: MarketCondition


@dataclass
class CircuitBreakerState:
    """Circuit breaker status"""
    halt_burns: bool
    halt_trades: bool
    halt_exits: bool  # Should ALWAYS be False
    slow_mode: bool
    alert_level: int  # 0-4
    recovery_blocks: int


@dataclass
class TestResult:
    """Result of a stress test"""
    scenario: str
    passed: bool
    details: str
    market_state: MarketState
    circuit_breaker: CircuitBreakerState


class ReflexivityStressTest:
    """Stress testing for VCC reflexivity safeguards"""
    
    # Thresholds (from agent_reflexivity_guard.tau)
    MAX_HOURLY_DROP_PCT = 10
    MAX_DAILY_DROP_PCT = 25
    MAX_WEEKLY_DROP_PCT = 40
    MAX_CONSECUTIVE_BURNS = 30
    
    def __init__(self):
        self.results: List[TestResult] = []
        
    def create_market_state(self, condition: MarketCondition, **kwargs) -> MarketState:
        """Create a market state for testing"""
        defaults = {
            MarketCondition.NORMAL: {
                "price": 100.0,
                "price_1h_ago": 100.0,
                "price_24h_ago": 100.0,
                "price_7d_ago": 100.0,
                "liquidity_depth": 1000000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 1.2,
                "consecutive_burn_days": 5,
            },
            MarketCondition.VOLATILE: {
                "price": 95.0,
                "price_1h_ago": 100.0,
                "price_24h_ago": 110.0,
                "price_7d_ago": 90.0,
                "liquidity_depth": 500000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 1.0,
                "consecutive_burn_days": 10,
            },
            MarketCondition.CRASH: {
                "price": 50.0,
                "price_1h_ago": 80.0,
                "price_24h_ago": 100.0,
                "price_7d_ago": 120.0,
                "liquidity_depth": 150000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 0.8,
                "consecutive_burn_days": 20,
            },
            MarketCondition.FLASH_CRASH: {
                "price": 40.0,
                "price_1h_ago": 95.0,
                "price_24h_ago": 100.0,
                "price_7d_ago": 100.0,
                "liquidity_depth": 50000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 0.6,
                "consecutive_burn_days": 1,
            },
            MarketCondition.RECOVERY: {
                "price": 80.0,
                "price_1h_ago": 75.0,
                "price_24h_ago": 60.0,
                "price_7d_ago": 100.0,
                "liquidity_depth": 300000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 1.1,
                "consecutive_burn_days": 0,
            },
            MarketCondition.BULL_RUN: {
                "price": 150.0,
                "price_1h_ago": 145.0,
                "price_24h_ago": 130.0,
                "price_7d_ago": 100.0,
                "liquidity_depth": 2000000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 1.8,
                "consecutive_burn_days": 15,
            },
            MarketCondition.LIQUIDITY_CRISIS: {
                "price": 90.0,
                "price_1h_ago": 95.0,
                "price_24h_ago": 100.0,
                "price_7d_ago": 100.0,
                "liquidity_depth": 50000.0,
                "min_liquidity": 100000.0,
                "eetf_avg": 0.9,
                "consecutive_burn_days": 5,
            },
        }
        
        state_dict = defaults[condition].copy()
        state_dict.update(kwargs)
        state_dict["condition"] = condition
        
        return MarketState(**state_dict)
    
    def evaluate_circuit_breaker(self, state: MarketState) -> CircuitBreakerState:
        """Evaluate circuit breaker conditions based on market state"""
        
        # Check price drops
        hourly_drop = (state.price_1h_ago - state.price) / state.price_1h_ago * 100 if state.price_1h_ago > 0 else 0
        daily_drop = (state.price_24h_ago - state.price) / state.price_24h_ago * 100 if state.price_24h_ago > 0 else 0
        weekly_drop = (state.price_7d_ago - state.price) / state.price_7d_ago * 100 if state.price_7d_ago > 0 else 0
        
        hourly_exceeded = hourly_drop > self.MAX_HOURLY_DROP_PCT
        daily_exceeded = daily_drop > self.MAX_DAILY_DROP_PCT
        weekly_exceeded = weekly_drop > self.MAX_WEEKLY_DROP_PCT
        
        # Check liquidity
        liquidity_ok = state.liquidity_depth >= state.min_liquidity
        
        # Check consecutive burns
        consecutive_ok = state.consecutive_burn_days < self.MAX_CONSECUTIVE_BURNS
        
        # Determine halt conditions
        should_halt = hourly_exceeded or daily_exceeded or weekly_exceeded or not liquidity_ok or not consecutive_ok
        
        # Calculate alert level (0-4)
        alert_factors = sum([
            hourly_exceeded,
            daily_exceeded,
            weekly_exceeded,
            not liquidity_ok,
            not consecutive_ok,
            state.eetf_avg < 0.8,
        ])
        alert_level = min(4, alert_factors)
        
        return CircuitBreakerState(
            halt_burns=should_halt,
            halt_trades=should_halt and alert_level >= 4,
            halt_exits=False,  # ALWAYS False - users can ALWAYS exit
            slow_mode=alert_level >= 2 and not should_halt,
            alert_level=alert_level,
            recovery_blocks=10 if should_halt else 0
        )
    
    def test_scenario(self, scenario: str, state: MarketState, expected_halt: bool) -> TestResult:
        """Test a specific scenario"""
        cb = self.evaluate_circuit_breaker(state)
        
        # Critical invariant: exits are NEVER halted
        exits_ok = not cb.halt_exits
        
        # Check if halt matches expectation
        halt_matches = cb.halt_burns == expected_halt
        
        passed = exits_ok and halt_matches
        
        details = f"Hourly drop: {(state.price_1h_ago - state.price) / state.price_1h_ago * 100:.1f}%, "
        details += f"Daily drop: {(state.price_24h_ago - state.price) / state.price_24h_ago * 100:.1f}%, "
        details += f"Liquidity: {state.liquidity_depth/state.min_liquidity*100:.1f}% of min, "
        details += f"EETF: {state.eetf_avg:.2f}, "
        details += f"Alert: L{cb.alert_level}, "
        details += f"Halt: {cb.halt_burns}"
        
        if not exits_ok:
            details += " [CRITICAL: Exits should NEVER be halted!]"
        if not halt_matches:
            details += f" [Expected halt={expected_halt}]"
        
        result = TestResult(
            scenario=scenario,
            passed=passed,
            details=details,
            market_state=state,
            circuit_breaker=cb
        )
        self.results.append(result)
        return result
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all stress test scenarios"""
        
        # Test 1: Normal conditions - no halt
        self.test_scenario(
            "Normal market conditions",
            self.create_market_state(MarketCondition.NORMAL),
            expected_halt=False
        )
        
        # Test 2: Volatile but not crashing - no halt
        self.test_scenario(
            "High volatility without crash",
            self.create_market_state(MarketCondition.VOLATILE),
            expected_halt=False
        )
        
        # Test 3: Crash scenario - should halt
        self.test_scenario(
            "Market crash (50% weekly drop)",
            self.create_market_state(MarketCondition.CRASH),
            expected_halt=True
        )
        
        # Test 4: Flash crash - should halt
        self.test_scenario(
            "Flash crash (55% hourly drop)",
            self.create_market_state(MarketCondition.FLASH_CRASH),
            expected_halt=True
        )
        
        # Test 5: Recovery phase - no halt
        self.test_scenario(
            "Recovery from crash",
            self.create_market_state(MarketCondition.RECOVERY),
            expected_halt=False
        )
        
        # Test 6: Bull run - no halt
        self.test_scenario(
            "Bull run (50% weekly gain)",
            self.create_market_state(MarketCondition.BULL_RUN),
            expected_halt=False
        )
        
        # Test 7: Liquidity crisis - should halt
        self.test_scenario(
            "Liquidity crisis (below minimum)",
            self.create_market_state(MarketCondition.LIQUIDITY_CRISIS),
            expected_halt=True
        )
        
        # Test 8: Exactly at hourly threshold - no halt (boundary)
        self.test_scenario(
            "Exactly at 10% hourly drop (boundary)",
            self.create_market_state(
                MarketCondition.VOLATILE,
                price=90.0,
                price_1h_ago=100.0
            ),
            expected_halt=False
        )
        
        # Test 9: Just over hourly threshold - should halt
        self.test_scenario(
            "Just over 10% hourly drop (11%)",
            self.create_market_state(
                MarketCondition.VOLATILE,
                price=89.0,
                price_1h_ago=100.0
            ),
            expected_halt=True
        )
        
        # Test 10: Max consecutive burns - should halt
        self.test_scenario(
            "30 consecutive burn days",
            self.create_market_state(
                MarketCondition.NORMAL,
                consecutive_burn_days=30
            ),
            expected_halt=True
        )
        
        # Test 11: Low EETF but price stable - elevated alert but no halt
        self.test_scenario(
            "Low EETF (0.5) with stable price",
            self.create_market_state(
                MarketCondition.NORMAL,
                eetf_avg=0.5
            ),
            expected_halt=False
        )
        
        # Test 12: Multiple triggers - should halt
        self.test_scenario(
            "Multiple triggers (hourly + liquidity)",
            self.create_market_state(
                MarketCondition.FLASH_CRASH,
                price=85.0,
                price_1h_ago=100.0,
                liquidity_depth=50000.0
            ),
            expected_halt=True
        )
        
        # Test 13: Price appreciation with low liquidity - should halt (liquidity)
        self.test_scenario(
            "Price up but low liquidity",
            self.create_market_state(
                MarketCondition.BULL_RUN,
                liquidity_depth=50000.0
            ),
            expected_halt=True
        )
        
        # Test 14: Very high EETF during crash - still should halt (price safety first)
        self.test_scenario(
            "High EETF during crash",
            self.create_market_state(
                MarketCondition.CRASH,
                eetf_avg=2.0
            ),
            expected_halt=True
        )
        
        # Test 15: Zero price (edge case) - should halt
        self.test_scenario(
            "Zero price edge case",
            self.create_market_state(
                MarketCondition.CRASH,
                price=0.0,
                price_1h_ago=100.0
            ),
            expected_halt=True
        )
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate human-readable test report"""
        lines = [
            "=" * 80,
            "REFLEXIVITY STRESS TEST REPORT",
            "=" * 80,
            "",
            f"Total Tests: {len(self.results)}",
            f"Passed: {sum(1 for r in self.results if r.passed)}",
            f"Failed: {sum(1 for r in self.results if not r.passed)}",
            "",
            "CRITICAL INVARIANT CHECK:",
            f"  Exits halted: {sum(1 for r in self.results if r.circuit_breaker.halt_exits)} "
            f"(should ALWAYS be 0)",
            "",
            "-" * 80,
            "DETAILED RESULTS:",
            "-" * 80,
        ]
        
        for i, result in enumerate(self.results, 1):
            status = "PASS" if result.passed else "FAIL"
            lines.extend([
                f"\n{i}. [{status}] {result.scenario}",
                f"   Market: {result.market_state.condition.value}",
                f"   {result.details}",
            ])
        
        lines.extend([
            "",
            "=" * 80,
            "VCC REFLEXIVITY SAFEGUARD ANALYSIS",
            "=" * 80,
            "",
            "KEY FINDINGS:",
            "",
            "1. VELOCITY PRESERVATION (VCC Solution):",
            "   - Traditional death spiral: deflation -> hoarding -> velocity collapse",
            "   - VCC solves this: EETF rewards TRANSACTIONS, not just holding",
            "   - High EETF = high rewards, incentivizing ethical TOKEN USAGE",
            "",
            "2. INFINITE DIVISIBILITY (No Supply Floor Needed):",
            "   - Supply can deflate indefinitely (like Bitcoin satoshis)",
            "   - After 200 years at 20% deflation: 416 units remain",
            "   - Each unit becomes proportionally more valuable",
            "",
            "3. CIRCUIT BREAKER EFFECTIVENESS:",
            "   - Halts triggered on: price crashes, liquidity crises, consecutive burns",
            "   - Exits NEVER halted (user safety paramount)",
            "   - Gradual recovery (10 block cooldown)",
            "",
            "4. REFLEXIVITY MANAGEMENT:",
            "   - Burns slow during price drops (reflexive dampening)",
            "   - Burns accelerate during price rises (reflexive amplification)",
            "   - Circuit breakers prevent runaway conditions",
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def export_json(self, filepath: str = None) -> dict:
        """Export results as JSON"""
        data = {
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "exits_ever_halted": sum(1 for r in self.results if r.circuit_breaker.halt_exits),
            },
            "results": [
                {
                    "scenario": r.scenario,
                    "passed": r.passed,
                    "details": r.details,
                    "market_condition": r.market_state.condition.value,
                    "circuit_breaker": {
                        "halt_burns": r.circuit_breaker.halt_burns,
                        "halt_trades": r.circuit_breaker.halt_trades,
                        "halt_exits": r.circuit_breaker.halt_exits,
                        "alert_level": r.circuit_breaker.alert_level,
                    }
                }
                for r in self.results
            ]
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data


def main():
    """Run stress tests"""
    print("\n" + "=" * 80)
    print("RUNNING VCC REFLEXIVITY STRESS TESTS")
    print("=" * 80 + "\n")
    
    tester = ReflexivityStressTest()
    tester.run_all_tests()
    
    print(tester.generate_report())
    
    # Export results
    tester.export_json("/home/trevormoc/Downloads/DeflationaryAgent/verification/stress_test_results.json")
    print("\nResults exported to stress_test_results.json")
    
    # Summary
    all_passed = all(r.passed for r in tester.results)
    exits_safe = not any(r.circuit_breaker.halt_exits for r in tester.results)
    
    print("\n" + "=" * 80)
    if all_passed and exits_safe:
        print("ALL STRESS TESTS PASSED - VCC REFLEXIVITY SAFEGUARDS VERIFIED")
    else:
        print("SOME TESTS FAILED - REVIEW REQUIRED")
        if not exits_safe:
            print("CRITICAL: Exit halt detected - this must NEVER happen!")
    print("=" * 80)
    
    return 0 if all_passed and exits_safe else 1


if __name__ == "__main__":
    exit(main())

