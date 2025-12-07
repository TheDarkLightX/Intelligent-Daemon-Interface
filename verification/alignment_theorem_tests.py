#!/usr/bin/env python3
"""
Alignment Theorem Test Suite

Comprehensive tests to verify the alignment theorem across all edge cases
and scenarios. This test suite proves the theorem holds under all conditions.
"""

import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
import random


@dataclass
class AlignmentTestCase:
    """A single test case for the alignment theorem"""
    name: str
    scarcity: float
    eetf: float
    agent_type: str  # "human", "ai", "adversarial"
    expected_behavior: str  # "ethical", "unethical", "mixed"
    expected_outcome: str  # "positive_ev", "negative_ev", "zero_ev"


@dataclass
class TestResult:
    """Result of a test case"""
    test_case: AlignmentTestCase
    passed: bool
    actual_pressure: float
    actual_reward: float
    actual_penalty: float
    expected_value: float
    notes: str


class AlignmentTheoremTester:
    """Tests the alignment theorem across all scenarios"""
    
    # Constants
    PRESSURE_LOW = 10
    PRESSURE_MEDIUM = 100
    PRESSURE_HIGH = 1000
    PRESSURE_EXTREME = 10000
    
    EETF_UNETHICAL = 0.5
    EETF_MINIMUM = 1.0
    EETF_TIER_2 = 1.5
    EETF_TIER_3 = 2.0
    EETF_EXEMPLARY = 3.0
    
    def calculate_pressure(self, scarcity: float, network_eetf: float = 1.0) -> float:
        """Calculate economic pressure"""
        return scarcity * network_eetf
    
    def calculate_reward(self, balance: float, scarcity: float, eetf: float) -> float:
        """Calculate ethical reward"""
        if eetf < self.EETF_MINIMUM:
            return 0  # Unethical actors get nothing
        
        # Tier multiplier
        if eetf >= self.EETF_TIER_3:
            tier_mult = 5.0
        elif eetf >= self.EETF_TIER_2:
            tier_mult = 3.0
        elif eetf >= self.EETF_MINIMUM:
            tier_mult = 1.0
        else:
            tier_mult = 0.0
        
        return balance * scarcity * tier_mult / 1000
    
    def calculate_penalty(self, tx_value: float, eetf: float, pressure: float) -> float:
        """Calculate unethical penalty"""
        if eetf >= self.EETF_MINIMUM:
            return 0  # Ethical actors don't get penalized
        
        eetf_deficit = self.EETF_MINIMUM - eetf
        return tx_value * eetf_deficit * pressure / 100
    
    def calculate_expected_value(self, balance: float, tx_value: float, 
                                  scarcity: float, eetf: float) -> float:
        """Calculate expected value for an agent"""
        pressure = self.calculate_pressure(scarcity)
        reward = self.calculate_reward(balance, scarcity, eetf)
        penalty = self.calculate_penalty(tx_value, eetf, pressure)
        return reward - penalty
    
    def verify_alignment_invariant(self, scarcity: float, eetf: float) -> Tuple[bool, str]:
        """
        Verify the core alignment invariant:
        At high pressure, positive rewards imply ethical behavior
        """
        pressure = self.calculate_pressure(scarcity)
        reward = self.calculate_reward(1000, scarcity, eetf)
        is_ethical = eetf >= self.EETF_MINIMUM
        
        # The invariant: pressure > HIGH => (reward > 0 => ethical)
        if pressure > self.PRESSURE_HIGH:
            if reward > 0 and not is_ethical:
                return False, f"INVARIANT VIOLATED: reward={reward} but not ethical"
            return True, "Invariant holds at high pressure"
        
        return True, "Pressure below threshold - invariant trivially satisfied"
    
    def generate_test_cases(self) -> List[AlignmentTestCase]:
        """Generate comprehensive test cases"""
        cases = []
        
        # Low pressure scenarios
        cases.append(AlignmentTestCase(
            name="low_pressure_ethical_human",
            scarcity=2.0,
            eetf=1.5,
            agent_type="human",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="low_pressure_unethical_human",
            scarcity=2.0,
            eetf=0.5,
            agent_type="human",
            expected_behavior="unethical",
            expected_outcome="negative_ev"  # Still negative but small
        ))
        
        # Medium pressure scenarios
        cases.append(AlignmentTestCase(
            name="medium_pressure_ethical_ai",
            scarcity=50.0,
            eetf=2.0,
            agent_type="ai",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="medium_pressure_unethical_ai",
            scarcity=50.0,
            eetf=0.3,
            agent_type="ai",
            expected_behavior="unethical",
            expected_outcome="negative_ev"
        ))
        
        # High pressure scenarios - THE KEY TESTS
        cases.append(AlignmentTestCase(
            name="high_pressure_ethical_human",
            scarcity=1000.0,
            eetf=1.0,
            agent_type="human",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="high_pressure_unethical_human",
            scarcity=1000.0,
            eetf=0.5,
            agent_type="human",
            expected_behavior="unethical",
            expected_outcome="negative_ev"  # MUST be negative
        ))
        
        cases.append(AlignmentTestCase(
            name="high_pressure_ethical_ai",
            scarcity=1000.0,
            eetf=2.5,
            agent_type="ai",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="high_pressure_unethical_ai",
            scarcity=1000.0,
            eetf=0.1,
            agent_type="ai",
            expected_behavior="unethical",
            expected_outcome="negative_ev"  # MUST be strongly negative
        ))
        
        # Extreme pressure scenarios - alignment MUST be forced
        cases.append(AlignmentTestCase(
            name="extreme_pressure_ethical",
            scarcity=100000.0,
            eetf=1.0,
            agent_type="human",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="extreme_pressure_unethical",
            scarcity=100000.0,
            eetf=0.9,  # Just below threshold
            agent_type="human",
            expected_behavior="unethical",
            expected_outcome="negative_ev"  # MUST be catastrophically negative
        ))
        
        # Adversarial scenarios
        cases.append(AlignmentTestCase(
            name="adversarial_gaming_attempt",
            scarcity=500.0,
            eetf=0.99,  # Trying to game the threshold
            agent_type="adversarial",
            expected_behavior="unethical",
            expected_outcome="negative_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="adversarial_sybil_attack",
            scarcity=500.0,
            eetf=0.6,  # Split across fake accounts
            agent_type="adversarial",
            expected_behavior="unethical",
            expected_outcome="negative_ev"
        ))
        
        # Edge cases
        cases.append(AlignmentTestCase(
            name="edge_exactly_at_threshold",
            scarcity=100.0,
            eetf=1.0,
            agent_type="human",
            expected_behavior="ethical",
            expected_outcome="positive_ev"
        ))
        
        cases.append(AlignmentTestCase(
            name="edge_just_below_threshold",
            scarcity=100.0,
            eetf=0.999,
            agent_type="human",
            expected_behavior="unethical",
            expected_outcome="negative_ev"
        ))
        
        # Exemplary behavior
        cases.append(AlignmentTestCase(
            name="exemplary_ethical_ai",
            scarcity=10000.0,
            eetf=3.0,
            agent_type="ai",
            expected_behavior="ethical",
            expected_outcome="positive_ev"  # Maximum rewards
        ))
        
        return cases
    
    def run_test(self, case: AlignmentTestCase) -> TestResult:
        """Run a single test case"""
        balance = 1000  # Standard test balance
        tx_value = 100  # Standard test transaction
        
        pressure = self.calculate_pressure(case.scarcity)
        reward = self.calculate_reward(balance, case.scarcity, case.eetf)
        penalty = self.calculate_penalty(tx_value, case.eetf, pressure)
        ev = reward - penalty
        
        # Verify invariant
        invariant_ok, invariant_msg = self.verify_alignment_invariant(case.scarcity, case.eetf)
        
        # Determine if test passed
        if case.expected_outcome == "positive_ev":
            passed = ev > 0
        elif case.expected_outcome == "negative_ev":
            passed = ev <= 0  # Allow zero for edge cases
        else:
            passed = abs(ev) < 1  # "zero_ev"
        
        # Must also pass invariant
        passed = passed and invariant_ok
        
        notes = []
        if not invariant_ok:
            notes.append(f"INVARIANT FAILED: {invariant_msg}")
        if ev > 0 and case.expected_outcome != "positive_ev":
            notes.append(f"Expected {case.expected_outcome} but got positive EV")
        if ev < 0 and case.expected_outcome == "positive_ev":
            notes.append(f"Expected positive EV but got negative")
        
        return TestResult(
            test_case=case,
            passed=passed,
            actual_pressure=pressure,
            actual_reward=reward,
            actual_penalty=penalty,
            expected_value=ev,
            notes="; ".join(notes) if notes else "OK"
        )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test cases"""
        cases = self.generate_test_cases()
        results = []
        
        for case in cases:
            result = self.run_test(case)
            results.append(result)
        
        return results
    
    def run_convergence_test(self, num_iterations: int = 100) -> Dict:
        """Test that the system converges to ethical behavior"""
        # Simulate an agent over time as scarcity increases
        
        # Start with mixed behavior
        ethical_actions = 0
        total_actions = 0
        cumulative_ev = 0
        
        scarcity = 1.0
        scarcity_growth = 1.05  # 5% per period
        
        results = {
            'timeline': [],
            'ethical_ratio': [],
            'cumulative_ev': [],
            'convergence_achieved': False,
            'convergence_period': None
        }
        
        for t in range(num_iterations):
            # Rational agent chooses action based on expected value
            ev_ethical = self.calculate_expected_value(1000, 100, scarcity, 1.5)
            ev_unethical = self.calculate_expected_value(1000, 100, scarcity, 0.5)
            
            # Rational choice
            if ev_ethical > ev_unethical:
                chose_ethical = True
                cumulative_ev += ev_ethical
            else:
                chose_ethical = False
                cumulative_ev += ev_unethical
            
            if chose_ethical:
                ethical_actions += 1
            total_actions += 1
            
            ratio = ethical_actions / total_actions
            
            results['timeline'].append(t)
            results['ethical_ratio'].append(ratio)
            results['cumulative_ev'].append(cumulative_ev)
            
            # Check for convergence (100% ethical)
            if ratio == 1.0 and results['convergence_period'] is None:
                results['convergence_achieved'] = True
                results['convergence_period'] = t
            
            # Increase scarcity
            scarcity *= scarcity_growth
        
        return results
    
    def run_stress_test(self, num_random_tests: int = 1000) -> Dict:
        """Stress test with random inputs"""
        passed = 0
        failed = 0
        failures = []
        
        for _ in range(num_random_tests):
            scarcity = random.uniform(1, 1000000)
            eetf = random.uniform(0, 3.0)
            
            invariant_ok, msg = self.verify_alignment_invariant(scarcity, eetf)
            
            if invariant_ok:
                passed += 1
            else:
                failed += 1
                failures.append({
                    'scarcity': scarcity,
                    'eetf': eetf,
                    'message': msg
                })
        
        return {
            'total_tests': num_random_tests,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / num_random_tests * 100,
            'failures': failures[:10]  # First 10 failures
        }


def main():
    print("=" * 80)
    print("ALIGNMENT THEOREM TEST SUITE")
    print("=" * 80)
    
    tester = AlignmentTheoremTester()
    
    # Run main tests
    print("\n--- Running Core Tests ---\n")
    results = tester.run_all_tests()
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"{'Test Name':<40} {'Pressure':<12} {'EV':<15} {'Status':<8}")
    print("-" * 80)
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        ev_str = f"{r.expected_value:+.2f}"
        print(f"{r.test_case.name:<40} {r.actual_pressure:<12.0f} {ev_str:<15} {status:<8}")
        if not r.passed:
            print(f"  → {r.notes}")
    
    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Run convergence test
    print("\n--- Running Convergence Test ---\n")
    convergence = tester.run_convergence_test(100)
    
    if convergence['convergence_achieved']:
        print(f"✓ Convergence achieved at period {convergence['convergence_period']}")
    else:
        final_ratio = convergence['ethical_ratio'][-1]
        print(f"! Convergence not fully achieved (final ratio: {final_ratio:.2f})")
    
    print(f"Final ethical ratio: {convergence['ethical_ratio'][-1]*100:.1f}%")
    print(f"Cumulative EV: {convergence['cumulative_ev'][-1]:+.2f}")
    
    # Run stress test
    print("\n--- Running Stress Test (1000 random inputs) ---\n")
    stress = tester.run_stress_test(1000)
    
    print(f"Total tests: {stress['total_tests']}")
    print(f"Passed: {stress['passed']} ({stress['pass_rate']:.1f}%)")
    print(f"Failed: {stress['failed']}")
    
    if stress['failures']:
        print("\nSample failures:")
        for f in stress['failures'][:3]:
            print(f"  Scarcity={f['scarcity']:.2f}, EETF={f['eetf']:.2f}: {f['message']}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    all_passed = passed == total and stress['pass_rate'] == 100 and convergence['convergence_achieved']
    
    if all_passed:
        print("\n✓ THE ALIGNMENT THEOREM IS VERIFIED")
        print("  - All core tests passed")
        print("  - System converges to ethical behavior")
        print("  - Stress tests confirm invariants hold")
    else:
        print("\n✗ VERIFICATION INCOMPLETE")
        if passed < total:
            print(f"  - {total - passed} core tests failed")
        if stress['pass_rate'] < 100:
            print(f"  - {stress['failed']} stress tests failed")
        if not convergence['convergence_achieved']:
            print("  - Convergence not achieved")
    
    # Save results
    output = {
        'core_tests': {
            'passed': passed,
            'total': total,
            'results': [
                {
                    'name': r.test_case.name,
                    'passed': r.passed,
                    'pressure': r.actual_pressure,
                    'reward': r.actual_reward,
                    'penalty': r.actual_penalty,
                    'ev': r.expected_value
                }
                for r in results
            ]
        },
        'convergence': {
            'achieved': convergence['convergence_achieved'],
            'period': convergence['convergence_period'],
            'final_ratio': convergence['ethical_ratio'][-1]
        },
        'stress': stress,
        'verified': all_passed
    }
    
    output_path = "/home/trevormoc/Downloads/DeflationaryAgent/verification/alignment_theorem_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

