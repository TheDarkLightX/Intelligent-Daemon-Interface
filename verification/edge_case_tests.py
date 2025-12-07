#!/usr/bin/env python3
"""
Exhaustive Edge Case Tests for Deflationary Agent

Tests ALL boundary conditions and edge cases systematically:
1. Timer boundaries (0, 1, 2, 3)
2. Nonce blocking scenarios
3. Entry condition failures (each condition individually)
4. Exit conditions (profit, timeout, failure, volume dropout)
5. State persistence across transitions
6. Burn accumulation
7. Multiple trade sequences
8. Adversarial patterns

Copyright DarkLightX/Dana Edwards
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import json

from fsm_model import DeflationaryAgentFSM, AgentState, InputVector, Outputs
from trace_analyzer import TraceAnalyzer, ExecutionTrace


@dataclass
class EdgeCaseTest:
    """A single edge case test"""
    name: str
    description: str
    category: str
    inputs: List[InputVector]
    expected_behavior: str
    assertions: List[Callable[[ExecutionTrace], Tuple[bool, str]]]
    
    def run(self, analyzer: TraceAnalyzer) -> Tuple[bool, ExecutionTrace, List[str]]:
        """Run the test and return (passed, trace, failures)"""
        trace = analyzer.analyze_sequence(self.name, self.description, self.inputs)
        failures = []
        
        for assertion in self.assertions:
            passed, msg = assertion(trace)
            if not passed:
                failures.append(msg)
        
        return len(failures) == 0, trace, failures


class EdgeCaseTestSuite:
    """
    Comprehensive edge case test suite for complete coverage
    """
    
    def __init__(self):
        self.analyzer = TraceAnalyzer()
        self.tests: List[EdgeCaseTest] = []
        self._build_all_tests()
    
    def _build_all_tests(self):
        """Build all edge case tests"""
        self._add_timer_tests()
        self._add_nonce_tests()
        self._add_entry_condition_tests()
        self._add_exit_condition_tests()
        self._add_persistence_tests()
        self._add_burn_tests()
        self._add_adversarial_tests()
        self._add_exhaustive_input_tests()
    
    # === Timer Edge Cases ===
    def _add_timer_tests(self):
        """Timer boundary tests"""
        
        # Timer T0 -> T1 transition
        self.tests.append(EdgeCaseTest(
            name="timer_t0_to_t1",
            description="Timer transitions from 00 to 01 on entry",
            category="timer",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
            ],
            expected_behavior="Timer starts at 1 (01) after entry",
            assertions=[
                lambda t: (t.ticks[0].curr_state.timer == 1, 
                          f"Expected timer=1, got {t.ticks[0].curr_state.timer}")
            ]
        ))
        
        # Timer T1 -> T2 transition
        self.tests.append(EdgeCaseTest(
            name="timer_t1_to_t2",
            description="Timer transitions from 01 to 10",
            category="timer",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry (T=1)
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue (T=2)
            ],
            expected_behavior="Timer increments to 2 (10)",
            assertions=[
                lambda t: (t.ticks[1].curr_state.timer == 2,
                          f"Expected timer=2, got {t.ticks[1].curr_state.timer}")
            ]
        ))
        
        # Timer T2 -> T3 transition
        self.tests.append(EdgeCaseTest(
            name="timer_t2_to_t3",
            description="Timer transitions from 10 to 11",
            category="timer",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry (T=1)
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue (T=2)
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue (T=3)
            ],
            expected_behavior="Timer reaches 3 (11) - timeout threshold",
            assertions=[
                lambda t: (t.ticks[2].curr_state.timer == 3,
                          f"Expected timer=3, got {t.ticks[2].curr_state.timer}")
            ]
        ))
        
        # Timer T3 forces exit (timeout)
        self.tests.append(EdgeCaseTest(
            name="timer_t3_timeout",
            description="Timer at 11 forces exit on next tick",
            category="timer",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry (T=1)
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # T=2
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # T=3
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Timeout exit
            ],
            expected_behavior="State exits (o0=0) due to timeout",
            assertions=[
                lambda t: (t.ticks[2].curr_state.timer == 3, "Timer should be 3"),
                lambda t: (t.ticks[3].curr_state.o0 == 0, 
                          f"Expected o0=0 after timeout, got {t.ticks[3].curr_state.o0}"),
                lambda t: (t.ticks[3].outputs.o3 == 1, "Expected sell signal on timeout")
            ]
        ))
        
        # Timer resets on exit
        self.tests.append(EdgeCaseTest(
            name="timer_reset_on_exit",
            description="Timer resets to 0 after exiting",
            category="timer",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit
            ],
            expected_behavior="Timer resets to 0 after exit",
            assertions=[
                lambda t: (t.ticks[1].curr_state.timer == 0,
                          f"Expected timer=0 after exit, got {t.ticks[1].curr_state.timer}")
            ]
        ))
    
    # === Nonce Edge Cases ===
    def _add_nonce_tests(self):
        """Nonce blocking tests"""
        
        # Nonce set on entry
        self.tests.append(EdgeCaseTest(
            name="nonce_set_on_entry",
            description="Nonce (o9) is set to 1 when entry occurs",
            category="nonce",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
            ],
            expected_behavior="o9=1 after entry",
            assertions=[
                lambda t: (t.ticks[0].outputs.o9 == 1,
                          f"Expected nonce=1, got {t.ticks[0].outputs.o9}")
            ]
        ))
        
        # Nonce clears on exit
        self.tests.append(EdgeCaseTest(
            name="nonce_clear_on_exit",
            description="Nonce clears when exiting idle",
            category="nonce",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit
            ],
            expected_behavior="o9=0 after exit",
            assertions=[
                lambda t: (t.ticks[1].outputs.o9 == 0,
                          f"Expected nonce=0 after exit, got {t.ticks[1].outputs.o9}")
            ]
        ))
        
        # Nonce blocks immediate re-entry
        self.tests.append(EdgeCaseTest(
            name="nonce_blocks_reentry",
            description="Entry blocked when nonce is set from previous trade",
            category="nonce",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Try re-entry
            ],
            expected_behavior="Entry blocked (o0=0) due to nonce still set",
            assertions=[
                # After exit, nonce persists in some form or clears depending on spec
                # The key is that immediate re-entry should be blocked
                lambda t: (t.ticks[2].outputs.o2 == 0 or t.ticks[2].curr_state.o0 == 0,
                          "Re-entry should be blocked or not produce buy signal")
            ]
        ))
    
    # === Entry Condition Tests ===
    def _add_entry_condition_tests(self):
        """Test each entry condition failing individually"""
        
        # Entry blocked: price high
        self.tests.append(EdgeCaseTest(
            name="entry_blocked_price_high",
            description="Entry blocked when price is high (i0=1)",
            category="entry_conditions",
            inputs=[
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Price high
            ],
            expected_behavior="No entry (o0=0) when price high",
            assertions=[
                lambda t: (t.ticks[0].outputs.o0 == 0,
                          f"Expected o0=0 with high price, got {t.ticks[0].outputs.o0}")
            ]
        ))
        
        # Entry blocked: volume low
        self.tests.append(EdgeCaseTest(
            name="entry_blocked_volume_low",
            description="Entry blocked when volume is low (i1=0)",
            category="entry_conditions",
            inputs=[
                InputVector(i0=0, i1=0, i2=1, i3=1, i4=0),  # Volume low
            ],
            expected_behavior="No entry when volume low",
            assertions=[
                lambda t: (t.ticks[0].outputs.o0 == 0,
                          f"Expected o0=0 with low volume, got {t.ticks[0].outputs.o0}")
            ]
        ))
        
        # Entry blocked: trend bearish
        self.tests.append(EdgeCaseTest(
            name="entry_blocked_trend_bearish",
            description="Entry blocked when trend is bearish (i2=0)",
            category="entry_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=0, i3=1, i4=0),  # Trend bearish
            ],
            expected_behavior="No entry when trend bearish",
            assertions=[
                lambda t: (t.ticks[0].outputs.o0 == 0,
                          f"Expected o0=0 with bearish trend, got {t.ticks[0].outputs.o0}")
            ]
        ))
        
        # Entry blocked: failure echo
        self.tests.append(EdgeCaseTest(
            name="entry_blocked_failure",
            description="Entry blocked when failure echo active (i4=1)",
            category="entry_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=1),  # Failure
            ],
            expected_behavior="No entry when failure echo",
            assertions=[
                lambda t: (t.ticks[0].outputs.o0 == 0,
                          f"Expected o0=0 with failure, got {t.ticks[0].outputs.o0}")
            ]
        ))
        
        # Entry succeeds: all conditions met
        self.tests.append(EdgeCaseTest(
            name="entry_success_all_conditions",
            description="Entry succeeds when all conditions met",
            category="entry_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # All good
            ],
            expected_behavior="Entry succeeds (o0=1, o2=1)",
            assertions=[
                lambda t: (t.ticks[0].outputs.o0 == 1,
                          f"Expected o0=1, got {t.ticks[0].outputs.o0}"),
                lambda t: (t.ticks[0].outputs.o2 == 1,
                          f"Expected o2=1 (buy), got {t.ticks[0].outputs.o2}")
            ]
        ))
    
    # === Exit Condition Tests ===
    def _add_exit_condition_tests(self):
        """Test all exit conditions"""
        
        # Exit: profitable (high price + guard)
        self.tests.append(EdgeCaseTest(
            name="exit_profitable",
            description="Profitable exit when price high and guard active",
            category="exit_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit high
            ],
            expected_behavior="Exit with profit (o11=1)",
            assertions=[
                lambda t: (t.ticks[1].outputs.o3 == 1, "Expected sell signal"),
                lambda t: (t.ticks[1].outputs.o11 == 1,
                          f"Expected profit=1, got {t.ticks[1].outputs.o11}")
            ]
        ))
        
        # Exit: no profit without guard
        self.tests.append(EdgeCaseTest(
            name="exit_no_profit_no_guard",
            description="Exit without profit when guard inactive",
            category="exit_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=0, i4=0),  # Exit high but no guard
            ],
            expected_behavior="Exit without profit (o11=0)",
            assertions=[
                lambda t: (t.ticks[1].outputs.o3 == 1, "Expected sell signal"),
                lambda t: (t.ticks[1].outputs.o11 == 0,
                          f"Expected profit=0 without guard, got {t.ticks[1].outputs.o11}")
            ]
        ))
        
        # Exit: volume dropout during execution
        self.tests.append(EdgeCaseTest(
            name="exit_volume_dropout",
            description="Exit forced when volume drops during execution",
            category="exit_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=0, i1=0, i2=1, i3=1, i4=0),  # Volume drops
            ],
            expected_behavior="State exits (o0=0) on volume dropout",
            assertions=[
                lambda t: (t.ticks[1].outputs.o0 == 0,
                          f"Expected o0=0 on volume dropout, got {t.ticks[1].outputs.o0}")
            ]
        ))
        
        # Exit: failure echo during execution
        self.tests.append(EdgeCaseTest(
            name="exit_failure_during_exec",
            description="Exit forced when failure echo during execution",
            category="exit_conditions",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=1),  # Failure
            ],
            expected_behavior="State exits immediately on failure",
            assertions=[
                lambda t: (t.ticks[1].outputs.o0 == 0,
                          f"Expected o0=0 on failure, got {t.ticks[1].outputs.o0}")
            ]
        ))
    
    # === State Persistence Tests ===
    def _add_persistence_tests(self):
        """Test state variable persistence"""
        
        # Holding persists during execution
        self.tests.append(EdgeCaseTest(
            name="holding_persists",
            description="Holding (o1) persists during execution",
            category="persistence",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue
            ],
            expected_behavior="o1=1 throughout execution",
            assertions=[
                lambda t: (all(tick.outputs.o1 == 1 for tick in t.ticks),
                          "Holding should persist")
            ]
        ))
        
        # Entry price captured on entry
        self.tests.append(EdgeCaseTest(
            name="entry_price_captured",
            description="Entry price (o10) captured at entry",
            category="persistence",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry at low
            ],
            expected_behavior="o10=0 (entered at low price)",
            assertions=[
                lambda t: (t.ticks[0].outputs.o10 == 0,
                          f"Expected entry price=0, got {t.ticks[0].outputs.o10}")
            ]
        ))
    
    # === Burn Tests ===
    def _add_burn_tests(self):
        """Test burn accumulation"""
        
        # Burn set on profitable exit
        self.tests.append(EdgeCaseTest(
            name="burn_on_profit",
            description="Burn flag (o13) set on profitable exit",
            category="burn",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Profit exit
            ],
            expected_behavior="o13=1 after profitable exit",
            assertions=[
                lambda t: (t.ticks[1].outputs.o13 == 1,
                          f"Expected burn=1, got {t.ticks[1].outputs.o13}")
            ]
        ))
        
        # Burn persists across trades
        self.tests.append(EdgeCaseTest(
            name="burn_persists",
            description="Burn flag persists after being set",
            category="burn",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Trade 1 entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Trade 1 profit exit
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Trade 2 entry
                InputVector(i0=1, i1=1, i2=1, i3=0, i4=0),  # Trade 2 no profit
            ],
            expected_behavior="o13=1 persists even without profit",
            assertions=[
                lambda t: (t.ticks[3].outputs.o13 == 1,
                          f"Burn should persist, got {t.ticks[3].outputs.o13}")
            ]
        ))
    
    # === Adversarial Tests ===
    def _add_adversarial_tests(self):
        """Test adversarial/attack patterns"""
        
        # Rapid entry attempts
        self.tests.append(EdgeCaseTest(
            name="adversarial_rapid_entry",
            description="Multiple rapid entry attempts",
            category="adversarial",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Attempt
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Attempt
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Attempt
            ],
            expected_behavior="Only one buy signal generated",
            assertions=[
                lambda t: (sum(tick.outputs.o2 for tick in t.ticks) == 1,
                          f"Expected exactly 1 buy, got {sum(tick.outputs.o2 for tick in t.ticks)}")
            ]
        ))
        
        # Alternating conditions
        self.tests.append(EdgeCaseTest(
            name="adversarial_oscillating",
            description="Rapidly oscillating input conditions",
            category="adversarial",
            inputs=[
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),
                InputVector(i0=1, i1=0, i2=0, i3=0, i4=1),
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),
                InputVector(i0=1, i1=0, i2=0, i3=0, i4=1),
            ],
            expected_behavior="Invariants hold under rapid changes",
            assertions=[
                lambda t: (all(tick.all_invariants_hold() for tick in t.ticks),
                          "All invariants should hold")
            ]
        ))
    
    # === Exhaustive Input Tests ===
    def _add_exhaustive_input_tests(self):
        """Test all 32 inputs from initial state"""
        for i in range(32):
            inp = InputVector.from_int(i)
            
            self.tests.append(EdgeCaseTest(
                name=f"exhaust_idle_input_{i:02d}",
                description=f"Test input {i} ({inp}) from idle state",
                category="exhaustive_idle",
                inputs=[inp],
                expected_behavior="Invariants hold, expected behavior based on conditions",
                assertions=[
                    lambda t, inp=inp: (t.ticks[0].all_invariants_hold(),
                                       f"Invariants should hold for input {inp}")
                ]
            ))
    
    def run_all(self) -> Tuple[int, int, List[Tuple[str, List[str]]]]:
        """Run all tests and return (passed, failed, failure_details)"""
        passed = 0
        failed = 0
        failures = []
        
        print("=" * 70)
        print("EDGE CASE TEST SUITE")
        print("=" * 70)
        
        by_category: Dict[str, List[EdgeCaseTest]] = {}
        for test in self.tests:
            if test.category not in by_category:
                by_category[test.category] = []
            by_category[test.category].append(test)
        
        for category, tests in sorted(by_category.items()):
            print(f"\n{category.upper()} ({len(tests)} tests)")
            print("-" * 50)
            
            for test in tests:
                success, trace, test_failures = test.run(self.analyzer)
                
                if success:
                    passed += 1
                    print(f"  ✓ {test.name}")
                else:
                    failed += 1
                    print(f"  ✗ {test.name}")
                    for msg in test_failures:
                        print(f"      └─ {msg}")
                    failures.append((test.name, test_failures))
        
        print("\n" + "=" * 70)
        print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
        print("=" * 70)
        
        return passed, failed, failures
    
    def run_category(self, category: str):
        """Run tests in a specific category"""
        tests = [t for t in self.tests if t.category == category]
        
        print(f"\nRunning {len(tests)} tests in category: {category}")
        
        for test in tests:
            success, trace, failures = test.run(self.analyzer)
            status = "✓" if success else "✗"
            print(f"  {status} {test.name}: {test.description}")
            
            if not success:
                for msg in failures:
                    print(f"      └─ {msg}")


def main():
    """Run comprehensive edge case tests"""
    suite = EdgeCaseTestSuite()
    
    print(f"Total tests: {len(suite.tests)}")
    
    # Run all tests
    passed, failed, failures = suite.run_all()
    
    # Save detailed results
    results = {
        'total': passed + failed,
        'passed': passed,
        'failed': failed,
        'failures': [{'test': name, 'errors': errs} for name, errs in failures]
    }
    
    output_file = Path(__file__).parent / "edge_case_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

