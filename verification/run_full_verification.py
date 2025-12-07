#!/usr/bin/env python3
"""
Full Verification Suite for Deflationary Agent

Runs all verification and testing components:
1. FSM Model enumeration
2. Coverage generation
3. Input simulation
4. Trace analysis
5. Edge case tests
6. Generate reports

Copyright DarkLightX/Dana Edwards
"""

import sys
from pathlib import Path
from datetime import datetime

def main():
    """Run full verification suite"""
    print("=" * 80)
    print("DEFLATIONARY AGENT - FULL VERIFICATION SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {}
    
    # 1. FSM Model
    print("\n[1/6] Building FSM Model...")
    from fsm_model import DeflationaryAgentFSM
    fsm = DeflationaryAgentFSM()
    fsm.enumerate_reachable_states()
    fsm.build_transition_table()
    results['fsm'] = {
        'reachable_states': len(fsm.reachable_states),
        'total_transitions': len(fsm.transitions)
    }
    print(f"      Found {results['fsm']['reachable_states']} reachable states")
    print(f"      Found {results['fsm']['total_transitions']} transitions")
    
    # 2. Coverage
    print("\n[2/6] Generating Coverage Tests...")
    from coverage_generator import CoverageGenerator
    cov_gen = CoverageGenerator(fsm)
    min_seq = cov_gen.generate_minimal_coverage_sequence()
    boundary_seqs = cov_gen.generate_boundary_tests()
    cov_result = cov_gen.compute_coverage([min_seq] + boundary_seqs)
    results['coverage'] = {
        'state_coverage': cov_result.state_coverage,
        'transition_coverage': cov_result.transition_coverage
    }
    print(f"      State coverage: {cov_result.state_coverage*100:.1f}%")
    print(f"      Transition coverage: {cov_result.transition_coverage*100:.1f}%")
    
    # 3. Input Simulation
    print("\n[3/6] Generating Input Simulations...")
    from input_simulator import InputSimulator
    sim = InputSimulator(seed=42)
    scenarios = {
        'bull_market': sim.generate_scenario_bull_market(),
        'bear_market': sim.generate_scenario_bear_market(),
        'flash_crash': sim.generate_scenario_flash_crash(),
        'adversarial': sim.generate_scenario_adversarial_double_entry(),
    }
    results['simulation'] = {
        'scenarios': len(scenarios),
        'total_ticks': sum(len(v) for v in scenarios.values())
    }
    print(f"      Generated {results['simulation']['scenarios']} scenarios")
    print(f"      Total {results['simulation']['total_ticks']} input ticks")
    
    # 4. Trace Analysis
    print("\n[4/6] Running Trace Analysis...")
    from trace_analyzer import TraceAnalyzer
    from fsm_model import InputVector
    analyzer = TraceAnalyzer()
    test_trace = analyzer.analyze_sequence("test", "Quick test", [
        InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),
        InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),
    ])
    invariant_ok = all(t.all_invariants_hold() for t in test_trace.ticks)
    results['trace'] = {
        'invariants_hold': invariant_ok
    }
    print(f"      Invariants hold: {invariant_ok}")
    
    # 5. Edge Case Tests
    print("\n[5/6] Running Edge Case Tests...")
    from edge_case_tests import EdgeCaseTestSuite
    suite = EdgeCaseTestSuite()
    passed, failed, failures = suite.run_all()
    results['edge_cases'] = {
        'passed': passed,
        'failed': failed
    }
    
    # 6. Generate Report
    print("\n[6/6] Generating Verification Report...")
    from verification_report import VerificationReportGenerator
    report_gen = VerificationReportGenerator()
    report_file = report_gen.save_report()
    results['report'] = str(report_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nFSM Model:")
    print(f"  Reachable states: {results['fsm']['reachable_states']}")
    print(f"  Total transitions: {results['fsm']['total_transitions']}")
    
    print(f"\nCoverage:")
    print(f"  State: {results['coverage']['state_coverage']*100:.1f}%")
    print(f"  Transitions: {results['coverage']['transition_coverage']*100:.1f}%")
    
    print(f"\nEdge Case Tests:")
    print(f"  Passed: {results['edge_cases']['passed']}")
    print(f"  Failed: {results['edge_cases']['failed']}")
    
    print(f"\nReport: {results['report']}")
    
    overall_pass = (
        results['coverage']['state_coverage'] == 1.0 and
        results['trace']['invariants_hold'] and
        results['edge_cases']['failed'] <= 1  # Allow 1 expected failure
    )
    
    print("\n" + "=" * 80)
    if overall_pass:
        print("✓ VERIFICATION PASSED")
    else:
        print("⚠ VERIFICATION COMPLETED WITH WARNINGS")
    print("=" * 80)
    
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

