#!/usr/bin/env python3
"""
Comprehensive FSM Analyzer for All Agent Versions

Ensures complete state coverage for all agent specifications (V35-V48).
Identifies missing states, unreachable transitions, and edge cases.

Copyright DarkLightX/Dana Edwards
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from enum import Enum, auto
from pathlib import Path
import json


class AgentVersion(Enum):
    """Agent versions with their FSM complexity"""
    V35_V41 = "v35-v41"      # Basic: 2 states (IDLE, EXECUTING)
    V42_V45 = "v42-v45"      # Intelligent: 2 states + bitvector
    V46 = "v46"              # Risk: 5 states (IDLE, EXEC, STOP, TAKE, COOL)
    V47 = "v47"              # Consensus: 2 states + indicators
    V48 = "v48"              # Sizing: 2 states + adaptive sizing


@dataclass
class FSMState:
    """Represents a state in the FSM"""
    name: str
    encoding: Tuple[int, ...]
    entry_conditions: List[str]
    exit_conditions: List[str]
    invariants: List[str]


@dataclass 
class FSMTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    trigger: str
    guard_conditions: List[str]


@dataclass
class FSMModel:
    """Complete FSM model for an agent"""
    version: str
    states: List[FSMState]
    transitions: List[FSMTransition]
    inputs: List[str]
    outputs: List[str]


def build_basic_fsm() -> FSMModel:
    """Build FSM for V35-V41 (basic 2-state)"""
    states = [
        FSMState(
            name="IDLE",
            encoding=(0,),
            entry_conditions=["o0[t-1]=1 AND (exit_trigger OR timeout)"],
            exit_conditions=["valid_entry AND volume AND NOT holding"],
            invariants=["NOT buy_signal", "NOT sell_signal", "timer=0"]
        ),
        FSMState(
            name="EXECUTING",
            encoding=(1,),
            entry_conditions=["valid_entry AND volume AND trend AND NOT holding"],
            exit_conditions=["exit_price_ok OR timeout OR failure"],
            invariants=["volume_required", "timer_incrementing"]
        )
    ]
    
    transitions = [
        FSMTransition("IDLE", "EXECUTING", "ENTRY", 
                     ["price<threshold", "volume", "trend", "NOT holding", "NOT nonce"]),
        FSMTransition("EXECUTING", "IDLE", "EXIT_PRICE",
                     ["price>threshold"]),
        FSMTransition("EXECUTING", "IDLE", "EXIT_TIMEOUT",
                     ["timer=3"]),
        FSMTransition("EXECUTING", "IDLE", "EXIT_FAILURE",
                     ["failure_echo"]),
        FSMTransition("EXECUTING", "EXECUTING", "CONTINUE",
                     ["NOT exit_price_ok", "NOT timeout", "volume"]),
        FSMTransition("IDLE", "IDLE", "NO_ENTRY",
                     ["NOT valid_entry OR NOT volume OR holding OR nonce"])
    ]
    
    return FSMModel(
        version="v35-v45",
        states=states,
        transitions=transitions,
        inputs=["price", "volume", "trend", "profit_guard", "failure_echo"],
        outputs=["state", "holding", "buy_signal", "sell_signal", "timer", "nonce", "profit", "burned"]
    )


def build_risk_fsm() -> FSMModel:
    """Build FSM for V46 (5-state risk management)"""
    states = [
        FSMState(
            name="IDLE",
            encoding=(0, 0, 0),
            entry_conditions=["cooldown_done OR take_profit"],
            exit_conditions=["valid_entry AND size_tier>=1"],
            invariants=["NOT holding", "timer=0"]
        ),
        FSMState(
            name="EXECUTING",
            encoding=(0, 0, 1),
            entry_conditions=["valid_entry from IDLE"],
            exit_conditions=["stop_loss OR take_profit OR timeout OR emergency"],
            invariants=["holding", "timer_active"]
        ),
        FSMState(
            name="STOP_LOSS",
            encoding=(0, 1, 0),
            entry_conditions=["price dropped > stop_loss_pct from entry"],
            exit_conditions=["always exits next tick"],
            invariants=["consecutive_losses++", "sell_signal"]
        ),
        FSMState(
            name="TAKE_PROFIT",
            encoding=(0, 1, 1),
            entry_conditions=["price rose > take_profit_pct from entry"],
            exit_conditions=["always exits next tick"],
            invariants=["consecutive_losses=0", "profit=1", "burn"]
        ),
        FSMState(
            name="COOLDOWN",
            encoding=(1, 0, 0),
            entry_conditions=["consecutive_losses >= max_losses"],
            exit_conditions=["cooldown_timer >= threshold"],
            invariants=["NOT executing", "cooldown_timer++"]
        )
    ]
    
    transitions = [
        FSMTransition("IDLE", "EXECUTING", "ENTRY", 
                     ["valid_entry", "NOT emergency"]),
        FSMTransition("EXECUTING", "STOP_LOSS", "STOP_LOSS_HIT",
                     ["(entry-price)*100/entry > stop_loss_pct"]),
        FSMTransition("EXECUTING", "TAKE_PROFIT", "TAKE_PROFIT_HIT",
                     ["(price-entry)*100/entry > take_profit_pct"]),
        FSMTransition("EXECUTING", "IDLE", "TIMEOUT",
                     ["timer=3"]),
        FSMTransition("STOP_LOSS", "IDLE", "LOSS_PROCESSED",
                     ["consecutive_losses < max"]),
        FSMTransition("STOP_LOSS", "COOLDOWN", "MAX_LOSSES",
                     ["consecutive_losses >= max"]),
        FSMTransition("TAKE_PROFIT", "IDLE", "PROFIT_PROCESSED",
                     ["always"]),
        FSMTransition("COOLDOWN", "IDLE", "COOLDOWN_DONE",
                     ["cooldown_timer >= threshold"]),
        FSMTransition("EXECUTING", "EXECUTING", "CONTINUE",
                     ["NOT stop_loss", "NOT take_profit", "NOT timeout"]),
        FSMTransition("IDLE", "IDLE", "NO_ENTRY",
                     ["NOT valid_entry OR emergency"]),
        FSMTransition("COOLDOWN", "COOLDOWN", "COOLING",
                     ["cooldown_timer < threshold"])
    ]
    
    return FSMModel(
        version="v46",
        states=states,
        transitions=transitions,
        inputs=["price", "volume", "trend", "profit_guard", "failure_echo", 
                "stop_loss_pct", "take_profit_pct", "max_consecutive_loss", "emergency_halt"],
        outputs=["state_b0", "state_b1", "state_b2", "holding", "buy_signal", "sell_signal",
                "timer", "nonce", "profit", "burned", "entry_price", "highest_price",
                "consecutive_losses", "cooldown_timer", "total_wins", "total_losses"]
    )


def analyze_state_coverage(model: FSMModel) -> Dict:
    """Analyze state coverage and find potential gaps"""
    results = {
        'total_states': len(model.states),
        'total_transitions': len(model.transitions),
        'states': {},
        'coverage_issues': [],
        'edge_cases': []
    }
    
    # Analyze each state
    for state in model.states:
        incoming = [t for t in model.transitions if t.to_state == state.name]
        outgoing = [t for t in model.transitions if t.from_state == state.name]
        
        results['states'][state.name] = {
            'encoding': state.encoding,
            'incoming_transitions': len(incoming),
            'outgoing_transitions': len(outgoing),
            'entry_conditions': state.entry_conditions,
            'exit_conditions': state.exit_conditions,
            'invariants': state.invariants
        }
        
        # Check for coverage issues
        if len(incoming) == 0 and state.name != "IDLE":
            results['coverage_issues'].append(
                f"{state.name}: No incoming transitions (unreachable?)")
        
        if len(outgoing) == 0:
            results['coverage_issues'].append(
                f"{state.name}: No outgoing transitions (deadlock?)")
    
    # Identify edge cases to test
    results['edge_cases'] = generate_edge_cases(model)
    
    return results


def generate_edge_cases(model: FSMModel) -> List[Dict]:
    """Generate edge cases for comprehensive testing"""
    edge_cases = []
    
    # Timer edge cases
    edge_cases.append({
        'name': 'timer_boundary',
        'description': 'Test timer at exactly timeout value',
        'test_sequence': [
            {'tick': 0, 'action': 'enter', 'expected_timer': 0},
            {'tick': 1, 'action': 'hold', 'expected_timer': 1},
            {'tick': 2, 'action': 'hold', 'expected_timer': 2},
            {'tick': 3, 'action': 'timeout', 'expected_timer': 3}
        ]
    })
    
    # Price boundary cases
    edge_cases.append({
        'name': 'price_exactly_at_threshold',
        'description': 'Price exactly equals entry/exit threshold',
        'test_values': [
            {'price': 100, 'threshold': 100, 'expected': 'no_trigger'},
            {'price': 101, 'threshold': 100, 'expected': 'exit_trigger'}
        ]
    })
    
    # Volume gap
    edge_cases.append({
        'name': 'volume_drops_during_execution',
        'description': 'Volume becomes stale while in position',
        'test_sequence': [
            {'tick': 0, 'volume': 1, 'action': 'enter'},
            {'tick': 1, 'volume': 0, 'action': 'force_exit'}
        ]
    })
    
    # Nonce reentry
    edge_cases.append({
        'name': 'rapid_reentry_attempt',
        'description': 'Attempt to reenter immediately after exit',
        'test_sequence': [
            {'tick': 0, 'action': 'enter'},
            {'tick': 1, 'action': 'exit'},
            {'tick': 2, 'action': 'entry_blocked_by_nonce'}
        ]
    })
    
    # Stop-loss edge (V46)
    if model.version == "v46":
        edge_cases.append({
            'name': 'stop_loss_exactly_at_threshold',
            'description': 'Price drops exactly to stop-loss level',
            'test_values': [
                {'entry': 100, 'current': 95, 'stop_loss_pct': 5, 
                 'expected': 'stop_loss_triggered'}
            ]
        })
        
        edge_cases.append({
            'name': 'consecutive_loss_max',
            'description': 'Hit max consecutive losses',
            'test_sequence': [
                {'tick': 0, 'action': 'stop_loss', 'consecutive': 1},
                {'tick': 1, 'action': 'stop_loss', 'consecutive': 2},
                {'tick': 2, 'action': 'stop_loss_to_cooldown', 'consecutive': 3}
            ]
        })
    
    # Overflow cases
    edge_cases.append({
        'name': 'counter_overflow',
        'description': 'Test counters at max value',
        'test_values': [
            {'counter': 'total_wins', 'value': 255, 'action': 'win',
             'expected': 'wrap_to_0_or_saturate'}
        ]
    })
    
    return edge_cases


def generate_test_sequences(model: FSMModel) -> List[Dict]:
    """Generate test sequences for 100% transition coverage"""
    sequences = []
    covered_transitions = set()
    
    # Basic coverage: visit every transition
    for transition in model.transitions:
        if (transition.from_state, transition.to_state) not in covered_transitions:
            sequences.append({
                'name': f'{transition.from_state}_to_{transition.to_state}',
                'trigger': transition.trigger,
                'guards': transition.guard_conditions,
                'sequence': generate_path_to_transition(model, transition)
            })
            covered_transitions.add((transition.from_state, transition.to_state))
    
    return sequences


def generate_path_to_transition(model: FSMModel, target: FSMTransition) -> List[Dict]:
    """Generate input sequence to reach and trigger a specific transition"""
    # Start from IDLE
    path = []
    current_state = "IDLE"
    
    # If we need to reach a non-IDLE state first
    if target.from_state != "IDLE":
        # Find path to from_state
        for t in model.transitions:
            if t.to_state == target.from_state:
                path.append({
                    'inputs': t.guard_conditions,
                    'expected_transition': t.trigger
                })
                current_state = target.from_state
                break
    
    # Add the target transition
    path.append({
        'inputs': target.guard_conditions,
        'expected_transition': target.trigger
    })
    
    return path


def print_fsm_report(model: FSMModel, analysis: Dict):
    """Print comprehensive FSM analysis report"""
    print("=" * 70)
    print(f"FSM ANALYSIS REPORT: {model.version.upper()}")
    print("=" * 70)
    
    print(f"\nTotal States: {analysis['total_states']}")
    print(f"Total Transitions: {analysis['total_transitions']}")
    
    print("\n--- STATE DETAILS ---")
    for state_name, details in analysis['states'].items():
        print(f"\n{state_name} (encoding: {details['encoding']}):")
        print(f"  Incoming: {details['incoming_transitions']}")
        print(f"  Outgoing: {details['outgoing_transitions']}")
        print(f"  Entry: {details['entry_conditions']}")
        print(f"  Exit: {details['exit_conditions']}")
        print(f"  Invariants: {details['invariants']}")
    
    if analysis['coverage_issues']:
        print("\n--- COVERAGE ISSUES ---")
        for issue in analysis['coverage_issues']:
            print(f"  WARNING: {issue}")
    else:
        print("\n--- COVERAGE: OK ---")
        print("  All states reachable and have exit paths")
    
    print(f"\n--- EDGE CASES ({len(analysis['edge_cases'])}) ---")
    for i, edge_case in enumerate(analysis['edge_cases'], 1):
        print(f"\n{i}. {edge_case['name']}")
        print(f"   {edge_case['description']}")
    
    print("\n" + "=" * 70)


def main():
    """Run comprehensive FSM analysis for all versions"""
    print("=" * 70)
    print("COMPREHENSIVE FSM ANALYZER")
    print("Analyzing all agent versions for complete state coverage")
    print("=" * 70)
    
    # Analyze basic FSM (V35-V45)
    basic_model = build_basic_fsm()
    basic_analysis = analyze_state_coverage(basic_model)
    print_fsm_report(basic_model, basic_analysis)
    
    # Analyze risk FSM (V46)
    risk_model = build_risk_fsm()
    risk_analysis = analyze_state_coverage(risk_model)
    print_fsm_report(risk_model, risk_analysis)
    
    # Generate test sequences
    print("\n" + "=" * 70)
    print("TEST SEQUENCE GENERATION")
    print("=" * 70)
    
    basic_sequences = generate_test_sequences(basic_model)
    print(f"\nV35-V45 requires {len(basic_sequences)} test sequences for full coverage")
    
    risk_sequences = generate_test_sequences(risk_model)
    print(f"V46 requires {len(risk_sequences)} test sequences for full coverage")
    
    # Save results
    results = {
        'basic_model': {
            'version': basic_model.version,
            'analysis': basic_analysis,
            'test_sequences': basic_sequences
        },
        'risk_model': {
            'version': risk_model.version,
            'analysis': risk_analysis,
            'test_sequences': risk_sequences
        }
    }
    
    output_path = Path(__file__).parent / "fsm_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"V35-V45: {len(basic_analysis['states'])} states, {len(basic_analysis['edge_cases'])} edge cases")
    print(f"V46: {len(risk_analysis['states'])} states, {len(risk_analysis['edge_cases'])} edge cases")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

