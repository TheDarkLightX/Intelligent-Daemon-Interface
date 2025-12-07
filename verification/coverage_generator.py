#!/usr/bin/env python3
"""
Coverage Generator for Deflationary Agent FSM

Generates minimal test sequences that achieve 100% transition coverage.
Uses the FSM model to systematically explore all (state, input) pairs.

Coverage Metrics:
1. State coverage: visit all reachable states
2. Transition coverage: exercise all (state, input) -> state' edges
3. Path coverage: representative paths through state graph
4. Boundary coverage: test state boundaries

Copyright DarkLightX/Dana Edwards
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
import json

# Import FSM model
from fsm_model import (
    DeflationaryAgentFSM, AgentState, InputVector, 
    Transition, Outputs
)


@dataclass
class CoverageResult:
    """Coverage analysis results"""
    total_states: int = 0
    covered_states: int = 0
    total_transitions: int = 0
    covered_transitions: int = 0
    total_paths: int = 0
    covered_paths: int = 0
    
    # Detailed tracking
    uncovered_states: Set[AgentState] = field(default_factory=set)
    uncovered_transitions: Set[Tuple[AgentState, InputVector]] = field(default_factory=set)
    
    @property
    def state_coverage(self) -> float:
        return self.covered_states / self.total_states if self.total_states else 0
    
    @property
    def transition_coverage(self) -> float:
        return self.covered_transitions / self.total_transitions if self.total_transitions else 0


@dataclass
class TestSequence:
    """A test sequence with expected results"""
    name: str
    description: str
    inputs: List[InputVector]
    expected_states: List[AgentState]
    expected_outputs: List[Outputs]
    covered_transitions: Set[Tuple[str, str]]  # (from_state, to_state)


class CoverageGenerator:
    """
    Generate test sequences for complete FSM coverage
    """
    
    def __init__(self, fsm: DeflationaryAgentFSM):
        self.fsm = fsm
        if not fsm.reachable_states:
            fsm.enumerate_reachable_states()
        if not fsm.transitions:
            fsm.build_transition_table()
        
        # Coverage tracking
        self.covered_states: Set[AgentState] = set()
        self.covered_transitions: Set[Tuple[AgentState, InputVector]] = set()
        self.test_sequences: List[TestSequence] = []
    
    def generate_all_input_combinations(self) -> List[Dict[str, int]]:
        """Generate all 32 possible input combinations"""
        combinations = []
        for i in range(32):
            inp = InputVector.from_int(i)
            combinations.append({
                'i0': inp.i0,
                'i1': inp.i1,
                'i2': inp.i2,
                'i3': inp.i3,
                'i4': inp.i4,
                'description': self._describe_input(inp)
            })
        return combinations
    
    def _describe_input(self, inp: InputVector) -> str:
        """Human-readable input description"""
        parts = []
        parts.append("price_high" if inp.i0 else "price_low")
        parts.append("volume_ok" if inp.i1 else "volume_low")
        parts.append("trend_bull" if inp.i2 else "trend_bear")
        parts.append("profit_guard" if inp.i3 else "no_guard")
        parts.append("FAILURE" if inp.i4 else "no_fail")
        return ", ".join(parts)
    
    def find_path_to_state(self, target: AgentState) -> Optional[List[Tuple[InputVector, AgentState]]]:
        """
        Find shortest path from initial state to target state using BFS.
        Returns list of (input, resulting_state) pairs.
        """
        initial = AgentState.initial()
        if target == initial:
            return []
        
        # BFS with path tracking
        queue = deque([(initial, [])])
        visited = {initial}
        
        while queue:
            state, path = queue.popleft()
            
            for i in range(32):
                inp = InputVector.from_int(i)
                next_state, _, _ = self.fsm.next_state(state, inp)
                
                if next_state == target:
                    return path + [(inp, next_state)]
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [(inp, next_state)]))
        
        return None  # Unreachable
    
    def generate_minimal_coverage_sequence(self) -> TestSequence:
        """
        Generate a minimal sequence that covers all states and transitions.
        Uses greedy approach: at each state, pick input that covers most new transitions.
        """
        inputs = []
        states = [AgentState.initial()]
        outputs_list = []
        covered_trans = set()
        
        current = AgentState.initial()
        self.covered_states.add(current)
        
        # Track which transitions we've covered
        all_transitions = set(self.fsm.transitions.keys())
        
        max_steps = 500  # Safety limit
        steps = 0
        
        while len(covered_trans) < len(all_transitions) and steps < max_steps:
            # Find input that covers most new transitions from current state
            best_inp = None
            best_gain = 0
            best_next = None
            best_outputs = None
            
            for i in range(32):
                inp = InputVector.from_int(i)
                trans_key = (current, inp)
                
                next_state, outputs, _ = self.fsm.next_state(current, inp)
                
                # Count new coverage gain
                gain = 0
                if trans_key not in covered_trans:
                    gain += 1
                if next_state not in self.covered_states:
                    gain += 5  # Prioritize new states
                
                if gain > best_gain:
                    best_gain = gain
                    best_inp = inp
                    best_next = next_state
                    best_outputs = outputs
            
            if best_gain == 0:
                # No new coverage from current state - need to navigate elsewhere
                # Find uncovered transition and path to its source state
                uncovered = all_transitions - covered_trans
                for (src_state, inp) in uncovered:
                    if src_state != current:
                        path = self.find_path_to_state(src_state)
                        if path:
                            # Follow path
                            for path_inp, path_state in path:
                                trans_key = (current, path_inp)
                                covered_trans.add(trans_key)
                                self.covered_transitions.add(trans_key)
                                self.covered_states.add(path_state)
                                
                                _, path_outputs, _ = self.fsm.next_state(current, path_inp)
                                
                                inputs.append(path_inp)
                                states.append(path_state)
                                outputs_list.append(path_outputs)
                                
                                current = path_state
                                steps += 1
                            break
                else:
                    # Can't find path - try random unexplored transition
                    for i in range(32):
                        inp = InputVector.from_int(i)
                        trans_key = (current, inp)
                        if trans_key not in covered_trans:
                            best_inp = inp
                            best_next, best_outputs, _ = self.fsm.next_state(current, inp)
                            break
                    else:
                        # All transitions from current covered - pick any
                        best_inp = InputVector.from_int(0)
                        best_next, best_outputs, _ = self.fsm.next_state(current, best_inp)
            
            if best_inp:
                trans_key = (current, best_inp)
                covered_trans.add(trans_key)
                self.covered_transitions.add(trans_key)
                self.covered_states.add(best_next)
                
                inputs.append(best_inp)
                states.append(best_next)
                outputs_list.append(best_outputs)
                
                current = best_next
                steps += 1
        
        return TestSequence(
            name="minimal_coverage",
            description=f"Minimal sequence covering {len(covered_trans)}/{len(all_transitions)} transitions in {steps} steps",
            inputs=inputs,
            expected_states=states[1:],  # Skip initial
            expected_outputs=outputs_list,
            covered_transitions={(str(s), str(self.fsm.transitions[(s, i)].to_state)) 
                                for s, i in covered_trans}
        )
    
    def generate_boundary_tests(self) -> List[TestSequence]:
        """
        Generate test sequences for boundary conditions.
        """
        sequences = []
        
        # 1. Timer boundary: reach timer=3 (timeout)
        # Entry at T0 -> T1 -> T2 -> T3 (timeout)
        entry_input = InputVector(i0=0, i1=1, i2=1, i3=1, i4=0)  # Valid entry
        continue_no_exit = InputVector(i0=0, i1=1, i2=1, i3=1, i4=0)  # Stay in
        
        timeout_seq = TestSequence(
            name="timer_timeout",
            description="Test timer reaching 11 (3) and forcing timeout exit",
            inputs=[entry_input] + [continue_no_exit] * 4,  # Entry + 3 continues = T3
            expected_states=[],  # Will be computed
            expected_outputs=[],
            covered_transitions=set()
        )
        
        # Compute expected states
        current = AgentState.initial()
        for inp in timeout_seq.inputs:
            next_state, outputs, trans_type = self.fsm.next_state(current, inp)
            timeout_seq.expected_states.append(next_state)
            timeout_seq.expected_outputs.append(outputs)
            timeout_seq.covered_transitions.add((str(current), str(next_state)))
            current = next_state
        
        sequences.append(timeout_seq)
        
        # 2. Nonce blocking: try to re-enter after entry
        nonce_seq = TestSequence(
            name="nonce_blocking",
            description="Test nonce prevents immediate re-entry after exit",
            inputs=[
                entry_input,  # Entry - nonce set
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit with profit
                entry_input,  # Try re-entry - should be blocked by nonce
                entry_input,  # Try again
            ],
            expected_states=[],
            expected_outputs=[],
            covered_transitions=set()
        )
        
        current = AgentState.initial()
        for inp in nonce_seq.inputs:
            next_state, outputs, _ = self.fsm.next_state(current, inp)
            nonce_seq.expected_states.append(next_state)
            nonce_seq.expected_outputs.append(outputs)
            nonce_seq.covered_transitions.add((str(current), str(next_state)))
            current = next_state
        
        sequences.append(nonce_seq)
        
        # 3. All entry conditions failing individually
        for i, condition in enumerate(['price', 'volume', 'trend', 'holding', 'nonce', 'failure']):
            # Create input that fails only this condition
            if condition == 'price':
                bad_input = InputVector(i0=1, i1=1, i2=1, i3=1, i4=0)  # price high
            elif condition == 'volume':
                bad_input = InputVector(i0=0, i1=0, i2=1, i3=1, i4=0)  # volume low
            elif condition == 'trend':
                bad_input = InputVector(i0=0, i1=1, i2=0, i3=1, i4=0)  # trend bearish
            elif condition == 'failure':
                bad_input = InputVector(i0=0, i1=1, i2=1, i3=1, i4=1)  # failure echo
            else:
                continue  # holding and nonce need state setup
            
            condition_seq = TestSequence(
                name=f"entry_fail_{condition}",
                description=f"Test entry blocked when {condition} condition fails",
                inputs=[bad_input],
                expected_states=[],
                expected_outputs=[],
                covered_transitions=set()
            )
            
            current = AgentState.initial()
            next_state, outputs, _ = self.fsm.next_state(current, bad_input)
            condition_seq.expected_states.append(next_state)
            condition_seq.expected_outputs.append(outputs)
            condition_seq.covered_transitions.add((str(current), str(next_state)))
            
            sequences.append(condition_seq)
        
        # 4. Failure echo exit
        failure_exit_seq = TestSequence(
            name="failure_echo_exit",
            description="Test failure echo (i4=1) forces immediate exit",
            inputs=[
                entry_input,  # Entry
                InputVector(i0=0, i1=1, i2=1, i3=1, i4=1),  # Failure echo
            ],
            expected_states=[],
            expected_outputs=[],
            covered_transitions=set()
        )
        
        current = AgentState.initial()
        for inp in failure_exit_seq.inputs:
            next_state, outputs, _ = self.fsm.next_state(current, inp)
            failure_exit_seq.expected_states.append(next_state)
            failure_exit_seq.expected_outputs.append(outputs)
            failure_exit_seq.covered_transitions.add((str(current), str(next_state)))
            current = next_state
        
        sequences.append(failure_exit_seq)
        
        # 5. Profitable exit with burn
        profit_burn_seq = TestSequence(
            name="profitable_exit_burn",
            description="Test profitable exit triggers burn tracking",
            inputs=[
                entry_input,  # Entry at low price
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit at high price with guard
            ],
            expected_states=[],
            expected_outputs=[],
            covered_transitions=set()
        )
        
        current = AgentState.initial()
        for inp in profit_burn_seq.inputs:
            next_state, outputs, _ = self.fsm.next_state(current, inp)
            profit_burn_seq.expected_states.append(next_state)
            profit_burn_seq.expected_outputs.append(outputs)
            profit_burn_seq.covered_transitions.add((str(current), str(next_state)))
            current = next_state
        
        sequences.append(profit_burn_seq)
        
        # 6. Multiple trades with burn accumulation
        multi_trade_seq = TestSequence(
            name="multi_trade_burn_accum",
            description="Test burn flag persists across multiple trades",
            inputs=[],
            expected_states=[],
            expected_outputs=[],
            covered_transitions=set()
        )
        
        # Do 3 profitable trades
        for _ in range(3):
            multi_trade_seq.inputs.extend([
                entry_input,  # Entry
                InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Profitable exit
            ])
        
        current = AgentState.initial()
        for inp in multi_trade_seq.inputs:
            next_state, outputs, _ = self.fsm.next_state(current, inp)
            multi_trade_seq.expected_states.append(next_state)
            multi_trade_seq.expected_outputs.append(outputs)
            multi_trade_seq.covered_transitions.add((str(current), str(next_state)))
            current = next_state
        
        sequences.append(multi_trade_seq)
        
        return sequences
    
    def generate_exhaustive_per_state(self) -> List[TestSequence]:
        """
        Generate test sequences that exhaustively test all 32 inputs from each reachable state.
        """
        sequences = []
        
        for state in sorted(self.fsm.reachable_states, key=lambda s: s.to_tuple()):
            # Find path to this state
            path = self.find_path_to_state(state)
            if path is None and state != AgentState.initial():
                continue  # Can't reach this state
            
            # Setup inputs to reach the state
            setup_inputs = [inp for inp, _ in (path or [])]
            
            # Test all 32 inputs from this state
            for i in range(32):
                test_inp = InputVector.from_int(i)
                
                seq = TestSequence(
                    name=f"exhaust_{state}_inp{i:02d}",
                    description=f"Exhaustive test: state {state} with input {test_inp}",
                    inputs=setup_inputs + [test_inp],
                    expected_states=[],
                    expected_outputs=[],
                    covered_transitions=set()
                )
                
                # Compute expected
                current = AgentState.initial()
                for inp in seq.inputs:
                    next_state, outputs, _ = self.fsm.next_state(current, inp)
                    seq.expected_states.append(next_state)
                    seq.expected_outputs.append(outputs)
                    seq.covered_transitions.add((str(current), str(next_state)))
                    current = next_state
                
                sequences.append(seq)
        
        return sequences
    
    def compute_coverage(self, sequences: List[TestSequence]) -> CoverageResult:
        """Compute coverage metrics for given test sequences"""
        result = CoverageResult()
        result.total_states = len(self.fsm.reachable_states)
        result.total_transitions = len(self.fsm.transitions)
        
        covered_states: Set[AgentState] = {AgentState.initial()}
        covered_trans: Set[Tuple[AgentState, InputVector]] = set()
        
        for seq in sequences:
            current = AgentState.initial()
            covered_states.add(current)
            
            for inp, expected_state in zip(seq.inputs, seq.expected_states):
                trans_key = (current, inp)
                covered_trans.add(trans_key)
                covered_states.add(expected_state)
                current = expected_state
        
        result.covered_states = len(covered_states)
        result.covered_transitions = len(covered_trans)
        result.uncovered_states = self.fsm.reachable_states - covered_states
        result.uncovered_transitions = set(self.fsm.transitions.keys()) - covered_trans
        
        return result
    
    def print_coverage_report(self, sequences: List[TestSequence], result: CoverageResult):
        """Print detailed coverage report"""
        print("=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)
        
        print(f"\nState Coverage: {result.covered_states}/{result.total_states} ({result.state_coverage*100:.1f}%)")
        print(f"Transition Coverage: {result.covered_transitions}/{result.total_transitions} ({result.transition_coverage*100:.1f}%)")
        print(f"Test Sequences: {len(sequences)}")
        print(f"Total Test Steps: {sum(len(s.inputs) for s in sequences)}")
        
        if result.uncovered_states:
            print(f"\nUncovered States ({len(result.uncovered_states)}):")
            for state in result.uncovered_states:
                print(f"  - {state}")
        
        if result.uncovered_transitions:
            print(f"\nUncovered Transitions (showing first 10 of {len(result.uncovered_transitions)}):")
            for state, inp in list(result.uncovered_transitions)[:10]:
                next_state = self.fsm.transition_matrix[state][inp]
                print(f"  - {state} --{inp}--> {next_state}")
        
        print("=" * 70)


def main():
    """Generate coverage tests and report"""
    # Build FSM
    print("Building FSM model...")
    fsm = DeflationaryAgentFSM()
    fsm.enumerate_reachable_states()
    fsm.build_transition_table()
    
    # Generate coverage
    print("Generating test sequences...")
    generator = CoverageGenerator(fsm)
    
    # Generate different types of tests
    minimal_seq = generator.generate_minimal_coverage_sequence()
    boundary_seqs = generator.generate_boundary_tests()
    
    all_sequences = [minimal_seq] + boundary_seqs
    
    # Compute coverage
    result = generator.compute_coverage(all_sequences)
    generator.print_coverage_report(all_sequences, result)
    
    # Save test sequences
    output_file = Path(__file__).parent / "coverage_tests.json"
    tests_data = []
    for seq in all_sequences:
        tests_data.append({
            'name': seq.name,
            'description': seq.description,
            'inputs': [inp.to_tuple() for inp in seq.inputs],
            'expected_states': [s.to_tuple() for s in seq.expected_states],
            'step_count': len(seq.inputs)
        })
    
    with open(output_file, 'w') as f:
        json.dump(tests_data, f, indent=2)
    
    print(f"\nTest sequences saved to {output_file}")
    
    # Print individual test summaries
    print("\nTest Sequence Summary:")
    print("-" * 50)
    for seq in all_sequences:
        print(f"  {seq.name}: {len(seq.inputs)} steps")
    
    return generator, all_sequences, result


if __name__ == "__main__":
    main()

