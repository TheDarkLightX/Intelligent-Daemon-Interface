#!/usr/bin/env python3
"""
Explicit Finite State Machine Model for Deflationary Agent V35

This module builds an explicit FSM from the tau specification,
enabling complete state enumeration and transition coverage analysis.

State Space:
- o0 (executing): 0 or 1
- o1 (holding): 0 or 1
- o6,o7 (timer): 0-3 (2 bits)
- o9 (nonce): 0 or 1
- o10 (entry_price): 0 or 1
- o13 (has_burned): 0 or 1

Theoretical: 2 * 2 * 4 * 2 * 2 * 2 = 128 states
Reachable: ~16 states (due to invariants)

Copyright DarkLightX/Dana Edwards
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import deque
import json


@dataclass(frozen=True)
class AgentState:
    """Immutable agent state for FSM analysis"""
    o0: int  # executing
    o1: int  # holding
    o6: int  # timer bit 0
    o7: int  # timer bit 1
    o9: int  # nonce
    o10: int  # entry_price
    o13: int  # has_burned
    last_sell: int  # previous sell pulse (o3[t-1])
    
    @property
    def timer(self) -> int:
        """2-bit timer value"""
        return self.o7 * 2 + self.o6
    
    @property
    def is_idle(self) -> bool:
        return self.o0 == 0
    
    @property
    def is_executing(self) -> bool:
        return self.o0 == 1
    
    def __str__(self) -> str:
        state_name = "EXEC" if self.o0 else "IDLE"
        hold = "H" if self.o1 else "-"
        timer = f"T{self.timer}"
        nonce = "N" if self.o9 else "-"
        entry = f"E{self.o10}"
        burned = "B" if self.o13 else "-"
        cooldown = "S" if self.last_sell else "-"
        return f"{state_name}({hold},{timer},{nonce},{entry},{burned},{cooldown})"
    
    def to_tuple(self) -> Tuple[int, ...]:
        return (self.o0, self.o1, self.o6, self.o7, self.o9, self.o10, self.o13, self.last_sell)
    
    @classmethod
    def from_tuple(cls, t: Tuple[int, ...]) -> 'AgentState':
        return cls(o0=t[0], o1=t[1], o6=t[2], o7=t[3], o9=t[4], o10=t[5], o13=t[6], last_sell=t[7])
    
    @classmethod
    def initial(cls) -> 'AgentState':
        """Initial state: all zeros"""
        return cls(o0=0, o1=0, o6=0, o7=0, o9=0, o10=0, o13=0, last_sell=0)


@dataclass(frozen=True)
class InputVector:
    """Input vector for FSM transitions"""
    i0: int  # price (0=low, 1=high)
    i1: int  # volume (0=low, 1=high)
    i2: int  # trend (0=bearish, 1=bullish)
    i3: int  # profit_guard
    i4: int  # failure_echo
    
    def __str__(self) -> str:
        price = "H" if self.i0 else "L"
        vol = "V" if self.i1 else "-"
        trend = "↑" if self.i2 else "↓"
        guard = "G" if self.i3 else "-"
        fail = "F" if self.i4 else "-"
        return f"({price},{vol},{trend},{guard},{fail})"
    
    def to_tuple(self) -> Tuple[int, ...]:
        return (self.i0, self.i1, self.i2, self.i3, self.i4)
    
    @classmethod
    def from_tuple(cls, t: Tuple[int, ...]) -> 'InputVector':
        return cls(i0=t[0], i1=t[1], i2=t[2], i3=t[3], i4=t[4])
    
    @classmethod
    def from_int(cls, n: int) -> 'InputVector':
        """Create from 5-bit integer"""
        return cls(
            i0=(n >> 0) & 1,
            i1=(n >> 1) & 1,
            i2=(n >> 2) & 1,
            i3=(n >> 3) & 1,
            i4=(n >> 4) & 1
        )


@dataclass
class Outputs:
    """All output signals for a single tick"""
    # Core state
    o0: int = 0   # executing
    o1: int = 0   # holding
    o2: int = 0   # buy_signal
    o3: int = 0   # sell_signal
    
    # Derived (V35)
    o4: int = 0   # lock (= o0)
    o5: int = 0   # oracle_fresh (= i1)
    
    # Timer
    o6: int = 0   # timer_b0
    o7: int = 0   # timer_b1
    
    # State
    o9: int = 0   # nonce
    o10: int = 0  # entry_price
    o11: int = 0  # profit
    o12: int = 0  # burn_event (= o11)
    o13: int = 0  # has_burned
    
    # Observable invariants (V35)
    o14: int = 1  # action_exclusivity
    o15: int = 1  # fresh_oracle_execution
    o16: int = 1  # burn_profit_coupling
    o17: int = 1  # nonce_effect
    o18: int = 0  # progress_flag
    
    def to_dict(self) -> Dict[str, int]:
        return {f'o{i}': getattr(self, f'o{i}') for i in 
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}


@dataclass
class Transition:
    """A single FSM transition"""
    from_state: AgentState
    input_vec: InputVector
    to_state: AgentState
    outputs: Outputs
    transition_type: str  # "entry", "continue", "exit_profit", "exit_timeout", "exit_failure", "idle"
    
    def __str__(self) -> str:
        return f"{self.from_state} --{self.input_vec}--> {self.to_state} [{self.transition_type}]"


class DeflationaryAgentFSM:
    """
    Explicit Finite State Machine for Deflationary Agent V35
    
    Implements the exact semantics from agent4_testnet_v35.tau
    """
    
    def __init__(self):
        self.reachable_states: Set[AgentState] = set()
        self.transitions: Dict[Tuple[AgentState, InputVector], Transition] = {}
        self.transition_matrix: Dict[AgentState, Dict[InputVector, AgentState]] = {}
        
    # === Helper predicates (from tau spec) ===
    
    def valid_entry(self, i0: int, i1: int, i2: int, o1_prev: int) -> bool:
        """valid_entry(p, v, t, h) := p' & v & t & h'"""
        return (not i0) and i1 and i2 and (not o1_prev)
    
    def valid_exit(self, i0: int, o1_prev: int) -> bool:
        """valid_exit(p, h) := p & h"""
        return i0 and o1_prev
    
    def timed_out(self, o7: int, o6: int) -> bool:
        """timed_out(b1, b0) := b1 & b0  (timer = 11 = 3)"""
        return o7 and o6
    
    # === State transition function ===
    
    def next_state(self, prev: AgentState, inp: InputVector) -> Tuple[AgentState, Outputs, str]:
        """
        Compute next state given previous state and input.
        Returns (next_state, outputs, transition_type)
        """
        outputs = Outputs()
        
        # Timer value
        timeout = self.timed_out(prev.o7, prev.o6)
        
        # === STATE MACHINE (o0) ===
        # Entry condition
        entry_cond = (
            (not prev.o0) and
            self.valid_entry(inp.i0, inp.i1, inp.i2, prev.o1) and
            inp.i1 and
            (not timeout) and
            (not prev.o9) and
            (not inp.i4) and
            (not prev.last_sell)
        )
        
        # Continue condition
        continue_cond = (
            prev.o0 and
            (not self.valid_exit(inp.i0, prev.o1)) and
            (not timeout) and
            inp.i1 and
            (not inp.i4)
        )
        
        o0_next = int(entry_cond or continue_cond)
        
        # === TRADING SIGNALS ===
        o2_next = int(o0_next and (not prev.o0) and (not prev.o1))  # buy
        o3_next = int(prev.o0 and (not o0_next) and prev.o1)        # sell
        o1_next = int(o2_next or ((not o3_next) and prev.o1))       # holding
        
        # === TIMER ===
        o6_next = int(o0_next and (not prev.o6))
        xor_val = (prev.o7 and not prev.o6) or (not prev.o7 and prev.o6)
        o7_next = int(o0_next and xor_val)
        
        # === NONCE ===
        o9_next = int(o2_next or (prev.o0 and (not o3_next) and prev.o9))
        
        # === ECONOMICS ===
        o10_next = int((o2_next and inp.i0) or 
                       (prev.o0 and (not o2_next) and (not o3_next) and prev.o10))
        o11_next = int(o3_next and inp.i0 and (not prev.o10) and inp.i3)
        
        # === BURN TRACKING ===
        o13_next = int(prev.o13 or o11_next)
        
        # === DERIVED OUTPUTS ===
        o4_next = o0_next  # lock = executing
        o5_next = inp.i1   # oracle_fresh = volume
        o12_next = o11_next  # burn_event = profit
        
        # === OBSERVABLE INVARIANTS ===
        o18_next = int(o2_next or o3_next or 
                       (prev.o7 and prev.o6) or 
                       (o0_next and (not prev.o0)))  # progress
        
        # Invariants (should always be 1)
        o14_next = int((not o18_next) or not (o2_next and o3_next))  # action_excl
        o15_next = int((not o18_next) or ((not o0_next) or o5_next))  # fresh_exec
        o16_next = int((not o18_next) or ((not o12_next) or o11_next))  # burn_profit
        o17_next = int((not o18_next) or ((not o2_next) or (not prev.o9)))  # nonce_effect
        
        # Build output struct
        outputs.o0 = o0_next
        outputs.o1 = o1_next
        outputs.o2 = o2_next
        outputs.o3 = o3_next
        outputs.o4 = o4_next
        outputs.o5 = o5_next
        outputs.o6 = o6_next
        outputs.o7 = o7_next
        outputs.o9 = o9_next
        outputs.o10 = o10_next
        outputs.o11 = o11_next
        outputs.o12 = o12_next
        outputs.o13 = o13_next
        outputs.o14 = o14_next
        outputs.o15 = o15_next
        outputs.o16 = o16_next
        outputs.o17 = o17_next
        outputs.o18 = o18_next
        
        # Build next state
        next_state = AgentState(
            o0=o0_next,
            o1=o1_next,
            o6=o6_next,
            o7=o7_next,
            o9=o9_next,
            o10=o10_next,
            o13=o13_next,
            last_sell=o3_next
        )
        
        # Determine transition type
        if not prev.o0 and o0_next:
            trans_type = "entry"
        elif prev.o0 and o0_next:
            trans_type = "continue"
        elif prev.o0 and not o0_next:
            if o11_next:
                trans_type = "exit_profit"
            elif timeout:
                trans_type = "exit_timeout"
            elif inp.i4:
                trans_type = "exit_failure"
            else:
                trans_type = "exit_other"
        else:
            trans_type = "idle"
        
        return next_state, outputs, trans_type
    
    # === State enumeration ===
    
    def enumerate_reachable_states(self) -> Set[AgentState]:
        """
        Use BFS to enumerate all reachable states from initial state.
        """
        initial = AgentState.initial()
        visited: Set[AgentState] = set()
        queue = deque([initial])
        
        while queue:
            state = queue.popleft()
            if state in visited:
                continue
            visited.add(state)
            
            # Try all 32 input combinations
            for i in range(32):
                inp = InputVector.from_int(i)
                next_state, _, _ = self.next_state(state, inp)
                
                if next_state not in visited:
                    queue.append(next_state)
        
        self.reachable_states = visited
        return visited
    
    def build_transition_table(self):
        """
        Build complete transition table for all reachable states
        and all input combinations.
        """
        if not self.reachable_states:
            self.enumerate_reachable_states()
        
        for state in self.reachable_states:
            self.transition_matrix[state] = {}
            
            for i in range(32):
                inp = InputVector.from_int(i)
                next_state, outputs, trans_type = self.next_state(state, inp)
                
                transition = Transition(
                    from_state=state,
                    input_vec=inp,
                    to_state=next_state,
                    outputs=outputs,
                    transition_type=trans_type
                )
                
                self.transitions[(state, inp)] = transition
                self.transition_matrix[state][inp] = next_state
    
    def get_state_graph(self) -> Dict[str, List[Dict]]:
        """
        Get state graph suitable for visualization.
        """
        if not self.transitions:
            self.build_transition_table()
        
        nodes = [{"id": str(s), "label": str(s)} for s in self.reachable_states]
        
        # Aggregate edges by (from, to) to reduce clutter
        edge_map: Dict[Tuple[str, str], List[str]] = {}
        for (state, inp), trans in self.transitions.items():
            key = (str(state), str(trans.to_state))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(f"{inp}:{trans.transition_type}")
        
        edges = [
            {
                "from": from_s, 
                "to": to_s, 
                "labels": labels[:3] + (["..."] if len(labels) > 3 else [])
            }
            for (from_s, to_s), labels in edge_map.items()
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    def print_summary(self):
        """Print FSM summary statistics"""
        if not self.reachable_states:
            self.enumerate_reachable_states()
        if not self.transitions:
            self.build_transition_table()
        
        print("=" * 60)
        print("DEFLATIONARY AGENT V35 - FSM MODEL SUMMARY")
        print("=" * 60)
        print(f"\nTheoretical state space: 128 states")
        print(f"Reachable states: {len(self.reachable_states)}")
        print(f"Input combinations: 32 per state")
        print(f"Total transitions: {len(self.transitions)}")
        
        # Count by type
        type_counts: Dict[str, int] = {}
        for trans in self.transitions.values():
            type_counts[trans.transition_type] = type_counts.get(trans.transition_type, 0) + 1
        
        print(f"\nTransition types:")
        for t, c in sorted(type_counts.items()):
            print(f"  {t}: {c}")
        
        print(f"\nReachable states:")
        for state in sorted(self.reachable_states, key=lambda s: s.to_tuple()):
            print(f"  {state}")
        
        # Check invariants
        print(f"\nInvariant check:")
        invariant_violations = 0
        for trans in self.transitions.values():
            if not trans.outputs.o14:
                print(f"  VIOLATION: action_excl at {trans}")
                invariant_violations += 1
            if not trans.outputs.o15:
                print(f"  VIOLATION: fresh_exec at {trans}")
                invariant_violations += 1
            if not trans.outputs.o16:
                print(f"  VIOLATION: burn_profit at {trans}")
                invariant_violations += 1
            if not trans.outputs.o17:
                print(f"  VIOLATION: nonce_effect at {trans}")
                invariant_violations += 1
        
        if invariant_violations == 0:
            print("  All invariants hold for all transitions ✓")
        else:
            print(f"  TOTAL VIOLATIONS: {invariant_violations}")
        
        print("=" * 60)


def main():
    """Build and analyze FSM model"""
    fsm = DeflationaryAgentFSM()
    
    # Enumerate reachable states
    print("Enumerating reachable states...")
    states = fsm.enumerate_reachable_states()
    
    # Build transition table
    print("Building transition table...")
    fsm.build_transition_table()
    
    # Print summary
    fsm.print_summary()
    
    # Save state graph
    graph = fsm.get_state_graph()
    graph_file = "/home/trevormoc/Downloads/DeflationaryAgent/verification/fsm_graph.json"
    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"\nState graph saved to {graph_file}")
    
    return fsm


if __name__ == "__main__":
    main()

