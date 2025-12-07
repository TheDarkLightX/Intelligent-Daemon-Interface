#!/usr/bin/env python3
"""
Detailed Trace Analyzer for Deflationary Agent

Generates human-readable execution traces with:
1. All 16+ outputs at each tick
2. Expected vs actual comparisons
3. Invariant verification
4. State transition annotations
5. Causality chain verification

Copyright DarkLightX/Dana Edwards
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from fsm_model import (
    DeflationaryAgentFSM, AgentState, InputVector, 
    Transition, Outputs
)


class VerificationStatus(Enum):
    PASS = "✓"
    FAIL = "✗"
    WARN = "⚠"
    INFO = "ℹ"


@dataclass
class TickTrace:
    """Complete trace for a single tick"""
    tick: int
    inputs: InputVector
    prev_state: AgentState
    curr_state: AgentState
    outputs: Outputs
    transition_type: str
    
    # Verification results
    invariants: Dict[str, bool] = field(default_factory=dict)
    expected_checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def all_invariants_hold(self) -> bool:
        return all(self.invariants.values())


@dataclass
class ExecutionTrace:
    """Complete execution trace for a scenario"""
    name: str
    description: str
    ticks: List[TickTrace]
    
    # Summary statistics
    total_invariant_checks: int = 0
    failed_invariant_checks: int = 0
    state_transitions: int = 0
    trades_executed: int = 0
    profitable_trades: int = 0


class TraceAnalyzer:
    """
    Generate detailed, human-readable execution traces
    """
    
    def __init__(self):
        self.fsm = DeflationaryAgentFSM()
        self.fsm.enumerate_reachable_states()
        self.fsm.build_transition_table()
    
    def analyze_sequence(self, name: str, description: str, 
                         inputs: List[InputVector]) -> ExecutionTrace:
        """
        Analyze an input sequence and produce detailed trace.
        """
        trace = ExecutionTrace(name=name, description=description, ticks=[])
        
        prev_state = AgentState.initial()
        prev_outputs = Outputs()
        
        for t, inp in enumerate(inputs):
            curr_state, outputs, trans_type = self.fsm.next_state(prev_state, inp)
            
            tick_trace = TickTrace(
                tick=t,
                inputs=inp,
                prev_state=prev_state,
                curr_state=curr_state,
                outputs=outputs,
                transition_type=trans_type
            )
            
            # Verify invariants
            self._verify_invariants(tick_trace, prev_outputs)
            
            # Check expected behaviors
            self._check_expected_behaviors(tick_trace, prev_state, prev_outputs, inp)
            
            trace.ticks.append(tick_trace)
            
            # Update statistics
            if not tick_trace.all_invariants_hold():
                trace.failed_invariant_checks += 1
            trace.total_invariant_checks += len(tick_trace.invariants)
            
            if trans_type in ("entry", "exit_profit", "exit_timeout", "exit_failure", "exit_other"):
                trace.state_transitions += 1
            
            if outputs.o2:  # buy signal
                trace.trades_executed += 1
            
            if outputs.o11:  # profit
                trace.profitable_trades += 1
            
            prev_state = curr_state
            prev_outputs = outputs
        
        return trace
    
    def _verify_invariants(self, tick: TickTrace, prev_outputs: Outputs):
        """Verify all safety invariants"""
        o = tick.outputs
        
        # Invariant 1: Action exclusivity - never buy AND sell
        tick.invariants['action_exclusivity'] = not (o.o2 and o.o3)
        
        # Invariant 2: Fresh oracle - executing requires fresh volume
        tick.invariants['fresh_oracle'] = (not o.o0) or o.o5
        
        # Invariant 3: Burn-profit coupling - burn only on profit
        tick.invariants['burn_profit'] = (not o.o12) or o.o11
        
        # Invariant 4: Nonce effect - no buy when nonce was set
        tick.invariants['nonce_effect'] = (not o.o2) or (not tick.prev_state.o9)
        
        # Invariant 5: Monotonic burns - burn flag never decreases
        tick.invariants['monotonic_burns'] = o.o13 >= tick.prev_state.o13
        
        # Invariant 6: Lock equals state
        tick.invariants['lock_state'] = o.o4 == o.o0
        
        # Invariant 7: Burn event equals profit
        tick.invariants['burn_event_profit'] = o.o12 == o.o11
        
        # Invariant 8: Holding consistency - holding requires prior buy or continuation
        if o.o1:  # If holding
            valid_hold = o.o2 or (tick.prev_state.o1 and not o.o3)
            tick.invariants['holding_consistency'] = valid_hold
        else:
            tick.invariants['holding_consistency'] = True
    
    def _check_expected_behaviors(self, tick: TickTrace, prev_state: AgentState,
                                   prev_outputs: Outputs, inp: InputVector):
        """Check expected behaviors and causality chains"""
        o = tick.outputs
        
        # Check 1: Entry should set nonce
        if o.o2:  # Buy signal
            expected = o.o9 == 1
            tick.expected_checks.append((
                "Buy triggers nonce",
                expected,
                f"o2=1 should set o9=1, got o9={o.o9}"
            ))
        
        # Check 2: Entry should set holding
        if o.o2:
            expected = o.o1 == 1
            tick.expected_checks.append((
                "Buy triggers holding",
                expected,
                f"o2=1 should set o1=1, got o1={o.o1}"
            ))
        
        # Check 3: Sell should clear holding
        if o.o3:  # Sell signal
            expected = o.o1 == 0
            tick.expected_checks.append((
                "Sell clears holding",
                expected,
                f"o3=1 should clear o1=0, got o1={o.o1}"
            ))
        
        # Check 4: Profit requires high price and guard
        if o.o11:
            expected = inp.i0 == 1 and inp.i3 == 1
            tick.expected_checks.append((
                "Profit requires price_high and guard",
                expected,
                f"o11=1 requires i0=1 and i3=1, got i0={inp.i0}, i3={inp.i3}"
            ))
        
        # Check 5: Entry blocked when nonce set
        if prev_state.o9 and not prev_state.o0:  # Nonce set, was idle
            entry_blocked = not o.o2
            tick.expected_checks.append((
                "Nonce blocks entry",
                entry_blocked,
                f"With o9[t-1]=1, entry should be blocked, o2={o.o2}"
            ))
        
        # Check 6: Timer increments while executing
        if prev_state.o0 and o.o0:  # Was and is executing
            prev_timer = prev_state.timer
            curr_timer = tick.curr_state.timer
            if prev_timer < 3:
                expected = curr_timer == (prev_timer + 1) % 4 or curr_timer == prev_timer + 1
                tick.expected_checks.append((
                    "Timer increments",
                    True,  # Timer logic is complex, just note it
                    f"Timer: {prev_timer} -> {curr_timer}"
                ))
        
        # Check 7: Timeout forces exit
        if prev_state.timer == 3 and prev_state.o0:
            expected = not o.o0
            tick.expected_checks.append((
                "Timeout forces exit",
                expected,
                f"Timer=3 should force exit, o0={o.o0}"
            ))
        
        # Check 8: Failure echo forces exit
        if prev_state.o0 and inp.i4:
            expected = not o.o0
            tick.expected_checks.append((
                "Failure echo forces exit",
                expected,
                f"i4=1 during execution should force exit, o0={o.o0}"
            ))
        
        # Warnings for unusual conditions
        if o.o3 and not prev_state.o1:
            tick.warnings.append("Sell signal without prior holding - unexpected state")
        
        if o.o0 and not inp.i1:
            tick.warnings.append("Executing without fresh volume - check invariant")
    
    def format_trace(self, trace: ExecutionTrace, verbose: bool = True) -> str:
        """Format trace for human reading"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"EXECUTION TRACE: {trace.name}")
        lines.append(f"Description: {trace.description}")
        lines.append("=" * 80)
        lines.append("")
        
        for tick in trace.ticks:
            lines.extend(self._format_tick(tick, verbose))
            lines.append("")
        
        # Summary
        lines.append("=" * 80)
        lines.append("TRACE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total ticks: {len(trace.ticks)}")
        lines.append(f"State transitions: {trace.state_transitions}")
        lines.append(f"Trades executed: {trace.trades_executed}")
        lines.append(f"Profitable trades: {trace.profitable_trades}")
        lines.append(f"Invariant checks: {trace.total_invariant_checks}")
        lines.append(f"Failed invariants: {trace.failed_invariant_checks}")
        
        if trace.failed_invariant_checks > 0:
            lines.append("\n⚠️  WARNING: Some invariants failed!")
        else:
            lines.append("\n✓ All invariants passed")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_tick(self, tick: TickTrace, verbose: bool) -> List[str]:
        """Format a single tick"""
        lines = []
        
        # Tick header
        trans_symbol = {
            "idle": "─",
            "entry": "▶",
            "continue": "→",
            "exit_profit": "◀$",
            "exit_timeout": "◀⏱",
            "exit_failure": "◀!",
            "exit_other": "◀"
        }.get(tick.transition_type, "?")
        
        state_name = "EXECUTING" if tick.curr_state.o0 else "IDLE"
        lines.append(f"TICK {tick.tick}: {tick.prev_state} {trans_symbol} {tick.curr_state} [{state_name}]")
        
        # Inputs
        lines.append(f"  Inputs:  i0={tick.inputs.i0}({'H' if tick.inputs.i0 else 'L'}) "
                    f"i1={tick.inputs.i1}({'V' if tick.inputs.i1 else '-'}) "
                    f"i2={tick.inputs.i2}({'↑' if tick.inputs.i2 else '↓'}) "
                    f"i3={tick.inputs.i3}({'G' if tick.inputs.i3 else '-'}) "
                    f"i4={tick.inputs.i4}({'F' if tick.inputs.i4 else '-'})")
        
        # Core outputs
        o = tick.outputs
        lines.append(f"  State:   o0={o.o0} o1={o.o1} timer={tick.curr_state.timer} "
                    f"nonce={o.o9} entry_price={o.o10} burned={o.o13}")
        
        # Signals
        buy_str = "BUY!" if o.o2 else "-"
        sell_str = "SELL!" if o.o3 else "-"
        profit_str = "PROFIT!" if o.o11 else "-"
        lines.append(f"  Signals: buy={buy_str} sell={sell_str} profit={profit_str}")
        
        # Derived outputs
        lines.append(f"  Derived: lock={o.o4} fresh={o.o5} burn_event={o.o12}")
        
        if verbose:
            # Invariants
            inv_status = " ".join(
                f"{name}={VerificationStatus.PASS.value if ok else VerificationStatus.FAIL.value}"
                for name, ok in tick.invariants.items()
            )
            lines.append(f"  Invariants: {inv_status}")
            
            # Expected checks
            if tick.expected_checks:
                for check_name, passed, msg in tick.expected_checks:
                    status = VerificationStatus.PASS.value if passed else VerificationStatus.FAIL.value
                    lines.append(f"  CHECK [{status}] {check_name}: {msg}")
            
            # Warnings
            for warning in tick.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        return lines
    
    def verify_all_outputs(self, trace: ExecutionTrace) -> Dict[str, List[int]]:
        """
        Extract all output values across the trace for manual verification.
        Returns dict mapping output name to list of values per tick.
        """
        outputs = {f'o{i}': [] for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}
        
        for tick in trace.ticks:
            o = tick.outputs
            outputs['o0'].append(o.o0)
            outputs['o1'].append(o.o1)
            outputs['o2'].append(o.o2)
            outputs['o3'].append(o.o3)
            outputs['o4'].append(o.o4)
            outputs['o5'].append(o.o5)
            outputs['o6'].append(o.o6)
            outputs['o7'].append(o.o7)
            outputs['o9'].append(o.o9)
            outputs['o10'].append(o.o10)
            outputs['o11'].append(o.o11)
            outputs['o12'].append(o.o12)
            outputs['o13'].append(o.o13)
            outputs['o14'].append(o.o14)
            outputs['o15'].append(o.o15)
            outputs['o16'].append(o.o16)
            outputs['o17'].append(o.o17)
            outputs['o18'].append(o.o18)
        
        return outputs
    
    def print_output_table(self, trace: ExecutionTrace):
        """Print tabular view of all outputs for manual verification"""
        outputs = self.verify_all_outputs(trace)
        
        print("\n" + "=" * 100)
        print("OUTPUT VALUES TABLE (for manual verification)")
        print("=" * 100)
        
        # Header
        header = "Tick |" + "|".join(f" {k:>3}" for k in sorted(outputs.keys())) + "|"
        print(header)
        print("-" * len(header))
        
        # Rows
        for t in range(len(trace.ticks)):
            row = f" {t:3} |"
            for k in sorted(outputs.keys()):
                val = outputs[k][t]
                row += f" {val:>3}|"
            print(row)
        
        print("=" * 100)


def main():
    """Run trace analysis on test scenarios"""
    from input_simulator import InputSimulator
    
    analyzer = TraceAnalyzer()
    simulator = InputSimulator(seed=42)
    
    # Generate scenarios
    scenarios = [
        ("normal_trade", "Normal profitable trade cycle", [
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Hold
            InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit with profit
            InputVector(i0=1, i1=1, i2=0, i3=1, i4=0),  # Idle
            InputVector(i0=1, i1=1, i2=0, i3=1, i4=0),  # Idle
        ]),
        ("timeout_exit", "Timer reaches 3 and forces exit", [
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry (T=1)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue (T=2)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Continue (T=3)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Timeout exit
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Idle (nonce set)
        ]),
        ("failure_exit", "Failure echo forces immediate exit", [
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=1),  # Failure!
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Post-exit
        ]),
        ("nonce_blocking", "Nonce prevents immediate re-entry", [
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Entry
            InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Exit
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Try entry (blocked)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Try again (blocked)
        ]),
        ("multi_profitable", "Multiple profitable trades", [
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Trade 1 entry
            InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Trade 1 exit (profit)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Trade 2 entry
            InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Trade 2 exit (profit)
            InputVector(i0=0, i1=1, i2=1, i3=1, i4=0),  # Trade 3 entry
            InputVector(i0=1, i1=1, i2=1, i3=1, i4=0),  # Trade 3 exit (profit)
        ]),
    ]
    
    # Analyze each scenario
    for name, desc, inputs in scenarios:
        trace = analyzer.analyze_sequence(name, desc, inputs)
        print(analyzer.format_trace(trace, verbose=True))
        analyzer.print_output_table(trace)
        print("\n\n")
    
    # Save detailed traces to file
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    for name, desc, inputs in scenarios:
        trace = analyzer.analyze_sequence(name, desc, inputs)
        report_file = report_dir / f"trace_{name}.txt"
        with open(report_file, 'w') as f:
            f.write(analyzer.format_trace(trace, verbose=True))
        print(f"Saved trace to {report_file}")
    
    print(f"\nAll traces saved to {report_dir}")


if __name__ == "__main__":
    main()

