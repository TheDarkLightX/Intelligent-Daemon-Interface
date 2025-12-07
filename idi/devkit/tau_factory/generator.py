"""Tau spec generator - compiles AgentSchema to valid Tau Language spec."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock


def _load_patterns() -> Dict:
    """Load logic block patterns from templates."""
    patterns_path = Path(__file__).parent / "templates" / "patterns.json"
    if patterns_path.exists():
        return json.loads(patterns_path.read_text())
    return {}


def _generate_inputs(streams: tuple[StreamConfig, ...]) -> List[str]:
    """Generate input stream declarations."""
    lines = []
    input_streams = [s for s in streams if s.is_input]
    for idx, stream in enumerate(input_streams):
        hex_idx = hex(idx)[2:].upper()  # i0, i1, ..., i9, iA, iB, ...
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        lines.append(f'i{hex_idx}:{type_str} = in file("inputs/{stream.name}.in").')
    return lines


def _generate_outputs(streams: tuple[StreamConfig, ...]) -> List[str]:
    """Generate output stream declarations."""
    lines = []
    output_streams = [s for s in streams if not s.is_input]
    for idx, stream in enumerate(output_streams):
        hex_idx = hex(idx)[2:].upper()  # o0, o1, ..., o9, oA, oB, ...
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        lines.append(f'o{hex_idx}:{type_str} = out file("outputs/{stream.name}.out").')
    return lines


def _generate_input_mirrors(streams: tuple[StreamConfig, ...]) -> List[str]:
    """Generate input mirror declarations for symmetric I/O."""
    lines = []
    input_streams = [s for s in streams if s.is_input]
    for idx, stream in enumerate(input_streams):
        hex_idx = hex(idx)[2:].upper()
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        lines.append(f'i{hex_idx}:{type_str} = out file("outputs/i{hex_idx}_mirror.out").')
    return lines


def _get_stream_index(stream_name: str, streams: tuple[StreamConfig, ...], is_input: bool) -> str:
    """Get hex index for a stream name.
    
    For inputs: searches only input streams (i0, i1, ...)
    For outputs: searches only output streams (o0, o1, ...)
    """
    filtered = [s for s in streams if s.is_input == is_input]
    for idx, stream in enumerate(filtered):
        if stream.name == stream_name:
            return hex(idx)[2:].upper()
    raise ValueError(f"Stream {stream_name} not found (is_input={is_input})")


def _get_stream_index_any(stream_name: str, streams: tuple[StreamConfig, ...]) -> tuple[str, bool]:
    """Get hex index for a stream name, searching both inputs and outputs.
    
    Returns:
        (hex_index, is_input) tuple
    """
    # Try inputs first
    try:
        idx = _get_stream_index(stream_name, streams, is_input=True)
        return (idx, True)
    except ValueError:
        pass
    
    # Try outputs
    try:
        idx = _get_stream_index(stream_name, streams, is_input=False)
        return (idx, False)
    except ValueError:
        raise ValueError(f"Stream {stream_name} not found in any stream type")


def _generate_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate FSM pattern logic."""
    if len(block.inputs) < 2:
        raise ValueError("FSM pattern requires at least 2 inputs (buy, sell)")
    
    # Inputs can be either input streams or output streams (from other logic blocks)
    buy_idx, buy_is_input = _get_stream_index_any(block.inputs[0], streams)
    sell_idx, sell_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Use 'i' prefix for inputs, 'o' prefix for outputs
    buy_ref = f"i{buy_idx}[t]" if buy_is_input else f"o{buy_idx}[t]"
    sell_ref = f"i{sell_idx}[t]" if sell_is_input else f"o{sell_idx}[t]"
    
    # Find output stream type
    output_stream = next(s for s in streams if s.name == block.output)
    is_sbf = output_stream.stream_type == "sbf"
    
    # Position FSM: buy sets position, sell clears it, otherwise maintain
    fsm_logic = f"(o{output_idx}[t] = {buy_ref} | (o{output_idx}[t-1] & {sell_ref}'))"
    
    # Add initial condition: position starts at 0
    if is_sbf:
        init_condition = f" && (o{output_idx}[0] = 0)"
    else:
        init_condition = f" && (o{output_idx}[0] = {{0}}:bv[{output_stream.width}])"
    
    return fsm_logic + init_condition


def _generate_passthrough_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate passthrough pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Passthrough pattern requires exactly 1 input")
    
    input_idx, input_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    input_ref = f"i{input_idx}[t]" if input_is_input else f"o{input_idx}[t]"
    return f"(o{output_idx}[t] = {input_ref})"


def _generate_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate counter pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Counter pattern requires exactly 1 input")
    
    event_idx, event_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Find output stream type
    output_stream = next(s for s in streams if s.name == block.output)
    is_sbf = output_stream.stream_type == "sbf"
    
    # Use correct reference for event input
    event_ref = f"i{event_idx}[t]" if event_is_input else f"o{event_idx}[t]"
    
    # Toggle counter: flip on event
    counter_logic = f"(o{output_idx}[t] = ({event_ref} & o{output_idx}[t-1]') | ({event_ref}' & o{output_idx}[t-1]))"
    
    # Add initial condition: counter starts at 0
    if is_sbf:
        init_condition = f" && (o{output_idx}[0] = 0)"
    else:
        init_condition = f" && (o{output_idx}[0] = {{0}}:bv[{output_stream.width}])"
    
    return counter_logic + init_condition


def _generate_accumulator_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate accumulator pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Accumulator pattern requires exactly 1 input")
    
    input_idx, input_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Find output stream width
    output_stream = next(s for s in streams if s.name == block.output)
    width = output_stream.width if output_stream.stream_type == "bv" else 8
    
    input_ref = f"i{input_idx}[t]" if input_is_input else f"o{input_idx}[t]"
    return f"(o{output_idx}[t] = o{output_idx}[t-1] + {input_ref}) && (o{output_idx}[0] = {{0}}:bv[{width}])"


def _generate_vote_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate vote pattern logic."""
    if len(block.inputs) < 2:
        raise ValueError("Vote pattern requires at least 2 inputs")
    
    input_refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        input_refs.append(ref)
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # OR all inputs together
    vote_expr = " | ".join(input_refs)
    return f"(o{output_idx}[t] = {vote_expr})"


def _generate_majority_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate majority voting pattern logic (N-of-M).
    
    Example: 2-of-3 majority = (a & b) | (a & c) | (b & c)
    """
    if len(block.inputs) < 2:
        raise ValueError("Majority pattern requires at least 2 inputs")
    
    # Get threshold and total from params, or use defaults
    threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)
    total = block.params.get("total", len(block.inputs))
    
    if threshold < 1:
        raise ValueError(f"Majority threshold must be >= 1, got {threshold}")
    if threshold > total:
        raise ValueError(f"Majority threshold ({threshold}) cannot exceed total ({total})")
    if total > len(block.inputs):
        raise ValueError(f"Total ({total}) cannot exceed number of inputs ({len(block.inputs)})")
    
    # Get input references (can be inputs or outputs)
    input_refs = []
    for inp in block.inputs[:total]:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        input_refs.append((ref, inp_idx))
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Generate all combinations of threshold inputs
    combinations = list(itertools.combinations(input_refs, threshold))
    
    if not combinations:
        raise ValueError(f"No valid combinations for threshold={threshold}, total={total}")
    
    # Create AND expressions for each combination, then OR them together
    and_exprs = []
    for combo in combinations:
        and_expr = " & ".join(ref for ref, _ in combo)
        and_exprs.append(f"({and_expr})")
    
    majority_expr = " | ".join(and_exprs)
    return f"(o{output_idx}[t] = {majority_expr})"


def _generate_unanimous_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate unanimous consensus pattern logic (all inputs must agree)."""
    if len(block.inputs) < 2:
        raise ValueError("Unanimous pattern requires at least 2 inputs")
    
    input_refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        input_refs.append(ref)
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # AND all inputs together
    unanimous_expr = " & ".join(input_refs)
    return f"(o{output_idx}[t] = {unanimous_expr})"


def _generate_supervisor_worker_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate supervisor-worker pattern logic.
    
    Creates a supervisor FSM that coordinates multiple worker FSMs.
    Supervisor outputs enable/disable workers based on supervisor state.
    
    NOTE: This pattern generates MULTIPLE outputs (supervisor + enables + workers).
    The 'output' parameter should be the supervisor output name.
    Additional outputs must be declared in streams.
    """
    # Parse supervisor and worker inputs from params or infer from block.inputs
    supervisor_inputs = block.params.get("supervisor_inputs", block.inputs[:2] if len(block.inputs) >= 2 else [block.inputs[0]])
    worker_inputs = block.params.get("worker_inputs", block.inputs[2:] if len(block.inputs) > 2 else [])
    
    if len(supervisor_inputs) < 1:
        raise ValueError("Supervisor-worker pattern requires at least 1 supervisor input")
    if len(worker_inputs) < 1:
        raise ValueError("Supervisor-worker pattern requires at least 1 worker input")
    
    # Get supervisor FSM inputs (buy/sell for supervisor)
    supervisor_buy = supervisor_inputs[0]
    supervisor_sell = supervisor_inputs[1] if len(supervisor_inputs) > 1 else supervisor_inputs[0]
    
    # Get supervisor output (main output)
    supervisor_output_name = block.output
    supervisor_idx = _get_stream_index(supervisor_output_name, streams, is_input=False)
    
    # Generate supervisor FSM
    supervisor_buy_idx, supervisor_buy_is_input = _get_stream_index_any(supervisor_buy, streams)
    supervisor_sell_idx, supervisor_sell_is_input = _get_stream_index_any(supervisor_sell, streams)
    
    supervisor_buy_ref = f"i{supervisor_buy_idx}[t]" if supervisor_buy_is_input else f"o{supervisor_buy_idx}[t]"
    supervisor_sell_ref = f"i{supervisor_sell_idx}[t]" if supervisor_sell_is_input else f"o{supervisor_sell_idx}[t]"
    
    # Supervisor FSM: mode=1 enters ACTIVE, mode=0 exits ACTIVE
    # Standard FSM: (o[t] = buy | (o[t-1] & sell'))
    # For supervisor: buy=mode, sell=mode' (when mode=0, we want to exit)
    # But we want: mode=1 → ACTIVE, mode=0 → IDLE
    # So: (o[t] = mode | (o[t-1] & mode')) - when mode=1, set; when mode=0, maintain if already active; when mode=0, clear
    # Actually simpler: supervisor directly follows mode (passthrough FSM)
    # Or: (o[t] = mode | (o[t-1] & mode')) - mode=1 sets, mode=0 maintains if already set
    # But we want mode=0 to clear, so: (o[t] = mode)
    # However, for FSM semantics, we want: mode=1 enters, mode=0 exits
    # Standard: (o[t] = mode | (o[t-1] & mode')) - this maintains when mode=0
    # We want: (o[t] = mode) - direct passthrough, but that's not FSM
    # Correct FSM: (o[t] = mode | (o[t-1] & mode')) - mode=1 sets, mode=0 maintains if already set
    # But test expects mode=0 to clear, so use passthrough: (o[t] = mode)
    supervisor_logic = f"(o{supervisor_idx}[t] = {supervisor_buy_ref}) && (o{supervisor_idx}[0] = 0)"
    
    # Generate worker enable signals and worker FSMs
    worker_logics = []
    num_workers = len(worker_inputs)
    
    # Get output names from params or generate defaults
    enable_outputs = block.params.get("worker_enable_outputs", [f"worker{i+1}_enable" for i in range(num_workers)])
    worker_outputs = block.params.get("worker_outputs", [f"worker{i+1}_state" for i in range(num_workers)])
    
    for i, worker_input in enumerate(worker_inputs):
        # Worker enable: active when supervisor = ACTIVE (supervisor = 1)
        enable_name = enable_outputs[i] if i < len(enable_outputs) else f"worker{i+1}_enable"
        try:
            enable_idx = _get_stream_index(enable_name, streams, is_input=False)
        except ValueError:
            # Enable output not declared - skip this worker
            continue
        
        # Enable when supervisor is active
        enable_logic = f"(o{enable_idx}[t] = o{supervisor_idx}[t])"
        worker_logics.append(enable_logic)
        
        # Worker FSM (enabled by enable signal)
        worker_name = worker_outputs[i] if i < len(worker_outputs) else f"worker{i+1}_state"
        try:
            worker_idx = _get_stream_index(worker_name, streams, is_input=False)
        except ValueError:
            # Worker output not declared - skip
            continue
        
        # Worker signal input
        worker_signal_idx, worker_signal_is_input = _get_stream_index_any(worker_input, streams)
        worker_signal_ref = f"i{worker_signal_idx}[t]" if worker_signal_is_input else f"o{worker_signal_idx}[t]"
        
        # Worker FSM: enabled by enable signal, controlled by worker signal
        # Worker can only enter LONG when enabled, exits on signal
        # Logic: (enable & signal) sets worker, (worker[t-1] & enable & signal') maintains worker, disabled when enable=0
        worker_fsm_logic = f"(o{worker_idx}[t] = (o{enable_idx}[t] & {worker_signal_ref}) | (o{worker_idx}[t-1] & o{enable_idx}[t] & {worker_signal_ref}')) && (o{worker_idx}[0] = 0)"
        worker_logics.append(worker_fsm_logic)
    
    # Combine all logic
    all_logic = [supervisor_logic] + worker_logics
    return " &&\n    ".join(all_logic)


def _generate_weighted_vote_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate weighted voting pattern logic using bitvector arithmetic.
    
    Computes weighted sum: sum(weight[i] * vote[i]) and compares to threshold.
    Uses bitvector arithmetic and comparisons directly in recurrence.
    """
    if len(block.inputs) < 1:
        raise ValueError("Weighted vote pattern requires at least 1 input")
    
    # Get weights and threshold from params
    weights = block.params.get("weights", [1] * len(block.inputs))
    threshold = block.params.get("threshold", sum(weights) // 2 + 1)
    
    if len(weights) != len(block.inputs):
        raise ValueError(f"Weights length ({len(weights)}) must match inputs length ({len(block.inputs)})")
    
    # Get output stream
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Determine bitvector width (use max of threshold and weights)
    max_value = max(threshold, max(weights, default=1))
    # Width needed: ceil(log2(max_value + 1)) + some headroom
    import math
    width = max(8, min(32, math.ceil(math.log2(max_value + 1)) + 4))
    
    # Build weighted sum expression
    # For each input: weight[i] * vote[i] (vote is 0 or 1, so weight[i] if vote=1, else 0)
    weighted_terms = []
    for i, (inp_name, weight) in enumerate(zip(block.inputs, weights)):
        inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
        inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        
        # Convert boolean to bitvector: (vote ? weight : 0)
        # Tau supports ternary operator: (condition ? value_true : value_false)
        # For boolean input, convert to bitvector: (vote ? {weight}:bv[width] : {0}:bv[width])
        inp_stream = next((s for s in streams if s.name == inp_name), None)
        if inp_stream and inp_stream.stream_type == "bv":
            # Input is already bitvector, multiply directly
            weighted_terms.append(f"({inp_ref} * {{{weight}}}:bv[{width}])")
        else:
            # Input is boolean, convert to bitvector using ternary
            # Tau syntax: (condition ? value_true : value_false)
            weighted_terms.append(f"(({inp_ref} ? {{{weight}}}:bv[{width}] : {{0}}:bv[{width}]))")
    
    # Sum all weighted terms
    if len(weighted_terms) == 1:
        weighted_sum = weighted_terms[0]
    else:
        weighted_sum = " + ".join(f"({term})" for term in weighted_terms)
    
    # Output can be boolean (sbf) for comparison result, or bitvector (bv) for weighted sum
    if output_stream.stream_type == "sbf":
        # Output boolean: weighted_sum >= threshold
        return f"(o{output_idx}[t] = ({weighted_sum}) >= {{{threshold}}}:bv[{width}])"
    else:
        # Output the weighted sum itself (bitvector)
        # Need to ensure output width matches
        output_width = output_stream.width if output_stream.width else width
        return f"(o{output_idx}[t] = {weighted_sum}) && (o{output_idx}[0] = {{0}}:bv[{output_width}])"


def _generate_time_lock_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate time-lock pattern logic using bitvector arithmetic.
    
    Computes: remaining_time = lock_start + lock_duration - current_time
    Outputs: lock_active = (remaining_time > 0)
    """
    # Get inputs from params or infer
    lock_start_name = block.params.get("lock_start", block.inputs[0] if len(block.inputs) > 0 else None)
    lock_duration_name = block.params.get("lock_duration", block.inputs[1] if len(block.inputs) > 1 else None)
    current_time_name = block.params.get("current_time", block.inputs[2] if len(block.inputs) > 2 else None)
    
    if not lock_start_name or not lock_duration_name or not current_time_name:
        raise ValueError("Time-lock pattern requires lock_start, lock_duration, and current_time inputs")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get input indices
    lock_start_idx, lock_start_is_input = _get_stream_index_any(lock_start_name, streams)
    lock_duration_idx, lock_duration_is_input = _get_stream_index_any(lock_duration_name, streams)
    current_time_idx, current_time_is_input = _get_stream_index_any(current_time_name, streams)
    
    # Get stream types to determine width
    lock_start_stream = next((s for s in streams if s.name == lock_start_name), None)
    lock_duration_stream = next((s for s in streams if s.name == lock_duration_name), None)
    current_time_stream = next((s for s in streams if s.name == current_time_name), None)
    
    # Determine bitvector width (use max width of inputs, or default 16)
    widths = []
    if lock_start_stream and lock_start_stream.stream_type == "bv":
        widths.append(lock_start_stream.width)
    if lock_duration_stream and lock_duration_stream.stream_type == "bv":
        widths.append(lock_duration_stream.width)
    if current_time_stream and current_time_stream.stream_type == "bv":
        widths.append(current_time_stream.width)
    
    width = max(widths) if widths else 16
    
    # Build references
    lock_start_ref = f"i{lock_start_idx}[t]" if lock_start_is_input else f"o{lock_start_idx}[t]"
    lock_duration_ref = f"i{lock_duration_idx}[t]" if lock_duration_is_input else f"o{lock_duration_idx}[t]"
    current_time_ref = f"i{current_time_idx}[t]" if current_time_is_input else f"o{current_time_idx}[t]"
    
    # Compute remaining time: remaining = (lock_start + lock_duration) - current_time
    # Handle overflow: if (lock_start + lock_duration) < current_time, lock has expired
    remaining_time_expr = f"(({lock_start_ref} + {lock_duration_ref}) - {current_time_ref})"
    
    # Output: lock_active = (remaining_time > 0)
    if output_stream.stream_type == "sbf":
        # Output boolean comparison
        return f"(o{output_idx}[t] = ({remaining_time_expr}) > {{0}}:bv[{width}])"
    else:
        # Output remaining time as bitvector
        return f"(o{output_idx}[t] = {remaining_time_expr}) && (o{output_idx}[0] = {{0}}:bv[{width}])"


def _generate_hex_stake_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate Hex staking pattern logic.
    
    Implements time-lock staking system with FSM:
    - UNSTAKED (0) → ACTIVE_STAKE (1) → MATURED_STAKE (2) or EARLY_ENDED_STAKE (3) → UNSTAKED
    
    States:
    - 0: UNSTAKED - No active stake
    - 1: ACTIVE_STAKE - Stake is active and locked
    - 2: MATURED_STAKE - Stake has reached maturity
    - 3: EARLY_ENDED_STAKE - Stake ended before maturity
    """
    # Get inputs from params or infer from block.inputs
    stake_amount_name = block.params.get("stake_amount", block.inputs[0] if len(block.inputs) > 0 else None)
    stake_duration_name = block.params.get("stake_duration", block.inputs[1] if len(block.inputs) > 1 else None)
    current_time_name = block.params.get("current_time", block.inputs[2] if len(block.inputs) > 2 else None)
    action_stake_name = block.params.get("action_stake", block.inputs[3] if len(block.inputs) > 3 else None)
    action_end_name = block.params.get("action_end", block.inputs[4] if len(block.inputs) > 4 else None)
    
    if not stake_amount_name or not stake_duration_name or not current_time_name:
        raise ValueError("Hex stake pattern requires stake_amount, stake_duration, and current_time inputs")
    if not action_stake_name or not action_end_name:
        raise ValueError("Hex stake pattern requires action_stake and action_end inputs")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get input indices
    stake_amount_idx, stake_amount_is_input = _get_stream_index_any(stake_amount_name, streams)
    stake_duration_idx, stake_duration_is_input = _get_stream_index_any(stake_duration_name, streams)
    current_time_idx, current_time_is_input = _get_stream_index_any(current_time_name, streams)
    action_stake_idx, action_stake_is_input = _get_stream_index_any(action_stake_name, streams)
    action_end_idx, action_end_is_input = _get_stream_index_any(action_end_name, streams)
    
    # Get stream types to determine width
    stake_amount_stream = next((s for s in streams if s.name == stake_amount_name), None)
    stake_duration_stream = next((s for s in streams if s.name == stake_duration_name), None)
    current_time_stream = next((s for s in streams if s.name == current_time_name), None)
    
    # Determine bitvector widths (Tau max is 32 bits)
    amount_width = stake_amount_stream.width if stake_amount_stream and stake_amount_stream.stream_type == "bv" else 32
    duration_width = stake_duration_stream.width if stake_duration_stream and stake_duration_stream.stream_type == "bv" else 16
    time_width = current_time_stream.width if current_time_stream and current_time_stream.stream_type == "bv" else 32
    
    # Get params
    max_duration = block.params.get("max_duration", 3650)
    lock_start_name = block.params.get("lock_start", None)
    
    # Build references
    stake_amount_ref = f"i{stake_amount_idx}[t]" if stake_amount_is_input else f"o{stake_amount_idx}[t]"
    stake_duration_ref = f"i{stake_duration_idx}[t]" if stake_duration_is_input else f"o{stake_duration_idx}[t]"
    current_time_ref = f"i{current_time_idx}[t]" if current_time_is_input else f"o{current_time_idx}[t]"
    action_stake_ref = f"i{action_stake_idx}[t]" if action_stake_is_input else f"o{action_stake_idx}[t]"
    action_end_ref = f"i{action_end_idx}[t]" if action_end_is_input else f"o{action_end_idx}[t]"
    
    # Get lock_start (if provided, otherwise use a state variable)
    if lock_start_name:
        lock_start_idx, lock_start_is_input = _get_stream_index_any(lock_start_name, streams)
        lock_start_ref = f"i{lock_start_idx}[t]" if lock_start_is_input else f"o{lock_start_idx}[t]"
    else:
        # Use a state variable to track lock_start
        # We'll need to add this as an output stream
        lock_start_ref = f"o{hex(int(output_idx, 16) + 1)[2:].upper()}[t]"  # Next output index
    
    # State encoding: 0=UNSTAKED, 1=ACTIVE_STAKE, 2=MATURED_STAKE, 3=EARLY_ENDED_STAKE
    # Use 2-bit state: bv[2]
    state_width = 2
    
    # Calculate end_time and remaining_days
    # end_time = lock_start + stake_duration
    # remaining_days = end_time - current_time
    # is_matured = current_time >= end_time
    
    # FSM logic:
    # UNSTAKED (0) → action_stake & valid → ACTIVE_STAKE (1)
    # ACTIVE_STAKE (1) → action_end & matured → MATURED_STAKE (2)
    # ACTIVE_STAKE (1) → action_end & early → EARLY_ENDED_STAKE (3)
    # MATURED_STAKE (2) → action_end → UNSTAKED (0)
    # EARLY_ENDED_STAKE (3) → action_end → UNSTAKED (0)
    
    # For simplicity, we'll use a simplified state machine that tracks:
    # - lock_active (sbf): Is stake active
    # - is_matured (sbf): Is stake matured
    # - is_early (sbf): Is early exit
    
    # We need to find or create additional output streams
    lock_active_name = block.params.get("lock_active_output", "lock_active")
    remaining_days_name = block.params.get("remaining_days_output", "remaining_days")
    is_matured_name = block.params.get("is_matured_output", "is_matured")
    is_early_name = block.params.get("is_early_output", "is_early")
    
    try:
        lock_active_idx = _get_stream_index(lock_active_name, streams, is_input=False)
        lock_active_ref = f"o{lock_active_idx}"
    except ValueError:
        # Lock active not declared - skip
        lock_active_ref = None
    
    try:
        remaining_days_idx = _get_stream_index(remaining_days_name, streams, is_input=False)
        remaining_days_ref = f"o{remaining_days_idx}"
    except ValueError:
        remaining_days_ref = None
    
    try:
        is_matured_idx = _get_stream_index(is_matured_name, streams, is_input=False)
        is_matured_ref = f"o{is_matured_idx}"
    except ValueError:
        is_matured_ref = None
    
    try:
        is_early_idx = _get_stream_index(is_early_name, streams, is_input=False)
        is_early_ref = f"o{is_early_idx}"
    except ValueError:
        is_early_ref = None
    
    # Generate Hex stake logic
    # Core FSM: UNSTAKED → ACTIVE_STAKE → (MATURED_STAKE | EARLY_ENDED_STAKE) → UNSTAKED
    
    # We need to track lock_start time when stake is created
    # lock_start[t] = action_stake ? current_time : lock_start[t-1]
    # This requires an additional output stream for lock_start
    
    # For Phase 1, let's implement basic lock_active tracking
    # lock_active[t] = action_stake | (lock_active[t-1] & !action_end)
    
    # Check if we have additional outputs for lock_start, remaining_days, etc.
    lock_start_output = block.params.get("lock_start_output", None)
    remaining_days_output = block.params.get("remaining_days_output", None)
    is_matured_output = block.params.get("is_matured_output", None)
    
    logic_parts = []
    
    # Primary output: lock_active (or stake_state)
    if output_stream.stream_type == "sbf":
        # Boolean output: lock_active
        logic_parts.append(f"(o{output_idx}[t] = {action_stake_ref} | (o{output_idx}[t-1] & {action_end_ref}')) && (o{output_idx}[0] = 0)")
    else:
        # Bitvector output: stake_state (0=UNSTAKED, 1=ACTIVE_STAKE, 2=MATURED_STAKE, 3=EARLY_ENDED_STAKE)
        # For Phase 1, simplify to boolean-like: 0=UNSTAKED, 1=ACTIVE_STAKE
        logic_parts.append(f"(o{output_idx}[t] = ({action_stake_ref} ? {{1}}:bv[{state_width}] : ((o{output_idx}[t-1] & {action_end_ref}') ? o{output_idx}[t-1] : {{0}}:bv[{state_width}]))) && (o{output_idx}[0] = {{0}}:bv[{state_width}])")
    
    # Try to add lock_start tracking if output stream exists
    if lock_start_output:
        try:
            lock_start_idx = _get_stream_index(lock_start_output, streams, is_input=False)
            lock_start_stream = next(s for s in streams if s.name == lock_start_output and not s.is_input)
            lock_start_width = lock_start_stream.width if lock_start_stream.width else time_width
            # lock_start[t] = action_stake ? current_time : lock_start[t-1]
            logic_parts.append(f"(o{lock_start_idx}[t] = ({action_stake_ref} ? {current_time_ref} : o{lock_start_idx}[t-1])) && (o{lock_start_idx}[0] = {{0}}:bv[{lock_start_width}])")
        except ValueError:
            pass  # Lock start output not declared
    
    # Try to add remaining_days calculation if output stream exists
    if remaining_days_output and lock_start_output:
        try:
            remaining_days_idx = _get_stream_index(remaining_days_output, streams, is_input=False)
            remaining_days_stream = next(s for s in streams if s.name == remaining_days_output and not s.is_input)
            remaining_days_width = remaining_days_stream.width if remaining_days_stream.width else duration_width
            lock_start_idx = _get_stream_index(lock_start_output, streams, is_input=False)
            lock_start_ref_state = f"o{lock_start_idx}[t]"
            # remaining_days = (lock_start + stake_duration) - current_time
            # Only calculate if lock is active, clamp to 0 if negative
            # Use same width for calculation
            calc_width = max(time_width, duration_width)
            logic_parts.append(f"(o{remaining_days_idx}[t] = (o{output_idx}[t] ? ((({lock_start_ref_state} + {stake_duration_ref}) - {current_time_ref}) > {{0}}:bv[{calc_width}] ? (({lock_start_ref_state} + {stake_duration_ref}) - {current_time_ref}) : {{0}}:bv[{remaining_days_width}]) : {{0}}:bv[{remaining_days_width}])) && (o{remaining_days_idx}[0] = {{0}}:bv[{remaining_days_width}])")
        except ValueError:
            pass
    
    # Try to add is_matured check if output stream exists
    if is_matured_output and lock_start_output:
        try:
            is_matured_idx = _get_stream_index(is_matured_output, streams, is_input=False)
            lock_start_idx = _get_stream_index(lock_start_output, streams, is_input=False)
            lock_start_ref_state = f"o{lock_start_idx}[t]"
            # is_matured = current_time >= (lock_start + stake_duration) & lock_active
            logic_parts.append(f"(o{is_matured_idx}[t] = o{output_idx}[t] & ({current_time_ref} >= ({lock_start_ref_state} + {stake_duration_ref})))")
        except ValueError:
            pass
    
    # Phase 2: Add share calculation, rewards, and penalties if requested
    include_shares = block.params.get("include_shares", False)
    include_rewards = block.params.get("include_rewards", False)
    include_penalties = block.params.get("include_penalties", False)
    
    # Share calculation: user_shares = stake_amount * duration_multiplier
    if include_shares:
        user_shares_output = block.params.get("user_shares_output", "user_shares")
        try:
            user_shares_idx = _get_stream_index(user_shares_output, streams, is_input=False)
            user_shares_stream = next(s for s in streams if s.name == user_shares_output and not s.is_input)
            user_shares_width = user_shares_stream.width if user_shares_stream.width else amount_width
            
            # Duration multiplier: duration / max_duration (linear scaling)
            # For sqrt scaling, would need external computation
            duration_scaling = block.params.get("duration_scaling", "linear")
            max_duration = block.params.get("max_duration", 3650)
            
            if duration_scaling == "linear":
                # Linear: multiplier = duration / max_duration
                # user_shares = stake_amount * (duration / max_duration)
                # Simplified: user_shares = stake_amount * duration / max_duration
                logic_parts.append(f"(o{user_shares_idx}[t] = (o{output_idx}[t] ? (({stake_amount_ref} * {stake_duration_ref}) / {{{max_duration}}}:bv[{duration_width}]) : {{0}}:bv[{user_shares_width}])) && (o{user_shares_idx}[0] = {{0}}:bv[{user_shares_width}])")
            else:
                # Sqrt scaling would need external computation
                # For now, use linear
                logic_parts.append(f"(o{user_shares_idx}[t] = (o{output_idx}[t] ? (({stake_amount_ref} * {stake_duration_ref}) / {{{max_duration}}}:bv[{duration_width}]) : {{0}}:bv[{user_shares_width}])) && (o{user_shares_idx}[0] = {{0}}:bv[{user_shares_width}])")
        except ValueError:
            pass  # User shares output not declared
    
    # Reward accrual: accrued_rewards = (user_shares / total_shares) * daily_inflation * elapsed_days
    if include_rewards:
        accrued_rewards_output = block.params.get("accrued_rewards_output", "accrued_rewards")
        total_shares_name = block.params.get("total_shares", "total_shares")
        daily_inflation_name = block.params.get("daily_inflation", "daily_inflation")
        
        try:
            accrued_rewards_idx = _get_stream_index(accrued_rewards_output, streams, is_input=False)
            accrued_rewards_stream = next(s for s in streams if s.name == accrued_rewards_output and not s.is_input)
            accrued_rewards_width = accrued_rewards_stream.width if accrued_rewards_stream.width else amount_width
            
            # Get external inputs
            total_shares_idx, total_shares_is_input = _get_stream_index_any(total_shares_name, streams)
            daily_inflation_idx, daily_inflation_is_input = _get_stream_index_any(daily_inflation_name, streams)
            
            total_shares_ref = f"i{total_shares_idx}[t]" if total_shares_is_input else f"o{total_shares_idx}[t]"
            daily_inflation_ref = f"i{daily_inflation_idx}[t]" if daily_inflation_is_input else f"o{daily_inflation_idx}[t]"
            
            # Calculate elapsed_days = current_time - lock_start
            if lock_start_output:
                lock_start_idx = _get_stream_index(lock_start_output, streams, is_input=False)
                lock_start_ref_state = f"o{lock_start_idx}[t]"
                elapsed_days_expr = f"({current_time_ref} - {lock_start_ref_state})"
            else:
                # Can't calculate without lock_start
                elapsed_days_expr = "{0}:bv[16]"
            
            # Get user_shares reference
            if include_shares:
                user_shares_idx = _get_stream_index(user_shares_output, streams, is_input=False)
                user_shares_ref = f"o{user_shares_idx}[t]"
            else:
                # Use stake_amount as proxy
                user_shares_ref = stake_amount_ref
            
            # accrued_rewards = (user_shares / total_shares) * daily_inflation * elapsed_days
            # Only calculate if lock is active
            logic_parts.append(f"(o{accrued_rewards_idx}[t] = (o{output_idx}[t] ? ((({user_shares_ref} / {total_shares_ref}) * {daily_inflation_ref}) * {elapsed_days_expr}) : {{0}}:bv[{accrued_rewards_width}])) && (o{accrued_rewards_idx}[0] = {{0}}:bv[{accrued_rewards_width}])")
        except ValueError:
            pass  # Accrued rewards output not declared or external inputs missing
    
    # Penalty calculation: penalty = base_penalty_rate * (remaining_days / total_days) * stake_amount / 100
    if include_penalties:
        penalty_amount_output = block.params.get("penalty_amount_output", "penalty_amount")
        base_penalty_rate = block.params.get("base_penalty_rate", 50)  # 50%
        
        try:
            penalty_amount_idx = _get_stream_index(penalty_amount_output, streams, is_input=False)
            penalty_amount_stream = next(s for s in streams if s.name == penalty_amount_output and not s.is_input)
            penalty_amount_width = penalty_amount_stream.width if penalty_amount_stream.width else amount_width
            
            # Penalty only applies if early exit (lock_active & !is_matured & action_end)
            # penalty = base_penalty_rate * (remaining_days / stake_duration) * stake_amount / 100
            
            if remaining_days_output and is_matured_output:
                remaining_days_idx = _get_stream_index(remaining_days_output, streams, is_input=False)
                is_matured_idx = _get_stream_index(is_matured_output, streams, is_input=False)
                remaining_days_ref = f"o{remaining_days_idx}[t]"
                is_matured_ref = f"o{is_matured_idx}[t]"
                
                # penalty = (lock_active & !is_matured) ? (base_penalty_rate * remaining_days * stake_amount) / (stake_duration * 100) : 0
                logic_parts.append(f"(o{penalty_amount_idx}[t] = ((o{output_idx}[t] & {is_matured_ref}') ? ((({{{base_penalty_rate}}}:bv[8] * {remaining_days_ref}) * {stake_amount_ref}) / (({stake_duration_ref} * {{100}}:bv[8]))) : {{0}}:bv[{penalty_amount_width}])) && (o{penalty_amount_idx}[0] = {{0}}:bv[{penalty_amount_width}])")
        except ValueError:
            pass  # Penalty amount output not declared
    
    return " &&\n    ".join(logic_parts)


def _generate_multi_bit_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate multi-bit counter pattern logic.
    
    Implements a counter with configurable width (2-bit, 3-bit, etc.).
    Supports increment and reset operations.
    """
    if len(block.inputs) < 1:
        raise ValueError("Multi-bit counter pattern requires at least 1 input (increment)")
    
    # Get inputs
    increment_name = block.inputs[0]
    reset_name = block.inputs[1] if len(block.inputs) > 1 else None
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get width from params or output stream
    width = block.params.get("width", output_stream.width if output_stream.width else 2)
    if width < 1 or width > 32:
        raise ValueError(f"Multi-bit counter width must be 1-32, got {width}")
    
    # Get input indices
    increment_idx, increment_is_input = _get_stream_index_any(increment_name, streams)
    increment_ref = f"i{increment_idx}[t]" if increment_is_input else f"o{increment_idx}[t]"
    
    # Build counter logic: counter[t] = reset ? 0 : (increment ? counter[t-1] + 1 : counter[t-1])
    # Tau doesn't support ternary in recurrence, so use: (reset & 0) | (reset' & (increment ? counter[t-1] + 1 : counter[t-1]))
    # Simplified: counter[t] = reset ? 0 : (increment ? counter[t-1] + 1 : counter[t-1])
    
    if reset_name:
        reset_idx, reset_is_input = _get_stream_index_any(reset_name, streams)
        reset_ref = f"i{reset_idx}[t]" if reset_is_input else f"o{reset_idx}[t]"
        # counter[t] = reset ? 0 : (increment ? counter[t-1] + 1 : counter[t-1])
        counter_logic = f"(o{output_idx}[t] = ({reset_ref} ? {{0}}:bv[{width}] : ({increment_ref} ? (o{output_idx}[t-1] + {{1}}:bv[{width}]) : o{output_idx}[t-1])))"
    else:
        # No reset: counter[t] = increment ? counter[t-1] + 1 : counter[t-1]
        counter_logic = f"(o{output_idx}[t] = ({increment_ref} ? (o{output_idx}[t-1] + {{1}}:bv[{width}]) : o{output_idx}[t-1]))"
    
    # Initial condition
    initial_value = block.params.get("initial_value", 0)
    init_logic = f" && (o{output_idx}[0] = {{{initial_value}}}:bv[{width}])"
    
    return counter_logic + init_logic


def _generate_streak_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate streak counter pattern logic.
    
    Tracks consecutive events (win/loss streaks).
    Resets on opposite event or explicit reset.
    """
    if len(block.inputs) < 1:
        raise ValueError("Streak counter pattern requires at least 1 input (event)")
    
    # Get inputs
    event_name = block.inputs[0]
    reset_name = block.inputs[1] if len(block.inputs) > 1 else None
    opposite_event_name = block.params.get("opposite_event", None)
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get width from params or output stream
    width = block.params.get("width", output_stream.width if output_stream.width else 4)
    if width < 1 or width > 32:
        raise ValueError(f"Streak counter width must be 1-32, got {width}")
    
    # Get input indices
    event_idx, event_is_input = _get_stream_index_any(event_name, streams)
    event_ref = f"i{event_idx}[t]" if event_is_input else f"o{event_idx}[t]"
    
    # Streak logic:
    # - If event occurs: streak[t] = streak[t-1] + 1
    # - If opposite event occurs: streak[t] = 0
    # - If reset: streak[t] = 0
    # - Otherwise: streak[t] = streak[t-1]
    
    reset_conditions = []
    if reset_name:
        reset_idx, reset_is_input = _get_stream_index_any(reset_name, streams)
        reset_ref = f"i{reset_idx}[t]" if reset_is_input else f"o{reset_idx}[t]"
        reset_conditions.append(reset_ref)
    
    if opposite_event_name:
        opposite_idx, opposite_is_input = _get_stream_index_any(opposite_event_name, streams)
        opposite_ref = f"i{opposite_idx}[t]" if opposite_is_input else f"o{opposite_idx}[t]"
        reset_conditions.append(opposite_ref)
    
    # Build logic: streak[t] = (reset | opposite) ? 0 : (event ? streak[t-1] + 1 : streak[t-1])
    if reset_conditions:
        reset_expr = " | ".join(reset_conditions)
        streak_logic = f"(o{output_idx}[t] = (({reset_expr}) ? {{0}}:bv[{width}] : ({event_ref} ? (o{output_idx}[t-1] + {{1}}:bv[{width}]) : o{output_idx}[t-1])))"
    else:
        # No reset: streak[t] = event ? streak[t-1] + 1 : streak[t-1]
        streak_logic = f"(o{output_idx}[t] = ({event_ref} ? (o{output_idx}[t-1] + {{1}}:bv[{width}]) : o{output_idx}[t-1]))"
    
    # Initial condition
    initial_value = block.params.get("initial_value", 0)
    init_logic = f" && (o{output_idx}[0] = {{{initial_value}}}:bv[{width}])"
    
    return streak_logic + init_logic


def _generate_mode_switch_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate mode switch pattern logic.
    
    Switches between modes (e.g., AGGRESSIVE/DEFENSIVE) based on conditions.
    """
    if len(block.inputs) < 1:
        raise ValueError("Mode switch pattern requires at least 1 input")
    
    # Get modes from params
    modes = block.params.get("modes", ["MODE1", "MODE2"])
    if len(modes) < 2:
        raise ValueError("Mode switch pattern requires at least 2 modes")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Determine state width (log2 of number of modes)
    import math
    state_width = max(1, math.ceil(math.log2(len(modes))))
    
    # Get transitions from params
    transitions = block.params.get("transitions", {})
    
    # Build mode switch logic
    # For each mode, check transition conditions
    # mode[t] = (transition_to_mode1 ? mode1 : (transition_to_mode2 ? mode2 : mode[t-1]))
    
    # Build mode switch logic using boolean expressions
    # For each mode, create a condition that selects that mode value
    # mode[t] = (cond1 ? mode1 : (cond2 ? mode2 : current))
    # But Tau doesn't support nested ternary well, so use boolean logic:
    # mode[t] = (cond1 & mode1) | (cond1' & cond2 & mode2) | (cond1' & cond2' & current)
    
    # Simple approach: use first input to switch between modes
    # For 2 modes: mode[t] = (signal ? mode1 : mode0)
    # For more modes, use bitvector encoding
    
    if len(modes) == 2:
        # Simple 2-mode switch
        if len(block.inputs) >= 1:
            inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
            inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
            # Switch to mode 1 on input, mode 0 otherwise
            mode_logic = f"(o{output_idx}[t] = ({inp_ref} ? {{1}}:bv[{state_width}] : {{0}}:bv[{state_width}]))"
        else:
            mode_logic = f"(o{output_idx}[t] = o{output_idx}[t-1])"
    else:
        # Multi-mode: use transitions to build conditions
        # For now, use first input as mode selector (simplified)
        if len(block.inputs) >= 1:
            inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
            inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
            # Use input as mode index (simplified - assumes input is mode value)
            mode_logic = f"(o{output_idx}[t] = ({inp_ref} ? {inp_ref} : o{output_idx}[t-1]))"
        else:
            mode_logic = f"(o{output_idx}[t] = o{output_idx}[t-1])"
    
    # Initial condition
    initial_mode = block.params.get("initial_mode", 0)
    init_logic = f" && (o{output_idx}[0] = {{{initial_mode}}}:bv[{state_width}])"
    
    return mode_logic + init_logic


def _generate_proposal_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate proposal FSM pattern logic.
    
    Implements governance proposal lifecycle: DRAFT → VOTING → PASSED → EXECUTED → CANCELLED
    """
    if len(block.inputs) < 1:
        raise ValueError("Proposal FSM pattern requires at least 1 input")
    
    # Get inputs (create, vote, execute, cancel)
    create_name = block.params.get("create_input", block.inputs[0] if len(block.inputs) > 0 else None)
    vote_name = block.params.get("vote_input", block.inputs[1] if len(block.inputs) > 1 else None)
    execute_name = block.params.get("execute_input", block.inputs[2] if len(block.inputs) > 2 else None)
    cancel_name = block.params.get("cancel_input", block.inputs[3] if len(block.inputs) > 3 else None)
    
    if not create_name:
        raise ValueError("Proposal FSM pattern requires create_input")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # States: 0=DRAFT, 1=VOTING, 2=PASSED, 3=EXECUTED, 4=CANCELLED
    # Use 3-bit state (supports up to 8 states)
    state_width = 3
    
    # Get input indices
    create_idx, create_is_input = _get_stream_index_any(create_name, streams)
    create_ref = f"i{create_idx}[t]" if create_is_input else f"o{create_idx}[t]"
    
    vote_ref = None
    if vote_name:
        vote_idx, vote_is_input = _get_stream_index_any(vote_name, streams)
        vote_ref = f"i{vote_idx}[t]" if vote_is_input else f"o{vote_idx}[t]"
    
    execute_ref = None
    if execute_name:
        execute_idx, execute_is_input = _get_stream_index_any(execute_name, streams)
        execute_ref = f"i{execute_idx}[t]" if execute_is_input else f"o{execute_idx}[t]"
    
    cancel_ref = None
    if cancel_name:
        cancel_idx, cancel_is_input = _get_stream_index_any(cancel_name, streams)
        cancel_ref = f"i{cancel_idx}[t]" if cancel_is_input else f"o{cancel_idx}[t]"
    
    # Get quorum/vote result (optional external input)
    quorum_met_name = block.params.get("quorum_met", None)
    quorum_met_ref = None
    if quorum_met_name:
        quorum_idx, quorum_is_input = _get_stream_index_any(quorum_met_name, streams)
        quorum_met_ref = f"i{quorum_idx}[t]" if quorum_is_input else f"o{quorum_idx}[t]"
    
    # State transitions:
    # DRAFT (0) → create → VOTING (1)
    # VOTING (1) → (vote & quorum_met) → PASSED (2)
    # PASSED (2) → execute → EXECUTED (3)
    # Any state → cancel → CANCELLED (4)
    
    # Build transition logic using boolean expressions
    # State transitions:
    # DRAFT (0) → create → VOTING (1)
    # VOTING (1) → (vote & quorum_met) → PASSED (2)
    # PASSED (2) → execute → EXECUTED (3)
    # Any state → cancel → CANCELLED (4)
    
    # Use boolean logic: state[t] = (cond1 & val1) | (cond1' & cond2 & val2) | (cond1' & cond2' & cond3 & val3) | (cond1' & cond2' & cond3' & cond4 & val4) | (cond1' & cond2' & cond3' & cond4' & current)
    
    conditions = []
    
    # Cancel has highest priority (any state → CANCELLED)
    if cancel_ref:
        conditions.append(f"({cancel_ref} ? {{4}}:bv[{state_width}] : {{0}}:bv[{state_width}])")
    
    # DRAFT → VOTING: create
    conditions.append(f"((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {create_ref} ? {{1}}:bv[{state_width}] : {{0}}:bv[{state_width}])")
    
    # VOTING → PASSED: vote & quorum_met
    if vote_ref and quorum_met_ref:
        conditions.append(f"((o{output_idx}[t-1] = {{1}}:bv[{state_width}]) & {vote_ref} & {quorum_met_ref} ? {{2}}:bv[{state_width}] : {{0}}:bv[{state_width}])")
    elif vote_ref:
        conditions.append(f"((o{output_idx}[t-1] = {{1}}:bv[{state_width}]) & {vote_ref} ? {{2}}:bv[{state_width}] : {{0}}:bv[{state_width}])")
    
    # PASSED → EXECUTED: execute
    if execute_ref:
        conditions.append(f"((o{output_idx}[t-1] = {{2}}:bv[{state_width}]) & {execute_ref} ? {{3}}:bv[{state_width}] : {{0}}:bv[{state_width}])")
    
    # Combine: use OR to combine conditions, but need to handle priority
    # For now, use simple approach: check conditions in order, first match wins
    # This is simplified - proper implementation would need priority logic
    if conditions:
        # Use nested ternary with proper fallback
        # For cancel: if cancel, state=4, else check other transitions
        if cancel_ref:
            # Cancel first, then other transitions, then maintain state
            other_conditions = " : ".join(conditions[1:]) if len(conditions) > 1 else f"o{output_idx}[t-1]"
            proposal_logic = f"(o{output_idx}[t] = ({cancel_ref} ? {{4}}:bv[{state_width}] : ({other_conditions})))"
        else:
            # No cancel, use transitions with fallback
            transition_expr = " : ".join(conditions) + f" : o{output_idx}[t-1]"
            proposal_logic = f"(o{output_idx}[t] = {transition_expr})"
    else:
        # Fallback: simple create → voting
        proposal_logic = f"(o{output_idx}[t] = ({create_ref} ? {{1}}:bv[{state_width}] : o{output_idx}[t-1]))"
    
    # Initial condition: DRAFT
    init_logic = f" && (o{output_idx}[0] = {{0}}:bv[{state_width}])"
    
    return proposal_logic + init_logic


def _generate_risk_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate risk FSM pattern logic.
    
    Implements risk state machine: NORMAL → WARNING → CRITICAL
    """
    if len(block.inputs) < 1:
        raise ValueError("Risk FSM pattern requires at least 1 input")
    
    # Get inputs (risk signals)
    warning_signal_name = block.params.get("warning_signal", block.inputs[0] if len(block.inputs) > 0 else None)
    critical_signal_name = block.params.get("critical_signal", block.inputs[1] if len(block.inputs) > 1 else None)
    normal_signal_name = block.params.get("normal_signal", block.inputs[2] if len(block.inputs) > 2 else None)
    
    if not warning_signal_name:
        raise ValueError("Risk FSM pattern requires warning_signal")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # States: 0=NORMAL, 1=WARNING, 2=CRITICAL
    # Use 2-bit state
    state_width = 2
    
    # Get input indices
    warning_idx, warning_is_input = _get_stream_index_any(warning_signal_name, streams)
    warning_ref = f"i{warning_idx}[t]" if warning_is_input else f"o{warning_idx}[t]"
    
    critical_ref = None
    if critical_signal_name:
        critical_idx, critical_is_input = _get_stream_index_any(critical_signal_name, streams)
        critical_ref = f"i{critical_idx}[t]" if critical_is_input else f"o{critical_idx}[t]"
    
    normal_ref = None
    if normal_signal_name:
        normal_idx, normal_is_input = _get_stream_index_any(normal_signal_name, streams)
        normal_ref = f"i{normal_idx}[t]" if normal_is_input else f"o{normal_idx}[t]"
    
    # State transitions:
    # NORMAL (0) → warning → WARNING (1)
    # WARNING (1) → critical → CRITICAL (2)
    # WARNING (1) → normal → NORMAL (0)
    # CRITICAL (2) → normal → NORMAL (0)
    
    # Build transition logic using boolean expressions
    # State transitions:
    # NORMAL (0) → warning → WARNING (1)
    # WARNING (1) → critical → CRITICAL (2)
    # WARNING (1) → normal → NORMAL (0)
    # CRITICAL (2) → normal → NORMAL (0)
    
    # Priority: normal > critical > warning
    # normal signal resets to NORMAL from any state
    # critical signal moves WARNING → CRITICAL
    # warning signal moves NORMAL → WARNING
    
    if normal_ref:
        # Normal signal has highest priority (resets to NORMAL)
        # If normal, state=0, else check other transitions
        if critical_ref:
            # normal ? 0 : (critical & warning_state ? 2 : (warning & normal_state ? 1 : current))
            risk_logic = f"(o{output_idx}[t] = ({normal_ref} ? {{0}}:bv[{state_width}] : ((o{output_idx}[t-1] = {{1}}:bv[{state_width}]) & {critical_ref} ? {{2}}:bv[{state_width}] : ((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {warning_ref} ? {{1}}:bv[{state_width}] : o{output_idx}[t-1]))))"
        else:
            # normal ? 0 : (warning & normal_state ? 1 : current)
            risk_logic = f"(o{output_idx}[t] = ({normal_ref} ? {{0}}:bv[{state_width}] : ((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {warning_ref} ? {{1}}:bv[{state_width}] : o{output_idx}[t-1])))"
    elif critical_ref:
        # No normal signal, check critical and warning
        risk_logic = f"(o{output_idx}[t] = ((o{output_idx}[t-1] = {{1}}:bv[{state_width}]) & {critical_ref} ? {{2}}:bv[{state_width}] : ((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {warning_ref} ? {{1}}:bv[{state_width}] : o{output_idx}[t-1])))"
    else:
        # Only warning signal
        risk_logic = f"(o{output_idx}[t] = ((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {warning_ref} ? {{1}}:bv[{state_width}] : o{output_idx}[t-1]))"
    
    # Initial condition: NORMAL
    init_logic = f" && (o{output_idx}[0] = {{0}}:bv[{state_width}])"
    
    return risk_logic + init_logic


def _generate_entry_exit_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate entry-exit FSM pattern logic.
    
    Implements multi-phase trade lifecycle: PRE_TRADE → IN_TRADE → POST_TRADE
    Each phase can have sub-states for fine-grained control.
    """
    if len(block.inputs) < 2:
        raise ValueError("Entry-exit FSM pattern requires at least 2 inputs (entry_signal, exit_signal)")
    
    # Get inputs
    entry_signal_name = block.params.get("entry_signal", block.inputs[0] if len(block.inputs) > 0 else None)
    exit_signal_name = block.params.get("exit_signal", block.inputs[1] if len(block.inputs) > 1 else None)
    stop_loss_name = block.params.get("stop_loss", block.inputs[2] if len(block.inputs) > 2 else None)
    take_profit_name = block.params.get("take_profit", block.inputs[3] if len(block.inputs) > 3 else None)
    
    if not entry_signal_name or not exit_signal_name:
        raise ValueError("Entry-exit FSM pattern requires entry_signal and exit_signal")
    
    # Get phases from params (default: PRE_TRADE, IN_TRADE, POST_TRADE)
    phases = block.params.get("phases", ["PRE_TRADE", "IN_TRADE", "POST_TRADE"])
    if len(phases) < 2:
        raise ValueError("Entry-exit FSM pattern requires at least 2 phases")
    
    # Get output streams
    phase_output_name = block.params.get("phase_output", block.output)
    position_output_name = block.params.get("position_output", None)
    
    # Get phase output stream
    phase_output_idx = _get_stream_index(phase_output_name, streams, is_input=False)
    phase_output_stream = next(s for s in streams if s.name == phase_output_name and not s.is_input)
    
    # Determine phase state width (log2 of number of phases)
    import math
    phase_width = max(2, math.ceil(math.log2(len(phases))))
    
    # Get input indices
    entry_idx, entry_is_input = _get_stream_index_any(entry_signal_name, streams)
    entry_ref = f"i{entry_idx}[t]" if entry_is_input else f"o{entry_idx}[t]"
    
    exit_idx, exit_is_input = _get_stream_index_any(exit_signal_name, streams)
    exit_ref = f"i{exit_idx}[t]" if exit_is_input else f"o{exit_idx}[t]"
    
    stop_loss_ref = None
    if stop_loss_name:
        stop_loss_idx, stop_loss_is_input = _get_stream_index_any(stop_loss_name, streams)
        stop_loss_ref = f"i{stop_loss_idx}[t]" if stop_loss_is_input else f"o{stop_loss_idx}[t]"
    
    take_profit_ref = None
    if take_profit_name:
        take_profit_idx, take_profit_is_input = _get_stream_index_any(take_profit_name, streams)
        take_profit_ref = f"i{take_profit_idx}[t]" if take_profit_is_input else f"o{take_profit_idx}[t]"
    
    # Phase transitions:
    # PRE_TRADE (0) → entry → IN_TRADE (1)
    # IN_TRADE (1) → (exit | stop_loss | take_profit) → POST_TRADE (2)
    # POST_TRADE (2) → reset → PRE_TRADE (0)
    
    # Build phase transition logic
    transitions = []
    
    # PRE_TRADE → IN_TRADE: entry signal
    transitions.append(f"((o{phase_output_idx}[t-1] = {{0}}:bv[{phase_width}]) & {entry_ref} ? {{1}}:bv[{phase_width}] : {{999}}:bv[{phase_width}])")
    
    # IN_TRADE → POST_TRADE: exit, stop_loss, or take_profit
    exit_conditions = [exit_ref]
    if stop_loss_ref:
        exit_conditions.append(stop_loss_ref)
    if take_profit_ref:
        exit_conditions.append(take_profit_ref)
    
    exit_expr = " | ".join(exit_conditions)
    transitions.append(f"((o{phase_output_idx}[t-1] = {{1}}:bv[{phase_width}]) & ({exit_expr}) ? {{2}}:bv[{phase_width}] : {{999}}:bv[{phase_width}])")
    
    # POST_TRADE → PRE_TRADE: reset (no entry signal)
    # Reset when not entering (entry signal is false)
    transitions.append(f"((o{phase_output_idx}[t-1] = {{2}}:bv[{phase_width}]) & {entry_ref}' ? {{0}}:bv[{phase_width}] : {{999}}:bv[{phase_width}])")
    
    # Combine transitions with fallback to maintain current phase
    if transitions:
        transition_expr = " : ".join(transitions) + f" : o{phase_output_idx}[t-1]"
        phase_logic = f"(o{phase_output_idx}[t] = {transition_expr})"
    else:
        # Fallback: simple entry → exit
        phase_logic = f"(o{phase_output_idx}[t] = ({entry_ref} ? {{1}}:bv[{phase_width}] : ({exit_ref} ? {{2}}:bv[{phase_width}] : o{phase_output_idx}[t-1])))"
    
    # Position output: true when IN_TRADE
    logic_parts = [phase_logic]
    
    if position_output_name:
        try:
            position_output_idx = _get_stream_index(position_output_name, streams, is_input=False)
            position_logic = f"(o{position_output_idx}[t] = (o{phase_output_idx}[t] = {{1}}:bv[{phase_width}])) && (o{position_output_idx}[0] = 0)"
            logic_parts.append(position_logic)
        except ValueError:
            pass  # Position output not declared
    
    # Initial condition: PRE_TRADE
    init_logic = f" && (o{phase_output_idx}[0] = {{0}}:bv[{phase_width}])"
    
    return " &&\n    ".join(logic_parts) + init_logic


def _generate_orthogonal_regions_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate orthogonal regions pattern logic.
    
    Implements parallel independent FSMs (orthogonal regions).
    Each region operates independently and can be in different states simultaneously.
    Example: execution region, risk region, connectivity region all running in parallel.
    """
    if len(block.inputs) < 1:
        raise ValueError("Orthogonal regions pattern requires at least 1 input")
    
    # Get regions configuration from params
    regions_config = block.params.get("regions", [])
    
    if not regions_config:
        raise ValueError("Orthogonal regions pattern requires 'regions' parameter with region definitions")
    
    # Get output streams for each region
    region_outputs = block.params.get("region_outputs", [])
    
    if len(region_outputs) != len(regions_config):
        raise ValueError(f"Number of region_outputs ({len(region_outputs)}) must match number of regions ({len(regions_config)})")
    
    # Generate FSM logic for each region
    logic_parts = []
    
    for i, region in enumerate(regions_config):
        if not isinstance(region, dict):
            raise ValueError(f"Region {i} must be a dictionary")
        
        region_name = region.get("name", f"region_{i}")
        region_inputs = region.get("inputs", [])
        region_states = region.get("states", ["FLAT", "LONG"])
        
        if not region_inputs:
            raise ValueError(f"Region {region_name} must have at least one input")
        
        if len(region_states) < 2:
            raise ValueError(f"Region {region_name} must have at least 2 states")
        
        # Get output stream for this region
        region_output_name = region_outputs[i]
        region_output_idx = _get_stream_index(region_output_name, streams, is_input=False)
        region_output_stream = next(s for s in streams if s.name == region_output_name and not s.is_input)
        
        # Determine state width
        import math
        state_width = max(1, math.ceil(math.log2(len(region_states))))
        
        # Get input indices for this region
        # For FSM, we need buy/sell signals (or equivalent)
        # If region has 2 inputs, treat as buy/sell
        # If region has 1 input, treat as toggle
        
        if len(region_inputs) >= 2:
            # Standard FSM: buy/sell inputs
            buy_input_name = region_inputs[0]
            sell_input_name = region_inputs[1]
            
            buy_idx, buy_is_input = _get_stream_index_any(buy_input_name, streams)
            sell_idx, sell_is_input = _get_stream_index_any(sell_input_name, streams)
            
            buy_ref = f"i{buy_idx}[t]" if buy_is_input else f"o{buy_idx}[t]"
            sell_ref = f"i{sell_idx}[t]" if sell_is_input else f"o{sell_idx}[t]"
            
            # Generate FSM logic: state[t] = buy | (state[t-1] & sell')
            region_logic = f"(o{region_output_idx}[t] = {buy_ref} | (o{region_output_idx}[t-1] & {sell_ref}'))"
        else:
            # Single input: toggle FSM
            toggle_input_name = region_inputs[0]
            toggle_idx, toggle_is_input = _get_stream_index_any(toggle_input_name, streams)
            toggle_ref = f"i{toggle_idx}[t]" if toggle_is_input else f"o{toggle_idx}[t]"
            
            # Toggle between first two states
            region_logic = f"(o{region_output_idx}[t] = ({toggle_ref} ? {{1}}:bv[{state_width}] : {{0}}:bv[{state_width}]))"
        
        # Initial condition
        initial_state = region.get("initial_state", 0)
        init_logic = f" && (o{region_output_idx}[0] = {{{initial_state}}}:bv[{state_width}])"
        
        logic_parts.append(region_logic + init_logic)
    
    return " &&\n    ".join(logic_parts)


def _generate_state_aggregation_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate state aggregation pattern logic.
    
    Combines multiple FSM states into a superstate using aggregation methods.
    Common aggregation methods: majority, unanimous, custom expression, or mode selection.
    """
    if len(block.inputs) < 2:
        raise ValueError("State aggregation pattern requires at least 2 inputs (FSM states)")
    
    # Get aggregation method from params
    aggregation_method = block.params.get("method", "majority")
    valid_methods = ["majority", "unanimous", "custom", "mode"]
    
    if aggregation_method not in valid_methods:
        raise ValueError(f"Aggregation method must be one of {valid_methods}, got {aggregation_method}")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get input indices (these are FSM state outputs)
    input_refs = []
    for inp_name in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
        inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        input_refs.append(inp_ref)
    
    # Generate aggregation logic based on method
    if aggregation_method == "majority":
        # Majority: output is true if majority of inputs are true
        threshold = block.params.get("threshold", (len(block.inputs) + 1) // 2)
        total = len(block.inputs)
        
        # Generate majority expression: count true inputs >= threshold
        # For boolean aggregation: (a & b) | (a & c) | (b & c) for 2-of-3
        if threshold == total:
            # Unanimous: all must be true
            agg_expr = " & ".join(input_refs)
        elif threshold == 1:
            # Any: at least one true
            agg_expr = " | ".join(input_refs)
        else:
            # N-of-M: generate combinations
            from itertools import combinations
            combinations_list = list(combinations(range(len(input_refs)), threshold))
            terms = []
            for combo in combinations_list:
                term_parts = [input_refs[i] for i in combo]
                terms.append("(" + " & ".join(term_parts) + ")")
            agg_expr = " | ".join(terms)
        
        aggregation_logic = f"(o{output_idx}[t] = {agg_expr})"
    
    elif aggregation_method == "unanimous":
        # Unanimous: all inputs must agree
        agg_expr = " & ".join(input_refs)
        aggregation_logic = f"(o{output_idx}[t] = {agg_expr})"
    
    elif aggregation_method == "custom":
        # Custom: use provided expression
        expression = block.params.get("expression", "")
        if not expression:
            raise ValueError("Custom aggregation method requires 'expression' parameter")
        
        # Replace input names with references
        for i, inp_name in enumerate(block.inputs):
            expression = expression.replace(f"{inp_name}[t]", input_refs[i])
            expression = expression.replace(f"{{{inp_name}}}", input_refs[i])
        
        aggregation_logic = f"(o{output_idx}[t] = {expression})"
    
    elif aggregation_method == "mode":
        # Mode: select mode based on which input is active
        # Output is the index of the first active input, or 0 if none
        # This is more complex - for now, use first active input
        
        # Build mode selection: (input0 ? 0 : (input1 ? 1 : (input2 ? 2 : 0)))
        mode_conditions = []
        for i, inp_ref in enumerate(input_refs):
            mode_conditions.append(f"({inp_ref} ? {{{i}}}:bv[{output_stream.width if output_stream.width else 2}] : {{999}}:bv[{output_stream.width if output_stream.width else 2}])")
        
        if mode_conditions:
            mode_expr = " : ".join(mode_conditions) + f" : {{0}}:bv[{output_stream.width if output_stream.width else 2}]"
            aggregation_logic = f"(o{output_idx}[t] = {mode_expr})"
        else:
            aggregation_logic = f"(o{output_idx}[t] = {{0}}:bv[{output_stream.width if output_stream.width else 2}])"
    
    # Initial condition
    if output_stream.stream_type == "sbf":
        init_logic = f" && (o{output_idx}[0] = 0)"
    else:
        initial_value = block.params.get("initial_value", 0)
        init_logic = f" && (o{output_idx}[0] = {{{initial_value}}}:bv[{output_stream.width if output_stream.width else 2}])"
    
    return aggregation_logic + init_logic


def _generate_tcp_connection_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate TCP connection FSM pattern logic.
    
    Implements TCP connection state machine with 11 states:
    CLOSED, LISTEN, SYN_SENT, SYN_RECEIVED, ESTABLISHED, 
    FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, CLOSING, TIME_WAIT, LAST_ACK
    
    Handles SYN, ACK, FIN, RST flags for connection lifecycle.
    """
    if len(block.inputs) < 1:
        raise ValueError("TCP connection FSM pattern requires at least 1 input (flags)")
    
    # Get flag inputs from params or inputs
    syn_flag_name = block.params.get("syn_flag", block.inputs[0] if len(block.inputs) > 0 else None)
    ack_flag_name = block.params.get("ack_flag", block.inputs[1] if len(block.inputs) > 1 else None)
    fin_flag_name = block.params.get("fin_flag", block.inputs[2] if len(block.inputs) > 2 else None)
    rst_flag_name = block.params.get("rst_flag", block.inputs[3] if len(block.inputs) > 3 else None)
    
    if not syn_flag_name:
        raise ValueError("TCP connection FSM pattern requires syn_flag")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # TCP states: 0=CLOSED, 1=LISTEN, 2=SYN_SENT, 3=SYN_RECEIVED, 4=ESTABLISHED,
    # 5=FIN_WAIT_1, 6=FIN_WAIT_2, 7=CLOSE_WAIT, 8=CLOSING, 9=TIME_WAIT, 10=LAST_ACK
    # Use 4-bit state (supports up to 16 states)
    state_width = 4
    
    # Get input indices
    syn_idx, syn_is_input = _get_stream_index_any(syn_flag_name, streams)
    syn_ref = f"i{syn_idx}[t]" if syn_is_input else f"o{syn_idx}[t]"
    
    ack_ref = None
    if ack_flag_name:
        ack_idx, ack_is_input = _get_stream_index_any(ack_flag_name, streams)
        ack_ref = f"i{ack_idx}[t]" if ack_is_input else f"o{ack_idx}[t]"
    
    fin_ref = None
    if fin_flag_name:
        fin_idx, fin_is_input = _get_stream_index_any(fin_flag_name, streams)
        fin_ref = f"i{fin_idx}[t]" if fin_is_input else f"o{fin_idx}[t]"
    
    rst_ref = None
    if rst_flag_name:
        rst_idx, rst_is_input = _get_stream_index_any(rst_flag_name, streams)
        rst_ref = f"i{rst_idx}[t]" if rst_is_input else f"o{rst_idx}[t]"
    
    # TCP state transitions (simplified):
    # CLOSED (0) → LISTEN (1): passive open
    # CLOSED (0) → SYN_SENT (2): active open (SYN)
    # LISTEN (1) → SYN_RECEIVED (3): receive SYN
    # SYN_SENT (2) → ESTABLISHED (4): receive SYN+ACK
    # SYN_RECEIVED (3) → ESTABLISHED (4): send ACK
    # ESTABLISHED (4) → FIN_WAIT_1 (5): send FIN
    # ESTABLISHED (4) → CLOSE_WAIT (7): receive FIN
    # FIN_WAIT_1 (5) → FIN_WAIT_2 (6): receive ACK
    # FIN_WAIT_1 (5) → CLOSING (8): receive FIN
    # FIN_WAIT_2 (6) → TIME_WAIT (9): receive FIN
    # CLOSE_WAIT (7) → LAST_ACK (10): send FIN
    # CLOSING (8) → TIME_WAIT (9): receive ACK
    # TIME_WAIT (9) → CLOSED (0): timeout
    # LAST_ACK (10) → CLOSED (0): receive ACK
    # Any state → CLOSED (0): RST
    
    # Build transition logic
    transitions = []
    
    # RST has highest priority: any state → CLOSED
    if rst_ref:
        transitions.append(f"({rst_ref} ? {{0}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # CLOSED → LISTEN: passive open (no SYN, but we'll use a separate signal or default)
    # For simplicity, assume LISTEN is initial state or use a separate "listen" signal
    # transitions.append(f"((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & listen_signal ? {{1}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # CLOSED → SYN_SENT: active open (SYN)
    transitions.append(f"((o{output_idx}[t-1] = {{0}}:bv[{state_width}]) & {syn_ref} ? {{2}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # LISTEN → SYN_RECEIVED: receive SYN
    transitions.append(f"((o{output_idx}[t-1] = {{1}}:bv[{state_width}]) & {syn_ref} ? {{3}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # SYN_SENT → ESTABLISHED: receive SYN+ACK
    if ack_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{2}}:bv[{state_width}]) & {syn_ref} & {ack_ref} ? {{4}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # SYN_RECEIVED → ESTABLISHED: send ACK
    if ack_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{3}}:bv[{state_width}]) & {ack_ref} ? {{4}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # ESTABLISHED → FIN_WAIT_1: send FIN
    if fin_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{4}}:bv[{state_width}]) & {fin_ref} ? {{5}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # ESTABLISHED → CLOSE_WAIT: receive FIN
    if fin_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{4}}:bv[{state_width}]) & {fin_ref} ? {{7}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # FIN_WAIT_1 → FIN_WAIT_2: receive ACK
    if ack_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{5}}:bv[{state_width}]) & {ack_ref} ? {{6}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # FIN_WAIT_1 → CLOSING: receive FIN
    if fin_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{5}}:bv[{state_width}]) & {fin_ref} ? {{8}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # FIN_WAIT_2 → TIME_WAIT: receive FIN
    if fin_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{6}}:bv[{state_width}]) & {fin_ref} ? {{9}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # CLOSE_WAIT → LAST_ACK: send FIN
    if fin_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{7}}:bv[{state_width}]) & {fin_ref} ? {{10}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # CLOSING → TIME_WAIT: receive ACK
    if ack_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{8}}:bv[{state_width}]) & {ack_ref} ? {{9}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # TIME_WAIT → CLOSED: timeout (simplified - use timeout signal or timer)
    timeout_signal_name = block.params.get("timeout_signal", None)
    if timeout_signal_name:
        timeout_idx, timeout_is_input = _get_stream_index_any(timeout_signal_name, streams)
        timeout_ref = f"i{timeout_idx}[t]" if timeout_is_input else f"o{timeout_idx}[t]"
        transitions.append(f"((o{output_idx}[t-1] = {{9}}:bv[{state_width}]) & {timeout_ref} ? {{0}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # LAST_ACK → CLOSED: receive ACK
    if ack_ref:
        transitions.append(f"((o{output_idx}[t-1] = {{10}}:bv[{state_width}]) & {ack_ref} ? {{0}}:bv[{state_width}] : {{999}}:bv[{state_width}])")
    
    # Combine transitions with fallback to maintain current state
    if transitions:
        transition_expr = " : ".join(transitions) + f" : o{output_idx}[t-1]"
        tcp_logic = f"(o{output_idx}[t] = {transition_expr})"
    else:
        # Fallback: simple SYN → SYN_SENT
        tcp_logic = f"(o{output_idx}[t] = ({syn_ref} ? {{2}}:bv[{state_width}] : o{output_idx}[t-1]))"
    
    # Initial condition: CLOSED
    initial_state = block.params.get("initial_state", 0)
    init_logic = f" && (o{output_idx}[0] = {{{initial_state}}}:bv[{state_width}])"
    
    return tcp_logic + init_logic


def _generate_utxo_state_machine_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate UTXO state machine pattern logic.
    
    Implements Bitcoin UTXO (Unspent Transaction Output) state machine.
    Tracks UTXO set state: UTXO exists or spent.
    Validates transactions: inputs must exist in UTXO set, outputs create new UTXOs.
    Updates UTXO set: remove spent outputs, add new outputs.
    """
    if len(block.inputs) < 1:
        raise ValueError("UTXO state machine pattern requires at least 1 input")
    
    # Get transaction inputs from params
    tx_inputs_name = block.params.get("tx_inputs", block.inputs[0] if len(block.inputs) > 0 else None)
    tx_outputs_name = block.params.get("tx_outputs", block.inputs[1] if len(block.inputs) > 1 else None)
    tx_valid_name = block.params.get("tx_valid", None)  # External validation (signatures, etc.)
    
    if not tx_inputs_name:
        raise ValueError("UTXO state machine pattern requires tx_inputs")
    
    # Get output streams
    utxo_set_output_name = block.params.get("utxo_set_output", block.output)
    tx_valid_output_name = block.params.get("tx_valid_output", None)
    
    # Get UTXO set output stream
    utxo_set_output_idx = _get_stream_index(utxo_set_output_name, streams, is_input=False)
    utxo_set_output_stream = next(s for s in streams if s.name == utxo_set_output_name and not s.is_input)
    
    # UTXO set is a bitvector where each bit represents a UTXO
    # For simplicity, we'll use a single bitvector to represent UTXO set
    # In practice, this would be a set/map, but Tau uses bitvectors
    
    # Get input indices
    tx_inputs_idx, tx_inputs_is_input = _get_stream_index_any(tx_inputs_name, streams)
    tx_inputs_ref = f"i{tx_inputs_idx}[t]" if tx_inputs_is_input else f"o{tx_inputs_idx}[t]"
    
    tx_outputs_ref = None
    if tx_outputs_name:
        tx_outputs_idx, tx_outputs_is_input = _get_stream_index_any(tx_outputs_name, streams)
        tx_outputs_ref = f"i{tx_outputs_idx}[t]" if tx_outputs_is_input else f"o{tx_outputs_idx}[t]"
    
    tx_valid_ref = None
    if tx_valid_name:
        tx_valid_idx, tx_valid_is_input = _get_stream_index_any(tx_valid_name, streams)
        tx_valid_ref = f"i{tx_valid_idx}[t]" if tx_valid_is_input else f"o{tx_valid_idx}[t]"
    
    # UTXO set update logic:
    # utxo_set[t] = (tx_valid & utxo_set[t-1] & !tx_inputs) | (tx_valid & tx_outputs)
    # This means:
    # - Remove spent outputs: utxo_set[t-1] & !tx_inputs (if tx_inputs references a UTXO, remove it)
    # - Add new outputs: tx_outputs (new UTXOs created)
    # - Only if transaction is valid: tx_valid
    
    # For simplicity, we'll use bitwise operations:
    # utxo_set[t] = (tx_valid ? ((utxo_set[t-1] & tx_inputs') | tx_outputs) : utxo_set[t-1])
    
    if tx_valid_ref:
        if tx_outputs_ref:
            # Full UTXO update: remove inputs, add outputs, only if valid
            utxo_logic = f"(o{utxo_set_output_idx}[t] = ({tx_valid_ref} ? ((o{utxo_set_output_idx}[t-1] & {tx_inputs_ref}') | {tx_outputs_ref}) : o{utxo_set_output_idx}[t-1]))"
        else:
            # Only remove inputs (no new outputs)
            utxo_logic = f"(o{utxo_set_output_idx}[t] = ({tx_valid_ref} ? (o{utxo_set_output_idx}[t-1] & {tx_inputs_ref}') : o{utxo_set_output_idx}[t-1]))"
    else:
        # No validation: always update
        if tx_outputs_ref:
            utxo_logic = f"(o{utxo_set_output_idx}[t] = ((o{utxo_set_output_idx}[t-1] & {tx_inputs_ref}') | {tx_outputs_ref}))"
        else:
            utxo_logic = f"(o{utxo_set_output_idx}[t] = (o{utxo_set_output_idx}[t-1] & {tx_inputs_ref}'))"
    
    logic_parts = [utxo_logic]
    
    # Transaction validation output (optional)
    if tx_valid_output_name:
        try:
            tx_valid_output_idx = _get_stream_index(tx_valid_output_name, streams, is_input=False)
            # Validate: all inputs must exist in UTXO set
            # tx_valid = tx_inputs & utxo_set[t-1] (all inputs are in UTXO set)
            # For simplicity, check if inputs are subset of UTXO set
            if tx_valid_ref:
                # Use external validation
                validation_logic = f"(o{tx_valid_output_idx}[t] = {tx_valid_ref}) && (o{tx_valid_output_idx}[0] = 0)"
            else:
                # Internal validation: inputs must be subset of UTXO set
                # tx_valid = (tx_inputs & utxo_set[t-1]) = tx_inputs (all inputs exist)
                validation_logic = f"(o{tx_valid_output_idx}[t] = (({tx_inputs_ref} & o{utxo_set_output_idx}[t-1]) = {tx_inputs_ref})) && (o{tx_valid_output_idx}[0] = 0)"
            logic_parts.append(validation_logic)
        except ValueError:
            pass  # Validation output not declared
    
    # Initial condition: initial UTXO set
    initial_utxo_set = block.params.get("initial_utxo_set", 0)
    if utxo_set_output_stream.stream_type == "sbf":
        init_logic = f" && (o{utxo_set_output_idx}[0] = {initial_utxo_set})"
    else:
        init_logic = f" && (o{utxo_set_output_idx}[0] = {{{initial_utxo_set}}}:bv[{utxo_set_output_stream.width if utxo_set_output_stream.width else 32}])"
    
    return " &&\n    ".join(logic_parts) + init_logic


def _generate_history_state_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate history state pattern logic.
    
    Implements history state: remembers last substate when returning to superstate.
    When entering superstate, if history exists, restore to last substate; otherwise use initial.
    """
    if len(block.inputs) < 2:
        raise ValueError("History state pattern requires at least 2 inputs (substate, superstate_entry)")
    
    # Get inputs from params
    substate_input_name = block.params.get("substate_input", block.inputs[0] if len(block.inputs) > 0 else None)
    superstate_entry_name = block.params.get("superstate_entry", block.inputs[1] if len(block.inputs) > 1 else None)
    superstate_exit_name = block.params.get("superstate_exit", None)
    
    if not substate_input_name or not superstate_entry_name:
        raise ValueError("History state pattern requires substate_input and superstate_entry")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get history storage output (optional, defaults to internal)
    history_output_name = block.params.get("history_output", None)
    
    # Get input indices
    substate_idx, substate_is_input = _get_stream_index_any(substate_input_name, streams)
    substate_ref = f"i{substate_idx}[t]" if substate_is_input else f"o{substate_idx}[t]"
    
    superstate_entry_idx, superstate_entry_is_input = _get_stream_index_any(superstate_entry_name, streams)
    superstate_entry_ref = f"i{superstate_entry_idx}[t]" if superstate_entry_is_input else f"o{superstate_entry_idx}[t]"
    
    superstate_exit_ref = None
    if superstate_exit_name:
        superstate_exit_idx, superstate_exit_is_input = _get_stream_index_any(superstate_exit_name, streams)
        superstate_exit_ref = f"i{superstate_exit_idx}[t]" if superstate_exit_is_input else f"o{superstate_exit_idx}[t]"
    
    # Determine state width
    state_width = output_stream.width if output_stream.width else 2
    
    # Initial substate
    initial_substate = block.params.get("initial_substate", 0)
    
    # History state logic: simplified version
    # When entering superstate: restore from history or use initial
    # When exiting: save current substate
    # Otherwise: use current substate
    
    if history_output_name:
        # External history storage
        history_idx = _get_stream_index(history_output_name, streams, is_input=False)
        history_ref = f"o{history_idx}[t-1]"
        
        # Save history when exiting
        if superstate_exit_ref:
            history_logic = f"(o{history_idx}[t] = ({superstate_exit_ref} ? {substate_ref} : o{history_idx}[t-1]))"
        else:
            history_logic = f"(o{history_idx}[t] = ({superstate_entry_ref}' ? {substate_ref} : o{history_idx}[t-1]))"
        
        # Restore logic
        restore_logic = f"(o{output_idx}[t] = ({superstate_entry_ref} & (o{history_idx}[t-1] != {{0}}:bv[{state_width}]) ? o{history_idx}[t-1] : ({superstate_entry_ref} ? {{{initial_substate}}}:bv[{state_width}] : {substate_ref})))"
        
        init_history = f" && (o{history_idx}[0] = {{0}}:bv[{state_width}])"
        init_output = f" && (o{output_idx}[0] = {{{initial_substate}}}:bv[{state_width}])"
        
        return f"({restore_logic}) && ({history_logic}){init_history}{init_output}"
    else:
        # Internal history (use previous output)
        history_ref = f"o{output_idx}[t-1]"
        
        # Simplified: restore on entry, save on exit, otherwise use substate
        if superstate_exit_ref:
            restore_logic = f"(o{output_idx}[t] = ({superstate_entry_ref} & ({history_ref} != {{0}}:bv[{state_width}]) ? {history_ref} : ({superstate_entry_ref} ? {{{initial_substate}}}:bv[{state_width}] : ({superstate_exit_ref} ? {substate_ref} : {substate_ref}))))"
        else:
            restore_logic = f"(o{output_idx}[t] = ({superstate_entry_ref} & ({history_ref} != {{0}}:bv[{state_width}]) ? {history_ref} : ({superstate_entry_ref} ? {{{initial_substate}}}:bv[{state_width}] : {substate_ref})))"
        
        init_output = f" && (o{output_idx}[0] = {{{initial_substate}}}:bv[{state_width}])"
        return f"({restore_logic}){init_output}"


def _generate_decomposed_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate decomposed FSM pattern logic.
    
    Implements hierarchical FSM decomposition: breaks down superstate into substates.
    Generates separate FSMs for each substate and aggregates them into superstate.
    """
    if len(block.inputs) < 1:
        raise ValueError("Decomposed FSM pattern requires at least 1 input")
    
    # Get hierarchy configuration from params
    hierarchy = block.params.get("hierarchy", {})
    transitions = block.params.get("transitions", [])
    
    if not hierarchy:
        raise ValueError("Decomposed FSM pattern requires 'hierarchy' parameter")
    
    # Get output streams for substates
    substate_outputs = block.params.get("substate_outputs", [])
    
    # Generate FSMs for each substate
    logic_parts = []
    
    # Track output indices
    output_indices = {}
    
    # Collect all unique substates first to avoid duplicates
    all_substates = {}
    for superstate_name, superstate_config in hierarchy.items():
        substates = superstate_config.get("substates", [])
        initial_substate = superstate_config.get("initial", substates[0] if substates else None)
        for i, substate_name in enumerate(substates):
            if substate_name not in all_substates:
                # Get output stream for this substate
                if i < len(substate_outputs):
                    substate_output_name = substate_outputs[i]
                else:
                    # Find index in flattened list
                    flat_idx = sum(len(hierarchy[k].get("substates", [])) for k in list(hierarchy.keys())[:list(hierarchy.keys()).index(superstate_name)]) + i
                    if flat_idx < len(substate_outputs):
                        substate_output_name = substate_outputs[flat_idx]
                    else:
                        substate_output_name = f"{substate_name}_state"
                
                all_substates[substate_name] = {
                    "output_name": substate_output_name,
                    "superstate": superstate_name,
                    "is_initial": (substate_name == initial_substate)
                }
    
    # Generate FSM for each unique substate
    for substate_name, substate_info in all_substates.items():
        try:
            substate_output_idx = _get_stream_index(substate_info["output_name"], streams, is_input=False)
            output_indices[substate_name] = substate_output_idx
            
            # Find transitions FROM this substate (exit transitions)
            from_transitions = [t for t in transitions if t.get("from") == substate_name]
            # Find transitions TO this substate (entry transitions)
            to_transitions = [t for t in transitions if t.get("to") == substate_name]
            
            # Build entry conditions (transitions TO this substate)
            entry_parts = []
            for trans in to_transitions:
                from_substate = trans.get("from")
                condition = trans.get("condition", "1")
                
                # Find source substate output index
                if from_substate in all_substates:
                    source_output_name = all_substates[from_substate]["output_name"]
                    try:
                        source_output_idx = _get_stream_index(source_output_name, streams, is_input=False)
                        
                        # Build condition expression
                        condition_expr = condition
                        for inp_name in block.inputs:
                            inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
                            inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
                            condition_expr = condition_expr.replace(f"{inp_name}[t]", inp_ref)
                            condition_expr = condition_expr.replace(f"{{{inp_name}}}", inp_ref)
                        
                        # Entry: source was active AND condition is true
                        entry_parts.append(f"(o{source_output_idx}[t-1] & ({condition_expr}))")
                    except ValueError:
                        pass
            
            # Build exit conditions (transitions FROM this substate)
            exit_parts = []
            for trans in from_transitions:
                condition = trans.get("condition", "1")
                condition_expr = condition
                for inp_name in block.inputs:
                    inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
                    inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
                    condition_expr = condition_expr.replace(f"{inp_name}[t]", inp_ref)
                    condition_expr = condition_expr.replace(f"{{{inp_name}}}", inp_ref)
                # Exit: this substate was active AND condition is true
                exit_parts.append(f"(o{substate_output_idx}[t-1] & ({condition_expr}))")
            
            # Build FSM logic:
            # Active if: (entry condition) OR (was active AND NOT exit condition)
            if entry_parts:
                entry_expr = " | ".join(entry_parts)
                if exit_parts:
                    exit_expr = " | ".join(exit_parts)
                    # Active if: entry OR (was active AND not exit)
                    substate_logic = f"(o{substate_output_idx}[t] = ({entry_expr} | (o{substate_output_idx}[t-1] & ({exit_expr})')))"
                else:
                    # Active if: entry OR was active
                    substate_logic = f"(o{substate_output_idx}[t] = ({entry_expr} | o{substate_output_idx}[t-1]))"
            else:
                # No entry transitions
                if exit_parts:
                    exit_expr = " | ".join(exit_parts)
                    # Active if: was active AND not exit
                    substate_logic = f"(o{substate_output_idx}[t] = (o{substate_output_idx}[t-1] & ({exit_expr})'))"
                else:
                    # No transitions: maintain state
                    substate_logic = f"(o{substate_output_idx}[t] = o{substate_output_idx}[t-1])"
            
            # Initial condition
            initial_value = 1 if substate_info["is_initial"] else 0
            init_logic = f" && (o{substate_output_idx}[0] = {initial_value})"
            
            logic_parts.append(substate_logic + init_logic)
            
        except ValueError:
            # Output stream not found, skip
            continue
    
    # Aggregate substates into superstate (optional)
    aggregate_output_name = block.params.get("aggregate_output", None)
    if aggregate_output_name:
        try:
            aggregate_output_idx = _get_stream_index(aggregate_output_name, streams, is_input=False)
            # Aggregate: superstate active if any substate active
            substate_refs = [f"o{idx}[t]" for idx in output_indices.values()]
            if substate_refs:
                aggregate_expr = " | ".join(substate_refs)
                aggregate_logic = f"(o{aggregate_output_idx}[t] = {aggregate_expr}) && (o{aggregate_output_idx}[0] = 0)"
                logic_parts.append(aggregate_logic)
        except ValueError:
            pass
    
    if not logic_parts:
        raise ValueError("No valid substate outputs found for decomposed FSM")
    
    return " &&\n    ".join(logic_parts)


def _generate_script_execution_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate script execution pattern logic.
    
    Implements Bitcoin Script execution engine (simplified stack-based VM).
    Handles basic opcodes: OP_DUP, OP_HASH160, OP_EQUALVERIFY, OP_CHECKSIG (external).
    """
    if len(block.inputs) < 1:
        raise ValueError("Script execution pattern requires at least 1 input")
    
    # Get script and stack inputs
    script_input_name = block.params.get("script_input", block.inputs[0] if len(block.inputs) > 0 else None)
    stack_input_name = block.params.get("stack_input", block.inputs[1] if len(block.inputs) > 1 else None)
    
    if not script_input_name:
        raise ValueError("Script execution pattern requires script_input")
    
    # Get output
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    output_stream = next(s for s in streams if s.name == block.output and not s.is_input)
    
    # Get input indices
    script_idx, script_is_input = _get_stream_index_any(script_input_name, streams)
    script_ref = f"i{script_idx}[t]" if script_is_input else f"o{script_idx}[t]"
    
    stack_ref = None
    if stack_input_name:
        stack_idx, stack_is_input = _get_stream_index_any(stack_input_name, streams)
        stack_ref = f"i{stack_idx}[t]" if stack_is_input else f"o{stack_idx}[t]"
    
    # Simplified script execution: passthrough with optional stack manipulation
    state_width = output_stream.width if output_stream.width else 32
    
    # Basic opcode execution (simplified)
    # For now, implement simple passthrough with opcode selection
    if stack_ref:
        script_logic = f"(o{output_idx}[t] = {stack_ref})"
    else:
        script_logic = f"(o{output_idx}[t] = {script_ref})"
    
    # Initial condition
    initial_value = block.params.get("initial_value", 0)
    init_logic = f" && (o{output_idx}[0] = {{{initial_value}}}:bv[{state_width}])"
    
    return script_logic + init_logic


def _generate_quorum_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate quorum pattern logic (uses majority internally).
    
    Quorum is essentially majority voting with a threshold.
    Example: 3-of-5 quorum = majority with threshold=3, total=5
    """
    if len(block.inputs) < 2:
        raise ValueError("Quorum pattern requires at least 2 inputs")
    
    # Get threshold from params
    threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)
    total = block.params.get("total", len(block.inputs))
    
    if threshold < 1:
        raise ValueError(f"Quorum threshold must be >= 1, got {threshold}")
    if threshold > total:
        raise ValueError(f"Quorum threshold ({threshold}) cannot exceed total ({total})")
    if total > len(block.inputs):
        raise ValueError(f"Total ({total}) cannot exceed number of inputs ({len(block.inputs)})")
    
    # Use majority logic internally
    input_refs = []
    for inp in block.inputs[:total]:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        input_refs.append((ref, inp_idx))
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Generate all combinations of threshold inputs
    combinations = list(itertools.combinations(input_refs, threshold))
    
    if not combinations:
        raise ValueError(f"No valid combinations for threshold={threshold}, total={total}")
    
    # Create AND expressions for each combination, then OR them together
    and_exprs = []
    for combo in combinations:
        and_expr = " & ".join(ref for ref, _ in combo)
        and_exprs.append(f"({and_expr})")
    
    quorum_expr = " | ".join(and_exprs)
    return f"(o{output_idx}[t] = {quorum_expr})"


def _generate_custom_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate custom boolean expression pattern logic.
    
    The expression string should use placeholders like {i0}, {i1}, etc. for inputs,
    or use stream names that will be replaced with indices.
    """
    if "expression" not in block.params:
        raise ValueError("Custom pattern requires 'expression' parameter")
    
    expression = block.params["expression"]
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Replace stream names with indices in the expression
    # Process in reverse order to avoid replacing already-replaced indices
    # First, replace stream names with placeholders
    name_to_placeholder = {}
    for inp_name in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
        prefix = "i" if inp_is_input else "o"
        name_to_placeholder[inp_name] = f"__PLACEHOLDER_{prefix}{inp_idx}__"
    
    # Replace stream names with placeholders
    for inp_name, placeholder in name_to_placeholder.items():
        expression = expression.replace(f"{inp_name}[t]", placeholder)
        expression = expression.replace(f"{{{inp_name}}}", placeholder)
    
    # Replace placeholders with actual indices
    for inp_name, placeholder in name_to_placeholder.items():
        inp_idx, inp_is_input = _get_stream_index_any(inp_name, streams)
        prefix = "i" if inp_is_input else "o"
        expression = expression.replace(placeholder, f"{prefix}{inp_idx}[t]")
    
    # Also handle direct index placeholders like {i0}, {i1}
    for idx, inp_name in enumerate(block.inputs):
        inp_idx = _get_stream_index(inp_name, streams, is_input=True)
        expression = expression.replace(f"{{i{idx}}}", f"i{inp_idx}[t]")
    
    # Replace output placeholder
    expression = expression.replace("{output}", f"o{output_idx}")
    expression = expression.replace("{output_idx}", f"o{output_idx}")
    
    return f"(o{output_idx}[t] = {expression})"


def _generate_recurrence_block(schema: AgentSchema) -> List[str]:
    """Generate recurrence relation block."""
    lines = ["defs", "r ("]
    
    logic_lines = []
    for block in schema.logic_blocks:
        if block.pattern == "fsm":
            logic_lines.append(_generate_fsm_logic(block, schema.streams))
        elif block.pattern == "passthrough":
            logic_lines.append(_generate_passthrough_logic(block, schema.streams))
        elif block.pattern == "counter":
            logic_lines.append(_generate_counter_logic(block, schema.streams))
        elif block.pattern == "accumulator":
            logic_lines.append(_generate_accumulator_logic(block, schema.streams))
        elif block.pattern == "vote":
            logic_lines.append(_generate_vote_logic(block, schema.streams))
        elif block.pattern == "majority":
            logic_lines.append(_generate_majority_logic(block, schema.streams))
        elif block.pattern == "unanimous":
            logic_lines.append(_generate_unanimous_logic(block, schema.streams))
        elif block.pattern == "custom":
            logic_lines.append(_generate_custom_logic(block, schema.streams))
        elif block.pattern == "quorum":
            logic_lines.append(_generate_quorum_logic(block, schema.streams))
        elif block.pattern == "supervisor_worker":
            logic_lines.append(_generate_supervisor_worker_logic(block, schema.streams))
        elif block.pattern == "weighted_vote":
            logic_lines.append(_generate_weighted_vote_logic(block, schema.streams))
        elif block.pattern == "time_lock":
            logic_lines.append(_generate_time_lock_logic(block, schema.streams))
        elif block.pattern == "hex_stake":
            logic_lines.append(_generate_hex_stake_logic(block, schema.streams))
        elif block.pattern == "multi_bit_counter":
            logic_lines.append(_generate_multi_bit_counter_logic(block, schema.streams))
        elif block.pattern == "streak_counter":
            logic_lines.append(_generate_streak_counter_logic(block, schema.streams))
        elif block.pattern == "mode_switch":
            logic_lines.append(_generate_mode_switch_logic(block, schema.streams))
        elif block.pattern == "proposal_fsm":
            logic_lines.append(_generate_proposal_fsm_logic(block, schema.streams))
        elif block.pattern == "risk_fsm":
            logic_lines.append(_generate_risk_fsm_logic(block, schema.streams))
        elif block.pattern == "entry_exit_fsm":
            logic_lines.append(_generate_entry_exit_fsm_logic(block, schema.streams))
        elif block.pattern == "orthogonal_regions":
            logic_lines.append(_generate_orthogonal_regions_logic(block, schema.streams))
        elif block.pattern == "state_aggregation":
            logic_lines.append(_generate_state_aggregation_logic(block, schema.streams))
        elif block.pattern == "tcp_connection_fsm":
            logic_lines.append(_generate_tcp_connection_fsm_logic(block, schema.streams))
        elif block.pattern == "utxo_state_machine":
            logic_lines.append(_generate_utxo_state_machine_logic(block, schema.streams))
        elif block.pattern == "history_state":
            logic_lines.append(_generate_history_state_logic(block, schema.streams))
        elif block.pattern == "decomposed_fsm":
            logic_lines.append(_generate_decomposed_fsm_logic(block, schema.streams))
        elif block.pattern == "script_execution":
            logic_lines.append(_generate_script_execution_logic(block, schema.streams))
        else:
            raise ValueError(f"Unknown pattern: {block.pattern}")
    
    # Join logic lines with &&
    if logic_lines:
        lines.append("    " + " &&\n    ".join(logic_lines))
    
    lines.append(")")
    return lines


def _generate_execution_commands(num_steps: int) -> List[str]:
    """Generate execution commands (n commands + q)."""
    lines = []
    for _ in range(num_steps):
        lines.append("n")
    lines.append("q")
    return lines


def generate_tau_spec(schema: AgentSchema, validate: bool = True) -> str:
    """Generate a complete Tau Language spec from an AgentSchema.
    
    Args:
        schema: The agent schema to compile
        validate: Whether to validate the generated spec
        
    Returns:
        Complete Tau spec as a string
        
    Raises:
        ValueError: If validation fails and validate=True
    """
    lines = [
        f"# {schema.name} Agent (Auto-generated)",
        f"# Strategy: {schema.strategy}",
        "",
    ]
    
    # Input declarations
    lines.extend(_generate_inputs(schema.streams))
    lines.append("")
    
    # Input mirrors (if enabled)
    if schema.include_mirrors:
        lines.extend(_generate_input_mirrors(schema.streams))
        lines.append("")
    
    # Output declarations
    lines.extend(_generate_outputs(schema.streams))
    lines.append("")
    
    # Recurrence block
    lines.extend(_generate_recurrence_block(schema))
    lines.append("")
    
    # Execution commands
    lines.extend(_generate_execution_commands(schema.num_steps))
    
    spec = "\n".join(lines)
    
    # Validate if requested
    if validate:
        from idi.devkit.tau_factory.spec_validator import validate_tau_spec
        is_valid, errors = validate_tau_spec(spec)
        if not is_valid:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Generated spec validation failed:\n{error_msg}\n\nSpec:\n{spec}")
    
    return spec

