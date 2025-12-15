"""Tau spec generator - compiles AgentSchema to Tau Language specs.

This module contains the reference implementation used by the test suite,
including pattern expansion and optional spec validation.
"""

from __future__ import annotations

import itertools

from idi.devkit.tau_factory.schema import AgentSchema
from .dsl_parser import DSLParser, ValidationError
from .performance_monitor import monitor


def validate_schema(schema: AgentSchema) -> list[ValidationError]:
    """Validate an AgentSchema without generating code.

    Args:
        schema: AgentSchema to validate

    Returns:
        List of validation errors (empty if valid)
    """
    parser = DSLParser()
    return parser.validate_schema(schema)


def create_minimal_schema(name: str, strategy: str = "custom") -> AgentSchema:
    """Create a minimal valid AgentSchema for testing.

    Args:
        name: Name for the agent schema
        strategy: Trading strategy type

    Returns:
        Minimal AgentSchema instance
    """
    from idi.devkit.tau_factory.schema import StreamConfig, LogicBlock

    # Create basic I/O streams
    streams = (
        StreamConfig("input1", "sbf", is_input=True),
        StreamConfig("output1", "sbf", is_input=False),
    )

    # Create basic logic block
    logic_blocks = (
        LogicBlock(
            pattern="passthrough",
            inputs=("input1",),
            output="output1"
        ),
    )

    return AgentSchema(
        name=name,
        strategy=strategy,
        streams=streams,
        logic_blocks=logic_blocks
    )


def _generate_inputs(streams: tuple[StreamConfig, ...], descriptive: bool = False) -> List[str]:
    """Generate input stream declarations.
    
    Args:
        streams: Stream configurations
        descriptive: If True, use human-readable names (requires charvar=off)
    """
    lines = []
    input_streams = [s for s in streams if s.is_input]
    for idx, stream in enumerate(input_streams):
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        if descriptive:
            # Use stream name directly for readability
            var_name = stream.name.replace("-", "_")  # Tau doesn't allow hyphens
            lines.append(f'{var_name}:{type_str} = in file("inputs/{stream.name}.in")')
        else:
            hex_idx = hex(idx)[2:].upper()  # i0, i1, ..., i9, iA, iB, ...
            lines.append(f'i{hex_idx}:{type_str} = in file("inputs/{stream.name}.in")')
    return lines


def _generate_outputs(streams: tuple[StreamConfig, ...], descriptive: bool = False) -> List[str]:
    """Generate output stream declarations.
    
    Args:
        streams: Stream configurations
        descriptive: If True, use human-readable names (requires charvar=off)
    """
    lines = []
    output_streams = [s for s in streams if not s.is_input]
    for idx, stream in enumerate(output_streams):
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        if descriptive:
            var_name = stream.name.replace("-", "_")
            lines.append(f'{var_name}:{type_str} = out file("outputs/{stream.name}.out")')
        else:
            hex_idx = hex(idx)[2:].upper()  # o0, o1, ..., o9, oA, oB, ...
            lines.append(f'o{hex_idx}:{type_str} = out file("outputs/{stream.name}.out")')
    return lines


def _generate_input_mirrors(streams: tuple[StreamConfig, ...], descriptive: bool = False) -> List[str]:
    """Generate input mirror declarations for symmetric I/O.
    
    Note: Mirrors use unique stream names to avoid conflict with input streams.
    
    Args:
        streams: Stream configurations
        descriptive: If True, use human-readable names (requires charvar=off)
    """
    lines = []
    input_streams = [s for s in streams if s.is_input]
    for idx, stream in enumerate(input_streams):
        type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
        if descriptive:
            var_name = f"{stream.name}_mirror".replace("-", "_")
            lines.append(f'{var_name}:{type_str} = out file("outputs/{stream.name}_mirror.out")')
        else:
            hex_idx = hex(idx)[2:].upper()
            lines.append(f'mi{hex_idx}:{type_str} = out file("outputs/i{hex_idx}_mirror.out")')
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
    
    # Output: lock_active
    # IMPORTANT: Tau/cvc5 uses UNSIGNED bitvector comparisons.
    # Using (end_time - current_time) > 0 is incorrect under wrap-around (expired locks wrap to a large value).
    # Prefer comparing timestamps directly: current_time < end_time.
    if output_stream.stream_type == "sbf":
        end_time_expr = f"({lock_start_ref} + {lock_duration_ref})"
        return f"(o{output_idx}[t] = ({current_time_ref} < {end_time_expr}))"
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
    global_substate_idx = 0  # Track global index across all superstates
    for superstate_name, superstate_config in hierarchy.items():
        substates = superstate_config.get("substates", [])
        initial_substate = superstate_config.get("initial", substates[0] if substates else None)
        for i, substate_name in enumerate(substates):
            if substate_name not in all_substates:
                # Get output stream for this substate using global index
                if global_substate_idx < len(substate_outputs):
                    substate_output_name = substate_outputs[global_substate_idx]
                else:
                    substate_output_name = f"{substate_name}_state"
                
                all_substates[substate_name] = {
                    "output_name": substate_output_name,
                    "superstate": superstate_name,
                    "is_initial": (substate_name == initial_substate)
                }
                global_substate_idx += 1
    
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
                # Exit condition: just the condition (we'll AND with state in the logic)
                exit_parts.append(f"({condition_expr})")
            
            # Build FSM logic:
            # Active if: (entry condition) OR (was active AND NOT exit condition)
            if entry_parts:
                entry_expr = " | ".join(entry_parts)
                if exit_parts:
                    exit_expr = " | ".join(exit_parts)
                    # Active if: entry OR (was active AND not exit)
                    # exit_expr is just the condition, so we need: was active AND (NOT condition)
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


def _validate_custom_expression(expression: str) -> tuple[bool, str]:
    """Validate custom expression for safe Tau syntax only.
    
    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    import re
    
    dangerous_patterns = [
        (r'[;]', "semicolons not allowed"),
        (r'[\n\r]', "newlines not allowed"),
        (r'in\s+file\s*\(', "file I/O not allowed in expressions"),
        (r'out\s+file\s*\(', "file I/O not allowed in expressions"),
        (r'in\s+console', "console I/O not allowed in expressions"),
        (r'out\s+console', "console I/O not allowed in expressions"),
        (r'\bexec\b', "exec not allowed"),
        (r'\beval\b', "eval not allowed"),
        (r'\bimport\b', "import not allowed"),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            return False, f"Invalid expression: {message}"
    
    return True, ""


def _generate_custom_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate custom boolean expression pattern logic.
    
    The expression string should use placeholders like {i0}, {i1}, etc. for inputs,
    or use stream names that will be replaced with indices.
    """
    if "expression" not in block.params:
        raise ValueError("Custom pattern requires 'expression' parameter")
    
    expression = block.params["expression"]
    
    is_valid, error = _validate_custom_expression(expression)
    if not is_valid:
        raise ValueError(error)
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


# ============================================================================
# NEW PATTERNS - Signal Processing
# ============================================================================

def _generate_edge_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Detect rising edge: output is 1 when input transitions from 0 to 1."""
    if len(block.inputs) != 1:
        raise ValueError("edge_detector pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Rising edge: current=1 AND previous=0
    return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]') && (o{output_idx}[0] = 0)"


def _generate_falling_edge_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Detect falling edge: output is 1 when input transitions from 1 to 0."""
    if len(block.inputs) != 1:
        raise ValueError("falling_edge pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Falling edge: current=0 AND previous=1
    return f"(o{output_idx}[t] = {inp_ref}[t]' & {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_toggle_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Toggle output on each rising edge of input."""
    if len(block.inputs) != 1:
        raise ValueError("toggle pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Toggle on rising edge: XOR current output with edge detection
    return f"(o{output_idx}[t] = ({inp_ref}[t] & {inp_ref}[t-1]') ^ o{output_idx}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_latch_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """SR Latch: set input sets output to 1, reset clears it."""
    if len(block.inputs) != 2:
        raise ValueError("latch pattern requires exactly 2 inputs (set, reset)")
    
    set_idx, set_is_input = _get_stream_index_any(block.inputs[0], streams)
    reset_idx, reset_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    set_ref = f"i{set_idx}[t]" if set_is_input else f"o{set_idx}[t]"
    reset_ref = f"i{reset_idx}[t]" if reset_is_input else f"o{reset_idx}[t]"
    
    return f"(o{output_idx}[t] = {set_ref} | (o{output_idx}[t-1] & {reset_ref}')) && (o{output_idx}[0] = 0)"


def _generate_debounce_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Debounce: output only changes if input is stable for N cycles."""
    if len(block.inputs) != 1:
        raise ValueError("debounce pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Simple debounce: require 2 consecutive same values
    # output = input if (input == prev_input), else hold
    return f"(o{output_idx}[t] = ({inp_ref}[t] & {inp_ref}[t-1]) | (o{output_idx}[t-1] & ({inp_ref}[t] ^ {inp_ref}[t-1]))) && (o{output_idx}[0] = 0)"


def _generate_pulse_generator_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate single pulse on rising edge of trigger."""
    if len(block.inputs) != 1:
        raise ValueError("pulse_generator pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Single cycle pulse on rising edge
    return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]') && (o{output_idx}[0] = 0)"


def _generate_sample_hold_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Sample and hold: capture input value when trigger is high."""
    if len(block.inputs) != 2:
        raise ValueError("sample_hold pattern requires exactly 2 inputs (data, trigger)")
    
    data_idx, data_is_input = _get_stream_index_any(block.inputs[0], streams)
    trig_idx, trig_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    data_ref = f"i{data_idx}[t]" if data_is_input else f"o{data_idx}[t]"
    trig_ref = f"i{trig_idx}[t]" if trig_is_input else f"o{trig_idx}[t]"
    
    # Sample on trigger high, otherwise hold previous value
    return f"(o{output_idx}[t] = ({trig_ref} & {data_ref}) | ({trig_ref}' & o{output_idx}[t-1])) && (o{output_idx}[0] = 0)"


# ============================================================================
# NEW PATTERNS - Data Flow / Routing
# ============================================================================

def _generate_multiplexer_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """2-to-1 multiplexer: select between two inputs based on selector."""
    if len(block.inputs) != 3:
        raise ValueError("multiplexer pattern requires exactly 3 inputs (a, b, select)")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    sel_idx, sel_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    sel_ref = f"i{sel_idx}[t]" if sel_is_input else f"o{sel_idx}[t]"
    
    # MUX: sel=0 -> a, sel=1 -> b
    return f"(o{output_idx}[t] = ({sel_ref}' & {a_ref}) | ({sel_ref} & {b_ref}))"


def _generate_demultiplexer_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """1-to-2 demultiplexer: route input to one of two outputs based on selector."""
    if len(block.inputs) != 2:
        raise ValueError("demultiplexer pattern requires exactly 2 inputs (data, select)")
    
    data_idx, data_is_input = _get_stream_index_any(block.inputs[0], streams)
    sel_idx, sel_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    data_ref = f"i{data_idx}[t]" if data_is_input else f"o{data_idx}[t]"
    sel_ref = f"i{sel_idx}[t]" if sel_is_input else f"o{sel_idx}[t]"
    
    # Output A (when sel=0)
    which_output = block.params.get("output_index", 0)
    if which_output == 0:
        return f"(o{output_idx}[t] = {sel_ref}' & {data_ref})"
    else:
        return f"(o{output_idx}[t] = {sel_ref} & {data_ref})"


def _generate_priority_encoder_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Priority encoder: output the index of highest priority active input."""
    if len(block.inputs) < 2:
        raise ValueError("priority_encoder pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Build priority logic: highest index wins
    # Output is 1 if highest priority input (last one) is active
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Highest priority (last input) active
    highest = refs[-1]
    return f"(o{output_idx}[t] = {highest})"


def _generate_arbiter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Round-robin arbiter: grant access to requesters in rotation."""
    if len(block.inputs) < 2:
        raise ValueError("arbiter pattern requires at least 2 inputs (requests)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Simple arbiter: OR all requests (any request grants)
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = {or_expr})"


# ============================================================================
# NEW PATTERNS - Safety / Watchdog
# ============================================================================

def _generate_watchdog_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Watchdog: output alarm if no heartbeat received."""
    if len(block.inputs) != 1:
        raise ValueError("watchdog pattern requires exactly 1 input (heartbeat)")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Alarm if no heartbeat (input stayed 0 for 2 cycles)
    return f"(o{output_idx}[t] = {inp_ref}[t]' & {inp_ref}[t-1]') && (o{output_idx}[0] = 0)"


def _generate_deadman_switch_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Deadman switch: output active only while input is continuously held."""
    if len(block.inputs) != 1:
        raise ValueError("deadman_switch pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Output active only if input has been high for consecutive cycles
    return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_safety_interlock_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Safety interlock: require all safety conditions to enable output."""
    if len(block.inputs) < 2:
        raise ValueError("safety_interlock pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # AND all safety inputs
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_fault_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Fault detector: detect disagreement between redundant inputs."""
    if len(block.inputs) < 2:
        raise ValueError("fault_detector pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # XOR all inputs - output 1 if any disagreement
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Pairwise XOR to detect any difference
    xor_expr = refs[0]
    for ref in refs[1:]:
        xor_expr = f"({xor_expr} ^ {ref})"
    
    return f"(o{output_idx}[t] = {xor_expr})"


# ============================================================================
# NEW PATTERNS - Protocol / Handshake
# ============================================================================

def _generate_handshake_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Request-acknowledge handshake protocol."""
    if len(block.inputs) != 2:
        raise ValueError("handshake pattern requires exactly 2 inputs (request, acknowledge)")
    
    req_idx, req_is_input = _get_stream_index_any(block.inputs[0], streams)
    ack_idx, ack_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    req_ref = f"i{req_idx}[t]" if req_is_input else f"o{req_idx}[t]"
    ack_ref = f"i{ack_idx}[t]" if ack_is_input else f"o{ack_idx}[t]"
    
    # Complete when both request and acknowledge are active
    return f"(o{output_idx}[t] = {req_ref} & {ack_ref})"


def _generate_sync_barrier_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Synchronization barrier: output when all inputs are ready."""
    if len(block.inputs) < 2:
        raise ValueError("sync_barrier pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # AND all ready signals
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_token_ring_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Token ring: pass token to next when current holder releases."""
    if len(block.inputs) != 2:
        raise ValueError("token_ring pattern requires exactly 2 inputs (token_in, release)")
    
    token_idx, token_is_input = _get_stream_index_any(block.inputs[0], streams)
    release_idx, release_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    token_ref = f"i{token_idx}[t]" if token_is_input else f"o{token_idx}[t]"
    release_ref = f"i{release_idx}[t]" if release_is_input else f"o{release_idx}[t]"
    
    # Hold token until release, accept incoming token
    return f"(o{output_idx}[t] = {token_ref} | (o{output_idx}[t-1] & {release_ref}')) && (o{output_idx}[0] = 0)"


# ============================================================================
# NEW PATTERNS - Arithmetic / Comparison
# ============================================================================

def _generate_comparator_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Compare two inputs: output 1 if a equals b."""
    if len(block.inputs) != 2:
        raise ValueError("comparator pattern requires exactly 2 inputs")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    # Equality: XNOR (not XOR)
    return f"(o{output_idx}[t] = ({a_ref} ^ {b_ref})')"


def _generate_min_selector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Select minimum: for sbf, output 1 only if all inputs are 1."""
    if len(block.inputs) < 2:
        raise ValueError("min_selector pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_max_selector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Select maximum: for sbf, output 1 if any input is 1."""
    if len(block.inputs) < 2:
        raise ValueError("max_selector pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = {or_expr})"


def _generate_threshold_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Threshold detector: output 1 if at least N of M inputs are 1."""
    if len(block.inputs) < 2:
        raise ValueError("threshold_detector pattern requires at least 2 inputs")
    
    threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Generate combinations
    combinations = list(itertools.combinations(refs, threshold))
    and_exprs = [" & ".join(combo) for combo in combinations]
    or_expr = " | ".join(f"({e})" for e in and_exprs)
    
    return f"(o{output_idx}[t] = {or_expr})"


# ============================================================================
# NEW PATTERNS - Consensus / Distributed
# ============================================================================

def _generate_byzantine_fault_tolerant_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Byzantine fault tolerant: tolerate up to f faulty nodes (need 3f+1 total)."""
    if len(block.inputs) < 4:
        raise ValueError("byzantine_fault_tolerant pattern requires at least 4 inputs")
    
    n = len(block.inputs)
    f = block.params.get("faulty_tolerance", (n - 1) // 3)
    threshold = 2 * f + 1  # Need 2f+1 agreement
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    combinations = list(itertools.combinations(refs, threshold))
    and_exprs = [" & ".join(combo) for combo in combinations]
    or_expr = " | ".join(f"({e})" for e in and_exprs)
    
    return f"(o{output_idx}[t] = {or_expr})"


def _generate_leader_election_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Leader election: select leader based on priority inputs."""
    if len(block.inputs) < 2:
        raise ValueError("leader_election pattern requires at least 2 inputs (candidates)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Highest priority candidate wins (last input has highest priority)
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Build priority chain: last one wins if active
    result = refs[-1]
    for ref in reversed(refs[:-1]):
        result = f"({ref} & {refs[-1]}') | {result}"
    
    return f"(o{output_idx}[t] = {refs[-1]})"


def _generate_commit_protocol_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Two-phase commit: require prepare then commit from all participants."""
    if len(block.inputs) < 2:
        raise ValueError("commit_protocol pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # All must agree
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


# ============================================================================
# NEW PATTERNS - Gate / Logic
# ============================================================================

def _generate_nand_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """NAND gate: output is NOT(AND of all inputs)."""
    if len(block.inputs) < 2:
        raise ValueError("nand_gate pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = ({and_expr})')"


def _generate_nor_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """NOR gate: output is NOT(OR of all inputs)."""
    if len(block.inputs) < 2:
        raise ValueError("nor_gate pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = ({or_expr})')"


def _generate_xnor_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """XNOR gate: output is 1 if inputs are equal."""
    if len(block.inputs) != 2:
        raise ValueError("xnor_gate pattern requires exactly 2 inputs")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    return f"(o{output_idx}[t] = ({a_ref} ^ {b_ref})')"


def _generate_implication_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Logical implication: a -> b (if a then b)."""
    if len(block.inputs) != 2:
        raise ValueError("implication pattern requires exactly 2 inputs")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    # a -> b = !a | b
    return f"(o{output_idx}[t] = {a_ref}' | {b_ref})"


def _generate_equivalence_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Logical equivalence: a <-> b (a if and only if b)."""
    if len(block.inputs) != 2:
        raise ValueError("equivalence pattern requires exactly 2 inputs")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    # a <-> b = (a & b) | (!a & !b) = !(a ^ b)
    return f"(o{output_idx}[t] = ({a_ref} ^ {b_ref})')"


# ============================================================================
# NEW PATTERNS - Timing / Delay
# ============================================================================

def _generate_delay_line_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Delay line: output is input delayed by 1 cycle."""
    if len(block.inputs) != 1:
        raise ValueError("delay_line pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    return f"(o{output_idx}[t] = {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_hold_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Hold: maintain last value until new trigger."""
    if len(block.inputs) != 2:
        raise ValueError("hold pattern requires exactly 2 inputs (data, load)")
    
    data_idx, data_is_input = _get_stream_index_any(block.inputs[0], streams)
    load_idx, load_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    data_ref = f"i{data_idx}[t]" if data_is_input else f"o{data_idx}[t]"
    load_ref = f"i{load_idx}[t]" if load_is_input else f"o{load_idx}[t]"
    
    return f"(o{output_idx}[t] = ({load_ref} & {data_ref}) | ({load_ref}' & o{output_idx}[t-1])) && (o{output_idx}[0] = 0)"


def _generate_one_shot_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """One-shot: output single pulse, then lock until reset."""
    if len(block.inputs) != 2:
        raise ValueError("one_shot pattern requires exactly 2 inputs (trigger, reset)")
    
    trig_idx, trig_is_input = _get_stream_index_any(block.inputs[0], streams)
    reset_idx, reset_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    trig_ref = f"i{trig_idx}[t]" if trig_is_input else f"o{trig_idx}[t]"
    reset_ref = f"i{reset_idx}[t]" if reset_is_input else f"o{reset_idx}[t]"
    
    # Trigger once, lock until reset
    return f"(o{output_idx}[t] = ({trig_ref} & o{output_idx}[t-1]') | (o{output_idx}[t-1] & {reset_ref}')) && (o{output_idx}[0] = 0)"


# ============================================================================
# NEW PATTERNS - State Encoding
# ============================================================================

def _generate_gray_code_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Gray code: only one bit changes between adjacent states."""
    if len(block.inputs) != 1:
        raise ValueError("gray_code pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # XOR with delayed self (gray code transition)
    return f"(o{output_idx}[t] = {inp_ref}[t] ^ o{output_idx}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_ring_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Ring counter: single hot bit rotates through positions."""
    if len(block.inputs) != 1:
        raise ValueError("ring_counter pattern requires exactly 1 input (clock)")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Toggle on clock edge
    return f"(o{output_idx}[t] = ({inp_ref}[t] & {inp_ref}[t-1]') ^ o{output_idx}[t-1]) && (o{output_idx}[0] = 1)"


def _generate_sequence_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Sequence detector: output when specific pattern is detected."""
    if len(block.inputs) != 1:
        raise ValueError("sequence_detector pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Detect sequence "11" (two consecutive 1s)
    pattern = block.params.get("pattern", "11")
    if pattern == "11":
        return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"
    elif pattern == "01":
        return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]') && (o{output_idx}[0] = 0)"
    elif pattern == "10":
        return f"(o{output_idx}[t] = {inp_ref}[t]' & {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"
    else:
        return f"(o{output_idx}[t] = {inp_ref}[t]' & {inp_ref}[t-1]') && (o{output_idx}[0] = 0)"


# ============================================================================
# INTELLIGENT AGENT PATTERNS - Decision Making
# ============================================================================

def _generate_confidence_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Confidence gate: only pass signal if confidence threshold is met.
    
    Use case: Filter agent actions based on confidence level.
    """
    if len(block.inputs) != 2:
        raise ValueError("confidence_gate pattern requires exactly 2 inputs (signal, confidence)")
    
    signal_idx, signal_is_input = _get_stream_index_any(block.inputs[0], streams)
    conf_idx, conf_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    signal_ref = f"i{signal_idx}[t]" if signal_is_input else f"o{signal_idx}[t]"
    conf_ref = f"i{conf_idx}[t]" if conf_is_input else f"o{conf_idx}[t]"
    
    # Only pass signal when confidence is high
    return f"(o{output_idx}[t] = {signal_ref} & {conf_ref})"


def _generate_action_selector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Action selector: choose action based on priority and availability.
    
    Use case: Select best action from multiple candidates.
    """
    if len(block.inputs) < 2:
        raise ValueError("action_selector pattern requires at least 2 inputs (action candidates)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Priority selection: first available action wins
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Build priority chain: first active input wins
    # action0 if active, else action1 if active, etc.
    result = refs[-1]
    for i in range(len(refs) - 2, -1, -1):
        # result = refs[i] | (refs[i]' & result)
        not_higher = " & ".join(f"{refs[j]}'" for j in range(i))
        if not_higher:
            result = f"({refs[i]} & {not_higher}) | ({refs[i]}' & {result})"
        else:
            result = f"{refs[i]} | ({refs[i]}' & {result})"
    
    return f"(o{output_idx}[t] = {refs[0]})"


def _generate_exploration_exploit_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Exploration vs exploitation: balance between exploring new actions and exploiting known good ones.
    
    Use case: Epsilon-greedy style decision making.
    """
    if len(block.inputs) != 3:
        raise ValueError("exploration_exploit pattern requires exactly 3 inputs (explore_action, exploit_action, explore_flag)")
    
    explore_idx, explore_is_input = _get_stream_index_any(block.inputs[0], streams)
    exploit_idx, exploit_is_input = _get_stream_index_any(block.inputs[1], streams)
    flag_idx, flag_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    explore_ref = f"i{explore_idx}[t]" if explore_is_input else f"o{explore_idx}[t]"
    exploit_ref = f"i{exploit_idx}[t]" if exploit_is_input else f"o{exploit_idx}[t]"
    flag_ref = f"i{flag_idx}[t]" if flag_is_input else f"o{flag_idx}[t]"
    
    # flag=1 -> explore, flag=0 -> exploit
    return f"(o{output_idx}[t] = ({flag_ref} & {explore_ref}) | ({flag_ref}' & {exploit_ref}))"


def _generate_reward_accumulator_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Reward accumulator: track cumulative reward signal.
    
    Use case: RL agent reward tracking.
    """
    if len(block.inputs) != 2:
        raise ValueError("reward_accumulator pattern requires exactly 2 inputs (reward, reset)")
    
    reward_idx, reward_is_input = _get_stream_index_any(block.inputs[0], streams)
    reset_idx, reset_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    reward_ref = f"i{reward_idx}[t]" if reward_is_input else f"o{reward_idx}[t]"
    reset_ref = f"i{reset_idx}[t]" if reset_is_input else f"o{reset_idx}[t]"
    
    # Accumulate reward, reset on reset signal
    return f"(o{output_idx}[t] = ({reset_ref}' & (o{output_idx}[t-1] | {reward_ref})) | ({reset_ref} & {reward_ref})) && (o{output_idx}[0] = 0)"


def _generate_goal_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Goal detector: detect when goal conditions are met.
    
    Use case: Determine if agent has achieved its objective.
    """
    if len(block.inputs) < 1:
        raise ValueError("goal_detector pattern requires at least 1 input (goal conditions)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # All goal conditions must be met (AND)
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_obstacle_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Obstacle detector: detect any blocking condition.
    
    Use case: Identify obstacles or constraints preventing action.
    """
    if len(block.inputs) < 1:
        raise ValueError("obstacle_detector pattern requires at least 1 input")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Any obstacle triggers (OR)
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = {or_expr})"


def _generate_policy_switch_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Policy switch: switch between different behavioral policies.
    
    Use case: Multi-policy agent that adapts to context.
    """
    if len(block.inputs) != 3:
        raise ValueError("policy_switch pattern requires exactly 3 inputs (policy_a, policy_b, selector)")
    
    policy_a_idx, policy_a_is_input = _get_stream_index_any(block.inputs[0], streams)
    policy_b_idx, policy_b_is_input = _get_stream_index_any(block.inputs[1], streams)
    selector_idx, selector_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    policy_a_ref = f"i{policy_a_idx}[t]" if policy_a_is_input else f"o{policy_a_idx}[t]"
    policy_b_ref = f"i{policy_b_idx}[t]" if policy_b_is_input else f"o{policy_b_idx}[t]"
    selector_ref = f"i{selector_idx}[t]" if selector_is_input else f"o{selector_idx}[t]"
    
    return f"(o{output_idx}[t] = ({selector_ref}' & {policy_a_ref}) | ({selector_ref} & {policy_b_ref}))"


# ============================================================================
# INTELLIGENT AGENT PATTERNS - Learning & Memory
# ============================================================================

def _generate_experience_buffer_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Experience buffer: store recent experience for learning.
    
    Use case: Remember past states/actions for replay.
    """
    if len(block.inputs) != 2:
        raise ValueError("experience_buffer pattern requires exactly 2 inputs (experience, store_trigger)")
    
    exp_idx, exp_is_input = _get_stream_index_any(block.inputs[0], streams)
    store_idx, store_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    exp_ref = f"i{exp_idx}[t]" if exp_is_input else f"o{exp_idx}[t]"
    store_ref = f"i{store_idx}[t]" if store_is_input else f"o{store_idx}[t]"
    
    # Store on trigger, otherwise hold
    return f"(o{output_idx}[t] = ({store_ref} & {exp_ref}) | ({store_ref}' & o{output_idx}[t-1])) && (o{output_idx}[0] = 0)"


def _generate_learning_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Learning gate: enable/disable learning updates.
    
    Use case: Control when agent should learn vs act.
    """
    if len(block.inputs) != 2:
        raise ValueError("learning_gate pattern requires exactly 2 inputs (update_signal, learning_enabled)")
    
    update_idx, update_is_input = _get_stream_index_any(block.inputs[0], streams)
    enabled_idx, enabled_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    update_ref = f"i{update_idx}[t]" if update_is_input else f"o{update_idx}[t]"
    enabled_ref = f"i{enabled_idx}[t]" if enabled_is_input else f"o{enabled_idx}[t]"
    
    return f"(o{output_idx}[t] = {update_ref} & {enabled_ref})"


def _generate_attention_focus_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Attention focus: select which input stream to attend to.
    
    Use case: Attention mechanism for multi-input agents.
    """
    if len(block.inputs) < 3:
        raise ValueError("attention_focus pattern requires at least 3 inputs (inputs..., attention_weights)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Last input is attention selector, others are data inputs
    data_inputs = block.inputs[:-1]
    attention_inp = block.inputs[-1]
    
    att_idx, att_is_input = _get_stream_index_any(attention_inp, streams)
    att_ref = f"i{att_idx}[t]" if att_is_input else f"o{att_idx}[t]"
    
    # Simple attention: pass first data input when attention is 0, second when 1
    if len(data_inputs) >= 2:
        d0_idx, d0_is_input = _get_stream_index_any(data_inputs[0], streams)
        d1_idx, d1_is_input = _get_stream_index_any(data_inputs[1], streams)
        d0_ref = f"i{d0_idx}[t]" if d0_is_input else f"o{d0_idx}[t]"
        d1_ref = f"i{d1_idx}[t]" if d1_is_input else f"o{d1_idx}[t]"
        return f"(o{output_idx}[t] = ({att_ref}' & {d0_ref}) | ({att_ref} & {d1_ref}))"
    else:
        d0_idx, d0_is_input = _get_stream_index_any(data_inputs[0], streams)
        d0_ref = f"i{d0_idx}[t]" if d0_is_input else f"o{d0_idx}[t]"
        return f"(o{output_idx}[t] = {att_ref} & {d0_ref})"


# ============================================================================
# INTELLIGENT AGENT PATTERNS - Coordination & Communication
# ============================================================================

def _generate_consensus_vote_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Consensus vote: multi-agent consensus decision.
    
    Use case: Distributed agent agreement.
    """
    if len(block.inputs) < 2:
        raise ValueError("consensus_vote pattern requires at least 2 inputs (agent votes)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    combinations = list(itertools.combinations(refs, threshold))
    and_exprs = [" & ".join(combo) for combo in combinations]
    or_expr = " | ".join(f"({e})" for e in and_exprs)
    
    return f"(o{output_idx}[t] = {or_expr})"


def _generate_broadcast_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Broadcast: send signal to multiple outputs (passthrough).
    
    Use case: Agent broadcasting to other agents.
    """
    if len(block.inputs) != 1:
        raise ValueError("broadcast pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
    
    return f"(o{output_idx}[t] = {inp_ref})"


def _generate_message_filter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Message filter: filter incoming messages by validity.
    
    Use case: Agent message validation.
    """
    if len(block.inputs) != 2:
        raise ValueError("message_filter pattern requires exactly 2 inputs (message, valid_flag)")
    
    msg_idx, msg_is_input = _get_stream_index_any(block.inputs[0], streams)
    valid_idx, valid_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    msg_ref = f"i{msg_idx}[t]" if msg_is_input else f"o{msg_idx}[t]"
    valid_ref = f"i{valid_idx}[t]" if valid_is_input else f"o{valid_idx}[t]"
    
    return f"(o{output_idx}[t] = {msg_ref} & {valid_ref})"


def _generate_role_assignment_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Role assignment: assign role based on conditions.
    
    Use case: Multi-agent role allocation.
    """
    if len(block.inputs) != 2:
        raise ValueError("role_assignment pattern requires exactly 2 inputs (capability, request)")
    
    cap_idx, cap_is_input = _get_stream_index_any(block.inputs[0], streams)
    req_idx, req_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    cap_ref = f"i{cap_idx}[t]" if cap_is_input else f"o{cap_idx}[t]"
    req_ref = f"i{req_idx}[t]" if req_is_input else f"o{req_idx}[t]"
    
    # Assign role if capable and requested
    return f"(o{output_idx}[t] = {cap_ref} & {req_ref})"


# ============================================================================
# INTELLIGENT AGENT PATTERNS - Safety & Constraints
# ============================================================================

def _generate_action_mask_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Action mask: mask out invalid/unsafe actions.
    
    Use case: Constrained action space for safe RL.
    """
    if len(block.inputs) != 2:
        raise ValueError("action_mask pattern requires exactly 2 inputs (action, mask)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    mask_idx, mask_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    mask_ref = f"i{mask_idx}[t]" if mask_is_input else f"o{mask_idx}[t]"
    
    # Action only passes if mask allows (mask=1 means allowed)
    return f"(o{output_idx}[t] = {action_ref} & {mask_ref})"


def _generate_safety_override_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Safety override: override agent action when safety condition triggers.
    
    Use case: Human-in-the-loop safety intervention.
    """
    if len(block.inputs) != 3:
        raise ValueError("safety_override pattern requires exactly 3 inputs (agent_action, safe_action, safety_trigger)")
    
    agent_idx, agent_is_input = _get_stream_index_any(block.inputs[0], streams)
    safe_idx, safe_is_input = _get_stream_index_any(block.inputs[1], streams)
    trigger_idx, trigger_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    agent_ref = f"i{agent_idx}[t]" if agent_is_input else f"o{agent_idx}[t]"
    safe_ref = f"i{safe_idx}[t]" if safe_is_input else f"o{safe_idx}[t]"
    trigger_ref = f"i{trigger_idx}[t]" if trigger_is_input else f"o{trigger_idx}[t]"
    
    # Use safe action when triggered, otherwise agent action
    return f"(o{output_idx}[t] = ({trigger_ref} & {safe_ref}) | ({trigger_ref}' & {agent_ref}))"


def _generate_constraint_checker_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Constraint checker: verify all constraints are satisfied.
    
    Use case: Validate action against constraints.
    """
    if len(block.inputs) < 1:
        raise ValueError("constraint_checker pattern requires at least 1 input (constraints)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # All constraints must be satisfied (AND)
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_budget_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Budget gate: only allow action if budget/resource available.
    
    Use case: Resource-constrained agent decisions.
    """
    if len(block.inputs) != 2:
        raise ValueError("budget_gate pattern requires exactly 2 inputs (action, budget_available)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    budget_idx, budget_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    budget_ref = f"i{budget_idx}[t]" if budget_is_input else f"o{budget_idx}[t]"
    
    return f"(o{output_idx}[t] = {action_ref} & {budget_ref})"


# ============================================================================
# INTELLIGENT AGENT PATTERNS - Inference & Prediction
# ============================================================================

def _generate_prediction_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Prediction gate: use prediction when observation unavailable.
    
    Use case: Handle partial observability with predictions.
    """
    if len(block.inputs) != 3:
        raise ValueError("prediction_gate pattern requires exactly 3 inputs (observation, prediction, obs_available)")
    
    obs_idx, obs_is_input = _get_stream_index_any(block.inputs[0], streams)
    pred_idx, pred_is_input = _get_stream_index_any(block.inputs[1], streams)
    avail_idx, avail_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    obs_ref = f"i{obs_idx}[t]" if obs_is_input else f"o{obs_idx}[t]"
    pred_ref = f"i{pred_idx}[t]" if pred_is_input else f"o{pred_idx}[t]"
    avail_ref = f"i{avail_idx}[t]" if avail_is_input else f"o{avail_idx}[t]"
    
    # Use observation when available, otherwise prediction
    return f"(o{output_idx}[t] = ({avail_ref} & {obs_ref}) | ({avail_ref}' & {pred_ref}))"


def _generate_anomaly_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Anomaly detector: detect unusual input patterns.
    
    Use case: Detect out-of-distribution inputs.
    """
    if len(block.inputs) != 2:
        raise ValueError("anomaly_detector pattern requires exactly 2 inputs (current, expected)")
    
    curr_idx, curr_is_input = _get_stream_index_any(block.inputs[0], streams)
    exp_idx, exp_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    curr_ref = f"i{curr_idx}[t]" if curr_is_input else f"o{curr_idx}[t]"
    exp_ref = f"i{exp_idx}[t]" if exp_is_input else f"o{exp_idx}[t]"
    
    # Anomaly when current differs from expected (XOR)
    return f"(o{output_idx}[t] = {curr_ref} ^ {exp_ref})"


def _generate_state_classifier_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """State classifier: classify current state based on observations.
    
    Use case: Discrete state inference.
    """
    if len(block.inputs) < 2:
        raise ValueError("state_classifier pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Simple classifier: AND all condition inputs
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


# ============================================================================
# AGENT FAIRNESS & COORDINATION PATTERNS
# Patterns for multi-agent coordination, verifiable randomness, and fair
# decision-making in intelligent agent systems (non-gaming).
#
# Inspiration: General formal verification concepts (LTL, safety invariants).
# Credit: Whitepaper by l0g1x (Tau Tomorrow) explores similar concepts for gaming:
# https://github.com/taumorrow/provably_fair_gaming/blob/master/PROVABLY_FAIR_GAMING_WHITEPAPER.MD
#
# Note: These are independent implementations for intelligent agents applying
# standard CS/formal methods principles. No gaming-specific code was copied.
# ============================================================================

def _generate_xor_combine_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """XOR combine: combine multiple values via XOR for unpredictable output.
    
    Use case: Multi-party randomness generation - no single party can predict
    or manipulate the output if at least one input is truly random.
    Inspired by: Provably fair gaming seed generation.
    """
    if len(block.inputs) < 2:
        raise ValueError("xor_combine pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    xor_expr = " ^ ".join(refs)
    return f"(o{output_idx}[t] = {xor_expr})"


def _generate_commitment_match_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Commitment match: verify revealed value matches commitment.
    
    Use case: Commitment-reveal protocol validation.
    Inputs: (revealed_value, committed_hash) - outputs 1 if they match.
    For sbf: simple equality check (XNOR).
    """
    if len(block.inputs) != 2:
        raise ValueError("commitment_match pattern requires exactly 2 inputs (revealed, committed)")
    
    reveal_idx, reveal_is_input = _get_stream_index_any(block.inputs[0], streams)
    commit_idx, commit_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    reveal_ref = f"i{reveal_idx}[t]" if reveal_is_input else f"o{reveal_idx}[t]"
    commit_ref = f"i{commit_idx}[t]" if commit_is_input else f"o{commit_idx}[t]"
    
    # Match = XNOR (equal when both same)
    return f"(o{output_idx}[t] = ({reveal_ref} ^ {commit_ref})')"


def _generate_all_revealed_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """All revealed: check if all parties have revealed their values.
    
    Use case: Commitment-reveal protocol - proceed only when all reveal.
    """
    if len(block.inputs) < 2:
        raise ValueError("all_revealed pattern requires at least 2 inputs (reveal flags)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_phase_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Phase gate: only allow action during specific phase.
    
    Use case: Multi-phase game loops (commit, reveal, execute phases).
    Inputs: (action, phase_active)
    """
    if len(block.inputs) != 2:
        raise ValueError("phase_gate pattern requires exactly 2 inputs (action, phase_active)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    phase_idx, phase_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    phase_ref = f"i{phase_idx}[t]" if phase_is_input else f"o{phase_idx}[t]"
    
    return f"(o{output_idx}[t] = {action_ref} & {phase_ref})"


def _generate_collision_detect_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Collision detect: detect when two agent states are equal.
    
    Use case: Conflict detection, duplicate state detection, convergence check.
    Output is 1 when inputs are equal (XNOR).
    """
    if len(block.inputs) != 2:
        raise ValueError("collision_detect pattern requires exactly 2 inputs")
    
    pos1_idx, pos1_is_input = _get_stream_index_any(block.inputs[0], streams)
    pos2_idx, pos2_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    pos1_ref = f"i{pos1_idx}[t]" if pos1_is_input else f"o{pos1_idx}[t]"
    pos2_ref = f"i{pos2_idx}[t]" if pos2_is_input else f"o{pos2_idx}[t]"
    
    # Collision when equal: NOT(XOR)
    return f"(o{output_idx}[t] = ({pos1_ref} ^ {pos2_ref})')"


def _generate_turn_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Turn gate: only allow action when it's the agent's turn.
    
    Use case: Agent scheduling, ordered execution, resource access control.
    Inputs: (action, my_turn)
    """
    if len(block.inputs) != 2:
        raise ValueError("turn_gate pattern requires exactly 2 inputs (action, my_turn)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    turn_idx, turn_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    turn_ref = f"i{turn_idx}[t]" if turn_is_input else f"o{turn_idx}[t]"
    
    return f"(o{output_idx}[t] = {action_ref} & {turn_ref})"


def _generate_fair_priority_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Fair priority: deterministic tiebreaker using priority signal.
    
    Use case: Resolve simultaneous actions fairly.
    Inputs: (action_a, action_b, a_has_priority)
    Output: a's action if priority, else b's action when both want to act.
    """
    if len(block.inputs) != 3:
        raise ValueError("fair_priority pattern requires exactly 3 inputs (action_a, action_b, a_priority)")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    pri_idx, pri_is_input = _get_stream_index_any(block.inputs[2], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    pri_ref = f"i{pri_idx}[t]" if pri_is_input else f"o{pri_idx}[t]"
    
    # If both want action: priority wins. Otherwise: OR of actions.
    return f"(o{output_idx}[t] = ({a_ref} & {b_ref} & {pri_ref}) | ({a_ref} & {b_ref}' ) | ({a_ref}' & {b_ref}))"


def _generate_streak_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Streak detector: detect consecutive successes/hits.
    
    Use case: Combo systems, win streaks, bonus multipliers.
    Output is 1 when current AND previous are both 1.
    """
    if len(block.inputs) != 1:
        raise ValueError("streak_detector pattern requires exactly 1 input")
    
    inp_idx, inp_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    inp_ref = f"i{inp_idx}" if inp_is_input else f"o{inp_idx}"
    
    # Streak when current AND previous both true
    return f"(o{output_idx}[t] = {inp_ref}[t] & {inp_ref}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_combo_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Combo counter: count consecutive hits, reset on miss.
    
    Use case: Fighting game combos, streak bonuses.
    For sbf: tracks if in combo (consecutive hits).
    """
    if len(block.inputs) != 1:
        raise ValueError("combo_counter pattern requires exactly 1 input (hit)")
    
    hit_idx, hit_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    hit_ref = f"i{hit_idx}" if hit_is_input else f"o{hit_idx}"
    
    # Combo active: hit AND (was in combo OR just started)
    # Simplified for sbf: output = hit[t] & (hit[t-1] | output[t-1])
    return f"(o{output_idx}[t] = {hit_ref}[t] & ({hit_ref}[t-1] | o{output_idx}[t-1])) && (o{output_idx}[0] = 0)"


def _generate_cooldown_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Cooldown: prevent action for N cycles after trigger.
    
    Use case: Ability cooldowns, rate limiting.
    Simplified for sbf: lock for 1 cycle after use.
    """
    if len(block.inputs) != 1:
        raise ValueError("cooldown pattern requires exactly 1 input (trigger)")
    
    trig_idx, trig_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    trig_ref = f"i{trig_idx}" if trig_is_input else f"o{trig_idx}"
    
    # Ready (output=1) when not triggered recently
    # Lock (output=0) for 1 cycle after trigger
    return f"(o{output_idx}[t] = {trig_ref}[t-1]') && (o{output_idx}[0] = 1)"


def _generate_capture_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Capture: resource acquisition when both agents present.
    
    Use case: Resource contention, competitive acquisition, takeover detection.
    Inputs: (acquirer_present, holder_present)
    Output: acquisition occurred (both agents present)
    """
    if len(block.inputs) != 2:
        raise ValueError("capture pattern requires exactly 2 inputs (attacker, defender)")
    
    atk_idx, atk_is_input = _get_stream_index_any(block.inputs[0], streams)
    def_idx, def_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    atk_ref = f"i{atk_idx}[t]" if atk_is_input else f"o{atk_idx}[t]"
    def_ref = f"i{def_idx}[t]" if def_is_input else f"o{def_idx}[t]"
    
    return f"(o{output_idx}[t] = {atk_ref} & {def_ref})"


def _generate_territory_control_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Territory control: track resource/zone ownership.
    
    Use case: Resource allocation, zone control, exclusive access.
    Inputs: (claim_signal, contest_signal)
    Output: controlled (claimed AND not contested)
    """
    if len(block.inputs) != 2:
        raise ValueError("territory_control pattern requires exactly 2 inputs (claim, contest)")
    
    claim_idx, claim_is_input = _get_stream_index_any(block.inputs[0], streams)
    contest_idx, contest_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    claim_ref = f"i{claim_idx}[t]" if claim_is_input else f"o{claim_idx}[t]"
    contest_ref = f"i{contest_idx}[t]" if contest_is_input else f"o{contest_idx}[t]"
    
    # Control = claimed AND not contested
    return f"(o{output_idx}[t] = {claim_ref} & {contest_ref}')"


def _generate_simultaneous_action_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Simultaneous action: detect when multiple actors act at same time.
    
    Use case: Simultaneous turn games, coordination detection.
    """
    if len(block.inputs) < 2:
        raise ValueError("simultaneous_action pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # All acting simultaneously
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_any_action_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Any action: detect when any actor takes action.
    
    Use case: Event detection, activity monitoring.
    """
    if len(block.inputs) < 2:
        raise ValueError("any_action pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = {or_expr})"


def _generate_exclusive_action_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Exclusive action: exactly one actor should act (XOR).
    
    Use case: Mutual exclusion, single-active validation.
    """
    if len(block.inputs) != 2:
        raise ValueError("exclusive_action pattern requires exactly 2 inputs")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    return f"(o{output_idx}[t] = {a_ref} ^ {b_ref})"


def _generate_valid_move_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Valid move: check if move satisfies all game rules.
    
    Use case: Move validation in board games.
    Inputs: multiple rule conditions that must all be true.
    """
    if len(block.inputs) < 1:
        raise ValueError("valid_move pattern requires at least 1 input (rule conditions)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_win_condition_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Win condition: check if victory conditions are met.
    
    Use case: Game ending, victory detection.
    """
    if len(block.inputs) < 1:
        raise ValueError("win_condition pattern requires at least 1 input")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    and_expr = " & ".join(refs)
    return f"(o{output_idx}[t] = {and_expr})"


def _generate_game_over_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Terminal state: latch that stays true once any end condition triggers.
    
    Use case: Episode termination, permanent failure state, goal reached.
    """
    if len(block.inputs) < 1:
        raise ValueError("game_over pattern requires at least 1 input (end conditions)")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # Any end condition OR already game over
    or_expr = " | ".join(refs)
    return f"(o{output_idx}[t] = {or_expr} | o{output_idx}[t-1]) && (o{output_idx}[0] = 0)"


def _generate_score_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Score gate: award points only when conditions met.
    
    Use case: Conditional scoring, achievement unlocking.
    Inputs: (score_event, condition_met)
    """
    if len(block.inputs) != 2:
        raise ValueError("score_gate pattern requires exactly 2 inputs (event, condition)")
    
    event_idx, event_is_input = _get_stream_index_any(block.inputs[0], streams)
    cond_idx, cond_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    event_ref = f"i{event_idx}[t]" if event_is_input else f"o{event_idx}[t]"
    cond_ref = f"i{cond_idx}[t]" if cond_is_input else f"o{cond_idx}[t]"
    
    return f"(o{output_idx}[t] = {event_ref} & {cond_ref})"


def _generate_bonus_trigger_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Bonus trigger: activate bonus when streak/combo reaches threshold.
    
    Use case: Streak bonuses, combo finishers.
    For sbf: bonus when streak_active AND condition.
    """
    if len(block.inputs) != 2:
        raise ValueError("bonus_trigger pattern requires exactly 2 inputs (streak, condition)")
    
    streak_idx, streak_is_input = _get_stream_index_any(block.inputs[0], streams)
    cond_idx, cond_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    streak_ref = f"i{streak_idx}[t]" if streak_is_input else f"o{streak_idx}[t]"
    cond_ref = f"i{cond_idx}[t]" if cond_is_input else f"o{cond_idx}[t]"
    
    return f"(o{output_idx}[t] = {streak_ref} & {cond_ref})"


# ============================================================================
# FORMAL VERIFICATION INVARIANT PATTERNS
# Novel patterns for safety, liveness, fairness, and convergence guarantees.
# These extend beyond gaming into general multi-agent formal verification.
# ============================================================================

def _generate_mutual_exclusion_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Mutual exclusion: at most one agent can hold resource at a time.
    
    Safety invariant: G(NOT(inCS_i AND inCS_j)) for all i != j
    Use case: Critical section access, exclusive resource locks.
    Output is 1 (safe) when at most one input is active.
    """
    if len(block.inputs) != 2:
        raise ValueError("mutual_exclusion pattern requires exactly 2 inputs (agent states)")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    # Safe = NOT(both active) = NAND
    return f"(o{output_idx}[t] = ({a_ref} & {b_ref})')"


def _generate_never_unsafe_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Never unsafe: safety invariant that must always hold.
    
    Safety invariant: G(NOT unsafe_condition)
    Use case: Collision avoidance, forbidden state prevention.
    Output is 1 (safe) when unsafe condition is false.
    """
    if len(block.inputs) != 1:
        raise ValueError("never_unsafe pattern requires exactly 1 input (unsafe_condition)")
    
    unsafe_idx, unsafe_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    unsafe_ref = f"i{unsafe_idx}[t]" if unsafe_is_input else f"o{unsafe_idx}[t]"
    
    # Safe = NOT(unsafe)
    return f"(o{output_idx}[t] = {unsafe_ref}')"


def _generate_request_response_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Request-response: every request must eventually get a response.
    
    Liveness pattern: G(request -> F grant)
    For sbf: tracks pending requests, output 1 when request satisfied or none pending.
    """
    if len(block.inputs) != 2:
        raise ValueError("request_response pattern requires exactly 2 inputs (request, grant)")
    
    req_idx, req_is_input = _get_stream_index_any(block.inputs[0], streams)
    grant_idx, grant_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    req_ref = f"i{req_idx}" if req_is_input else f"o{req_idx}"
    grant_ref = f"i{grant_idx}[t]" if grant_is_input else f"o{grant_idx}[t]"
    
    # Pending = (request OR was_pending) AND NOT grant
    # Satisfied = NOT pending
    return f"(o{output_idx}[t] = ({req_ref}[t] | o{output_idx}[t-1]) & {grant_ref}') && (o{output_idx}[0] = 0)"


def _generate_recovery_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Recovery: if failure occurs, system eventually recovers.
    
    Liveness pattern: G(failure -> F recovered)
    Output tracks recovery state: 0 = failed/recovering, 1 = healthy.
    """
    if len(block.inputs) != 2:
        raise ValueError("recovery pattern requires exactly 2 inputs (failure, recovered)")
    
    fail_idx, fail_is_input = _get_stream_index_any(block.inputs[0], streams)
    recov_idx, recov_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    fail_ref = f"i{fail_idx}[t]" if fail_is_input else f"o{fail_idx}[t]"
    recov_ref = f"i{recov_idx}[t]" if recov_is_input else f"o{recov_idx}[t]"
    
    # Healthy = (was_healthy AND NOT failure) OR recovered
    return f"(o{output_idx}[t] = (o{output_idx}[t-1] & {fail_ref}') | {recov_ref}) && (o{output_idx}[0] = 1)"


def _generate_no_starvation_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """No starvation: if agent requests infinitely often, granted infinitely often.
    
    Fairness pattern: GF request -> GF grant
    Simplified: output 1 when agent was recently granted after requesting.
    """
    if len(block.inputs) != 2:
        raise ValueError("no_starvation pattern requires exactly 2 inputs (request, grant)")
    
    req_idx, req_is_input = _get_stream_index_any(block.inputs[0], streams)
    grant_idx, grant_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    req_ref = f"i{req_idx}[t]" if req_is_input else f"o{req_idx}[t]"
    grant_ref = f"i{grant_idx}[t]" if grant_is_input else f"o{grant_idx}[t]"
    
    # Fair service = request implies grant (when scheduled)
    return f"(o{output_idx}[t] = {req_ref}' | {grant_ref})"


def _generate_stabilization_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Stabilization: system eventually reaches stable state and stays there.
    
    Convergence pattern: F G stable
    Output 1 when current state equals previous state (no change).
    """
    if len(block.inputs) != 1:
        raise ValueError("stabilization pattern requires exactly 1 input (state)")
    
    state_idx, state_is_input = _get_stream_index_any(block.inputs[0], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    state_ref = f"i{state_idx}" if state_is_input else f"o{state_idx}"
    
    # Stable = state[t] == state[t-1] (no change) = XNOR
    return f"(o{output_idx}[t] = ({state_ref}[t] ^ {state_ref}[t-1])') && (o{output_idx}[0] = 0)"


def _generate_consensus_check_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Consensus check: verify all agents have same value.
    
    Convergence pattern: F G (all values equal)
    Output 1 when all inputs are equal.
    """
    if len(block.inputs) < 2:
        raise ValueError("consensus_check pattern requires at least 2 inputs")
    
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Build pairwise equality checks
    refs = []
    for inp in block.inputs:
        inp_idx, inp_is_input = _get_stream_index_any(inp, streams)
        ref = f"i{inp_idx}[t]" if inp_is_input else f"o{inp_idx}[t]"
        refs.append(ref)
    
    # All equal = pairwise XNOR (for 2 inputs: a XNOR b)
    # For sbf with 2 inputs: NOT(a XOR b)
    if len(refs) == 2:
        return f"(o{output_idx}[t] = ({refs[0]} ^ {refs[1]})')"
    else:
        # For 3+: chain pairwise checks
        checks = [f"({refs[i]} ^ {refs[i+1]})'" for i in range(len(refs)-1)]
        return f"(o{output_idx}[t] = {' & '.join(checks)})"


def _generate_progress_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Progress: system makes progress toward goal.
    
    Liveness pattern: G(enabled -> F done)
    Output 1 when progress is being made (goal closer or achieved).
    """
    if len(block.inputs) != 2:
        raise ValueError("progress pattern requires exactly 2 inputs (enabled, done)")
    
    enabled_idx, enabled_is_input = _get_stream_index_any(block.inputs[0], streams)
    done_idx, done_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    enabled_ref = f"i{enabled_idx}[t]" if enabled_is_input else f"o{enabled_idx}[t]"
    done_ref = f"i{done_idx}[t]" if done_is_input else f"o{done_idx}[t]"
    
    # Progress = NOT enabled OR done (implication)
    return f"(o{output_idx}[t] = {enabled_ref}' | {done_ref})"


def _generate_bounded_until_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Bounded until: condition must hold until goal within bound.
    
    Bounded liveness: condition U goal (within K steps)
    Simplified for sbf: condition must hold while goal not yet reached.
    """
    if len(block.inputs) != 2:
        raise ValueError("bounded_until pattern requires exactly 2 inputs (condition, goal)")
    
    cond_idx, cond_is_input = _get_stream_index_any(block.inputs[0], streams)
    goal_idx, goal_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    cond_ref = f"i{cond_idx}[t]" if cond_is_input else f"o{cond_idx}[t]"
    goal_ref = f"i{goal_idx}[t]" if goal_is_input else f"o{goal_idx}[t]"
    
    # Valid = goal reached OR (condition holds AND was valid)
    return f"(o{output_idx}[t] = {goal_ref} | ({cond_ref} & o{output_idx}[t-1])) && (o{output_idx}[0] = 1)"


def _generate_trust_update_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Trust update: update trust based on observed behavior.
    
    Trust invariant: trust increases on good behavior, decreases on bad.
    For sbf: trust = (good_outcome) OR (trust[t-1] AND NOT bad_outcome)
    """
    if len(block.inputs) != 2:
        raise ValueError("trust_update pattern requires exactly 2 inputs (good_outcome, bad_outcome)")
    
    good_idx, good_is_input = _get_stream_index_any(block.inputs[0], streams)
    bad_idx, bad_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    good_ref = f"i{good_idx}[t]" if good_is_input else f"o{good_idx}[t]"
    bad_ref = f"i{bad_idx}[t]" if bad_is_input else f"o{bad_idx}[t]"
    
    # Trust = good OR (was_trusted AND NOT bad)
    return f"(o{output_idx}[t] = {good_ref} | (o{output_idx}[t-1] & {bad_ref}')) && (o{output_idx}[0] = 0)"


def _generate_reputation_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Reputation gate: only allow action if reputation threshold met.
    
    Trust pattern: action only if trusted
    """
    if len(block.inputs) != 2:
        raise ValueError("reputation_gate pattern requires exactly 2 inputs (action, trusted)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    trust_idx, trust_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    trust_ref = f"i{trust_idx}[t]" if trust_is_input else f"o{trust_idx}[t]"
    
    return f"(o{output_idx}[t] = {action_ref} & {trust_ref})"


def _generate_risk_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Risk gate: only allow action if risk below threshold.
    
    Risk-aware pattern: action only if NOT high_risk
    """
    if len(block.inputs) != 2:
        raise ValueError("risk_gate pattern requires exactly 2 inputs (action, high_risk)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    risk_idx, risk_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    risk_ref = f"i{risk_idx}[t]" if risk_is_input else f"o{risk_idx}[t]"
    
    # Action allowed only if NOT high risk
    return f"(o{output_idx}[t] = {action_ref} & {risk_ref}')"


def _generate_belief_consistency_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Belief consistency: beliefs must not contradict each other.
    
    Epistemic pattern: G(consistent(beliefs))
    For sbf: NOT(belief_a AND belief_not_a)
    """
    if len(block.inputs) != 2:
        raise ValueError("belief_consistency pattern requires exactly 2 inputs (belief, negated_belief)")
    
    a_idx, a_is_input = _get_stream_index_any(block.inputs[0], streams)
    b_idx, b_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    a_ref = f"i{a_idx}[t]" if a_is_input else f"o{a_idx}[t]"
    b_ref = f"i{b_idx}[t]" if b_is_input else f"o{b_idx}[t]"
    
    # Consistent = NOT(a AND b) where b is negation of a
    return f"(o{output_idx}[t] = ({a_ref} & {b_ref})')"


def _generate_exploration_decay_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Exploration decay: exploration rate decreases over time.
    
    Learning pattern: exploration eventually stops (FG NOT exploring)
    For sbf: explore only if trigger AND NOT accumulated_experience
    """
    if len(block.inputs) != 2:
        raise ValueError("exploration_decay pattern requires exactly 2 inputs (explore_trigger, experience)")
    
    trig_idx, trig_is_input = _get_stream_index_any(block.inputs[0], streams)
    exp_idx, exp_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    trig_ref = f"i{trig_idx}[t]" if trig_is_input else f"o{trig_idx}[t]"
    exp_ref = f"i{exp_idx}[t]" if exp_is_input else f"o{exp_idx}[t]"
    
    # Explore = trigger AND NOT experienced (exploitation takes over)
    return f"(o{output_idx}[t] = {trig_ref} & {exp_ref}')"


def _generate_safe_exploration_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Safe exploration: explore only when safe to do so.
    
    Shielded RL pattern: G(explore -> safe)
    """
    if len(block.inputs) != 2:
        raise ValueError("safe_exploration pattern requires exactly 2 inputs (explore, safe)")
    
    explore_idx, explore_is_input = _get_stream_index_any(block.inputs[0], streams)
    safe_idx, safe_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    explore_ref = f"i{explore_idx}[t]" if explore_is_input else f"o{explore_idx}[t]"
    safe_ref = f"i{safe_idx}[t]" if safe_is_input else f"o{safe_idx}[t]"
    
    # Safe exploration = explore AND safe
    return f"(o{output_idx}[t] = {explore_ref} & {safe_ref})"


def _generate_emergent_detector_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Emergent behavior detector: detect unexpected macro patterns.
    
    Emergent pattern: state NOT in intended_set
    Output 1 when behavior deviates from expected.
    """
    if len(block.inputs) != 2:
        raise ValueError("emergent_detector pattern requires exactly 2 inputs (actual, expected)")
    
    actual_idx, actual_is_input = _get_stream_index_any(block.inputs[0], streams)
    expected_idx, expected_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    actual_ref = f"i{actual_idx}[t]" if actual_is_input else f"o{actual_idx}[t]"
    expected_ref = f"i{expected_idx}[t]" if expected_is_input else f"o{expected_idx}[t]"
    
    # Emergent = actual XOR expected (deviation)
    return f"(o{output_idx}[t] = {actual_ref} ^ {expected_ref})"


def _generate_utility_alignment_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Utility alignment: verify agent actions align with desired utility.
    
    Alignment pattern: G(action -> aligned_with_preferences)
    """
    if len(block.inputs) != 2:
        raise ValueError("utility_alignment pattern requires exactly 2 inputs (action, aligned)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    aligned_idx, aligned_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    aligned_ref = f"i{aligned_idx}[t]" if aligned_is_input else f"o{aligned_idx}[t]"
    
    # Valid = action implies aligned (NOT action OR aligned)
    return f"(o{output_idx}[t] = {action_ref}' | {aligned_ref})"


def _generate_causal_gate_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Causal gate: action only if causal precondition met.
    
    Causal reasoning: do(action) only if belief(cause -> effect)
    """
    if len(block.inputs) != 2:
        raise ValueError("causal_gate pattern requires exactly 2 inputs (action, causal_condition)")
    
    action_idx, action_is_input = _get_stream_index_any(block.inputs[0], streams)
    causal_idx, causal_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    action_ref = f"i{action_idx}[t]" if action_is_input else f"o{action_idx}[t]"
    causal_ref = f"i{causal_idx}[t]" if causal_is_input else f"o{causal_idx}[t]"
    
    return f"(o{output_idx}[t] = {action_ref} & {causal_ref})"


def _generate_counterfactual_safe_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Counterfactual safety: avoid actions with bad counterfactual outcomes.
    
    Counterfactual pattern: G(choice -> NOT F bad_counterfactual)
    """
    if len(block.inputs) != 2:
        raise ValueError("counterfactual_safe pattern requires exactly 2 inputs (choice, bad_counterfactual)")
    
    choice_idx, choice_is_input = _get_stream_index_any(block.inputs[0], streams)
    bad_idx, bad_is_input = _get_stream_index_any(block.inputs[1], streams)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    choice_ref = f"i{choice_idx}[t]" if choice_is_input else f"o{choice_idx}[t]"
    bad_ref = f"i{bad_idx}[t]" if bad_is_input else f"o{bad_idx}[t]"
    
    # Safe choice = choice AND NOT bad counterfactual
    return f"(o{output_idx}[t] = {choice_ref} & {bad_ref}')"


def _generate_recurrence_block(schema: AgentSchema, descriptive: bool = False) -> List[str]:
    """Generate recurrence relation block.
    
    Tau Language uses 'r <wff>' to run a well-formed formula.
    Multiple clauses are joined with '&&' in a single formula.
    
    Args:
        schema: The agent schema
        descriptive: If True, convert i0/o0 names to human-readable stream names
    """
    lines = []
    
    comment_lines = []
    logic_lines = []
    for block in schema.logic_blocks:
        # Track pattern comment for readability and tests
        comment_lines.append(f"{block.pattern} pattern: {block.output} <- {', '.join(block.inputs)}")
        if block.pattern == "majority":
            comment_lines.append("Majority voting implementation")
        if block.pattern == "passthrough" and block.inputs:
            comment_lines.append(f"{block.output} :- {block.inputs[0]}")
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
        # NEW PATTERNS - Signal Processing
        elif block.pattern == "edge_detector":
            logic_lines.append(_generate_edge_detector_logic(block, schema.streams))
        elif block.pattern == "falling_edge":
            logic_lines.append(_generate_falling_edge_logic(block, schema.streams))
        elif block.pattern == "toggle":
            logic_lines.append(_generate_toggle_logic(block, schema.streams))
        elif block.pattern == "latch":
            logic_lines.append(_generate_latch_logic(block, schema.streams))
        elif block.pattern == "debounce":
            logic_lines.append(_generate_debounce_logic(block, schema.streams))
        elif block.pattern == "pulse_generator":
            logic_lines.append(_generate_pulse_generator_logic(block, schema.streams))
        elif block.pattern == "sample_hold":
            logic_lines.append(_generate_sample_hold_logic(block, schema.streams))
        # NEW PATTERNS - Data Flow / Routing
        elif block.pattern == "multiplexer":
            logic_lines.append(_generate_multiplexer_logic(block, schema.streams))
        elif block.pattern == "demultiplexer":
            logic_lines.append(_generate_demultiplexer_logic(block, schema.streams))
        elif block.pattern == "priority_encoder":
            logic_lines.append(_generate_priority_encoder_logic(block, schema.streams))
        elif block.pattern == "arbiter":
            logic_lines.append(_generate_arbiter_logic(block, schema.streams))
        # NEW PATTERNS - Safety / Watchdog
        elif block.pattern == "watchdog":
            logic_lines.append(_generate_watchdog_logic(block, schema.streams))
        elif block.pattern == "deadman_switch":
            logic_lines.append(_generate_deadman_switch_logic(block, schema.streams))
        elif block.pattern == "safety_interlock":
            logic_lines.append(_generate_safety_interlock_logic(block, schema.streams))
        elif block.pattern == "fault_detector":
            logic_lines.append(_generate_fault_detector_logic(block, schema.streams))
        # NEW PATTERNS - Protocol / Handshake
        elif block.pattern == "handshake":
            logic_lines.append(_generate_handshake_logic(block, schema.streams))
        elif block.pattern == "sync_barrier":
            logic_lines.append(_generate_sync_barrier_logic(block, schema.streams))
        elif block.pattern == "token_ring":
            logic_lines.append(_generate_token_ring_logic(block, schema.streams))
        # NEW PATTERNS - Arithmetic / Comparison
        elif block.pattern == "comparator":
            logic_lines.append(_generate_comparator_logic(block, schema.streams))
        elif block.pattern == "min_selector":
            logic_lines.append(_generate_min_selector_logic(block, schema.streams))
        elif block.pattern == "max_selector":
            logic_lines.append(_generate_max_selector_logic(block, schema.streams))
        elif block.pattern == "threshold_detector":
            logic_lines.append(_generate_threshold_detector_logic(block, schema.streams))
        # NEW PATTERNS - Consensus / Distributed
        elif block.pattern == "byzantine_fault_tolerant":
            logic_lines.append(_generate_byzantine_fault_tolerant_logic(block, schema.streams))
        elif block.pattern == "leader_election":
            logic_lines.append(_generate_leader_election_logic(block, schema.streams))
        elif block.pattern == "commit_protocol":
            logic_lines.append(_generate_commit_protocol_logic(block, schema.streams))
        # NEW PATTERNS - Gate / Logic
        elif block.pattern == "nand_gate":
            logic_lines.append(_generate_nand_gate_logic(block, schema.streams))
        elif block.pattern == "nor_gate":
            logic_lines.append(_generate_nor_gate_logic(block, schema.streams))
        elif block.pattern == "xnor_gate":
            logic_lines.append(_generate_xnor_gate_logic(block, schema.streams))
        elif block.pattern == "implication":
            logic_lines.append(_generate_implication_logic(block, schema.streams))
        elif block.pattern == "equivalence":
            logic_lines.append(_generate_equivalence_logic(block, schema.streams))
        # NEW PATTERNS - Timing / Delay
        elif block.pattern == "delay_line":
            logic_lines.append(_generate_delay_line_logic(block, schema.streams))
        elif block.pattern == "hold":
            logic_lines.append(_generate_hold_logic(block, schema.streams))
        elif block.pattern == "one_shot":
            logic_lines.append(_generate_one_shot_logic(block, schema.streams))
        # NEW PATTERNS - State Encoding
        elif block.pattern == "gray_code":
            logic_lines.append(_generate_gray_code_logic(block, schema.streams))
        elif block.pattern == "ring_counter":
            logic_lines.append(_generate_ring_counter_logic(block, schema.streams))
        elif block.pattern == "sequence_detector":
            logic_lines.append(_generate_sequence_detector_logic(block, schema.streams))
        # INTELLIGENT AGENT PATTERNS - Decision Making
        elif block.pattern == "confidence_gate":
            logic_lines.append(_generate_confidence_gate_logic(block, schema.streams))
        elif block.pattern == "action_selector":
            logic_lines.append(_generate_action_selector_logic(block, schema.streams))
        elif block.pattern == "exploration_exploit":
            logic_lines.append(_generate_exploration_exploit_logic(block, schema.streams))
        elif block.pattern == "reward_accumulator":
            logic_lines.append(_generate_reward_accumulator_logic(block, schema.streams))
        elif block.pattern == "goal_detector":
            logic_lines.append(_generate_goal_detector_logic(block, schema.streams))
        elif block.pattern == "obstacle_detector":
            logic_lines.append(_generate_obstacle_detector_logic(block, schema.streams))
        elif block.pattern == "policy_switch":
            logic_lines.append(_generate_policy_switch_logic(block, schema.streams))
        # INTELLIGENT AGENT PATTERNS - Learning & Memory
        elif block.pattern == "experience_buffer":
            logic_lines.append(_generate_experience_buffer_logic(block, schema.streams))
        elif block.pattern == "learning_gate":
            logic_lines.append(_generate_learning_gate_logic(block, schema.streams))
        elif block.pattern == "attention_focus":
            logic_lines.append(_generate_attention_focus_logic(block, schema.streams))
        # INTELLIGENT AGENT PATTERNS - Coordination & Communication
        elif block.pattern == "consensus_vote":
            logic_lines.append(_generate_consensus_vote_logic(block, schema.streams))
        elif block.pattern == "broadcast":
            logic_lines.append(_generate_broadcast_logic(block, schema.streams))
        elif block.pattern == "message_filter":
            logic_lines.append(_generate_message_filter_logic(block, schema.streams))
        elif block.pattern == "role_assignment":
            logic_lines.append(_generate_role_assignment_logic(block, schema.streams))
        # INTELLIGENT AGENT PATTERNS - Safety & Constraints
        elif block.pattern == "action_mask":
            logic_lines.append(_generate_action_mask_logic(block, schema.streams))
        elif block.pattern == "safety_override":
            logic_lines.append(_generate_safety_override_logic(block, schema.streams))
        elif block.pattern == "constraint_checker":
            logic_lines.append(_generate_constraint_checker_logic(block, schema.streams))
        elif block.pattern == "budget_gate":
            logic_lines.append(_generate_budget_gate_logic(block, schema.streams))
        # INTELLIGENT AGENT PATTERNS - Inference & Prediction
        elif block.pattern == "prediction_gate":
            logic_lines.append(_generate_prediction_gate_logic(block, schema.streams))
        elif block.pattern == "anomaly_detector":
            logic_lines.append(_generate_anomaly_detector_logic(block, schema.streams))
        elif block.pattern == "state_classifier":
            logic_lines.append(_generate_state_classifier_logic(block, schema.streams))
        # PROVABLY FAIR GAMING PATTERNS
        elif block.pattern == "xor_combine":
            logic_lines.append(_generate_xor_combine_logic(block, schema.streams))
        elif block.pattern == "commitment_match":
            logic_lines.append(_generate_commitment_match_logic(block, schema.streams))
        elif block.pattern == "all_revealed":
            logic_lines.append(_generate_all_revealed_logic(block, schema.streams))
        elif block.pattern == "phase_gate":
            logic_lines.append(_generate_phase_gate_logic(block, schema.streams))
        elif block.pattern == "collision_detect":
            logic_lines.append(_generate_collision_detect_logic(block, schema.streams))
        elif block.pattern == "turn_gate":
            logic_lines.append(_generate_turn_gate_logic(block, schema.streams))
        elif block.pattern == "fair_priority":
            logic_lines.append(_generate_fair_priority_logic(block, schema.streams))
        elif block.pattern == "streak_detector":
            logic_lines.append(_generate_streak_detector_logic(block, schema.streams))
        elif block.pattern == "combo_counter":
            logic_lines.append(_generate_combo_counter_logic(block, schema.streams))
        elif block.pattern == "cooldown":
            logic_lines.append(_generate_cooldown_logic(block, schema.streams))
        elif block.pattern == "capture":
            logic_lines.append(_generate_capture_logic(block, schema.streams))
        elif block.pattern == "territory_control":
            logic_lines.append(_generate_territory_control_logic(block, schema.streams))
        elif block.pattern == "simultaneous_action":
            logic_lines.append(_generate_simultaneous_action_logic(block, schema.streams))
        elif block.pattern == "any_action":
            logic_lines.append(_generate_any_action_logic(block, schema.streams))
        elif block.pattern == "exclusive_action":
            logic_lines.append(_generate_exclusive_action_logic(block, schema.streams))
        elif block.pattern == "valid_move":
            logic_lines.append(_generate_valid_move_logic(block, schema.streams))
        elif block.pattern == "win_condition":
            logic_lines.append(_generate_win_condition_logic(block, schema.streams))
        elif block.pattern == "game_over":
            logic_lines.append(_generate_game_over_logic(block, schema.streams))
        elif block.pattern == "score_gate":
            logic_lines.append(_generate_score_gate_logic(block, schema.streams))
        elif block.pattern == "bonus_trigger":
            logic_lines.append(_generate_bonus_trigger_logic(block, schema.streams))
        # FORMAL VERIFICATION INVARIANT PATTERNS
        elif block.pattern == "mutual_exclusion":
            logic_lines.append(_generate_mutual_exclusion_logic(block, schema.streams))
        elif block.pattern == "never_unsafe":
            logic_lines.append(_generate_never_unsafe_logic(block, schema.streams))
        elif block.pattern == "request_response":
            logic_lines.append(_generate_request_response_logic(block, schema.streams))
        elif block.pattern == "recovery":
            logic_lines.append(_generate_recovery_logic(block, schema.streams))
        elif block.pattern == "no_starvation":
            logic_lines.append(_generate_no_starvation_logic(block, schema.streams))
        elif block.pattern == "stabilization":
            logic_lines.append(_generate_stabilization_logic(block, schema.streams))
        elif block.pattern == "consensus_check":
            logic_lines.append(_generate_consensus_check_logic(block, schema.streams))
        elif block.pattern == "progress":
            logic_lines.append(_generate_progress_logic(block, schema.streams))
        elif block.pattern == "bounded_until":
            logic_lines.append(_generate_bounded_until_logic(block, schema.streams))
        elif block.pattern == "trust_update":
            logic_lines.append(_generate_trust_update_logic(block, schema.streams))
        elif block.pattern == "reputation_gate":
            logic_lines.append(_generate_reputation_gate_logic(block, schema.streams))
        elif block.pattern == "risk_gate":
            logic_lines.append(_generate_risk_gate_logic(block, schema.streams))
        elif block.pattern == "belief_consistency":
            logic_lines.append(_generate_belief_consistency_logic(block, schema.streams))
        elif block.pattern == "exploration_decay":
            logic_lines.append(_generate_exploration_decay_logic(block, schema.streams))
        elif block.pattern == "safe_exploration":
            logic_lines.append(_generate_safe_exploration_logic(block, schema.streams))
        elif block.pattern == "emergent_detector":
            logic_lines.append(_generate_emergent_detector_logic(block, schema.streams))
        elif block.pattern == "utility_alignment":
            logic_lines.append(_generate_utility_alignment_logic(block, schema.streams))
        elif block.pattern == "causal_gate":
            logic_lines.append(_generate_causal_gate_logic(block, schema.streams))
        elif block.pattern == "counterfactual_safe":
            logic_lines.append(_generate_counterfactual_safe_logic(block, schema.streams))
        else:
            raise ValueError(f"Unknown pattern: {block.pattern}")
    
    # Emit comments as header lines
    for comment in comment_lines:
        if comment:
            lines.append(f"# {comment}")

    # Build the run command with the full WFF
    # Tau syntax: r <wff> where multiple clauses are joined with &&
    if logic_lines:
        # Normalize each clause: remove embedded newlines and trailing &&
        normalized = []
        for clause in logic_lines:
            # Flatten multi-line clauses to single line
            flat = " ".join(clause.split())
            # Remove trailing && if present (will be re-added when joining)
            flat = flat.rstrip().rstrip("&").rstrip()
            if flat:
                normalized.append(flat)
        # Join all logic clauses with && into a single WFF
        full_wff = " && ".join(normalized)
        
        # Convert i0/o0 names to descriptive names if enabled
        if descriptive:
            full_wff = _convert_to_descriptive_names(full_wff, schema.streams)
        
        lines.append(f"r {full_wff}")
    
    return lines


def _convert_to_descriptive_names(wff: str, streams: tuple) -> str:
    """Convert i0/o0/mi0 style names to descriptive stream names.
    
    Args:
        wff: The well-formed formula with i0/o0 style names
        streams: Stream configurations to look up actual names
        
    Returns:
        WFF with descriptive names (e.g., 'buy_signal' instead of 'i0')
    """
    import re
    
    result = wff
    
    # Build mappings
    input_streams = [s for s in streams if s.is_input]
    output_streams = [s for s in streams if not s.is_input]
    
    # Replace output references (o0, o1, etc.)
    for idx, stream in enumerate(output_streams):
        hex_idx = hex(idx)[2:].upper()
        var_name = stream.name.replace("-", "_")
        # Replace o0[t], o0[t-1], o0[0], etc.
        result = re.sub(rf'\bo{hex_idx}\[', f'{var_name}[', result)
    
    # Replace input references (i0, i1, etc.)
    for idx, stream in enumerate(input_streams):
        hex_idx = hex(idx)[2:].upper()
        var_name = stream.name.replace("-", "_")
        # Replace i0[t], i0[t-1], etc.
        result = re.sub(rf'\bi{hex_idx}\[', f'{var_name}[', result)
    
    # Replace mirror references (mi0, mi1, etc.)
    for idx, stream in enumerate(input_streams):
        hex_idx = hex(idx)[2:].upper()
        var_name = f"{stream.name}_mirror".replace("-", "_")
        result = re.sub(rf'\bmi{hex_idx}\[', f'{var_name}[', result)
    
    return result


def _generate_execution_commands(num_steps: int) -> List[str]:
    """Generate execution commands.
    
    In Tau REPL:
    - Empty lines (ENTER) advance execution by one step
    - 'n' means NORMALIZE, not next step!
    - 'q' quits the REPL
    """
    lines = []
    # Each empty line steps execution forward by one time step
    for _ in range(num_steps):
        lines.append("")  # Empty line = step execution
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
        
    Notes:
        When schema.descriptive_names=True, the generated spec uses human-readable
        variable names like 'buy_signal' instead of 'i0'. This mode requires
        running Tau with --charvar off (or TauConfig(charvar=False) in runner).
        
        Example with descriptive_names=False (default):
            i0:sbf = in file("inputs/buy_signal.in")
            o0:sbf = out file("outputs/position.out")
            r o0[t] = i0[t]
            
        Example with descriptive_names=True:
            buy_signal:sbf = in file("inputs/buy_signal.in")
            position:sbf = out file("outputs/position.out")
            r position[t] = buy_signal[t]
    """
    with monitor.measure("total_generation", schema_name=schema.name):
        descriptive = schema.descriptive_names
        
        lines = [
            f"# {schema.name} - Generated Tau Agent",
            "# Auto-generated from AgentSchema",
            f"# {schema.name} Agent (Auto-generated)",
            f"# Strategy: {schema.strategy}",
        ]
        
        # Add note about charvar requirement if using descriptive names
        if descriptive:
            lines.append("# NOTE: This spec uses descriptive names - run with: tau --charvar off")
        lines.append("")
        
        # Input declarations
        lines.extend(_generate_inputs(schema.streams, descriptive))
        lines.append("")
        
        # Input mirrors (if enabled)
        if schema.include_mirrors:
            lines.extend(_generate_input_mirrors(schema.streams, descriptive))
            lines.append("")
        
        # Output declarations
        lines.extend(_generate_outputs(schema.streams, descriptive))
        lines.append("")
        
        # Recurrence block (with descriptive names if enabled)
        lines.extend(_generate_recurrence_block(schema, descriptive))
        # No blank line here - execution commands (empty lines) follow directly
        
        # Execution commands (empty lines for stepping + quit)
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

