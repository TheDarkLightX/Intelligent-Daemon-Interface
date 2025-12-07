"""Tau spec generator - compiles AgentSchema to valid Tau Language spec."""

from __future__ import annotations

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
    """Get hex index for a stream name."""
    filtered = [s for s in streams if s.is_input == is_input]
    for idx, stream in enumerate(filtered):
        if stream.name == stream_name:
            return hex(idx)[2:].upper()
    raise ValueError(f"Stream {stream_name} not found")


def _generate_fsm_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate FSM pattern logic."""
    if len(block.inputs) < 2:
        raise ValueError("FSM pattern requires at least 2 inputs (buy, sell)")
    
    buy_idx = _get_stream_index(block.inputs[0], streams, is_input=True)
    sell_idx = _get_stream_index(block.inputs[1], streams, is_input=True)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Position FSM: buy sets position, sell clears it, otherwise maintain
    return f"(o{output_idx}[t] = i{buy_idx}[t] | (o{output_idx}[t-1] & i{sell_idx}[t]'))"


def _generate_passthrough_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate passthrough pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Passthrough pattern requires exactly 1 input")
    
    input_idx = _get_stream_index(block.inputs[0], streams, is_input=True)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    return f"(o{output_idx}[t] = i{input_idx}[t])"


def _generate_counter_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate counter pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Counter pattern requires exactly 1 input")
    
    event_idx = _get_stream_index(block.inputs[0], streams, is_input=True)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Toggle counter: flip on event
    return f"(o{output_idx}[t] = (i{event_idx}[t] & o{output_idx}[t-1]') | (i{event_idx}[t]' & o{output_idx}[t-1]))"


def _generate_accumulator_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate accumulator pattern logic."""
    if len(block.inputs) != 1:
        raise ValueError("Accumulator pattern requires exactly 1 input")
    
    input_idx = _get_stream_index(block.inputs[0], streams, is_input=True)
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # Find output stream width
    output_stream = next(s for s in streams if s.name == block.output)
    width = output_stream.width if output_stream.stream_type == "bv" else 8
    
    return f"(o{output_idx}[t] = o{output_idx}[t-1] + i{input_idx}[t]) && (o{output_idx}[0] = {{0}}:bv[{width}])"


def _generate_vote_logic(block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
    """Generate vote pattern logic."""
    if len(block.inputs) < 2:
        raise ValueError("Vote pattern requires at least 2 inputs")
    
    input_indices = [_get_stream_index(inp, streams, is_input=True) for inp in block.inputs]
    output_idx = _get_stream_index(block.output, streams, is_input=False)
    
    # OR all inputs together
    vote_expr = " | ".join(f"i{idx}[t]" for idx in input_indices)
    return f"(o{output_idx}[t] = {vote_expr})"


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


def generate_tau_spec(schema: AgentSchema) -> str:
    """Generate a complete Tau Language spec from an AgentSchema.
    
    Args:
        schema: The agent schema to compile
        
    Returns:
        Complete Tau spec as a string
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
    
    return "\n".join(lines)

