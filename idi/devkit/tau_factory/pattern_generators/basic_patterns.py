"""Basic pattern generators: FSM, Counter, Accumulator, Vote, Passthrough.

These are the fundamental building blocks with simpler logic and lower complexity.
"""

from __future__ import annotations

from typing import Dict, Any, List
from idi.devkit.tau_factory.schema import LogicBlock, StreamConfig


class BasicPatternGenerator:
    """Generator for basic FSM patterns with low complexity."""

    def generate(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate Tau code for basic patterns."""
        pattern = block.pattern

        if pattern == "fsm":
            return self._generate_basic_fsm(block, streams)
        elif pattern == "counter":
            return self._generate_counter(block, streams)
        elif pattern == "accumulator":
            return self._generate_accumulator(block, streams)
        elif pattern == "vote":
            return self._generate_vote(block, streams)
        elif pattern == "passthrough":
            return self._generate_passthrough(block, streams)
        else:
            raise ValueError(f"Unknown basic pattern: {pattern}")

    def _generate_basic_fsm(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate basic FSM logic in Tau syntax."""
        states = block.params.get("states", [])
        transitions = block.params.get("transitions", {})

        if not states or not transitions:
            raise ValueError("FSM requires 'states' and 'transitions' parameters")

        # For now, implement as simple state machine
        # This is a simplified implementation - full FSM would need more complex logic
        input_idx, _ = self._get_stream_index_any(block.inputs[0], streams)
        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Simple pass-through for basic FSM
        return f"    (o{output_idx}[t] = i{input_idx}[t])"

    def _generate_counter(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate counter logic in Tau syntax."""
        if len(block.inputs) != 1:
            raise ValueError("Counter pattern requires exactly 1 input (increment signal)")

        max_value = block.params.get("max_value", 10)

        input_idx, _ = self._get_stream_index_any(block.inputs[0], streams)
        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Counter logic: increment on input, modulo max_value
        # This is a simplified implementation - real counters would need state tracking
        return f"    (o{output_idx}[t] = (o{output_idx}[t-1] + i{input_idx}[t]) % {max_value})"

    def _generate_accumulator(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate accumulator logic in Tau syntax."""
        operation = block.params.get("operation", "+")

        input_idx, _ = self._get_stream_index_any(block.inputs[0], streams)
        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Accumulator: running total with operation
        if operation == "+":
            return f"    (o{output_idx}[t] = o{output_idx}[t-1] + i{input_idx}[t])"
        elif operation == "*":
            return f"    (o{output_idx}[t] = o{output_idx}[t-1] * i{input_idx}[t])"
        else:
            return f"    (o{output_idx}[t] = o{output_idx}[t-1] + i{input_idx}[t])"  # Default to addition

    def _generate_vote(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate voting logic (OR-based) in Tau syntax."""
        # OR all inputs together
        input_refs = []
        for inp in block.inputs:
            input_idx, _ = self._get_stream_index_any(inp, streams)
            input_refs.append(f"i{input_idx}[t]")

        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Generate OR expression
        or_expr = " | ".join(input_refs)
        return f"    (o{output_idx}[t] = {or_expr})"

    def _generate_passthrough(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate passthrough logic in Tau syntax."""
        # Get indices for input and output streams
        input_idx, _ = self._get_stream_index_any(block.inputs[0], streams)
        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Generate Tau assignment: o{output_idx}[t] = i{input_idx}[t]
        return f"    (o{output_idx}[t] = i{input_idx}[t])"

    def _get_stream_index(self, stream_name: str, streams: tuple[StreamConfig, ...], is_input: bool) -> str:
        """Get hex index for a stream name."""
        filtered = [s for s in streams if s.is_input == is_input]
        for idx, stream in enumerate(filtered):
            if stream.name == stream_name:
                return hex(idx)[2:].upper()
        raise ValueError(f"Stream {stream_name} not found (is_input={is_input})")

    def _get_stream_index_any(self, stream_name: str, streams: tuple[StreamConfig, ...]) -> tuple[str, bool]:
        """Get hex index for a stream name, searching both inputs and outputs."""
        # Try inputs first
        try:
            idx = self._get_stream_index(stream_name, streams, is_input=True)
            return (idx, True)
        except ValueError:
            pass

        # Try outputs
        try:
            idx = self._get_stream_index(stream_name, streams, is_input=False)
            return (idx, False)
        except ValueError:
            raise ValueError(f"Stream {stream_name} not found in any stream type")
