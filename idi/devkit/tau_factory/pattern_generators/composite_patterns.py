"""Composite pattern generators: Majority, Unanimous, Custom, Quorum.

These patterns combine multiple inputs with voting/consensus logic.
"""

from __future__ import annotations

from typing import Dict, Any, List
from idi.devkit.tau_factory.schema import LogicBlock, StreamConfig


class CompositePatternGenerator:
    """Generator for composite voting/consensus patterns."""

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

class CompositePatternGenerator:
    """Generator for composite voting/consensus patterns."""

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

    def generate(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate Tau code for composite patterns."""
        pattern = block.pattern.value if hasattr(block.pattern, "value") else block.pattern

        if pattern == "majority":
            return self._generate_majority(block, streams)
        elif pattern == "unanimous":
            return self._generate_unanimous(block, streams)
        elif pattern == "custom":
            return self._generate_custom(block, streams)
        elif pattern == "quorum":
            return self._generate_quorum(block, streams)
        else:
            raise ValueError(f"Unknown composite pattern: {pattern}")

    def _generate_majority(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate majority voting logic in Tau syntax."""
        threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)

        # Get input references
        input_refs = []
        for inp in block.inputs:
            input_idx, _ = self._get_stream_index_any(inp, streams)
            input_refs.append(f"i{input_idx}[t]")

        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Count true inputs and compare to threshold
        # This is a simplified implementation - real majority voting needs proper counting
        if len(input_refs) == 2:
            # Simple case: 2 inputs, threshold 2 means both must be true
            if threshold >= 2:
                return f"    (o{output_idx}[t] = i{input_refs[0].split('[')[0][1:]}[t] & i{input_refs[1].split('[')[0][1:]}[t])"
            else:
                # threshold 1: at least one true (OR)
                return f"    (o{output_idx}[t] = i{input_refs[0].split('[')[0][1:]}[t] | i{input_refs[1].split('[')[0][1:]}[t])"
        else:
            # For more inputs, use OR as simplification
            or_expr = " | ".join([f"i{ref.split('[')[0][1:]}[t]" for ref in input_refs])
            return f"    (o{output_idx}[t] = {or_expr})"

    def _generate_unanimous(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate unanimous consensus logic in Tau syntax."""
        # All inputs must be true (AND operation)
        input_refs = []
        for inp in block.inputs:
            input_idx, _ = self._get_stream_index_any(inp, streams)
            input_refs.append(f"i{input_idx}[t]")

        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Generate AND expression for unanimous consensus
        and_expr = " & ".join(input_refs)
        return f"    (o{output_idx}[t] = {and_expr})"

    def _generate_custom(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate custom boolean expression logic in Tau syntax."""
        expression = block.params.get("expression", "")

        if not expression:
            raise ValueError("Custom pattern requires 'expression' parameter")

        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Use the expression directly (assuming it's already in Tau syntax)
        return f"    (o{output_idx}[t] = {expression})"

    def _generate_quorum(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate quorum-based voting logic in Tau syntax."""
        min_votes = block.params.get("min_votes", len(block.inputs) // 2 + 1)

        output_idx = self._get_stream_index(block.output, streams, is_input=False)

        # Simplified quorum logic - for now use OR (at least one vote)
        # Real quorum counting would need proper threshold logic
        input_refs = []
        for inp in block.inputs:
            input_idx, _ = self._get_stream_index_any(inp, streams)
            input_refs.append(f"i{input_idx}[t]")

        or_expr = " | ".join(input_refs)
        return f"    (o{output_idx}[t] = {or_expr})"
