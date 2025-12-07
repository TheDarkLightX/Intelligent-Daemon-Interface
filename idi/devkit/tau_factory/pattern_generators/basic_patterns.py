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
        """Generate basic FSM logic."""
        states = block.params.get("states", [])
        transitions = block.params.get("transitions", {})

        if not states or not transitions:
            raise ValueError("FSM requires 'states' and 'transitions' parameters")

        lines = [
            "% Basic FSM implementation",
            f"{block.output} :- {block.inputs[0]}."
        ]

        # Add state transition logic
        for state in states:
            if state in transitions:
                next_state = transitions[state]
                lines.append(f"{block.output} :- {state} -> {next_state}.")

        return "\n".join(lines)

    def _generate_counter(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate counter logic."""
        max_value = block.params.get("max_value", 10)
        reset_condition = block.params.get("reset_condition", "")

        lines = [
            "% Counter implementation",
            f"{block.output} :- {block.inputs[0]} + 1."
        ]

        if max_value:
            lines.append(f"{block.output} :- {block.output} mod {max_value}.")

        if reset_condition:
            lines.append(f"{block.output} :- {reset_condition} -> 0.")

        return "\n".join(lines)

    def _generate_accumulator(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate accumulator logic."""
        operation = block.params.get("operation", "+")

        lines = [
            "% Accumulator implementation",
            f"{block.output} :- {block.inputs[0]} {operation} {block.output}."
        ]

        return "\n".join(lines)

    def _generate_vote(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate voting logic (OR-based)."""
        lines = [
            "% Vote implementation (OR logic)",
            f"{block.output} :- {' | '.join(block.inputs)}."
        ]

        return "\n".join(lines)

    def _generate_passthrough(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate passthrough logic."""
        lines = [
            "% Passthrough implementation",
            f"{block.output} :- {block.inputs[0]}."
        ]

        return "\n".join(lines)
