"""Composite pattern generators: Majority, Unanimous, Custom, Quorum.

These patterns combine multiple inputs with voting/consensus logic.
"""

from __future__ import annotations

from typing import Dict, Any, List
from idi.devkit.tau_factory.schema import LogicBlock, StreamConfig


class CompositePatternGenerator:
    """Generator for composite voting/consensus patterns."""

    def generate(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate Tau code for composite patterns."""
        pattern = block.pattern

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
        """Generate majority voting logic."""
        threshold = block.params.get("threshold", len(block.inputs) // 2 + 1)

        lines = [
            "% Majority voting implementation",
            f"% Requires {threshold} out of {len(block.inputs)} votes"
        ]

        # Generate counting logic
        count_expr = " + ".join(f"({inp} ? 1 : 0)" for inp in block.inputs)
        lines.append(f"vote_count :- {count_expr}.")
        lines.append(f"{block.output} :- vote_count >= {threshold}.")

        return "\n".join(lines)

    def _generate_unanimous(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate unanimous consensus logic."""
        lines = [
            "% Unanimous voting implementation",
            f"% Requires all {len(block.inputs)} votes to agree"
        ]

        # All inputs must be true
        conditions = " & ".join(block.inputs)
        lines.append(f"{block.output} :- {conditions}.")

        return "\n".join(lines)

    def _generate_custom(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate custom boolean expression logic."""
        expression = block.params.get("expression", "")

        if not expression:
            raise ValueError("Custom pattern requires 'expression' parameter")

        lines = [
            "% Custom boolean expression implementation",
            f"{block.output} :- {expression}."
        ]

        return "\n".join(lines)

    def _generate_quorum(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate quorum-based voting logic."""
        min_votes = block.params.get("min_votes", len(block.inputs) // 2 + 1)

        lines = [
            "% Quorum voting implementation",
            f"% Requires minimum {min_votes} votes"
        ]

        # Count true votes
        count_expr = " + ".join(f"({inp} ? 1 : 0)" for inp in block.inputs)
        lines.append(f"vote_count :- {count_expr}.")
        lines.append(f"{block.output} :- vote_count >= {min_votes}.")

        return "\n".join(lines)
