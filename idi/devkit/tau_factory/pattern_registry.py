"""Pattern registry for managing Tau code generators.

Provides a clean interface for registering and accessing pattern generators
without tight coupling between the main generator and individual patterns.
"""

from __future__ import annotations

from typing import Dict, Protocol, Any
from idi.devkit.tau_factory.schema import LogicBlock, StreamConfig

from .pattern_generators.basic_patterns import BasicPatternGenerator
from .pattern_generators.composite_patterns import CompositePatternGenerator


class PatternGenerator(Protocol):
    """Protocol for pattern generators."""

    def generate(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate Tau code for a logic block."""
        ...


class PatternRegistry:
    """Registry for pattern generators with dependency injection support."""

    def __init__(self) -> None:
        self._generators: Dict[str, PatternGenerator] = {}
        self._register_builtin_generators()

    def _register_builtin_generators(self) -> None:
        """Register built-in pattern generators."""
        basic_gen = BasicPatternGenerator()
        composite_gen = CompositePatternGenerator()

        # Basic patterns
        basic_patterns = [
            "fsm", "counter", "accumulator", "vote", "passthrough"
        ]
        for pattern in basic_patterns:
            self._generators[pattern] = basic_gen

        # Composite patterns
        composite_patterns = [
            "majority", "unanimous", "custom", "quorum"
        ]
        for pattern in composite_patterns:
            self._generators[pattern] = composite_gen

    def register(self, pattern: str, generator: PatternGenerator) -> None:
        """Register a custom pattern generator."""
        self._generators[pattern] = generator

    def get_generator(self, pattern: str) -> PatternGenerator:
        """Get generator for a pattern."""
        if pattern not in self._generators:
            available = list(self._generators.keys())
            raise ValueError(
                f"No generator registered for pattern '{pattern}'. "
                f"Available patterns: {available}"
            )
        return self._generators[pattern]

    def supports_pattern(self, pattern: str) -> bool:
        """Check if a pattern is supported."""
        return pattern in self._generators

    def list_patterns(self) -> list[str]:
        """List all registered patterns."""
        return list(self._generators.keys())

    def generate(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate code for a logic block."""
        pattern_name = block.pattern.value if hasattr(block.pattern, 'value') else block.pattern
        generator = self.get_generator(pattern_name)
        return generator.generate(block, streams)


# Global registry instance
registry = PatternRegistry()
