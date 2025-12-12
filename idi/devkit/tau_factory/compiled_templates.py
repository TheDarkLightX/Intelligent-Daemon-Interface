"""Compiled template system for high-performance rendering.

Implements pre-compiled templates with integer variable IDs for O(1) lookups,
following performance optimization best practices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

from .template_engine import TauTemplate, TemplateVariable


@dataclass(frozen=True)
class CompiledVariable:
    """Compiled variable with integer ID for fast lookup."""

    original_name: str
    slot_id: int
    type_hint: Optional[str] = None

    def __str__(self) -> str:
        return f"var_{self.slot_id}"


@dataclass
class CompiledFragment:
    """Compiled template fragment with resolved variable references."""

    parts: List[Union[str, CompiledVariable]] = field(default_factory=list)

    def render(self, values: List[str]) -> str:
        """Render fragment using direct array access for variables."""
        result_parts = []
        for part in self.parts:
            if isinstance(part, CompiledVariable):
                # O(1) array access instead of dictionary lookup
                result_parts.append(values[part.slot_id])
            else:
                result_parts.append(part)
        return "".join(result_parts)


@dataclass
class CompiledTemplate:
    """Pre-compiled template with optimized structure."""

    name: str
    fragments: List[CompiledFragment] = field(default_factory=list)
    variable_map: Dict[str, CompiledVariable] = field(default_factory=dict)
    next_slot_id: int = 0

    def add_fragment(self, content: str) -> None:
        """Compile a template fragment."""
        fragment = self._compile_fragment(content)
        self.fragments.append(fragment)

    def _compile_fragment(self, content: str) -> CompiledFragment:
        """Compile template string to optimized fragment."""
        # Simple compilation: split by ${} placeholders
        # In a full implementation, this would parse more complex syntax
        parts = []
        remaining = content

        while "${" in remaining:
            # Split on first ${
            before, after = remaining.split("${", 1)
            if before:
                parts.append(before)

            # Extract variable name
            if "}" in after:
                var_name, remaining = after.split("}", 1)

                # Get or create compiled variable
                if var_name not in self.variable_map:
                    compiled_var = CompiledVariable(
                        original_name=var_name,
                        slot_id=self.next_slot_id
                    )
                    self.variable_map[var_name] = compiled_var
                    self.next_slot_id += 1
                else:
                    compiled_var = self.variable_map[var_name]

                parts.append(compiled_var)
            else:
                # Malformed - treat as literal
                parts.append("${" + after)
                break
        else:
            # No more variables
            if remaining:
                parts.append(remaining)

        return CompiledFragment(parts)

    def render(self, context: Dict[str, Any]) -> str:
        """Render compiled template with optimized variable access."""
        # Convert context to array for O(1) access
        values = [""] * self.next_slot_id

        for var_name, compiled_var in self.variable_map.items():
            if var_name in context:
                values[compiled_var.slot_id] = str(context[var_name])

        # Render all fragments
        result_parts = []
        for fragment in self.fragments:
            result_parts.append(fragment.render(values))

        return "\n".join(result_parts)

    def get_required_slots(self) -> int:
        """Get number of variable slots needed."""
        return self.next_slot_id


class TemplateCache:
    """LRU cache for compiled templates."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, CompiledTemplate] = {}
        self.access_order: List[str] = []

    def get_or_compile(self, name: str, template_source: TauTemplate) -> CompiledTemplate:
        """Get cached compiled template or compile and cache it."""
        cache_key = self._make_cache_key(name, template_source)

        if cache_key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]

        # Compile new template
        compiled = self._compile_template(name, template_source)

        # Add to cache
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[cache_key] = compiled
        self.access_order.append(cache_key)

        return compiled

    def _make_cache_key(self, name: str, template: TauTemplate) -> str:
        """Create cache key from template content."""
        # Hash template content for cache key
        content = name
        for fragment in template.fragments:
            content += fragment.content
            for var in fragment.variables:
                content += str(var)

        # Use SHA-256 instead of MD5 to avoid collision vulnerabilities
        return hashlib.sha256(content.encode()).hexdigest()

    def _compile_template(self, name: str, template: TauTemplate) -> CompiledTemplate:
        """Compile TauTemplate to CompiledTemplate."""
        compiled = CompiledTemplate(name=name)

        for fragment in template.fragments:
            compiled.add_fragment(fragment.content)

        return compiled

    def clear(self) -> None:
        """Clear all cached templates."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global cache instance
template_cache = TemplateCache()
