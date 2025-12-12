"""Structured template engine for Tau code generation.

Replaces string-based templates with structured, analyzable templates
that support validation, composition, and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class TemplateVariable:
    """A variable reference in a template."""

    name: str
    type_hint: Optional[str] = None

    def __str__(self) -> str:
        return f"${{{self.name}}}"


@dataclass
class TemplateFragment:
    """A fragment of Tau code with variable references."""

    content: str
    variables: List[TemplateVariable] = field(default_factory=list)

    def render(self, context: Dict[str, Any]) -> str:
        """Render fragment with variable substitution."""
        result = self.content
        for var in self.variables:
            if var.name not in context:
                raise ValueError(f"Missing required variable: {var.name}")
            value = context[var.name]

            # Security: prevent template injection by validating variable content
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Variable {var.name} must be a primitive type, got {type(value)}")

            # Limit variable value length to prevent DoS
            value_str = str(value)
            if len(value_str) > 1000:  # 1KB limit per variable
                raise ValueError(f"Variable {var.name} value too long (>1000 chars)")

            result = result.replace(str(var), value_str)
        return result


@dataclass
class TauTemplate:
    """Complete Tau code template with structured composition."""

    name: str
    fragments: List[TemplateFragment] = field(default_factory=list)

    def add_fragment(self, content: str, variables: List[TemplateVariable] = None) -> None:
        """Add a code fragment to the template."""
        if variables is None:
            variables = []
        self.fragments.append(TemplateFragment(content, variables))

    def render(self, context: Dict[str, Any]) -> str:
        """Render complete template."""
        return "\n".join(fragment.render(context) for fragment in self.fragments)

    def validate_context(self, context: Dict[str, Any]) -> List[str]:
        """Validate that context provides all required variables."""
        errors = []
        required_vars = set()
        for fragment in self.fragments:
            for var in fragment.variables:
                required_vars.add(var.name)

        for var_name in required_vars:
            if var_name not in context:
                errors.append(f"Missing required variable: {var_name}")

        return errors


class TemplateRegistry:
    """Registry for managing Tau templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, TauTemplate] = {}

    def register(self, template: TauTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template

    def get(self, name: str) -> TauTemplate:
        """Get a template by name."""
        if name not in self._templates:
            raise ValueError(f"Template not found: {name}")
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())


# Global registry instance
registry = TemplateRegistry()


def create_io_template() -> TauTemplate:
    """Create template for I/O stream declarations."""
    template = TauTemplate("io_streams")

    # Input streams
    template.add_fragment(
        "i${idx}:${type} = in file(\"inputs/${name}.in\").",
        [TemplateVariable("idx"), TemplateVariable("type"), TemplateVariable("name")]
    )

    # Output streams
    template.add_fragment(
        "o${idx}:${type} = out file(\"outputs/${name}.out\").",
        [TemplateVariable("idx"), TemplateVariable("type"), TemplateVariable("name")]
    )

    # Mirror streams for symmetric I/O
    template.add_fragment(
        "i${idx}:${type} = out file(\"outputs/i${idx}_mirror.out\").",
        [TemplateVariable("idx"), TemplateVariable("type")]
    )

    return template


def create_logic_template() -> TauTemplate:
    """Create template for logic block insertion."""
    template = TauTemplate("logic_block")

    template.add_fragment(
        "% ${description}",
        [TemplateVariable("description")]
    )

    template.add_fragment(
        "${logic_code}",
        [TemplateVariable("logic_code")]
    )

    return template


# Register default templates
registry.register(create_io_template())
registry.register(create_logic_template())
