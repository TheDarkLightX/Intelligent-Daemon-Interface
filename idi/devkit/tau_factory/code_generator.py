"""Tau code generator using structured templates and pattern registry.

Replaces the monolithic generator with a clean, modular architecture
that separates concerns and enables better testing and maintenance.
"""

from __future__ import annotations

from typing import List
from idi.devkit.tau_factory.schema import AgentSchema

from .template_engine import registry as template_registry
from .pattern_registry import registry as pattern_registry
from .dsl_parser import DSLParser


class TauCodeGenerator:
    """Clean Tau code generator with separated concerns.

    This class orchestrates the Tau code generation process using
    the modular architecture: parser for validation, templates for
    structured code emission, and pattern registry for logic generation.
    """

    def __init__(self) -> None:
        """Initialize the code generator with required components."""
        self.parser = DSLParser()
        self.templates = template_registry
        self.patterns = pattern_registry

    def generate(self, schema: AgentSchema) -> str:
        """Generate complete Tau spec from agent schema."""
        # Parse and validate schema
        parsed_schema = self.parser.parse(schema)

        # Generate code sections
        sections = []
        sections.append(self._generate_header(schema))
        sections.append(self._generate_io_streams(parsed_schema))
        sections.append(self._generate_logic_blocks(parsed_schema))

        return "\n\n".join(sections)

    def _generate_header(self, schema: AgentSchema) -> str:
        """Generate spec header."""
        return f"""% {schema.name} - Generated Tau Agent
% Auto-generated from AgentSchema
% Do not edit manually"""

    def _generate_io_streams(self, parsed_schema) -> str:
        """Generate I/O stream declarations."""
        io_template = self.templates.get("io_streams")
        lines = []

        # Input streams
        for idx, stream in enumerate(s for s in parsed_schema.streams if not s.is_input):
            stream_type = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
            context = {
                "idx": hex(idx)[2:].upper(),
                "type": stream_type,
                "name": stream.name
            }
            lines.append(io_template.render(context))

        # Output streams
        for idx, stream in enumerate(s for s in parsed_schema.streams if s.is_input):
            stream_type = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
            context = {
                "idx": hex(idx)[2:].upper(),
                "type": stream_type,
                "name": stream.name
            }
            lines.append(io_template.render(context))

        return "\n".join(lines)

    def _generate_logic_blocks(self, parsed_schema) -> str:
        """Generate logic block implementations."""
        logic_template = self.templates.get("logic_block")
        lines = []

        for block in parsed_schema.logic_blocks:
            # Generate pattern-specific logic
            logic_code = self.patterns.generate(block, tuple(parsed_schema.streams))

            # Wrap in template
            context = {
                "description": f"{block.pattern} pattern: {block.output} <- {', '.join(block.inputs)}",
                "logic_code": logic_code
            }
            lines.append(logic_template.render(context))

        return "\n\n".join(lines)
