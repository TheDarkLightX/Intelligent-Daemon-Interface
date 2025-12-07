"""Tau Agent Factory - Parameterized agent generation system."""

__version__ = "0.1.0"

# Clean public API - hide internal complexity
from .generator import generate_tau_spec, validate_schema, create_minimal_schema
from .schema import AgentSchema, StreamConfig, LogicBlock
from .dsl_parser import ValidationError
from .dsl_linter import DSLLinter
from .migration_tools import migrate_schema, validate_migration, SchemaVersion

__all__ = [
    "generate_tau_spec",
    "validate_schema",
    "create_minimal_schema",
    "AgentSchema",
    "StreamConfig",
    "LogicBlock",
    "ValidationError",
    "DSLLinter",
    "migrate_schema",
    "validate_migration",
    "SchemaVersion",
]

