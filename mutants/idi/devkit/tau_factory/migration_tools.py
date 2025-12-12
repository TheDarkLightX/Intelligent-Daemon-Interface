"""Migration tools for Tau Agent Schema evolution.

Handles schema upgrades when DSL features change or are deprecated.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .schema import AgentSchema, StreamConfig, LogicBlock


class SchemaVersion(Enum):
    """Schema version identifiers."""

    V1_0 = "1.0"  # Initial version
    V1_1 = "1.1"  # Added validation improvements
    V1_2 = "1.2"  # Added performance optimizations
    CURRENT = "1.2"


@dataclass
class MigrationRule:
    """A migration rule that transforms schema elements."""

    name: str
    description: str
    from_version: str
    to_version: str
    transform_func: Callable[[AgentSchema], AgentSchema]

    def can_apply(self, current_version: str) -> bool:
        """Check if this rule can be applied."""
        return current_version == self.from_version

    def apply(self, schema: AgentSchema) -> AgentSchema:
        """Apply the migration transformation."""
        return self.transform_func(schema)


class SchemaMigrator:
    """Handles schema migrations between versions."""

    def __init__(self) -> None:
        self.migration_rules: Dict[str, List[MigrationRule]] = {}
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register built-in migration rules."""

        # Example migration: Add default strategy if missing
        def add_default_strategy(schema: AgentSchema) -> AgentSchema:
            if not hasattr(schema, 'strategy') or not schema.strategy:
                # Create new schema with default strategy
                return AgentSchema(
                    name=schema.name,
                    strategy="custom",  # Default strategy
                    streams=list(schema.streams),
                    logic_blocks=list(schema.logic_blocks),
                    num_steps=getattr(schema, 'num_steps', 10),
                    include_mirrors=getattr(schema, 'include_mirrors', True)
                )
            return schema

        self._add_rule(MigrationRule(
            name="add_default_strategy",
            description="Add default 'custom' strategy if missing",
            from_version="1.0",
            to_version="1.1",
            transform_func=add_default_strategy
        ))

        # Example migration: Normalize stream naming
        def normalize_stream_names(schema: AgentSchema) -> AgentSchema:
            normalized_streams = []
            for stream in schema.streams:
                # Ensure stream names are valid identifiers
                normalized_name = stream.name.replace("-", "_").lower()
                if normalized_name != stream.name:
                    # Create new stream with normalized name
                    normalized_streams.append(StreamConfig(
                        name=normalized_name,
                        stream_type=stream.stream_type,
                        width=stream.width,
                        is_input=stream.is_input
                    ))
                else:
                    normalized_streams.append(stream)

            return AgentSchema(
                name=schema.name,
                strategy=getattr(schema, 'strategy', 'custom'),
                streams=normalized_streams,
                logic_blocks=list(schema.logic_blocks),
                num_steps=getattr(schema, 'num_steps', 10),
                include_mirrors=getattr(schema, 'include_mirrors', True)
            )

        self._add_rule(MigrationRule(
            name="normalize_stream_names",
            description="Normalize stream names to use underscores and lowercase",
            from_version="1.1",
            to_version="1.2",
            transform_func=normalize_stream_names
        ))

    def _add_rule(self, rule: MigrationRule) -> None:
        """Add a migration rule."""
        if rule.from_version not in self.migration_rules:
            self.migration_rules[rule.from_version] = []
        self.migration_rules[rule.from_version].append(rule)

    def migrate_schema(self, schema: AgentSchema, from_version: str,
                      to_version: str = SchemaVersion.CURRENT.value) -> AgentSchema:
        """Migrate a schema from one version to another."""
        current_schema = schema
        current_version = from_version

        # Apply migration rules in sequence
        while current_version != to_version:
            if current_version not in self.migration_rules:
                raise ValueError(f"No migration path from version {current_version}")

            applicable_rules = [
                rule for rule in self.migration_rules[current_version]
                if rule.to_version == to_version
            ]

            if not applicable_rules:
                # Try intermediate migrations
                found_path = False
                for rule in self.migration_rules[current_version]:
                    try:
                        current_schema = rule.apply(current_schema)
                        current_version = rule.to_version
                        found_path = True
                        break
                    except Exception:
                        continue

                if not found_path:
                    raise ValueError(f"Cannot migrate from {current_version} to {to_version}")
            else:
                # Direct migration available
                rule = applicable_rules[0]
                current_schema = rule.apply(current_schema)
                current_version = to_version

        return current_schema

    def get_migration_path(self, from_version: str,
                          to_version: str = SchemaVersion.CURRENT.value) -> List[str]:
        """Get the migration path from one version to another."""
        path = [from_version]
        current = from_version

        while current != to_version:
            if current not in self.migration_rules:
                raise ValueError(f"No migration path from version {current}")

            # Find a rule that gets us closer to target
            next_version = None
            for rule in self.migration_rules[current]:
                if rule.to_version == to_version:
                    next_version = to_version
                    break
                elif rule.to_version not in path:  # Avoid cycles
                    next_version = rule.to_version
                    break

            if next_version is None:
                raise ValueError(f"Cannot reach {to_version} from {current}")

            path.append(next_version)
            current = next_version

        return path

    def get_available_versions(self) -> List[str]:
        """Get all available schema versions."""
        versions = set()
        for rules in self.migration_rules.values():
            for rule in rules:
                versions.add(rule.from_version)
                versions.add(rule.to_version)
        return sorted(versions)

    def validate_migration(self, original: AgentSchema, migrated: AgentSchema,
                          from_version: str, to_version: str) -> List[str]:
        """Validate that a migration preserved functionality."""
        issues = []

        # Basic structure checks
        if len(original.streams) != len(migrated.streams):
            issues.append("Migration changed number of streams")

        if len(original.logic_blocks) != len(migrated.logic_blocks):
            issues.append("Migration changed number of logic blocks")

        # Check that logic blocks still reference valid streams
        migrated_stream_names = {s.name for s in migrated.streams}
        for block in migrated.logic_blocks:
            for input_name in block.inputs:
                if input_name not in migrated_stream_names:
                    issues.append(f"Migrated block references unknown input: {input_name}")
            if block.output not in migrated_stream_names:
                issues.append(f"Migrated block references unknown output: {block.output}")

        return issues


# Global migrator instance
migrator = SchemaMigrator()


def migrate_schema(schema: AgentSchema, from_version: str,
                  to_version: str = SchemaVersion.CURRENT.value) -> AgentSchema:
    """Convenience function to migrate a schema."""
    return migrator.migrate_schema(schema, from_version, to_version)


def validate_migration(original: AgentSchema, migrated: AgentSchema,
                      from_version: str, to_version: str) -> List[str]:
    """Convenience function to validate a migration."""
    return migrator.validate_migration(original, migrated, from_version, to_version)
