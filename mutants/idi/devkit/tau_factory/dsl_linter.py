"""DSL Linter for Tau Agent Schemas.

Provides static analysis, best practice checks, and optimization suggestions
for Tau agent schemas.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from .schema import AgentSchema, StreamConfig, LogicBlock
from .dsl_parser import ValidationError


@dataclass
class LintRule:
    """A linting rule with check function and metadata."""

    name: str
    description: str
    severity: str  # "error", "warning", "info"
    check_func: callable

    def check(self, schema: AgentSchema) -> List[LintIssue]:
        """Run the rule check on a schema."""
        return self.check_func(schema)


@dataclass
class LintIssue:
    """A linting issue found in a schema."""

    rule: str
    severity: str
    message: str
    location: str = ""
    suggestion: str = ""
    context: Optional[Dict[str, Any]] = None


class DSLLinter:
    """Linter for Tau agent schemas with comprehensive analysis."""

    def __init__(self) -> None:
        self.rules: List[LintRule] = []
        self._register_builtin_rules()

    def _register_builtin_rules(self) -> None:
        """Register built-in linting rules."""

        # Schema structure rules
        self.rules.append(LintRule(
            name="schema_naming",
            description="Schema names should follow naming conventions",
            severity="warning",
            check_func=self._check_schema_naming
        ))

        self.rules.append(LintRule(
            name="unused_streams",
            description="Detect streams that are not used in any logic blocks",
            severity="warning",
            check_func=self._check_unused_streams
        ))

        self.rules.append(LintRule(
            name="stream_naming_consistency",
            description="Stream names should follow consistent naming patterns",
            severity="info",
            check_func=self._check_stream_naming_consistency
        ))

        # Logic block rules
        self.rules.append(LintRule(
            name="logic_block_coverage",
            description="Ensure logic blocks cover all input scenarios",
            severity="info",
            check_func=self._check_logic_block_coverage
        ))

        self.rules.append(LintRule(
            name="pattern_complexity",
            description="Flag overly complex patterns that could be simplified",
            severity="warning",
            check_func=self._check_pattern_complexity
        ))

        # Performance rules
        self.rules.append(LintRule(
            name="large_schema_warning",
            description="Large schemas may impact performance",
            severity="info",
            check_func=self._check_large_schema
        ))

        self.rules.append(LintRule(
            name="redundant_patterns",
            description="Detect redundant or equivalent logic patterns",
            severity="warning",
            check_func=self._check_redundant_patterns
        ))

    def lint(self, schema: AgentSchema) -> List[LintIssue]:
        """Run all linting rules on a schema."""
        issues = []

        for rule in self.rules:
            try:
                rule_issues = rule.check(schema)
                issues.extend(rule_issues)
            except Exception as e:
                # Don't let linting errors break the process
                issues.append(LintIssue(
                    rule="linter_error",
                    severity="error",
                    message=f"Linter rule '{rule.name}' failed: {e}",
                    location="linter"
                ))

        return issues

    def _check_schema_naming(self, schema: AgentSchema) -> List[LintIssue]:
        """Check schema naming conventions."""
        issues = []

        if not schema.name:
            issues.append(LintIssue(
                rule="schema_naming",
                severity="error",
                message="Schema name cannot be empty",
                location="schema.name",
                suggestion="Provide a descriptive name for the agent schema"
            ))

        if len(schema.name) < 3:
            issues.append(LintIssue(
                rule="schema_naming",
                severity="warning",
                message="Schema name is very short",
                location="schema.name",
                suggestion="Use a more descriptive name (3+ characters)"
            ))

        if not schema.name.replace("_", "").replace("-", "").isalnum():
            issues.append(LintIssue(
                rule="schema_naming",
                severity="warning",
                message="Schema name contains special characters",
                location="schema.name",
                suggestion="Use only letters, numbers, underscores, and hyphens"
            ))

        return issues

    def _check_unused_streams(self, schema: AgentSchema) -> List[LintIssue]:
        """Check for streams that are not used in logic blocks."""
        issues = []

        # Collect all stream names
        all_streams = {s.name for s in schema.streams}

        # Collect streams used in logic blocks
        used_inputs = set()
        used_outputs = set()

        for block in schema.logic_blocks:
            used_inputs.update(block.inputs)
            used_outputs.add(block.output)

        used_streams = used_inputs | used_outputs

        # Find unused streams
        unused_streams = all_streams - used_streams

        for stream_name in unused_streams:
            issues.append(LintIssue(
                rule="unused_streams",
                severity="warning",
                message=f"Stream '{stream_name}' is declared but never used",
                location=f"streams.{stream_name}",
                suggestion="Remove unused stream or add logic that uses it"
            ))

        return issues

    def _check_stream_naming_consistency(self, schema: AgentSchema) -> List[LintIssue]:
        """Check for consistent stream naming patterns."""
        issues = []

        # Check for mixed naming conventions
        snake_case = 0
        camel_case = 0
        kebab_case = 0

        for stream in schema.streams:
            name = stream.name
            if "_" in name and name.islower():
                snake_case += 1
            elif any(c.isupper() for c in name):
                camel_case += 1
            elif "-" in name:
                kebab_case += 1

        total = len(schema.streams)
        if total > 2:  # Only check if we have enough streams
            # Check if we have inconsistent naming
            conventions = [snake_case, camel_case, kebab_case]
            active_conventions = sum(1 for c in conventions if c > 0)

            if active_conventions > 1:
                issues.append(LintIssue(
                    rule="stream_naming_consistency",
                    severity="info",
                    message="Mixed naming conventions detected in stream names",
                    location="streams",
                    suggestion="Use consistent naming: snake_case, camelCase, or kebab-case"
                ))

        return issues

    def _check_logic_block_coverage(self, schema: AgentSchema) -> List[LintIssue]:
        """Check if logic blocks provide adequate coverage."""
        issues = []

        # Basic coverage check: ensure we have logic for most inputs
        input_streams = {s.name for s in schema.streams if not s.is_input}
        output_streams = {s.name for s in schema.streams if s.is_input}

        # Check if all output streams are produced by logic blocks
        produced_outputs = {block.output for block in schema.logic_blocks}
        uncovered_outputs = output_streams - produced_outputs

        if uncovered_outputs:
            for output in uncovered_outputs:
                issues.append(LintIssue(
                    rule="logic_block_coverage",
                    severity="warning",
                    message=f"Output stream '{output}' is not produced by any logic block",
                    location=f"streams.{output}",
                    suggestion="Add a logic block that produces this output"
                ))

        return issues

    def _check_pattern_complexity(self, schema: AgentSchema) -> List[LintIssue]:
        """Check for overly complex patterns."""
        issues = []

        for i, block in enumerate(schema.logic_blocks):
            # Check for patterns with many inputs
            if len(block.inputs) > 5:
                issues.append(LintIssue(
                    rule="pattern_complexity",
                    severity="warning",
                    message=f"Logic block has {len(block.inputs)} inputs, consider simplifying",
                    location=f"logic_blocks[{i}]",
                    suggestion="Break complex logic into smaller, simpler blocks"
                ))

            # Check for complex parameter configurations
            if len(block.params) > 3:
                issues.append(LintIssue(
                    rule="pattern_complexity",
                    severity="info",
                    message=f"Logic block has {len(block.params)} parameters",
                    location=f"logic_blocks[{i}]",
                    suggestion="Review if all parameters are necessary"
                ))

        return issues

    def _check_large_schema(self, schema: AgentSchema) -> List[LintIssue]:
        """Check for schemas that might be too large."""
        issues = []

        total_elements = len(schema.streams) + len(schema.logic_blocks)

        if total_elements > 50:
            issues.append(LintIssue(
                rule="large_schema_warning",
                severity="info",
                message=f"Schema has {total_elements} elements, consider splitting into smaller schemas",
                location="schema",
                suggestion="Break large schemas into smaller, focused agents"
            ))

        return issues

    def _check_redundant_patterns(self, schema: AgentSchema) -> List[LintIssue]:
        """Check for redundant or equivalent logic patterns."""
        issues = []

        # Simple check: look for identical logic blocks
        seen_blocks = {}
        for i, block in enumerate(schema.logic_blocks):
            key = (block.pattern, tuple(sorted(block.inputs)), block.output)
            if key in seen_blocks:
                issues.append(LintIssue(
                    rule="redundant_patterns",
                    severity="warning",
                    message=f"Logic block {i} is identical to block {seen_blocks[key]}",
                    location=f"logic_blocks[{i}]",
                    suggestion="Remove duplicate logic block or merge functionality"
                ))
            else:
                seen_blocks[key] = i

        return issues

    def get_rule_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available rules."""
        return {rule.name: rule.description for rule in self.rules}

    def lint_with_report(self, schema: AgentSchema) -> Dict[str, Any]:
        """Run linting and return a structured report."""
        issues = self.lint(schema)

        # Group issues by severity
        by_severity = {
            "error": [],
            "warning": [],
            "info": []
        }

        for issue in issues:
            by_severity[issue.severity].append({
                "rule": issue.rule,
                "message": issue.message,
                "location": issue.location,
                "suggestion": issue.suggestion
            })

        return {
            "summary": {
                "total_issues": len(issues),
                "errors": len(by_severity["error"]),
                "warnings": len(by_severity["warning"]),
                "info": len(by_severity["info"])
            },
            "issues": by_severity
        }
