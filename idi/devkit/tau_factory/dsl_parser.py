"""Formal DSL parser for AgentSchema validation and processing.

Implements a formal grammar for validating agent schemas and provides
structured parsing with better error messages and tooling support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from idi.devkit.tau_factory.schema import AgentSchema, LogicBlock, StreamConfig


class ValidationError(Exception):
    """Validation error with context information."""

    def __init__(self, message: str, path: str = "", suggestion: str = ""):
        """Initialize validation error.

        Args:
            message: Human-readable error message
            path: Path/context where error occurred
            suggestion: Suggested fix for the error
        """
        self.message = message
        self.path = path
        self.suggestion = suggestion
        super().__init__(f"{path}: {message}" + (f" (suggestion: {suggestion})" if suggestion else ""))


class PatternType(Enum):
    """Valid pattern types with validation rules."""

    FSM = "fsm"
    COUNTER = "counter"
    ACCUMULATOR = "accumulator"
    VOTE = "vote"
    PASSTHROUGH = "passthrough"
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    CUSTOM = "custom"
    QUORUM = "quorum"
    SUPERVISOR_WORKER = "supervisor_worker"
    WEIGHTED_VOTE = "weighted_vote"
    TIME_LOCK = "time_lock"
    HEX_STAKE = "hex_stake"
    MULTI_BIT_COUNTER = "multi_bit_counter"
    STREAK_COUNTER = "streak_counter"
    MODE_SWITCH = "mode_switch"
    PROPOSAL_FSM = "proposal_fsm"
    RISK_FSM = "risk_fsm"
    ENTRY_EXIT_FSM = "entry_exit_fsm"
    ORTHOGONAL_REGIONS = "orthogonal_regions"
    STATE_AGGREGATION = "state_aggregation"
    TCP_CONNECTION_FSM = "tcp_connection_fsm"
    UTXO_STATE_MACHINE = "utxo_state_machine"
    HISTORY_STATE = "history_state"
    DECOMPOSED_FSM = "decomposed_fsm"
    SCRIPT_EXECUTION = "script_execution"

    @classmethod
    def validate(cls, name: str) -> PatternType:
        """Validate and return pattern type."""
        try:
            return cls(name)
        except ValueError:
            valid_names = [p.value for p in cls]
            raise ValidationError(
                f"Invalid pattern type: {name}",
                f"pattern",
                f"Must be one of: {', '.join(valid_names)}"
            )


@dataclass
class ParsedLogicBlock:
    """Parsed and validated logic block."""

    pattern: PatternType
    inputs: List[str]
    output: str
    params: Dict[str, Any]
    source_location: Optional[str] = None

    @classmethod
    def from_logic_block(cls, block: LogicBlock, location: str = "") -> ParsedLogicBlock:
        """Create from schema LogicBlock with validation."""
        pattern = PatternType.validate(block.pattern)  # Validate and store enum
        return cls(
            pattern=pattern,
            inputs=list(block.inputs),
            output=block.output,
            params=dict(block.params),
            source_location=location
        )

    def validate(self) -> List[ValidationError]:
        """Validate this logic block."""
        errors = []

        # Pattern-specific validation
        if self.pattern == PatternType.FSM:
            errors.extend(self._validate_fsm())
        elif self.pattern == PatternType.HEX_STAKE:
            errors.extend(self._validate_hex_stake())
        # Add more pattern validations as needed

        return errors

    def _validate_fsm(self) -> List[ValidationError]:
        """Validate FSM pattern requirements."""
        errors = []
        required_params = ["states", "transitions"]
        for param in required_params:
            if param not in self.params:
                errors.append(ValidationError(
                    f"FSM pattern requires '{param}' parameter",
                    f"{self.source_location}",
                    f"Add {param} to block params"
                ))
        return errors

    def _validate_hex_stake(self) -> List[ValidationError]:
        """Validate Hex stake pattern requirements."""
        errors = []
        required_inputs = 5  # stake_amount, stake_duration, current_time, action_stake, action_end
        if len(self.inputs) != required_inputs:
            errors.append(ValidationError(
                f"Hex stake pattern requires {required_inputs} inputs, got {len(self.inputs)}",
                f"{self.source_location}",
                f"Provide inputs: [stake_amount, stake_duration, current_time, action_stake, action_end]"
            ))
        return errors


@dataclass
class ParsedAgentSchema:
    """Parsed and validated agent schema."""

    name: str
    streams: List[StreamConfig]
    logic_blocks: List[ParsedLogicBlock]
    metadata: Dict[str, Any]

    @classmethod
    def from_agent_schema(cls, schema: AgentSchema) -> ParsedAgentSchema:
        """Create from schema AgentSchema with validation."""
        logic_blocks = []
        for i, block in enumerate(schema.logic_blocks):
            parsed_block = ParsedLogicBlock.from_logic_block(
                block, f"logic_blocks[{i}]"
            )
            logic_blocks.append(parsed_block)

        return cls(
            name=schema.name,
            streams=list(schema.streams),
            logic_blocks=logic_blocks,
            metadata=getattr(schema, 'metadata', {})
        )

    def validate(self) -> List[ValidationError]:
        """Validate the entire schema."""
        errors = []

        # Basic field validation
        if not self.name.strip():
            errors.append(ValidationError(
                "Schema name cannot be empty",
                "schema.name",
                "Provide a non-empty name for the agent schema"
            ))

        # Validate streams
        errors.extend(self._validate_streams())

        # Validate logic blocks
        for block in self.logic_blocks:
            errors.extend(block.validate())

        # Validate cross-references
        errors.extend(self._validate_references())

        return errors

    def _validate_streams(self) -> List[ValidationError]:
        """Validate stream configurations."""
        errors = []
        seen_names = set()

        for stream in self.streams:
            if stream.name in seen_names:
                errors.append(ValidationError(
                    f"Duplicate stream name: {stream.name}",
                    "streams",
                    "Use unique names for all streams"
                ))
            seen_names.add(stream.name)

        return errors

    def _validate_references(self) -> List[ValidationError]:
        """Validate that logic blocks reference valid streams."""
        errors = []
        all_stream_names = {s.name for s in self.streams}
        output_names = {s.name for s in self.streams if not s.is_input}

        for block in self.logic_blocks:
            for input_name in block.inputs:
                if input_name not in all_stream_names:
                    errors.append(ValidationError(
                        f"Logic block references unknown input stream: {input_name}",
                        f"{block.source_location}",
                        f"Available streams: {sorted(all_stream_names)}"
                    ))

            if block.output not in output_names:
                errors.append(ValidationError(
                    f"Logic block outputs to undefined stream: {block.output}",
                    f"{block.source_location}",
                    f"Available output streams: {sorted(output_names)}"
                ))

        return errors


class DSLParser:
    """Parser for AgentSchema DSL with formal validation."""

    def parse(self, schema: AgentSchema) -> ParsedAgentSchema:
        """Parse and validate an agent schema."""
        parsed = ParsedAgentSchema.from_agent_schema(schema)
        errors = parsed.validate()

        if errors:
            error_messages = [str(e) for e in errors]
            raise ValidationError(
                f"Schema validation failed with {len(errors)} errors:\n" +
                "\n".join(f"  - {msg}" for msg in error_messages),
                "schema",
                "Fix validation errors and re-parse"
            )

        return parsed

    def validate_schema(self, schema: AgentSchema) -> List[ValidationError]:
        """Validate schema without raising exceptions."""
        try:
            parsed = ParsedAgentSchema.from_agent_schema(schema)
            return parsed.validate()
        except Exception as e:
            return [ValidationError(f"Parse error: {e}", "schema")]
