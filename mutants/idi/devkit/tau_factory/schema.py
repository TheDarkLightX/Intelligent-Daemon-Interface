"""AgentSchema definitions for parameterized Tau agent generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Tuple


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for a single input or output stream."""

    name: str
    stream_type: Literal["sbf", "bv"]
    width: int = 8  # For bv[N], ignored for sbf
    is_input: bool = True

    def __post_init__(self):
        """Validate stream configuration."""
        if self.stream_type not in ("sbf", "bv"):
            raise ValueError(f"Invalid stream_type: {self.stream_type}")
        if self.stream_type == "bv":
            if self.width is None:
                raise ValueError("BV streams must specify width")
            if self.width < 1:
                raise ValueError(f"BV streams must have width >= 1, got {self.width}")
            if self.width > 32:
                raise ValueError(f"BV streams max width is 32, got {self.width}")


@dataclass(frozen=True)
class LogicBlock:
    """A reusable logic pattern block."""

    pattern: Literal["fsm", "counter", "accumulator", "vote", "passthrough", "majority", "unanimous", "custom", "quorum", "supervisor_worker", "weighted_vote", "time_lock", "hex_stake", "multi_bit_counter", "streak_counter", "mode_switch", "proposal_fsm", "risk_fsm", "entry_exit_fsm", "orthogonal_regions", "state_aggregation", "tcp_connection_fsm", "utxo_state_machine", "history_state", "decomposed_fsm", "script_execution"]
    inputs: Tuple[str, ...]
    output: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate logic block."""
        valid_patterns = ("fsm", "counter", "accumulator", "vote", "passthrough", "majority", "unanimous", "custom", "quorum", "supervisor_worker", "weighted_vote", "time_lock", "hex_stake", "multi_bit_counter", "streak_counter", "mode_switch", "proposal_fsm", "risk_fsm", "entry_exit_fsm", "orthogonal_regions", "state_aggregation", "tcp_connection_fsm", "utxo_state_machine", "history_state", "decomposed_fsm", "script_execution")
        if self.pattern not in valid_patterns:
            raise ValueError(f"Invalid pattern: {self.pattern}. Must be one of {valid_patterns}")


@dataclass(frozen=True)
class AgentSchema:
    """Complete schema for generating a Tau agent spec."""

    name: str
    strategy: Literal["momentum", "mean_reversion", "regime_aware", "custom"]
    streams: Tuple[StreamConfig, ...]
    logic_blocks: Tuple[LogicBlock, ...]
    num_steps: int = 10
    include_mirrors: bool = True

    def __post_init__(self):
        """Basic field validation only - comprehensive validation moved to DSL parser."""
        # Only validate basic field constraints, not cross-references
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")


def validate_schema(schema: AgentSchema) -> None:
    """Validate an AgentSchema for correctness.

    Raises:
        ValueError: If schema is invalid.
    """
    # Additional validation beyond __post_init__
    if not schema.streams:
        raise ValueError("Schema must have at least one stream")
    if not schema.logic_blocks:
        raise ValueError("Schema must have at least one logic block")

    # Check for duplicate stream names
    stream_names = [s.name for s in schema.streams]
    if len(stream_names) != len(set(stream_names)):
        raise ValueError("Duplicate stream names found")

    # Check for duplicate output names in logic blocks
    output_names = [b.output for b in schema.logic_blocks]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Multiple logic blocks output to the same stream")

