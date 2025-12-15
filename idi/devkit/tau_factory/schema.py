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
        valid_patterns = (
            # Original patterns (26)
            "fsm", "counter", "accumulator", "vote", "passthrough", "majority", 
            "unanimous", "custom", "quorum", "supervisor_worker", "weighted_vote", 
            "time_lock", "hex_stake", "multi_bit_counter", "streak_counter", 
            "mode_switch", "proposal_fsm", "risk_fsm", "entry_exit_fsm", 
            "orthogonal_regions", "state_aggregation", "tcp_connection_fsm", 
            "utxo_state_machine", "history_state", "decomposed_fsm", "script_execution",
            # Signal Processing patterns
            "edge_detector", "falling_edge", "toggle", "latch", "debounce",
            "pulse_generator", "sample_hold",
            # Data Flow / Routing patterns
            "multiplexer", "demultiplexer", "priority_encoder", "arbiter",
            # Safety / Watchdog patterns
            "watchdog", "deadman_switch", "safety_interlock", "fault_detector",
            # Protocol / Handshake patterns
            "handshake", "sync_barrier", "token_ring",
            # Arithmetic / Comparison patterns
            "comparator", "min_selector", "max_selector", "threshold_detector",
            # Consensus / Distributed patterns
            "byzantine_fault_tolerant", "leader_election", "commit_protocol",
            # Gate / Logic patterns
            "nand_gate", "nor_gate", "xnor_gate", "implication", "equivalence",
            # Timing / Delay patterns
            "delay_line", "hold", "one_shot",
            # State Encoding patterns
            "gray_code", "ring_counter", "sequence_detector",
            # Intelligent Agent - Decision Making patterns
            "confidence_gate", "action_selector", "exploration_exploit",
            "reward_accumulator", "goal_detector", "obstacle_detector", "policy_switch",
            # Intelligent Agent - Learning & Memory patterns
            "experience_buffer", "learning_gate", "attention_focus",
            # Intelligent Agent - Coordination & Communication patterns
            "consensus_vote", "broadcast", "message_filter", "role_assignment",
            # Intelligent Agent - Safety & Constraints patterns
            "action_mask", "safety_override", "constraint_checker", "budget_gate",
            # Intelligent Agent - Inference & Prediction patterns
            "prediction_gate", "anomaly_detector", "state_classifier",
            # Agent Fairness & Coordination patterns
            "xor_combine", "commitment_match", "all_revealed", "phase_gate",
            "collision_detect", "turn_gate", "fair_priority", "streak_detector",
            "combo_counter", "cooldown", "capture", "territory_control",
            "simultaneous_action", "any_action", "exclusive_action", "valid_move",
            "win_condition", "game_over", "score_gate", "bonus_trigger",
            # Formal Verification Invariant patterns
            "mutual_exclusion", "never_unsafe", "request_response", "recovery",
            "no_starvation", "stabilization", "consensus_check", "progress",
            "bounded_until", "trust_update", "reputation_gate", "risk_gate",
            "belief_consistency", "exploration_decay", "safe_exploration",
            "emergent_detector", "utility_alignment", "causal_gate", "counterfactual_safe",
        )
        if self.pattern not in valid_patterns:
            raise ValueError(f"Invalid pattern: {self.pattern}. Must be one of {valid_patterns}")


@dataclass(frozen=True)
class AgentSchema:
    """Complete schema for generating a Tau agent spec.
    
    Attributes:
        name: Agent name for comments/documentation
        strategy: Trading/logic strategy type
        streams: Input and output stream configurations
        logic_blocks: Logic pattern blocks defining behavior
        num_steps: Number of execution steps
        include_mirrors: Whether to mirror inputs to output files
        descriptive_names: Use human-readable variable names instead of i0/o0.
                          Requires Tau to run with --charvar off.
                          When True, stream names like 'buy_signal' are used directly.
                          When False (default), compact names like 'i0', 'o0' are used.
    """

    name: str
    strategy: Literal["momentum", "mean_reversion", "regime_aware", "custom"]
    streams: Tuple[StreamConfig, ...]
    logic_blocks: Tuple[LogicBlock, ...]
    num_steps: int = 10
    include_mirrors: bool = False  # Disabled by default - mirrors require WFF wiring
    descriptive_names: bool = False  # Use readable names (requires charvar=off)

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
    if not schema.name.strip():
        raise ValueError("Schema name cannot be empty")

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

    # Cross-reference validation
    all_stream_names = {s.name for s in schema.streams}
    declared_outputs = {s.name for s in schema.streams if not s.is_input}
    for block in schema.logic_blocks:
        for input_name in block.inputs:
            if input_name not in all_stream_names:
                raise ValueError(f"Logic block references unknown input stream: {input_name}")
        if block.output not in declared_outputs:
            raise ValueError(f"Logic block outputs to undefined stream: {block.output}")

