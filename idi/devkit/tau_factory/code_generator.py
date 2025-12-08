"""Tau code generator tuned for original template correctness.

Implements the legacy output format expected by the test suite:
- Header comments
- Input/output declarations
- Optional input mirrors (when include_mirrors=True)
- defs / recurrence block / execution commands
This keeps monitoring hooks but favors correctness over modular templates.
"""

from __future__ import annotations

from typing import List, Dict
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock

from .dsl_parser import DSLParser
from .performance_monitor import monitor
# Import pattern generators dynamically to avoid circular imports
# We'll import the module and access functions as needed
import idi.devkit.tau_factory.generator as generator_module


class TauCodeGenerator:
    """Generate Tau specs matching legacy template expectations."""

    def __init__(self) -> None:
        self.parser = DSLParser()

    def generate(self, schema: AgentSchema) -> str:
        """Generate complete Tau spec from agent schema."""
        with monitor.measure("code_generation", schema_name=schema.name):
            parsed_schema = self.parser.parse(schema)
            lines: List[str] = []

            # Header
            lines.append(f"# {schema.name} Agent (Auto-generated)")
            lines.append(f"# Strategy: {schema.strategy}")
            lines.append("")

            inputs = [s for s in parsed_schema.streams if s.is_input]
            outputs = [s for s in parsed_schema.streams if not s.is_input]

            # Input declarations - use hex indices (i0, i1, ..., i9, iA, iB, ...)
            for idx, stream in enumerate(inputs):
                hex_idx = hex(idx)[2:].upper()
                type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
                lines.append(f'i{hex_idx}:{type_str} = in file("inputs/{stream.name}.in").')
            lines.append("")

            # Input mirrors (only if enabled) - use hex indices
            if schema.include_mirrors:
                for idx, stream in enumerate(inputs):
                    hex_idx = hex(idx)[2:].upper()
                    type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
                    lines.append(f'i{hex_idx}:{type_str} = out file("outputs/i{hex_idx}_mirror.out").')
                lines.append("")

            # Output declarations - use hex indices (o0, o1, ..., o9, oA, oB, ...)
            for idx, stream in enumerate(outputs):
                hex_idx = hex(idx)[2:].upper()
                type_str = f"bv[{stream.width}]" if stream.stream_type == "bv" else "sbf"
                lines.append(f'o{hex_idx}:{type_str} = out file("outputs/{stream.name}.out").')
            lines.append("")

            # defs and recurrence
            lines.append("defs")
            lines.append("r (")

            # Convert parsed blocks back to LogicBlock for pattern generators
            # Pattern generators expect LogicBlock with original schema streams
            for parsed_block in parsed_schema.logic_blocks:
                # Add pattern comment to match legacy output
                lines.append(f"% {parsed_block.pattern} pattern: {parsed_block.output} <- {', '.join(parsed_block.inputs)}")
                
                # Convert ParsedLogicBlock back to LogicBlock for compatibility
                logic_block = LogicBlock(
                    pattern=parsed_block.pattern,
                    inputs=tuple(parsed_block.inputs),
                    output=parsed_block.output,
                    params=parsed_block.params
                )
                
                # Use pattern generators from generator.py
                expr = self._generate_pattern_logic(logic_block, tuple(parsed_schema.streams))
                lines.append(f"    {expr}")

            lines.append(")")
            # Execution commands: n repeated num_steps, then q
            for _ in range(schema.num_steps):
                lines.append("n")
            lines.append("q")

            return "\n".join(lines)

    def _generate_pattern_logic(self, block: LogicBlock, streams: tuple[StreamConfig, ...]) -> str:
        """Generate logic expression using pattern generators from generator.py."""
        pattern = block.pattern.lower()
        
        # Map pattern names to generator function names
        pattern_map = {
            "fsm": "_generate_fsm_logic",
            "passthrough": "_generate_passthrough_logic",
            "counter": "_generate_counter_logic",
            "accumulator": "_generate_accumulator_logic",
            "vote": "_generate_vote_logic",
            "majority": "_generate_majority_logic",
            "unanimous": "_generate_unanimous_logic",
            "quorum": "_generate_quorum_logic",
            "custom": "_generate_custom_logic",
            "supervisor_worker": "_generate_supervisor_worker_logic",
            "weighted_vote": "_generate_weighted_vote_logic",
            "time_lock": "_generate_time_lock_logic",
            "hex_stake": "_generate_hex_stake_logic",
            "multi_bit_counter": "_generate_multi_bit_counter_logic",
            "streak_counter": "_generate_streak_counter_logic",
            "mode_switch": "_generate_mode_switch_logic",
            "proposal_fsm": "_generate_proposal_fsm_logic",
            "risk_fsm": "_generate_risk_fsm_logic",
            "entry_exit_fsm": "_generate_entry_exit_fsm_logic",
            "orthogonal_regions": "_generate_orthogonal_regions_logic",
            "state_aggregation": "_generate_state_aggregation_logic",
            "tcp_connection_fsm": "_generate_tcp_connection_fsm_logic",
            "utxo_state_machine": "_generate_utxo_state_machine_logic",
            "history_state": "_generate_history_state_logic",
            "decomposed_fsm": "_generate_decomposed_fsm_logic",
            "script_execution": "_generate_script_execution_logic",
        }
        
        try:
            func_name = pattern_map.get(pattern, "_generate_passthrough_logic")
            generator_func = getattr(generator_module, func_name)
            return generator_func(block, streams)
        except AttributeError:
            # Fallback to passthrough if pattern not found
            return generator_module._generate_passthrough_logic(block, streams)
        except Exception as e:
            raise ValueError(f"Error generating {pattern} pattern: {e}") from e
