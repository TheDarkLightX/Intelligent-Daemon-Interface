"""Output validator - checks agent outputs against expected patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig


@dataclass
class ValidationResult:
    """Result of output validation."""

    passed: bool
    checks: List[Tuple[str, bool, str]]  # (name, passed, message)


def _check_position_binary(outputs: Dict[str, List[str]], schema: AgentSchema) -> Optional[Tuple[str, bool, str]]:
    """Check that position outputs are binary (0 or 1)."""
    position_streams = [s for s in schema.streams if "position" in s.name.lower() and not s.is_input]
    
    for stream in position_streams:
        output_key = f"{stream.name}.out"
        if output_key in outputs:
            values = outputs[output_key]
            invalid = [v for v in values if v not in ("0", "1")]
            if invalid:
                return (
                    "position_binary",
                    False,
                    f"Position stream {stream.name} contains non-binary values: {invalid[:3]}",
                )
            return ("position_binary", True, f"Position stream {stream.name} is valid binary")
    
    return None


def _check_mutual_exclusion(outputs: Dict[str, List[str]], schema: AgentSchema) -> Optional[Tuple[str, bool, str]]:
    """Check that buy and sell signals are mutually exclusive."""
    buy_streams = [s for s in schema.streams if "buy" in s.name.lower() and not s.is_input]
    sell_streams = [s for s in schema.streams if "sell" in s.name.lower() and not s.is_input]
    
    if not buy_streams or not sell_streams:
        return None
    
    # Check first buy/sell pair
    buy_stream = buy_streams[0]
    sell_stream = sell_streams[0]
    
    buy_key = f"{buy_stream.name}.out"
    sell_key = f"{sell_stream.name}.out"
    
    if buy_key not in outputs or sell_key not in outputs:
        return None
    
    buy_values = outputs[buy_key]
    sell_values = outputs[sell_key]
    
    # Check lengths match
    if len(buy_values) != len(sell_values):
        return (
            "mutual_exclusion_length",
            False,
            f"Buy and sell streams have different lengths: {len(buy_values)} vs {len(sell_values)}",
        )
    
    # Check for simultaneous 1s
    violations = []
    for i, (b, s) in enumerate(zip(buy_values, sell_values)):
        if b == "1" and s == "1":
            violations.append(i)
    
    if violations:
        return (
            "mutual_exclusion",
            False,
            f"Buy and sell signals both 1 at timesteps: {violations[:5]}",
        )
    
    return ("mutual_exclusion", True, "Buy and sell signals are mutually exclusive")


def _check_output_lengths(outputs: Dict[str, List[str]], schema: AgentSchema) -> Optional[Tuple[str, bool, str]]:
    """Check that all outputs have the same length."""
    output_streams = [s for s in schema.streams if not s.is_input]
    
    if len(output_streams) < 2:
        return None
    
    lengths = {}
    for stream in output_streams:
        output_key = f"{stream.name}.out"
        if output_key in outputs:
            lengths[stream.name] = len(outputs[output_key])
    
    if len(set(lengths.values())) > 1:
        length_str = ", ".join(f"{name}: {len}" for name, len in lengths.items())
        return (
            "output_lengths",
            False,
            f"Output streams have different lengths: {length_str}",
        )
    
    return ("output_lengths", True, f"All outputs have length {list(lengths.values())[0] if lengths else 0}")


def _check_bitvector_ranges(outputs: Dict[str, List[str]], schema: AgentSchema) -> List[Tuple[str, bool, str]]:
    """Check that bitvector outputs are within valid ranges."""
    checks = []
    bv_streams = [s for s in schema.streams if s.stream_type == "bv" and not s.is_input]
    
    for stream in bv_streams:
        output_key = f"{stream.name}.out"
        if output_key not in outputs:
            continue
        
        max_value = (1 << stream.width) - 1
        values = outputs[output_key]
        
        invalid = []
        for i, val_str in enumerate(values):
            try:
                val = int(val_str)
                if val < 0 or val > max_value:
                    invalid.append((i, val))
            except ValueError:
                invalid.append((i, val_str))
        
        if invalid:
            checks.append((
                f"bv_range_{stream.name}",
                False,
                f"Stream {stream.name} (bv[{stream.width}]) has out-of-range values: {invalid[:3]}",
            ))
        else:
            checks.append((
                f"bv_range_{stream.name}",
                True,
                f"Stream {stream.name} values are within bv[{stream.width}] range [0, {max_value}]",
            ))
    
    return checks


def _check_required_outputs(outputs: Dict[str, List[str]], schema: AgentSchema) -> Optional[Tuple[str, bool, str]]:
    """Check that all expected output streams are present."""
    output_streams = [s for s in schema.streams if not s.is_input]
    expected_keys = {f"{s.name}.out" for s in output_streams}
    actual_keys = set(outputs.keys())
    
    missing = expected_keys - actual_keys
    if missing:
        return (
            "required_outputs",
            False,
            f"Missing output streams: {sorted(missing)[:5]}",
        )
    
    return ("required_outputs", True, f"All {len(expected_keys)} expected outputs present")


def validate_agent_outputs(
    outputs: Dict[str, List[str]],
    schema: AgentSchema,
) -> ValidationResult:
    """Validate agent outputs against schema expectations.
    
    Args:
        outputs: Dictionary mapping output file names to lists of values
        schema: The agent schema to validate against
        
    Returns:
        ValidationResult with all checks performed
    """
    checks = []
    
    # Check required outputs exist
    required_check = _check_required_outputs(outputs, schema)
    if required_check:
        checks.append(required_check)
    
    # Check output lengths match
    length_check = _check_output_lengths(outputs, schema)
    if length_check:
        checks.append(length_check)
    
    # Check position binary
    position_check = _check_position_binary(outputs, schema)
    if position_check:
        checks.append(position_check)
    
    # Check mutual exclusion
    exclusion_check = _check_mutual_exclusion(outputs, schema)
    if exclusion_check:
        checks.append(exclusion_check)
    
    # Check bitvector ranges
    bv_checks = _check_bitvector_ranges(outputs, schema)
    checks.extend(bv_checks)
    
    # If no checks were performed, that's a problem
    if not checks:
        checks.append(("no_checks", False, "No validation checks could be performed"))
    
    passed = all(check[1] for check in checks)
    
    return ValidationResult(passed=passed, checks=checks)

