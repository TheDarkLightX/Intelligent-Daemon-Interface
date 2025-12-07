"""Validate generated Tau specs for correctness before execution."""

from __future__ import annotations

from typing import List, Tuple


def validate_tau_spec(spec: str) -> Tuple[bool, List[str]]:
    """Validate a generated Tau spec for common issues.
    
    Args:
        spec: The Tau spec string to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for required keywords
    if "defs" not in spec:
        errors.append("Missing 'defs' keyword")
    
    if "r (" not in spec:
        errors.append("Missing recurrence block 'r ('")
    
    if not spec.strip().endswith("q"):
        errors.append("Missing 'q' quit command")
    
    # Check balanced parentheses
    open_parens = spec.count("(")
    close_parens = spec.count(")")
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
    
    # Check for proper I/O declarations
    if '= in file(' not in spec and '= in file("' not in spec:
        errors.append("No input file declarations found")
    
    if '= out file(' not in spec and '= out file("' not in spec:
        errors.append("No output file declarations found")
    
    # Check for proper stream naming (i0, i1, o0, o1, etc.)
    has_inputs = any(f"i{i}:sbf" in spec or f"i{i}:bv" in spec for i in range(10))
    has_outputs = any(f"o{i}:sbf" in spec or f"o{i}:bv" in spec for i in range(10))
    
    if not has_inputs:
        errors.append("No input streams found (i0, i1, etc.)")
    
    if not has_outputs:
        errors.append("No output streams found (o0, o1, etc.)")
    
    # Check for execution commands
    n_count = spec.count("\nn\n") + spec.count("n\n")
    if n_count == 0:
        errors.append("No execution steps ('n' commands) found")
    
    # Check for common syntax errors
    if "&&" in spec and "r (" in spec:
        # Check that && is used correctly in recurrence block
        r_block_start = spec.find("r (")
        r_block_end = spec.find(")", r_block_start)
        if r_block_end > r_block_start:
            r_block = spec[r_block_start:r_block_end]
            # Should have && between clauses, not at start/end
            if r_block.strip().startswith("&&"):
                errors.append("Recurrence block starts with && (should start with clause)")
            if r_block.strip().endswith("&&"):
                errors.append("Recurrence block ends with && (should end with clause)")
    
    # Check for proper initial conditions format
    if "[0]" in spec:
        # Initial conditions should use && syntax
        init_patterns = ["&& (", "&&("]
        has_proper_init = any(pattern in spec for pattern in init_patterns)
        if not has_proper_init and "[0]" in spec:
            # Might be okay, but warn if no && nearby
            pass
    
    return len(errors) == 0, errors

