"""Validate generated Tau specs for correctness before execution."""

from __future__ import annotations

from typing import List, Tuple


def validate_tau_spec(spec: str) -> Tuple[bool, List[str]]:
    """Validate a generated Tau spec for common issues.
    
    Tau Language REPL syntax (verified via tau.tgf grammar):
    - Input: varname:type = in file("path")
    - Output: varname:type = out file("path")
    - Run: r <wff> (formula directly, no block wrapper)
    - Stepping: Empty lines advance execution
    - Quit: q
    
    Args:
        spec: The Tau spec string to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for run command (r <wff>)
    if "r " not in spec:
        errors.append("Missing run command 'r <wff>'")
    
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
    
    # Check for execution steps (empty lines between 'r' command and 'q')
    # In Tau REPL, empty lines advance execution, NOT 'n' commands
    lines = spec.split('\n')
    r_idx = -1
    q_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('r '):
            r_idx = i
        if line.strip() == 'q':
            q_idx = i
    
    if r_idx >= 0 and q_idx > r_idx:
        # Count empty lines between r command and q (these are execution steps)
        empty_count = sum(1 for line in lines[r_idx+1:q_idx] if line.strip() == '')
        if empty_count == 0:
            errors.append("No execution steps (empty lines between 'r' and 'q') found")
    
    # Check for common syntax errors in run command
    if "r " in spec:
        # Find the run command line
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('r '):
                wff = stripped[2:]  # Everything after 'r '
                # Check that WFF doesn't start or end with &&
                if wff.strip().startswith('&&'):
                    errors.append("Run command WFF starts with && (should start with clause)")
                if wff.strip().endswith('&&'):
                    errors.append("Run command WFF ends with && (should end with clause)")
                break
    
    return len(errors) == 0, errors

