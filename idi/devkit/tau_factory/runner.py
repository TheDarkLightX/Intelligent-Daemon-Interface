"""Tau runner - executes Tau specs and captures outputs."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TauResult:
    """Result from running a Tau spec."""

    success: bool
    outputs: Dict[str, List[str]]
    errors: List[str]
    duration_ms: float


def _parse_output_files(outputs_dir: Path) -> Dict[str, List[str]]:
    """Parse output files from outputs directory."""
    outputs = {}
    if not outputs_dir.exists():
        return outputs
    
    for out_file in sorted(outputs_dir.glob("*.out")):
        try:
            content = out_file.read_text().strip()
            # Split by newlines, filter empty
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            outputs[out_file.name] = lines
        except Exception as e:
            # Skip files that can't be read
            continue
    
    return outputs


def _parse_errors(stderr: bytes, returncode: int) -> List[str]:
    """Parse error messages from stderr."""
    errors = []
    
    if returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="ignore")
        if stderr_text.strip():
            # Extract error lines (lines containing "Error" or starting with "(")
            for line in stderr_text.split("\n"):
                line = line.strip()
                if "Error" in line or line.startswith("(") and "Error" in line:
                    errors.append(line)
        
        if not errors:
            errors.append(f"Tau exited with code {returncode}")
    
    return errors


def run_tau_spec(spec_path: Path, tau_bin: Path, timeout: float = 30.0) -> TauResult:
    """Run a Tau spec file and capture outputs.
    
    Args:
        spec_path: Path to the .tau spec file
        tau_bin: Path to the Tau binary executable
        timeout: Maximum execution time in seconds
        
    Returns:
        TauResult with success status, outputs, errors, and duration
    """
    start_time = time.perf_counter()
    outputs_dir = spec_path.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if tau binary exists
    if not tau_bin.exists():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Tau binary not found: {tau_bin}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    # Check if spec file exists
    if not spec_path.exists():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Spec file not found: {spec_path}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    try:
        # Run Tau with spec piped via stdin
        with spec_path.open("rb") as spec_file:
            proc = subprocess.run(
                [str(tau_bin)],
                stdin=spec_file,
                capture_output=True,
                timeout=timeout,
                cwd=str(spec_path.parent),
            )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse outputs
        outputs = _parse_output_files(outputs_dir)
        
        # Parse errors
        errors = _parse_errors(proc.stderr, proc.returncode)
        
        success = proc.returncode == 0 and len(errors) == 0
        
        return TauResult(
            success=success,
            outputs=outputs,
            errors=errors,
            duration_ms=duration_ms,
        )
    
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - start_time) * 1000
        outputs = _parse_output_files(outputs_dir)
        return TauResult(
            success=False,
            outputs=outputs,
            errors=[f"Execution timed out after {timeout}s"],
            duration_ms=duration_ms,
        )
    
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Execution failed: {str(e)}"],
            duration_ms=duration_ms,
        )

