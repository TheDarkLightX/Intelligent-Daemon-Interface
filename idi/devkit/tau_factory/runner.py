"""Tau runner - executes Tau specs and captures outputs.

Supports multiple execution modes:
- REPL/stdin mode: Pipes spec to tau binary via stdin (default)
- File mode: Passes spec file path as argument to tau binary
- Both modes support file-based I/O via ifile()/ofile() or = in file()/= out file()

Key Tau CLI options:
- -V, --charvar: Enable character variables (single-letter vars). Default: on
- -x, --experimental: Use new spec runner
- -S, --severity: Log level (trace/debug/info/error)
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable


_PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 0.5


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _validate_path(path: Path, description: str) -> tuple[Path, Optional[str]]:
    """Validate and canonicalize a path to prevent directory traversal.
    
    Returns:
        (resolved_path, error_message) - error_message is None if valid
    """
    try:
        resolved = path.resolve()
        if any(part == ".." for part in path.parts):
            return resolved, f"{description} contains directory traversal: {path}"
        return resolved, None
    except (OSError, ValueError) as e:
        return path, f"Invalid {description}: {e}"


def _should_fallback_to_file_mode(*, config: TauConfig, stderr: str, returncode: int) -> bool:
    if config.mode != TauMode.STDIN or config.experimental:
        return False
    if returncode != 2:
        return False

    clean_stderr = _strip_ansi(stderr).lower()
    required_tokens = ("usage:", "<spec", "tau")
    return all(token in clean_stderr for token in required_tokens)


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=_PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=_PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: str,
    timeout: float,
    stdin: Optional[object],
) -> subprocess.CompletedProcess[bytes]:
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    proc = subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        raise exc


class TauMode(Enum):
    """Tau execution mode."""
    STDIN = "stdin"   # Pipe spec via stdin (REPL mode)
    FILE = "file"     # Pass spec file as argument


@dataclass
class TauConfig:
    """Configuration for Tau execution.
    
    Attributes:
        charvar: Enable character variables (single-letter vars like x0, y1).
                 When False, descriptive names are allowed (input_signal, etc.)
        experimental: Use experimental spec runner
        severity: Log level (trace/debug/info/error)
        mode: Execution mode (stdin or file)
        timeout: Maximum execution time in seconds
    """
    charvar: bool = True
    experimental: bool = False
    severity: str = "info"
    mode: TauMode = TauMode.STDIN
    timeout: float = 30.0


@dataclass
class TauResult:
    """Result from running a Tau spec.
    
    Attributes:
        success: Whether execution completed without errors
        outputs: Dict mapping output filename to list of output values
        errors: List of error messages encountered
        duration_ms: Execution time in milliseconds
        stdout: Raw stdout from Tau (for debugging)
        stderr: Raw stderr from Tau (for debugging)
        returncode: Process return code
    """
    success: bool
    outputs: Dict[str, List[str]]
    errors: List[str]
    duration_ms: float
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _parse_output_files(outputs_dir: Path) -> Dict[str, List[str]]:
    """Parse output files from outputs directory.
    
    Returns dict mapping filename to list of output values (one per line).
    """
    outputs = {}
    if not outputs_dir.exists():
        return outputs
    
    for out_file in sorted(outputs_dir.glob("*.out")):
        try:
            content = out_file.read_text().strip()
            # Split by newlines, filter empty
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            outputs[out_file.name] = lines
        except Exception:
            continue
    
    return outputs


def _parse_errors(stdout: str, stderr: str, returncode: int) -> List[str]:
    """Parse error messages from stdout/stderr.
    
    Tau outputs errors to stdout in REPL mode with "(Error)" prefix.
    Also detects: Type mismatch, unsat specifications, unresolved symbols,
    failed stream binding, and syntax errors.
    """
    errors = []

    clean_stdout = _strip_ansi(stdout).replace("\r", "\n")
    clean_stderr = _strip_ansi(stderr).replace("\r", "\n")

    error_patterns = [
        "(Error",
        "Error)",
        "Syntax Error",
        "Type mismatch",
        "is unsat",
        "Unresolved function",
        "Failed to find output stream",
        "Failed to find input stream",
        "Failed to parse",
        "cannot contain a relative offset",
        "cannot depend on a future state",
    ]

    for line in clean_stdout.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if "[repl]" in stripped and "Error" in stripped:
            errors.append(stripped)
            continue
        for pattern in error_patterns:
            if pattern in stripped:
                errors.append(stripped)
                break

    if clean_stderr.strip():
        for line in clean_stderr.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if "Error" in stripped or "error" in stripped:
                errors.append(stripped)
    
    # Add returncode error if non-zero and no other errors
    if returncode != 0 and not errors:
        errors.append(f"Tau exited with code {returncode}")
    
    return errors


def _clear_output_files(outputs_dir: Path) -> None:
    """Clear existing output files before running spec."""
    if outputs_dir.exists():
        for out_file in outputs_dir.glob("*.out"):
            try:
                out_file.unlink()
            except Exception:
                pass


def find_tau_binary() -> Optional[Path]:
    """Find Tau binary in common locations.
    
    Returns Path to tau binary or None if not found.
    """
    # Check environment variable first
    if env_tau := os.environ.get("TAU_BIN"):
        path = Path(env_tau)
        if path.exists() and path.is_file():
            return path
    
    # Common locations relative to this file
    # runner.py is at idi/devkit/tau_factory/runner.py
    # IDI workspace is 4 levels up
    this_file = Path(__file__)
    workspace_candidates = [
        this_file.parent.parent.parent.parent.parent,  # IDI workspace
        this_file.parent.parent.parent.parent.parent.parent,  # parent of IDI
    ]
    
    search_paths = []
    for ws in workspace_candidates:
        search_paths.extend([
            ws / "external" / "tau-lang" / "build-Release" / "tau",
            ws / "tau-lang" / "build-Release" / "tau",
        ])
    
    # Also check PATH and common system locations
    search_paths.extend([
        Path("/usr/local/bin/tau"),
        Path("/usr/bin/tau"),
        Path.home() / "Downloads" / "IDI" / "external" / "tau-lang" / "build-Release" / "tau",
    ])
    
    for path in search_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


def run_tau_spec(
    spec_path: Path,
    tau_bin: Optional[Path] = None,
    config: Optional[TauConfig] = None,
    timeout: Optional[float] = None,
) -> TauResult:
    """Run a Tau spec file and capture outputs.
    
    Args:
        spec_path: Path to the .tau spec file
        tau_bin: Path to Tau binary (auto-detected if None)
        config: Tau configuration options
        timeout: Override timeout (uses config.timeout if None)
        
    Returns:
        TauResult with success status, outputs, errors, and duration
        
    Preconditions:
        - spec_path must exist and be readable
        - tau_bin must exist and be executable (or auto-detectable)
        
    Postconditions:
        - TauResult.outputs contains all output files created
        - TauResult.errors is empty iff TauResult.success is True
    """
    config = config or TauConfig()
    timeout = timeout or config.timeout
    
    start_time = time.perf_counter()
    
    # Validate and canonicalize spec_path first
    spec_path, path_error = _validate_path(spec_path, "spec_path")
    if path_error:
        return TauResult(
            success=False,
            outputs={},
            errors=[path_error],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    # Auto-detect tau binary if not provided
    if tau_bin is None:
        tau_bin = find_tau_binary()
        if tau_bin is None:
            return TauResult(
                success=False,
                outputs={},
                errors=["Tau binary not found. Set TAU_BIN env var or build tau-lang."],
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    # Validate and canonicalize tau_bin
    tau_bin, bin_error = _validate_path(tau_bin, "tau_bin")
    if bin_error:
        return TauResult(
            success=False,
            outputs={},
            errors=[bin_error],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    # Validate inputs exist
    if not tau_bin.exists():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Tau binary not found: {tau_bin}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    if not tau_bin.is_file():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Tau binary path is not a file: {tau_bin}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    if not os.access(tau_bin, os.X_OK):
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Tau binary is not executable: {tau_bin}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    if not spec_path.exists():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Spec file not found: {spec_path}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    if not spec_path.is_file():
        return TauResult(
            success=False,
            outputs={},
            errors=[f"Spec path is not a file: {spec_path}"],
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    # Setup directories
    outputs_dir = spec_path.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _clear_output_files(outputs_dir)  # Clear old outputs
    
    # Build command line arguments
    cmd = [str(tau_bin)]

    # Tau expects: tau <specification file> [ options ]
    if config.mode == TauMode.FILE:
        cmd.append(str(spec_path))
    
    if not config.charvar:
        cmd.extend(["--charvar", "off"])
    
    if config.experimental:
        cmd.append("-x")
    
    if config.severity != "info":
        cmd.extend(["-S", config.severity])
    
    try:
        if config.mode == TauMode.STDIN:
            # REPL/stdin mode: pipe spec via stdin
            with spec_path.open("rb") as spec_file:
                proc = _run_subprocess(
                    cmd,
                    cwd=str(spec_path.parent),
                    timeout=timeout,
                    stdin=spec_file,
                )
        else:
            # File mode: spec path already in cmd
            proc = _run_subprocess(
                cmd,
                cwd=str(spec_path.parent),
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        
        stdout = proc.stdout.decode("utf-8", errors="ignore")
        stderr = proc.stderr.decode("utf-8", errors="ignore")

        # Some Tau builds only support: tau <spec.tau>. If so, retry in FILE mode.
        if _should_fallback_to_file_mode(config=config, stderr=stderr, returncode=proc.returncode):
            _clear_output_files(outputs_dir)

            cmd_file = [str(tau_bin), str(spec_path)]
            if not config.charvar:
                cmd_file.extend(["--charvar", "off"])
            if config.experimental:
                cmd_file.append("-x")
            if config.severity != "info":
                cmd_file.extend(["-S", config.severity])

            proc = _run_subprocess(
                cmd_file,
                cwd=str(spec_path.parent),
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            stdout = proc.stdout.decode("utf-8", errors="ignore")
            stderr = proc.stderr.decode("utf-8", errors="ignore")

        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse outputs
        outputs = _parse_output_files(outputs_dir)
        
        # Parse errors from both stdout and stderr
        errors = _parse_errors(stdout, stderr, proc.returncode)

        if config.mode == TauMode.FILE:
            clean_stdout = _strip_ansi(stdout).replace("\r", "\n")
            if clean_stdout.lstrip().startswith("tau>"):
                errors.append(
                    "Tau entered REPL in FILE mode; invocation likely incorrect (expected: tau <specfile> [options])."
                )
        
        success = proc.returncode == 0 and len(errors) == 0
        
        return TauResult(
            success=success,
            outputs=outputs,
            errors=errors,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
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


def verify_outputs(
    result: TauResult,
    expected: Dict[str, List[str]],
) -> tuple[bool, List[str]]:
    """Verify Tau outputs match expected values.
    
    Args:
        result: TauResult from run_tau_spec
        expected: Dict mapping output filename to expected values
        
    Returns:
        (all_match, list_of_mismatches)
        
    Example:
        >>> result = run_tau_spec(spec_path, tau_bin)
        >>> ok, mismatches = verify_outputs(result, {"decision.out": ["0", "1", "1", "0"]})
    """
    mismatches = []
    
    for filename, expected_values in expected.items():
        if filename not in result.outputs:
            mismatches.append(f"Missing output file: {filename}")
            continue
        
        actual_values = result.outputs[filename]
        
        if len(actual_values) != len(expected_values):
            mismatches.append(
                f"{filename}: length mismatch - expected {len(expected_values)}, "
                f"got {len(actual_values)}"
            )
            continue
        
        for i, (exp, act) in enumerate(zip(expected_values, actual_values)):
            if exp != act:
                mismatches.append(
                    f"{filename}[{i}]: expected '{exp}', got '{act}'"
                )
    
    # Check for unexpected outputs
    for filename in result.outputs:
        if filename not in expected:
            mismatches.append(f"Unexpected output file: {filename}")
    
    return len(mismatches) == 0, mismatches


def exhaustive_verify(
    spec_generator: Callable[[Dict[str, List[str]]], str],
    input_combinations: List[Dict[str, List[str]]],
    oracle: Callable[[Dict[str, List[str]]], Dict[str, List[str]]],
    tau_bin: Optional[Path] = None,
    config: Optional[TauConfig] = None,
) -> tuple[bool, List[str]]:
    """Exhaustively verify a spec against all input combinations.
    
    This is a formal verification helper that tests all possible states.
    
    Args:
        spec_generator: Function that generates spec string from inputs
        input_combinations: List of input dicts to test
        oracle: Function that computes expected outputs from inputs
        tau_bin: Path to Tau binary
        config: Tau configuration
        
    Returns:
        (all_passed, list_of_failures)
        
    Example:
        >>> def oracle(inputs):
        ...     # Majority vote logic
        ...     v1 = [int(x) for x in inputs["v1.in"]]
        ...     v2 = [int(x) for x in inputs["v2.in"]]
        ...     v3 = [int(x) for x in inputs["v3.in"]]
        ...     return {"decision.out": [str((a+b+c) >= 2) for a,b,c in zip(v1,v2,v3)]}
        >>> exhaustive_verify(gen_spec, all_combos, oracle)
    """
    import tempfile
    
    failures = []
    
    for i, inputs in enumerate(input_combinations):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            inputs_dir = tmp / "inputs"
            outputs_dir = tmp / "outputs"
            inputs_dir.mkdir()
            outputs_dir.mkdir()
            
            # Write input files
            for filename, values in inputs.items():
                (inputs_dir / filename).write_text("\n".join(values) + "\n")
            
            # Generate and write spec
            spec = spec_generator(inputs)
            spec_path = tmp / "test.tau"
            spec_path.write_text(spec)
            
            # Run spec
            result = run_tau_spec(spec_path, tau_bin, config)
            
            if not result.success:
                failures.append(f"Case {i}: Execution failed - {result.errors}")
                continue
            
            # Compute expected outputs
            expected = oracle(inputs)
            
            # Verify
            ok, mismatches = verify_outputs(result, expected)
            if not ok:
                failures.append(f"Case {i}: {mismatches}")
    
    return len(failures) == 0, failures

