#!/usr/bin/env python3
"""
Run every Tau specification under specification/ through the native Tau
interpreter, capture execution logs, and summarize generated traces.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "specification"
RUN_ROOT = ROOT / "outputs" / "tau_runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
TAU_BIN = ROOT / "tau_daemon_alpha" / "bin" / "tau"
SHARED_INPUTS = SPEC_ROOT / "inputs"
TIMEOUT = 300  # seconds

# Tau daemon binary currently expects legacy "sbf i0 = ifile(...)." syntax.
# Most specs were normalized to "i0 : sbf = in file(...)." for readability.
# Convert back to the legacy format when producing run copies.
NEW_INPUT_RE = re.compile(
    r'(?m)^(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<type>[A-Za-z0-9_\[\]:]+)\s*=\s*in\s+file\((?P<path>[^)]+)\)\.'
)
NEW_OUTPUT_RE = re.compile(
    r'(?m)^(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<type>[A-Za-z0-9_\[\]:]+)\s*=\s*out\s+file\((?P<path>[^)]+)\)\.'
)


def ensure_symlink(target: Path, source: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.symlink_to(source.resolve())


def line_count(path: Path) -> int:
    try:
        with path.open() as fh:
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


def gather_outputs(outputs_dir: Path) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    if outputs_dir.exists():
        for out_file in sorted(outputs_dir.glob("*.out")):
            summary[out_file.name] = line_count(out_file)
    return summary


def scan_errors(log_text: str) -> List[str]:
    errors: List[str] = []
    for line in log_text.splitlines():
        if "Error" in line or "unsat" in line:
            errors.append(line.strip())
    return errors


@dataclass
class SpecRunResult:
    spec: str
    status: str
    duration_sec: float
    outputs: Dict[str, int]
    log_path: str
    errors: List[str]


def resolve_spec_paths(selections: List[str]) -> List[Path]:
    if not selections:
        return sorted(SPEC_ROOT.rglob("*.tau"))

    resolved: List[Path] = []
    for selection in selections:
        raw = Path(selection)
        if not raw.is_absolute():
            raw = (ROOT / selection).resolve()
        if not raw.exists():
            print(f"[tau-run] Selection not found: {selection}", file=sys.stderr)
            sys.exit(1)
        if raw.is_dir():
            resolved.extend(sorted(raw.rglob("*.tau")))
        elif raw.suffix == ".tau":
            resolved.append(raw)
        else:
            print(f"[tau-run] Selection must be .tau file or directory: {selection}", file=sys.stderr)
            sys.exit(1)

    # Preserve deterministic order based on relative path
    unique = []
    seen = set()
    for path in resolved:
        rel = path.relative_to(ROOT)
        if rel not in seen:
            seen.add(rel)
            unique.append(path)
    return unique


def convert_io_directives(text: str) -> str:
    """Normalize IO declarations to the legacy syntax required by tau."""

    def repl_input(match: re.Match[str]) -> str:
        return f"{match.group('type')} {match.group('name')} = ifile({match.group('path')})."

    def repl_output(match: re.Match[str]) -> str:
        return f"{match.group('type')} {match.group('name')} = ofile({match.group('path')})."

    updated = NEW_INPUT_RE.sub(repl_input, text)
    updated = NEW_OUTPUT_RE.sub(repl_output, updated)
    return updated


def run_spec(spec_path: Path) -> SpecRunResult:
    rel_spec = spec_path.relative_to(ROOT)
    run_dir = RUN_ROOT / rel_spec.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    run_spec_path = run_dir / spec_path.name
    raw_text = spec_path.read_text()
    processed_text = convert_io_directives(raw_text)
    run_spec_path.write_text(processed_text)

    # Inputs: prefer spec-local, then shared specification/inputs
    inputs_src: Optional[Path] = None
    local_inputs = spec_path.parent / "inputs"
    if local_inputs.exists():
        inputs_src = local_inputs
    elif SHARED_INPUTS.exists():
        inputs_src = SHARED_INPUTS

    if inputs_src is not None:
        ensure_symlink(run_dir / "inputs", inputs_src)

    outputs_dir = run_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir()

    log_path = run_dir / f"{spec_path.stem}.tau.log"
    start = time.time()
    status = "ok"
    try:
        with log_path.open("w") as log_fh:
            proc = subprocess.run(
                [str(TAU_BIN), run_spec_path.name],
                cwd=run_dir,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        if proc.returncode != 0:
            status = f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        with log_path.open("a") as log_fh:
            log_fh.write(f"\nTIMEOUT after {TIMEOUT} seconds\n")
    duration = time.time() - start

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    outputs_summary = gather_outputs(outputs_dir)
    errors = scan_errors(log_text)

    return SpecRunResult(
        spec=str(rel_spec),
        status=status,
        duration_sec=duration,
        outputs=outputs_summary,
        log_path=str(log_path.relative_to(ROOT)),
        errors=errors,
    )


def main(argv: List[str]) -> None:
    if not TAU_BIN.exists():
        print(f"Tau binary not found at {TAU_BIN}", file=sys.stderr)
        sys.exit(1)

    spec_files = resolve_spec_paths(argv)
    if not spec_files:
        print("[tau-run] No specifications matched selection", file=sys.stderr)
        sys.exit(1)

    results: List[SpecRunResult] = []

    for spec in spec_files:
        print(f"[tau-run] Executing {spec.relative_to(ROOT)}")
        result = run_spec(spec)
        results.append(result)

    summary_path = RUN_ROOT / "summary.json"
    with summary_path.open("w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])

