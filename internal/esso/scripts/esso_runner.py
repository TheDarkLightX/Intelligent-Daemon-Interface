#!/usr/bin/env python3
"""ESSO Master Runner - One-stop interface for all ESSO operations.

Usage:
    python3 esso_runner.py <command> [options]

Commands:
    validate    Validate all REQ specifications
    compile     Compile REQs to Python kernels
    analyze     Analyze model complexity and evolution feasibility
    ice         Run ICE strengthening (fast invariant discovery)
    verify      Quick verification with SMT
    evolve      Run evolution on suitable models
    all         Run full pipeline: validate → analyze → ice → compile

Examples:
    python3 esso_runner.py validate
    python3 esso_runner.py compile --only mempool_lifecycle
    python3 esso_runner.py analyze --json
    python3 esso_runner.py ice --timeout-ms 3000
    python3 esso_runner.py verify fraud_proof
    python3 esso_runner.py evolve --generations 3
    python3 esso_runner.py all
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_script(name: str, args: list[str]) -> int:
    """Run an ESSO script with given arguments."""
    script_path = SCRIPT_DIR / f"esso_{name}.py"
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return 1
    
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd)
    return result.returncode


def cmd_validate(args: list[str]) -> int:
    return run_script("validate_all", args)


def cmd_compile(args: list[str]) -> int:
    return run_script("compile_all", args)


def cmd_analyze(args: list[str]) -> int:
    return run_script("analyze", args)


def cmd_ice(args: list[str]) -> int:
    return run_script("ice_sweep", args)


def cmd_verify(args: list[str]) -> int:
    return run_script("quick_verify", args)


def cmd_evolve(args: list[str]) -> int:
    return run_script("evolve_fast", args)


def cmd_all(args: list[str]) -> int:
    """Run full pipeline."""
    print("=" * 70)
    print("ESSO Full Pipeline")
    print("=" * 70)
    
    steps = [
        ("validate", "Validating REQ specifications..."),
        ("analyze", "Analyzing model complexity..."),
        ("ice_sweep", "Running ICE strengthening..."),
        ("compile_all", "Compiling kernels..."),
    ]
    
    results = []
    for script, desc in steps:
        print(f"\n{'='*70}")
        print(f"STEP: {desc}")
        print("=" * 70)
        
        code = run_script(script, [])
        results.append((script, code))
        
        if code != 0:
            print(f"\n⚠️  Step '{script}' failed with code {code}")
    
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    
    for script, code in results:
        status = "✓ PASS" if code == 0 else f"✗ FAIL (code {code})"
        print(f"  {script}: {status}")
    
    failed = sum(1 for _, c in results if c != 0)
    return 1 if failed > 0 else 0


def print_help():
    print(__doc__)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        return 0
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    commands = {
        "validate": cmd_validate,
        "compile": cmd_compile,
        "analyze": cmd_analyze,
        "ice": cmd_ice,
        "verify": cmd_verify,
        "evolve": cmd_evolve,
        "all": cmd_all,
    }
    
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {list(commands.keys())}")
        return 1
    
    return commands[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
