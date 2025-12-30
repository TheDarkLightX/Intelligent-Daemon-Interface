#!/usr/bin/env python3
"""Validate all ESSO REQ specifications.

Usage:
    python3 esso_validate_all.py [--fix]
    
Options:
    --fix   Attempt to fix common issues (not implemented yet)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add ESSO to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "ESSO"))

from ESSO.foundry.compiler import compile_req
from ESSO.evolve import ir_hash


def validate_all(req_dir: Path, fix: bool = False) -> tuple[int, int]:
    """Validate all REQ files in directory.
    
    Returns:
        (passed, failed) counts
    """
    req_files = sorted(req_dir.glob("*.req.yaml"))
    
    if not req_files:
        print(f"No REQ files found in {req_dir}")
        return 0, 0
    
    print(f"Validating {len(req_files)} REQ specifications...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for req_path in req_files:
        try:
            compiled = compile_req(
                req_path=req_path,
                semantics_profile=None,
                style_profile=None,
            )
            h = ir_hash(compiled.model)[:16]
            n_actions = len(compiled.model.actions)
            n_vars = len(compiled.model.state_vars)
            n_inv = len(compiled.model.invariants)
            
            print(f"✓ {req_path.name}")
            print(f"    hash={h}... actions={n_actions} vars={n_vars} inv={n_inv}")
            passed += 1
            
        except Exception as e:
            print(f"✗ {req_path.name}")
            print(f"    ERROR: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return passed, failed


def main() -> int:
    fix = "--fix" in sys.argv
    
    # Repo-local REQ specs live under internal/esso/requirements.
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    
    passed, failed = validate_all(req_dir, fix=fix)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
