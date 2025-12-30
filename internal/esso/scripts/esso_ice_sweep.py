#!/usr/bin/env python3
"""Run ICE (Inductive Counterexample) strengthening on all models.

ICE is much faster than evolution and finds missing invariants.

Usage:
    python3 esso_ice_sweep.py [--timeout-ms MS] [--max-rounds N]
    
Options:
    --timeout-ms MS    Per-round solver timeout (default: 2000)
    --max-rounds N     Max ICE rounds per model (default: 10)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add ESSO to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "ESSO"))

import yaml
from ESSO.foundry.compiler import compile_req
from ESSO.verify.ice_loop import ICEConfig, strengthen_with_ice


def run_ice_on_model(req_path: Path, timeout_ms: int, max_rounds: int) -> dict:
    """Run ICE strengthening on a single model.
    
    Returns:
        Result dict with status and any new invariants
    """
    try:
        compiled = compile_req(
            req_path=req_path,
            semantics_profile=None,
            style_profile=None,
        )
        model = compiled.model
        
        config = ICEConfig(
            solver_random_seed=42,
            solver_timeout_ms=timeout_ms,
            max_ice_rounds=max_rounds,
            max_atoms=20,
            max_samples=100,
            max_tier=2,
        )
        
        strengthened_model, result = strengthen_with_ice(model, config=config)
        
        return {
            "name": req_path.stem.replace(".req", ""),
            "ok": result.ok,
            "rounds": result.rounds,
            "added_invariants": len(result.added_invariants),
            "new_invariants": [str(inv) for inv in result.added_invariants],
            "error": result.last_error,
        }
        
    except Exception as e:
        return {
            "name": req_path.stem.replace(".req", ""),
            "ok": False,
            "rounds": 0,
            "added_invariants": 0,
            "new_invariants": [],
            "error": str(e),
        }


def main() -> int:
    timeout_ms = 2000
    max_rounds = 10
    
    for i, arg in enumerate(sys.argv):
        if arg == "--timeout-ms" and i + 1 < len(sys.argv):
            timeout_ms = int(sys.argv[i + 1])
        if arg == "--max-rounds" and i + 1 < len(sys.argv):
            max_rounds = int(sys.argv[i + 1])
    
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    req_files = sorted(req_dir.glob("*.req.yaml"))
    
    if not req_files:
        print(f"No REQ files found in {req_dir}")
        return 1
    
    print(f"Running ICE strengthening on {len(req_files)} models...")
    print(f"  timeout: {timeout_ms}ms, max_rounds: {max_rounds}")
    print("=" * 70)
    
    results = []
    for req_path in req_files:
        print(f"Processing {req_path.name}...", end=" ", flush=True)
        result = run_ice_on_model(req_path, timeout_ms, max_rounds)
        results.append(result)
        
        if result["ok"]:
            if result["added_invariants"] > 0:
                print(f"✓ +{result['added_invariants']} invariants")
                for inv in result["new_invariants"]:
                    print(f"    NEW: {inv}")
            else:
                print("✓ already 1-inductive")
        else:
            print(f"✗ {result['error']}")
    
    print("=" * 70)
    
    # Summary
    total_new = sum(r["added_invariants"] for r in results)
    failed = sum(1 for r in results if not r["ok"])
    
    print(f"\nSummary:")
    print(f"  Models processed: {len(results)}")
    print(f"  New invariants found: {total_new}")
    print(f"  Failures: {failed}")
    
    if total_new > 0:
        print(f"\n⚠️  {total_new} new invariants found! Consider updating REQ specs.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
