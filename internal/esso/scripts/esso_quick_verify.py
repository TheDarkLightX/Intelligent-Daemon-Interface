#!/usr/bin/env python3
"""Quick verification of ESSO models with targeted SMT checks.

Uses z3-only mode for speed (skip cvc5 cross-check).

Usage:
    python3 esso_quick_verify.py [MODEL_NAME] [--full] [--timeout-ms MS]
    
Options:
    MODEL_NAME      Only verify specific model (without .req.yaml)
    --full          Use both z3 and cvc5 (slower but more thorough)
    --timeout-ms MS Per-query timeout (default: 3000)
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
from ESSO.verify.multi_solver import verify_ir_multi_solver, SolverResult


def verify_model(req_path: Path, timeout_ms: int, solvers: list[str]) -> dict:
    """Verify a single model with SMT.
    
    Returns:
        Result dict with verdict and query details
    """
    try:
        compiled = compile_req(
            req_path=req_path,
            semantics_profile=None,
            style_profile=None,
        )
        model = compiled.model
        
        results = verify_ir_multi_solver(
            model,
            timeout_ms=timeout_ms,
            solvers=solvers,
            produce_proofs=False,
            solver_seed=0,
        )
        
        passed = 0
        failed = 0
        unknown = 0
        details = []
        
        for query_name, result in results.items():
            if result.final_result == SolverResult.UNSAT:
                passed += 1
                details.append(f"  ✓ {query_name}")
            elif result.final_result == SolverResult.SAT:
                failed += 1
                details.append(f"  ✗ {query_name} (counterexample found)")
            else:
                unknown += 1
                details.append(f"  ? {query_name} ({result.final_result.value})")
        
        verdict = "PASS" if failed == 0 and unknown == 0 else ("FAIL" if failed > 0 else "UNKNOWN")
        
        return {
            "name": req_path.stem.replace(".req", ""),
            "verdict": verdict,
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "total": len(results),
            "details": details,
            "error": None,
        }
        
    except Exception as e:
        return {
            "name": req_path.stem.replace(".req", ""),
            "verdict": "ERROR",
            "passed": 0,
            "failed": 0,
            "unknown": 0,
            "total": 0,
            "details": [],
            "error": str(e),
        }


def main() -> int:
    timeout_ms = 3000
    solvers = ["z3"]  # Default: z3 only for speed
    model_filter = None
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--timeout-ms":
            if i + 1 >= len(args):
                print("Missing value for --timeout-ms")
                return 1
            timeout_ms = int(args[i + 1])
            i += 2
            continue
        if arg == "--full":
            solvers = ["z3", "cvc5"]
            i += 1
            continue
        if arg.startswith("--"):
            i += 1
            continue
        model_filter = arg
        i += 1
    
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    req_files = sorted(req_dir.glob("*.req.yaml"))
    
    if model_filter:
        req_files = [f for f in req_files if model_filter in f.stem]
        if not req_files:
            print(f"No matching REQ files for '{model_filter}'")
            return 1
    
    if not req_files:
        print(f"No REQ files found in {req_dir}")
        return 1
    
    print(f"Quick verification of {len(req_files)} models...")
    print(f"  solvers: {solvers}, timeout: {timeout_ms}ms")
    print("=" * 70)
    
    results = []
    for req_path in req_files:
        print(f"\n{req_path.stem.replace('.req', '')}:")
        result = verify_model(req_path, timeout_ms, solvers)
        results.append(result)
        
        if result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            for detail in result["details"]:
                print(detail)
            print(f"  → {result['verdict']} ({result['passed']}/{result['total']} queries passed)")
    
    print("\n" + "=" * 70)
    
    # Summary
    passed_models = sum(1 for r in results if r["verdict"] == "PASS")
    failed_models = sum(1 for r in results if r["verdict"] == "FAIL")
    unknown_models = sum(1 for r in results if r["verdict"] == "UNKNOWN")
    error_models = sum(1 for r in results if r["verdict"] == "ERROR")
    
    print(f"\nSummary:")
    print(f"  PASS: {passed_models}")
    print(f"  FAIL: {failed_models}")
    print(f"  UNKNOWN: {unknown_models}")
    print(f"  ERROR: {error_models}")
    
    if failed_models > 0:
        print(f"\n⚠️  {failed_models} model(s) have counterexamples - review invariants!")
    
    return 0 if failed_models == 0 and error_models == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
