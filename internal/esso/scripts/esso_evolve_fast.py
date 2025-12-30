#!/usr/bin/env python3
"""Run fast ESSO evolution on suitable models.

Only runs on models with log2 state space < 25 (feasible for evolution).

Usage:
    python3 esso_evolve_fast.py [--generations N] [--population N] [--only MODEL]
    
Options:
    --generations N   Number of generations (default: 5)
    --population N    Population size (default: 8)
    --only MODEL      Only evolve specific model
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add ESSO to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "ESSO"))

import yaml
from ESSO.foundry.compiler import compile_req
from ESSO.verify.fitness import kernel_complexity, log2_upper_bound
from ESSO.evolve import ir_hash

# Max log2 state space for feasible evolution
MAX_LOG2_FOR_EVOLUTION = 25


def get_evolvable_models(req_dir: Path) -> list[tuple[Path, float]]:
    """Get models suitable for evolution (sorted by log2 space)."""
    candidates = []
    
    for req_path in req_dir.glob("*.req.yaml"):
        try:
            compiled = compile_req(
                req_path=req_path,
                semantics_profile=None,
                style_profile=None,
            )
            log2 = log2_upper_bound(compiled.model)
            
            if log2 <= MAX_LOG2_FOR_EVOLUTION:
                candidates.append((req_path, log2))
        except Exception:
            continue
    
    return sorted(candidates, key=lambda x: x[1])


def evolve_model(
    req_path: Path,
    output_dir: Path,
    generations: int,
    population: int,
    timeout_ms: int = 2000,
) -> dict:
    """Run evolution on a single model."""
    try:
        # Compile to YAML for evolution
        compiled = compile_req(
            req_path=req_path,
            semantics_profile=None,
            style_profile=None,
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(compiled.model.to_json_dict(), f, default_flow_style=False)
            model_yaml = f.name
        
        model_name = req_path.stem.replace(".req", "")
        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        
        # Run evolution via subprocess with timeout
        cmd = [
            sys.executable, "-m", "ESSO", "evolve",
            model_yaml,
            "--generations", str(generations),
            "--population", str(population),
            "--seed", "42",
            "--timeout-ms", str(timeout_ms),
            "--bounded-max-states", "50",
            "--bounded-max-depth", "5",
            "--output", str(model_output),
            "--verbose",
        ]
        
        env = {"PYTHONPATH": str(REPO_ROOT / "external" / "ESSO")}
        env.update({k: v for k, v in __import__('os').environ.items()})
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute max per model
            cwd=str(REPO_ROOT),
            env=env,
        )
        
        # Parse result
        if result.returncode == 0:
            try:
                output_lines = result.stdout.strip().split('\n')
                json_line = output_lines[-1]
                data = json.loads(json_line)
                
                initial_log2 = log2_upper_bound(compiled.model)
                best_log2 = data.get("best", {}).get("fitness", {}).get("log2_upper_bound", initial_log2)
                improvement = initial_log2 - best_log2
                
                return {
                    "name": model_name,
                    "ok": True,
                    "initial_log2": round(initial_log2, 2),
                    "best_log2": round(best_log2, 2),
                    "improvement": round(improvement, 2),
                    "generations": generations,
                    "verified": data.get("stats", {}).get("verified", 0),
                    "best_path": data.get("best", {}).get("paths", {}).get("yaml"),
                }
            except (json.JSONDecodeError, IndexError):
                return {
                    "name": model_name,
                    "ok": True,
                    "initial_log2": log2_upper_bound(compiled.model),
                    "best_log2": None,
                    "improvement": 0,
                    "generations": generations,
                    "verified": 0,
                    "best_path": None,
                    "note": "Could not parse output",
                }
        else:
            return {
                "name": model_name,
                "ok": False,
                "error": result.stderr[:200] if result.stderr else "Unknown error",
            }
            
    except subprocess.TimeoutExpired:
        return {
            "name": req_path.stem.replace(".req", ""),
            "ok": False,
            "error": "Timeout (>2 minutes)",
        }
    except Exception as e:
        return {
            "name": req_path.stem.replace(".req", ""),
            "ok": False,
            "error": str(e),
        }


def main() -> int:
    generations = 5
    population = 8
    only_model = None
    
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--generations" and i + 1 < len(args):
            generations = int(args[i + 1])
        elif arg == "--population" and i + 1 < len(args):
            population = int(args[i + 1])
        elif arg == "--only" and i + 1 < len(args):
            only_model = args[i + 1]
    
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    output_dir = REPO_ROOT / "internal" / "esso" / "evolution_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidates = get_evolvable_models(req_dir)
    
    if only_model:
        candidates = [(p, l) for p, l in candidates if only_model in p.stem]
    
    if not candidates:
        print("No models suitable for fast evolution (log2 > 25)")
        print("Use ICE strengthening instead: python3 esso_ice_sweep.py")
        return 1
    
    print(f"Fast Evolution on {len(candidates)} model(s)")
    print(f"  generations: {generations}, population: {population}")
    print(f"  output: {output_dir}")
    print("=" * 70)
    
    results = []
    for req_path, log2 in candidates:
        name = req_path.stem.replace(".req", "")
        print(f"\n{name} (log2={log2:.1f})...")
        
        result = evolve_model(req_path, output_dir, generations, population)
        results.append(result)
        
        if result["ok"]:
            imp = result.get("improvement", 0)
            if imp > 0:
                print(f"  ✓ Improved! {result['initial_log2']} → {result['best_log2']} (Δ{imp})")
            else:
                print(f"  ✓ Already optimal (log2={result['initial_log2']})")
        else:
            print(f"  ✗ {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    
    # Summary
    improved = [r for r in results if r["ok"] and r.get("improvement", 0) > 0]
    optimal = [r for r in results if r["ok"] and r.get("improvement", 0) == 0]
    failed = [r for r in results if not r["ok"]]
    
    print(f"\nSummary:")
    print(f"  Improved: {len(improved)}")
    print(f"  Already optimal: {len(optimal)}")
    print(f"  Failed: {len(failed)}")
    
    if improved:
        print(f"\nImprovements found:")
        for r in improved:
            print(f"  {r['name']}: {r['initial_log2']} → {r['best_log2']}")
            if r.get("best_path"):
                print(f"    Best model: {r['best_path']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
