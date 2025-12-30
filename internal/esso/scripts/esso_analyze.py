#!/usr/bin/env python3
"""Analyze ESSO models for complexity and evolution feasibility.

Usage:
    python3 esso_analyze.py [--json]
    
Options:
    --json   Output as JSON for scripting
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add ESSO to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "ESSO"))

from ESSO.foundry.compiler import compile_req
from ESSO.verify.fitness import kernel_complexity, log2_upper_bound
from ESSO.evolve import ir_hash


@dataclass
class ModelAnalysis:
    name: str
    ir_hash: str
    n_state_vars: int
    n_actions: int
    n_invariants: int
    complexity: int
    log2_state_space: float
    evolution_speed: str
    recommended_strategy: str
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ir_hash": self.ir_hash,
            "state_vars": self.n_state_vars,
            "actions": self.n_actions,
            "invariants": self.n_invariants,
            "complexity": self.complexity,
            "log2_state_space": round(self.log2_state_space, 2),
            "evolution_speed": self.evolution_speed,
            "recommended_strategy": self.recommended_strategy,
        }


def classify_speed(log2: float) -> str:
    if log2 < 10:
        return "VERY_FAST"
    elif log2 < 20:
        return "FAST"
    elif log2 < 30:
        return "MEDIUM"
    elif log2 < 45:
        return "SLOW"
    else:
        return "VERY_SLOW"


def recommend_strategy(log2: float, complexity: int) -> str:
    if log2 < 15:
        return "evolve --generations 10 --population 20"
    elif log2 < 25:
        return "evolve --generations 5 --population 10 --timeout-ms 2000"
    elif log2 < 35:
        return "ice --max-rounds 10 (evolution too slow)"
    else:
        return "verify-multi only (state space too large for evolution)"


def analyze_model(req_path: Path) -> ModelAnalysis | None:
    try:
        compiled = compile_req(
            req_path=req_path,
            semantics_profile=None,
            style_profile=None,
        )
        m = compiled.model
        
        h = ir_hash(m)
        cplx = kernel_complexity(m)
        log2 = log2_upper_bound(m)
        speed = classify_speed(log2)
        strategy = recommend_strategy(log2, cplx)
        
        return ModelAnalysis(
            name=req_path.stem.replace(".req", ""),
            ir_hash=h[:16] + "...",
            n_state_vars=len(m.state_vars),
            n_actions=len(m.actions),
            n_invariants=len(m.invariants),
            complexity=cplx,
            log2_state_space=log2,
            evolution_speed=speed,
            recommended_strategy=strategy,
        )
    except Exception as e:
        print(f"Error analyzing {req_path.name}: {e}", file=sys.stderr)
        return None


def main() -> int:
    as_json = "--json" in sys.argv
    
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    req_files = sorted(req_dir.glob("*.req.yaml"))
    
    if not req_files:
        print(f"No REQ files found in {req_dir}")
        return 1
    
    analyses = []
    for req_path in req_files:
        analysis = analyze_model(req_path)
        if analysis:
            analyses.append(analysis)
    
    if as_json:
        print(json.dumps([a.to_dict() for a in analyses], indent=2))
    else:
        print("ESSO Model Analysis")
        print("=" * 90)
        print(f"{'Model':<25} {'Vars':<6} {'Acts':<6} {'Cplx':<8} {'log2':<8} {'Speed':<12} Strategy")
        print("-" * 90)
        
        for a in sorted(analyses, key=lambda x: x.log2_state_space):
            print(f"{a.name:<25} {a.n_state_vars:<6} {a.n_actions:<6} {a.complexity:<8} "
                  f"{a.log2_state_space:<8.1f} {a.evolution_speed:<12} {a.recommended_strategy}")
        
        print("-" * 90)
        
        # Summary
        fast = [a for a in analyses if a.evolution_speed in ("VERY_FAST", "FAST")]
        slow = [a for a in analyses if a.evolution_speed in ("SLOW", "VERY_SLOW")]
        
        print()
        print("Summary:")
        print(f"  Fast evolution candidates: {[a.name for a in fast]}")
        print(f"  Use ICE/verify-only: {[a.name for a in slow]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
