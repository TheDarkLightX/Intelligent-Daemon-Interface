#!/usr/bin/env python3
"""Compile all ESSO REQ specifications to Python kernels.

Usage:
    python3 esso_compile_all.py [--dry-run] [--only NAME]
    
Options:
    --dry-run   Show what would be compiled without writing files
    --only NAME Only compile the specified REQ (without .req.yaml suffix)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

# Add ESSO to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "ESSO"))

from ESSO.foundry.compiler import compile_req
from ESSO.export.python import generate_python_model
from ESSO.evolve import ir_hash


class KernelMapping(NamedTuple):
    req_name: str
    kernel_name: str
    output_path: Path


# Define REQ -> Kernel mappings
KERNEL_MAPPINGS = [
    KernelMapping(
        "bond_status",
        "bond_status_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "bond_status_fsm_ref.py",
    ),
    KernelMapping(
        "challenge_bond",
        "challenge_bond_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "challenge_bond_fsm_ref.py",
    ),
    KernelMapping(
        "evaluation_quorum",
        "evaluation_quorum_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "evaluation_quorum_fsm_ref.py",
    ),
    KernelMapping(
        "fast_lane",
        "fast_lane_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "fast_lane_fsm_ref.py",
    ),
    KernelMapping(
        "fraud_proof",
        "fraud_proof_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "fraud_proof_fsm_ref.py",
    ),
    KernelMapping(
        "mempool_lifecycle",
        "mempool_lifecycle_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "mempool_lifecycle_fsm_ref.py",
    ),
    KernelMapping(
        "peer_tier",
        "peer_tier_fsm_ref",
        REPO_ROOT / "idi" / "ian" / "network" / "kernels" / "peer_tier_fsm_ref.py",
    ),
]


def compile_kernel(mapping: KernelMapping, req_dir: Path, dry_run: bool = False) -> bool:
    """Compile a single REQ to kernel.
    
    Returns:
        True if successful
    """
    req_path = req_dir / f"{mapping.req_name}.req.yaml"
    
    if not req_path.exists():
        print(f"✗ {mapping.req_name}: REQ file not found at {req_path}")
        return False
    
    try:
        compiled = compile_req(
            req_path=req_path,
            semantics_profile=None,
            style_profile=None,
        )
        
        code = generate_python_model(compiled.model)
        h = ir_hash(compiled.model)[:16]
        
        if dry_run:
            print(f"○ {mapping.req_name} -> {mapping.kernel_name}")
            print(f"    Would write to: {mapping.output_path}")
            print(f"    Hash: {h}...")
        else:
            mapping.output_path.parent.mkdir(parents=True, exist_ok=True)
            mapping.output_path.write_text(code)
            print(f"✓ {mapping.req_name} -> {mapping.kernel_name}")
            print(f"    Written to: {mapping.output_path}")
            print(f"    Hash: {h}...")
        
        return True
        
    except Exception as e:
        print(f"✗ {mapping.req_name}: {e}")
        return False


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    only = None
    
    for i, arg in enumerate(sys.argv):
        if arg == "--only" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]
    
    req_dir = REPO_ROOT / "internal" / "esso" / "requirements"
    
    mappings = KERNEL_MAPPINGS
    if only:
        mappings = [m for m in mappings if m.req_name == only]
        if not mappings:
            print(f"Unknown REQ: {only}")
            print(f"Available: {[m.req_name for m in KERNEL_MAPPINGS]}")
            return 1
    
    print(f"Compiling {len(mappings)} kernels..." + (" (dry run)" if dry_run else ""))
    print("=" * 60)
    
    passed = sum(1 for m in mappings if compile_kernel(m, req_dir, dry_run))
    failed = len(mappings) - passed
    
    print("=" * 60)
    print(f"Results: {passed} compiled, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
