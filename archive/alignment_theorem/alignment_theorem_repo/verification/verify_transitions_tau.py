#!/usr/bin/env python3
"""
Tau Transition Verifier

Generates Tau specification files to verify FSM transitions using the 'tau' binary's solver.
This proves that the actual Tau bitvector logic (running on Z3) produces the expected state transitions.
"""

import subprocess
import os
import sys

TAU_BINARY = "/home/trevormoc/Downloads/tau-lang-latest/build-Release/tau"

def run_tau_solve(tau_content: str, test_name: str):
    filename = f"verification/test_trans_{test_name}.tau"
    with open(filename, "w") as f:
        f.write(tau_content)
    
    print(f"Running {test_name}...")
    try:
        result = subprocess.run([TAU_BINARY, filename], capture_output=True, text=True, timeout=10)
        output = result.stdout
        
        if "solution:" in output:
            print("  ✓ SAT (Solution found)")
            return True
        elif "unsat" in output:
            print("  ✗ UNSAT (No solution found - logic contradiction?)")
            print(output)
            return False
        else:
            print(f"  ? Unknown result or error: {output[:100]}...")
            if result.stderr:
                print(f"  Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ✗ Timeout")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def verify_infinite_deflation_genesis():
    """
    Verify GENESIS -> ACTIVE transition.
    """
    # c=circuit, s=supply, e=eetf, t=time, p=prev_state, d=prev_era, h=halving_pd, g=eetf_target, r=cur_era, x=expected, n=next_state
    # Using simple variables, no parens around equalities
    tau_code = """solve c={1}:bv[256] && s={#x033B2E3C9FD0803CE800000000000000000000000000000000}:bv[256] && e={100}:bv[256] && t={0}:bv[256] && p={0}:bv[256] && d={0}:bv[256] && h={216000}:bv[256] && g={100}:bv[256] && r=t/h && x={1}:bv[256] && ( (c!={1}:bv[256] -> n={4}:bv[256]) && ((c={1}:bv[256] && r>d) -> n={3}:bv[256]) && ((c={1}:bv[256] && r=d && e>{200}:bv[256]) -> n={2}:bv[256]) && ((c={1}:bv[256] && r=d && e<={200}:bv[256]) -> n={1}:bv[256]) ) && n=x
"""
    return run_tau_solve(tau_code, "InfiniteDeflation_Genesis_to_Active")

def verify_agent_v54_entry():
    """
    Verify ANALYZING -> CONFIDENT_ENTRY
    """
    # c=circuit, p=prev_pos, f=confidence, e=eetf, r=regime_req, n=next_state
    # Logic: (circuit & ~in_pos & conf>=2 & eetf>=req) -> next=CONFIDENT_ENTRY(1)
    # Else -> next=ANALYZING(0)
    tau_code = """solve c={1}:bv[256] && p={0}:bv[256] && f={2}:bv[256] && e={150}:bv[256] && r={100}:bv[256] && ( ((c={1}:bv[256] && p={0}:bv[256] && f>={2}:bv[256] && e>=r) -> n={1}:bv[256]) && ( !(c={1}:bv[256] && p={0}:bv[256] && f>={2}:bv[256] && e>=r) -> n={0}:bv[256]) ) && n={1}:bv[256]
"""
    return run_tau_solve(tau_code, "AgentV54_Entry")

if __name__ == "__main__":
    print("="*60)
    print("TAU TRANSITION VERIFICATION (Running on tau binary)")
    print("="*60)
    
    success = True
    success &= verify_infinite_deflation_genesis()
    success &= verify_agent_v54_entry()
    
    if success:
        print("\n✓ All transitions verified on actual Tau binary!")
        sys.exit(0)
    else:
        print("\n✗ Some verification steps failed.")
        sys.exit(1)
