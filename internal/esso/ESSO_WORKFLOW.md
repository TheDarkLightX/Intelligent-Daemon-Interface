# ESSO Workflow for IAN/IDI

## Overview

ESSO (Evolutionary State Space Optimizer) is used to:
1. **Compile** requirement specs (REQ) into verified FSM kernels
2. **Verify** kernels with multi-solver SMT checking (z3 + cvc5)
3. **Evolve** models to find more efficient implementations
4. **Synthesize** hole expressions via SyGuS
5. **Beautify** generated code for readability

## Quick Start

```bash
cd /home/trevormoc/Downloads/IDI/Intelligent-Daemon-Interface
export PYTHONPATH=external/ESSO
```

## Commands

### 1. Validate a REQ spec
```bash
python3 -m ESSO validate internal/esso/requirements/evaluation_quorum.req.yaml
```

### 2. Compile REQ → Python Kernel
```python
from pathlib import Path
from ESSO.foundry.compiler import compile_req
from ESSO.export.python import generate_python_model

compiled = compile_req(
    req_path=Path('internal/esso/requirements/evaluation_quorum.req.yaml'),
    semantics_profile=None,
    style_profile=None
)
code = generate_python_model(compiled.model)
Path('idi/ian/network/kernels/evaluation_quorum_fsm_ref.py').write_text(code)
```

### 3. Verify with Multi-Solver SMT
```bash
python3 -m ESSO verify-multi /tmp/model.yaml \
    --timeout-ms 5000 \
    --solvers z3,cvc5 \
    --output /tmp/verify_output \
    --write-report
```

**Interpreters:**
- `UNSAT` = invariant holds (good)
- `SAT` = counterexample found (bug!)
- `UNKNOWN` = solver timeout/undecidable

### 4. Evolution (Find Efficient Models)
```bash
python3 -m ESSO evolve model.yaml \
    --generations 10 \
    --population 20 \
    --seed 42 \
    --timeout-ms 5000 \
    --output /tmp/evolve_output \
    --verbose
```

### 5. Synthesis (CGS - Constraint-Guided Synthesis)
```bash
python3 -m ESSO synth model.yaml \
    --synth synth.json \
    --max-iters 100 \
    --solvers z3,cvc5 \
    --output /tmp/synth_output
```

### 6. Beautify (Clean Up Synthesized Code)
```bash
python3 -m ESSO beautify candidate_dir \
    --synth synth.json \
    --solvers z3,cvc5 \
    --output /tmp/beautify_output
```

## REQ Spec Format

```yaml
name: my_fsm
schema: "foundry-req/v1"
req_id: "my_fsm"
req_version: "0.0.1"

state:
  status: [IDLE, ACTIVE, DONE]  # enum
  count: { type: int, min: 0, max: 100 }  # bounded int
  flag: { type: bool }  # boolean

actions:
  - name: start
    params: {}
    pre: ["status = IDLE"]  # guard conditions
    eff:
      - "status' = ACTIVE"  # post-state assignments
      - "count' = 0"

invariants:
  - name: count_bounded
    expr: "count <= 100"  # safety invariant

observations:
  - name: is_active
    expr: "status = ACTIVE"
```

## Key Lessons Learned

### 1. Bound Guards Are Critical
ESSO found a bug: `vote_accept` could push `votes_received` to 101 when max was 100.

**Fix:** Add explicit guards:
```yaml
pre: ["status = COLLECTING", "votes_received < 100"]
```

### 2. SMT Verification is Slow
- Queries can take 5+ seconds each
- Use `--timeout-ms` to prevent hangs
- cvc5 is often faster than z3 for certain queries

### 3. Multi-Solver Cross-Check
When z3 and cvc5 disagree, the model needs review:
- `z3: unknown, cvc5: unsat` → z3 timed out, cvc5 proved it
- `z3: sat, cvc5: sat` → both found counterexample (real bug)

## Current Kernels (17 verified)

| Module | Kernel | REQ Path |
|--------|--------|----------|
| evaluation.py | evaluation_quorum_fsm_ref | requirements/evaluation_quorum.req.yaml |
| fast_lane.py | fast_lane_fsm_ref | requirements/fast_lane.req.yaml |
| peer_tiers.py | peer_tier_fsm_ref | requirements/peer_tier.req.yaml |
| skiplist_mempool.py | mempool_lifecycle_fsm_ref | requirements/mempool_lifecycle.req.yaml |
| economics.py | bond_status_fsm_ref | requirements/bond_status.req.yaml |
| economics.py | challenge_bond_fsm_ref | requirements/challenge_bond.req.yaml |
| fraud.py | fraud_proof_fsm_ref | requirements/fraud_proof.req.yaml |

## Troubleshooting

### cvc5 Not Found
Ensure cvc5 is in PATH or set `CVC5_PATH`:
```bash
export CVC5_PATH=/usr/bin/cvc5
```

### Evolution Too Slow
Reduce parameters:
```bash
--generations 3 --population 5 --timeout-ms 2000
```

### Solver Disagreement
Check the verification report for counterexamples and review the invariants.
