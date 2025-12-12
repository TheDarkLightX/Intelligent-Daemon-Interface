# A-Grade Security Specifications

## Overview

This document defines formal specifications for eliminating remaining security vulnerabilities
identified in the external security review (B+ → A upgrade path).

---

## Specification 1: Safe Q-Table Serialization

### Target Files
- `tau_q_agents/phase9_scaled/scaled_q_system.py` (ScaledQSystem.save/load)
- `tau_q_agents/phase10_training/full_training_pipeline.py` (MegaQTable.save/load)

### Current State (INSECURE)
Uses `pickle.dump`/`pickle.load` which can execute arbitrary code during deserialization.

### Required Behavior
Replace pickle with a **data-only** format that cannot execute code.

**Format**: JSON metadata + NumPy `.npz` for array data.

```
model_v1/
├── metadata.json    # Hyperparameters, config, non-array data
└── arrays.npz       # NumPy arrays (q_table, visits, etc.)
```

### Design by Contract

```python
def save(self, path: Path) -> None:
    """
    Save model to disk using safe serialization.
    
    Preconditions:
        - path is a valid writable path
        - path does not contain '..' or start with '/'
        - Model state is consistent (no NaN/Inf in arrays)
    
    Postconditions:
        - Files written atomically (temp + rename)
        - SHA-256 digest stored in metadata for integrity verification
        - load(path) yields semantically equivalent model
    
    Invariants:
        - No code execution during save
        - Format is deterministic (same model → same bytes)
    """

def load(cls, path: Path, *, verify_integrity: bool = True) -> "Model":
    """
    Load model from disk using safe deserialization.
    
    Preconditions:
        - path exists and is a directory with expected structure
        - path is within allowed directory policy
    
    Postconditions:
        - Returned model is valid and consistent
        - If verify_integrity=True, SHA-256 digest matches
    
    Invariants:
        - No code execution during load (JSON + NPZ only)
        - Load fails safely on corrupted/tampered files
    
    Raises:
        - ValueError: If integrity check fails
        - FileNotFoundError: If required files missing
    """
```

### Acceptance Tests
1. **Round-trip property**: `save(model) → load() → save()` produces identical bytes
2. **No pickle**: `grep -r "pickle" path/` returns empty
3. **Integrity check**: Tampered file → `ValueError`
4. **Legacy migration**: One-time script converts old pickle to new format

---

## Specification 2: Remove Insecure Crypto Fallback

### Target File
- `idi/ian/network/node.py`

### Current State (PARTIALLY SECURE)
HMAC fallback exists but is gated by `IAN_PRODUCTION` environment variable.
Risk: Developer might run in non-production mode with real data.

### Required Behavior
**Remove the fallback entirely.** Ed25519 is a hard requirement for all environments.

### Design by Contract

```python
# At module load time:
"""
Preconditions:
    - cryptography library is installed
    
Postconditions:
    - Ed25519 signing/verification is available
    - No HMAC fallback code exists
    
Invariants:
    - All signatures are Ed25519 (64 bytes)
    - All public keys are Ed25519 (32 bytes)

Raises:
    - ImportError: If cryptography not installed (with clear install instructions)
"""
```

### Acceptance Tests
1. **No fallback code**: `grep -r "HMAC\|HAS_CRYPTO" node.py` returns empty
2. **Import fails gracefully**: Missing `cryptography` → clear error message
3. **Signing works**: Unit tests pass with `cryptography` installed

---

## Specification 3: Subprocess Security (shell=True Elimination)

### Target Files
- Any first-party Python file using `subprocess` with `shell=True`

### Current State
Main codebase already uses `shell=False` with `shlex.split()`.
The `verification/performance_benchmark.py` mentioned in report does not exist in HEAD.
Third-party code in `.venv/` is out of scope.

### Required Behavior
Ensure **no first-party code** uses `shell=True`. Add static analysis check.

### Design by Contract

```python
def run_external_command(argv: List[str], *, timeout: float = 60.0) -> CompletedProcess:
    """
    Execute external command safely.
    
    Preconditions:
        - argv is a non-empty list of strings
        - argv[0] is an executable that exists
        - No element of argv contains shell metacharacters if user-supplied
    
    Postconditions:
        - Command executed with shell=False
        - Timeout enforced
    
    Invariants:
        - No shell interpretation of argv elements
    """
```

### Acceptance Tests
1. **Static check**: `grep -rn "shell=True" idi/ tau_q_agents/ --include="*.py"` returns empty
2. **Existing tests pass**: `pytest idi/zk/tests/test_proof_manager_timeout.py`

---

## Specification 4: Path Validation for File I/O

### Scope
All file save/load operations with user-controllable paths.

### Required Behavior
Validate paths before any file I/O:
- No `..` components
- No absolute paths (or must be within allowed base)
- No null bytes
- Reasonable length limit (< 4096 chars)

### Design by Contract

```python
def validate_safe_path(path: Path, *, base_dir: Optional[Path] = None) -> Path:
    """
    Validate that path is safe for file operations.
    
    Preconditions:
        - path is a Path or str
    
    Postconditions:
        - Returned path is normalized and safe
        - If base_dir provided, path is within base_dir
    
    Raises:
        - ValueError: Path contains '..' or null bytes
        - ValueError: Path is absolute when base_dir requires relative
        - ValueError: Path exceeds length limit
    """
```

---

## Implementation Order

1. **Spec 2 (Crypto)**: Smallest change, highest impact
2. **Spec 1 (Pickle → JSON/NPZ)**: Medium change, critical security
3. **Spec 3 (shell=True audit)**: Verification only (already done)
4. **Spec 4 (Path validation)**: Add utility function, integrate

## Verification

After implementation:
- All unit tests pass
- `mypy --strict` on changed files
- Manual security review of diffs
