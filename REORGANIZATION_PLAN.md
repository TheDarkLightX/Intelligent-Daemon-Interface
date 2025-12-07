# Repository Reorganization Plan - IDI Focus

## Goal
Reorganize repository to focus on **Intelligent Daemon Interface (IDI)** project, archiving unrelated Alignment Theorem content.

## Current Repository Analysis

### IDI Project Content (Keep & Organize)
- ✅ `idi/` - Core IDI project (devkit, training, zk, examples, specs)
- ✅ `tau_daemon_alpha/` - Rust daemon for Tau spec execution
- ✅ `specification/` - Tau agent specifications (V35-V54, libraries)
- ✅ `tau_q_agents/` - Q-learning agent implementations
- ✅ `verification/` - Verification tools and tests
- ✅ `scripts/` - Build and test scripts
- ✅ `docs/IDI_IAN_ARCHITECTURE.md` - IDI documentation
- ✅ `docs/TauNet_Deflationary_Ecosystem_DeepDive.md` - IDI-related

### Unrelated Content (Archive)
- ❌ `AlignmentTheorem/` - Alignment Theorem project (2.2M)
- ❌ `alignment_theorem_package/` - Package version (908K)
- ❌ `alignment_theorem_repo/` - Repo version (348K)
- ❌ `alignment_theorem_package.zip` - Archive file
- ❌ `proofs/AlignmentTheorem.lean` - Lean proof
- ❌ `.lake/` - Lean package cache (566M - should be gitignored)
- ❌ `lakefile.lean`, `lean-toolchain` - Lean project files
- ❌ `docs/Alignment_Theorem_*.{pdf,tex,docx,fodt,html}` - Alignment docs
- ❌ `docs/alignment_theorem_visualizer.html` - Alignment visualizer
- ❌ `docs/THE_ALIGNMENT_THEOREM.md` - Alignment Theorem doc
- ❌ `docs/figures/alignment_convergence.png` - Alignment figure

### External Dependencies (Gitignore)
- ⚠️ `tau-lang-latest/` - Tau language source (external, users build locally)
- ⚠️ `inputs/`, `outputs/` - Runtime directories (generated)
- ⚠️ `.lake/` - Lean cache (huge, not needed)

## Proposed Structure

```
idi/                          # Main IDI project
├── devkit/                   # Agent development toolkit (tau_factory)
├── training/                 # Q-learning training (Python + Rust)
├── zk/                       # Zero-knowledge proof integration
├── examples/                 # Example agents
├── practice/                 # Practice agents
├── specs/                    # Agent specifications
└── docs/                     # IDI documentation

tau_daemon_alpha/              # Rust daemon for Tau execution
specification/                 # Tau agent specs (V35-V54, libraries)
tau_q_agents/                  # Legacy Q-learning implementations
verification/                  # Verification tools
scripts/                       # Build and test scripts

archive/                       # Archived unrelated content
├── alignment_theorem/         # All Alignment Theorem content
│   ├── AlignmentTheorem/
│   ├── alignment_theorem_package/
│   ├── alignment_theorem_repo/
│   ├── alignment_theorem_package.zip
│   └── README.md              # Explains archived content
└── lean_proofs/              # Lean proof system
    ├── lakefile.lean
    ├── lean-toolchain
    └── proofs/

docs/                          # IDI-focused documentation
├── IDI_IAN_ARCHITECTURE.md
├── TauNet_Deflationary_Ecosystem_DeepDive.md
└── ... (other IDI docs)

README.md                      # IDI-focused README
.gitignore                     # Updated for IDI
LICENSE                        # Keep
ARCHIVE.md                     # Documents archived content
```

## Implementation Steps

### Step 1: Create Archive Structure
```bash
mkdir -p archive/alignment_theorem archive/lean_proofs
```

### Step 2: Move Alignment Theorem Content
```bash
mv AlignmentTheorem archive/alignment_theorem/
mv alignment_theorem_package archive/alignment_theorem/
mv alignment_theorem_repo archive/alignment_theorem/
mv alignment_theorem_package.zip archive/alignment_theorem/
```

### Step 3: Move Lean Proof System
```bash
mv lakefile.lean archive/lean_proofs/
mv lean-toolchain archive/lean_proofs/
mv proofs/AlignmentTheorem.lean archive/lean_proofs/proofs/ 2>/dev/null || true
```

### Step 4: Move Alignment Theorem Docs
```bash
mkdir -p archive/alignment_theorem/docs
mv docs/Alignment_Theorem_* archive/alignment_theorem/docs/ 2>/dev/null || true
mv docs/alignment-theorem-deep-dive.html archive/alignment_theorem/docs/ 2>/dev/null || true
mv docs/alignment_theorem_visualizer.html archive/alignment_theorem/docs/ 2>/dev/null || true
mv docs/THE_ALIGNMENT_THEOREM.md archive/alignment_theorem/docs/ 2>/dev/null || true
mv docs/figures/alignment_convergence.png archive/alignment_theorem/docs/figures/ 2>/dev/null || true
```

### Step 5: Update .gitignore
Add to `.gitignore`:
```
# Archive (keep in repo but document as archived)
archive/

# External dependencies
tau-lang-latest/

# Lean cache (huge)
.lake/

# Runtime directories
inputs/
outputs/
*.out
*.in

# Build artifacts
__pycache__/
*.pyc
.pytest_cache/
target/
*.lock
```

### Step 6: Update README.md
Rewrite to focus on IDI project.

### Step 7: Create ARCHIVE.md
Document what's archived and why.

## Files to Update

1. **README.md** - Rewrite for IDI focus
2. **.gitignore** - Add archive, tau-lang, runtime dirs, Lean cache
3. **ARCHIVE.md** - Document archived content
4. **docs/** - Keep only IDI-related docs

## Verification Checklist

- [ ] All IDI functionality still works
- [ ] Scripts updated if paths changed
- [ ] Documentation updated
- [ ] Git history preserved (archive, don't delete)
- [ ] .gitignore updated
- [ ] README reflects IDI focus
