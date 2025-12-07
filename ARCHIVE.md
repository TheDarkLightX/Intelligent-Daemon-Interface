# Archive Directory

This directory contains archived content that is not part of the main IDI (Intelligent Daemon Interface) project.

## Contents

### `alignment_theorem/`
Complete Alignment Theorem project including:
- `AlignmentTheorem/` - Main project folder with demos, docs, proofs, verification
- `alignment_theorem_package/` - Package version
- `alignment_theorem_repo/` - Repository version  
- `alignment_theorem_package.zip` - Archive file
- `docs/` - All Alignment Theorem documentation (PDFs, LaTeX, HTML, visualizers)

**Status:** Archived - preserved for historical reference but not actively maintained as part of IDI.

**Reason:** The Alignment Theorem is a separate research project focused on formal economic proofs, while IDI focuses on intelligent agent development toolkits.

### `lean_proofs/`
Lean proof system files:
- `lakefile.lean` - Lean project configuration
- `lean-toolchain` - Lean version specification
- `lake-manifest.json` - Lean package manifest
- `proofs/AlignmentTheorem.lean` - Formal proof of Alignment Theorem

**Status:** Archived - related to Alignment Theorem formal verification.

**Reason:** Not needed for IDI development (IDI uses Tau Language, not Lean).

## Accessing Archived Content

To access archived content:
```bash
cd archive/alignment_theorem/AlignmentTheorem
# or
cd archive/lean_proofs/
```

## Restoring Archived Content

If you need to restore archived content to root:
```bash
# Restore Alignment Theorem
mv archive/alignment_theorem/AlignmentTheorem ./
mv archive/alignment_theorem/alignment_theorem_package ./
# etc.

# Restore Lean proofs
mv archive/lean_proofs/lakefile.lean ./
mv archive/lean_proofs/lean-toolchain ./
```

## Git Status

The `archive/` directory is tracked in git (preserves history). To ignore it completely, uncomment `archive/` in `.gitignore`.

